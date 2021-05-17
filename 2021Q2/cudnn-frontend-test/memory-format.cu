#include <cassert>
#include <cudnn_frontend.h>
#include <random>
#include <vector>

constexpr int64_t DIMS = 4;

struct Tensor {
  int64_t shape[DIMS];
  int64_t strides[DIMS];
  float *data;
  float &operator()(int64_t i, int64_t j, int64_t k, int64_t l) {
    int64_t index =
        i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
    return data[index];
  }
};

Tensor new_tensor(const std::vector<int64_t> &shape,
                  const std::vector<int64_t> &dim_order) {
  Tensor ret;

  for (int i = 0; i < DIMS; i++) {
    ret.shape[i] = shape[i];
  }

  int64_t size = 1;
  for (int i = 0; i < DIMS; i++) {
    auto dim = dim_order[i];
    ret.strides[dim] = size;
    size *= shape[dim];
  }
  cudaMallocManaged(&ret.data, size * sizeof(float));
  return ret;
}

void copy(Tensor &to, Tensor &from) {
  for (int i = 0; i < from.shape[0]; i++) {
    for (int j = 0; j < from.shape[1]; j++) {
      for (int k = 0; k < from.shape[2]; k++) {
        for (int l = 0; l < from.shape[3]; l++) {
          to(i, j, k, l) = from(i, j, k, l);
        }
      }
    }
  }
}

float maxdiff(Tensor &to, Tensor &from) {
  float result = -1;
  for (int i = 0; i < from.shape[0]; i++) {
    for (int j = 0; j < from.shape[1]; j++) {
      for (int k = 0; k < from.shape[2]; k++) {
        for (int l = 0; l < from.shape[3]; l++) {
          float diff = std::abs(to(i, j, k, l) - from(i, j, k, l));
          if (diff > result) {
            result = diff;
          }
        }
      }
    }
  }
  return result;
}

void random_fill(Tensor &t) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (int i = 0; i < t.shape[0]; i++) {
    for (int j = 0; j < t.shape[1]; j++) {
      for (int k = 0; k < t.shape[2]; k++) {
        for (int l = 0; l < t.shape[3]; l++) {
          t(i, j, k, l) = distribution(generator);
        }
      }
    }
  }
}

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uint64_t address = reinterpret_cast<uint64_t>(t.data);
  while (address % alignment == 0 && alignment < 16)
    alignment *= 2;
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, int64_t id) {
  return cudnn_frontend::TensorBuilder()
      .setDim(DIMS, t.shape)
      .setStrides(DIMS, t.strides)
      .setId(id)
      .setAlignment(getAlignment(t))
      .setDataType(CUDNN_DATA_FLOAT)
      .build();
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(std::vector<int64_t> padding,
                                              std::vector<int64_t> stride,
                                              std::vector<int64_t> dilation) {
  uint64_t convDim = stride.size();
  return cudnn_frontend::ConvDescBuilder()
      .setDataType(CUDNN_DATA_FLOAT)
      .setMathMode(CUDNN_CROSS_CORRELATION)
      .setNDims(convDim)
      .setStrides(convDim, stride.data())
      .setPrePadding(convDim, padding.data())
      .setPostPadding(convDim, padding.data())
      .setDilation(convDim, dilation.data())
      .build();
}

void filterEngineConfigs(cudnn_frontend::EngineConfigList &from,
                         cudnn_frontend::EngineConfigList &to,
                         bool deterministic, bool allow_tf32) {
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<
              CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c))
        return true;
    }
    if (!allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<
              CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c))
        return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c))
        return true;
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}

void convolution(Tensor input, Tensor weight, Tensor output,
                 std::vector<int64_t> padding, std::vector<int64_t> stride,
                 std::vector<int64_t> dilation, bool deterministic,
                 bool allow_tf32) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  auto run = [&](cudnn_frontend::ManagedOpaqueDescriptor cfg) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(cfg)
                    .build();

    auto workspace_size = plan.getWorkspaceSize();
    void *workspace;
    cudaMalloc(&workspace, workspace_size);
    void *data_ptrs[] = {input.data, output.data, weight.data};

    int64_t uids[] = {'x', 'y', 'w'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace)
                           .setDataPointers(3, data_ptrs)
                           .setUids(3, uids)
                           .build();
    cudnnBackendExecute(handle, plan.get_raw_desc(),
                        variantPack.get_raw_desc());
  };

  auto op = cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(getTensorDescriptor(input, 'x'))
                .setyDesc(getTensorDescriptor(output, 'y'))
                .setwDesc(getTensorDescriptor(weight, 'w'))
                .setcDesc(getConvDescriptor(padding, stride, dilation))
                .build();
  // std::cout << op.describe() << std::endl;

  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(1, ops.data())
                     .build();
  // std::cout << opGraph.describe() << std::endl;

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(opGraph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();
  auto fallback =
      cudnn_frontend::EngineFallbackListBuilder()
          .setOperationGraph(opGraph)
          .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
          .build();

  auto &engine_configs =
      heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto &fallback_list = fallback.getFallbackList();

  cudnn_frontend::EngineConfigList filtered_configs;
  filterEngineConfigs(engine_configs, filtered_configs, deterministic,
                      allow_tf32);
  filterEngineConfigs(fallback_list, filtered_configs, deterministic,
                      allow_tf32);

  for (auto &cfg : filtered_configs) {
    try {
      run(cfg);
      return;
    } catch (cudnn_frontend::cudnnException &e) {
    }
  }
}

int main() {}
