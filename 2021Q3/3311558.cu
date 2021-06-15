#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <iostream>
#include <random>
#include <vector>

constexpr int64_t DIMS = 4;

template <typename T> struct Tensor {
  using dtype = T;
  int64_t shape[DIMS];
  int64_t strides[DIMS];
  T *data;
  T &operator()(int64_t i, int64_t j, int64_t k, int64_t l) {
    int64_t index =
        i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
    return data[index];
  }
  const T &operator()(int64_t i, int64_t j, int64_t k, int64_t l) const {
    int64_t index =
        i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
    return data[index];
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor<T> &t) {
  out << "Tensor(shape=[" << t.shape[0] << "," << t.shape[1] << ","
      << t.shape[2] << "," << t.shape[3] << "], stride=[" << t.strides[0] << ","
      << t.strides[1] << "," << t.strides[2] << "," << t.strides[3]
      << "], data = [\n";
  bool firsti = true;
  for (int64_t i = 0; i < t.shape[0]; i++) {
    if (!firsti) {
      out << ",\n";
    }
    out << " [";
    bool firstj = true;
    for (int64_t j = 0; j < t.shape[1]; j++) {
      if (!firstj) {
        out << ",\n  ";
      }
      out << "[";
      bool firstk = true;
      for (int64_t k = 0; k < t.shape[2]; k++) {
        if (!firstk) {
          out << ", ";
        }
        out << "[";
        bool firstl = true;
        for (int64_t l = 0; l < t.shape[3]; l++) {
          if (!firstl) {
            out << ", ";
          }
          out << t(i, j, k, l);
          firstl = false;
        }
        out << "]";
        firstk = false;
      }
      out << "]";
      firstj = false;
    }
    out << "]";
    firsti = false;
  }
  out << "]);";
  return out;
}

template <typename T> inline cudnnDataType_t getDataType() {
  if (std::is_same<T, float>::value) {
    return CUDNN_DATA_FLOAT;
  } else if (std::is_same<T, __half>::value) {
    return CUDNN_DATA_HALF;
  } else if (std::is_same<T, double>::value) {
    return CUDNN_DATA_DOUBLE;
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    return CUDNN_DATA_BFLOAT16;
  }
  throw std::runtime_error(
      "TensorDescriptor only supports double, float and half tensors");
}

template <typename T>
Tensor<T> new_tensor(const std::vector<int64_t> &shape,
                     const std::vector<int64_t> &dim_order) {
  Tensor<T> ret;

  for (int i = 0; i < DIMS; i++) {
    ret.shape[i] = shape[i];
  }

  int64_t size = 1;
  for (int i = 0; i < DIMS; i++) {
    auto dim = dim_order[i];
    ret.strides[dim] = size;
    size *= shape[dim];
  }
  cudaMallocManaged(&ret.data, size * sizeof(T));
  return ret;
}

template <typename T1, typename T2>
void copy(Tensor<T1> &to, Tensor<T2> &from) {
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

template <typename T1, typename T2>
double maxdiff(Tensor<T1> &to, Tensor<T2> &from) {
  double result = -1;
  for (int i = 0; i < from.shape[0]; i++) {
    for (int j = 0; j < from.shape[1]; j++) {
      for (int k = 0; k < from.shape[2]; k++) {
        for (int l = 0; l < from.shape[3]; l++) {
          auto diff = std::abs(to(i, j, k, l) - from(i, j, k, l));
          if (diff > result) {
            result = diff;
          }
        }
      }
    }
  }
  return result;
}

template <typename T> void random_fill(Tensor<T> &t) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-3, 3);
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

class CuDNNError : public std::runtime_error {
  using runtime_error::runtime_error;
};

#define CUDNN_CHECK(EXPR, ...)                                                 \
  do {                                                                         \
    cudnnStatus_t status = EXPR;                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      throw CuDNNError("cuDNN error");                                         \
    }                                                                          \
  } while (0)

template <typename T> uint8_t getAlignment(const Tensor<T> &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uint64_t address = reinterpret_cast<uint64_t>(t.data);
  while (address % alignment == 0 && alignment < 16)
    alignment *= 2;
  return alignment;
}

template <typename T>
cudnn_frontend::Tensor getTensorDescriptor(const Tensor<T> &t, int64_t id) {
  return cudnn_frontend::TensorBuilder()
      .setDim(DIMS, t.shape)
      .setStrides(DIMS, t.strides)
      .setId(id)
      .setAlignment(getAlignment(t))
      .setDataType(getDataType<T>())
      .build();
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dtype,
                                              std::vector<int64_t> padding,
                                              std::vector<int64_t> stride,
                                              std::vector<int64_t> dilation) {
  uint64_t convDim = stride.size();
  return cudnn_frontend::ConvDescBuilder()
      .setDataType(dtype)
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

template <typename T1, typename T2, typename T3>
void convolution(Tensor<T1> input, Tensor<T2> weight, Tensor<T3> output,
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
    CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(),
                                    variantPack.get_raw_desc()));
  };

  auto op = cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(getTensorDescriptor(input, 'x'))
                .setyDesc(getTensorDescriptor(output, 'y'))
                .setwDesc(getTensorDescriptor(weight, 'w'))
                .setcDesc(getConvDescriptor(getDataType<T3>(), padding, stride,
                                            dilation))
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
    } catch (CuDNNError &e) {
    }
  }
}

int main() {
  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> dilation = {1, 1};

  // float nchw
  auto input = new_tensor<float>({2, 8, 4, 4}, {3, 2, 1, 0});
  random_fill(input);
  // std::cout << "input = " << input << std::endl;
  auto weight = new_tensor<float>({4, 8, 3, 3}, {3, 2, 1, 0});
  random_fill(weight);
  auto output = new_tensor<float>({2, 4, 2, 2}, {3, 2, 1, 0});
  convolution(input, weight, output, padding, stride, dilation, false, true);

  // double nchw
  auto input2 = new_tensor<double>({2, 8, 4, 4}, {3, 2, 1, 0});
  copy(input2, input);
  auto weight2 = new_tensor<double>({4, 8, 3, 3}, {3, 2, 1, 0});
  copy(weight2, weight);
  auto output2 = new_tensor<double>({2, 4, 2, 2}, {3, 2, 1, 0});
  convolution(input2, weight2, output2, padding, stride, dilation, false, true);

  std::cout << "diff(output, output2) = " << maxdiff(output, output2)
            << std::endl;
}