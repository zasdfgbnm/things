#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/extension.h>

using namespace torch::jit::fuser::cuda;

void func() {
  Fusion fusion;
  FusionGuard fg(&fusion);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
