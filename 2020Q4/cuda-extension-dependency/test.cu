#include <torch/extension.h>
#include "test.cuh"

TORCH_LIBRARY(my_ops, m) {
  m.def("is_installed() -> bool", &is_installed);
}
