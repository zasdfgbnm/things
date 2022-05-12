import torch
from torch.utils.cpp_extension import load_inline

source = r"""
#include <thrust/thrust.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
"""