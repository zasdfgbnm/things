from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='test',
    ext_modules=[
        CUDAExtension('test', ['test.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
