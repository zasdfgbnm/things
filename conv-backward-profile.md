code:
```python
import torch

input_ = torch.zeros(128, 3, 224, 224).cuda()
conv1 = torch.nn.Conv2d(3, 64, 3, 3, bias=True).cuda()
conv2 = torch.nn.Conv2d(3, 64, 3, 3, bias=False).cuda()
r1 = conv1(input_).sum()
r2 = (conv2(input_) + conv1.bias.view(1, -1, 1, 1)).sum()
torch.cuda.synchronize()

torch.cuda.profiler.start()
torch.autograd.grad(r1, conv1.bias, retain_graph=True)

torch.cuda.synchronize()

torch.autograd.grad(r2, conv1.bias, retain_graph=True)
torch.cuda.synchronize()

torch.cuda.profiler.stop()
```

result
```
nvprof --profile-from-start on --print-gpu-trace python conv-bias2.py
```

```
4477== NVPROF is profiling process 24477, command: python conv-bias2.py
==24477== Profiling application: python conv-bias2.py
==24477== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
3.24575s  7.7809ms                    -               -         -         -         -  73.500MB  9.2248GB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
3.25740s  2.0800us                    -               -         -         -         -  6.7500KB  3.0949GB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
3.25751s  1.6000us                    -               -         -         -         -      256B  152.59MB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
3.25769s  2.0800us                    -               -         -         -         -  6.7500KB  3.0949GB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
5.14458s  1.8880us                    -               -         -         -         -  15.000KB  7.5769GB/s      Device           -  Tesla V100-DGXS         1        25  [CUDA memset]
5.14460s  2.0160us                    -               -         -         -         -  15.000KB  7.0958GB/s      Device           -  Tesla V100-DGXS         1        26  [CUDA memset]
5.14461s  2.0480us                    -               -         -         -         -  15.000KB  6.9849GB/s      Device           -  Tesla V100-DGXS         1        27  [CUDA memset]
5.14463s  2.0480us                    -               -         -         -         -  15.000KB  6.9849GB/s      Device           -  Tesla V100-DGXS         1        28  [CUDA memset]
5.14536s  1.6000us                    -               -         -         -         -      112B  66.757MB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
5.14680s  2.1120us             (43 1 1)       (128 1 1)        16        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) [1121]
5.14681s  443.14us           (5476 1 1)       (128 1 1)       128  16.000KB        0B         -           -           -           -  Tesla V100-DGXS         1         7  volta_scudnn_128x64_relu_interior_nn_v1 [1124]
5.14726s  468.35us          (43808 1 1)       (256 1 1)        18        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  void op_generic_tensor_kernel<int=4, float, float, float, int=256, cudnnGenericOp_t=0, cudnnNanPropagation_t=0, int=0>(cudnnTensorStruct, float*, cudnnTensorStruct, float const *, cudnnTensorStruct, float const *, float, float, float, float, reducedDivisorArray, int) [1127]
5.14773s  1.3440us                    -               -         -         -         -        4B  2.8383MB/s      Device           -  Tesla V100-DGXS         1         7  [CUDA memset]
5.14774s  220.45us           (1 5476 1)       (512 1 1)        32       16B  2.0000KB         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native13reduce_kernelILi512ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_15sum_kernel_implIfffEEvRNS_14TensorIteratorEEUlffE_EEjfLi4EEEEEvT0_ [1146]
5.14796s  2.1760us             (43 1 1)       (128 1 1)        16        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) [1168]
5.14796s  438.05us           (5476 1 1)       (128 1 1)       128  16.000KB        0B         -           -           -           -  Tesla V100-DGXS         1         7  volta_scudnn_128x64_relu_interior_nn_v1 [1171]
5.14840s  456.74us          (87616 1 1)       (128 1 1)        16        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native18elementwise_kernelILi128ELi4EZNS0_15gpu_kernel_implIZZZNS0_15add_kernel_cudaERNS_14TensorIteratorEN3c106ScalarEENKUlvE_clEvENKUlvE2_clEvEUlffE_EEvS4_RKT_EUliE2_EEviT1_ [1184]
5.14886s  1.3440us                    -               -         -         -         -        4B  2.8383MB/s      Device           -  Tesla V100-DGXS         1         7  [CUDA memset]
5.14886s  218.59us           (1 5476 1)       (512 1 1)        32       16B  2.0000KB         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native13reduce_kernelILi512ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_15sum_kernel_implIfffEEvRNS_14TensorIteratorEEUlffE_EEjfLi4EEEEEvT0_ [1200]
5.14925s  2.2720us              (1 1 1)       (128 1 1)        16        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native18elementwise_kernelILi128ELi4EZNS0_15gpu_kernel_implIZZZNS0_16fill_kernel_cudaERNS_14TensorIteratorEN3c106ScalarEENKUlvE_clEvENKUlvE2_clEvEUlvE_EEvS4_RKT_EUliE2_EEviT1_ [1218]
5.15016s  201.95us          (87616 1 1)       (512 1 1)        32        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native18elementwise_kernelILi512ELi1EZNS0_15gpu_kernel_implIZZZNS0_21copy_device_to_deviceERNS_14TensorIteratorEbENKUlvE_clEvENKUlvE2_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_ [1241]
5.15126s  1.8560us                    -               -         -         -         -  15.000KB  7.7075GB/s      Device           -  Tesla V100-DGXS         1        37  [CUDA memset]
5.15130s  2.0480us                    -               -         -         -         -  15.000KB  6.9849GB/s      Device           -  Tesla V100-DGXS         1        38  [CUDA memset]
5.15134s  2.0160us                    -               -         -         -         -  15.000KB  7.0958GB/s      Device           -  Tesla V100-DGXS         1        39  [CUDA memset]
5.15137s  2.0160us                    -               -         -         -         -  15.000KB  7.0958GB/s      Device           -  Tesla V100-DGXS         1        40  [CUDA memset]
5.15153s  1.3440us                    -               -         -         -         -      112B  79.473MB/s    Pageable      Device  Tesla V100-DGXS         1         7  [CUDA memcpy HtoD]
5.15375s  2.0480us                    -               -         -         -         -  6.7500KB  3.1432GB/s      Device           -  Tesla V100-DGXS         1         7  [CUDA memset]
5.15381s  768.83us           (1 2 1408)         (8 8 1)        96  3.2500KB        0B         -           -           -           -  Tesla V100-DGXS         1         7  void cudnn::detail::wgrad_alg0_engine<float, int=512, int=6, int=5, int=3, int=3, int=3, bool=1, int=512>(int, int, int, float const *, int, cudnn::detail::wgrad_alg0_engine<float, int=512, int=6, int=5, int=3, int=3, int=3, bool=1, int=512>*, float const , kernel_grad_params, int, float, int, int, int, int) [1444]
5.15458s  1.9070ms             (64 1 1)       (128 1 1)        16      512B        0B         -           -           -           -  Tesla V100-DGXS         1         7  void calc_bias_diff<int=2, float, float, int=128, int=0>(cudnnTensorStruct, float const *, cudnnTensorStruct, float*, float, float, int) [1457]
5.15654s  1.4720us              (1 1 1)       (128 1 1)        16        0B        0B         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native18elementwise_kernelILi128ELi4EZNS0_15gpu_kernel_implIZZZNS0_16fill_kernel_cudaERNS_14TensorIteratorEN3c106ScalarEENKUlvE_clEvENKUlvE2_clEvEUlvE_EEvS4_RKT_EUliE2_EEviT1_ [1476]
5.15673s  1.3440us                    -               -         -         -         -        8B  5.6766MB/s      Device           -  Tesla V100-DGXS         1         7  [CUDA memset]
5.15677s  114.24us           (2 2738 1)       (32 16 1)        32       16B  2.0000KB         -           -           -           -  Tesla V100-DGXS         1         7  _ZN2at6native13reduce_kernelILi512ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_15sum_kernel_implIfffEEvRNS_14TensorIteratorEEUlffE_EEjfLi4EEEEEvT0_ [1496]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
```
