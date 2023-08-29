import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id1(fd : FusionDefinition) -> None :                                                                                      
    T0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False)
    T1 = fd.define_tensor(symbolic_sizes=[1, 1, -1, -1], contiguity=[None, None, True, True], dtype=DataType.Float, is_cpu=False)
    S2 = fd.define_constant(0.125000, dtype=DataType.Double)
    T3 = fd.ops.mul(T0, S2)                               
    T4 = fd.ops.slice(T1, start_indices=[0, 0, 0, 0], end_indices=[1, 1, 128, 128], strides=[1, 1, 1, 1])                                    
    S5 = fd.define_constant(0.00000, dtype=DataType.Double)
    T6 = fd.ops.eq(T4, S5)                                     
    T7 = fd.ops.broadcast_in_dim(T6, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3])
    S8 = fd.define_constant(float("-inf"), dtype=DataType.Double)
    T9 = fd.ops.where(T7, S8, T3)
    T10 = fd.ops.max(T9, axes=[3], keepdim=False, dtype=DataType.Null) 
    T11 = fd.ops.broadcast_in_dim(T10, output_shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2])
    T12 = fd.ops.broadcast_in_dim(T11, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3])
    T13 = fd.ops.sub(T9, T12)                                                                                                                
    T14 = fd.ops.exp(T13)                                                                                                                    
    T15 = fd.ops.sum(T14, axes=[3], keepdim=False, dtype=DataType.Null)                              
    T16 = fd.ops.broadcast_in_dim(T15, output_shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2])      
    T17 = fd.ops.broadcast_in_dim(T16, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3])
    T18 = fd.ops.div(T14, T17)                           
    S19 = fd.define_constant(0.00000, dtype=DataType.Double)
    S20 = fd.define_constant(1.00000, dtype=DataType.Double)
    S21 = fd.define_constant(16, dtype=DataType.Int) 
    S22 = fd.define_constant(16, dtype=DataType.Int)                                                                                           
    S23 = fd.define_constant(128, dtype=DataType.Int)        
    S24 = fd.define_constant(128, dtype=DataType.Int)
    T25 = fd.ops.uniform(S19, S20, shape=[S21, S22, S23, S24], dtype=DataType.Float)
    S26 = fd.define_constant(0.900000, dtype=DataType.Double)
    T27 = fd.ops.lt(T25, S26)
    T28 = fd.ops.cast(T27, dtype=DataType.Float)
    T29 = fd.ops.mul(T18, T28)
    S30 = fd.define_constant(1.11111, dtype=DataType.Double)
    T31 = fd.ops.mul(T29, S30)
    fd.add_output(T31)

with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

inputs = [
    torch.randn(16, 16, 128, 128, device='cuda'),
    torch.randn(1, 1, 128, 128, device='cuda'),
]

for _ in range(5):
    out = fd.execute(inputs)
