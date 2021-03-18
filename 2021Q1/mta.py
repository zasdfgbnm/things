import torch
device = "cuda"
dtype = torch.float32
out = [torch.ones(3, device=device, dtype=dtype)]
out = torch._foreach_mul(out, 10.0)
print(out) #all zeros

