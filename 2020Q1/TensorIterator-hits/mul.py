import torch

a = torch.zeros(1048576, device='cuda')
a.mul_(0.9)
