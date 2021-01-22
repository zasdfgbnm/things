import torch
torch.randn(5, requires_grad=True).sum().backward()
print("done")
