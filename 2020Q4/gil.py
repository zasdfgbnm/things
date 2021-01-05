import torch
import torch.nn as nn

model = nn.Sequential(*(nn.Linear(512, 512) for i in range(100)))
inp = torch.rand(32, 512)
out = model(inp)
out.backward(torch.ones_like(out))
