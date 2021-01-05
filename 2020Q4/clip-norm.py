import torch
import torch.nn as nn
import time

print(torch.__version__)

model = nn.Sequential(*(nn.Linear(512, 512) for i in range(100))).cuda()
elapsed = 0

for _ in range(1):
    inp = torch.rand(32, 512).cuda()
    out = model(inp)
    out.backward(torch.ones_like(out))

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-7)
    torch.cuda.synchronize()
    t2 = time.time()
    elapsed += (t2 - t1) * 1000


print(elapsed, 'ms')
