import torch
import timeit

cfgs = [
    {"dtype": torch.float16, "dim": 0, "sizes": [torch.Size([0]),torch.Size([128, 16, 1024])]},
    {"dtype": torch.float16, "dim": 0, "sizes": [torch.Size([19997, 1024]),torch.Size([3, 1024])]},
    {"dtype": torch.float16, "dim": 0, "sizes": [torch.Size([19997]),torch.Size([3])]},
    {"dtype": torch.float16, "dim": -1, "sizes": [torch.Size([128, 512]),torch.Size([128, 512])]},
    {"dtype": torch.float16, "dim": 3, "sizes": [torch.Size([16, 16, 128, 1]),torch.Size([16, 16, 128, 128])]},
    {"dtype": torch.int64, "dim": 0, "sizes": [torch.Size([1664, 16]),torch.Size([15348, 16])]},
]

torch.arange(888888, device='cuda')

for c in cfgs:
    tensors = [torch.zeros(shape, dtype=c['dtype'], device='cuda') for shape in c['sizes']]
    def f():
        torch.cat(tensors, dim=c['dim'])
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(timeit.timeit(f, number=10000))
