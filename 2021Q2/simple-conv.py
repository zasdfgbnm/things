import torch
import torch.utils.benchmark

conv = torch.nn.Conv2d(1, 1, 1).cuda()
input = torch.randn((100, 1, 100, 100), device='cuda')

t = torch.utils.benchmark.Timer(
    stmt='conv(input)',
    setup='import torch; conv = torch.nn.Conv2d(1, 1, 1).cuda(); input = torch.randn((100, 1, 100, 100), device="cuda")')

result = t.timeit()
print(result)

result = t.collect_callgrind()
print(result)
