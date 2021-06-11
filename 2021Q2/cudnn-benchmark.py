import torch
torch.backends.cudnn.benchmark = True

conv = torch.nn.Conv2d(1, 1, 1).cuda()
input = torch.randn((100, 1, 100, 100), device='cuda')
conv(input)
