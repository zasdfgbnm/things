import torch
import torchvision.models as models

torch.backends.cudnn.benchmark = True

dtype = torch.float
m0 = models.resnet50(pretrained=True).to(device='cuda:0', dtype=dtype)
ipt0 = torch.randn((64, 3, 224, 224), dtype=dtype, device='cuda:0')

m1 = models.resnet50(pretrained=True).to(device='cuda:1', dtype=dtype)
ipt1 = torch.randn((64, 3, 224, 224), dtype=dtype, device='cuda:1')

m0(ipt0).sum().backward()
m1(ipt1).sum().backward()
