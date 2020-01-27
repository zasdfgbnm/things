import torchvision
import torch

model = torchvision.models.resnet50().cuda()
input_ = torch.randn(64, 3, 224, 224, device="cuda")
torch.cuda.synchronize()

output_ = model(input_)
torch.cuda.synchronize()
print("==============================")

target = torch.empty(64, dtype=torch.long, device="cuda").random_(1000)
loss = torch.nn.CrossEntropyLoss()
o = torch.optim.Adam(model.parameters())
torch.cuda.synchronize()

print("==============================")
l = loss(output_, target)
l.backward()
torch.cuda.synchronize()
print("==============================")

o.step()
torch.cuda.synchronize()
