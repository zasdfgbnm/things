import torchvision
import torch

print("running resnet50")
model = torchvision.models.resnet50().cuda()
input_ = torch.randn(64, 3, 224, 224, device="cuda")
torch.cuda.synchronize()

print("==============================")
print("forward")
output_ = model(input_)
torch.cuda.synchronize()
print("end forward")
print("==============================")

target = torch.empty(64, dtype=torch.long, device="cuda").random_(1000)
loss = torch.nn.CrossEntropyLoss()
o = torch.optim.Adam(model.parameters())
torch.cuda.synchronize()

print("==============================")
print("backward")
l = loss(output_, target)
l.backward()
torch.cuda.synchronize()
print("end backward")
print("==============================")

print("optimizer.step()")
o.step()
torch.cuda.synchronize()
print("end optimizer.step()")
print("==============================")
