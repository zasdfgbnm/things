import torch

torch.backends.cudnn.allow_tf32 = False

conv_cpu = torch.nn.Conv2d(5, 4, 3)
input_cpu = torch.randn((64, 5, 100, 100))

input_cuda = input_cpu.cuda()
conv_cuda = torch.nn.Conv2d(5, 4, 3)
conv_cuda.load_state_dict(conv_cpu.state_dict())
conv_cuda.cuda()

input_cpu.requires_grad_()
input_cuda.requires_grad_()

output_cpu = conv_cpu(input_cpu)
grad_output_cpu = torch.randn_like(output_cpu)
output_cpu.backward(grad_output_cpu)

output_cuda = conv_cuda(input_cuda)
grad_output_cuda = grad_output_cpu.cuda()
output_cuda.backward(grad_output_cuda)

print((output_cpu - output_cuda.cpu()).abs().max())
print((input_cpu.grad - input_cuda.grad.cpu()).abs().max())
print((conv_cpu.weight.grad - conv_cuda.weight.grad.cpu()).abs().max())
print((conv_cpu.bias.grad - conv_cuda.bias.grad.cpu()).abs().max())
