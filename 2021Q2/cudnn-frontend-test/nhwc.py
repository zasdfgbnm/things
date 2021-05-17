import torch
from torch import nn

dtype = torch.float
device = 'cuda'

def assertTrue(x):
    assert x

def assertEqual(a, b, exact_dtype=True):
    diff = (a - b).abs().max()
    assert diff < 1e-6, f"diff = {diff}"

def helper(n, c, h, w, out_channels, kernel_size, groups):
    input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device)\
        .to(memory_format=torch.channels_last)
    input.requires_grad_()
    conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)\
        .to(device='cuda', dtype=dtype, memory_format=torch.channels_last)
    for p in conv.parameters():
        p.data = torch.randint_like(p, -3, 3)

    # use FP64 channels-first conv as reference
    ref_input = input.detach().clone().contiguous().double().requires_grad_()
    ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
    # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
    ref_conv.load_state_dict(conv.state_dict())
    ref_conv = ref_conv.to(device='cuda', dtype=torch.double, memory_format=torch.contiguous_format)

    out = conv(input)
    ref_out = ref_conv(ref_input)

    print("input.shape =", input.shape)
    print("out.shape =", out.shape)
    print("conv.weight.shape =", conv.weight.shape)
    print("conv.padding =", conv.padding)
    print("conv.stride =", conv.stride)
    print("conv.dilation =", conv.dilation)

    grad = torch.randint_like(out, -3, 3)
    ref_grad = grad.detach().clone().double().contiguous()

    out.backward(grad)
    ref_out.backward(ref_grad)

    assertTrue(out.is_contiguous(memory_format=torch.channels_last))
    assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
    assertTrue(conv.weight.grad.is_contiguous(memory_format=torch.channels_last))

    assertTrue(ref_out.is_contiguous())
    assertTrue(ref_input.grad.is_contiguous())
    assertTrue(ref_conv.weight.grad.is_contiguous())

    assertEqual(out, ref_out, exact_dtype=False)
    assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
    assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
    assertEqual(input.grad, ref_input.grad, exact_dtype=False)

helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)