import torch

def mykron(a, b):
    assert a.dim() == b.dim()
    a_view_shape = []
    b_view_shape = []
    ab_view_shape = []
    for i in range(a.dim()):
        a_view_shape.append(a.size(i))
        a_view_shape.append(1)
        b_view_shape.append(1)
        b_view_shape.append(b.size(i))
        ab_view_shape.append(a.size(i) * b.size(i))
    return (a.reshape(a_view_shape) * b.reshape(b_view_shape)).reshape(ab_view_shape)

a = torch.randn(5, 5, 5)
b = torch.randn(6, 6, 6)

r1 = torch.kron(a, b)
r2 = mykron(a, b)

diff = (r1 - r2).abs().max()
print(diff)