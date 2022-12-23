import torch

def my_index_select(self, dim, index):
    broadcast_shape = [1] * len(self.shape)
    broadcast_shape[dim] = index.shape[0]
    expand_shape = list(self.shape)
    expand_shape[dim] = index.shape[0]
    index = index.view(broadcast_shape).expand(expand_shape)
    return self.gather(dim, index)

x = torch.randn(5, 6, 7)
index = torch.tensor([1, 2, 3])

assert torch.eq(x.index_select(0, index), my_index_select(x, 0, index)).all()
assert torch.eq(x.index_select(1, index), my_index_select(x, 1, index)).all()
assert torch.eq(x.index_select(2, index), my_index_select(x, 2, index)).all()
