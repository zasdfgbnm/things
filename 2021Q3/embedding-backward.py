import torch
import itertools

num_embeddings = (119547, 50265, 32000, 8000, 3052)
num_tokens = (4096, 16384)
hidden_sizes = (512, 768)

for ne, nt, nh in itertools.product(num_embeddings, num_tokens, hidden_sizes):
    print(f"Embedding size: {ne}, Tokens: {nt}, Hidden size: {nh}")
    embedding = torch.nn.Embedding(ne, nh).cuda()
    input_ = torch.randint(ne, (nt,), device='cuda')
    out = embedding(input_)
    torch.cuda.synchronize()
    out.backward(out, retain_graph=True); torch.cuda.synchronize()