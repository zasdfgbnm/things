# torchrun --nproc_per_node 2 ucc_barrier.py

import torch
import torch.distributed as dist
import time

dist.init_process_group("ucc")
rank = dist.get_rank()

x = torch.zeros(5, device=f"cuda:{rank}")
dist.all_reduce(x)

for i in range(10):
    time.sleep(rank)

    print(f"{i}:1")
    dist.barrier()
    print(f"{i}:2")
