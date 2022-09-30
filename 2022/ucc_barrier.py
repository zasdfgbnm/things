import torch
import torch.distributed as dist
import time

dist.init_process_group("ucc")
rank = dist.get_rank()

x = torch.zeros(5, device=f"cuda:{rank}")
dist.all_reduce(x)

time.sleep(rank)

print("1")
dist.barrier()
print("2")

