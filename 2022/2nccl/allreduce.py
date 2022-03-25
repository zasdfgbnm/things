import os
import torch
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("namespace")
parser.add_argument("rank", type=int)
args = parser.parse_args()

rank = args.rank
world_size = 2

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# First PG
print("Initializing First PG")
os.environ["NCCL_SOCKET_IFNAME"] = f"{args.namespace}net1"
dist.init_process_group('nccl', init_method="file:///tmp/tmpfile1",
                        rank=rank, world_size=world_size)
print("First PG Initialized")

# Second PG
print("Initializing Second PG")
os.environ["NCCL_SOCKET_IFNAME"] = f"{args.namespace}net2"
pg = dist.new_group(backend="nccl")
print("Second PG Initialized")

# Data
t1 = torch.empty(1000, device=f"cuda:{rank}").fill_(rank + 1)
t2 = torch.empty(1000000, device=f"cuda:{rank}").fill_(rank + 1)

# Run
torch.cuda.synchronize()
print("all reduce 1")
work = dist.all_reduce(t1)
# work.wait()
torch.cuda.synchronize()
print("recv1[:10]", t1[:10])

print("all reduce 2")
work = dist.all_reduce(t2, group=pg)
# work.wait()
torch.cuda.synchronize()
print("recv2[:10]", t2[:10])
