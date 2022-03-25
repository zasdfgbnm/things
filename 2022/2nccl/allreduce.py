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
os.environ["NCCL_SOCKET_IFNAME"] = f"{args.namespace}net1"
dist.init_process_group('nccl', init_method="file:///tmp/tmpfile1",
                        rank=rank, world_size=world_size)

# Second PG
os.environ["NCCL_SOCKET_IFNAME"] = f"{args.namespace}net2"
pg = dist.new_group(backend="nccl")

# Data
send1 = torch.empty(1000, device=f"cuda:{rank}").fill_(rank + 1)
send2 = torch.empty(1000000, device=f"cuda:{rank}").fill_(rank + 1)
recv1 = torch.empty(1000, device=f"cuda:{rank}")
recv2 = torch.empty(1000000, device=f"cuda:{rank}")

# Run
torch.cuda.synchronize()
dist.all_reduce(recv1, send1)
torch.cuda.synchronize()
dist.all_reduce(recv2, send2, group=pg)
torch.cuda.synchronize()

print("recv1[:10]", recv1[:10])
print("recv2[:10]", recv2[:10])
