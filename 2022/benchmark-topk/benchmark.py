import torch
import time
import torch
import pandas as pd
import numpy as np
import torch.utils.benchmark as benchmark


torch.manual_seed(1)
dtype = torch.float
data = []

for d1 in [1, 10, 20, 40, 60, 80, 100, 200, 400, 800, 1000, 2000, 4000, 6000, 8000, 10000, 100000, 500000]:
    if d1 <= 1000:
        D2 = [100, 200, 300, 400, 800, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 20000, 30000, 40000, 80000, 100000, 200000, 300000, 400000, 500000]
    else:
        D2 = [100, 200, 300, 400, 800, 1000, 5000, 10000, 20000, 30000]
    for d2 in D2:
        k = 2000 if d2 >= 2000 else d2 // 2
        print(f"----------------- D1 = {d1}, D2 = {d2} -----------------")
        try:
            x = torch.randn((d1, d2), dtype=dtype, device="cuda")
            m = benchmark.Timer(
                stmt='x.topk(k=k, dim=1, sorted=False, largest=True)',
                globals={'x': x, 'k': k},
                num_threads=1,
            ).blocked_autorange(min_run_time=1)
            print(m)
            time_ms = m.median * 1000
        except RuntimeError: # OOM
            time_ms = -1
        data.append([d1, d2, k, time_ms])

df = pd.DataFrame(data=data, columns=['D1', 'D2', 'k', 'time(ms)'])
print(df)
df.to_csv('benchmark.csv')