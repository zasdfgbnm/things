import os

seed = 6198
#seed = 0x1e92_492a_b192_746d
dataset_length = 865284096 // 2
batch_size = 1024
sequence_length = 2048

import torch

# old method
def indices_from_generator():
    device = 'cuda'
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    indices = torch.randperm(dataset_length, generator=g, device=device)
    print(indices.device)
    return indices.cpu()


# true random (but can't use a seed, so we can't use this method)
def indices_from_system_randomness():
    buffer = os.urandom(dataset_length * 8)
    indices = torch.frombuffer(buffer, dtype=torch.int64)
    indices = indices % dataset_length
    return indices


# numpy version
def indices_from_numpy():
    import numpy
    rng = numpy.random.Generator(numpy.random.PCG64(seed=seed))
    indices = numpy.arange(dataset_length, dtype=numpy.int64)
    rng.shuffle(indices)
    return torch.from_numpy(indices)


indices = indices_from_generator()
true_average = indices.mean(dtype=torch.float64)
batch_averages = indices.unfold(0, batch_size, batch_size).mean(dim=1, dtype=torch.float64)

# make rolling average of batch averages
rolling_average_window = 1000
rolling_batch_averages = batch_averages.unfold(0, rolling_average_window, 1).mean(dim=1)

# plot the results
import matplotlib.pyplot as plt

start_token_count = 400e9
end_token_count = 800e9
start = int(start_token_count / (sequence_length * batch_size))
end = int(end_token_count / (sequence_length * batch_size))
rolling_batch_averages = rolling_batch_averages[start:end]
batch_numbers = [x + (start // batch_size) for x in range(len(rolling_batch_averages))]
token_counts = [start_token_count + ((x + rolling_average_window) * 1024 * 2048) for x in batch_numbers]

fig, ax = plt.subplots()
ax.plot(token_counts, rolling_batch_averages)
ax.set(xlabel="tokens", ylabel="index rolling average")
plt.show()
