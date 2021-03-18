import math

q = 0.9

def left(M, N):
    return (M / (M - N)) ** (M - N + 0.5)

def right(N):
    return q / math.sqrt(2 * math.pi) * math.exp(N + 1)

for i in range(65):
    N = 2 ** i
    j = i + 1
    M = 2 ** j
    while left(M, N) <= right(N):
        j += 1
        M = 2 ** j
    print(i, j)
