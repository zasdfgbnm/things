import math

q = 0.9

def left(M, N):
    term1 = (M - N + 0.5) * math.log(M / (M - N))
    term2 = 1 / (12 * M + 1) - 1 / (12 * (M - N))
    return term1 + term2

def right(N):
    return N + math.log(q)

for i in range(65):
    N = 2 ** i
    j = i + 1
    M = 2 ** j
    while left(M, N) <= right(N):
        print(left(M, N), right(N))
        j += 1
        M = 2 ** j
    print(i, j)
