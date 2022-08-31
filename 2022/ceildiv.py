import math

def ceildiv(i, j):
    if j > 0:
        return (i + j - 1) // j
    else:
        return (i + j + 1) // j

for i in range(-10, 10):
    for j in range(-10, 10):
        if j == 0:
            continue
        assert math.ceil(i/j) == ceildiv(i, j)
