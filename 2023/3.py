import math

x = math.log(3, 2)

print(x)

def error(i):
    return abs(i * x - round(i * x))

nums = [i for i in range(1000)]
sorted_nums = sorted(nums, key=error)

for i in sorted_nums:
    print(i, error(i))
