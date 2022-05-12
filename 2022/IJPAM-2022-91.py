lambdas = [10, 0, 0, 0]
r = 2
j = 0
f = lambda x: x

n = len(lambdas)
m = sum((f(x) for x in lambdas)) / n
left = sum(((f(x) - m) ** (2 * r) for x in lambdas))
right = (1 + (n - 1) ** (2 * r - 1)) / ((n - 1) ** (2 * r - 1)) * (f(lambdas[j]) - m) ** (2 * r)
print(left, right)
