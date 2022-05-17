from regex import B


def f(k, r):
    lambdas = [1.4257, 4.7754, 6.423, 9.3759]
    j = 0
    f = lambda x: x ** k

    n = len(lambdas)
    m = sum((f(x) for x in lambdas)) / n
    trB2r = sum(((f(x) - m) ** (2 * r) for x in lambdas))
    trB2 = sum(((f(x) - m) ** 2 for x in lambdas))
    a = ((n - 1) ** (2 * r - 1)) / (1 + (n - 1) ** (2 * r - 1))
    b = (a * trB2r) ** (1 / (2 * r))
    f1high = round(m + b, 4)
    fnlow = round(m - b, 4)
    c = trB2 / n * (1 / (a * trB2r)) ** (1 / (2 * r))
    f1low = round(m + c, 4)
    fnhigh = round(m - c, 4)
    print(k, r, (f1low, f1high), (fnlow, fnhigh))

for k in range(1, 4):
    for r in range(1, 4):
        f(k, r)
