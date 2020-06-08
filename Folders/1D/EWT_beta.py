def EWT_beta(x):
    bm = (x >= 0)*(x <= 1)*(x**4 * (35 - 84*x + 70*x**2 - 20*x**3))
    bm += (x > 1)
    return bm