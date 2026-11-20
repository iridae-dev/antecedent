"""Independent exact-moment calculation of the nonorthogonal three-parent test.

Integrate the Taylor polynomial of exp(-RSS/2), using signed-uniform moments.
All input numbers are tenths, so polynomial coefficients are integers after
scaling -RSS/2 by 200. For these rows RSS/2 <= 3.2 over every sign box.
The degree-36 Taylor remainder is bounded by 3.2**37 / 37! < 4e-25.
"""
from collections import defaultdict
from decimal import Decimal, getcontext
from math import factorial

getcontext().prec = 60
rows = [(2, -4, 1, 3), (-3, 1, -2, -4), (1, 2, 3, 2)]
dim = 3
zero = (0,) * dim
quadratic = defaultdict(int)
for *x, y in rows:
    terms = [(zero, y)]
    for i in range(dim):
        terms.append((tuple(int(j == i) for j in range(dim)), -x[i]))
    for a, ca in terms:
        for b, cb in terms:
            quadratic[tuple(i + j for i, j in zip(a, b))] -= ca * cb

def moment(k):
    return Decimal(0) if k % 2 else 2 * (1 - Decimal(2) ** (-k - 1)) / (k + 1)

moments = [moment(i) for i in range(74)]
power = {zero: 1}
integral = Decimal(1)
for k in range(1, 37):
    product = defaultdict(int)
    for (a, b, c), v in power.items():
        for (i, j, l), w in quadratic.items():
            product[a + i, b + j, c + l] += v * w
    power = product
    value = sum(v * moments[a] * moments[b] * moments[c]
                for (a, b, c), v in power.items() if not (a | b | c) % 2)
    integral += value / (200**k * factorial(k))
pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
print(integral.ln() - Decimal(len(rows)) / 2 * (2 * pi).ln())
