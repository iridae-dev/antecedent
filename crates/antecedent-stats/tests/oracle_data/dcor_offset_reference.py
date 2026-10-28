# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Distance correlation in exact rational arithmetic, for series far from zero.

Regenerate with:  uv run --offline --script dcor_offset_reference.py > dcor_offset_reference.json

The data are the exact values of the doubles written to the file.  The V-statistic distance
covariances are formed from the double-centred L1 distance matrices of the definition (Szekely,
Rizzo and Bakirov 2007) with Fractions, and the correlation sqrt(dCov2_xy / sqrt(dVar2_x dVar2_y))
is taken to 50 digits.  No sorting or Fenwick identity of the implementation is shared.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60
_state = 0x1F83D9ABFB41BD6B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def centred_distances(s):
    n = len(s)
    a = [[abs(s[i] - s[j]) for j in range(n)] for i in range(n)]
    row = [sum(a[i]) / n for i in range(n)]
    grand = sum(row) / n
    return [[a[i][j] - row[i] - row[j] + grand for j in range(n)] for i in range(n)]


def dcor(x, y):
    n = len(x)
    X = [F(v) for v in x]
    Y = [F(v) for v in y]
    a, b = centred_distances(X), centred_distances(Y)
    dxy = sum(a[i][j] * b[i][j] for i in range(n) for j in range(n)) / (n * n)
    dxx = sum(a[i][j] * a[i][j] for i in range(n) for j in range(n)) / (n * n)
    dyy = sum(b[i][j] * b[i][j] for i in range(n) for j in range(n)) / (n * n)
    dec = lambda q: Decimal(q.numerator) / Decimal(q.denominator)
    return str((dec(dxy) / (dec(dxx) * dec(dyy)).sqrt()).sqrt())


def series(n, ties):
    x = [u01() * 4 - 2 for _ in range(n)]
    if ties:
        x = [round(v * 3) / 3 for v in x]
    y = [v * v + 0.3 * (u01() - 0.5) for v in x]
    if ties:
        y = [round(v * 2) / 2 for v in y]
    return x, y


def case(name, x, y, x_offset, y_offset):
    xs = [v + x_offset for v in x]
    ys = [v + y_offset for v in y]
    return {"name": name, "x": xs, "y": ys, "x_offset": x_offset, "y_offset": y_offset,
            "dcor": dcor(xs, ys)}


cases = []
x, y = series(40, False)
cases.append(case("no_offset", x, y, 0.0, 0.0))
cases.append(case("offset_1e6_both", x, y, 1.0e6, 1.0e6))
cases.append(case("offset_1e8_x_only", x, y, 1.0e8, 0.0))
cases.append(case("offset_1e9_y_only", x, y, 0.0, 1.0e9))
x, y = series(37, True)
cases.append(case("ties_offset_1e6", x, y, 1.0e6, -1.0e6))
x, y = series(9, False)
cases.append(case("few_rows_offset_1e7", x, y, 1.0e7, 1.0e7))
print(json.dumps({"cases": cases}, separators=(",", ":")))
