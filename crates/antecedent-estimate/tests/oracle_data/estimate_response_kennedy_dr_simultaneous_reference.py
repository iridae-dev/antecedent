"""Fixed-grid multiplier band from an influence matrix, from the definition.

Regenerate with:  python3 estimate_response_kennedy_dr_simultaneous_reference.py > estimate_response_kennedy_dr_simultaneous_reference.json

For a G x n influence matrix psi with row standard errors se_g = sqrt(sum_i psi_gi^2), the band draws
B sign vectors from a SplitMix64 stream seeded with seed xor 0x9E3779B97F4A7C15 (each row consumes one
64-bit output, sign minus when its low bit is zero), forms for every draw the maximum over the grid of
|sum_i psi_gi eps_i| / se_g, sorts the B maxima and takes the ceil(level (B + 1) - 1e-9)-th smallest as
the critical value (infinite when that rank exceeds B).  The band is mean +- max(critical, z) se with
z the normal quantile.  The influence matrices are dyadic rationals and the sums are exact.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import math
from fractions import Fraction as F

M = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
_state = 0x1B873593CC9E2D51


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def mix64(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


class SplitMix:
    def __init__(self, seed):
        self.state = (seed ^ GAMMA) & M

    def next(self):
        self.state = (self.state + GAMMA) & M
        return mix64(self.state)


def band(psi, level, B, seed):
    G, n = len(psi), len(psi[0])
    se = [math.sqrt(float(sum(v * v for v in row))) for row in psi]
    rng = SplitMix(seed)
    maxima = []
    for _ in range(B):
        eps = [-1 if rng.next() & 1 == 0 else 1 for _ in range(n)]
        maxima.append(max(abs(float(sum(psi[g][i] * eps[i] for i in range(n)))) / se[g] for g in range(G)))
    maxima.sort()
    rank = math.ceil(level * (B + 1) - 1e-9)
    rank = max(rank, 1)
    critical = float("inf") if rank > B else maxima[rank - 1]
    return se, maxima, rank, critical


cases = []
for name, G, n, B, level, seed, scale in [
    ("three_points_ordinary", 3, 40, 400, 0.95, 7, F(1, 4)),
    ("one_point", 1, 25, 200, 0.9, 11, F(1)),
    ("nearly_collinear_columns", 4, 60, 500, 0.95, 12345, F(1, 8)),
    ("rank_does_not_exist", 2, 30, 100, 0.995, 3, F(1, 2)),
    ("rank_is_the_maximum", 2, 30, 100, 0.99, 3, F(1, 2)),
]:
    common = [F(int(round((u01() - 0.5) * 4096)), 1024) for _ in range(n)]
    psi = []
    for g in range(G):
        if name == "nearly_collinear_columns":
            row = [common[i] + F(int(round((u01() - 0.5) * 16)), 4096) for i in range(n)]
        else:
            row = [F(int(round((u01() - 0.5) * 4096)), 1024) * scale for _ in range(n)]
        psi.append(row)
    se, maxima, rank, critical = band(psi, level, B, seed)
    cases.append({
        "name": name, "level": level, "replicates": B, "seed": seed,
        "psi": [[str(float(v)) for v in row] for row in psi],
        "se": se, "rank": rank,
        "critical": None if critical == float("inf") else critical,
        "maxima_at_rank_neighbours": [maxima[max(rank - 2, 0)], maxima[min(rank - 1, B - 1)]],
    })
first = SplitMix(2024)
print(json.dumps({"cases": cases, "stream": {"seed": 2024, "first_outputs": [str(first.next()) for _ in range(6)]}}, indent=1))
