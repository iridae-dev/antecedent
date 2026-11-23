# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the effect-width objective (signed reduction of the treatment SE).

Regenerate with:  uv run --script effect_width_reference.py > effect_width_reference.json

Written from the definition at 80 digits: for an OLS design with Gram G and residual variance
s2, the standard error of coefficient t is sqrt(s2 * [G^-1]_tt), computed here by a full
matrix inverse.  Every candidate's reduction is se(baseline) - se(updated design), where the
updated design is built the way the candidate describes:
  more samples n0 -> n1      G scaled by n1 / n0 (rows of the same population)
  environment with a Gram    G + G_env, with the environment's variance if it declares one
  environment without a Gram rows scale as above
  measurement                G bordered by each measured column in turn, variance replaced by
                             the last declared sigma2_after
  intervention               the supplied design Gram and variance.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf


def mat(rows):
    return mp.matrix([[mpf(x) for x in r] for r in rows])


def se(G, s2, t):
    return mp.sqrt(mpf(s2) * (G ** -1)[t, t])


def border(G, cross, self_dot):
    p = G.rows
    out = mp.matrix(p + 1, p + 1)
    for i in range(p):
        for j in range(p):
            out[i, j] = G[i, j]
        out[i, p] = mpf(cross[i])
        out[p, i] = mpf(cross[i])
    out[p, p] = mpf(self_dot)
    return out


def hilbert_like(p, scale):
    return [[scale / (i + j + 1) for j in range(p)] for i in range(p)]


# A well-conditioned correlated 3x3 Gram, treatment is column 1.
G = [[40.0, 12.0, -5.0], [12.0, 55.0, 9.0], [-5.0, 9.0, 31.0]]
S2 = 2.25
T = 1
N = 40
ENV = [[18.0, 4.0, 1.0], [4.0, 22.0, -3.0], [1.0, -3.0, 14.0]]
ENV_S2 = 3.5
INT = [[40.0, 2.0, 0.5], [2.0, 90.0, 1.0], [0.5, 1.0, 31.0]]
INT_S2 = 1.8
MEASURE = {
    7: {"cross": [6.0, -4.0, 3.0], "self_dot": 35.0, "sigma2_after": 1.9},
    8: {"cross": [2.0, 7.0, 1.0, 5.0], "self_dot": 28.0, "sigma2_after": None},
}

g = mat(G)
se0 = se(g, S2, T)
cases = []
cases.append({"name": "sampling", "param": 60, "want": se0 - se(g * mpf(100) / 40, S2, T)})
cases.append({"name": "sampling_zero", "param": 0, "want": mpf(0)})
cases.append({"name": "env_with_gram", "want": se0 - se(g + mat(ENV), ENV_S2, T)})
cases.append({"name": "env_rows_only", "param": 25, "want": se0 - se(g * mpf(65) / 40, S2, T)})
cases.append({"name": "env_rows_zero", "param": 0, "want": mpf(0)})
m1 = border(g, MEASURE[7]["cross"], MEASURE[7]["self_dot"])
cases.append({"name": "measure_one", "vars": [7], "want": se0 - se(m1, 1.9, T)})
m2 = border(m1, MEASURE[8]["cross"], MEASURE[8]["self_dot"])
cases.append({"name": "measure_two_keep_last_sigma", "vars": [7, 8], "want": se0 - se(m2, 1.9, T)})
m3 = border(g, MEASURE[8]["cross"][:3], MEASURE[8]["self_dot"]) if False else None
cases.append({"name": "measure_unmatched", "vars": [99], "want": mpf(0)})
cases.append({"name": "measure_matched_and_unmatched", "vars": [99, 7], "want": se0 - se(m1, 1.9, T)})
cases.append({"name": "intervene", "want": se0 - se(mat(INT), INT_S2, T)})

# Ill-conditioned Gram (Hilbert-like, p = 5): the Cholesky path against the exact inverse.
H = hilbert_like(5, 30.0)
for i in range(5):
    H[i][i] += 1e-3
hg = mat(H)
hse0 = se(hg, 1.0, 2)
ill = {
    "gram": H,
    "treatment_col": 2,
    "n": 30,
    "se0": hse0,
    "sampling_param": 300,
    "sampling_want": hse0 - se(hg * mpf(330) / 30, 1.0, 2),
    "cond": mp.mp.mpf(mp.norm(hg, 1) * mp.norm(hg ** -1, 1)),
}

out = {
    "gram": G,
    "sigma2": S2,
    "treatment_col": T,
    "n": N,
    "env_gram": ENV,
    "env_sigma2": ENV_S2,
    "intervention_gram": INT,
    "intervention_sigma2": INT_S2,
    "measure": {str(k): v for k, v in MEASURE.items()},
    "se0": float(se0),
    "cases": [
        {k: (float(v) if k == "want" else v) for k, v in c.items()} for c in cases
    ],
    "ill": {
        k: (float(v) if k in ("se0", "sampling_want", "cond") else v) for k, v in ill.items()
    },
}
print(json.dumps(out, indent=1))
