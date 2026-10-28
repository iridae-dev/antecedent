# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""50-digit reference for the plug-in and the multi-dataset delta-method standard error of the
sampled two-source joint counterfactual transport estimate.

Regenerate with:  uv run --offline --script sampled_transport_multi_table_reference.py > sampled_transport_multi_table_reference.json

The estimand is P*(Y_{x=1} = 1, X = 0) of the confounded covariate model with a selection node
on Z.  Its identified functional reads THREE independent datasets (count tables): the source
observational joint over (Z, X, Y) (8 cells, index z*4 + x*2 + y), the source do(X = 1)
experiment over (Z, Y) (4 cells, index z*2 + y) and the target marginal of Z (2 cells):

    f(w) = sum_z  Wt(z) / sum Wt  *  [  Wd(z,1) / (Wd(z,0) + Wd(z,1))
                                       - Wo(z,1,1) / sum_{x,y} Wo(z,x,y)  ]

written for homogeneous cell weights w, so that the multinomial delta-method variance of one
dataset of N independent units with cell shares p is

    (1 / N) [ sum_c p_c g_c^2 - ( sum_c p_c g_c )^2 ],     g_c = d f / d w_c  at  w = p,

and the variance of the estimator is the sum over the three independent datasets (nothing is
shared between them).  Gradients are 50-digit central differences of the closed form, step 1e-25.
With additive smoothing a (applied to a table with an unseen cell only), that table's shares are
(n + a) / (N + K a) over its K cells and the variance uses those shares with the table's own N.
Nothing here calls the product.  Repository-authored (MIT OR Apache-2.0).
"""
import json

import mpmath as mp

mp.mp.dps = 50
H = mp.mpf(10) ** -25


def f(wo, wd, wt):
    total = mp.mpf(0)
    nt = wt[0] + wt[1]
    for z in range(2):
        d_z = wd[z * 2] + wd[z * 2 + 1]
        o_z = sum(wo[z * 4 + k] for k in range(4))
        total += (wt[z] / nt) * (wd[z * 2 + 1] / d_z - wo[z * 4 + 3] / o_z)
    return total


def shares(counts, smoothing):
    n = sum(counts)
    if smoothing is not None and 0 in counts:
        a = mp.mpf(smoothing)
        k = len(counts)
        return [(mp.mpf(c) + a) / (n + k * a) for c in counts]
    return [mp.mpf(c) / n for c in counts]


def case(name, obs, do1, tgt, smoothing=None):
    po, pd, pt = shares(obs, smoothing), shares(do1, smoothing), shares(tgt, smoothing)
    tables = [po, pd, pt]
    ns = [sum(obs), sum(do1), sum(tgt)]
    point = f(po, pd, pt)
    var = mp.mpf(0)
    for t in range(3):
        grads = []
        for c in range(len(tables[t])):
            up = [list(x) for x in tables]
            dn = [list(x) for x in tables]
            up[t][c] += H
            dn[t][c] -= H
            grads.append((f(*up) - f(*dn)) / (2 * H))
        p = tables[t]
        s1 = sum(pc * g for pc, g in zip(p, grads))
        s2 = sum(pc * g * g for pc, g in zip(p, grads))
        var += (s2 - s1 * s1) / ns[t]
    return {"name": name, "obs": obs, "do1": do1, "target": tgt, "smoothing": smoothing,
            "point": float(point), "se": float(mp.sqrt(var))}


def main():
    out = [
        case("balanced", [120, 90, 40, 150, 70, 55, 30, 45], [100, 60, 80, 60], [90, 110]),
        case("tiny_samples", [4, 3, 2, 5, 3, 3, 1, 4], [6, 3, 5, 6], [8, 12]),
        case("unbalanced_sizes", [20000, 15000, 7000, 30000, 12000, 9000, 6000, 1000],
             [40, 30, 25, 45], [5, 995]),
        case("rare_cells", [100000, 1, 90000, 2, 3, 100000, 4, 80000], [100000, 1, 100, 100000],
             [1, 1000000]),
        case("one_cell_unseen_smoothed", [60, 0, 30, 25, 40, 22, 18, 35], [50, 40, 30, 60], [40, 70],
             smoothing=0.5),
        case("one_cell_unseen_smoothed_small", [8, 0, 4, 3, 5, 3, 2, 4], [6, 5, 4, 7], [9, 12],
             smoothing=1.0),
    ]
    json.dump(out, open("/dev/stdout", "w"), separators=(",", ":"))


main()
