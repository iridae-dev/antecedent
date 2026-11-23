# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the anomaly score, its Shapley attribution and the arrow strength.

Regenerate with:  uv run --script anomaly_arrow_reference.py > anomaly_arrow_reference.json

Written from the stated definitions at 80 digits:
  score(v)  = -ln P(|Z| >= |v - c| / s) = -ln erfc(|v - c| / (s sqrt 2))
  empirical reference: c = median(y), s = 1.4826022185056018 * MAD(y); if MAD = 0, s = mean |y - c|,
  and 1 if that is 0 as well.
  chain X -> Y -> Z plus X -> Z with additive noise: node = intercept + sum(beta * parent) + noise,
  the noise of a row is the residual of the observed row, the reference noise is 0, and the game
  v(S) = score(z reconstructed with the noise of S restored and the rest at 0) is averaged over
  all 3! orders of its three players.
  arrow strength: Var(node) from the full covariance of the linear system (solved by the
  matrix inverse (I - B)^-1 diag(sigma^2) (I - B)^-T), variance_contribution(edge) =
  beta^2 Var(parent), and the strength of the edge is the Gaussian Kullback-Leibler divergence
  KL(N(0, S) || N(0, S_cut)) = 1/2 (tr(S_cut^-1 S) - n + ln(det S_cut / det S)) between the joint
  covariance S and the covariance S_cut of the model with the edge cut (the child is fed an
  independent root copy of the parent, same marginal variance; S_cut is the extended covariance
  restricted to the original nodes). For comparison it also records the closed form
  1/2 ln(1 + beta^2 Var(parent) / sigma_child^2) and the normalised form
  -1/2 ln(1 - beta^2 Var(parent) / Var(child)), which equals the divergence only when the edge
  is the child's only parent.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf
MAD_SCALE = mpf("1.482602218505602")


def tail_score(v, c, s):
    z = abs(mpf(v) - mpf(c)) / mpf(s)
    return -mp.log(mp.erfc(z / mp.sqrt(2)))


def median(xs):
    xs = sorted(mpf(x) for x in xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def empirical_reference(ys):
    c = median(ys)
    dev = [abs(mpf(y) - c) for y in ys]
    mad = median(dev)
    if mad > 0:
        return c, MAD_SCALE * mad
    mean_abs = sum(dev) / len(dev)
    return c, (mean_abs if mean_abs > 0 else mpf(1))


# Chain model: x ~ a0 + n; y = a1 + b1 x + n; z = a2 + b2 x + c2 y + n.
MODEL = {"a0": 1.5, "a1": -0.5, "b1": 0.8, "a2": 2.0, "b2": -1.2, "c2": 0.7}
ROWS = [
    [1.5, 0.7, 2.3],      # exactly on the model mean path: zero noise everywhere
    [2.0, 1.9, 4.5],
    [0.0, -1.1, -3.0],
    [4.5, 3.0, 0.5],
    [1.7, 0.2, 40.0],     # extreme target
    [1.4, 0.9, -250.0],   # extreme, far tail (z beyond the asymptotic switch)
    [3.0, -9.0, 2.0],
]


def reconstruct(row, mask):
    m = {k: mpf(v) for k, v in MODEL.items()}
    nx = mpf(row[0]) - m["a0"]
    ny = mpf(row[1]) - (m["a1"] + m["b1"] * mpf(row[0]))
    nz = mpf(row[2]) - (m["a2"] + m["b2"] * mpf(row[0]) + m["c2"] * mpf(row[1]))
    ex = nx if mask[0] else mpf(0)
    ey = ny if mask[1] else mpf(0)
    ez = nz if mask[2] else mpf(0)
    x = m["a0"] + ex
    y = m["a1"] + m["b1"] * x + ey
    return m["a2"] + m["b2"] * x + m["c2"] * y + ez


def shapley(row, c, s):
    v = {}
    for mask in itertools.product((0, 1), repeat=3):
        v[mask] = tail_score(reconstruct(row, mask), c, s)
    phi = [mpf(0)] * 3
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        mask = [0, 0, 0]
        for p in perm:
            before = v[tuple(mask)]
            mask[p] = 1
            phi[p] += v[tuple(mask)] - before
    return [x / len(perms) for x in phi], v[(0, 0, 0)], v[(1, 1, 1)]


def fmt(x):
    return float(x)


def anomaly_case(ref_kind, center=None, scale=None):
    zs = [r[2] for r in ROWS]
    if ref_kind == "empirical":
        c, s = empirical_reference(zs)
    else:
        c, s = mpf(center), mpf(scale)
    out = {"reference": ref_kind, "center": fmt(c), "scale": fmt(s), "rows": []}
    for row in ROWS:
        phi, base, full = shapley(row, c, s)
        out["rows"].append(
            {"score": fmt(full), "baseline": fmt(base), "phi": [fmt(p) for p in phi]}
        )
    return out


# A degenerate bulk: more than half of the target column identical (MAD = 0) and the mean-abs fallback.
DEGENERATE_Z = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 9.0, -1.0]
c_d = median(DEGENERATE_Z)
c_all_same, s_all_same = empirical_reference([5.0] * 6)
c_d, s_d = empirical_reference(DEGENERATE_Z)

# Far tail: the product switches to the log-space asymptotic once 2 sf(z) underflows.
TAIL = [{"z": z, "score": fmt(tail_score(z, 0, 1))} for z in (0.0, 0.5, 1.0, 5.0, 12.0, 30.0, 37.0, 38.0, 38.5, 39.0, 40.0, 100.0, 1000.0)]

# Arrow strengths on a diamond with a chord: a -> b, a -> c, b -> d, c -> d, a -> d.
NODES = ["a", "b", "c", "d"]
SIGMA = {"a": 1.3, "b": 0.6, "c": 0.9, "d": 0.4}
BETA = {("a", "b"): 0.8, ("a", "c"): -1.1, ("b", "d"): 0.5, ("c", "d"): 0.9, ("a", "d"): 0.3}


def arrow_reference():
    n = len(NODES)
    B = mp.matrix(n, n)
    for (p, c), b in BETA.items():
        B[NODES.index(c), NODES.index(p)] = mpf(b)
    inv = (mp.eye(n) - B) ** -1
    D = mp.diag([mpf(SIGMA[k]) ** 2 for k in NODES])
    cov = inv * D * inv.T
    var = {k: cov[i, i] for i, k in enumerate(NODES)}
    edges = []
    for (p, c), b in BETA.items():
        contribution = mpf(b) ** 2 * var[p]
        ratio = contribution / var[c]
        # Extended system: node 0 is the independent copy of p, with p's marginal variance.
        Be = mp.matrix(n + 1, n + 1)
        for (pp, cc), bb in BETA.items():
            src = 0 if (pp, cc) == (p, c) else NODES.index(pp) + 1
            Be[NODES.index(cc) + 1, src] = mpf(bb)
        De = mp.diag([var[p]] + [mpf(SIGMA[k]) ** 2 for k in NODES])
        inve = (mp.eye(n + 1) - Be) ** -1
        cove = inve * De * inve.T
        cut = cove[1:, 1:]
        divergence = (mp.fsum(((cut ** -1) * cov)[i, i] for i in range(n)) - n + mp.log(mp.det(cut) / mp.det(cov))) / 2
        closed = mp.log(1 + contribution / mpf(SIGMA[c]) ** 2) / 2
        assert abs(divergence - closed) < mpf(10) ** -60
        n_parents = sum(1 for (_, cc) in BETA if cc == c)
        edges.append(
            {
                "parent": p,
                "child": c,
                "beta": b,
                "strength": fmt(divergence),
                "variance_contribution": fmt(contribution),
                "normalised_form": fmt(-mp.log(1 - ratio) / 2) if ratio < 1 else None,
                "child_parents": n_parents,
                "ratio": fmt(ratio),
            }
        )
    return {"variance": {k: fmt(v) for k, v in var.items()}, "edges": edges}


result = {
    "model": MODEL,
    "rows": ROWS,
    "fixed": anomaly_case("fixed", 1.0, 2.5),
    "empirical": anomaly_case("empirical"),
    "degenerate": {
        "z": DEGENERATE_Z,
        "center": fmt(c_d),
        "scale": fmt(s_d),
        "all_same_scale": fmt(s_all_same),
    },
    "tail": TAIL,
    "arrow": arrow_reference(),
}
print(json.dumps(result, indent=1))
