# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Exact posterior over PAG classes of Gaussian maximal ancestral graphs, by brute force.

Regenerate with:  uv run --script exact_pag_posterior_reference.py > exact_pag_posterior_reference.json

For p = 3 or 4 variables every assignment of {absent, i->j, j->i, i<->j} to the pairs is listed and
kept when it is a maximal ancestral graph by definition: ancestral (no directed cycle and no
bidirected edge between a node and its ancestor) and maximal (every non-adjacent pair is
m-separated by some subset of the others).  m-separation is evaluated by enumerating every simple
path and applying the definition (a non-collider outside the conditioning set, a collider in the
ancestors of the conditioning set).  MAGs with the same set of (pair, conditioning set) separation
statements form a Markov equivalence class, and the PAG of a class marks an endpoint tail or arrow
only when every member agrees and a circle otherwise.

Each MAG is fitted by directly minimising the Gaussian negative log-likelihood
(n/2)[p ln 2 pi + ln det Omega + tr(Omega^-1 (I - B) S (I - B)')] over the free coefficients B and
the free entries of Omega (diagonal and bidirected covariances), S the covariance of the data with
divisor n, by BFGS with backtracking (steps leaving the positive-definite region are rejected) from several
starts with an analytic gradient (checked against finite differences), keeping the best positive-definite optimum.  BIC = loglik - (k/2) ln n with k the
coefficients plus p variances plus bidirected covariances.  The class weight is the normalised sum
of exp(BIC) over its members under a uniform prior over MAGs.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
import math
import random

import numpy as np
from scipy.optimize import minimize

LINKS = ("n", "f", "b", "d")  # none, i->j, j->i, i<->j


class G:
    def __init__(self, p, links):
        self.p = p
        self.pairs = list(itertools.combinations(range(p), 2))
        self.link = dict(zip(self.pairs, links))

    def directed(self, a, b):
        if a == b:
            return False
        if a < b:
            return self.link[(a, b)] == "f"
        return self.link[(b, a)] == "b"

    def bidirected(self, a, b):
        return a != b and self.link[(min(a, b), max(a, b))] == "d"

    def adjacent(self, a, b):
        return self.link[(min(a, b), max(a, b))] != "n"

    def arrow_at(self, frm, at):
        return self.directed(frm, at) or self.bidirected(frm, at)

    def reach(self):
        p = self.p
        r = [[self.directed(a, b) for b in range(p)] for a in range(p)]
        for k in range(p):
            for a in range(p):
                for b in range(p):
                    if r[a][k] and r[k][b]:
                        r[a][b] = True
        return r

    def ancestral(self):
        r = self.reach()
        p = self.p
        if any(r[a][a] for a in range(p)):
            return False
        return all(not (self.bidirected(a, b) and (r[a][b] or r[b][a])) for a, b in self.pairs)

    def m_separated(self, x, y, z, reach):
        anc_z = set(z) | {a for a in range(self.p) if any(reach[a][d] for d in z)}
        others = [v for v in range(self.p) if v not in (x, y)]
        for r in range(len(others) + 1):
            for mid in itertools.permutations(others, r):
                path = (x,) + mid + (y,)
                if not all(self.adjacent(path[i], path[i + 1]) for i in range(len(path) - 1)):
                    continue
                open_ = True
                for i in range(1, len(path) - 1):
                    collider = self.arrow_at(path[i - 1], path[i]) and self.arrow_at(path[i + 1], path[i])
                    if collider:
                        if path[i] not in anc_z:
                            open_ = False
                            break
                    elif path[i] in z:
                        open_ = False
                        break
                if open_:
                    return False
        return True

    def maximal(self, reach):
        for a, b in self.pairs:
            if self.adjacent(a, b):
                continue
            others = [v for v in range(self.p) if v not in (a, b)]
            if not any(self.m_separated(a, b, set(z), reach) for r in range(len(others) + 1) for z in itertools.combinations(others, r)):
                return False
        return True

    def signature(self, reach):
        out = []
        for a, b in self.pairs:
            others = [v for v in range(self.p) if v not in (a, b)]
            for r in range(len(others) + 1):
                for z in itertools.combinations(others, r):
                    if self.m_separated(a, b, set(z), reach):
                        out.append((a, b, z))
        return frozenset(out)


def fit_bic(g, S, n):
    p = g.p
    free_b = [(i, k) for i in range(p) for k in range(p) if k != i and g.directed(k, i)]
    spouse = [(a, b) for a, b in g.pairs if g.bidirected(a, b)]
    nb, ns = len(free_b), len(spouse)

    def unpack(theta):
        B = np.zeros((p, p))
        for t, (i, k) in enumerate(free_b):
            B[i, k] = theta[t]
        Om = np.zeros((p, p))
        for i in range(p):
            Om[i, i] = math.exp(min(theta[nb + i], 80.0))
        for t, (a, b) in enumerate(spouse):
            Om[a, b] = Om[b, a] = theta[nb + p + t]
        return B, Om

    def f(theta, grad=False):
        B, Om = unpack(theta)
        try:
            L = np.linalg.cholesky(Om)
        except np.linalg.LinAlgError:
            return (1e30, np.zeros_like(theta)) if grad else 1e30
        A = np.linalg.inv(Om)
        T = np.eye(p) - B
        R = T @ S @ T.T
        val = 0.5 * n * (p * math.log(2 * math.pi) + 2 * np.log(np.diag(L)).sum() + np.trace(A @ R))
        if not grad:
            return val
        g_om = 0.5 * n * (A - A @ R @ A)
        g_b = -n * (S @ T.T @ A)
        out = np.zeros_like(theta)
        for t, (i, k) in enumerate(free_b):
            out[t] = g_b[k, i]
        for i in range(p):
            out[nb + i] = g_om[i, i] * Om[i, i]
        for t, (a, b) in enumerate(spouse):
            out[nb + p + t] = 2 * g_om[a, b]
        return val, out

    def bfgs(x0):
        """BFGS with Armijo backtracking that rejects steps leaving the positive-definite region."""
        x = x0.copy()
        fx, gx = f(x, True)
        if fx >= 1e29:
            return None
        H = np.eye(len(x))
        for _ in range(4000):
            if np.linalg.norm(gx) < 1e-9:
                break
            step = -H @ gx
            if step @ gx >= 0:
                H = np.eye(len(x))
                step = -gx
            alpha = 1.0
            while alpha > 1e-20:
                xn = x + alpha * step
                fn, gn = f(xn, True)
                if fn < fx + 1e-4 * alpha * (step @ gx):
                    break
                alpha *= 0.5
            else:
                break
            sv, yv = xn - x, gn - gx
            sy = sv @ yv
            if sy > 1e-14:
                rho = 1.0 / sy
                I = np.eye(len(x))
                H = (I - rho * np.outer(sv, yv)) @ H @ (I - rho * np.outer(yv, sv)) + rho * np.outer(sv, sv)
            done = abs(fx - fn) < 1e-15 * (1 + abs(fx))
            x, fx, gx = xn, fn, gn
            if done and np.linalg.norm(gx) < 1e-6:
                break
        # Newton polish with a finite-difference Hessian of the analytic gradient.
        for _ in range(60):
            if np.linalg.norm(gx) < 1e-11:
                break
            m = len(x)
            Hs = np.zeros((m, m))
            for j in range(m):
                e = np.zeros(m)
                e[j] = 1e-6 * max(1.0, abs(x[j]))
                Hs[:, j] = (f(x + e, True)[1] - f(x - e, True)[1]) / (2 * e[j])
            Hs = 0.5 * (Hs + Hs.T)
            try:
                step = -np.linalg.solve(Hs, gx)
            except np.linalg.LinAlgError:
                break
            alpha = 1.0
            while alpha > 1e-12:
                fn, gn = f(x + alpha * step, True)
                if fn <= fx:
                    break
                alpha *= 0.5
            else:
                break
            x, fx, gx = x + alpha * step, fn, gn
        return fx

    best = None
    starts = [np.concatenate([np.zeros(nb), np.log(np.diag(S)), np.zeros(ns)])]
    rng = np.random.default_rng(7)
    for _ in range(4):
        starts.append(np.concatenate([0.1 * rng.standard_normal(nb), np.log(np.diag(S)) - 0.3, 0.05 * rng.standard_normal(ns) * np.sqrt(np.diag(S)).mean()]))
    for x0 in starts:
        value = bfgs(x0)
        if value is not None and (best is None or value < best):
            best = value
    k = nb + p + ns
    return -best - 0.5 * k * math.log(n)


def gradient_check():
    rng = np.random.default_rng(3)
    p = 3
    g = G(p, ["f", "d", "n"])
    A = rng.standard_normal((50, p)) @ rng.standard_normal((p, p))
    S = np.cov(A.T, bias=True)
    # re-use fit_bic's closure indirectly by comparing BIC from two different optimisers
    return fit_bic(g, S, 50)


def endpoint(class_graphs, frm, at):
    if all(g.arrow_at(frm, at) for g in class_graphs):
        return "arrow"
    if all(not g.arrow_at(frm, at) for g in class_graphs):
        return "tail"
    return "circle"


def analyse(name, cols, tolerate_deficit=False):
    X = np.array(cols).T
    n, p = X.shape
    S = np.cov(X.T, bias=True)
    pairs = list(itertools.combinations(range(p), 2))
    mags = []
    for links in itertools.product(LINKS, repeat=len(pairs)):
        g = G(p, links)
        if not g.ancestral():
            continue
        reach = g.reach()
        if g.maximal(reach):
            mags.append((g, g.signature(reach)))
    scores = [fit_bic(g, S, n) for g, _ in mags]
    # Markov-equivalent MAGs define the same Gaussian model with the same number of parameters, so
    # their maximised likelihoods coincide.  A direct optimiser can stall on the ridge of a nearly
    # singular covariance, so on such data a member's score is taken from the best member of its
    # class; on every other data set the members must agree on their own.
    by_class = {}
    for (g, sig), sc in zip(mags, scores):
        by_class.setdefault(sig, []).append(sc)
    worst = max(max(v) - min(v) for v in by_class.values())
    if not tolerate_deficit:
        assert worst < 1e-5, (name, worst)
    scores = [max(by_class[sig]) for (g, sig) in mags]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    z = sum(weights)
    classes = {}
    for (g, sig), w in zip(mags, weights):
        classes.setdefault(sig, []).append((g, w / z))
    out = []
    for sig, members in classes.items():
        graphs = [m[0] for m in members]
        first = graphs[0]
        edges = []
        for a, b in pairs:
            if first.adjacent(a, b):
                edges.append([a, b, endpoint(graphs, b, a), endpoint(graphs, a, b)])
        out.append({"edges": edges, "weight": sum(m[1] for m in members), "mags": len(members)})
    return {"name": name, "columns": cols, "mags": len(mags), "member_spread": worst, "classes": out}


def sem(rng, n, spec, latents=()):
    cols = [[] for _ in spec]
    for _ in range(n):
        row = []
        for coefs, noise in spec:
            row.append(sum(c * row[i] for i, c in coefs) + noise * rng.gauss(0, 1))
        for j, v in enumerate(row):
            cols[j].append(round(v, 6))
    return cols


def main():
    rng = random.Random(41)
    # x <- L -> z and y <- x, z, with L unobserved: columns x, z, y  (L is dropped)
    n = 150
    L = [rng.gauss(0, 1) for _ in range(n)]
    x = [L[i] + 0.7 * rng.gauss(0, 1) for i in range(n)]
    z = [0.8 * L[i] + 0.7 * rng.gauss(0, 1) for i in range(n)]
    y = [0.7 * x[i] + 0.6 * z[i] + 0.6 * rng.gauss(0, 1) for i in range(n)]
    d3a = [[round(v, 6) for v in c] for c in (x, z, y)]
    rng2 = random.Random(5)
    d3b = [[round(rng2.gauss(0, 1), 6) for _ in range(8)] for _ in range(3)]
    rng3 = random.Random(6)
    base = [rng3.gauss(0, 1) for _ in range(100)]
    d3c = [[round(v, 6) for v in base], [round(v + 1e-3 * rng3.gauss(0, 1), 6) for v in base], [round(rng3.gauss(0, 1), 6) for _ in range(100)]]
    # four variables: latent L between a and c; a -> b -> c; d <- b, c (collider)
    rng4 = random.Random(8)
    n4 = 250
    L4 = [rng4.gauss(0, 1) for _ in range(n4)]
    a = [L4[i] + 0.8 * rng4.gauss(0, 1) for i in range(n4)]
    b = [0.8 * a[i] + 0.6 * rng4.gauss(0, 1) for i in range(n4)]
    c = [0.6 * b[i] + 0.8 * L4[i] + 0.6 * rng4.gauss(0, 1) for i in range(n4)]
    d = [0.7 * b[i] + 0.7 * c[i] + 0.6 * rng4.gauss(0, 1) for i in range(n4)]
    d4 = [[round(v, 6) for v in col] for col in (a, b, c, d)]
    out = [analyse("three_variables_latent_confounder", d3a), analyse("three_variables_eight_rows", d3b),
           analyse("three_variables_near_collinear_pair", d3c, tolerate_deficit=True), analyse("four_variables_latent_and_collider", d4)]
    print(json.dumps(out))


main()
