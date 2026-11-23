# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact linear g-formula on a small unfolded discrete-time SCM, with its bootstrap and Bayesian pieces.

Regenerate with:  uv run --script temporal_sequential_reference.py > temporal_sequential_reference.json

The structure (hand-derived, asserted against the product's unfolding by the test) has four
variables observed on a regular grid: Z (exogenous confounder), A (treatment), M (mediator) and
Y (outcome), with template edges
    Z_t -> A_t,  Z_t -> Y_t,  Z_{t-1} -> Y_t,  A_{t-1} -> M_t,  A_{t-2} -> M_t,
    M_t -> Y_t,  M_{t-1} -> Y_t,  A_t -> Y_t.
The outcome is Y at offset 0 and the unfolded window is offsets -3..0.  Time-varying confounding
affected by earlier treatment is a lagged cycle the identifier refuses, so the mediator carries the
treatment history into the outcome.

Definitions used (nothing here reads the product):
  * Needed nodes: the outcome and, recursively, the parents of every needed node that is not hard-set.
  * Each needed node that is not hard-set and has parents gets its own ordinary-least-squares fit,
    exact in rational arithmetic, of its value on an intercept and its parents (parents ordered by
    (variable id, lag)), over the rows r = 0..n-1 whose unfolded window ends at series time
    max_lag + r, max_lag being the deepest lag any needed node reads.
  * A needed node without a fitted mechanism (a root, or a hard-set node) enters through the
    sample mean of its column over the rows (a hard-set node through its level).
  * Level: walk the nodes in topological order; a hard node takes its level, any other node the
    overlay-assigned value of its fitted mean at the already-propagated parent values.  A shift
    adds, a multiplier scales, a bound clamps the shifted mean.
  * Contrast: every intervened node carries the active-minus-control difference delta, every other
    node the slope-weighted sum of its parents' differences.
  * Replicates of the block bootstrap refit every mechanism on a frozen row map; the published
    standard error is the sample standard deviation of the replicates times the circular fixed-b
    factor (1.96 + 2.4389 b + 3.7072 b^2 - 2.1055 b^3)/1.96, b = block/rows, times a kernel-bias
    factor.  The block length is max(rule, ceil(b_PW rows^(1/6))) capped at rows/3, the Politis-White
    flat-top bandwidth b_PW being the largest over the contrast influence and every mechanism's
    normal-equation scores.
  * The Bayesian composition fits each stationary mechanism by conjugate Normal-Inverse-Gamma
    regression with rows weighted by 1/kappa (kappa read from the product's own tempering note and
    frozen in the inputs file) and composes the target from independent draws; its moments follow
    from the Student-t marginals.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import os
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
Z, A, M, Y = 0, 1, 2, 3
EDGES = [(Z, 0, A, 0), (Z, 0, Y, 0), (Z, 1, Y, 0), (A, 1, M, 0), (A, 2, M, 0),
         (M, 0, Y, 0), (M, 1, Y, 0), (A, 0, Y, 0)]
LOW = -3


def series(n, seed, persistence=0.0):
    """Linear SCM data on a 1/64 grid; `persistence` makes the exogenous and outcome noise AR(1)."""
    state = seed

    def u():
        nonlocal state
        state = (state * 1103515245 + 12345) % 2**31
        return (state % 2001 - 1000) / 1000.0

    zs, as_, ms, ys = [], [], [], []

    def back(v, k):
        return v[-k] if len(v) >= k else 0.0

    ez = ey = 0.0
    for _ in range(n):
        ez = persistence * ez + u()
        ey = persistence * ey + 0.4 * u()
        z = ez
        a = 0.5 * z + 0.7 * u()
        m = 0.6 * back(as_, 1) + 0.3 * back(as_, 2) + 0.4 * u()
        y = 1.0 + 1.2 * a + 0.8 * m + 0.5 * back(ms, 1) + 0.7 * z + 0.4 * back(zs, 1) + ey
        for lst, x in ((zs, z), (as_, a), (ms, m), (ys, y)):
            lst.append(round(x * 64) / 64)
    return {"z": zs, "a": as_, "m": ms, "y": ys}


# ---------------------------------------------------------------- exact linear algebra
def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [v / piv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def gram(x):
    p = len(x[0])
    return [[sum(r[i] * r[j] for r in x) for j in range(p)] for i in range(p)]


def ols(x, y, w=None):
    w = w or [F(1)] * len(y)
    p = len(x[0])
    xtx = [[sum(wr * r[i] * r[j] for wr, r in zip(w, x)) for j in range(p)] for i in range(p)]
    xty = [sum(wr * r[i] * v for wr, r, v in zip(w, x, y)) for i in range(p)]
    return solve(xtx, xty), xtx


# ---------------------------------------------------------------- structure
def unfolded_parents(node):
    var, off = node
    out = []
    for (u, lu, v, lv) in EDGES:
        if v == var:
            shift = off + lv
            p = (u, shift - lu)
            if p[1] >= LOW:
                out.append(p)
    return sorted(out, key=lambda p: (p[0], off - p[1]))


def all_nodes():
    return [(v, o) for o in range(LOW, 1) for v in (Z, A, M, Y)]


def needed_nodes(outcome, hard):
    seen, stack = set(), [outcome]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n not in hard:
            stack.extend(unfolded_parents(n))
    return sorted(seen, key=lambda n: (n[1], n[0]))


class Setup:
    """The needed unfolded nodes, the intact lag windows and the fitted mechanisms of one query."""

    def __init__(self, data, outcome, overlays, missing=None, weights=None, modifiers=()):
        self.series = [data["z"], data["a"], data["m"], data["y"]]
        self.missing = missing or {}
        self.outcome = outcome
        self.overlay = {o["node"]: o for o in overlays}
        self.hard = {n for n, o in self.overlay.items() if o.get("level") is not None}
        self.needed = needed_nodes(outcome, self.hard)
        self.anchor = max(0, max(o for _, o in self.needed))
        self.max_lag = max(self.anchor - o for _, o in self.needed)
        self.fitted = [n for n in self.needed if n not in self.hard and unfolded_parents(n)]
        self.free = [n for n in self.needed if n in self.hard or not unfolded_parents(n)]
        self.modifiers = list(modifiers)
        total = len(self.series[0]) - self.max_lag
        columns = {(v, self.anchor - o) for v, o in self.needed}
        if self.modifiers:
            for node in self.fitted:
                if self.has_intervened_parent(node):
                    columns |= {(m, self.anchor - node[1]) for m in self.modifiers}
        self.win = [r for r in range(total)
                    if all(self.present(v, self.max_lag + r - lag) for v, lag in columns)]
        self.n = len(self.win)
        self.window_count = total
        self.weights = None
        if weights is not None:
            self.weights = [F(weights[self.max_lag + r]) for r in self.win]

    def present(self, var, t):
        return t not in self.missing.get(var, ())

    def has_intervened_parent(self, node):
        return node not in self.hard and any(p in self.overlay for p in unfolded_parents(node))

    def colv(self, var, lag, rows=None):
        rs = range(self.n) if rows is None else rows
        return [F(self.series[var][self.max_lag + self.win[r] - lag]) for r in rs]

    def col(self, node, rows=None):
        return self.colv(node[0], self.anchor - node[1], rows)

    def weights_on(self, rows=None):
        if self.weights is None:
            return None
        return list(self.weights) if rows is None else [self.weights[r] for r in rows]

    def design(self, node, rows=None):
        """Columns: intercept, parents, then (modifiers) main effects and parent-by-modifier products."""
        ps = unfolded_parents(node)
        cols = [self.col(p, rows) for p in ps]
        n = len(self.col(node, rows))
        x = [[F(1)] + [c[r] for c in cols] for r in range(n)]
        self.slots = getattr(self, "slots", {})
        if self.modifiers and self.has_intervened_parent(node):
            lag = self.anchor - node[1]
            parent_vars = {p[0] for p in ps}
            wcols = [self.colv(m, lag, rows) for m in self.modifiers]
            for m, w in zip(self.modifiers, wcols):
                if m not in parent_vars:
                    for r in range(n):
                        x[r].append(w[r])
            slots = {}
            for mi, w in enumerate(wcols):
                for pi, p in enumerate(ps):
                    if p in self.overlay:
                        slots[(pi, mi)] = len(x[0])
                        for r in range(n):
                            x[r].append(cols[pi][r] * w[r])
            self.slots[node] = slots
        return x, self.col(node, rows)

    def fit(self, rows=None):
        coef = {}
        w = self.weights_on(rows)
        for node in self.fitted:
            x, y = self.design(node, rows)
            coef[node] = ols(x, y, w)[0]
        return coef

    def modifier_mean(self, node, mi, rows=None):
        c = self.colv(self.modifiers[mi], self.anchor - node[1], rows if rows else None)
        return sum(c) / len(c)

    def slope(self, node, coef, k, rows=None):
        out = coef[node][k + 1]
        for mi in range(len(self.modifiers)):
            slot = self.slots.get(node, {}).get((k, mi))
            if slot is not None:
                out += coef[node][slot] * self.modifier_mean(node, mi, rows)
        return out

    def factual(self, rows=None):
        out = {}
        for node in self.free:
            c = self.col(node, rows)
            out[node] = sum(c) / len(c)
        return out


def assign(overlay, mean):
    if overlay.get("level") is not None:
        return F(overlay["level"])
    v = F(overlay.get("mult", 1)) * mean + F(overlay.get("shift", 0))
    b = overlay.get("bounds")
    if b is not None:
        v = min(max(v, F(b[0])), F(b[1]))
    return v


def level_value(setup, coef, factual):
    vals = {}
    for node in setup.needed:
        ov = setup.overlay.get(node)
        if node in setup.hard:
            vals[node] = F(ov["level"])
            continue
        if node in coef:
            c = coef[node]
            ps = unfolded_parents(node)
            mean = c[0] + sum(ci * vals[p] for ci, p in zip(c[1:], ps))
        else:
            mean = factual[node]
        vals[node] = assign(ov, mean) if ov else mean
    return vals[setup.outcome]


def contrast_value(setup, coef, delta, rows=None):
    diff = {}
    for node in setup.needed:
        if node in setup.overlay:
            diff[node] = F(delta)
        elif node in coef:
            ps = unfolded_parents(node)
            diff[node] = sum(setup.slope(node, coef, k, rows) * diff[p] for k, p in enumerate(ps))
        else:
            diff[node] = F(0)
    return diff[setup.outcome]


def value(setup, kind, delta, rows=None, coef=None):
    coef = setup.fit(rows) if coef is None else coef
    if kind == "contrast":
        return contrast_value(setup, coef, delta, rows)
    return level_value(setup, coef, setup.factual(rows))


# ---------------------------------------------------------------- influence and scores
def exact_gradient(setup, kind, delta, coef, factual):
    """Derivative of the propagated value in every coefficient and every free mean (multilinear)."""
    base = level_value(setup, coef, factual) if kind == "level" else contrast_value(setup, coef, delta)
    grads = {}
    for node, c in coef.items():
        g = []
        for k in range(len(c)):
            bumped = {n: list(v) for n, v in coef.items()}
            bumped[node][k] += 1
            up = level_value(setup, bumped, factual) if kind == "level" else contrast_value(setup, bumped, delta)
            g.append(up - base)
        grads[node] = g
    root = {}
    if kind == "level":
        for node in setup.free:
            if node in setup.hard:
                continue
            bumped = dict(factual)
            bumped[node] += 1
            slope = level_value(setup, coef, bumped) - base
            if slope != 0:
                root[node] = slope
    return grads, root


def influence_and_scores(setup, kind, delta):
    n = setup.n
    coef = setup.fit()
    factual = setup.factual()
    grads, root = exact_gradient(setup, kind, delta, coef, factual)
    score = [F(0)] * n
    for node, slope in root.items():
        c = setup.col(node)
        for r in range(n):
            score[r] += slope * (c[r] - factual[node])
    root_scores = list(score) if root else None
    normal = []
    for node in setup.fitted:
        x, y = setup.design(node)
        c = coef[node]
        resid = [yv - sum(a * b for a, b in zip(row, c)) for row, yv in zip(x, y)]
        v = solve(gram(x), grads[node])
        for r in range(n):
            xv = sum(a * b for a, b in zip(x[r], v))
            score[r] += n * xv * resid[r]
        for j in range(len(c)):
            normal.append([x[r][j] * resid[r] for r in range(n)])
    return score, normal, root_scores, coef, factual


# ---------------------------------------------------------------- block length (Politis-White)
def flt(x):
    return [float(v) for v in x]


def autocov(s, mean, k):
    n = len(s)
    return sum((s[t] - mean) * (s[t + k] - mean) for t in range(n - k)) / n


def politis_white(scores):
    s = flt(scores)
    n = len(s)
    if n < 8:
        return None, None
    mean = sum(s) / n
    gamma0 = autocov(s, mean, 0)
    if gamma0 <= 1e-20:
        return None, None
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    rho = []

    def through(lag):
        while len(rho) <= min(lag, m_max):
            rho.append(autocov(s, mean, len(rho)) / gamma0)

    m_hat = max(m_max - k_n, 0)
    for m in range(0, max(m_max - k_n, 0) + 1):
        through(m + k_n)
        if all(abs(rho[m + j]) < thr if m + j < len(rho) else True for j in range(1, k_n + 1)):
            m_hat = m
            break
    big_m = max(min(2 * m_hat, m_max), 1)
    through(big_m)

    def flat(t):
        t = abs(t)
        if t <= 0.5:
            return 1.0
        return 2.0 * (1.0 - t) if t <= 1.0 else 0.0

    g0, big_g = gamma0, 0.0
    for k in range(1, min(big_m, len(rho) - 1) + 1):
        w = flat(k / big_m) * rho[k] * gamma0
        g0 += 2.0 * w
        big_g += 2.0 * k * w
    if g0 <= 0.0:
        return None, None
    d_cb = 4.0 / 3.0 * g0 * g0
    b = (2.0 * big_g * big_g / d_cb) ** (1 / 3) * n ** (1 / 3)
    cap = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    clamped = min(max(b, 1.0), max(cap, 1))
    return math.ceil(clamped), b


def cube_root_ceil(n):
    r = 0
    while r ** 3 < n:
        r += 1
    return r


def block_length(span, n, score_sets):
    rule = min(max(span, cube_root_ceil(n)), n)
    rule = max(rule, 1)
    pws = [politis_white(s)[0] for s in score_sets]
    pws = [p for p in pws if p is not None]
    pw = max(pws) if pws else 0
    testing = math.ceil(pw * n ** (1 / 6)) if pw else 0
    return min(max(rule, min(testing, n // 3)), max(n, 1)), rule, pw, testing


def fixed_b(block, rows):
    b = min(max(block / rows, 0.0), 1.0)
    return (1.96 + 2.4389 * b + 3.7072 * b * b - 2.1055 * b ** 3) / 1.96


def sample_sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# ---------------------------------------------------------------- serial-dependence mean standard error
def kendall_rho(s):
    num = sum(s[t] * s[t - 1] for t in range(1, len(s)))
    den = sum(v * v for v in s[:-1])
    rh = num / den if den > 0 else 0.0
    return min(max(rh + (1 + 3 * rh) / len(s), -0.97), 0.97)


def bartlett_lrv(u, bandwidth):
    m = len(u)
    lags = min(bandwidth, max(m - 1, 0))
    ac = lambda k: sum(u[t] * u[t - k] for t in range(k, m))
    lrv = ac(0)
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * ac(k)
    return max(lrv / m, 0.0)


def lrv_ratio(s, bw):
    g0 = sum(v * v for v in s) / len(s)
    if g0 <= 0:
        return 1.0
    rho = kendall_rho(s)
    u = [s[t] - rho * s[t - 1] for t in range(1, len(s))]
    return bartlett_lrv(u, bw) / (1.0 - rho) ** 2 / g0


def bic_ar(s, limit=4):
    n = len(s)
    mo = min(limit, max(n - 2, 0))
    gamma = [sum(s[t] * s[t - k] for t in range(k, n)) / n for k in range(mo + 1)]
    if gamma[0] <= 0:
        return []
    phi, var = [], gamma[0]
    best_bic, best = n * math.log(var), []
    for order in range(1, mo + 1):
        acc = gamma[order] - sum(f * gamma[order - 1 - j] for j, f in enumerate(phi))
        refl = acc / var
        if 1.0 - refl * refl <= 64 * 2.220446049250313e-16:
            break
        nxt = [phi[j] - refl * phi[order - 2 - j] for j in range(order - 1)] + [refl]
        phi = nxt
        var *= 1.0 - refl * refl
        if not (var > 64 * 2.220446049250313e-16 * gamma[0]):
            break
        bic = n * math.log(var) + order * math.log(n)
        if bic < best_bic:
            best_bic, best = bic, list(phi)
    return best


def ar_prewhitened_ratio(s, phi, bw):
    n, q = len(s), len(phi)
    g0 = sum(v * v for v in s) / n
    if g0 <= 0 or n <= q + 1:
        return 1.0
    u = [s[t] - sum(f * s[t - 1 - j] for j, f in enumerate(phi)) for t in range(q, n)]
    rec = max(1.0 - sum(phi), 1.0 - 0.97)
    return bartlett_lrv(u, bw) / rec ** 2 / g0


def mean_standard_error(series_):
    s0 = flt(series_)
    n = len(s0)
    if n < 3:
        return 0.0
    mean = sum(s0) / n
    s = [v - mean for v in s0]
    g0 = sum(v * v for v in s) / n
    if g0 <= 0:
        return 0.0
    bw = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))
    ratio = lrv_ratio(s, bw)
    phi = bic_ar(s)
    if len(phi) >= 2:
        ratio = max(ratio, ar_prewhitened_ratio(s, phi, bw))
    return math.sqrt(g0 * max(ratio, 1.0) / n)


# ---------------------------------------------------------------- kernel-bias factor
TAIL = 1e-12


def bic_ar_model(s, limit):
    """Levinson-Durbin AR fit with BIC order choice: (phi, autocorrelations up to the order)."""
    n = len(s)
    mo = min(limit, max(n - 2, 0))
    gamma = [sum(s[t] * s[t - k] for t in range(k, n)) / n for k in range(mo + 1)]
    phi, var = [], gamma[0]
    best_bic, best = n * math.log(var), []
    for order in range(1, mo + 1):
        acc = gamma[order] - sum(f * gamma[order - 1 - j] for j, f in enumerate(phi))
        refl = acc / var
        if 1.0 - refl * refl <= 64 * 2.220446049250313e-16:
            break
        phi = [phi[j] - refl * phi[order - 2 - j] for j in range(order - 1)] + [refl]
        var *= 1.0 - refl * refl
        if not (var > 64 * 2.220446049250313e-16 * gamma[0]):
            break
        bic = n * math.log(var) + order * math.log(n)
        if bic < best_bic:
            best_bic, best = bic, list(phi)
    return best, [g / gamma[0] for g in gamma[:len(best) + 1]]


def ar1_acf(rho, block):
    decay = abs(rho)
    tail = math.ceil(math.log(TAIL) / math.log(decay)) if 0 < decay < 1 else 0
    limit = max(block, tail) + 2
    acf, value = [1.0], 1.0
    for k in range(1, limit):
        value *= rho
        if k > block and abs(value) < TAIL:
            break
        acf.append(value)
    return acf


def ar_acf(phi, autocorr, rows, block):
    q = len(phi)
    rho = list(autocorr)
    resolvable = math.ceil(-math.log(TAIL) * max(rows, 1))
    limit = max(block + resolvable, q + 2)
    for k in range(q + 1, limit):
        nxt = sum(f * rho[k - 1 - j] for j, f in enumerate(phi))
        if not math.isfinite(nxt):
            break
        rho.append(nxt)
        if k > block and all(abs(r) < TAIL for r in rho[k + 1 - q:k + 1]):
            break
    return rho


def bartlett_capture(acf, block):
    full = 1.0 + 2.0 * sum(acf[1:])
    kept = 1.0 + 2.0 * sum((1.0 - k / block) * acf[k] for k in range(1, min(len(acf), block)))
    if not math.isfinite(full) or not math.isfinite(kept) or full <= 1.0 or kept <= 0.0:
        return 1.0
    return min(kept / full, 1.0)


def kernel_bias(influence, block):
    s0 = flt(influence)
    n = len(s0)
    if n < 8 or block == 0:
        return 1.0
    mean = sum(s0) / n
    s = [v - mean for v in s0]
    g0 = sum(v * v for v in s) / n
    if g0 <= 0:
        return 1.0
    fraction = 1.0
    rho1 = kendall_rho(s)
    if rho1 > 0:
        fraction = min(fraction, bartlett_capture(ar1_acf(rho1, block), block))
    schwert = max(math.floor(12.0 * (n / 100.0) ** 0.25), 4)
    phi, autocorr = bic_ar_model(s, schwert)
    if len(phi) >= 2:
        fraction = min(fraction, bartlett_capture(ar_acf(phi, autocorr, n, block), block))
    return 1.0 / math.sqrt(fraction) if 0 < fraction < 1 else 1.0


# ---------------------------------------------------------------- Bayesian composition
A0 = B0 = mp.mpf("0.001")


def mechanism_rows(data, var, parents, missing=None):
    """Unique observed rows of one stationary mechanism: parents (var, lag) and the child."""
    ser = [data["z"], data["a"], data["m"], data["y"]]
    gone = missing or {}
    max_lag = max([l for _, l in parents], default=0)
    n = len(ser[0])
    times = [t for t in range(max_lag, n)
             if t not in gone.get(var, ()) and all(t - l not in gone.get(v, ()) for v, l in parents)]
    x = [[mp.mpf(1)] + [mp.mpf(ser[v][t - l]) for v, l in parents] for t in times]
    y = [mp.mpf(ser[var][t]) for t in times]
    return x, y


def nig(x, y, kappa, c0):
    w = mp.mpf(1) / mp.mpf(kappa)
    n = len(y)
    p = len(x[0])
    X = mp.matrix(x)
    Yv = mp.matrix(y)
    prec = w * (X.T * X) + mp.eye(p) / mp.mpf(c0) ** 2
    vn = prec ** -1
    mn = vn * (w * (X.T * Yv))
    res = Yv - X * mn
    rss = w * (res.T * res)[0]
    prior_q = (mn.T * mn)[0] / mp.mpf(c0) ** 2
    an = A0 + w * n / 2
    bn = B0 + (rss + prior_q) / 2
    return dict(mean=[mn[i] for i in range(p)], v=vn, an=an, bn=bn,
                e_s=bn / (an - 1), e_s2=bn ** 2 / ((an - 1) * (an - 2)))


def cov_of(post):
    p = len(post["mean"])
    return mp.matrix([[post["e_s"] * post["v"][i, j] for j in range(p)] for i in range(p)])


def second_moment(post):
    mu = mp.matrix(post["mean"])
    return mu * mu.T + cov_of(post)


def bayes_contrast_moments(posts, delta):
    """Composed contrast = delta*(bYA + bYM0 (b1+b2) + bYM1 b1) over independent Y and M mechanisms."""
    py, pm = posts["Y"], posts["M"]
    # Y coefficients ordered [1, Z0, Z1, A0, M0, M1]; M coefficients [1, A1, A2]
    muy = mp.matrix([py["mean"][3], py["mean"][4], py["mean"][5]])
    sy = cov_of(py)
    sub = lambda m_, idx: mp.matrix([[m_[i, j] for j in idx] for i in idx])
    cy = sub(sy, [3, 4, 5])
    mum = pm["mean"]
    cm = cov_of(pm)
    # g(m) = (1, b1 + b2, b1) = g0 + G m with m = (1, b1, b2) -> rows: [1,0,0] const handled below
    G = mp.matrix([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    g0 = mp.matrix([1, 0, 0])
    mum_v = mp.matrix(mum)
    mu_g = g0 + G * mum_v
    cov_g = G * cm * G.T
    egg = mu_g * mu_g.T + cov_g
    mean = (muy.T * mu_g)[0] * delta
    e2 = ((muy.T * egg * muy)[0] + sum(cy[i, j] * egg[i, j] for i in range(3) for j in range(3))) * delta ** 2
    return mean, mp.sqrt(e2 - mean ** 2)


def bayes_level_moments(posts, zbar, level_a):
    """Level under hard A = level_a at offsets -2,-1,0 (A_{-3} natural).  Independent Y, M, A mechanisms.

    Y_0 = bY + bZ0 z0 + bZ1 z1 + bA a + bM0 M_0 + bM1 M_{-1}
    M_0 = c0 + c1 a + c2 a ;  M_{-1} = c0 + c1 a + c2 A_{-3} ;  A_{-3} = d0 + d1 z_{-3}."""
    py, pm, pa = posts["Y"], posts["M"], posts["A"]
    z0, z1, z3 = zbar
    a = level_a
    mu_y = mp.matrix(py["mean"])
    cy = cov_of(py)
    e_y2 = mu_y * mu_y.T + cy
    # g = (1, z0, z1, a, M0, M1)
    mm = pm["mean"]
    cm = cov_of(pm)
    e_m2 = second_moment(pm)
    ma = pa["mean"]
    e_aval = ma[0] + ma[1] * z3
    ca = cov_of(pa)
    e_aval2 = e_aval ** 2 + ca[0, 0] + 2 * ca[0, 1] * z3 + ca[1, 1] * z3 ** 2
    # M0 = c0 + (c1 + c2) a ; M1 = c0 + c1 a + c2 aval
    m0_mean = mm[0] + (mm[1] + mm[2]) * a
    m1_mean = mm[0] + mm[1] * a + mm[2] * e_aval
    # E[M0^2], E[M0 M1], E[M1^2] via second moments of (c0, c1, c2)
    def quad(u, v):
        return sum(u[i] * v[j] * e_m2[i, j] for i in range(3) for j in range(3))
    u0 = [1, a, a]  # M0 coefficients on (c0, c1, c2)
    u1 = [1, a, 0]  # M1 constant part on (c0, c1); c2 multiplies aval
    e_m0m0 = quad(u0, u0)
    e_m0m1 = quad(u0, u1) + (e_m2[0, 2] + a * e_m2[1, 2] + a * e_m2[2, 2]) * e_aval
    e_m1m1 = quad(u1, u1) + 2 * (e_m2[0, 2] + a * e_m2[1, 2]) * e_aval + e_m2[2, 2] * e_aval2
    consts = [1, z0, z1, a]
    eg = mp.matrix(consts + [m0_mean, m1_mean])
    egg = mp.matrix(6, 6)
    for i in range(4):
        for j in range(4):
            egg[i, j] = consts[i] * consts[j]
        egg[i, 4] = consts[i] * m0_mean
        egg[4, i] = egg[i, 4]
        egg[i, 5] = consts[i] * m1_mean
        egg[5, i] = egg[i, 5]
    egg[4, 4], egg[4, 5], egg[5, 4], egg[5, 5] = e_m0m0, e_m0m1, e_m0m1, e_m1m1
    mean = (mu_y.T * eg)[0]
    e2 = (mu_y.T * egg * mu_y)[0] + sum(cy[i, j] * egg[i, j] for i in range(6) for j in range(6))
    return mean, e2 - mean ** 2


def main():
    inputs_path = os.path.join(HERE, "temporal_sequential_reference_inputs.json")
    inputs = json.load(open(inputs_path)) if os.path.exists(inputs_path) else {}
    data = series(24, 7)
    short = series(10, 11)
    persistent = series(64, 23, 0.85)
    out = {"data": data, "short": short, "persistent": persistent}

    out["structure"] = {f"{v}@{o}": [f"{pv}@{po}" for pv, po in unfolded_parents((v, o))]
                        for (v, o) in all_nodes()}

    def hard(node, level):
        return {"node": node, "level": level}

    window = [(A, -2), (A, -1), (A, 0)]
    hard_all = [hard(n, 0.5) for n in window]
    cases = {
        "contrast_unit": ("contrast", 1.0, []),
        "contrast_dose": ("contrast", 3.0, []),
        "level_hard": ("level", None, hard_all),
        "level_mixture": ("level", None, [hard(window[0], 0.5),
                                          {"node": window[1], "shift": 0.75},
                                          hard(window[2], 0.5)]),
        "level_multiplicative": ("level", None, [hard(window[0], 0.5),
                                                 {"node": window[1], "mult": 1.5, "shift": 0.25},
                                                 hard(window[2], 0.5)]),
        "level_bounded_upper": ("level", None, [hard(window[0], 0.5),
                                                {"node": window[1], "shift": 2.0, "bounds": [-0.3, 0.3]},
                                                hard(window[2], 0.5)]),
        "level_bounded_lower": ("level", None, [hard(window[0], 0.5),
                                                {"node": window[1], "shift": -2.0, "bounds": [-0.3, 0.3]},
                                                hard(window[2], 0.5)]),
        "level_bounded_slack": ("level", None, [hard(window[0], 0.5),
                                                {"node": window[1], "shift": 0.1, "bounds": [-5.0, 5.0]},
                                                hard(window[2], 0.5)]),
        "level_partial": ("level", None, [hard(window[1], 0.5)]),
        "level_joint_mediator": ("level", None, [hard(window[1], 0.5), hard((M, 0), 0.5)]),
        "level_shift_mediator": ("level", None, [hard(window[0], 0.5),
                                                 {"node": (M, 0), "mult": 2.0, "shift": -0.1}]),
        "contrast_late_window": ("contrast", 2.0, []),
        "level_two_soft": ("level", None, [{"node": window[0], "mult": 0.5, "shift": -0.2},
                                           {"node": window[1], "shift": 0.3},
                                           hard(window[2], 0.5)]),
    }
    contrast_overlays = [{"node": n, "level": 1.0} for n in window]
    late = [(A, -1), (A, 0)]
    point = []
    for name, (kind, delta, overlays) in cases.items():
        if kind == "contrast":
            overlays = [{"node": n, "level": delta} for n in (late if name == "contrast_late_window" else window)]
        s = Setup(data, (Y, 0), overlays)
        coef = s.fit()
        val = value(s, kind, delta, coef=coef)
        point.append(dict(
            name=name, kind=kind, delta=delta, overlays=overlays, value=float(val),
            window=[-1, 0] if name == "contrast_late_window" else [-2, 0],
            max_lag=s.max_lag, rows=s.n,
            fitted={f"{v}@{o}": [float(c) for c in cs] for (v, o), cs in coef.items()},
            factual={f"{v}@{o}": float(m) for (v, o), m in s.factual().items()},
        ))
    out["point"] = point

    # bootstrap pieces
    boots = {}
    for tag, dat, name in (("short_contrast", short, "contrast_unit"),
                           ("long_contrast", data, "contrast_unit"),
                           ("long_level", data, "level_hard"),
                           ("persistent_contrast", persistent, "contrast_unit")):
        kind, delta, overlays = cases[name]
        if kind == "contrast":
            overlays = contrast_overlays
        s = Setup(dat, (Y, 0), overlays)
        infl, normal, root_scores, coef, factual = influence_and_scores(s, kind, delta)
        span = s.max_lag + 1
        bl, rule, pw, testing = block_length(span, s.n, [infl] + normal)
        pws = [politis_white(x)[1] for x in [infl] + normal]
        entry = dict(case=name, value=float(value(s, kind, delta, coef=coef)), rows=s.n, span=span,
                     influence=flt(infl), block_length=bl, rule=rule, pw=pw, testing=testing,
                     pw_continuous=[p for p in pws if p is not None],
                     fixed_b=fixed_b(bl, s.n), kernel_bias=kernel_bias(infl, bl))
        row_maps = inputs.get("rows", {}).get(tag)
        if row_maps:
            reps = []
            for rows in row_maps:
                try:
                    reps.append(float(value(s, kind, delta, rows=rows)))
                except StopIteration:  # singular normal equations: the replicate cannot be fit
                    reps.append(None)
            ok = [r for r in reps if r is not None]
            entry["replicates"] = reps
            entry["replicates_ok"] = len(ok)
            entry["raw_sd"] = sample_sd(ok) if len(ok) >= 2 and (len(reps) - len(ok)) / len(reps) <= 0.5 else None
        boots[tag] = entry
    out["bootstrap"] = boots

    # a replicate on a frozen, non-RNG row map (with duplicates and wrap-around)
    s = Setup(data, (Y, 0), contrast_overlays)
    fixed_rows = [3, 4, 5, 6, 19, 20, 0, 1, 2, 11, 12, 13, 14, 15, 7, 8, 9, 9, 10, 17, 18, 18, 2]
    out["fixed_rows"] = dict(
        rows=fixed_rows,
        contrast=float(value(s, "contrast", 1.0, rows=fixed_rows)),
        level=float(value(Setup(data, (Y, 0), hard_all), "level", None, rows=fixed_rows)),
    )

    # effect modifiers, a series with missing time points, and sampling weights (all on the contrast)
    modifiers = Setup(data, (Y, 0), contrast_overlays, modifiers=[Z])
    out["modifier"] = dict(
        modifiers=[Z], value=float(value(modifiers, "contrast", 1.0)),
        unmodified=float(value(Setup(data, (Y, 0), contrast_overlays), "contrast", 1.0)),
        widths={f"{v}@{o}": len(modifiers.design((v, o))[0][0]) for (v, o) in modifiers.fitted},
    )
    missing = {Y: [9], Z: [17], M: [12]}
    gapped = Setup(data, (Y, 0), contrast_overlays, missing={k: set(v) for k, v in missing.items()})
    gap = dict(missing=missing, value=float(value(gapped, "contrast", 1.0)), rows=gapped.n,
               windows=gapped.window_count,
               complete=[r in set(gapped.win) for r in range(gapped.window_count)])
    block = max(min(max(gapped.max_lag + 1, cube_root_ceil(gapped.window_count)), gapped.window_count), 1)
    gap["block_length"] = block
    gap["fixed_b"] = fixed_b(block, gapped.window_count)
    row_of = {w: i for i, w in enumerate(gapped.win)}
    position_maps = inputs.get("rows", {}).get("gapped")
    if position_maps:
        reps = []
        for positions in position_maps:
            rows = [row_of[q] for q in positions if q in row_of]
            try:
                reps.append(float(value(gapped, "contrast", 1.0, rows=rows)) if rows else None)
            except StopIteration:
                reps.append(None)
        ok = [r for r in reps if r is not None]
        gap["replicates"] = reps
        gap["replicates_ok"] = len(ok)
        gap["raw_sd"] = sample_sd(ok) if len(ok) >= 2 and (len(reps) - len(ok)) / len(reps) <= 0.5 else None
    out["gaps"] = gap
    weights = [0.0 if t == 10 else 0.5 + ((7 * t) % 5) / 4 for t in range(len(data["z"]))]
    weighted = Setup(data, (Y, 0), contrast_overlays, weights=weights)
    out["weights"] = dict(weights=weights, value=float(value(weighted, "contrast", 1.0)),
                          unweighted=float(value(Setup(data, (Y, 0), contrast_overlays), "contrast", 1.0)))

    # Bayesian composition
    bayes = {}
    kappas = inputs.get("kappa")
    c0 = 4.0
    if kappas:
        mech_def = {"A": (A, [(Z, 0)]), "M": (M, [(A, 1), (A, 2)]),
                    "Y": (Y, [(Z, 0), (Z, 1), (A, 0), (M, 0), (M, 1)])}
        s = Setup(data, (Y, 0), hard_all)
        fac = s.factual()
        zbar = [mp.mpf(fac[(Z, 0)].numerator) / fac[(Z, 0)].denominator,
                mp.mpf(fac[(Z, -1)].numerator) / fac[(Z, -1)].denominator,
                mp.mpf(fac[(Z, -3)].numerator) / fac[(Z, -3)].denominator]
        infl, normal, root_scores, coef, factual = influence_and_scores(s, "level", None)
        root_se = mean_standard_error(root_scores)
        for kind in ("contrast", "level"):
            posts = {}
            for key, (var, ps) in mech_def.items():
                x, y = mechanism_rows(data, var, ps)
                posts[key] = nig(x, y, kappas[kind][key], c0)
            posterior = {k: dict(mean=[float(m) for m in p["mean"]], an=float(p["an"]), bn=float(p["bn"]),
                                 rows=len(mechanism_rows(data, *mech_def[k])[1]))
                         for k, p in posts.items()}
            if kind == "contrast":
                mean_c, sd_c = bayes_contrast_moments(posts, 1.0)
                bayes[kind] = dict(kappa=kappas[kind], mean=float(mean_c), sd=float(sd_c), posterior=posterior)
            else:
                mean_l, var_l = bayes_level_moments(posts, zbar, mp.mpf("0.5"))
                bayes[kind] = dict(kappa=kappas[kind], mean=float(mean_l),
                                   mechanism_sd=float(mp.sqrt(var_l)), root_mean_se=root_se,
                                   sd=float(mp.sqrt(var_l + mp.mpf(root_se) ** 2)), posterior=posterior)
        bayes["prior_scale"] = c0
        if "gapped" in kappas:
            gone = {k: set(v) for k, v in missing.items()}
            posts = {}
            counts = {}
            for key, (var, ps) in mech_def.items():
                x, y = mechanism_rows(data, var, ps, gone)
                counts[key] = len(y)
                posts[key] = nig(x, y, kappas["gapped"][key], c0)
            mean_g, sd_g = bayes_contrast_moments(posts, 1.0)
            bayes["gapped"] = dict(kappa=kappas["gapped"], rows=counts, mean=float(mean_g), sd=float(sd_g))
    out["bayes"] = bayes
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
