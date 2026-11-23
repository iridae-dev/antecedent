# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for group-sequential boundaries, crossing probabilities and stagewise inference.

Regenerate with:  uv run --script group_sequential_reference.py > group_sequential_reference.json

Written from the model, not from the product source, and on a different scale: the product
recurses on the standardized statistic Z_k with the transition N(u sqrt(t_{k-1}/t_k), ...), this
reference recurses on the score B_k = sqrt(t_k) Z_k, a sum of independent increments
B_k - B_{k-1} ~ N(theta (t_k - t_{k-1}), t_k - t_{k-1}), with its own Gauss-Legendre panels whose
width follows the narrowest transition (so closely spaced looks are resolved) and 20-digit
arithmetic.  Spending functions are the Lan-DeMets O'Brien-Fleming-type 2 - 2 Phi(z_{a/2}/sqrt t)
and Pocock-type a ln(1 + (e - 1) t); the boundary at a look solves "first crossing there has the
spent increment" by bisection on the 20-digit continuation integral.  Two-look crossing
probabilities are also checked against a direct one-dimensional integral of the bivariate normal.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 20
mpf = mp.mpf


def Phi(x):
    return mp.ncdf(x)


def phi(x):
    return mp.npdf(x)


def gl_rule(n):
    nodes, weights = [], []
    for i in range(n):
        x = mp.cos(mp.pi * (i + mpf(3) / 4) / (n + mpf(1) / 2))
        for _ in range(60):
            p = mp.legendre(n, x)
            dp = n * (x * p - mp.legendre(n - 1, x)) / (x * x - 1)
            step = p / dp
            x -= step
            if abs(step) < mpf(10) ** (-mp.mp.dps + 2):
                break
        p = mp.legendre(n, x)
        dp = n * (x * p - mp.legendre(n - 1, x)) / (x * x - 1)
        nodes.append(x)
        weights.append(2 / ((1 - x * x) * dp * dp))
    return nodes, weights


RULE = gl_rule(6)


def composite(lo, hi, width):
    panels = max(1, int(mp.ceil((hi - lo) / width)))
    h = (hi - lo) / panels
    nodes, weights = [], []
    for p in range(panels):
        centre = lo + h * (p + mpf(1) / 2)
        for u, w in zip(*RULE):
            nodes.append(centre + h * u / 2)
            weights.append(h * w / 2)
    return nodes, weights


class Design:
    def __init__(self, fractions, two_sided, drift):
        self.t = [mpf(f) for f in fractions]
        self.two = two_sided
        self.drift = mpf(drift)
        dts = [self.t[0]] + [self.t[i] - self.t[i - 1] for i in range(1, len(self.t))]
        self.width = min(mpf(1) / 2, mp.sqrt(min(dts)) * mpf(3) / 2)
        self.stage = None  # (fraction, nodes, weights, density, mass) on the score scale

    def mean_sd(self, k):
        if self.stage is None:
            return None, self.drift * self.t[k], mp.sqrt(self.t[k])
        dt = self.t[k] - self.stage[0]
        return dt, self.drift * dt, mp.sqrt(dt)

    def crossing(self, k, z, upper_only=False):
        """P(still continuing before look k and Z_k beyond z) on the score scale."""
        b = z * mp.sqrt(self.t[k])
        both = self.two and not upper_only
        dt, shift, sd = self.mean_sd(k)
        if self.stage is None:
            lower = Phi((-b - shift) / sd) if both else 0
            return lower + Phi(-(b - shift) / sd)
        _, nodes, weights, density, mass = self.stage
        inside = mpf(0)
        for u, w, g in zip(nodes, weights, density):
            up = Phi((b - u - shift) / sd)
            lo = Phi((-b - u - shift) / sd) if both else 0
            inside += w * g * (up - lo)
        return mass - inside

    def advance(self, k, z):
        b = z * mp.sqrt(self.t[k])
        lo, hi = (-b, b) if self.two else (-10 * mp.sqrt(self.t[k]) - 2 * abs(self.drift) * self.t[k], b)
        nodes, weights = composite(lo, hi, self.width * mp.sqrt(self.t[k]) / mp.sqrt(self.t[k]))
        dt, shift, sd = self.mean_sd(k)
        if self.stage is None:
            density = [phi((x - shift) / sd) / sd for x in nodes]
        else:
            _, pn, pw, pg, _ = self.stage
            density = [sum(w * g * phi((x - u - shift) / sd) / sd for u, w, g in zip(pn, pw, pg)) for x in nodes]
        mass = sum(w * g for w, g in zip(weights, density))
        self.stage = (self.t[k], nodes, weights, density, mass)


def spend(kind, fraction, alpha):
    fraction, alpha = mpf(fraction), mpf(alpha)
    if kind == "obf":
        z = mp.sqrt(2) * mp.erfinv(1 - alpha)
        return 2 * Phi(-z / mp.sqrt(fraction))
    return alpha * mp.log(1 + (mp.e - 1) * fraction)


def bisect(f, lo, hi, target, iters=48):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def boundaries(fractions, alpha, kind, two):
    d = Design(fractions, two, 0)
    crit, prev = [], mpf(0)
    for k in range(len(fractions)):
        cum = spend(kind, fractions[k], alpha)
        inc = cum - prev
        c = bisect(lambda z: d.crossing(k, z), mpf(0) if two else mpf(-8), mpf(12), inc)
        d.advance(k, c)
        crit.append(c)
        prev = cum
    return crit


def crossings(fractions, crit, two, drift):
    d = Design(fractions, two, drift)
    out = []
    for k, c in enumerate(crit):
        out.append(d.crossing(k, mpf(c)))
        d.advance(k, mpf(c))
    return out


def stage_p_all(fractions, crit, two, drift, stats):
    """Upper stagewise p-value at every look k for the statistic stats[k], in one pass."""
    d = Design(fractions, two, drift)
    prefix = mpf(0)
    out = []
    for k in range(len(fractions)):
        out.append(prefix + d.crossing(k, mpf(stats[k]), True))
        if k + 1 < len(fractions):
            prefix += d.crossing(k, mpf(crit[k]), True)
            d.advance(k, mpf(crit[k]))
    return out


def two_look_exact(t1, t2, c1, c2, drift):
    """P(first crossing at look 2) for the two-sided rule, one-dimensional integral of the
    bivariate normal: Z1 ~ N(drift sqrt t1, 1), Z2 | Z1 ~ N(...)."""
    t1, t2, c1, c2, th = mpf(t1), mpf(t2), mpf(c1), mpf(c2), mpf(drift)
    rho = mp.sqrt(t1 / t2)
    sd = mp.sqrt(1 - t1 / t2)
    shift = th * (t2 - t1) / mp.sqrt(t2)

    def inner(z1):
        m = z1 * rho + shift
        return phi(z1 - th * mp.sqrt(t1)) * (Phi((c2 - m) / sd) - Phi((-c2 - m) / sd))

    cuts = sorted({-c1, c1} | {x for x in ((s * c2 - shift) / rho for s in (1, -1)) if -c1 < x < c1})
    inside = mp.quad(inner, cuts)
    first = Phi(-c1 - th * mp.sqrt(t1)) + Phi(-(c1 - th * mp.sqrt(t1)))
    return 1 - first - inside


def ns(x):
    return mp.nstr(x, 18)


def main():
    out = {"boundaries": [], "stagewise": [], "two_look": []}
    designs = [
        [1.0],
        [0.5, 1.0],
        [1 / 3, 2 / 3, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.05, 0.3, 0.31, 0.9, 1.0],
        [0.95, 0.99, 1.0],
        [0.3, 0.7],
    ]
    for fractions in designs:
        for kind in ("obf", "pocock"):
            for two in (False, True):
                if len(fractions) >= 10 and not two:
                    continue
                for alpha in (0.05,) if len(fractions) > 4 else (0.05, 0.001):
                    crit = boundaries(fractions, alpha, kind, two)
                    rec = {"fractions": fractions, "alpha": alpha, "spending": kind, "two_sided": two,
                           "critical": [ns(c) for c in crit], "drifts": []}
                    for drift in (0.0, 1.5):
                        rec["drifts"].append({"drift": drift, "probabilities": [ns(p) for p in crossings(fractions, crit, two, drift)]})
                    out["boundaries"].append(rec)
                    if fractions in ([0.5, 1.0], [1 / 3, 2 / 3, 1.0]) and alpha == 0.05:
                        for offset in (0.0, 0.7):
                            stats = [float(c) + offset if k < len(fractions) - 1 else float(c) - 0.4 + offset for k, c in enumerate(crit)]
                            for drift in (0.0, 1.0):
                                ps = stage_p_all(fractions, crit, two, drift, stats)
                                for look in range(len(fractions)):
                                    out["stagewise"].append({"fractions": fractions, "alpha": alpha, "spending": kind, "two_sided": two,
                                                             "critical": [ns(c) for c in crit], "look": look, "statistic": stats[look], "drift": drift,
                                                             "p": ns(ps[look])})
    for t1, t2, c1, c2, th in ((0.5, 1.0, 2.5, 2.0, 0.0), (0.5, 1.0, 2.5, 2.0, 2.0), (0.01, 1.0, 5.0, 2.0, 1.0), (0.9, 1.0, 2.2, 2.0, 0.5), (0.999, 1.0, 2.0, 2.0, 0.5)):
        for two in (True,):
            crit = [c1, c2]
            out["two_look"].append({"fractions": [t1, t2], "critical": crit, "drift": th,
                                    "exact": ns(two_look_exact(t1, t2, c1, c2, th)),
                                    "recursion": ns(crossings([t1, t2], crit, True, th)[1])})
    print(json.dumps(out, separators=(",", ":")))


main()
