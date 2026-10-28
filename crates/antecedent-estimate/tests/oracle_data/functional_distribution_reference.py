"""Empirical plug-in of identified functionals, evaluated exactly from the contingency counts.

Regenerate with:  python3 functional_distribution_reference.py > functional_distribution_reference.json

Each case is a contingency table (rows with counts).  The reference evaluates the textbook
identities with exact rational arithmetic on the empirical probabilities, with no estimator code:
  backdoor     P(y | do(t)) = sum_z P(y | t, z) P(z)
  front door   P(y | do(t)) = sum_m P(m | t) sum_t' P(y | m, t') P(t')
  napkin       P(y | do(x)) = [sum_w2 P(x, y | w1, w2) P(w2)] / [sum_w2 P(x | w1, w2) P(w2)]
               at each w1 with positive support, averaged by the empirical P(w1) over the supported
               values (the free-variable rule of the module).
  mediation    E[Y_{t1 on the mediator path, t0 on the direct path}] =
               sum_{c,m} P(c) P(m | t1, c) E[Y | t0, m, c]
The contrast of an ATE is the difference of the two interventional means.  A cell the identity
needs but the table lacks is recorded as unsupported (the product must refuse, not zero-fill).
Values are written as the nearest binary64 of the exact rational.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from itertools import product


class Table:
    def __init__(self, names, rows):
        self.names, self.rows = names, rows  # rows: list of (tuple of values, count)
        self.n = sum(c for _, c in rows)

    def idx(self, v):
        return self.names.index(v)

    def count(self, **fix):
        return sum(c for r, c in self.rows
                   if all(r[self.idx(k)] == v for k, v in fix.items()))

    def p(self, **fix):
        return F(self.count(**fix), self.n)

    def cond(self, target, given):
        d = self.count(**given)
        if d == 0:
            raise KeyError("unsupported")
        return F(self.count(**{**given, **target}), d)

    def levels(self, v):
        return sorted({r[self.idx(v)] for r, _ in self.rows})


def backdoor_dist(tb, t, y_levels):
    out = []
    for yv in y_levels:
        s = F(0)
        for z in tb.levels("z"):
            s += tb.cond({"y": yv}, {"t": t, "z": z}) * tb.p(z=z)
        out.append(s)
    return out


def mean(levels, probs):
    return sum(F(l) * p for l, p in zip(levels, probs))


def backdoor_case(name, rows):
    tb = Table(["z", "t", "y"], rows)
    yl = tb.levels("y")
    res = {"name": name, "kind": "backdoor", "rows": rows, "y_levels": yl}
    for t in (0, 1):
        try:
            d = backdoor_dist(tb, t, yl)
            res[f"dist_t{t}"] = [float(v) for v in d]
            res[f"mean_t{t}"] = float(mean(yl, d))
        except KeyError:
            res[f"dist_t{t}"] = None
            res[f"mean_t{t}"] = None
    if res["mean_t0"] is not None and res["mean_t1"] is not None:
        d1, d0 = backdoor_dist(tb, 1, yl), backdoor_dist(tb, 0, yl)
        res["ate"] = float(mean(yl, d1) - mean(yl, d0))
    else:
        res["ate"] = None
    return res


def frontdoor_mean(tb, t, y_levels):
    s = F(0)
    for m in tb.levels("m"):
        pm = tb.cond({"m": m}, {"t": t})
        g = F(0)
        for tp in tb.levels("t"):
            g += mean(y_levels, [tb.cond({"y": yv}, {"m": m, "t": tp}) for yv in y_levels]) * tb.p(t=tp)
        s += pm * g
    return s


def frontdoor_case(name, rows):
    tb = Table(["t", "m", "y"], rows)
    yl = tb.levels("y")
    return {"name": name, "kind": "frontdoor", "rows": rows,
            "ate": float(frontdoor_mean(tb, 1, yl) - frontdoor_mean(tb, 0, yl))}


def napkin_case(name, rows):
    tb = Table(["w2", "w1", "x", "y"], rows)
    out = {"name": name, "kind": "napkin", "rows": rows}
    for x in (0, 1):
        acc = []  # per y level
        yl = tb.levels("y")
        weight = F(0)
        total = [F(0)] * len(yl)
        for w1 in tb.levels("w1"):
            try:
                den = sum(tb.cond({"x": x}, {"w1": w1, "w2": w2}) * tb.p(w2=w2)
                          for w2 in tb.levels("w2"))
                num = [sum(tb.cond({"x": x, "y": yv}, {"w1": w1, "w2": w2}) * tb.p(w2=w2)
                           for w2 in tb.levels("w2")) for yv in yl]
            except KeyError:
                continue
            pw = tb.p(w1=w1)
            weight += pw
            total = [a + pw * b / den for a, b in zip(total, num)]
        dist = [a / weight for a in total]
        out[f"dist_x{x}"] = [float(v) for v in dist]
    out["y_levels"] = tb.levels("y")
    return out


def mediation_case(name, rows):
    tb = Table(["c", "t", "m", "y"], rows)
    yl = tb.levels("y")

    def ey(t_med, t_dir):
        s = F(0)
        for c in tb.levels("c"):
            for m in tb.levels("m"):
                pm = tb.cond({"m": m}, {"t": t_med, "c": c})
                ym = mean(yl, [tb.cond({"y": yv}, {"t": t_dir, "m": m, "c": c}) for yv in yl])
                s += tb.p(c=c) * pm * ym
        return s

    return {"name": name, "kind": "mediation", "rows": rows,
            "nie": float(ey(1, 0) - ey(0, 0))}


def grid(levels, counts):
    return [(k, c) for k, c in zip(product(*levels), counts) if c > 0]


cases = []
# z in {0,1,2}, t in {0,1}, y in {0,1,2}: unbalanced counts, singleton cells.
counts = [9, 3, 1, 2, 7, 4, 1, 6, 2, 5, 1, 3, 8, 2, 2, 4, 6, 1]
cases.append(backdoor_case("backdoor_ternary_unbalanced", grid([(0, 1, 2), (0, 1), (0, 1, 2)], counts)))
sparse = list(counts)
for k in range(3):  # remove every row of the cell z=2, t=0
    sparse[(2 * 2 + 0) * 3 + k] = 0
cases.append(backdoor_case("backdoor_missing_control_cell", grid([(0, 1, 2), (0, 1), (0, 1, 2)], sparse)))
cases.append(frontdoor_case("frontdoor_ternary_mediator",
                            grid([(0, 1), (0, 1, 2), (0, 1)],
                                 [7, 3, 4, 5, 2, 6, 3, 4, 5, 6, 8, 1])))
cases.append(napkin_case("napkin_binary", grid([(0, 1), (0, 1), (0, 1), (0, 1)],
                                               [6, 2, 1, 4, 3, 5, 2, 2, 4, 3, 6, 1, 2, 5, 3, 7])))
cases.append(mediation_case("mediation_edge", grid([(0, 1), (0, 1), (0, 1), (0, 1)],
                                                   [5, 3, 2, 4, 3, 6, 2, 5, 4, 2, 6, 3, 1, 7, 4, 4])))
print(json.dumps({"cases": cases}))
