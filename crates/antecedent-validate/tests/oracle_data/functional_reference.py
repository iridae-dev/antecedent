# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact reference for the identified-functional stability refuters (path subsets, interventional
distribution table checks, window-contrast stretches / rotated treatment / split halves).

Stages (the Rust test regenerates every perturbation from the library RNG streams and asserts it
equals the frozen one):

  python3 functional_reference.py data > data.json
  (Rust: FUNCTIONAL_REFERENCE_DATA=data.json FUNCTIONAL_REFERENCE_OUT=perturbations.json
         cargo test -p antecedent-validate --test functional_reference functional_emit -- --ignored)
  python3 functional_reference.py build data.json perturbations.json > functional_reference.json

Written from the definitions, with no product code. Every functional is evaluated from the
contingency table of the kept rows with fractions.Fraction:
  backdoor effect   ey(t) = sum_c P(c) E[y | t, c];   effect = ey(1) - ey(0)
  backdoor law      P(y | do(t=1)) = sum_c P(c) P(y | t=1, c)   (unconditional query), or
                    P(y | t=1, c) for each c   (query conditioned on c)
  path effect       E[Y with the mediator drawn at t=1 and the direct arm at t=0]
                    - E[Y with both at t=0], each sum_c P(c) sum_m P(m | t_med, c) E[y | 0, m, c]
A cell the identity reads that a kept set lacks makes the refit undefined; the refuter must then
refuse (the case records "error"). Refuters are evaluated as follows:
  mean = average of the replicate values, sd = sample SD (n-1), z = (null - mean)/sd,
  p = erfc(|z|/sqrt 2), passed = p >= 0.05; a spread of at most 1e-12 scale gives p = 1 when the
  null is within 1e-9 scale of the mean and 0 otherwise.
  distribution table distance: for every conditioning group of the union of both tables, 1 when
  the group is missing from either, else half the L1 distance over the union of outcomes; the
  maximum over groups, averaged over the replicates, passes when at most 1/10.
  window stretch    rows[start : start + L], L = clamp(floor(n f), 1, n - 1), start =
                    min(floor(u (n - L + 1)), n - L);
  window rotation   rotations = max(n + 1 - 2 span, 0); offset = min(floor(u rotations),
                    rotations - 1); the treatment column read (span + offset) mod n rows further
                    down; none possible when rotations = 0;
  split half        first n // 2 rows against the rest, z = (a - b)/sqrt(se_a^2 + se_b^2) with the
                    supplied bootstrap standard errors, p = erfc(|z|/sqrt 2); unusable (not
                    informative, not passed) when a half has no standard error or is covered by
                    one resampling block (the supplied block lengths).
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import math
import sys
from fractions import Fraction as F

ALPHA = 0.05
REPS = 20
# (full validation, shift of the supplied original from the exact full-table value)
RUNS = [(False, 0.0), (True, 0.0), (True, 0.15)]


class Undefined(Exception):
    pass


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 40) / (1 << 24)


def backdoor_rows(n, seed, drift=0.0, tcut=0.5, ylevels=3):
    """(t, y, c): c ternary, t binary leaning on c, y on {0,1,2} leaning on t and c."""
    g = lcg(seed)
    rows = []
    for i in range(n):
        u = [next(g) for _ in range(3)]
        c = min(int(u[0] * 3), 2)
        t = 1 if u[1] + 0.12 * (c - 1) > tcut else 0
        eff = 0.30 + drift * (1 if i >= n // 2 else 0)
        s = u[2] + eff * t * (1 if drift == 0 else 1) + 0.15 * c - 0.15
        y = 0 if s < 0.4 else (1 if s < 0.8 else 2)
        rows.append((t, y, c))
    return rows


def path_rows(n, seed):
    """(t, m, y, c): c ternary -> t, m, y; t -> m, y; m -> y; all binary but c."""
    g = lcg(seed)
    rows = []
    for _ in range(n):
        u = [next(g) for _ in range(4)]
        c = min(int(u[0] * 3), 2)
        t = 1 if u[1] + 0.1 * (c - 1) > 0.5 else 0
        m = 1 if u[2] + 0.35 * t + 0.1 * c > 0.75 else 0
        y = 1 if u[3] + 0.3 * m + 0.1 * t + 0.1 * c > 0.85 else 0
        rows.append((t, m, y, c))
    return rows


CASES = [
    # name, kind, rows, extra
    dict(name="window_stationary", kind="window", rows=backdoor_rows(96, 11), span=2),
    dict(name="window_drifting", kind="window", rows=backdoor_rows(120, 23, drift=0.9), span=3),
    dict(name="window_rotation_single", kind="window", rows=backdoor_rows(60, 5), span=30),
    dict(name="window_too_short_to_rotate", kind="window", rows=backdoor_rows(48, 7), span=25),
    dict(name="window_sparse", kind="window", rows=backdoor_rows(12, 3), span=2),
    dict(name="distribution_small", kind="distribution", rows=backdoor_rows(90, 31), conditioned=False),
    dict(name="distribution_large", kind="distribution", rows=backdoor_rows(600, 37), conditioned=False),
    dict(name="distribution_conditioned", kind="distribution", rows=backdoor_rows(150, 41), conditioned=True),
    dict(name="path_dense", kind="path", rows=path_rows(400, 53)),
    dict(name="path_small", kind="path", rows=path_rows(110, 59)),
    dict(name="path_sparse", kind="path", rows=path_rows(24, 61)),
]


def columns(rows):
    return [[float(r[k]) for r in rows] for k in range(len(rows[0]))]


def data_stage():
    out = []
    for c in CASES:
        out.append({"name": c["name"], "kind": c["kind"], "columns": columns(c["rows"]),
                    "span": c.get("span"), "conditioned": c.get("conditioned")})
    print(json.dumps(out))


# ---- exact functionals -----------------------------------------------------------------

def ey_backdoor(rows, t):
    n = len(rows)
    s = F(0)
    for c in sorted({r[2] for r in rows}):
        cell = [r[1] for r in rows if r[0] == t and r[2] == c]
        if not cell:
            raise Undefined
        s += F(sum(1 for r in rows if r[2] == c), n) * F(sum(cell), len(cell))
    return s


def effect_backdoor(rows):
    return ey_backdoor(rows, 1) - ey_backdoor(rows, 0)


def law_backdoor(rows, conditioned):
    """{group: {y: prob}}; the unconditional law has the single group ()."""
    n = len(rows)
    cs = sorted({r[2] for r in rows})
    ys = sorted({r[1] for r in rows})
    out = {}
    if conditioned:
        for c in cs:
            cell = [r for r in rows if r[0] == 1 and r[2] == c]
            if not cell:
                raise Undefined
            out[(c,)] = {y: F(sum(1 for r in cell if r[1] == y), len(cell)) for y in ys}
    else:
        law = {y: F(0) for y in ys}
        for c in cs:
            cell = [r for r in rows if r[0] == 1 and r[2] == c]
            if not cell:
                raise Undefined
            for y in ys:
                law[y] += F(sum(1 for r in rows if r[2] == c), n) * F(sum(1 for r in cell if r[1] == y), len(cell))
        out[()] = law
    return out


def effect_path(rows):
    n = len(rows)
    cs = sorted({r[3] for r in rows})
    ms = sorted({r[1] for r in rows})

    def ey(t_med):
        s = F(0)
        for c in cs:
            den = [r for r in rows if r[0] == t_med and r[3] == c]
            if not den:
                raise Undefined
            for m in ms:
                pm = F(sum(1 for r in den if r[1] == m), len(den))
                cell = [r[2] for r in rows if r[0] == 0 and r[1] == m and r[3] == c]
                if not cell:
                    if pm == 0:
                        continue
                    raise Undefined
                s += F(sum(1 for r in rows if r[3] == c), n) * pm * F(sum(cell), len(cell))
        return s

    return ey(1) - ey(0)


def tv(original, estimate):
    worst = F(0)
    for g in sorted(set(original) | set(estimate)):
        if g not in original or g not in estimate:
            return F(1)
        a, b = original[g], estimate[g]
        worst = max(worst, F(1, 2) * sum(abs(a.get(y, F(0)) - b.get(y, F(0))) for y in set(a) | set(b)))
    return worst


# ---- p-values ----------------------------------------------------------------------------

def p_value(samples, null):
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    m, h = float(mean), float(null)
    sd = math.sqrt(float(var))
    scale = max(abs(m), abs(h), 1.0)
    if sd <= 1e-12 * scale:
        return float(mean), (1.0 if abs(h - m) <= 1e-9 * scale else 0.0)
    z = (h - m) / sd
    return float(mean), min(max(math.erfc(abs(z) / math.sqrt(2)), 0.0), 1.0)


def report(rid, original, samples, null, informative=False):
    mean, p = p_value(samples, null)
    return {"id": rid, "original": float(original), "refuted": mean, "comparison": p,
            "passed": p >= ALPHA, "informative": informative, "replicates": len(samples)}


# ---- refuters ----------------------------------------------------------------------------

def kept(rows, mask):
    return [r for r, k in zip(rows, mask) if k]


def stretch(rows, frac, u):
    n = len(rows)
    length = min(max(math.floor(n * frac), 1), n - 1)
    span = n - length
    start = min(math.floor(u * (span + 1.0)), span)
    return rows[start:start + length]


def rotate(rows, col, shift):
    n = len(rows)
    by = shift % n
    out = []
    for i in range(n):
        r = list(rows[i])
        r[col] = rows[(i + by) % n][col]
        out.append(tuple(r))
    return out


def window_runs(case, pert):
    rows, span = case["rows"], case["span"]
    n = len(rows)
    exact = effect_backdoor(rows)
    results = []
    for full, delta in RUNS:
        original = F(float(exact) + delta)
        try:
            reports = []
            for frac, rid, key in ([(0.8, "window.stretch_80", "80")] +
                                   ([(0.5, "window.stretch_50", "50")] if full else [])):
                vals = [effect_backdoor(stretch(rows, frac, u)) for u in pert["stretch_u"][key]]
                reports.append(report(rid, original, vals, original))
            rotations = max(n + 1 - 2 * span, 0)
            if rotations == 0:
                reports.append({"id": "window.placebo_treatment", "original": float(original),
                                "refuted": 0.0, "comparison": 0.0, "passed": False,
                                "informative": False, "replicates": 0})
            else:
                vals = []
                for u in pert["shift_u"]:
                    offset = min(math.floor(u * rotations), rotations - 1)
                    vals.append(effect_backdoor(rotate(rows, 0, span + offset)))
                reports.append(report("window.placebo_treatment", original, vals, F(0)))
            if full:
                half = n // 2
                a, b = effect_backdoor(rows[:half]), effect_backdoor(rows[half:])
                se_a, se_b = pert["half_se"]
                # A half that one circular block covers cannot be resampled: its standard error is
                # not an estimate, so the test is unusable rather than a z test with zero spread.
                one_block = [bl >= rows_ for bl, rows_ in zip(pert["half_block"], (half, n - half))]
                variance = (None if se_a is None or se_b is None or any(one_block)
                            else se_a * se_a + se_b * se_b)
                if variance is None:
                    reports.append({"id": "window.split_half", "original": float(a), "refuted": float(b),
                                    "comparison": 0.0, "passed": False, "informative": False,
                                    "replicates": 40})
                    results.append({"full": full, "delta": delta, "original": float(original), "reports": reports})
                    continue
                if variance > 0:
                    z = (float(a) - float(b)) / math.sqrt(variance)
                    p = min(max(math.erfc(abs(z) / math.sqrt(2)), 0.0), 1.0)
                elif abs(float(a - b)) <= 1e-12:
                    p = 1.0
                else:
                    p = 0.0
                reports.append({"id": "window.split_half", "original": float(a), "refuted": float(b),
                                "comparison": p, "passed": p >= ALPHA, "informative": True,
                                "replicates": 40})
            results.append({"full": full, "delta": delta, "original": float(original), "reports": reports})
        except Undefined:
            results.append({"full": full, "delta": delta, "original": float(original), "reports": "error"})
    return results


def distribution_runs(case, pert):
    rows, conditioned = case["rows"], case["conditioned"]
    original = law_backdoor(rows, conditioned)
    normal = all(abs(float(sum(g.values())) - 1.0) <= 1e-8 for g in original.values())
    results = []
    for full in (False, True):
        try:
            reports = [{"id": "distribution.normalization", "original": 1.0, "refuted": float(normal),
                        "comparison": float(normal), "passed": normal, "informative": True,
                        "replicates": 1}]
            for slot, (rid, key) in enumerate([("distribution.subset_tv", "s0"),
                                               ("distribution.half_sample_tv", "s1")][:2 if full else 1]):
                dists = [tv(original, law_backdoor(kept(rows, m), conditioned)) for m in pert["masks"][key]]
                mean = sum(dists) / len(dists)
                reports.append({"id": rid, "original": 0.0, "refuted": float(mean), "comparison": float(mean),
                                "passed": mean <= F(1, 10), "informative": True, "replicates": REPS,
                                "boundary": abs(float(mean) - 0.1) < 1e-9})
            results.append({"full": full, "reports": reports})
        except Undefined:
            results.append({"full": full, "reports": "error"})
    law = [[list(g), y, float(p)] for g, d in original.items() for y, p in d.items()]
    tampered = []
    for s in (1.0, 1.01, 1 + 1e-9, 1 + 1e-7, 1 - 5e-9, 0.99):
        ok = all(abs(float(sum(F(v) * F(s) for v in g.values())) - 1.0) <= 1e-8
                 and all(0 <= F(v) * F(s) <= 1 for v in g.values()) for g in original.values())
        tampered.append({"scale": s, "normalized": ok})
    return {"law": law, "tampered": tampered, "runs": results}


def path_runs(case, pert):
    rows = case["rows"]
    exact = effect_path(rows)
    results = []
    for full, delta in RUNS:
        original = F(float(exact) + delta)
        try:
            reports = []
            for rid, key in [("path.subset", "s0"), ("path.half_sample", "s1")][:2 if full else 1]:
                vals = [effect_path(kept(rows, m)) for m in pert["masks"][key]]
                reports.append(report(rid, original, vals, original))
            results.append({"full": full, "delta": delta, "original": float(original), "reports": reports})
        except Undefined:
            results.append({"full": full, "delta": delta, "original": float(original), "reports": "error"})
    return results


def build_stage(data_path, pert_path):
    data = json.load(open(data_path))
    perts = json.load(open(pert_path))
    out = []
    for case, d in zip(CASES, data):
        pert = perts[case["name"]]
        runner = {"window": window_runs, "distribution": distribution_runs, "path": path_runs}[case["kind"]]
        res = runner(case, pert)
        entry = {"case": d, "perturbations": pert}
        if isinstance(res, dict):
            entry.update(res)
        else:
            entry["runs"] = res
        out.append(entry)
    print(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "data":
        data_stage()
    else:
        build_stage(sys.argv[2], sys.argv[3])
