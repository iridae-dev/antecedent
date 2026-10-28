# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Honest causal forest against an exact-rational tree reference with a ported random stream.

Regenerate with:  uv run --offline --script estimate_causal_forest_reference.py > estimate_causal_forest_reference.json

The forest is a deterministic function of its data, its settings and the raw 64-bit draws of one
SplitMix64 stream per tree.  This script ports that stream (seed mixing, stream id packing and
the SplitMix64 step, written here from the published SplitMix64 definition and the mixing
constants the library documents) and then evaluates the forest by its definition with `Fraction`
arithmetic on the exact binary values of the inputs:

* subsample: a partial Fisher-Yates shuffle of the row ids with draw modulo the remaining count,
  truncated to about half of the rows (never below two minimum leaves); with honesty the first
  half of the shuffled rows places splits and the second half estimates leaf effects; for a
  declared cluster unit the same shuffle runs over the clusters in ascending label order;
* split search: every candidate cut (midpoints of consecutive distinct values, thinned by a
  stride) of every chosen feature; a cut is admitted when both children keep at least the
  minimum leaf and both arms; the score is (nl nr / n^2) (tau_left - tau_right)^2 and the first
  strictly larger score wins in feature-then-cut order; the script refuses a case whose best
  score is within 1e-9 relative of a different candidate, so no float tie-break is in play;
* honest leaf: the difference of arm means on the estimation rows that reach the leaf, with the
  two-sample variance of that difference when each arm has two rows, and the parent's estimate
  when the leaf holds no two-arm rows;
* aggregation: the per-row forest effect is the mean over trees with an estimate, the leaf
  dispersion the root of the mean honest leaf variance over trees that have one.

The product's candidate-cut convention (consecutive values within 1e-12 are one value; the cut is
0.5 a + 0.5 b in binary64) is reproduced as a convention, not derived.  Values are written as
nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import random
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60
M = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def mix64(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def derive_stream(master, stream_id):
    z = (master * 0xD6E8FEB86659FD93) & M
    z ^= (stream_id * 0xC2B2AE3D27D4EB4F) & M
    z = ((z ^ (z >> 32)) * 0x165667B19E3779F9) & M
    z = ((z ^ (z >> 32)) * 0xD6E8FEB86659FD93) & M
    return z ^ (z >> 32)


ESTIMATE_TAG = 14


class Stream:
    def __init__(self, master, tree):
        index = derive_stream(0xC0F00000, tree)
        stream_id = ((ESTIMATE_TAG * 0xD1B54A32D192ED03) & M) ^ index
        self.state = derive_stream(master, stream_id) ^ GAMMA
        self.used = []

    def next(self):
        self.state = (self.state + GAMMA) & M
        out = mix64(self.state)
        self.used.append(out)
        return out


def round_half_up(v):
    return math.floor(v + 0.5)


def candidate_thresholds(values, max_cuts):
    if len(values) < 2:
        return []
    vals = [values[0]]
    for v in values[1:]:
        if abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    if len(vals) < 2:
        return []
    gaps = len(vals) - 1
    stride = max(-(-gaps // max_cuts), 1)
    return [0.5 * vals[i] + 0.5 * vals[i + 1] for i in range(0, gaps, stride)]


class Case:
    def __init__(self, x, y, t, min_leaf, max_depth, max_cuts, honesty, clusters=None):
        self.x, self.y, self.t = x, y, t  # x[feature][row]
        self.n, self.p = len(y), len(x)
        self.min_leaf, self.max_depth, self.max_cuts = min_leaf, max_depth, max_cuts
        self.honesty = honesty
        self.clusters = clusters
        self.min_margin = None

    def arm_sums(self, rows):
        s1 = n1 = s0 = n0 = 0
        for r in rows:
            if self.t[r] == 1.0:
                s1 += F(self.y[r])
                n1 += 1
            else:
                s0 += F(self.y[r])
                n0 += 1
        return s1, n1, s0, n0

    def leaf_stat(self, rows):
        s1, n1, s0, n0 = self.arm_sums(rows)
        if n1 < 1 or n0 < 1:
            return None
        tau = s1 / n1 - s0 / n0
        if n1 < 2 or n0 < 2:
            return (tau, None)
        m1, m0 = s1 / n1, s0 / n0
        v1 = sum((F(self.y[r]) - m1) ** 2 for r in rows if self.t[r] == 1.0) / (n1 - 1)
        v0 = sum((F(self.y[r]) - m0) ** 2 for r in rows if self.t[r] != 1.0) / (n0 - 1)
        return (tau, v1 / n1 + v0 / n0)

    def best_split(self, rows, stream):
        p = self.p
        max_cuts = self.max_cuts if self.max_cuts is not None else max(16, math.isqrt(len(rows)))
        feats = list(range(p))
        if p > 4:
            mtry = math.ceil(math.sqrt(p))
            for i in range(mtry):
                j = i + (stream.next() % (p - i))
                feats[i], feats[j] = feats[j], feats[i]
            feats = feats[:mtry]
        scores = []
        for f in feats:
            order = sorted(rows, key=lambda r: self.x[f][r])
            values = [self.x[f][r] for r in order]
            for thr in candidate_thresholds(values, max_cuts):
                left = [r for r in order if self.x[f][r] <= thr]
                right = [r for r in order if self.x[f][r] > thr]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                a, b = self.leaf_stat_point(left), self.leaf_stat_point(right)
                if a is None or b is None:
                    continue
                nl, nr = len(left), len(right)
                score = F(nl * nr, (nl + nr) ** 2) * (a - b) ** 2
                scores.append((score, f, thr))
        if not scores:
            return None
        best = None
        for s in scores:
            if best is None or s[0] > best[0]:
                best = s
        others = [s[0] for s in scores if s is not best]
        if others and best[0] > 0:
            gap = float((best[0] - max(others)) / best[0])
            self.min_margin = gap if self.min_margin is None else min(self.min_margin, gap)
            if gap < 1e-9:
                raise SystemExit("split score tie within 1e-9: change the case")
        return best[1], best[2]

    def leaf_stat_point(self, rows):
        s1, n1, s0, n0 = self.arm_sums(rows)
        if n1 < 1 or n0 < 1:
            return None
        return s1 / n1 - s0 / n0

    def grow(self, rows, depth, stream):
        if depth >= self.max_depth or len(rows) < 2 * self.min_leaf or self.p == 0:
            return ["leaf", self.leaf_stat(rows)]
        split = self.best_split(rows, stream)
        if split is None:
            return ["leaf", self.leaf_stat(rows)]
        f, thr = split
        left = [r for r in rows if self.x[f][r] <= thr]
        right = [r for r in rows if self.x[f][r] > thr]
        lnode = self.grow(left, depth + 1, stream)
        rnode = self.grow(right, depth + 1, stream)
        return ["split", f, thr, lnode, rnode]

    def fill(self, node, rows, parent):
        honest = self.leaf_stat(rows) or parent
        if node[0] == "leaf":
            node[1] = honest
            return
        _, f, thr, l, r = node
        self.fill(l, [i for i in rows if self.x[f][i] <= thr], honest)
        self.fill(r, [i for i in rows if self.x[f][i] > thr], honest)

    def tree(self, stream):
        n = self.n
        if self.clusters is None:
            want = round_half_up(n * 0.5)
            k = min(max(want, min(2 * self.min_leaf, n)), n)
            idx = list(range(n))
            for i in range(k):
                rest = n - i
                j = i + (stream.next() % rest)
                idx[i], idx[j] = idx[j], idx[i]
            sample = idx[:k]
            if self.honesty:
                split_rows, est_rows = sample[: k // 2], sample[k // 2 :]
            else:
                split_rows, est_rows = sample, sample
        else:
            groups = [
                [r for r in range(n) if self.clusters[r] == label]
                for label in sorted(set(self.clusters))
            ]
            g = len(groups)
            want = round_half_up(g * 0.5)
            k = min(max(want, min(2, g)), g)
            order = list(range(g))
            for i in range(k):
                rest = g - i
                j = i + (stream.next() % rest)
                order[i], order[j] = order[j], order[i]
            order = order[:k]
            flat = lambda cs: [r for c in cs for r in groups[c]]
            if self.honesty:
                split_rows, est_rows = flat(order[: k // 2]), flat(order[k // 2 :])
            else:
                split_rows = est_rows = flat(order)
        root = self.grow(split_rows, 0, stream)
        self.fill(root, est_rows, None)
        return root


def preorder(node, out):
    if node[0] == "leaf":
        out.append({"leaf": None if node[1] is None else float(node[1][0])})
        return
    index = len(out)
    out.append(None)
    preorder(node[3], out)
    right = len(out)
    preorder(node[4], out)
    out[index] = {"feature": node[1], "threshold": node[2], "left": index + 1, "right": right}


def leaf_at(node, case, row):
    while node[0] == "split":
        node = node[3] if case.x[node[1]][row] <= node[2] else node[4]
    return node[1]


def sqrt_float(fr):
    if fr == 0:
        return 0.0
    return float((Decimal(fr.numerator) / Decimal(fr.denominator)).sqrt())


def run(case, trees, seed):
    streams, nodes = [], []
    for b in range(trees):
        s = Stream(seed, b)
        nodes.append(case.tree(s))
        streams.append(s.used)
    n = case.n
    cate, dispersion = [], []
    hits_zero = False
    complete = True
    for i in range(n):
        taus = [leaf_at(nd, case, i) for nd in nodes]
        taus = [t for t in taus if t is not None]
        if not taus:
            hits_zero = True
            cate.append(None)
            dispersion.append(None)
            continue
        cate.append(float(sum(t[0] for t in taus) / len(taus)))
        vars_ = [t[1] for t in taus if t[1] is not None]
        if vars_:
            dispersion.append(sqrt_float(sum(vars_) / len(vars_)))
        else:
            complete = False
            dispersion.append(None)
    out = {
        "seed": seed,
        "draws": [[str(d) for d in used] for used in streams],
        "trees": [],
        "refused_no_estimate": hits_zero,
        "dispersion_complete": complete,
    }
    for nd in nodes:
        flat = []
        preorder(nd, flat)
        out["trees"].append(flat)
    if not hits_zero:
        out["cate"] = cate
        out["dispersion"] = dispersion if complete else None
    return out


def make(name, n, p, seed, *, kind="interaction", ml=4, depth=4, cuts=None, honesty=True,
         trees=8, seeds=(3,), levels=None, y_offset=0.0, x_offset=0.0, clusters=None,
         treated_rows=None):
    rng = random.Random(seed)
    x = [[0.0] * n for _ in range(p)]
    t, y = [0.0] * n, [0.0] * n
    for i in range(n):
        for f in range(p):
            if levels:
                x[f][i] = float(rng.randrange(levels[f]))
            else:
                x[f][i] = round(rng.gauss(0, 1), 3) + x_offset
        z = x[0][i] - x_offset
        pr = 1 / (1 + math.exp(-(-0.3 + 0.6 * z)))
        t[i] = 1.0 if rng.random() < pr else 0.0
        y[i] = round((1 + z) * t[i] + 0.4 * z + 0.4 * rng.gauss(0, 1) + y_offset, 6)
    if treated_rows is not None:
        t = [1.0 if i in treated_rows else 0.0 for i in range(n)]
    labels = None
    if clusters:
        labels = [clusters[i // (n // len(clusters))] for i in range(n)]
    case = Case(x, y, t, ml, depth, cuts, honesty, labels)
    return {
        "name": name,
        "t": t,
        "y": y,
        "x": [[x[f][i] for f in range(p)] for i in range(n)],
        "min_leaf": ml,
        "max_depth": depth,
        "max_cuts": cuts,
        "honesty": honesty,
        "n_trees": trees,
        "clusters": labels,
        "runs": [run(case, trees, s) for s in seeds],
        "min_margin": case.min_margin,
    }, case


def with_treated_split(name, n, position_in, seed, **kw):
    """n = 2 min_leaf, one tree.

    Five treated rows, so the marginal cross-fit has a treated row in every fold.  Either all five
    sit in the placement half of the shuffle (the estimation half then has no treated row) or
    exactly one sits in the estimation half (a one-row arm, so a point estimate with no variance).
    """
    probe = Stream(seed, 0)
    idx = list(range(n))
    for i in range(n):
        j = i + (probe.next() % (n - i))
        idx[i], idx[j] = idx[j], idx[i]
    half = n // 2
    rows = set(idx[:5]) if position_in == "placement" else set(idx[:4]) | {idx[half]}
    return make(name, n, 1, 99, treated_rows=rows, seeds=(seed,), trees=1, ml=n // 2, **kw)[0]


def main():
    cases = []
    cases.append(make("single_covariate_interaction", 60, 1, 11, ml=4, depth=4, trees=10)[0])
    cases.append(make("tied_covariates_without_honesty", 48, 2, 12, ml=3, depth=3, honesty=False,
                      levels=[6, 4], trees=8)[0])
    cases.append(make("feature_subsampling_six_covariates", 64, 6, 13, ml=4, depth=3, trees=8)[0])
    cases.append(make("thinned_cuts", 80, 1, 14, ml=5, depth=3, cuts=3, trees=8)[0])
    cases.append(make("depth_one_stump", 60, 2, 15, ml=3, depth=1, trees=6)[0])
    cases.append(make("min_leaf_is_half_the_table_honest", 16, 1, 16, ml=8, depth=3, trees=12)[0])
    cases.append(make("half_table_without_honesty_is_seed_free", 16, 1, 17, ml=8, depth=3,
                      honesty=False, trees=6, seeds=(1, 7, 99))[0])
    cases.append(with_treated_split("treated_rows_only_in_placement_half", 16, "placement", 5))
    cases.append(with_treated_split("one_treated_row_in_estimation_half", 16, "estimation", 5))
    cases.append(make("outcome_offset_one_e7", 60, 1, 18, ml=5, depth=3, trees=8, y_offset=1e7)[0])
    cases.append(make("covariate_offset_one_e6", 60, 2, 19, ml=4, depth=3, trees=8,
                      x_offset=1e6)[0])
    cases.append(make("whole_cluster_honesty", 60, 1, 20, ml=3, depth=3, trees=8,
                      clusters=[30, 10, 20, 50, 40, 70, 60, 5, 15, 25, 35, 45])[0])
    json.dump({"cases": cases}, sys.stdout)


if __name__ == "__main__":
    main()
