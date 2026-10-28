# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Independent tabulation for the discovery-stability wrappers.

Reads the `inputs` section of `discovery_stability_reference.json` (what the library's own
perturbation streams and the reviewed discovery engines produced on each perturbed series) and
rewrites the `expected` section from the definitions alone, in exact rational arithmetic:

* the gapped block layout (a separator of `gap` missing rows wherever the next drawn row does not
  follow the previous one, values and validity of every position, and the property that no window
  of `gap + 1` valid rows straddles a junction);
* link-retention frequencies (count / replicates) and their order (frequency descending, ties in
  link order), for block bootstrap, orientation (directed, undirected, conflict rate), regime and
  parameter grids;
* the lagged-link family size, the standard error of a null-hit rate (the larger of the binomial
  value and the run-to-run spread), the false-positive check verdict and the synthetic-null band;
* the agreement of two link sets (shared fraction and Jaccard, undefined on an empty denominator).

Standard library only. The discovery results themselves are inputs, not recomputed here: they
are produced by the engines reviewed in their own records.

Run: uv run --offline crates/antecedent-validate/tests/oracle_data/discovery_stability_reference.py
"""

import json
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 60
PATH = Path(__file__).with_name("discovery_stability_reference.json")


def layout(case):
    n, block, gap, idx = case["n"], case["block"], case["gap"], case["idx"]
    assert len(idx) == n and all(0 <= i < n for i in idx)
    # moving-block structure: inside a block of `block` draws the rows are consecutive
    for pos in range(1, n):
        if pos % block != 0:
            assert idx[pos] == idx[pos - 1] + 1, "row inside a block must follow its predecessor"
    source_rows, valid, v0, v1 = [], [], [], []
    for pos, row in enumerate(idx):
        if pos > 0 and row != idx[pos - 1] + 1:
            for _ in range(gap):
                source_rows.append(idx[pos - 1])
                valid.append(False)
                v0.append(0.0)
                v1.append(0.0)
        source_rows.append(row)
        valid.append(True)
        v0.append(float(row))
        v1.append(-float(row) - 1.0)
    length = len(valid)
    straddles = 0
    for start in range(0, length - gap):
        window = range(start, start + gap + 1)
        if all(valid[i] for i in window):
            if not all(source_rows[i] == source_rows[i - 1] + 1 for i in list(window)[1:]):
                straddles += 1
    assert straddles == 0
    return {
        "len": length,
        "source_rows": source_rows,
        "valid": valid,
        "values0": v0,
        "values1": v1,
        "junctions": sum(1 for p in range(1, n) if idx[p] != idx[p - 1] + 1),
    }


def table(reps, count, block_size, key=lambda k: tuple(k)):
    counts = {}
    for links in reps:
        assert len({key(l) for l in links}) == len(links), "a replicate lists a link twice"
        for link in {key(l) for l in links}:
            counts[link] = counts.get(link, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: (-Fraction(kv[1], count), kv[0]))
    return {
        "replicates": count,
        "block_size": block_size,
        "frequencies": [[list(k), float(Fraction(c, count))] for k, c in rows],
    }


def to_dec(fr):
    return Decimal(fr.numerator) / Decimal(fr.denominator)


def null_rate_se_sq(alpha, family, hits):
    alpha = Fraction(alpha)
    runs = len(hits)
    binom = alpha * (1 - alpha) / (runs * family)
    if runs < 2:
        return binom
    mean = Fraction(sum(hits), runs)
    var = sum((Fraction(h) - mean) ** 2 for h in hits) / (runs - 1)
    return max(binom, var / runs / family**2)


def dec_sqrt(fr):
    return to_dec(fr).sqrt()


def family_size(n, lo, hi):
    if lo == 0 or hi < lo:
        return None
    return n * n * (hi - lo + 1)


def margin_ok(value):
    assert abs(value) > Decimal("1e-9"), f"verdict too close to its boundary: {value}"


def fp_report(case, one_sided):
    fam = family_size(case["variables"], case["min_lag"], case["max_lag"])
    hits = case["hits"]
    runs = len(hits)
    alpha = Fraction(case["alpha"])
    mean = Fraction(sum(hits), runs)
    se = dec_sqrt(null_rate_se_sq(alpha, fam, hits))
    if one_sided:
        fpr = mean / fam
        gap = Decimal(alpha.numerator) / Decimal(alpha.denominator) + 3 * se - to_dec(fpr)
        margin_ok(gap)
        return {
            "mean_edge_count": float(mean),
            "empirical_fpr": float(fpr),
            "se": float(se),
            "family": fam,
            "passed": gap >= 0,
        }
    fpr = Fraction(sum(hits), runs * fam)
    diff = abs(to_dec(fpr) - to_dec(alpha))
    band = Decimal(str(case["band_tol"])) * se
    margin_ok(band - diff)
    return {"empirical_fpr": float(fpr), "se": float(se), "family": fam, "within_band": diff <= band}


def agreement(train, holdout):
    t, h = {tuple(l) for l in train}, {tuple(l) for l in holdout}
    shared, union = len(t & h), len(t | h)
    return {
        "shared": None if not t else float(Fraction(shared, len(t))),
        "jaccard": None if union == 0 else float(Fraction(shared, union)),
    }


def orientation(block):
    reps, count = block["reps"], block["replicates"]
    directed = table([r["directed"] for r in reps], count, block["block_size"])
    und = {}
    for r in reps:
        for pair in {tuple(p) for p in r["undirected"]}:
            und[pair] = und.get(pair, 0) + 1
    und_rows = sorted(und.items(), key=lambda kv: (-Fraction(kv[1], count), kv[0]))
    conflicts = sum(1 for r in reps if r["conflict"])
    return {
        "directed": directed,
        "undirected": [[list(k), float(Fraction(c, count))] for k, c in und_rows],
        "conflict_rate": float(Fraction(conflicts, count)),
    }


def regime(block):
    count = block["replicates"]
    out = {}
    for regime_id in block["regimes"]:
        per = [
            links
            for rep in block["reps"]
            for rid, links in rep
            if rid == regime_id
        ]
        out[str(regime_id)] = table(per, count, block["block_size"])
    return out


def main():
    data = json.loads(PATH.read_text())
    inp = data["inputs"]
    exp = {}
    exp["layout"] = [layout(c) for c in inp["layout"]]
    b = inp["block"]
    exp["block"] = table(b["links"], b["replicates"], b["block_size"])
    exp["orientation"] = {k: orientation(v) for k, v in inp["orientation"].items()}
    exp["regime"] = regime(inp["regime"])
    exp["false_positive"] = {k: fp_report(v, True) for k, v in inp["false_positive"].items()}
    exp["null"] = fp_report(inp["null"], False)
    exp["env"] = [agreement(c["train"], c["holdout"]) for c in inp["env"]]
    exp["grid"] = {k: table(v, len(v), 0) for k, v in inp["grid"].items()}
    ar = inp["arithmetic"]
    exp["arithmetic"] = {
        "se": [
            float(dec_sqrt(null_rate_se_sq(c["alpha"], c["family"], c["hits"])))
            for c in ar["hits_cases"]
        ],
        "family": [family_size(*c) for c in ar["family_cases"]],
    }
    data["expected"] = exp
    PATH.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", PATH, file=sys.stderr)


if __name__ == "__main__":
    main()
