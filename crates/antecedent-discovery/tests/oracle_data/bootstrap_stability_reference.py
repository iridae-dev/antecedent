# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Selection frequencies of a structure search over resamples, by direct counting, exactly.

Regenerate with:  uv run --script bootstrap_stability_reference.py > bootstrap_stability_reference.json

A structure is a set of marked node pairs (a < b, mark at a, mark at b) with marks in
{tail, arrow, circle}.  Given the structures a search returned on `succeeded` of `requested`
resamples and the structure it returned on the full sample, the definition-level frequencies are

  adjacency(a, b)      = #{resamples joining a-b} / succeeded
  mark(a, b, ma, mb)   = #{resamples with those marks on a-b} / succeeded
  structure(s)         = #{resamples equal to s as a mark set} / succeeded
  reference_frequency  = structure(reference)

and the table is refused unless succeeded >= 2 and 2 * succeeded >= requested.  Everything here
is Fraction arithmetic over seeded pseudo-random structures that include skewed distributions,
many ties between whole-structure counts, a reference no resample returned, and a single
distinct structure.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

MARKS = ["tail", "arrow", "circle"]
NODES = 4


def random_structure(rng, density):
    edges = {}
    for a in range(NODES):
        for b in range(a + 1, NODES):
            if rng.random() < density:
                edges[(a, b)] = (rng.choice(MARKS), rng.choice(MARKS))
    return edges


def to_list(s):
    return [[a, b, ma, mb] for (a, b), (ma, mb) in sorted(s.items())]


def frac(f):
    return [f.numerator, f.denominator]


def case(name, reference, resamples, requested):
    succeeded = len(resamples)
    ok = requested > 0 and succeeded <= requested and succeeded >= 2 and 2 * succeeded >= requested
    out = {"name": name, "requested": requested, "reference": to_list(reference),
           "resamples": [to_list(s) for s in resamples], "ok": ok}
    if not ok:
        return out
    total = F(succeeded)
    pairs = {}
    for s in resamples:
        for k, m in s.items():
            adj, marks = pairs.setdefault(k, [0, {}])
            pairs[k][0] += 1
            marks[m] = marks.get(m, 0) + 1
    for k in reference:
        pairs.setdefault(k, [0, {}])
    out["edges"] = [
        {"a": a, "b": b, "frequency": frac(F(adj) / total), "in_reference": (a, b) in reference,
         "marks": [[ma, mb, frac(F(c) / total)] for (ma, mb), c in marks.items()]}
        for (a, b), (adj, marks) in sorted(pairs.items())
    ]
    counts = {}
    for s in resamples:
        key = tuple(sorted(s.items()))
        counts[key] = counts.get(key, 0) + 1
    out["structures"] = [{"edges": [[a, b, ma, mb] for (a, b), (ma, mb) in k], "count": c,
                          "frequency": frac(F(c) / total)} for k, c in counts.items()]
    out["reference_frequency"] = frac(F(counts.get(tuple(sorted(reference.items())), 0)) / total)
    out["succeeded"] = succeeded
    return out


def main():
    rng = random.Random(20260428)
    cases = []
    # Skewed pool: a few structures drawn with very unequal weights, many resamples.
    pool = [random_structure(rng, 0.6) for _ in range(6)]
    weights = [50, 20, 12, 8, 6, 4]
    skew = [rng.choices(pool, weights)[0] for _ in range(97)]
    cases.append(case("skewed_pool_with_failures", pool[1], skew, 120))
    # Fully tied whole-structure counts: four structures, 7 resamples each.
    tied_pool = [random_structure(rng, 0.5) for _ in range(4)]
    tied = [s for s in tied_pool for _ in range(7)]
    rng.shuffle(tied)
    cases.append(case("tied_counts", tied_pool[2], tied, 28))
    # Many distinct structures (little repetition).
    many = [random_structure(rng, 0.7) for _ in range(60)]
    cases.append(case("many_distinct", many[0], many, 60))
    # A single structure returned every time.
    one = random_structure(rng, 0.8)
    cases.append(case("single_structure", one, [one] * 9, 9))
    # A reference that no resample returned; its edges are listed with zero frequency.
    other = [random_structure(rng, 0.3) for _ in range(5)]
    ref = {(0, 1): ("circle", "arrow"), (2, 3): ("tail", "tail")}
    cases.append(case("reference_never_returned", ref, other, 8))
    # Empty structures among the resamples.
    cases.append(case("empty_structures", one, [{}, {}, one, {}], 4))
    # Acceptance boundary: exactly half, just under half, one success, none requested.
    cases.append(case("exactly_half", one, [one, {}], 4))
    cases.append(case("just_under_half", one, [one, {}], 5))
    cases.append(case("one_success", one, [one], 1))
    cases.append(case("none_requested", one, [one], 0))
    cases.append(case("more_than_requested", one, [one, one, one], 2))
    print(json.dumps(cases, indent=1))


main()
