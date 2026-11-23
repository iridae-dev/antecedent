# /// script
# requires-python = ">=3.11"
# ///
"""Exact-rational reference for the posterior-mass mixture of per-atom refutation reports.

  uv run --offline --script dbn_posterior_reference.py > dbn_posterior_reference.json

Written from the stated definition, with no product code. For per-atom reports (w_k, r_k) of one
refuter, atoms with w_k <= 0 and reports whose refuter id differs from the first positive-weight
report are dropped; with W = sum w_k over the kept atoms:
  original = sum w_k original_k / W,  refuted = sum w_k refuted_k / W,
  comparison = sum w_k comparison_k / W;  informative = any informative_k;
  replicates = max replicates_k;  passed = every kept atom passed (a failing atom of any positive
  weight, however small, fails the mixture);  failure = the first failing atom's text.
`mix_weighted` groups a list of per-atom report vectors by refuter id, in order of first
appearance, skips atoms with non-positive weight, and mixes each group.
All sums are exact Fractions of the supplied doubles; the expected values are rounded once.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F


def rep(refuter, original, refuted, comparison, informative, passed, replicates, failure=None):
    return {"refuter": refuter, "original": original, "refuted": refuted, "comparison": comparison,
            "informative": informative, "passed": passed, "replicates": replicates,
            "failure": failure}


def mix(items):
    """items: list of (weight, report). Returns None or the mixed report dict."""
    first = next((r for w, r in items if w > 0), None)
    if first is None:
        return None
    kept = [(F(w), r) for w, r in items if w > 0 and r["refuter"] == first["refuter"]]
    total = sum(w for w, _ in kept)
    failing = [r for _, r in kept if not r["passed"]]
    return {
        "refuter": first["refuter"],
        "original": float(sum(w * F(r["original"]) for w, r in kept) / total),
        "refuted": float(sum(w * F(r["refuted"]) for w, r in kept) / total),
        "comparison": float(sum(w * F(r["comparison"]) for w, r in kept) / total),
        "informative": any(r["informative"] for _, r in kept),
        "passed": not failing,
        "failure": None if not failing else failing[0]["failure"],
        "replicates": max(r["replicates"] for _, r in kept),
    }


CASES = []


def case(name, items):
    CASES.append({"name": name, "items": [{"weight": w, "report": r} for w, r in items],
                  "expected": mix(items)})


ok = lambda i, **kw: rep("placebo.treatment", 1.0 + i, 0.1 * i, 0.5 + 0.01 * i, False, True, 20 + i, **kw)
bad = lambda i, text: rep("placebo.treatment", 2.0 + i, -0.3 * i, 0.01, True, False, 7, failure=text)

case("all_pass_ordinary_masses", [(0.5, ok(1)), (0.3, ok(2)), (0.2, ok(3))])
case("one_failure_mixture_fails", [(0.5, ok(1)), (0.3, bad(2, "atom 2 failed")), (0.2, ok(3))])
case("first_failure_text_wins", [(0.1, bad(1, "first")), (0.6, bad(2, "second")), (0.3, ok(3))])
case("failing_atom_of_negligible_mass_still_fails", [(1.0, ok(1)), (1e-14, bad(2, "tiny atom failed"))])
case("failing_atom_of_mass_1e-300_still_fails", [(1.0, ok(1)), (1e-300, bad(2, "denormal-scale atom"))])
case("zero_and_negative_weights_dropped", [(0.0, bad(1, "zero")), (-0.4, bad(2, "negative")), (0.7, ok(3)), (0.3, ok(4))])
case("mismatched_refuter_dropped", [(0.4, ok(1)), (0.4, rep("dummy.outcome", 9.0, 9.0, 0.9, True, False, 99, failure="other")), (0.2, ok(3))])
case("first_positive_atom_names_the_refuter", [(0.0, rep("dummy.outcome", 1.0, 1.0, 1.0, False, True, 1)), (0.5, ok(1)), (0.5, ok(2))])
case("huge_equal_weights_do_not_overflow", [(1e300, rep("rcc", 5e10, 6e10, 0.3, False, True, 20)), (1e300, rep("rcc", 7e10, 8e10, 0.5, False, True, 20))])
case("huge_weights_with_a_failure", [(1e305, rep("rcc", 5e10, 6e10, 0.3, False, True, 20)), (1e305, rep("rcc", 7e10, 8e10, 0.5, True, False, 20, failure="rcc failed"))])
case("tiny_weights_underflow_free", [(1e-300, ok(1)), (3e-300, ok(2))])
case("mixed_signs_cancel", [(0.5, rep("x", 1e16, -1e16, 0.0, False, True, 1)), (0.5, rep("x", -1e16, 1e16, 0.0, False, True, 1))])
case("single_atom", [(0.37, ok(5))])
case("no_positive_weight", [(0.0, ok(1)), (-1.0, ok(2))])

PLAN = []
suites = [
    (0.6, [rep("mediation.placebo_mediator", 0.0, 0.01, 0.7, False, True, 20),
           rep("mediation.random_common_cause", 1.2, 1.1, 0.4, False, True, 20)]),
    (0.4, [rep("mediation.placebo_mediator", 0.0, -0.02, 0.2, False, False, 20, failure="placebo"),
           rep("mediation.random_common_cause", 0.8, 0.9, 0.6, False, True, 20),
           rep("mediation.contiguous_window", 0.8, 0.85, 0.9, True, True, 19)]),
    (0.0, [rep("mediation.placebo_mediator", 5.0, 5.0, 0.0, True, False, 20, failure="zero weight")]),
]
groups = {}
order = []
for w, reports in suites:
    if w <= 0:
        continue
    for r in reports:
        if r["refuter"] not in groups:
            groups[r["refuter"]] = []
            order.append(r["refuter"])
        groups[r["refuter"]].append((w, r))
PLAN.append({"name": "two_atoms_three_refuters",
             "atoms": [{"weight": w, "reports": rs} for w, rs in suites],
             "expected": [mix(groups[k]) for k in order]})

print(json.dumps({"cases": CASES, "plans": PLAN}))
