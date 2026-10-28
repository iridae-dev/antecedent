# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.procedure_ordered_p_value.

Regenerate with:  uv run --offline --script procedure_ordered_reference.py > procedure_ordered_reference.json

The exact law of a fixed-randomization two-arm procedure with Wald boundaries is path-enumerated in
rational arithmetic (procedure_exact.py).  The stagewise ordering ranks a terminal state by
(class, key, estimate): a rule stop for the plus arm ranks above everything (earlier stops higher), a
rule stop for the minus arm below everything (later stops higher), every other terminal state in
between by its estimate (arm-1 mean minus arm-0 mean); a terminal state with an undefined estimate
(an arm without observations) is counted in both tails.  upper = mass of states ranked at least as
high as the observed one, lower = mass ranked at least as low.  The inversion accepts a grid contrast
when, for every baseline of the grid keeping both arm probabilities in [0, 1], the larger of each
tail probability under the (baseline, baseline + contrast) truth is at least alpha / 2.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

F = pe.F


def rank(look, stopped, decision, estimate):
    if stopped and decision == 1:
        return (2, -look, estimate)
    if stopped and decision == 0:
        return (0, look, estimate)
    return (1, 0, estimate)


def p_values(atoms, observed):
    target = rank(*observed)
    upper = lower = undefined = F(0)
    for (look, stopped, obs, dec), p in atoms.items():
        s0, f0, s1, f1 = obs
        if s0 + f0 == 0 or s1 + f1 == 0:
            undefined += p
            continue
        est = F(s1, s1 + f1) - F(s0, s0 + f0)
        r = rank(look, stopped, dec, est)
        if r > target:
            upper += p
        elif r < target:
            lower += p
        else:
            upper += p
            lower += p
    return upper + undefined, lower + undefined, undefined


DECL = {
    "arms": 2,
    "assignment": {"kind": "fixed", "probabilities": ["1/2", "1/2"]},
    "enrolment": [2, 2, 2],
    "stopping": {"kind": "wald", "critical": ["1.9", "1.6", "1.3"], "sidedness": "two_sided"},
    "selection": {"kind": "none"},
}
SINGLES = dict(DECL, enrolment=[1, 1, 1, 1], stopping={"kind": "wald", "critical": ["1.2", "1.2", "1.2", "1.2"], "sidedness": "two_sided"})
UPPER = dict(DECL, enrolment=[3, 3], stopping={"kind": "wald", "critical": ["1.1", "1.0"], "sidedness": "upper"})

cases = []
for name, decl, truths in (
    ("two_sided_pairs", DECL, [["1/4", "3/4"], ["1/2", "1/2"], ["3/4", "1/4"]]),
    ("single_unit_looks_with_undefined_estimates", SINGLES, [["1/2", "1/2"], ["1/8", "7/8"]]),
    ("upper_boundaries", UPPER, [["1/3", "2/3"]]),
):
    for truth in truths:
        atoms = pe.enumerate_law(decl, ("fixed", truth))
        observed = []
        keys = sorted(atoms, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3]))
        for (look, stopped, obs, dec) in keys:
            s0, f0, s1, f1 = obs
            if s0 + f0 == 0 or s1 + f1 == 0:
                continue
            est = F(s1, s1 + f1) - F(s0, s0 + f0)
            up, lo, und = p_values(atoms, (look, stopped, dec, est))
            observed.append(
                {
                    "look": look,
                    "stopped": stopped,
                    "decision": dec,
                    "estimate": f"{est.numerator}/{est.denominator}",
                    "upper": float(up),
                    "lower": float(lo),
                    "undefined": float(und),
                }
            )
        cases.append({"name": name, "declaration": decl, "truth": truth, "observed": observed})

# Inversion on a dyadic grid.
contrasts = [F(k, 8) for k in range(-6, 7)]
baselines = [F(k, 8) for k in range(1, 8)]
inversions = []
smallest = 1.0
for alpha_text, term in (("1/10", (2, False, None, F(3, 8))), ("1/5", (1, False, None, F(-1, 4))), ("1/20", (0, True, 1, F(1, 2)))):
    alpha = pe.frac(alpha_text)
    accepted, evaluated = [], 0
    for delta in contrasts:
        up = lo = F(0)
        for base in baselines:
            other = base + delta
            if not (0 <= base <= 1 and 0 <= other <= 1):
                continue
            evaluated += 1
            atoms = pe.enumerate_law(DECL, ("fixed", [base, other]))
            u, l, _ = p_values(atoms, term)
            up, lo = max(up, u), max(lo, l)
        smallest = min(smallest, float(abs(up - alpha / 2)), float(abs(lo - alpha / 2)))
        if up >= alpha / 2 and lo >= alpha / 2:
            accepted.append(delta)
    inversions.append(
        {
            "alpha": alpha_text,
            "observed": {"look": term[0], "stopped": term[1], "decision": term[2], "estimate": f"{term[3].numerator}/{term[3].denominator}"},
            "contrasts": [f"{c.numerator}/{c.denominator}" for c in contrasts],
            "baselines": [f"{c.numerator}/{c.denominator}" for c in baselines],
            "accepted": [f"{c.numerator}/{c.denominator}" for c in accepted],
            "laws_evaluated": evaluated,
        }
    )
assert smallest > 1e-9, smallest
print(json.dumps({"declaration": DECL, "cases": cases, "inversions": inversions, "smallest_alpha_gap": smallest}, indent=1))
