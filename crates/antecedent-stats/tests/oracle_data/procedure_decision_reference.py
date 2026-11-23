# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.decision_theoretic_stopping.

Regenerate with:  uv run --offline --script procedure_decision_reference.py > procedure_decision_reference.json

Two parts, both in exact rational arithmetic (procedure_exact.py):

* `posteriors`: for a grid of Beta-Bernoulli counts of two arms, the posterior probability that each
  arm is best and the posterior expected loss E[max theta] - E[theta_k] of choosing it, from the
  binomial-tail identity for P(Y > X) and E[(Y - X)+] = E[Y; Y > X] - E[X; Y > X].
* `decisions`: the decision report of a posterior-expected-loss or futility rule under a truth,
  from the exact path-enumerated law: probability of a rule stop at each look, no stop, expected
  enrolled units, probability of deciding each arm or none, wrong-decision probability and expected
  regret E[(max theta - theta_decided) 1{decided}].

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

F = pe.F
pe.check_exceedance()

posteriors = []
grid = [(0, 0), (1, 0), (0, 1), (3, 1), (1, 3), (5, 5), (12, 4), (0, 9), (9, 0), (40, 25)]
for (s0, f0) in grid:
    for (s1, f1) in grid:
        counts = [s0, f0, s1, f1]
        best = pe.best_probabilities([1, 1], counts)
        loss = pe.losses([1, 1], counts)
        # E[max] - E[theta_k]: the loss of an arm equals the expected shortfall to the other arm.
        posteriors.append(
            {
                "counts": counts,
                "best": [float(x) for x in best],
                "loss": [float(x) for x in loss],
                "best_exact": [f"{x.numerator}/{x.denominator}" for x in best],
            }
        )

decisions = []


def decision(name, assignment, enrol, truth, stopping, selection, **extra):
    decl = {
        "arms": 2,
        "assignment": assignment,
        "enrolment": enrol,
        "stopping": stopping,
        "selection": selection,
    }
    decl.update(extra)
    atoms = pe.enumerate_law(decl, ("fixed", truth))
    arms = 2
    theta = [pe.frac(t) for t in truth]
    best = max(theta)
    looks = len(enrol)
    stops = [F(0)] * looks
    horizon = F(0)
    enrolled = F(0)
    decided = [F(0)] * arms
    undecided = F(0)
    for (look, stopped, observed, dec), p in atoms.items():
        if stopped:
            stops[look] += p
        else:
            horizon += p
        enrolled += p * sum(enrol[: look + 1])
        if dec is None:
            undecided += p
        else:
            decided[dec] += p
    wrong = sum((decided[a] for a in range(arms) if theta[a] != best), F(0))
    regret = sum((decided[a] * (best - theta[a]) for a in range(arms)), F(0))
    decisions.append(
        {
            "name": name,
            "declaration": decl,
            "truth": truth,
            "stop_probabilities": [float(x) for x in stops],
            "horizon_probability": float(horizon),
            "expected_enrolled": float(enrolled),
            "decision_probabilities": [float(x) for x in decided],
            "undecided_probability": float(undecided),
            "wrong_decision_probability": float(wrong),
            "expected_regret": float(regret),
        }
    )


FIXED = {"kind": "fixed", "probabilities": ["1/2", "1/2"]}
THOMPSON = {"kind": "thompson", "prior": [1, 1]}
LOSS = lambda t: {"kind": "loss", "threshold": t, "prior": [1, 1]}
NO_SELECTION = {"kind": "none"}
decision("loss_rule_unequal_truth", FIXED, [2, 2, 2], ["1/5", "4/5"], LOSS("7/100"), NO_SELECTION)
decision("loss_rule_equal_truth_has_no_wrong_decision", FIXED, [2, 2, 2], ["1/2", "1/2"], LOSS("7/100"), NO_SELECTION)
decision("loss_rule_thompson_with_selection", THOMPSON, [1] * 6, ["1/10", "3/5"], LOSS("1/15"), {"kind": "mean"})
decision("loss_rule_lagged", FIXED, [2, 2, 2, 2], ["1/4", "3/4"], LOSS("1/20"), NO_SELECTION, lag=1)
decision(
    "futility_rule_decides_the_control",
    FIXED,
    [2, 2, 2],
    ["1/2", "3/10"],
    {"kind": "futility", "margin": "0", "threshold": "3/20", "prior": [1, 1]},
    NO_SELECTION,
)
decision(
    "futility_rule_rare_stop_under_advantage",
    FIXED,
    [2, 2, 2],
    ["1/5", "4/5"],
    {"kind": "futility", "margin": "0", "threshold": "1/40", "prior": [1, 1]},
    NO_SELECTION,
)
print(json.dumps({"posteriors": posteriors, "decisions": decisions}, indent=1))
