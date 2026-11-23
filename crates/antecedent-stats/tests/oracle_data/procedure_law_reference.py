# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference law of declared adaptive procedures for adaptive_inference.procedure_law.

Regenerate with:  uv run --offline --script procedure_law_reference.py > procedure_law_reference.json

Each case is a complete procedure declaration and a hypothesized truth (fixed success
probabilities or independent Beta priors, i.e. the prior predictive), expanded path by path in
exact rational arithmetic by procedure_exact.py.  Repository-authored (MIT OR Apache-2.0); standard
library only.  Cases cover the assignment rules, delay, outcome loss, exposure failure with both
attributions, every stopping rule, every final selection, ties, three arms, extreme truths and
tiny samples.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

pe.check_exceedance()

FIXED = lambda *p: {"kind": "fixed", "probabilities": list(p)}
NO_STOP = {"kind": "none"}
SEL_MEAN = {"kind": "mean"}
SEL_NONE = {"kind": "none"}


def case(name, arms, assignment, enrol, kernel, stopping=NO_STOP, selection=SEL_MEAN, ties=False, **extra):
    decl = {
        "arms": arms,
        "assignment": assignment,
        "enrolment": enrol,
        "stopping": stopping,
        "selection": selection,
    }
    decl.update(extra)
    pe.Margin.smallest = pe.F(1)
    atoms = pe.enumerate_law(decl, kernel)
    return {
        "name": name,
        "declaration": decl,
        "kernel": {"kind": kernel[0], "values": kernel[1]},
        "atoms": pe.atoms_json(atoms),
        "smallest_threshold_gap": float(pe.Margin.smallest),
        "threshold_ties": ties,
    }


cases = [
    case("fixed_unequal_batches", 2, FIXED("1/3", "2/3"), [2, 1, 2], ("fixed", ["1/4", "3/5"])),
    case(
        "three_arms_equal_truth_ties_select_first",
        3,
        FIXED("1/2", "1/4", "1/4"),
        [3, 2],
        ("fixed", ["1/2", "1/2", "1/2"]),
    ),
    case(
        "epsilon_greedy_with_lag_one",
        2,
        {"kind": "epsilon_greedy", "epsilon": "1/5", "prior": [1, 1]},
        [1, 1, 1, 1, 1],
        ("fixed", ["2/5", "3/5"]),
        selection={"kind": "posterior_mean", "prior": [2, 2]},
        lag=1,
    ),
    case(
        "epsilon_greedy_three_arms_tie_split",
        3,
        {"kind": "epsilon_greedy", "epsilon": "1/4", "prior": [1, 1]},
        [1, 1, 1, 1],
        ("fixed", ["1/3", "1/2", "2/3"]),
        selection=SEL_NONE,
    ),
    case(
        "thompson_stops_on_probability_best",
        2,
        {"kind": "thompson", "prior": [1, 1]},
        [1] * 6,
        ("fixed", ["1/5", "4/5"]),
        stopping={"kind": "best", "threshold": "73/100", "prior": [1, 1]},
    ),
    case(
        "thompson_floored_extreme_truth",
        2,
        {"kind": "thompson_floored", "prior": [1, 1], "floor": "3/40"},
        [1] * 6,
        ("fixed", ["1/1000", "999/1000"]),
        selection={"kind": "posterior_mean", "prior": [1, 1]},
    ),
    case(
        "ucb_three_arms_resource_cap",
        3,
        {"kind": "ucb", "exploration": "2"},
        [1] * 6,
        ("fixed", ["1/4", "1/2", "3/4"]),
        stopping={"kind": "cap", "max_units": 5},
    ),
    case(
        "exposure_failure_attributed_to_assigned",
        2,
        FIXED("1/2", "1/2"),
        [2, 2],
        ("fixed", ["1/3", "2/3"]),
        exposure={"failure": ["0", "1/4"], "fallback": 0, "attribution": "assigned"},
    ),
    case(
        "exposure_failure_attributed_to_delivered",
        2,
        FIXED("1/2", "1/2"),
        [2, 2],
        ("fixed", ["1/3", "2/3"]),
        exposure={"failure": ["0", "1/4"], "fallback": 0, "attribution": "delivered"},
    ),
    case(
        "outcome_dependent_loss_with_policy_feedback",
        2,
        {"kind": "epsilon_greedy", "epsilon": "1/5", "prior": [1, 1]},
        [1, 1, 1, 1],
        ("fixed", ["1/2", "3/4"]),
        retain=["1/2", "1"],
    ),
    case(
        "loss_and_lag_two_with_exposure",
        2,
        {"kind": "epsilon_greedy", "epsilon": "1/2", "prior": [1, 1]},
        [1, 1, 1, 1],
        ("fixed", ["1/2", "3/4"]),
        retain=["3/4", "1/2"],
        lag=2,
        exposure={"failure": ["1/5", "0"], "fallback": 1, "attribution": "delivered"},
    ),
    case(
        "prior_predictive_thompson_expected_loss_stop",
        2,
        {"kind": "thompson", "prior": [1, 1]},
        [1] * 6,
        ("predictive", [[1, 1], [2, 3]]),
        stopping={"kind": "loss", "threshold": "7/100", "prior": [1, 1]},
    ),
    case(
        "prior_predictive_lagged_fixed",
        2,
        FIXED("1/2", "1/2"),
        [2, 1, 1],
        ("predictive", [[2, 1], [1, 2]]),
        lag=1,
        retain=["4/5", "4/5"],
    ),
    case(
        "futility_stop_selects_the_control",
        2,
        FIXED("1/2", "1/2"),
        [2, 2, 2],
        ("fixed", ["1/5", "4/5"]),
        stopping={"kind": "futility", "margin": "0", "threshold": "3/20", "prior": [1, 1]},
        selection=SEL_NONE,
    ),
    case(
        "wald_two_sided_boundaries",
        2,
        FIXED("1/2", "1/2"),
        [3, 3, 3],
        ("fixed", ["1/4", "3/4"]),
        stopping={"kind": "wald", "critical": ["2.0", "1.5", "1.4"], "sidedness": "two_sided"},
    ),
    case(
        "wald_upper_boundaries_adaptive_batches",
        2,
        {"kind": "epsilon_greedy", "epsilon": "1/2", "prior": [1, 1]},
        [2, 2, 2],
        ("fixed", ["1/4", "3/4"]),
        stopping={"kind": "wald", "critical": ["1.2", "1.3", "1.4"], "sidedness": "upper"},
        selection={"kind": "posterior_mean", "prior": [1, 1]},
    ),
]
# Thresholds equal to an exact posterior value: the rule stops when the value reaches the threshold
# (best >= threshold, loss < threshold), so the exact answer at equality is a stop for the first rule
# and a continue for the second.  The product integrates numerically; these cases pin down that a
# rounding error of the integral must not decide the comparison.
cases.append(case("threshold_equal_to_an_exact_probability_best", 2, FIXED("1/2", "1/2"), [1, 1, 1, 1],
                  ("fixed", ["1/3", "2/3"]), stopping={"kind": "best", "threshold": "7/10", "prior": [1, 1]},
                  selection=SEL_NONE, ties=True))
cases.append(case("threshold_equal_to_an_exact_futility_probability", 2, FIXED("1/2", "1/2"), [1, 1, 1, 1],
                  ("fixed", ["1/3", "2/3"]), stopping={"kind": "futility", "margin": "0", "threshold": "1/5", "prior": [1, 1]},
                  selection=SEL_NONE, ties=True))
bad = [(c["name"], c["smallest_threshold_gap"]) for c in cases if c["smallest_threshold_gap"] <= 1e-9 and not c["threshold_ties"]]
assert not bad, bad
print(json.dumps({"cases": cases}, indent=1))
