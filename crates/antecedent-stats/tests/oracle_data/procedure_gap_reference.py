# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.procedure_bayesian_martingale_gap.

Regenerate with:  uv run --offline --script procedure_gap_reference.py > procedure_gap_reference.json

For a declared procedure and independent Beta priors, the expectation of the terminal posterior
mean of each cell under the prior predictive (a Polya urn per delivered cell), expanded path by
path in exact rational arithmetic (procedure_exact.py), and its gap from the prior mean:

    gap_c = E[(alpha_c + s_c) / (alpha_c + beta_c + s_c + f_c)] - alpha_c / (alpha_c + beta_c)

where (s_c, f_c) are the observed counts of the cell at the stopping time.  The gap is exactly zero
when policy and stopping read observed history only and loss of an outcome does not depend on its
value; it is not zero under outcome-dependent loss, or when the cell counts are attributed to the
assigned arm while exposure fails.  `report` rows give the Beta posteriors, probability best and
expected loss of two arms at observed counts under non-uniform priors (exact binomial-tail values).

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

F = pe.F
pe.check_exceedance()

THOMPSON = {"kind": "thompson", "prior": [1, 1]}
NO_STOP = {"kind": "none"}
SEL_MEAN = {"kind": "mean"}

gaps = []


def gap_case(name, arms, assignment, enrol, priors, stopping=NO_STOP, selection=SEL_MEAN, **extra):
    decl = {
        "arms": arms,
        "assignment": assignment,
        "enrolment": enrol,
        "stopping": stopping,
        "selection": selection,
    }
    decl.update(extra)
    atoms = pe.enumerate_law(decl, ("predictive", priors))
    cells = []
    for c, (a, b) in enumerate(priors):
        expected = F(0)
        for (look, stopped, observed, dec), p in atoms.items():
            s, f = observed[2 * c], observed[2 * c + 1]
            expected += p * F(a + s, a + b + s + f)
        prior_mean = F(a, a + b)
        gap = expected - prior_mean
        cells.append(
            {
                "prior_mean": float(prior_mean),
                "expected_posterior_mean": float(expected),
                "gap": float(gap),
                "gap_exact": f"{gap.numerator}/{gap.denominator}",
            }
        )
    gaps.append({"name": name, "declaration": decl, "priors": priors, "cells": cells})


gap_case("thompson_probability_best_stop", 2, THOMPSON, [1] * 6, [[1, 1], [2, 3]],
         stopping={"kind": "best", "threshold": "73/100", "prior": [1, 1]})
gap_case("epsilon_greedy_lagged_administrative_loss", 2,
         {"kind": "epsilon_greedy", "epsilon": "1/5", "prior": [1, 1]}, [1, 1, 1, 1, 1],
         [[1, 1], [3, 2]], lag=1, retain=["3/4", "3/4"])
gap_case("futility_stop_with_posterior_mean_selection", 2,
         {"kind": "fixed", "probabilities": ["1/2", "1/2"]}, [2, 2, 2], [[2, 2], [1, 3]],
         stopping={"kind": "futility", "margin": "0", "threshold": "3/20", "prior": [1, 1]},
         selection={"kind": "posterior_mean", "prior": [1, 1]})
gap_case("ucb_three_arms_resource_cap", 3, {"kind": "ucb", "exploration": "2"}, [1] * 5,
         [[1, 1], [2, 1], [1, 2]], stopping={"kind": "cap", "max_units": 4})
gap_case("outcome_dependent_loss_breaks_the_martingale", 2,
         {"kind": "fixed", "probabilities": ["1/2", "1/2"]}, [1, 1, 1], [[1, 1], [1, 1]],
         retain=["1/2", "1"])
gap_case("loss_of_failures_with_adaptive_assignment", 2,
         {"kind": "epsilon_greedy", "epsilon": "1/5", "prior": [1, 1]}, [1, 1, 1, 1], [[1, 1], [2, 2]],
         retain=["1", "1/3"])
gap_case("exposure_failure_attributed_to_the_assigned_arm", 2,
         {"kind": "fixed", "probabilities": ["1/2", "1/2"]}, [2, 2], [[1, 1], [4, 1]],
         exposure={"failure": ["0", "1/2"], "fallback": 0, "attribution": "assigned"})
gap_case("exposure_failure_attributed_to_the_delivered_arm", 2,
         {"kind": "fixed", "probabilities": ["1/2", "1/2"]}, [2, 2], [[1, 1], [4, 1]],
         exposure={"failure": ["0", "1/2"], "fallback": 0, "attribution": "delivered"})

report = []
for prior in ([1, 1], [2, 3], [4, 1]):
    for counts in ([0, 0, 0, 0], [3, 1, 1, 3], [7, 2, 2, 8], [0, 5, 5, 0], [12, 12, 12, 12], [1, 0, 0, 1]):
        best = pe.best_probabilities(prior, counts)
        loss = pe.losses(prior, counts)
        report.append(
            {
                "prior": prior,
                "counts": counts,
                "posteriors": [list(pe.posterior(prior, counts, a)) for a in range(2)],
                "best": [float(x) for x in best],
                "loss": [float(x) for x in loss],
            }
        )
print(json.dumps({"gaps": gaps, "report": report}, indent=1))
