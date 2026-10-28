# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.procedure_weighting_simulation.

Regenerate with:  uv run --offline --script procedure_weighting_simulation_reference.py > procedure_weighting_simulation_reference.json

The sampling law of the adaptively weighted estimator at the terminal look of a declared procedure,
exactly, by path enumeration (procedure_exact.py) with the estimator evaluated at 60 digits from
its definition (Hadad et al. 2021 as restated in the weighting record): for the retained, revealed
units in enrolment order, with e_i(a) the probability the policy (or the delivered-arm mixture it
induces) gave each arm at the unit's look,

    m_i(a) = (strength * mean + sum of earlier outcomes attributed to a) / (strength + earlier count),
    Gamma_i(a) = m_i(a) + 1[A_i = a] (Y_i - m_i(a)) / e_i(a),
    h_i = 1  or  (sum over the estimand's arms of 1 / e_i(a))^(-1/2),
    estimate = sum h Gamma / sum h,   variance = sum h^2 (Gamma - estimate)^2 / (sum h)^2.

An estimate exists when at least two units are retained and revealed.  For each case the reference
gives the exact probabilities of the (look, rule stop) pairs, the exact distribution of the number
of analysed units, the expected enrolled units, and the exact conditional mean and standard
deviation of the estimate and of the plug-in variance given that an estimate exists.  The test
compares a seeded whole-procedure simulation with these at five Monte Carlo standard errors.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys
from decimal import Decimal, getcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

getcontext().prec = 60
F = pe.F


def dec(x):
    f = F(x)
    return Decimal(f.numerator) / Decimal(f.denominator)


def estimator(units, arms, estimand, scheme, strength, mean):
    """(estimate, variance) of the retained revealed units, or None."""
    if len(units) < 2:
        return None
    sums, counts = [Decimal(0)] * arms, [0] * arms
    gammas, weights = [], []
    for (_, attributed, success, _, _, eff) in units:
        y = Decimal(1 if success else 0)
        row = []
        for a in range(arms):
            m = (Decimal(strength) * dec(mean) + sums[a]) / (Decimal(strength) + counts[a])
            row.append(m + ((y - m) / dec(eff[a]) if a == attributed else Decimal(0)))
        sums[attributed] += y
        counts[attributed] += 1
        if estimand[0] == "arm":
            v = row[estimand[1]]
            arms_in = [estimand[1]]
        else:
            v = row[estimand[1]] - row[estimand[2]]
            arms_in = [estimand[1], estimand[2]]
        gammas.append(v)
        if scheme == "uniform":
            weights.append(Decimal(1))
        else:
            weights.append((sum(1 / dec(eff[a]) for a in arms_in)) ** Decimal("-0.5"))
    total = sum(weights)
    est = sum(h * g for h, g in zip(weights, gammas)) / total
    var = sum(h * h * (g - est) ** 2 for h, g in zip(weights, gammas)) / total**2
    return est, var


cases = []


def case(name, decl, truth, estimand, scheme, strength, mean, replications):
    arms = decl["arms"]
    enrol = decl["enrolment"]
    cumulative = [sum(enrol[: k + 1]) for k in range(len(enrol))]
    stats = {"defined": Decimal(0), "e1": Decimal(0), "e2": Decimal(0), "v1": Decimal(0), "v2": Decimal(0)}
    stop_mass = [F(0)] * len(enrol)
    horizon = F(0)
    analysed = {}
    enrolled = F(0)
    decided = {}

    def visit(look, stopped, decision, revealed, prob):
        nonlocal horizon, enrolled
        if stopped:
            stop_mass[look] += prob
        else:
            horizon += prob
        enrolled += prob * cumulative[look]
        analysed[len(revealed)] = analysed.get(len(revealed), F(0)) + prob
        key = "none" if decision is None else str(decision)
        decided[key] = decided.get(key, F(0)) + prob
        out = estimator(revealed, arms, estimand, scheme, strength, mean)
        if out is not None:
            p = dec(prob)
            est, var = out
            stats["defined"] += p
            stats["e1"] += p * est
            stats["e2"] += p * est * est
            stats["v1"] += p * var
            stats["v2"] += p * var * var

    pe.enumerate_paths(decl, ("fixed", truth), visit)
    d = stats["defined"]
    mean_est = stats["e1"] / d
    sd_est = (stats["e2"] / d - mean_est**2).sqrt()
    mean_var = stats["v1"] / d
    sd_var = (stats["v2"] / d - mean_var**2).sqrt()
    cases.append(
        {
            "name": name,
            "declaration": decl,
            "truth": truth,
            "estimand": estimand,
            "scheme": scheme,
            "outcome_prior": {"mean": mean, "strength": strength},
            "replications": replications,
            "stop_probability": float(sum(stop_mass, F(0))),
            "stop_by_look": [float(x) for x in stop_mass],
            "horizon_probability": float(horizon),
            "expected_enrolled": float(enrolled),
            "analysed_units": {str(k): float(v) for k, v in sorted(analysed.items())},
            "decided": {k: float(v) for k, v in sorted(decided.items())},
            "defined_probability": float(d),
            "estimate_mean": float(mean_est),
            "estimate_sd": float(sd_est),
            "variance_mean": float(mean_var),
            "variance_sd": float(sd_var),
        }
    )


FIXED2 = {"kind": "fixed", "probabilities": ["1/2", "1/2"]}


def decl(arms, assignment, enrol, stopping, selection=None, **extra):
    d = {"arms": arms, "assignment": assignment, "enrolment": enrol, "stopping": stopping,
         "selection": selection or {"kind": "mean"}}
    d.update(extra)
    return d


case("thompson_contrast_uniform_weights_with_a_stop",
     decl(2, {"kind": "thompson", "prior": [1, 1]}, [1] * 6, {"kind": "best", "threshold": "73/100", "prior": [1, 1]}),
     ["1/3", "2/3"], ["contrast", 1, 0], "uniform", 2, "1/2", 60000)
case("thompson_arm_mean_variance_stabilizing",
     decl(2, {"kind": "thompson", "prior": [1, 1]}, [1] * 6, {"kind": "none"}),
     ["1/4", "3/4"], ["arm", 1], "variance_stabilizing", 1, "1/2", 60000)
case("lagged_epsilon_greedy_with_random_loss",
     decl(2, {"kind": "epsilon_greedy", "epsilon": "1/2", "prior": [1, 1]}, [1] * 5, {"kind": "none"},
          lag=1, retain=["3/4", "3/4"]),
     ["2/5", "3/5"], ["contrast", 1, 0], "variance_stabilizing", 4, "1/2", 60000)
case("exposure_failure_delivered_attribution_batches",
     decl(2, {"kind": "epsilon_greedy", "epsilon": "1/2", "prior": [1, 1]}, [2, 2], {"kind": "none"},
          exposure={"failure": ["0", "1/4"], "fallback": 0, "attribution": "delivered"}),
     ["1/3", "2/3"], ["arm", 0], "uniform", 1, "1/3", 60000)
case("exposure_failure_assigned_attribution_with_cap",
     decl(2, FIXED2, [2, 2, 2], {"kind": "cap", "max_units": 4},
          exposure={"failure": ["1/5", "0"], "fallback": 1, "attribution": "assigned"}),
     ["1/3", "2/3"], ["contrast", 0, 1], "uniform", 2, "1/2", 60000)
print(json.dumps({"cases": cases}, indent=1))
