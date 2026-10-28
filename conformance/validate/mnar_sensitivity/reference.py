"""Independent reference for the missing-not-at-random sensitivity analysis.

Pure standard-library Python; it imports no Antecedent code and shares no formula text with the
Rust implementation. Two departure models of the outcome-missingness mechanism are evaluated
for a two-arm mean contrast.

Pattern-mixture shift. Missing outcomes of an arm are the observed outcomes shifted by
``loading * delta``: ``E[Y | R=0, arm] = E[Y | R=1, arm] + loading * delta``.

Selection model (exponential tilting). The probability of observing an outcome follows
``logit P(R=1 | y, arm) = alpha - t * y`` with ``t = loading * theta``. Bayes' rule on the
selection mechanism, written out explicitly here (``f(y) ~ f_obs(y) / pi(y)`` and
``f(y | R=0) ~ f(y) (1 - pi(y))``), gives the missing-pattern law; ``alpha`` is arbitrary and
drops out, which the script asserts by evaluating two values of it.

The arm's full-population mean is ``pi_hat * ybar + (1 - pi_hat) * E[Y | R=0]`` with ``pi_hat``
the observed fraction. The sampling variance conditional on the parameter is the delta-method
variance computed here from a numerical influence function (a central finite difference of the
arm mean under a perturbed weight of each observed outcome), not from the closed-form influence
the implementation uses.

Cases:

* ``shift_closed_form`` -- treated outcomes 1..6 observed and 4 missing, control outcomes 0..4
  observed and 5 missing; effect(delta) = 1.5 + delta * (l1 * 0.4 - l0 * 0.5); the common-loading
  sign tipping point is delta = 15 and the treated-only decision "effect > 1" tips at -1.25.
* ``tilt_binary_closed_form`` -- binary treated outcomes (three ones, three zeros, four
  missing) against a fully observed control with mean 0.25; the missing-pattern mean is
  expit(theta) and the decision "effect > 0.3" tips at theta = ln(5/3).
* ``tilt_continuous`` -- a seeded continuous sample with opposite-direction loadings, whose
  tipping points are found by a fine scan plus bisection on this script's own mean function.

``--write`` regenerates ``expected.json``; ``--check`` compares it (``math.isclose``).

    python3 conformance/validate/mnar_sensitivity/reference.py --check
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"
Z975 = 1.959963984540054


def tilted_missing_mean(obs: list[float], t: float, alpha: float = 0.0) -> float:
    """E[Y | R=0] from the logistic selection model, by explicit Bayes on the mechanism."""
    n = len(obs)

    def pi(y: float) -> float:
        return 1.0 / (1.0 + math.exp(-(alpha - t * y)))

    full = [1.0 / (n * pi(y)) for y in obs]  # f(y) ~ f_obs(y) / pi(y)
    missing = [f * (1.0 - pi(y)) for f, y in zip(full, obs)]  # f(y, R=0)
    total = sum(missing)
    return sum(m * y for m, y in zip(missing, obs)) / total


def tilted_missing_mean_weighted(
    obs: list[float], weights: list[float], t: float, shift: float
) -> float:
    """Missing-pattern mean with observation weights (for the numerical influence)."""
    centre = max(obs) if t >= 0 else min(obs)
    w = [s * math.exp(t * (y - centre)) for s, y in zip(weights, obs)]
    return sum(a * y for a, y in zip(w, obs)) / sum(w) + shift


def arm_mean_weighted(
    obs: list[float], missing: int, model: str, param: float, weights: list[float]
) -> float:
    n_obs = len(obs)
    pi_hat = n_obs / (n_obs + missing)
    ybar = sum(s * y for s, y in zip(weights, obs)) / sum(weights)
    if model == "delta":
        g = ybar + param
    else:
        g = tilted_missing_mean_weighted(obs, weights, param, 0.0)
    return pi_hat * ybar + (1.0 - pi_hat) * g


def arm_moments(obs: list[float], missing: int, model: str, param: float) -> tuple[float, float]:
    """Mean and delta-method variance of one arm's full-population mean."""
    n_obs, n = len(obs), len(obs) + missing
    pi_hat = n_obs / n
    ones = [1.0] * n_obs
    mean = arm_mean_weighted(obs, missing, model, param, ones)
    if missing == 0:
        ybar = sum(obs) / n_obs
        return mean, sum((y - ybar) ** 2 for y in obs) / (n_obs - 1) / n_obs
    h = 1e-5
    influence = []
    for i in range(n_obs):
        up, down = ones.copy(), ones.copy()
        up[i] += h
        down[i] -= h
        derivative = (
            arm_mean_weighted(obs, missing, model, param, up)
            - arm_mean_weighted(obs, missing, model, param, down)
        ) / (2 * h)
        influence.append(n_obs * derivative)
    centre = sum(influence) / n_obs
    var_obs = sum((z - centre) ** 2 for z in influence) / (n_obs - 1) / n_obs
    ybar = sum(obs) / n_obs
    g = (mean - pi_hat * ybar) / (1.0 - pi_hat)
    return mean, (ybar - g) ** 2 * pi_hat * (1.0 - pi_hat) / n + var_obs


def split(data: dict) -> tuple[tuple[list[float], int], tuple[list[float], int]]:
    arms = {1.0: ([], 0), 0.0: ([], 0)}
    for t, y in zip(data["treatment"], data["outcome"]):
        obs, miss = arms[t]
        if y is None:
            arms[t] = (obs, miss + 1)
        else:
            obs.append(y)
    return arms[1.0], arms[0.0]


def effect_at(case: dict, param: float, with_variance: bool = True) -> tuple[float, float]:
    (t_obs, t_miss), (c_obs, c_miss) = split(case["data"])
    lt, lc = case["loading"]
    if not with_variance:
        ones1, ones0 = [1.0] * len(t_obs), [1.0] * len(c_obs)
        mu1 = arm_mean_weighted(t_obs, t_miss, case["model"], lt * param, ones1)
        mu0 = arm_mean_weighted(c_obs, c_miss, case["model"], lc * param, ones0)
        return mu1 - mu0, 0.0
    mu1, v1 = arm_moments(t_obs, t_miss, case["model"], lt * param)
    mu0, v0 = arm_moments(c_obs, c_miss, case["model"], lc * param)
    return mu1 - mu0, math.sqrt(v1 + v0)


def first_change(
    case: dict, label, direction: int, bound: float, with_variance: bool = True
) -> float | None:
    """Smallest |parameter| past MAR in `direction` at which `label` differs from its MAR value."""
    at_mar = label(*effect_at(case, 0.0, with_variance))
    steps = 400
    previous = 0.0
    for k in range(1, steps + 1):
        p = direction * bound * k / steps
        if label(*effect_at(case, p, with_variance)) != at_mar:
            lo, hi = previous, p
            for _ in range(80):
                mid = (lo + hi) / 2
                if label(*effect_at(case, mid, with_variance)) == at_mar:
                    lo = mid
                else:
                    hi = mid
            return hi
        previous = p
    return None


def sign_label(effect: float, se: float) -> int:
    return (effect > 0) - (effect < 0)


def null_label(effect: float, se: float) -> int:
    lower, upper = effect - Z975 * se, effect + Z975 * se
    return 1 if lower > 0 else (-1 if upper < 0 else 0)


def decision_label(threshold: float):
    return lambda effect, se: int(effect > threshold)


def build_case(case: dict) -> dict:
    out = {k: v for k, v in case.items() if k not in ("probe", "decisions")}
    out["probe"] = []
    for p in case["probe"]:
        effect, se = effect_at(case, p)
        out["probe"].append({"parameter": p, "effect": effect, "standard_error": se})
    out["tipping"] = {}
    labels = {"sign": sign_label, "excludes_null": null_label}
    out["decisions"] = case["decisions"]
    for name, threshold in case["decisions"].items():
        labels[name] = decision_label(threshold)
    for name, label in labels.items():
        needs_se = name == "excludes_null"
        negative = first_change(case, label, -1, abs(case["lower"]), needs_se)
        positive = first_change(case, label, 1, case["upper"], needs_se)
        out["tipping"][name] = {"negative": negative, "positive": positive}
    return out


def shift_data() -> dict:
    treatment = [1.0] * 10 + [0.0] * 10
    outcome = [float(v) for v in range(1, 7)] + [None] * 4
    outcome += [float(v) for v in range(0, 5)] + [None] * 5
    return {"treatment": treatment, "outcome": outcome}


def tilt_binary_data() -> dict:
    treatment = [1.0] * 10 + [0.0] * 8
    outcome = [1.0] * 3 + [0.0] * 3 + [None] * 4 + [1.0] * 2 + [0.0] * 6
    return {"treatment": treatment, "outcome": outcome}


def continuous_data() -> dict:
    rng = random.Random(20240611)
    treatment, outcome = [], []
    for arm, mean, miss_rate in ((1.0, 1.0, 0.3), (0.0, 0.0, 0.25)):
        for _ in range(60):
            y = round(rng.gauss(mean, 1.0), 6)
            treatment.append(arm)
            outcome.append(None if rng.random() < miss_rate else y)
    return {"treatment": treatment, "outcome": outcome}


def cases() -> list[dict]:
    return [
        {
            "name": "shift_closed_form",
            "grid_points": 81,
            "model": "delta",
            "loading": [1.0, 1.0],
            "lower": -20.0,
            "upper": 20.0,
            "data": shift_data(),
            "probe": [-20.0, -5.0, 0.0, 2.0, 7.5, 20.0],
            "decisions": {"treat": 1.0},
        },
        {
            "name": "shift_treated_only",
            "grid_points": 81,
            "model": "delta",
            "loading": [1.0, 0.0],
            "lower": -10.0,
            "upper": 10.0,
            "data": shift_data(),
            "probe": [-10.0, -3.75, 0.0, 4.0, 10.0],
            "decisions": {"treat": 1.0},
        },
        {
            "name": "tilt_binary_closed_form",
            "grid_points": 81,
            "model": "tilt",
            "loading": [1.0, 0.0],
            "lower": -5.0,
            "upper": 5.0,
            "data": tilt_binary_data(),
            "probe": [-5.0, -1.0, 0.0, 0.5, 2.0, 5.0],
            "decisions": {"treat": 0.3},
        },
        {
            "name": "tilt_continuous",
            "grid_points": 61,
            "model": "tilt",
            "loading": [1.0, -1.0],
            "lower": -3.0,
            "upper": 3.0,
            "data": continuous_data(),
            "probe": [-3.0, -1.0, 0.0, 0.7, 3.0],
            "decisions": {"treat": 1.2},
        },
    ]


def analytic_checks() -> dict:
    """Closed forms asserted against the numeric evaluation."""
    shift = {c["name"]: c for c in cases()}
    common = shift["shift_closed_form"]
    for p in common["probe"]:
        assert math.isclose(effect_at(common, p)[0], 1.5 - 0.1 * p, abs_tol=1e-12)
    treated = shift["shift_treated_only"]
    for p in treated["probe"]:
        assert math.isclose(effect_at(treated, p)[0], 1.5 + 0.4 * p, abs_tol=1e-12)
    binary = shift["tilt_binary_closed_form"]
    for p in binary["probe"]:
        expit = 1.0 / (1.0 + math.exp(-p))
        assert math.isclose(effect_at(binary, p)[0], 0.05 + 0.4 * expit, abs_tol=1e-12)
    # alpha drops out of the selection-model missing-pattern mean
    obs = [0.2, 1.1, 0.7, 1.9, -0.4]
    assert math.isclose(
        tilted_missing_mean(obs, 0.8, alpha=0.0), tilted_missing_mean(obs, 0.8, alpha=1.7), abs_tol=1e-12
    )
    return {
        "shift_common_sign_tipping": 15.0,
        "shift_treated_only_sign_tipping": -3.75,
        "shift_treated_only_decision_tipping": -1.25,
        "tilt_binary_decision_tipping": math.log(5.0 / 3.0),
    }


def document() -> dict:
    closed = analytic_checks()
    built = [build_case(c) for c in cases()]
    by_name = {c["name"]: c for c in built}
    assert math.isclose(by_name["shift_closed_form"]["tipping"]["sign"]["positive"], 15.0, abs_tol=1e-8)
    assert math.isclose(by_name["shift_treated_only"]["tipping"]["sign"]["negative"], -3.75, abs_tol=1e-8)
    assert math.isclose(by_name["shift_treated_only"]["tipping"]["treat"]["negative"], -1.25, abs_tol=1e-8)
    assert math.isclose(
        by_name["tilt_binary_closed_form"]["tipping"]["treat"]["positive"], math.log(5 / 3), abs_tol=1e-8
    )
    return {
        "schema_version": 1,
        "tolerance_class": "StableFloat",
        "oracle": {
            "kind": "closed_form",
            "method": "closed forms for the shift and the binary tilt plus an independent evaluation of the logistic selection model by explicit Bayes on the selection mechanism, with a numerical-influence delta-method variance",
            "independence_basis": "reference.py is standard-library Python that imports no Antecedent code. The shift and the binary tilt are checked against hand-derived closed forms (effect, tipping points); the continuous tilt is evaluated through the selection mechanism f(y|R=0) ~ f_obs(y)(1-pi(y))/pi(y) with an arbitrary intercept that drops out, and its tipping points come from this script's own scan plus bisection. The author of the reference is the repository author and independence from the production implementation has not been reviewed.",
            "generator": "conformance/validate/mnar_sensitivity/reference.py",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "level": 0.95,
        "closed_forms": closed,
        "cases": built,
    }


def close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
    return a == b


def main(argv: list[str]) -> int:
    doc = document()
    if "--write" in argv:
        EXPECTED.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {EXPECTED}")
        return 0
    stored = json.loads(EXPECTED.read_text(encoding="utf-8"))
    if not close(json.loads(json.dumps(doc)), stored):
        print("expected.json differs from the reference; rerun with --write", file=sys.stderr)
        return 1
    print("mnar sensitivity reference check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
