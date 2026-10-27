#!/usr/bin/env python3
"""Estimate the weight change that follows quitting smoking from observational data.

Question: among cigarette smokers examined in 1971-75 and followed up in 1982, how much more
(or less) weight, in kilograms, did the average person gain between visits if they quit smoking
than if they kept smoking? This is the average effect of quitting. The data are the NHEFS
smoking-cessation subset used in Hernan and Robins, "Causal Inference: What If"; see
``examples/data/nhefs.toml`` for origin, citation, licence, checksum and preprocessing.

Data. By default this script runs on a **synthetic stand-in** with the same columns and size
(1,629 rows, 63 of them without a 1982 weight), whose average effect of quitting is a fixture
truth of the generator below. It uses the real file only when it is in the verified cache, or
when ``ANTECEDENT_DATA=download`` lets ``examples/data/fetch.py`` download and verify it. The
output says which table was used.

Declared graph. Every baseline covariate may affect both quitting and weight change; quitting
may affect weight change. No other cause is declared: that is the assumption of no unmeasured
confounding, which nothing here tests. Rows without a 1982 weight are dropped by a declared
complete-case policy, which assumes the outcome is missing independently of the outcome
itself given the model; multiple imputation is run as a check on that choice.

What a real-data run can and cannot say. The estimates are conditional on the declared graph
and the missing-data policy. Hernan and Robins report a similar analysis, but theirs also
weights for loss to follow-up, so it is not the same computation; neither number is causal truth.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import antecedent as ant
import numpy as np
from antecedent import population
from antecedent import validation as v
from antecedent.estimators import Aipw, PropensityWeighting

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import fetch  # noqa: E402  (examples/data/fetch.py; stdlib only)

DATASET = "nhefs"
N_ROWS, N_MISSING = 1629, 63
RAW = [
    "qsmk",
    "wt82_71",
    "sex",
    "race",
    "age",
    "education",
    "smokeintensity",
    "smokeyrs",
    "exercise",
    "active",
    "wt71",
]
SQUARED = ["age", "smokeintensity", "smokeyrs", "wt71"]
LEVELS = {"education": [2, 3, 4, 5], "exercise": [1, 2], "active": [1, 2]}
FIXTURE_EFFECT = 3.0


def design(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The book-style covariate set: squares of four continuous covariates, level indicators."""
    table = {k: raw[k] for k in ("qsmk", "wt82_71", "sex", "race", *SQUARED)}
    for name in SQUARED:
        table[f"{name}_sq"] = raw[name] ** 2
    for name, levels in LEVELS.items():
        for level in levels:
            table[f"{name}_{level}"] = (raw[name] == level).astype(float)
    return table


def standin_raw(seed: int = 3) -> dict[str, np.ndarray]:
    """1,629 smokers; quitting depends on the covariates; every quitter gains FIXTURE_EFFECT kg.

    The effect is constant, so the average effect of quitting is FIXTURE_EFFECT: a fixture
    truth of this generator, not an empirical claim. 63 outcomes are missing completely at
    random, a simpler mechanism than the real follow-up, so the stand-in cannot say whether
    complete-case analysis is adequate on the real data.
    """
    rng = np.random.default_rng(seed)
    n = N_ROWS
    sex = (rng.uniform(size=n) < 0.51).astype(float)
    race = (rng.uniform(size=n) < 0.13).astype(float)
    age = np.clip(np.round(43 + 12 * rng.normal(size=n)), 25, 74)
    education = np.clip(np.round(2.7 + 1.2 * rng.normal(size=n)), 1, 5)
    smokeintensity = np.clip(np.round(20 + 11 * rng.normal(size=n)), 1, 80)
    smokeyrs = np.clip(np.round(0.45 * (age - 18) + 6 * rng.normal(size=n)), 1, 60)
    exercise = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.4, 0.4, 0.2])
    active = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.45, 0.45, 0.1])
    wt71 = np.clip(70 + 15 * rng.normal(size=n) - 7 * sex, 40, 150)
    logit = (
        -1.1
        + 0.025 * (age - 43)
        - 0.02 * (smokeintensity - 20)
        + 0.12 * (education - 3)
        - 0.2 * sex
        - 0.3 * race
        - 0.01 * (wt71 - 70)
    )
    qsmk = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    wt82_71 = (
        FIXTURE_EFFECT * qsmk
        + 2.5
        - 0.12 * (age - 43)
        + 0.03 * (smokeintensity - 20)
        - 0.1 * (wt71 - 70)
        + 0.8 * sex
        - 1.0 * (exercise == 2)
        + 7.5 * rng.normal(size=n)
    )
    wt82_71[rng.choice(n, size=N_MISSING, replace=False)] = np.nan
    return {
        "qsmk": qsmk, "wt82_71": wt82_71, "sex": sex, "race": race, "age": age,
        "education": education, "smokeintensity": smokeintensity, "smokeyrs": smokeyrs,
        "exercise": exercise, "active": active, "wt71": wt71,
    }  # fmt: skip


def real_raw() -> dict[str, np.ndarray] | None:
    """The eleven used columns from the verified file, NaN where a value is empty; else None."""
    try:
        if os.environ.get("ANTECEDENT_DATA") == "download":
            paths = fetch.fetch(DATASET)
        else:
            paths = fetch.verified_paths(DATASET)
    except (fetch.DatasetUnavailable, OSError) as problem:
        print(f"Real file unavailable ({problem}); using the synthetic stand-in.")
        return None
    with paths["nhefs.csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == N_ROWS, len(rows)
    return {
        k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in RAW
    }


def main() -> None:
    raw = real_raw()
    real = raw is not None
    if raw is None:
        raw = standin_raw()
        print(
            f"DATA: synthetic stand-in (fixture average effect of quitting {FIXTURE_EFFECT} kg)."
        )
    else:
        print("DATA: NHEFS smoking-cessation subset (checksum verified).")
    data = design(raw)
    qsmk, outcome = data["qsmk"], data["wt82_71"]
    assert len(qsmk) == N_ROWS and int(np.isnan(outcome).sum()) == N_MISSING
    observed = ~np.isnan(outcome)
    unadjusted = (
        outcome[observed & (qsmk == 1)].mean() - outcome[observed & (qsmk == 0)].mean()
    )
    print(f"Unadjusted difference (quitters minus others): {unadjusted:.2f} kg")

    # Declare the graph: every covariate may affect quitting and weight change.
    covariates = [k for k in data if k not in ("qsmk", "wt82_71")]
    graph = [(z, "qsmk") for z in covariates] + [(z, "wt82_71") for z in covariates]
    graph.append(("qsmk", "wt82_71"))
    query = ant.AverageEffect("qsmk", "wt82_71")

    # Prepare and inspect before estimating. Rows without a 1982 weight are dropped by a
    # declared policy, and the result carries the policy's assumption.
    study = ant.prepare(
        data,
        graph=graph,
        query=query,
        estimator=PropensityWeighting(bootstrap=200),
        missing="complete_case",
        inference=ant.Frequentist(bootstrap=200),
        execution=ant.Execution(seed=7),
    )
    print(study.inspect())
    result = ant.outcomes.require_success(ant.estimate(study))
    est = result.estimate
    assert result.identification.status == "NonparametricallyIdentified"
    assert set(result.identification.adjustment_set) == set(covariates)

    missing = result.missing_data
    print(
        f"Missing data ({missing.policy}): {missing.accounting.rows_kept} of "
        f"{missing.accounting.rows_total} rows kept, {missing.accounting.rows_dropped} dropped."
    )
    print(f"Assumption: {missing.assumption}")
    assert missing.accounting.rows_total == N_ROWS
    assert missing.accounting.rows_dropped == N_MISSING
    if real:
        assert missing.accounting.rows_kept == 1566

    se = est.se_bootstrap or est.se_analytic
    assert se is not None and se > 0.0
    effect = result.answer.value
    lo, hi = effect - 2.576 * se, effect + 2.576 * se
    print(
        f"Propensity weighting, average effect of quitting: {effect:.2f} kg "
        f"(se {se:.2f}, 99% Wald interval [{lo:.2f}, {hi:.2f}])"
    )
    print(
        f"Overlap: smallest propensity {est.overlap_propensity_min:.3f}; effective sample "
        f"size {est.overlap_ess:.0f} of {missing.accounting.rows_kept} rows."
    )
    print("Calibration:", result.calibration.status)
    assert (
        est.overlap_propensity_min > 0.01
    )  # good overlap, unlike the NSW/PSID comparison
    assert np.isfinite([effect, lo, hi]).all() and lo < hi

    # Continued use: the effect among the people who quit, a different estimand.
    treated_query = ant.AverageEffect(
        "qsmk", "wt82_71", target_population=population.Treated()
    )
    among_quitters = ant.outcomes.require_success(
        ant.estimate(ant.reconfigure(study, query=treated_query))
    )
    print(f"Effect among those who quit: {among_quitters.answer.value:.2f} kg")

    # Cross-check with other estimators on the same declaration.
    aipw = ant.outcomes.require_success(
        ant.estimate(ant.reconfigure(study, estimator=Aipw(bootstrap=200)))
    )
    regression = ant.outcomes.require_success(
        ant.estimate(ant.reconfigure(study, estimator="linear.adjustment.ate"))
    )
    print(
        f"Cross-checks: AIPW {aipw.answer.value:.2f} kg, "
        f"outcome regression {regression.answer.value:.2f} kg"
    )

    # Check the complete-case choice with multiple imputation of the dropped outcomes.
    imputed = ant.outcomes.require_success(
        ant.estimate(ant.reconfigure(study, missing="multiple_imputation"))
    )
    print(
        f"Multiple imputation of the {N_MISSING} missing outcomes: {imputed.answer.value:.2f} kg"
    )
    estimates = [
        among_quitters.answer.value,
        aipw.answer.value,
        regression.answer.value,
        imputed.answer.value,
    ]
    assert np.isfinite(estimates).all()

    # Sensitivity: how strong must an unmeasured confounder be to explain the estimate away?
    # The E-value reads the effect on a standardized scale (effect / outcome standard deviation).
    sd = float(np.nanstd(outcome[qsmk == 0], ddof=1))
    sens = v.evalue(
        effect / sd, standard_error=se / sd, scale="standardized_mean_difference"
    )
    assert sens.range is not None and sens.range.break_even > 1.0
    print(
        f"E-value {sens.range.break_even:.2f}: an unmeasured confounder would need at least "
        "this risk-ratio association with both quitting and weight change to explain the "
        f"estimate away, under assumption '{sens.range.assumption}'."
    )

    if not real:
        # The stand-in has a stated truth, so recovery can be asserted. The real data have none.
        assert lo <= FIXTURE_EFFECT <= hi, (lo, hi, FIXTURE_EFFECT)
        print(
            f"Stand-in: the 99% interval contains the fixture effect, {FIXTURE_EFFECT} kg."
        )
    else:
        print(
            "Hernan and Robins report about 3.4 kg (95% CI about 2.4 to 4.5) by inverse-probability "
            "weighting. Their analysis also weights for loss to follow-up, so it is close to, "
            "but not the same as, this one."
        )


if __name__ == "__main__":
    main()
