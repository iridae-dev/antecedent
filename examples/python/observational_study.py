#!/usr/bin/env python3
"""Estimate a training programme's effect on earnings from observational data.

Question: for the men who entered a job-training programme, how much did it change their real
earnings in 1978 (the outcome) compared with what they would have earned without it? This is the
effect on the treated, the same estimand the NSW experiment estimates. The data are the National Supported Work
(NSW) demonstration with a Panel Study of Income Dynamics (PSID) comparison group; see
``examples/data/nsw_psid.toml`` for origin, citation, licence, checksums and preprocessing.

Data. By default this script runs on a **synthetic stand-in** with the same ten columns and
sizes, whose effect on the treated is a fixture truth of the generator below. It uses the real files
only when they are already in the verified cache, or when ``ANTECEDENT_DATA=download`` lets
``examples/data/fetch.py`` download and verify them. The output says which table was used. The
real files are never fetched on import, and the offline path needs no network.

Declared graph. Age, schooling, race, ethnicity, marital status, degree status and earnings in
1974 and 1975 may each affect both programme entry and 1978 earnings; the programme may affect
1978 earnings. No other cause is declared: that is the assumption of no unmeasured confounding.

What a real-data run can and cannot say. The NSW experiment gives a randomised estimate of the
programme's effect for its participants, the effect on the treated targeted here. Reproducing it from observational comparison data is
evidence about whether the declared adjustment works here; it does not make the observational
estimate causal truth, and a miss does not locate which assumption failed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import antecedent as _ant_api
import numpy as np
from antecedent import AverageEffect, analyze
from antecedent import population
from antecedent import validation as v
from antecedent.outcomes import require_success

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import fetch  # noqa: E402  (examples/data/fetch.py; stdlib only)

DATASET = "nsw_psid"
COLUMNS = [
    "treat",
    "age",
    "educ",
    "black",
    "hisp",
    "married",
    "nodegree",
    "re74",
    "re75",
    "re78",
]
COVARIATES = COLUMNS[1:-1]


def standin_table(seed: int = 5) -> tuple[dict[str, np.ndarray], float]:
    """185 treated and 2490 comparison rows with NSW-like columns, and the true effect on the treated.

    Entry into the programme depends on age, earnings capacity and race; earnings in 1978
    depend on the same covariates plus the programme. The effect varies with schooling,
    1000 + 100 * (11 - educ) dollars, so the effect on the treated is its mean over the 185
    treated rows: a fixture truth of this generator, not an empirical claim. Overlap is deliberately imperfect: treated rows sit in the lower-earnings tail.
    """
    rng = np.random.default_rng(seed)
    n = 24000
    capacity = rng.normal(size=n)
    age = np.clip(np.round(34 + 9 * rng.normal(size=n)), 18, 55)
    educ = np.clip(np.round(11 + 2 * rng.normal(size=n) + 0.5 * capacity), 3, 16)
    black = (rng.uniform(size=n) < 0.25).astype(float)
    hisp = ((rng.uniform(size=n) < 0.08) & (black == 0)).astype(float)
    married = (rng.uniform(size=n) < 0.25 + 0.4 * (age > 30)).astype(float)
    nodegree = (educ < 12).astype(float)
    re74 = np.maximum(0.0, 9000 + 5000 * capacity + 2500 * rng.normal(size=n))
    re75 = np.maximum(0.0, 0.8 * re74 + 1500 + 2500 * rng.normal(size=n))
    logit = -2.6 - 0.9 * capacity - 0.03 * (age - 34) + 0.6 * black - 0.3 * married
    entry = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))
    treated = np.flatnonzero(entry)[:185]
    comparison = np.flatnonzero(~entry)[:2490]
    rows = np.concatenate([treated, comparison])
    treat = np.concatenate([np.ones(len(treated)), np.zeros(len(comparison))])
    noise = 2500 * rng.normal(size=rows.size)
    effect = 1000 + 100 * (11 - educ[rows])
    re78 = (
        0.5 * re75[rows]
        + 0.3 * re74[rows]
        + 200 * (educ[rows] - 11)
        + effect * treat
        + 1500
        + noise
    )
    columns = {
        "age": age, "educ": educ, "black": black, "hisp": hisp, "married": married,
        "nodegree": nodegree, "re74": re74, "re75": re75,
    }  # fmt: skip
    table = {"treat": treat, **{k: x[rows].astype(float) for k, x in columns.items()}}
    table["re78"] = np.maximum(0.0, re78)
    return table, float(effect[treat == 1].mean())


def real_tables() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """(observational table, experimental table) from verified files, else None."""
    try:
        if os.environ.get("ANTECEDENT_DATA") == "download":
            paths = fetch.fetch(DATASET)
        else:
            paths = fetch.verified_paths(DATASET)
    except (fetch.DatasetUnavailable, OSError) as problem:
        print(f"Real files unavailable ({problem}); using the synthetic stand-in.")
        return None

    def load(name: str) -> np.ndarray:
        rows = np.loadtxt(paths[name], ndmin=2)
        assert rows.shape[1] == len(COLUMNS), (name, rows.shape)
        return rows

    def table(rows: np.ndarray) -> dict[str, np.ndarray]:
        return {name: rows[:, i].astype(float) for i, name in enumerate(COLUMNS)}

    treated = load("nswre74_treated.txt")
    control = load("nswre74_control.txt")
    psid = load("psid_controls.txt")
    return table(np.vstack([treated, psid])), table(np.vstack([treated, control]))


def mean_difference(table: dict[str, np.ndarray]) -> tuple[float, float]:
    """Difference in mean 1978 earnings between treated and untreated rows, with its standard error."""
    t, y = table["treat"] == 1, table["re78"]
    diff = y[t].mean() - y[~t].mean()
    return float(diff), float(
        np.sqrt(y[t].var(ddof=1) / t.sum() + y[~t].var(ddof=1) / (~t).sum())
    )


def main() -> None:
    real = real_tables()
    if real is None:
        (data, truth), benchmark = standin_table(), None
        print(
            f"DATA: synthetic stand-in (fixture effect on the treated ${truth:,.0f}); "
            "not the NSW or PSID records."
        )
    else:
        data, benchmark = real
        print(
            "DATA: NSW treated group with the PSID comparison group (checksums verified)."
        )
    n_treated = int(data["treat"].sum())
    print(f"rows={len(data['treat'])} treated={n_treated}")
    assert n_treated == 185 and len(data["treat"]) == 2675

    graph = [(z, "treat") for z in COVARIATES] + [(z, "re78") for z in COVARIATES]
    graph.append(("treat", "re78"))
    query = AverageEffect(
        treatment="treat", outcome="re78", target_population=population.Treated()
    )

    raw, raw_se = mean_difference(data)
    print(
        f"Unadjusted treated-minus-comparison difference: {raw:,.0f} (se {raw_se:,.0f})"
    )

    # Prepare and inspect before estimating: identification and overlap are readable first.
    study = _ant_api.prepare(
        data,
        graph=graph,
        query=query,
        estimator="propensity.weighting",
        refute="cheap",
        inference=_ant_api.Frequentist(bootstrap=100),
        execution=_ant_api.Execution(seed=11),
    )
    print(study.inspect())
    result = require_success(_ant_api.estimate(study))
    assert result.answer.kind == "point"
    assert result.identification.status == "NonparametricallyIdentified"
    assert set(result.identification.adjustment_set) == set(COVARIATES)
    assert result.estimate.overlap_ess is not None  # the overlap diagnostic is reported

    se = result.estimate.se_bootstrap or result.estimate.se_analytic
    assert se is not None and se > 0.0
    lo, hi = result.answer.value - 2.576 * se, result.answer.value + 2.576 * se
    print(
        f"Weighted effect on the treated: {result.answer.value:,.0f}, 99% Wald interval [{lo:,.0f}, {hi:,.0f}]; "
        f"overlap_ess={result.estimate.overlap_ess:.1f} of {len(data['treat'])} rows, "
        f"smallest estimated propensity={result.estimate.overlap_propensity_min}"
    )
    print("Calibration:", result.calibration.status)
    assert np.isfinite([lo, hi]).all() and lo < hi
    print("Refuters:")
    assert result.validation.ran and result.validation.count >= 1
    for report in result.validation.reports:
        print(
            f"  {report.refuter}: passed={report.passed} comparison={report.comparison:.4g}"
        )

    # Doubly robust cross-check from the same declaration (full-sample nuisances, no cross-fitting).
    aipw = require_success(
        analyze(
            data,
            graph=graph,
            query=query,
            estimator="aipw",
            refute="none",
            inference=_ant_api.Frequentist(bootstrap=100),
            execution=_ant_api.Execution(seed=11),
        )
    )
    assert aipw.answer.kind == "point" and aipw.estimate.estimator_id == "aipw"
    print(f"AIPW effect on the treated: {aipw.answer.value:,.0f}")

    # Sensitivity: how strong must an unmeasured confounder be to explain the whole effect away?
    # The E-value reads the effect on a standardized scale (effect / outcome standard deviation).
    sd = float(np.std(data["re78"][data["treat"] == 0], ddof=1))
    sens = v.evalue(
        result.answer.value / sd,
        standard_error=se / sd,
        scale="standardized_mean_difference",
    )
    assert sens.range is not None and sens.range.break_even > 1.0
    print(
        f"E-value {sens.range.break_even:.2f}: an unmeasured confounder would need this risk-ratio "
        "association with both programme entry and earnings to explain the estimate away, "
        f"under assumption '{sens.range.assumption}'. Calibration: {sens.range.calibration}."
    )

    if benchmark is None:
        # The stand-in has a stated truth, so recovery can be asserted. The real data have none.
        assert lo <= truth <= hi, (lo, hi, truth)
        print(
            f"Stand-in: the interval contains the fixture effect on the treated, {truth:,.0f}."
        )
    else:
        exp, exp_se = mean_difference(benchmark)
        print(
            f"Experimental benchmark (randomised NSW treated vs control): {exp:,.0f} (se {exp_se:,.0f}). "
            "Both numbers target the effect on the participants. Closeness shows the declared "
            "adjustment worked with this comparison group; it does not make either number "
            "the causal truth, and another comparison group or method can differ."
        )
        assert np.isfinite(exp) and exp_se > 0.0


if __name__ == "__main__":
    main()
