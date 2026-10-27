#!/usr/bin/env python3
"""Follow Antecedent's refusal when a comparison group does not overlap the treated.

Question: for the men who entered the National Supported Work (NSW) job-training programme,
how much did it change their real earnings in 1978 compared with what they would have earned
without it? This is the effect on the treated. The comparison group is drawn from the Panel
Study of Income Dynamics (PSID); see ``examples/data/nsw_psid.toml`` for origin, citation,
licence, checksums and preprocessing.

Most PSID respondents earned far more than anyone who entered the programme, so a propensity
model gives many of them essentially no chance of treatment. The script follows what Antecedent
reports at each step: the diagnostics of a default run, the ``positivity_violated`` refusal
when no overlap policy is declared, the common-support estimate once a trim is declared, and
how far the answer moves with that choice. On the real files it ends by comparing with the
randomised NSW benchmark, which estimates the same effect on the treated.

Data. By default this script runs on a **synthetic stand-in** with the same ten columns and
sizes and the same overlap problem, whose effect on the treated is a fixture truth of the
generator below. It uses the real files only when they are in the verified cache, or when
``ANTECEDENT_DATA=download`` lets ``examples/data/fetch.py`` download and verify them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import antecedent as ant
import numpy as np
from antecedent import population
from antecedent.estimators import Aipw, Overlap, PropensityWeighting

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
    treated rows: a fixture truth of this generator, not an empirical claim. As in the PSID
    comparison group, most comparison rows earn far more than any treated row, so overlap fails.
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
    re74 = np.maximum(0.0, 9000 + 9000 * capacity + 1000 * rng.normal(size=n))
    re75 = np.maximum(0.0, 0.8 * re74 + 1500 + 2500 * rng.normal(size=n))
    logit = -8.0 - 9.0 * capacity - 0.03 * (age - 34) + 0.6 * black - 0.3 * married
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


GRAPH = [(z, "treat") for z in COVARIATES] + [(z, "re78") for z in COVARIATES]
GRAPH.append(("treat", "re78"))
QUERY = ant.AverageEffect("treat", "re78", target_population=population.Treated())


def ask(data, estimator):
    return ant.analyze(
        data,
        graph=GRAPH,
        query=QUERY,
        estimator=estimator,
        refute="none",
        execution=ant.Execution(seed=11),
    )


def main() -> None:
    real = real_tables()
    if real is None:
        (data, truth), benchmark = standin_table(), None
        print(
            f"DATA: synthetic stand-in (fixture effect on the treated ${truth:,.0f})."
        )
    else:
        data, benchmark = real
        print(
            "DATA: NSW treated group with the PSID comparison group (checksums verified)."
        )
    n_treated = int(data["treat"].sum())
    assert n_treated == 185 and len(data["treat"]) == 2675

    # 1. A default run. Weighting clips propensities to [0.01, 0.99] by default.
    default = ant.outcomes.require_success(
        ask(data, PropensityWeighting(bootstrap=100))
    )
    est = default.estimate
    print(
        f"Default clip: effect on the treated {default.answer.value:,.0f}; "
        f"effective sample size {est.overlap_ess:.0f} of {len(data['treat'])} rows; "
        f"smallest raw propensity {est.overlap_propensity_min:.1e}"
    )
    assert est.overlap_ess < 0.2 * len(data["treat"])

    # 2. Ask without a clip: the scores cannot be used as they are, and Antecedent says why.
    refused = ask(data, PropensityWeighting(bootstrap=100, overlap=Overlap(clip=None)))
    refusal = refused.refusal
    assert refusal is not None and refusal.code == "positivity_violated"
    rows = refusal.fields["refusal_fields"]["boundary_count"]
    assert rows > 0
    print(f"Without a clip: {refusal.code}. {refusal.message.split(': ', 2)[-1]}")
    print(f"Remedy: {refusal.remedy}")

    # 3. Follow the remedy: declare a trim. The target becomes the treated rows with common support.
    trim = Overlap(clip=None, trim=0.01)
    trimmed = ant.outcomes.require_success(
        ask(data, PropensityWeighting(bootstrap=100, overlap=trim))
    )
    aipw = ant.outcomes.require_success(ask(data, Aipw(bootstrap=100, overlap=trim)))
    excluded = trimmed.estimate.overlap_excluded_fraction
    assert excluded is not None and 0.0 < excluded < 1.0
    se = trimmed.estimate.se_bootstrap or trimmed.estimate.se_analytic
    print(
        f"Trim at 0.01: {excluded:.0%} of rows excluded; weighting {trimmed.answer.value:,.0f} "
        f"(se {se:,.0f}), AIPW {aipw.answer.value:,.0f}, on the common-support treated rows."
    )

    # 4. How far the answer moves with the overlap choice is itself a finding.
    answers = [default.answer.value, trimmed.answer.value, aipw.answer.value]
    print(
        f"Spread across declared overlap choices and estimators: {max(answers) - min(answers):,.0f}"
    )
    assert all(np.isfinite(answers))

    if benchmark is None:
        print(
            f"Stand-in fixture effect over all treated rows: {truth:,.0f}. The trimmed estimate "
            "targets only the common-support treated rows, a different population."
        )
    else:
        exp, exp_se = mean_difference(benchmark)
        print(
            f"Randomised NSW benchmark for the same participants: {exp:,.0f} (se {exp_se:,.0f}). "
            "The experiment is the design the refusal's last remedy names: one that produces both "
            "treatments for people like the participants."
        )


if __name__ == "__main__":
    main()
