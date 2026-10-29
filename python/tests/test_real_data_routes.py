"""Every propensity and adjustment route keeps the host contract on observational study data.

The NSW/PSID comparison has severe overlap failure; NHEFS has good overlap. Bugs in how a route
reports a positivity problem surfaced only on such data, so each route is asked the same
question under each overlap declaration and target, and every answer is checked against the
contract rather than a number:

- a refusal carries a registered runtime refusal code, never an exception class name;
- an answer carrying a ``.limited`` diagnostic (overlap policy bound, dropped bootstrap
  replicates, weak sampler diagnostics) is never ``supported``;
- an answer's host status is one of the registered statuses.

The checks run on the examples' synthetic stand-ins, which reproduce each dataset's overlap
structure, so they always execute. When the verified real files are in the data cache
(``ANTECEDENT_DATA_DIR``; see ``examples/data/README.md``) they run on the real data too.
"""

from __future__ import annotations

import runpy
import tomllib
from collections.abc import Callable
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import host, population
from antecedent.estimators import (
    DML,
    Aipw,
    DistanceMatching,
    DRLearner,
    LinearAdjustment,
    Overlap,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
)

from _repo_text import REPO_ROOT, read_text

_REGISTRY = tomllib.loads(read_text(REPO_ROOT / "parity" / "reason_codes.toml"))
RUNTIME_REFUSAL_CODES = frozenset(
    entry["id"] for entry in _REGISTRY["code"] if "runtime_refusal" in entry["applies_to"]
)
HOST_STATUSES = {"supported", "partial", "unsupported"}

NSW = runpy.run_path(str(REPO_ROOT / "examples" / "python" / "when_overlap_fails.py"))
NHEFS = runpy.run_path(str(REPO_ROOT / "examples" / "python" / "observational_study.py"))


def _nsw(real: bool) -> tuple[dict[str, np.ndarray], str, str, list[str]] | None:
    if real:
        tables = NSW["real_tables"]()
        if tables is None:
            return None
        data = tables[0]
    else:
        data = NSW["standin_table"]()[0]
    return data, "treat", "re78", list(NSW["COVARIATES"])


def _nhefs(real: bool) -> tuple[dict[str, np.ndarray], str, str, list[str]] | None:
    raw = NHEFS["real_raw"]() if real else NHEFS["standin_raw"]()
    if raw is None:
        return None
    data = NHEFS["design"](raw)
    observed = ~np.isnan(data["wt82_71"])
    data = {name: column[observed] for name, column in data.items()}
    covariates = [name for name in data if name not in ("qsmk", "wt82_71")]
    return data, "qsmk", "wt82_71", covariates


DATASETS: dict[str, Callable[[bool], Any]] = {"nsw_psid": _nsw, "nhefs": _nhefs}

ESTIMATORS: dict[str, Callable[[Overlap | None], Any]] = {
    "weighting": lambda o: PropensityWeighting(bootstrap=0, overlap=o),
    "aipw": lambda o: Aipw(bootstrap=0, overlap=o),
    "matching": lambda o: PropensityMatching(bootstrap=0, overlap=o),
    "stratification": lambda o: PropensityStratification(bootstrap=0, overlap=o),
    "distance_matching": lambda o: DistanceMatching(bootstrap=0, overlap=o),
    "dml": lambda o: DML(overlap=o),
    "dr_learner": lambda o: DRLearner(overlap=o),
    "linear_adjustment": lambda o: LinearAdjustment(bootstrap=0),
}

OVERLAPS: dict[str, Overlap | None] = {
    "undeclared": None,
    "declared_clip": Overlap(clip=0.01),
    "declared_trim": Overlap(clip=None, trim=0.01),
}

TARGETS = {"all": None, "treated": population.Treated()}


_ANY = object()


def _expected_refusal(dataset: str, estimator: str, overlap: str, target: str):
    """The refusal code the stand-in data owe a route, from the documented contracts: a treated
    target is not estimable for the cross-fitted and linear routes; an undeclared overlap policy
    on the severe-overlap comparison is refused as a positivity violation by every propensity
    route; everything else answers."""
    if target == "treated" and estimator in {"dml", "dr_learner", "linear_adjustment"}:
        return "population_not_estimable"
    if dataset == "nsw_psid" and overlap == "undeclared" and estimator != "linear_adjustment":
        return "positivity_violated"
    return None


def _check(data, treatment, outcome, covariates, estimator, target, expected=_ANY) -> None:
    graph = [(z, treatment) for z in covariates] + [(z, outcome) for z in covariates]
    graph.append((treatment, outcome))
    query = (
        ant.AverageEffect(treatment, outcome)
        if target is None
        else ant.AverageEffect(treatment, outcome, target_population=target)
    )
    result = ant.analyze(
        data,
        graph=graph,
        query=query,
        estimator=estimator,
        refute="none",
        execution=ant.Execution(seed=7),
    )
    if expected is not _ANY:
        got = result.refusal.code if result.refusal is not None else None
        assert got == expected, (got, expected)
    if result.refusal is not None:
        assert result.refusal.code in RUNTIME_REFUSAL_CODES, (
            result.refusal.code,
            result.refusal.message,
        )
        return
    assert np.isfinite(result.answer.value)
    claim = result.inspect().contract["claim"]
    status = host.claim_status(claim)
    assert status in HOST_STATUSES
    if claim["support_domain"] == "limited":
        assert status == "partial"
    # A diagnostic that marks the result weakly supported (a code ending in ``.limited``:
    # overlap, dropped bootstrap replicates, weak sampler diagnostics) never leaves the claim
    # supported. An unlicensed cell (``outside_scope``) or a failed check (``contradicted``)
    # takes precedence over ``limited`` and makes the claim unsupported.
    if any(".limited" in line for line in result.value.diagnostics):
        assert claim["support_domain"] in {"limited", "outside_scope", "contradicted"}
        assert status != "supported"


def _cases(real: bool):
    for dataset in DATASETS:
        for name in ESTIMATORS:
            for overlap in OVERLAPS:
                for target in TARGETS:
                    yield pytest.param(
                        dataset, name, overlap, target, id=f"{dataset}-{name}-{overlap}-{target}"
                    )


@pytest.mark.parametrize(("dataset", "estimator", "overlap", "target"), list(_cases(False)))
def test_stand_in_routes_keep_the_host_contract(dataset, estimator, overlap, target) -> None:
    data, treatment, outcome, covariates = DATASETS[dataset](False)
    _check(
        data,
        treatment,
        outcome,
        covariates,
        ESTIMATORS[estimator](OVERLAPS[overlap]),
        TARGETS[target],
        expected=_expected_refusal(dataset, estimator, overlap, target),
    )


@pytest.mark.parametrize(("dataset", "estimator", "overlap", "target"), list(_cases(True)))
def test_real_data_routes_keep_the_host_contract(dataset, estimator, overlap, target) -> None:
    loaded = DATASETS[dataset](True)
    if loaded is None:
        pytest.skip(f"{dataset}: verified real files are not in the data cache")
    data, treatment, outcome, covariates = loaded
    _check(
        data,
        treatment,
        outcome,
        covariates,
        ESTIMATORS[estimator](OVERLAPS[overlap]),
        TARGETS[target],
    )


def test_the_nsw_comparison_refuses_without_a_declared_policy_on_every_propensity_route():
    data, treatment, outcome, covariates = _nsw(False)
    graph = [(z, treatment) for z in covariates] + [(z, outcome) for z in covariates]
    graph.append((treatment, outcome))
    for name in ("weighting", "aipw", "matching", "stratification", "dml", "dr_learner"):
        result = ant.analyze(
            data,
            graph=graph,
            query=ant.AverageEffect(treatment, outcome),
            estimator=ESTIMATORS[name](None),
            refute="none",
            execution=ant.Execution(seed=7),
        )
        assert result.refusal is not None, name
        assert result.refusal.code == "positivity_violated", (name, result.refusal.code)
