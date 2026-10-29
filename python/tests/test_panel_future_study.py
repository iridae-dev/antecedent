"""Future studies in a regime other than the observed one, ranked through the actual update.

The conditioned experiment retains the joint draws and their posterior weights. A candidate
study that runs in the observed regime is another sample of the observed estimand; one with its
own regime measures a different functional through its own licensed bridge and its own paired
per-draw affine mapping (here the study's mean is the source value divided by a slope of 2).
Truth: a same-regime study is ranked exactly as ``rank_next_samples`` ranks it; a regime changes
what each draw predicts the study will observe, so it is a different (and for a slope above one,
not more valuable) question with its own identity; a mapping whose evidence has been withdrawn
or whose source is another fitted dependency is refused.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
from antecedent import Execution, joint_panel
from antecedent import joint_panel as jp
from antecedent import scientific_bridge as sb
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration, SourceRevision
from antecedent.intervention_programs import minimum, var
from antecedent.outcomes import require_success

ROOT = Path(__file__).resolve().parents[2]
WORLDS = 2
# This routing fixture uses explicit permissive diagnostic thresholds, not a
# convergence/calibration claim. Keep enough retained draws for nonconstant 5%
# tail indicators; the original 24 draws legitimately failed the tail-ESS gate.
SAMPLER = {"chains": 2, "warmup": 20, "draws": 128, "proposal_scale": 0.01}


def _declarations():
    with (ROOT / "conformance/response/joint_panel/randomized_interaction.csv").open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    scenarios = []
    for group in range(4):
        selected = [row for row in rows if int(row["group"]) == group]
        keys = list(selected[0])
        data = {
            "a": [float(row[keys[4]]) for row in selected],
            "b": [float(row[keys[5]]) for row in selected],
            # Not an affine function of the other channels: the fit refuses a constant or
            # exactly collinear channel, whose separate effect the data cannot identify.
            "c": [2.0 + 0.5 * ((3 * index + group) % 5) for index in range(len(selected))],
            "y": [float(row[keys[6]]) if row[keys[6]] else None for row in selected],
            "z": [float(row[keys[3]]) if row[keys[3]] else None for row in selected],
            "season": [float(row[keys[2]]) for row in selected],
        }
        groups[group] = data
        scenarios.append(
            {
                "group": group,
                "history": {name: data[name][:2] for name in ("a", "b", "c")},
                "confounder": data["z"][2:],
                "season": [int(value) for value in data["season"][2:]],
            }
        )
    worlds = [
        {
            "id": "baseline",
            "reference": True,
            "channels": {"a": [0.0] * 24, "b": [0.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
        {
            "id": "joint",
            "reference": False,
            "channels": {"a": [4.0] * 24, "b": [4.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
    ]
    return groups, worlds


def _bridge(target):
    setting = sb.ExposureSetting(
        target.population_id,
        "program_period:23",
        "program_period:23",
        1.0,
        "declared_joint_program",
        1.0,
        "panel_period",
        target.regime_id,
    )
    return sb.BridgeContract(
        "parameter_intervention_response",
        replace(target, functional_id="experimental_sample_mean"),
        "model_parameter",
        target,
        "intervention_response",
        sb.Scope(target.population_id, target.regime_id),
        sb.AffineMapping(0.0, 1.0),
        assumptions=(
            sb.Assumption(
                "declared_experiment_likelihood",
                "parameter_interpretation",
                "The experimental Gaussian mean maps to this exact joint-response contrast",
                "geo_experiment",
            ),
        ),
        mapping_uncertainty=sb.MappingUncertainty("fixed_mapping", 0.0, 0.0, 0.0),
        source_dependence=sb.SourceDependence("slope_covariance", "retained_slots", 0.0),
        estimand_bridge=sb.EstimandBridge(
            "exact_program_translation",
            "geo_experiment",
            setting,
            setting,
            0.0,
            "panel_period",
            0.0,
            "declared_joint_program",
        ),
    )


@lru_cache(maxsize=1)
def _fitted():
    """One immutable actual fit; each update below receives a fresh evidence ledger."""
    groups, worlds = _declarations()
    prepared = joint_panel.prepare(
        groups,
        channels=["a", "b", "c"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions={"programmable": var("a") * var("b") * var("c") + minimum(var("c"), 0.25)},
        options={
            "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
            "inference": {
                "kind": "bayesian",
                "prior": {
                    "coefficient_sd": 10.0,
                    "log_half_saturation_mean": 0.0,
                    "log_half_saturation_sd": 1.0,
                    "lag": [],
                    "innovation_scale": 2.0,
                    "group_scale": 2.0,
                },
                "sampler": {**SAMPLER, "max_rhat": 1.0e6, "min_ess": 1.0},
            },
            "initial": None,
        },
        functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 0},
        execution=Execution(seed=20261021),
    )
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
    assert result.report()["diagnostics"]["posterior_converged"] is True
    return result


def _updated():
    """A contrast-path panel conditioned on one experiment with private ledger state."""
    result = _fitted()
    report = result.report()
    dependency = report["dependency"]
    draws = report["draws"]
    assert all(draw is not None for draw in draws), "every retained slot must be complete"
    predicted = [draw[1]["means"][-1] - draw[0]["means"][-1] for draw in draws]
    spread = max(predicted) - min(predicted)
    assert spread > 1e-10, "the fixture must retain uncertainty in its whole contrast"
    target = result.experiment_quantity("terminal_mean")
    bridge = _bridge(target)
    ledger = EvidenceLedger()
    model = ledger.add_source(SourceDeclaration(dependency, "fitted_model", [("model", "rows")]))
    experiment = ledger.add_source(
        SourceDeclaration("geo_experiment", "experiment", [("geo", "observed")])
    )
    mapping = ledger.add_source(SourceDeclaration("mapping", "fitted_model", [("mapping", "rows")]))
    slots = len(draws)
    updated = result.condition_experiment(
        bridge=bridge,
        ledger=ledger,
        observed_mean=max(predicted),
        observation_variance=100 * (spread / 6.0) ** 2,
        sample_size=100,
        mapping_dependency=dependency,
        intercepts=[0.0] * slots,
        slopes=[1.0] * slots,
        model=model,
        experiment=experiment,
        mapping=mapping,
    )
    posterior = updated.report(ledger)
    contrast = [-1.0, 1.0]
    mean = math.fsum(c * m for c, m in zip(contrast, posterior["posterior_means"], strict=True))
    actions = [
        {"id": "hold", "intercept": 0.0, "coefficients": [0.0] * WORLDS},
        {"id": "act", "intercept": -mean, "coefficients": contrast},
    ]
    return updated, ledger, bridge, dependency, actions, slots


def _regime(bridge, ledger, dependency, slots, slope: float, name: str) -> jp.PanelFutureRegime:
    evidence = ledger.add_source(SourceDeclaration(name, "fitted_model", [(name, "rows")]))
    # The retained finite mapping must average to the affine map the bridge declares.
    return jp.PanelFutureRegime(
        bridge=replace(bridge, mapping=sb.AffineMapping(0.0, slope)),
        mapping_dependency=dependency,
        intercepts=[0.0] * slots,
        slopes=[slope] * slots,
        mapping_evidence=evidence,
    )


def test_a_study_in_the_observed_regime_is_ranked_as_a_future_sample():
    updated, ledger, _bridge, _dependency, actions, _slots = _updated()
    samples = updated.rank_next_samples(
        actions=actions,
        candidates=[
            {"id": "same", "sample_size": 4, "observation_variance": 1.0, "cost_utility": 0.01}
        ],
        ledger=ledger,
        replicates=32,
    )["candidates"][0]
    studies = updated.rank_next_studies(
        actions=actions,
        studies=[jp.PanelFutureStudy("same", 4, 1.0, 0.01)],
        ledger=ledger,
        replicates=32,
    )["candidates"][0]
    # Both rank the same sampling model over the same retained update and actions.
    assert studies["question_identity"] == samples["question_identity"]
    assert studies["admissible_action_identity"] == samples["admissible_action_identity"]
    assert studies["dependency_identity"] == samples["dependency_identity"]
    slack = 3.0 * (studies["standard_error"] + samples["standard_error"]) + 1e-9
    assert abs(studies["gross_evsi"] - samples["gross_evsi"]) <= slack
    assert studies["net_value"] == pytest.approx(studies["gross_evsi"] - 0.01, abs=1e-9)
    assert studies["regime_identity"] is None


def test_a_regime_changes_what_the_study_observes_and_carries_its_own_identity():
    updated, ledger, bridge, dependency, actions, slots = _updated()
    halved = _regime(bridge, ledger, dependency, slots, 2.0, "future-mapping-2")
    tripled = _regime(bridge, ledger, dependency, slots, 3.0, "future-mapping-3")
    rows = {
        row["id"]: row
        for row in updated.rank_next_studies(
            actions=actions,
            studies=[
                jp.PanelFutureStudy("same", 4, 1.0, 0.01),
                jp.PanelFutureStudy("halved", 4, 1.0, 0.01, regime=halved),
                jp.PanelFutureStudy("tripled", 4, 1.0, 0.01, regime=tripled),
            ],
            ledger=ledger,
            replicates=32,
        )["candidates"]
    }
    assert rows["same"]["regime_identity"] is None
    assert len(rows["halved"]["regime_identity"]) == 64
    assert rows["halved"]["regime_identity"] != rows["tripled"]["regime_identity"]

    # A study that sees a fraction of the signal at the same noise cannot be worth more.
    def slack(a: str, b: str) -> float:
        return 3.0 * (rows[a]["standard_error"] + rows[b]["standard_error"]) + 1e-9

    assert rows["halved"]["gross_evsi"] <= rows["same"]["gross_evsi"] + slack("halved", "same")
    assert rows["tripled"]["gross_evsi"] <= rows["halved"]["gross_evsi"] + slack(
        "tripled", "halved"
    )


def test_a_regime_whose_mapping_evidence_is_withdrawn_is_refused():
    updated, ledger, bridge, dependency, actions, slots = _updated()
    regime = _regime(bridge, ledger, dependency, slots, 2.0, "future-mapping-2")
    require_success(
        ledger.revise_source(
            require_success(SourceRevision.withdraw(regime.mapping_evidence, "withdrawn mapping"))
        )
    )
    with pytest.raises((ValueError, CausalUnsupportedError), match="withdrawn|missing|superseded"):
        updated.rank_next_studies(
            actions=actions,
            studies=[jp.PanelFutureStudy("halved", 4, 1.0, 0.01, regime=regime)],
            ledger=ledger,
            replicates=8,
        )


def test_a_regime_mapped_over_another_fitted_source_is_refused():
    updated, ledger, bridge, _dependency, actions, slots = _updated()
    regime = _regime(bridge, ledger, "0" * 64, slots, 2.0, "future-mapping-2")
    with pytest.raises((ValueError, CausalUnsupportedError)):
        updated.rank_next_studies(
            actions=actions,
            studies=[jp.PanelFutureStudy("elsewhere", 4, 1.0, 0.01, regime=regime)],
            ledger=ledger,
            replicates=8,
        )


def test_declarations_that_are_not_future_studies_are_refused():
    updated, ledger, _bridge, _dependency, actions, _slots = _updated()
    with pytest.raises(CausalTypeError):
        updated.rank_next_studies(
            actions=actions,
            studies=[{"id": "x", "sample_size": 4, "observation_variance": 1.0}],  # type: ignore[list-item]
            ledger=ledger,
        )
    with pytest.raises(CausalTypeError):
        jp.PanelFutureStudy(
            "x",
            4,
            1.0,
            regime="elsewhere",  # type: ignore[arg-type]
        )._wire()


def test_updates_keep_the_cached_fit_and_other_ledgers_unchanged():
    fitted = _fitted()
    before = fitted.report()
    first, first_ledger, *_ = _updated()
    second, second_ledger, *_ = _updated()
    assert first_ledger is not second_ledger
    assert first.report(first_ledger) == second.report(second_ledger)
    assert fitted.report() == before
