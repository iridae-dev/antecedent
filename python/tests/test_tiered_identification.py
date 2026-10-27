"""Data-free tier declarations preserve structural scenarios before explicit estimation."""

from __future__ import annotations

import importlib

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalDataError, CausalUnsupportedError, CausalValueError
from antecedent.graph import TieredBackground, WithinTier
from antecedent.intervention import Set, Shift
from antecedent.outcomes import require_success


def _background(within=WithinTier.CODETERMINED):
    return TieredBackground(tiers=[["z"], ["t", "peer"], ["y"]], within_tier=within)


def test_codetermined_identification_has_no_data_or_estimation_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("identify-only must not ingest, prepare or estimate")

    monkeypatch.setattr(ant, "analyze", forbidden)
    monkeypatch.setattr(ant, "prepare", forbidden)
    estimation = importlib.import_module("antecedent.estimation")
    monkeypatch.setattr(estimation, "_prepared_columns", forbidden)
    monkeypatch.setattr(
        importlib.import_module("antecedent._native"), "load_float64_columns", forbidden
    )
    identified = ant.identify(graph=_background(), query=ant.AverageEffect("t", "y"))
    assert identified.status == "NonparametricallyIdentified"
    assert tuple(identified.adjustment_set) == ("z", "peer")
    assert identified.scenario_adjustment_sets == (("z", "peer"),)
    assert identified.certificate["within_tier"] == "codetermined"
    assert identified.assumption_statements
    assert any("tier-closure" in line for line in identified.derivation_statements)
    report = identified.inspect()
    assert report.uncertainty.available is False
    assert report.uncertainty.reason == "not_estimated"


def test_unknown_tiers_retain_both_declared_scenarios_without_claiming_class_bounds():
    identified = ant.identify(
        graph=_background(WithinTier.UNKNOWN),
        query=ant.AverageEffect("t", "y"),
    )
    assert identified.status == "GraphDependent"
    assert identified.scenario_adjustment_sets == (("z",), ("z", "peer"))
    assert tuple(identified.adjustment_set) == (), "two different sets must not collapse to one"
    assert identified.certificate["scenario_scope"] == "two_declared_orientation_scenarios"
    record = identified.certificate["cases"][0]["identification"]
    assert len(record["estimands"]) == 2
    assert any("not exhaustive" in str(item) for item in record["diagnostics"])
    assert "identified_weight" not in identified.certificate
    assert "unidentified_weight" not in identified.certificate


def test_tiered_identification_preserves_levels_population_and_functional():
    from antecedent.population import Treated
    from antecedent.query import Exceedance

    identified = ant.identify(
        graph=_background(),
        query=ant.AverageEffect(
            "t",
            "y",
            control_level=-1.0,
            active_level=2.0,
            target_population=Treated(),
            outcome_functional=Exceedance(0.7),
        ),
    )
    wire = identified.certificate["witness_query"]["average_effect"]
    assert wire["control"]["set"]["value"] == {"float64": -1.0}
    assert wire["active"]["set"]["value"] == {"float64": 2.0}
    assert wire["target_population"] == "treated"
    assert wire["outcome_functional"] == {"exceedance": 0.7}


def test_joint_tiered_identification_retains_requested_intervention_coordinates():
    identified = ant.identify(
        graph=_background(),
        query=ant.InterventionResponse("y", intervention=[Set("t", 1.0), Set("peer", 0.0)]),
    )
    assert identified.status == "NonparametricallyIdentified"
    assert identified.scenario_adjustment_sets == (("z",),)
    record = identified.certificate["cases"][0]["identification"]
    assert len(record["estimands"]) == 1
    wire = identified.certificate["witness_query"]["response"]["functional"][
        "intervention_response"
    ]
    assert wire["interventions"] == (
        {"set": {"variable": 1, "value": {"float64": 1.0}}},
        {"set": {"variable": 2, "value": {"float64": 0.0}}},
    )


def test_a_tiered_identification_can_explicitly_estimate_its_own_target():
    z = np.linspace(-1.0, 1.0, 300)
    t = 0.3 * z + np.sin(np.arange(300) * 1.7)
    y = 1.0 + 2.0 * t + 0.5 * z
    graph = TieredBackground(tiers=[["z"], ["t"], ["y"]])
    identified = ant.identify(graph=graph, query=ant.AverageEffect("t", "y"))
    result = require_success(
        identified.estimate(
            {"z": z, "t": t, "y": y},
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    )
    fresh = ant.analyze(
        {"z": z, "t": t, "y": y},
        graph=graph,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=1),
    )
    fresh = require_success(fresh)
    assert result.estimate.ate == pytest.approx(2.0, abs=1e-10)
    assert result.estimate.ate == pytest.approx(fresh.estimate.ate, abs=1e-12)
    assert result.program_id == fresh.program_id
    assert result.claim_id == fresh.claim_id


@pytest.mark.parametrize("identifier", ["backdoor.adjustment", "generalized.adjustment"])
def test_a_tier_declaration_rejects_a_conflicting_identifier(identifier):
    with pytest.raises(CausalUnsupportedError, match="selects its own identifier"):
        ant.identify(graph=_background(), query=ant.AverageEffect("t", "y"), identifier=identifier)


def test_unknown_joint_and_nonhard_interventions_are_refused():
    with pytest.raises(CausalUnsupportedError, match="Unknown-tier") as refused:
        ant.identify(
            graph=_background(WithinTier.UNKNOWN),
            query=ant.InterventionResponse("y", intervention=[Set("t", 1), Set("peer", 1)]),
        )
    assert refused.value.reason_code == "cell_not_licensed"
    with pytest.raises(CausalUnsupportedError, match="hard Set"):
        ant.identify(
            graph=_background(),
            query=ant.InterventionResponse("y", intervention=[Shift("t", 1), Set("peer", 1)]),
        )


def test_tiered_identification_resolves_an_explicit_schema_and_refuses_unknown_names():
    identified = ant.identify(
        graph=_background(), query=ant.AverageEffect("t", "y"), names=["y", "peer", "t", "z"]
    )
    assert set(identified.adjustment_set) == {"z", "peer"}
    assert identified.certificate["names"] == ("y", "peer", "t", "z")
    with pytest.raises((CausalValueError, CausalDataError), match="unknown"):
        ant.identify(graph=_background(), query=ant.AverageEffect("absent", "y"))
