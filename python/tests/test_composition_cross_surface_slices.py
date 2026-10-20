"""Cross-surface slice cases on the public Python API.

Scientific inputs and hand values are `conformance/cross_surface/slice_cases.json`.
Each case drops the producer and reads an independently loaded artifact.
Specified feedback has no Python binding; the Rust contract test covers it. The finite-discrete
horizon cell is a regime label on the same table, as on the Rust surface, because no facade
producer builds a horizon table of counts.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from antecedent import (
    Admg,
    AverageEffect,
    Bayesian,
    Frequentist,
    PulseEffect,
    ResponseCurve,
    analyze,
    load,
)
from antecedent import evidence_ledger as ledger_api
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as transport
from antecedent.transport._temporal_extensions import (
    MARGINALIZED_LABEL,
    TemporalExtensionRefusal,
    TemporalIntervalCandidate,
    _temporal_dependent_interval_candidate,
)

from _repo_text import REPO_ROOT, load_json
from test_observation_recovery import REQUESTS, decide, observed_law, truth

CASES = load_json(REPO_ROOT / "conformance" / "cross_surface" / "slice_cases.json")


# The checked-operation reference the independent consumer leaves unresolved. Only the static
# frequentist line has an independent replay; every other cell's body is readable and forwardable,
# and is not claimed as a verified program.
_UNRESOLVED = {
    (False, False): (),
    (False, True): ("dependencies.checked_bayesian_dag_ate_operation",),
    (True, False): ("dependencies.checked_temporal_dag_effect_operation",),
    (True, True): ("dependencies.checked_bayesian_temporal_dag_effect_operation",),
}
_MONTE_CARLO_STANDARD_ERRORS = 5.0


def _gaussian_cell(horizon: bool, bayesian: bool):
    gaussian = CASES["gaussian"]
    if horizon:
        data = {"t": gaussian["horizon_treatment"], "y": gaussian["horizon_outcome"]}
        graph = [("t", 1, "y", 0)]
        query = PulseEffect("t", "y", treatment_lag=1, horizon_steps=1)
        label = "gaussian-horizon-bayesian" if bayesian else "gaussian-horizon-frequentist"
    else:
        data = {"t": gaussian["treatment"], "y": gaussian["outcome"]}
        graph = [("t", "y")]
        query = AverageEffect("t", "y")
        label = "gaussian-static-bayesian" if bayesian else "gaussian-static-frequentist"
    inference = (
        Bayesian(
            n_draws=gaussian["posterior_draws"],
            prior_scale=gaussian["prior_scale"],
            backend="conjugate",
        )
        if bayesian
        else Frequentist()
    )
    result = analyze(
        data,
        graph=graph,
        query=query,
        inference=inference,
        bootstrap=0,
        refute="none",
        seed=1,
    )
    # A frequentist fit is exact on the noiseless line. A Bayesian point estimate is the mean of
    # iid conjugate draws, so it carries Monte Carlo error: five standard errors of that mean.
    if bayesian:
        assert result.posterior is not None
        tolerance = (
            _MONTE_CARLO_STANDARD_ERRORS
            * result.posterior.effect_sd
            / math.sqrt(result.posterior.n_draws)
        )
        assert result.posterior.effect_mean == pytest.approx(gaussian["effect"], abs=0.05)
        assert result.posterior.interval_type == "equal_tailed_95"
    else:
        assert result.posterior is None
        tolerance = 1e-8 if horizon else 1e-12
    assert result.estimate.ate == pytest.approx(gaussian["effect"], abs=tolerance)
    artifact = result.export(artifact_id=label)
    del result
    loaded = load(artifact)
    assert loaded.acceptance.recognized
    assert loaded.acceptance.unresolved == _UNRESOLVED[(horizon, bayesian)], label
    assert loaded.acceptance.verified == (not loaded.acceptance.unresolved)
    assert loaded.decoded_artifact.payload["estimate"] == pytest.approx(
        gaussian["effect"], abs=tolerance
    )


def test_four_way_gaussian_cells_reload_from_independent_artifacts():
    for horizon in (False, True):
        for bayesian in (False, True):
            _gaussian_cell(horizon, bayesian)


def _discrete_program(regime: str):
    discrete = CASES["finite_discrete"]
    graph = Admg.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    catalog = transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime(regime, discrete["population"], measured=["t", "z", "y"])]
    )
    query = transport.TransportQuery(
        ResponseCurve("t", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("source", discrete["population"], []),
        catalog=catalog,
    )
    identification = transport.identify(graph=graph, query=query)
    return identification, transport.reload_lowered_program(identification), catalog


def _law(counts, regime: str, snapshot: str):
    discrete = CASES["finite_discrete"]
    total = sum(counts)
    return transport.ExactDiscreteLaw(
        discrete["population"],
        regime,
        (("t", (0.0, 1.0)), ("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(count / total for count in counts),
        snapshot,
        empirical_counts=tuple(counts) if snapshot == discrete["snapshot"] else None,
    )


def _contrast(program, catalog, law) -> float:
    active = program.evaluate_exact(catalog, (law,), {"t": 1.0, "y": 1.0})
    control = program.evaluate_exact(catalog, (law,), {"t": 0.0, "y": 1.0})
    return active - control


def test_finite_discrete_adjustment_reloads_at_each_regime_label():
    discrete = CASES["finite_discrete"]
    assert discrete["static_regime"] != discrete["horizon_regime"]
    for regime in (str(discrete["static_regime"]), str(discrete["horizon_regime"])):
        identification, program, catalog = _discrete_program(regime)
        law = _law(discrete["counts"], regime, discrete["snapshot"])
        assert _contrast(program, catalog, law) == pytest.approx(discrete["contrast"])
        wire = program.to_wire_json()
        del program
        restored = transport.restore_lowered_program(identification, wire)
        assert _contrast(restored, catalog, law) == pytest.approx(discrete["contrast"])
        mixed = 0.0
        for index, draw in enumerate(discrete["posterior_draws"]):
            draw_law = _law(draw["counts"], regime, f"draw-{index}")
            contrast = _contrast(restored, catalog, draw_law)
            assert contrast == pytest.approx(draw["contrast"])
            mixed += draw["weight"] * contrast
        assert mixed == pytest.approx(discrete["posterior_mean"])


def _sources(session):
    mixed = CASES["mixed_evidence"]
    sources = []
    for index, (start, end) in enumerate(mixed["observation_ranges"]):
        sources.append(
            session.add_source(
                ledger_api.SourceDeclaration(
                    f"study_{index}",
                    "observation_set",
                    observations=ledger_api.rows("cohort", range(start, end)),
                )
            )
        )
    return [
        ledger_api.StudySummary(source, estimate, variance)
        for source, estimate, variance in zip(
            sources, mixed["estimates"], mixed["variances"], strict=True
        )
    ]


def test_mixed_evidence_pools_and_reloads_from_the_ledger():
    mixed = CASES["mixed_evidence"]
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    assert isinstance(pooled.value, ledger_api.FixedEffect)
    assert pooled.value.estimate == pytest.approx(mixed["pooled"])
    assert pooled.value.variance == pytest.approx(mixed["pooled_variance"])
    declared = sum(end - start for start, end in mixed["observation_ranges"])
    assert pooled.mass.distinct_observations == declared
    exported = session.export("mixed-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(exported.artifact, exported.identity)
    replay = consumed.synthesis_of(pooled.result)
    assert replay.values[0] == pytest.approx(mixed["pooled"])
    assert replay.values[1] == pytest.approx(mixed["pooled_variance"])


def test_reused_evidence_keeps_one_observation_count():
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    first = session.reuse_result("cited_once", pooled.result)
    second = session.reuse_result("cited_again", pooled.result)
    assert first != second
    counted = pooled.mass.distinct_observations
    for result_id in (pooled.result, first, second):
        assert session.result(result_id).mass.distinct_observations == counted
    exported = session.export("reused-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(exported.artifact, exported.identity)
    labels = {record.label: record.mass.distinct_observations for record in consumed.results}
    assert labels["pool"] == counted
    assert labels["cited_once"] == counted
    assert labels["cited_again"] == counted


def test_overlapping_studies_pool_under_a_declared_covariance_and_replace_the_disjoint_pool():
    mixed = CASES["mixed_evidence"]
    overlap = mixed["overlap"]
    session = ledger_api.EvidenceLedger()
    summaries = _sources(session)
    disjoint = session.pool_summaries("pool", summaries)
    first_source = summaries[0].source
    report = session.revise_source(
        ledger_api.SourceRevision.update(
            summaries[1].source,
            ledger_api.SourceDeclaration(
                "study_1_grown",
                "observation_set",
                observations=ledger_api.rows("cohort", range(*overlap["grown_range"])),
            ),
        )
    )
    grown = report.created_source
    assert grown is not None
    assert session.standing(disjoint.result) == "recompute"
    account = session.account_observations([first_source, grown])
    assert account.mass.distinct_observations == overlap["distinct_observations"]

    overlapping = [
        ledger_api.StudySummary(source, estimate, variance)
        for source, estimate, variance in zip(
            [first_source, grown], mixed["estimates"], mixed["variances"], strict=True
        )
    ]
    with pytest.raises(ledger_api.EvidenceLedgerRefusal) as refusal:
        session.pool_summaries("overlap_pool", overlapping, replaces=disjoint.result)
    assert refusal.value.reason_code == "evidence_double_counted"
    assert refusal.value.detail == "evidence_ledger.shared_observations"

    replacing = session.pool_summaries(
        "overlap_pool",
        overlapping,
        covariance=overlap["covariance"],
        replaces=disjoint.result,
    )
    assert replacing.value.estimate == pytest.approx(overlap["pooled"])
    assert replacing.value.variance == pytest.approx(overlap["pooled_variance"])
    assert replacing.mass.declared_slots == overlap["declared_slots"]
    assert replacing.mass.distinct_observations == overlap["distinct_observations"]
    assert replacing.mass.repeated_slots == overlap["shared"]
    record = session.result(replacing.result)
    assert record.dependence_declared
    assert record.replaces == disjoint.result
    assert session.result(disjoint.result).status == "replaced"

    exported = session.export("overlapping-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(exported.artifact, exported.identity)
    replay = consumed.synthesis_of(replacing.result)
    assert replay.values[0] == pytest.approx(overlap["pooled"])
    assert replay.values[1] == pytest.approx(overlap["pooled_variance"])
    statuses = {item.id: item.status for item in consumed.results}
    assert statuses[disjoint.result] == "replaced"


def _panel(shocks):
    temporal = CASES["temporal"]
    rows = []
    for unit in range(temporal["units"]):
        shock = shocks[unit % len(shocks)]
        time = 0
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        rows.append((unit, time, s0, a1, l2, a2, mean + shock))
                        time += 1
    return transport.TemporalUnitPanel.from_rows(temporal["snapshot"], rows)


def _interval_arguments():
    temporal = CASES["temporal"]
    return {
        "sequence": tuple(temporal["sequence"]),
        "target_law": transport.InitialStateLaw(
            temporal["target_snapshot"],
            {level: mass for level, mass in temporal["target_law"]},
        ),
        "seed": temporal["seed"],
        "replicates": temporal["replicates"],
        "level": temporal["level"],
        "min_units": temporal["min_units"],
        "max_failed_fraction": temporal["max_failed_fraction"],
    }


def _reload_candidate(panel, point):
    """Build the candidate over `panel`, drop it, and consume its artifact."""
    try:
        candidate = _temporal_dependent_interval_candidate(
            panel, estimand=MARGINALIZED_LABEL, **_interval_arguments()
        )
    except TemporalExtensionRefusal as refusal:
        pytest.skip(f"the candidate route is closed in this build: {refusal}")
    assert candidate.point == pytest.approx(point, abs=1e-12)
    assert candidate.calibration == "unmeasured"
    artifact = candidate.export()
    identity = candidate.identity
    del candidate
    return TemporalIntervalCandidate.load(artifact, expected_identity=identity)


def test_temporal_dependence_reloads_from_the_measured_artifact():
    temporal = CASES["temporal"]
    panel = _panel(temporal["unit_shocks"])
    loaded = _reload_candidate(panel, temporal["point"])
    assert loaded.point == pytest.approx(temporal["point"], abs=1e-12)

    # The measured route additionally needs a current attesting coverage record. Until the
    # coverage records are re-measured it refuses for exactly that reason and no other.
    try:
        result = transport.temporal_dependent_interval(panel, **_interval_arguments())
    except TemporalExtensionRefusal as refusal:
        assert "coverage_record_not_attesting" in str(refusal), refusal
        return
    artifact = result.export()
    expected = result.expected_identity
    del result
    measured = MeasuredInference.load(artifact, expected=expected)
    assert measured.scalar("response").point == pytest.approx(temporal["point"], abs=1e-12)


def _independent_rows_standard_error(panel_rows):
    mean = sum(panel_rows) / len(panel_rows)
    variance = sum((y - mean) ** 2 for y in panel_rows) / (len(panel_rows) - 1)
    return math.sqrt(variance / len(panel_rows))


def test_uneven_unit_shocks_widen_the_whole_unit_interval_beyond_independent_rows():
    temporal = CASES["temporal"]
    uneven = temporal["uneven"]
    shocks = uneven["unit_shocks"]
    loaded = _reload_candidate(_panel(shocks), uneven["point"])

    units = temporal["units"]
    mean_shock = sum(shocks[unit % len(shocks)] for unit in range(units)) / units
    assert abs(mean_shock) > 0.1, "the shocks do not cancel"
    assert loaded.point - temporal["point"] == pytest.approx(mean_shock, abs=1e-12)

    outcomes = []
    for unit in range(temporal["units"]):
        shock = shocks[unit % len(shocks)]
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        outcomes.append(mean + shock)
    rows_width = 2 * temporal["row_interval_z"] * _independent_rows_standard_error(outcomes)
    whole_unit_width = loaded.upper - loaded.lower
    assert whole_unit_width > rows_width, (whole_unit_width, rows_width)
    assert loaded.lower < loaded.point < loaded.upper


def test_recovered_feedback_reloads_the_observation_law():
    recovery = CASES["recovery"]
    law = observed_law(snapshot=recovery["python_snapshot"])
    assert law.population == recovery["population"]
    assert law.snapshot_identity == recovery["python_snapshot"]
    builder = decide()
    prepared = builder.prepare_exact(law, REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_observation_recovery_artifact(artifact))
    assert consumed["recovered"]["probabilities"] == pytest.approx(
        live["recovered"]["probabilities"]
    )
    assert consumed["recovered"]["probabilities"] == pytest.approx(truth())
    assert consumed["recovered"]["origin"] == "recovered"


def _operation_message(name: str) -> str:
    return re.escape(name.replace("in_place", "in-place").replace("_", " "))


def test_reduced_retention_serves_only_the_summary():
    from antecedent import _native
    from antecedent.errors import CausalStateError

    retention = CASES["retention"]
    mode = _native.RetentionMode.summary_only()
    assert mode.as_str() == retention["reduced"]
    reuse = _native.RetainedOperation.reuse_summary()
    assert mode.serves(reuse)
    mode.require(reuse)
    # The Python surface has no least-squares summary to serve, so the served quantity is covered
    # on the Rust surface; here each unavailable operation is refused with its reason.
    for name in retention["unserved"]:
        operation = getattr(_native.RetainedOperation, name)()
        assert not mode.serves(operation), name
        with pytest.raises(
            CausalStateError, match=f"summary-only cannot serve {_operation_message(name)}"
        ):
            mode.require(operation)
