"""2.3 A exit gate, boxes 5 to 8: the four scientific demonstrations through public Python routes.

Each test composes the public routes of the committed unit tests and re-derives the asserted
numbers by hand (copied builders; no private module is imported):

* box 5 - ``test_cpdag_scenarios.py`` and ``test_scenario_covariance.py``;
* box 6 - ``test_closed_pilots.py`` (joint Bayesian, nested Markov);
* box 7 - ``test_temporal_extensions.py`` and ``test_effect_constancy.py``;
* box 8 - ``test_temporal_counterfactual.py``, ``test_closed_pilots.py``,
  ``test_temporal_extensions.py``.

Closure is status-driven: a closed route is called and must refuse with the registry's reason code;
a route the registry has since opened is not called. No calibration is measured here. Calibration
of the four calibrated A claims is unmeasured at this commit (``parity/coverage_records.toml``
holds none of their records), so the box 6 demonstration can only be the "otherwise record the
failed gate and typed refusal" branch of the gate item.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from antecedent import Admg, Cpdag
from antecedent.errors import CausalUnsupportedError
from antecedent.temporal import (
    ConstancyConclusion,
    EffectEstimand,
    TemporalRefusal,
    effect_constancy,
)
from antecedent.temporal_counterfactual import (
    ActionHistory,
    NodeMechanism,
    TemporalMechanisms,
    TransportedPrerequisites,
    UnitHistory,
    temporal_fixed_population,
    transported_path_specific,
)
from antecedent.transport import advanced as transport
from antecedent.transport.advanced import (
    InitialStateLaw,
    TemporalExtensionRefusal,
    TemporalPremises,
    TemporalUnitPanel,
    TemporalWindow,
    temporal_dependent_interval,
    temporal_initial_state,
    temporal_new_period_refresh,
)

from _refusal import assert_registered_refusal

ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict[str, dict[str, Any]]:
    with (ROOT / "parity" / "promotion_2_3.toml").open("rb") as handle:
        records = tomllib.load(handle)["record"]
    return {r["id"]: r for r in records}


def _route_is_closed(record_id: str, route: str) -> bool:
    routes = {r["name"]: r for r in _registry()[record_id]["routes"]}
    return routes[route]["status"] == "closed"


def _refused_closed(call: Any, record_id: str, route: str, detail: str) -> None:
    """Refuse with ``cell_not_licensed`` and the frozen detail while the registry says closed.

    A route the registry has since opened is not called: its demonstration is a licence test.
    """
    if not _route_is_closed(record_id, route):
        return
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert caught.value.reason_code == "cell_not_licensed"
    assert detail in str(caught.value)
    assert_registered_refusal(caught.value)


# --------------------------------------------------------------------------------------------
# Box 5: a new graph/selection family (CPDAG completions) and measured-design covariance.
# --------------------------------------------------------------------------------------------

NAMES = ["a", "b", "c"]
FORWARD = frozenset({("a", "b"), ("b", "c")})
FORK = frozenset({("b", "a"), ("b", "c")})
BACKWARD = frozenset({("b", "a"), ("c", "b")})


def _joint(a: int, b: int, c: int) -> float:
    def p(one: bool, p_one: float) -> float:
        return p_one if one else 1.0 - p_one

    return p(a == 1, 0.4) * p(b == 1, (0.2, 0.7)[a]) * p(c == 1, (0.1, 0.8)[b])


def _catalog(measured: list[str]) -> transport.EvidenceCatalog:
    return transport.EvidenceCatalog(
        regimes=[transport.EvidenceRegime("obs", "target", measured=list(measured))]
    )


def _completions(evidence: Any = None, **overrides: Any) -> Any:
    table = tuple(_joint(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
    axes = tuple((n, (0.0, 1.0)) for n in NAMES)
    options: dict[str, Any] = {
        "outcomes": ["c"],
        "treatments": ["b"],
        "source": "src_pop",
        "target": "target",
        "coordinates": [transport.VariableCoordinate(n, "binary") for n in NAMES],
        "evidence": evidence or transport.CompletionEvidence(_catalog(NAMES), "target-law"),
        "laws": transport.ExactTransportData(
            (transport.ExactDiscreteLaw("target", "obs", axes, table, "target"),)
        ),
        "at": {"b": 1.0},
    }
    options.update(overrides)
    graph = Cpdag.from_directed_undirected(NAMES, [], [("a", "b"), ("b", "c")])
    return transport.cpdag_completion_scenarios(graph, **options)


def _bound(edges: frozenset[tuple[str, str]], identity: str, measured: list[str]) -> Any:
    return transport.CompletionEvidence(_catalog(measured), identity, completion=sorted(edges))


# Shared-data covariance fixture: six complete rows (z, x, y) of a tiny discrete sample.
ROWS = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0)]


def _row_table() -> Any:
    return transport.RowTable(["z", "x", "y"], ROWS, [f"u{i}" for i in range(len(ROWS))])


def _score_estimator(name: str, **overrides: Any) -> Any:
    scores = {
        "a": [(1.0, {"x": 1, "y": 1}), (-1.0, {"x": 0, "y": 1})],
        "b": [(1.0, {"x": 1, "y": 1}), (-1.0, {"x": 1, "y": 0})],
    }
    functional = transport.LinearScore([transport.ScoreTerm(c, w) for c, w in scores[name]])
    return transport.ScenarioEstimator(name, functional, **overrides)


def _covariance(*estimators: Any) -> Any:
    return transport.scenario_shared_covariance(
        _row_table(), list(estimators), method="exact_enumeration", max_compositions=10_000
    )


def test_a_exit_demo_graph_selection_family_has_scenario_evidence_and_honest_mass() -> None:
    # The chain a - b - c has the completions a->b->c, a<-b->c and a<-b<-c. With P(a=1)=0.4,
    # P(b=1|a)=(0.2,0.7), P(c=1|b)=(0.1,0.8): P(c=1|do(b=1)) = 0.8 for the first two (c does not
    # depend on a given b) and P(c=1) = 0.6*0.1 + 0.4*0.8 = 0.38 for the third (c is not a
    # descendant of b). The structural envelope is [0.38, 0.8], a range, not an interval.
    hand = {FORWARD: 0.8, FORK: 0.8, BACKWARD: 0.38}
    result = _completions()
    assert result.scope == "cpdag_completion_scenarios_structural_envelope"
    assert result.complete and result.exportable
    assert (result.counts.identified, result.counts.unidentified) == (3, 0)
    counts = result.counts
    assert (counts.unevaluated, counts.not_enumerated, counts.total) == (0, 0, 3)
    for completion in result.completions:
        assert completion.mean("c") == pytest.approx(hand[frozenset(completion.edges)], abs=1e-12)
        assert completion.evidence_identity == "target-law"
        assert len(completion.id) == 64
    assert result.envelope.mean_range("c") == pytest.approx((0.38, 0.8), abs=1e-12)
    assert "not_a_confidence_interval" in result.envelope.interpretation
    assert not hasattr(result, "weights") and not hasattr(result, "mass")
    with pytest.raises(transport.CpdagScenarioRefusal) as aggregate:
        result.aggregate_interval()
    assert aggregate.value.reason_code == "scenario_aggregate_not_licensed"
    assert aggregate.value.detail == "scenarios.shared_data_aggregate"

    # Scenario-specific evidence: a completion whose own evidence lacks a variable is missing
    # evidence and its mass is not redistributed; the envelope ranges over the others only.
    evidence = [
        _bound(FORWARD, "ev-forward", NAMES),
        _bound(FORK, "ev-fork", NAMES),
        _bound(BACKWARD, "ev-backward", ["a", "b"]),
    ]
    partial = _completions(evidence)
    status = {frozenset(c.edges): c.status for c in partial.completions}
    assert status == {FORWARD: "identified", FORK: "identified", BACKWARD: "missing_evidence"}
    assert (partial.counts.identified, partial.counts.unidentified) == (2, 1)
    assert partial.envelope.mean_range("c") == pytest.approx((0.8, 0.8), abs=1e-12)
    consumed = transport.consume_cpdag_scenarios_artifact(partial.export())
    assert consumed.counts == partial.counts and consumed.completions == partial.completions

    # Honest structural mass: a stopped enumeration counts what it never read as unevaluated and
    # exports an identical prefix; it claims no envelope.
    stopped = _completions(max_steps=3)
    assert not stopped.complete and len(stopped.completions) == 1
    assert (stopped.counts.not_enumerated, stopped.counts.unevaluated) == (3, 4)
    cut = stopped.counts
    assert (cut.identified, cut.unidentified, cut.total) == (0, 0, 4)
    assert not stopped.counts.exact and stopped.envelope is None
    assert stopped.receipt.stop == "search.operations"

    # Shared-data covariance, published only for the measured design (one row table, one snapshot,
    # shared units). Closed form Cov(a, b) = (E[ab] - E[a]E[b]) / n with n = 6: scores a and b are
    # +-1 on the cells (x=1,y=1), (x=0,y=1) and (x=1,y=0), giving 17/216 on both diagonals and
    # 11/216 off the diagonal, means 1/6 and 1/6; correlation 11/17.
    design = _covariance(_score_estimator("a"), _score_estimator("b"))
    assert design.claim == "point_only" and design.scope == "shared_row_covariance_point_only"
    assert design.entry("a", "a") == pytest.approx(17 / 216, abs=1e-12)
    assert design.entry("a", "b") == pytest.approx(11 / 216, abs=1e-12)
    assert design.entry("a", "b") == design.entry("b", "a")
    assert design.means == pytest.approx((1 / 6, 1 / 6), abs=1e-12)
    assert design.correlation("a", "b") == pytest.approx(11 / 17, abs=1e-12)
    assert not hasattr(design, "interval") and not hasattr(design, "lower")
    # Outside the measured design (declared independent or other-snapshot estimates) the shared
    # selection is unknown, so no covariance is published.
    for first, second in (
        (_score_estimator("a"), _score_estimator("b", dependence="independent_sample")),
        (_score_estimator("a", snapshot="snap:one"), _score_estimator("b", snapshot="snap:two")),
    ):
        with pytest.raises(transport.ScenarioCovarianceRefusal) as unknown:
            _covariance(first, second)
        assert unknown.value.reason_code == "route_not_supported"
        assert unknown.value.detail == "scenario_covariance.unknown_dependence"


# --------------------------------------------------------------------------------------------
# Box 6: calibrated joint Bayesian row and the nested-Markov pilot.
# --------------------------------------------------------------------------------------------

VERMA_NODES = ["X1", "X2", "X3", "X4"]
VERMA_DIRECTED = [("X1", "X2"), ("X2", "X3"), ("X3", "X4")]


def test_a_exit_demo_calibrated_joint_bayesian_row_records_failed_gate_and_typed_refusal() -> None:
    """The gate item is satisfiable here only as its "otherwise" branch.

    Both posterior candidates remain unmeasured and normal public routes stay closed.
    The continuous nested posterior has its own implemented method and frozen, unrun
    measurement suite. Allocated coverage identities are not measured records and
    cannot promote either the point fit or its separate Bayesian posterior.
    """
    registry = _registry()
    with (ROOT / "parity" / "coverage_records.toml").open("rb") as handle:
        collected = {r["id"] for r in tomllib.load(handle).get("record", [])}
    joint_record = registry["2.3A.X4.joint_bayesian_transport"]
    nested_record = registry["2.3A.X4.binary_nested_markov_pilot"]
    assert nested_record["inference_claim"] == "point_only"
    assert not nested_record.get("coverage_records", [])
    posterior = next(
        row
        for row in nested_record["inference_outputs"]
        if row["id"] == "binary_nested_markov_pilot.posterior"
    )
    assert posterior["implementation_status"] == "implemented_unmeasured"
    assert posterior["allocation_status"] == "frozen_harness_compiled_measurement_pending"
    assert (ROOT / posterior["measurement_suite"]).is_file()
    allocated = set(posterior["allocated_coverage_records"])
    assert allocated and allocated <= set(nested_record["candidate_coverage_records"])
    assert not allocated & collected
    for record in (joint_record,):
        assert record["inference_claim"] == "calibrated"
        if set(record["coverage_records"]) <= collected:
            # Measured: the record may legitimately be promoted; this demonstration then moves to
            # a licence test.
            continue
        # The failed gate is recorded and the record stays out of the promoted set.
        assert record["status"] == "carried_forward", record["id"]
        assert "calibrat" in record["inference_notes"].lower()
    oracle = ROOT / "crates/antecedent-estimate/tests/joint_bayesian_transport.rs"
    assert "x4_joint_posterior_oracle_single_source_matches_closed_form" in oracle.read_text(
        encoding="utf-8"
    )

    def joint(**overrides: Any) -> Any:
        kwargs: dict[str, Any] = {
            "sources": [{"id": "s1"}, {"id": "s2"}],
            "target": {"x": [0.1, 0.2, 0.3]},
            "features": ["x"],
            "draws": 1000,
            "seed": 7,
        }
        kwargs.update(overrides)
        return transport.joint_bayesian_transport(**kwargs)

    _refused_closed(
        joint,
        "2.3A.X4.joint_bayesian_transport",
        "antecedent.transport.joint_bayesian",
        "bayesian_transport.route_frozen",
    )
    # A request outside the pilot's scope refuses with its own scope detail.
    adjacent = Admg.from_edges(["x", "a", "y"], [("x", "a"), ("a", "y")], [("x", "y")])
    with pytest.raises(CausalUnsupportedError) as scope:
        joint(graph=adjacent)
    assert scope.value.reason_code == "route_not_supported"
    assert "bayesian_transport.unsupported_graph" in str(scope.value)

    verma = Admg.from_edges(VERMA_NODES, VERMA_DIRECTED, [("X2", "X4")])
    counts = [100.0 + index for index in range(16)]
    _refused_closed(
        lambda: transport.binary_nested_markov(graph=verma, regimes=[{"counts": counts}]),
        "2.3A.X4.binary_nested_markov_pilot",
        "antecedent.transport.binary_nested_markov",
        "nested_markov.route_frozen",
    )
    # The pilot's own scope refusal (an intervened regime) is not a nonidentification claim.
    with pytest.raises(CausalUnsupportedError) as outside:
        transport.binary_nested_markov(
            graph=verma, regimes=[{"counts": counts, "intervened": ["X2"]}]
        )
    assert "nested_markov.outside_binary_pilot" in str(outside.value)
    assert "nonidentif" not in str(outside.value)


# --------------------------------------------------------------------------------------------
# Box 7: initial-state temporal inference, new-period invalidation, partition constancy.
# --------------------------------------------------------------------------------------------

P_L1 = [[3, 7], [2, 8]]
M10 = [
    [[[3, 4], [4, 6]], [[3, 5], [5, 7]]],
    [[[4, 3], [7, 6]], [[6, 3], [5, 7]]],
]
SEQUENCE = (0, 0)


def _panel(
    snapshot: str, *, first_unit: int = 0, start: int = 0, lift: int = 0
) -> TemporalUnitPanel:
    units = []
    for unit in range(2):
        rows = []
        for s0 in range(2):
            for a1 in range(2):
                n_one = 20 * P_L1[s0][a1]
                for level in range(2):
                    n_level = n_one if level == 1 else 200 - n_one
                    for a2 in range(2):
                        n_cell = n_level // 2
                        ones = n_cell * (M10[s0][a1][level][a2] + lift) // 10
                        for k in range(n_cell):
                            y = 1.0 if k < ones else 0.0
                            time_id = start + len(rows)
                            rows.append((first_unit + unit, time_id, s0, a1, level, a2, y))
        units.extend(rows)
    return TemporalUnitPanel.from_rows(snapshot, units)


def _premises(**overrides: Any) -> TemporalPremises:
    fields = {
        "initial_state_variable": "s0",
        "time_order": ["s0", "a1", "l2", "a2", "y"],
        "source_regime": "source",
        "target_regime": "target",
        "graph_id": "two_slice_graph_v1",
        "proof_id": "proof_1",
    }
    fields.update(overrides)
    return TemporalPremises(**fields)


def _window(period: tuple[int, int], **overrides: Any) -> TemporalWindow:
    fields = {
        "lag_alignment": {"s0": 1, "a1": 1, "l2": 2, "a2": 2, "y": 2},
        "intervention_history": ["a1=0", "a2=0"],
        "selection_targets": ["s_t1", "s_t2"],
    }
    fields.update(overrides)
    return TemporalWindow(period=period, **fields)


def _held() -> Any:
    return temporal_initial_state(
        _panel("snap_old"),
        sequence=SEQUENCE,
        target_law=InitialStateLaw("target-state", {0: 1.0 - 0.7, 1: 0.7}),
        premises=_premises(),
        fixed_state=1,
        window=_window((0, 1000)),
    )


def _refusal(call: Any) -> TemporalExtensionRefusal:
    with pytest.raises(TemporalExtensionRefusal) as caught:
        call()
    assert isinstance(caught.value, CausalUnsupportedError)
    assert_registered_refusal(caught.value)
    return caught.value


def test_a_exit_demo_temporal_initial_state_refresh_invalidation_and_constancy() -> None:
    # Initial-state integration. The enumerated two-step SCM gives R(s0=0) = 0.33, R(s0=1) = 0.46
    # for the sequence (0, 0); the target law P(s0=1) = 0.7 marginalizes to 0.3*0.33 + 0.7*0.46 =
    # 0.421, while fixing the mode wrongly gives 0.46. The result is point-only with no interval.
    result = _held()
    assert result.value == pytest.approx(0.421, abs=1e-12)
    assert result.inference_claim == "point_only" and result.interval is None
    assert [c.response for c in result.contributions] == pytest.approx([0.33, 0.46], abs=1e-12)
    assert result.fixed.value == pytest.approx(0.46, abs=1e-12)
    assert result.label != result.fixed.label
    # A source-labelled law refuses as target evidence.
    source_law = InitialStateLaw("source-state", {0: 0.5, 1: 0.5}, population="source")
    refused = _refusal(
        lambda: temporal_initial_state(
            _panel("p"), sequence=SEQUENCE, target_law=source_law, premises=_premises()
        )
    )
    assert (refused.reason_code, refused.detail) == (
        "transport_missing_evidence",
        "initial_state.target_law_missing",
    )

    # New-period refresh. A replacement panel with every cell mean 0.1 higher re-evaluates to
    # 0.521 (0.3*0.43 + 0.7*0.56); the held interval is invalidated and never survives.
    refreshed = temporal_new_period_refresh(
        result,
        _panel("snap_new", first_unit=10, start=1000, lift=1),
        window=_window((1000, 2000)),
        interval_existed=True,
    )
    assert refreshed.value == pytest.approx(0.521, abs=1e-12)
    assert refreshed.previous_value == pytest.approx(0.421, abs=1e-12)
    assert refreshed.interval is None and refreshed.interval_invalidated is True
    assert refreshed.receipt.old_snapshot_id == "snap_old"
    assert refreshed.receipt.new_snapshot_id == "snap_new"
    assert result.value == pytest.approx(0.421, abs=1e-12), "the held result is unchanged"
    # Invalidation reasons: each premise change refuses with its own typed detail.
    changes = {
        "temporal_refresh.horizon_changed": {"horizon": 3},
        "temporal_refresh.lag_alignment_changed": {
            "lag_alignment": {"s0": 1, "a1": 2, "l2": 2, "a2": 2, "y": 2}
        },
        "temporal_refresh.intervention_history_changed": {"intervention_history": ["a1=0", "a2=1"]},
        "temporal_refresh.premises_changed": {"proof_id": "proof_2"},
    }
    for detail, overrides in changes.items():
        error = _refusal(
            lambda overrides=overrides: temporal_new_period_refresh(
                _held(),
                _panel("snap_new", first_unit=10, start=1000, lift=1),
                window=_window((1000, 2000), **overrides),
                interval_existed=True,
            )
        )
        assert (error.reason_code, error.detail) == ("route_not_supported", detail)
    stale = _refusal(
        lambda: temporal_new_period_refresh(
            _held(),
            _panel("snap_old", first_unit=10, start=1000, lift=1),
            window=_window((1000, 2000)),
        )
    )
    assert stale.detail == "temporal_refresh.stale_snapshot"

    # Dependent-sample interval: the route stays closed (calibration unmeasured).
    rows = [
        (unit, time_id, s0, 0, level, a2, ((unit * 7 + s0 * 3 + level * 5 + a2) % 10) / 10.0)
        for unit in range(24)
        for time_id, (s0, level, a2) in enumerate(
            (s0, level, a2) for s0 in range(2) for level in range(2) for a2 in range(2)
        )
    ]
    panel = TemporalUnitPanel.from_rows("interval-panel", rows)
    _refused_closed(
        lambda: temporal_dependent_interval(panel, sequence=SEQUENCE, replicates=40),
        "2.3A.X5.dependent_temporal_interval",
        "antecedent.transport.temporal_dependent_interval",
        "temporal_interval.route_frozen",
    )

    # Partition constancy at its declared coordinate. Q = (e2 - e1)^2 / (se1^2 + se2^2).
    effect = EffectEstimand(
        "ate_difference", "outcome_units", "treat_vs_control", "all_observed_h2"
    )
    equal = effect_constancy(
        [("p1", 1.0, 0.5), ("p2", 1.0, 0.5)], estimand=effect, coordinate="period"
    )
    assert equal.statistic == pytest.approx(0.0, abs=1e-12)
    assert equal.degrees_of_freedom == 1 and equal.p_value == pytest.approx(1.0, abs=1e-12)
    assert equal.conclusion is ConstancyConclusion.NOT_REJECTED
    # Effects 1 and 2 with variance 1/4 each: Q = 1 / (1/4 + 1/4) = 2 and p = erfc(1).
    varying = effect_constancy(
        [("p1", 1.0, 0.5), ("p2", 2.0, 0.5)], estimand=effect, coordinate="period"
    )
    assert varying.statistic == pytest.approx(2.0, abs=1e-9)
    assert varying.p_value == pytest.approx(math.erfc(1.0), abs=1e-9)
    assert varying.table() == [
        {"label": p, "coordinate": f"period:{p}", "support": "supported", "effect": e, "se": 0.5}
        for p, e in (("p1", 1.0), ("p2", 2.0))
    ]
    # Calibration is unmeasured and a non-rejection is not proof of constancy.
    assert varying.calibration == "unmeasured" and varying.inference_claim == "point_only"
    assert any("does not prove" in caveat for caveat in varying.caveats)
    assert (
        varying.identity
        == effect_constancy(
            [("p2", 2.0, 0.5), ("p1", 1.0, 0.5)], estimand=effect, coordinate="period"
        ).identity
    ), "partition order does not change the result identity"


# --------------------------------------------------------------------------------------------
# Box 8: temporal counterfactual, and the transported / sampled-recovery cells (closed).
# --------------------------------------------------------------------------------------------

NOISE = [[0.3, -0.2, 0.5], [-0.4, 0.6, -0.1], [0.0, 0.1, 0.9], [1.1, -0.7, -0.3]]
FACTUAL_ACTIONS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
PLUS = (1.0, 1.0)
MINUS = (0.0, 0.0)


def _simulate(noise: list[float], actions: Any) -> list[float]:
    l0 = 1.0 + noise[0]
    l1 = 0.5 + 0.8 * l0 + 1.5 * actions[0] + noise[1]
    y = -1.0 + 0.5 * l0 + 0.7 * actions[0] + 2.0 * l1 + 1.2 * actions[1] + noise[2]
    return [l0, actions[0], l1, actions[1], y]


def _histories() -> list[UnitHistory]:
    rows = []
    for i, (u, a) in enumerate(zip(NOISE, FACTUAL_ACTIONS, strict=True)):
        l0, a0, l1, a1, y = _simulate(u, a)
        rows.append(UnitHistory(f"unit-{i}", l0, a0, l1, a1, y, history=f"hist-{i}"))
    return rows


def _mechanisms(halfwidth: float | None = None) -> TemporalMechanisms:
    return TemporalMechanisms(
        "fit-known-coefficients",
        (
            NodeMechanism("covariate_0", 1.0, {}, halfwidth),
            NodeMechanism("covariate_1", 0.5, {"covariate_0": 0.8, "action_0": 1.5}, halfwidth),
            NodeMechanism(
                "outcome",
                -1.0,
                {"covariate_0": 0.5, "action_0": 0.7, "covariate_1": 2.0, "action_1": 1.2},
                halfwidth,
            ),
        ),
    )


def _counterfactual(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "plus": ActionHistory("always_treat", PLUS),
        "minus": ActionHistory("never_treat", MINUS),
        "mechanisms": _mechanisms(),
        "snapshot": "snapshot-1",
    }
    arguments.update(overrides)
    return temporal_fixed_population(_histories(), **arguments)


def test_a_exit_demo_temporal_counterfactual_and_closed_cells_stay_visibly_closed() -> None:
    # Closed-form SCM L0 = 1 + u0, L1 = 0.5 + 0.8 L0 + 1.5 A0 + u1,
    # Y = -1 + 0.5 L0 + 0.7 A0 + 2 L1 + 1.2 A1 + u2. Abduction recovers each unit's noise exactly
    # and the shared noise cancels: Y(1,1) - Y(0,0) = 0.7 + 1.2 + 2 * 1.5 = 4.9 for every unit.
    # Unit 0 (u = 0.3, -0.2, 0.5): L0 = 1.3, L1 = 0.5 + 1.04 + 1.5 - 0.2 = 2.84,
    # Y(1,1) = -1 + 0.65 + 0.7 + 5.68 + 1.2 + 0.5 = 7.73 and Y(0,0) = -1 + 0.65 + 2.68 + 0.5 = 2.83.
    result = _counterfactual()
    assert result.inference_claim == "point_only"
    assert result.contrast == pytest.approx(4.9, abs=1e-9)
    assert all(row.contrast == pytest.approx(4.9, abs=1e-9) for row in result.units)
    assert result.units[0].plus_outcome == pytest.approx(7.73, abs=1e-9)
    assert result.units[0].minus_outcome == pytest.approx(2.83, abs=1e-9)
    assert result.receipt.shared_by_both_worlds
    assert (result.receipt.n_units, result.receipt.n_worlds) == (4, 2)
    assert len({draw[2] for draw in result.receipt.unit_draws}) == 4, "one distinct draw per unit"
    # A noise term outside the declared bound refutes the history with its witness.
    with pytest.raises(TemporalRefusal) as refuting:
        _counterfactual(mechanisms=_mechanisms(halfwidth=0.45))
    assert refuting.value.detail == "temporal_counterfactual.refuting_history"
    assert refuting.value.witness["unit"] == "unit-0"
    assert refuting.value.witness["residual"] == pytest.approx(0.5, abs=1e-9)
    # Unpaired histories refuse with the offending unit.
    with pytest.raises(TemporalRefusal) as unpaired:
        _counterfactual(
            plus=ActionHistory("always_treat", PLUS, units=("unit-0", "unit-1", "unit-2"))
        )
    assert unpaired.value.detail == "temporal_counterfactual.unpaired_histories"
    assert unpaired.value.offending == "unit-3"

    # Transported counterfactual: closed, naming exactly the gates still missing.
    gates = (
        "transport_license_missing",
        "fixed_population_temporal_license_missing",
        "cross_population_assumptions_missing",
    )
    _refused_closed(
        lambda: transported_path_specific(required_factors=["source:rct", "target:field"]),
        "2.3A.X8.transported_path_specific_counterfactual",
        "antecedent.cross_world.transported_path_specific",
        "transported_counterfactual.route_frozen",
    )
    with pytest.raises(TemporalRefusal) as transported:
        transported_path_specific(required_factors=["source:rct", "target:field"])
    assert transported.value.missing_gates == gates
    assert transported.value.missing_factors == ("source:rct", "target:field")
    with pytest.raises(TemporalRefusal) as passed:
        transported_path_specific(
            prerequisites=TransportedPrerequisites(True, True, True),
            required_factors=["source:rct", "target:field"],
            supplied_factors={"source:rct": "e1", "target:field": "e2"},
        )
    # Every gate and factor present still does not open the cell: the joint cross-world theorem
    # is not derived.
    assert passed.value.reason_code == "cell_not_licensed"
    assert passed.value.detail == "transported_counterfactual.route_frozen"
    assert passed.value.missing_gates == ()

    # Sampled observation recovery: closed (calibration unmeasured) after real validation.
    query = transport.ObservationRecoveryQuery(
        population="clinic",
        observed_regime="observed",
        partially_observed=[transport.PartiallyObservedVariable("X", "R", "X_star")],
    )
    complete = [(index, 1, (0, 1)[index % 2], 0) for index in range(40)]
    _refused_closed(
        lambda: transport.sampled_observation_recovery(
            stage=SimpleNamespace(outcome="recovered"),
            query=query,
            rows=complete,
            snapshot="snap-observed",
            replicates=100,
            interval_method="bootstrap_percentile",
            seed=3,
        ),
        "2.3A.X10.sampled_observation_recovery",
        "antecedent.transport.sampled_observation_recovery",
        "sampled_recovery.route_frozen",
    )
