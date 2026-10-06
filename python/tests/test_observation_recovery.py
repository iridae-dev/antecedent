"""Exact binary observation recovery (2.2B X10) from Python.

The observed pattern law and the truth are enumerated here from binary
structural mechanisms, independently of the library. Treatment ``t`` and outcome
``y`` are item-missing with response indicators ``rt`` and ``ry`` and proxies
``t_obs`` and ``y_obs`` (levels 0, 1 and "?"); the confounder ``z`` is always
observed. ``rt`` depends on ``z`` and ``ry`` on ``t``: no response depends on
its own variable, so the full law is recoverable (graph-licensed recovery, not
MAR/IPCW), and the backdoor effect of ``t`` on ``y`` follows from it.
"""

import itertools
import json

import pytest
from antecedent import Admg
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced as transport

NAMES = ["t", "y", "z", "rt", "ry", "t_obs", "y_obs"]
OBSERVED = ["z", "rt", "ry", "t_obs", "y_obs"]

P_Z = 0.4
P_T = {0: 0.3, 1: 0.7}
P_Y = {(0, 0): 0.2, (0, 1): 0.5, (1, 0): 0.6, (1, 1): 0.85}
P_RT = {0: 0.8, 1: 0.5}
P_RY_BY_T = {0: 0.9, 1: 0.6}
P_RY_BY_Y = {0: 0.9, 1: 0.6}


def bern(p, value):
    return p if value else 1.0 - p


def joint(self_censoring=False):
    """Yield ((t, y, z, rt, ry), probability)."""
    for t, y, z, rt, ry in itertools.product((0, 1), repeat=5):
        p_ry = P_RY_BY_Y[y] if self_censoring else P_RY_BY_T[t]
        yield (
            (t, y, z, rt, ry),
            (
                bern(P_Z, z)
                * bern(P_T[z], t)
                * bern(P_Y[(t, z)], y)
                * bern(P_RT[z], rt)
                * bern(p_ry, ry)
            ),
        )


def observed_law(self_censoring=False, snapshot="snap-1"):
    index = {}
    levels = [(0, 1), (0, 1), (0, 1), (0, 1, "?"), (0, 1, "?")]
    for cell in itertools.product(*levels):
        index[cell] = 0.0
    for (t, y, z, rt, ry), p in joint(self_censoring):
        index[(z, rt, ry, t if rt else "?", y if ry else "?")] += p
    return transport.ExactDiscreteLaw(
        population="clinic",
        regime="observed",
        axes=[(name, values) for name, values in zip(OBSERVED, levels, strict=True)],
        probabilities=[index[cell] for cell in itertools.product(*levels)],
        snapshot_identity=snapshot,
    )


def truth():
    """P(t, y, z) with z fastest, in the recovered law's axis order (t, y, z)."""
    out = [0.0] * 8
    for (t, y, z, _rt, _ry), p in joint():
        out[t * 4 + y * 2 + z] += p
    return out


def effect_truth(level):
    return sum(bern(P_Z, z) * P_Y[(level, z)] for z in (0, 1))


def graph(self_censoring=False):
    edges = [
        ("z", "t"),
        ("z", "y"),
        ("t", "y"),
        ("z", "rt"),
        ("y", "ry") if self_censoring else ("t", "ry"),
        ("t", "t_obs"),
        ("rt", "t_obs"),
        ("y", "y_obs"),
        ("ry", "y_obs"),
    ]
    return Admg.from_edges(NAMES, edges, [])


def catalog(distribution="joint", measured=OBSERVED):
    return transport.EvidenceCatalog(
        environments=(
            transport.Environment(
                "clinic",
                tuple(
                    transport.VariableCoordinate(name, "categorical", cardinality=3)
                    if name.endswith("_obs")
                    else transport.VariableCoordinate(name, "binary")
                    for name in NAMES
                ),
            ),
        ),
        regimes=(
            transport.EvidenceRegime(
                "observed", "clinic", measured=list(measured), distribution=distribution
            ),
        ),
        bindings=(transport.RegimeBinding("observed", "snap-1", sampling="independent"),),
    )


def query(partially=None):
    return transport.ObservationRecoveryQuery(
        population="clinic",
        observed_regime="observed",
        partially_observed=partially
        or [
            transport.PartiallyObservedVariable("t", "rt", "t_obs"),
            transport.PartiallyObservedVariable("y", "ry", "y_obs"),
        ],
        fully_observed=["z"],
    )


def decide(self_censoring=False, effect=True, **kwargs):
    return transport.identify_observation_recovery(
        graph=graph(self_censoring),
        query=query(),
        catalog=kwargs.pop("catalog", catalog()),
        effect_outcomes=["y"] if effect else None,
        effect_treatments=["t"] if effect else None,
        **kwargs,
    )


REQUESTS = [{"t": 0.0}, {"t": 1.0}]


def p_one(effect):
    return effect["probabilities"][[atom[0] for atom in effect["atoms"]].index(1.0)]


def test_identify_observation_recovery_decides_once_and_executes_its_plan_after_disposal():
    builder = decide()
    assert builder.outcome == "recovered"
    decision = builder.decision()
    assert decision["route"] == "graph-licensed recovery, not MAR/IPCW"
    assert [f["conditioning"] for f in decision["factors"]] == [["z"], ["rt", "t_obs"]]
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    result = json.loads(prepared.estimate())
    assert result["recovered"]["probabilities"] == pytest.approx(truth(), abs=1e-12)
    assert prepared.plan()["effect"]["treatments"] == ["t"]


def test_prepare_exact_compiles_the_retained_derivation_after_builder_disposal():
    builder = decide()
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    plan = prepared.plan()
    assert plan["rule_version"] == "x10.recovery.v1"
    assert all(m["identity"].startswith("catalog_distribution.v1") for m in plan["margins"])
    result = json.loads(prepared.estimate())
    assert result["recovered"]["origin"] == "recovered"
    assert result["recovered"]["variables"] == ["t", "y", "z"]


def test_estimate_executes_every_request_from_the_retained_plan():
    builder = decide()
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    result = json.loads(prepared.estimate())
    assert [e["assignment"] for e in result["effects"]] == [{"t": 0.0}, {"t": 1.0}]
    for level, effect in enumerate(result["effects"]):
        assert p_one(effect) == pytest.approx(effect_truth(level), abs=1e-12)
    assert result["interval"] == {"available": False, "status": "point_only"}
    assert prepared.plan()["requests"] == 2


def test_refresh_rebinds_the_snapshot_law_and_re_executes_the_retained_plan():
    builder = decide()
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    before = json.loads(prepared.estimate())
    prepared.refresh(observed_law())
    after = json.loads(prepared.estimate())
    assert after["recovered"]["probabilities"] == before["recovered"]["probabilities"]
    assert prepared.plan()["rule_version"] == "x10.recovery.v1"
    with pytest.raises(CausalUnsupportedError) as refused:
        prepared.refresh(observed_law(snapshot="snap-2"))
    assert refused.value.reason_code == "invalid_argument"
    assert "recovery.invalid_observed_law" in str(refused.value)


def test_export_binds_names_and_the_consumer_recomputes_after_builder_disposal():
    builder = decide()
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    plan = prepared.plan()
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_observation_recovery_artifact(artifact))
    assert consumed["recovered"]["probabilities"] == live["recovered"]["probabilities"]
    assert consumed["effects"] == live["effects"]
    assert consumed["derivation"]["identity"] == plan["identity"]


def test_consumer_refuses_an_edited_artifact_and_limits_below_the_stored_ones():
    builder = decide()
    prepared = builder.prepare_exact(observed_law(), REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    plan = prepared.plan()
    artifact = prepared.export()
    consumed = json.loads(transport.consume_observation_recovery_artifact(artifact))
    assert consumed["effects"] == live["effects"]
    assert consumed["derivation"]["factors"] == plan["factors"]
    edited = bytearray(artifact)
    edited[-5] ^= 0x01
    with pytest.raises(CausalUnsupportedError, match=r"recovery\.invalid_derivation") as bad:
        transport.consume_observation_recovery_artifact(bytes(edited))
    assert bad.value.reason_code == "transport_not_certified"
    with pytest.raises(CausalUnsupportedError) as refused:
        transport.consume_observation_recovery_artifact(artifact, max_search_operations=10)
    assert refused.value.reason_code == "route_not_supported"


def test_self_censoring_is_nonrecoverable_with_a_verified_witness():
    stage = decide(self_censoring=True, effect=False)
    assert stage.outcome == "nonrecoverable"
    decision = stage.decision()
    assert decision["self_censoring_edge"] == ["y", "ry"]
    verified = decision["verified"]
    assert verified["observed_cells_equal"] > 0
    assert verified["target_masses"][0] != verified["target_masses"][1]
    with pytest.raises(CausalUnsupportedError) as refused:
        stage.prepare_exact(observed_law(self_censoring=True))
    assert refused.value.reason_code == "transport_proven_non_transportable"
    assert "recovery.nonrecoverable_witness" in str(refused.value)


def test_counted_laws_are_not_licensed_on_the_recovery_route():
    stage = decide()
    with pytest.raises(CausalUnsupportedError, match="cell_not_licensed") as refused:
        stage.prepare_empirical(observed_law(), REQUESTS)
    assert refused.value.reason_code == "cell_not_licensed"


def test_separate_marginals_refuse_as_a_missing_margin():
    with pytest.raises(CausalUnsupportedError) as refused:
        decide(catalog=catalog(distribution="separate_marginals"))
    assert refused.value.reason_code == "transport_missing_evidence"
    assert "recovery.missing_margin" in str(refused.value)


def test_proxy_relabelling_is_refused_never_aligned():
    swapped = [
        transport.PartiallyObservedVariable("t", "rt", "y_obs"),
        transport.PartiallyObservedVariable("y", "ry", "t_obs"),
    ]
    with pytest.raises(CausalUnsupportedError) as refused:
        transport.identify_observation_recovery(
            graph=graph(), query=query(swapped), catalog=catalog()
        )
    assert refused.value.reason_code == "invalid_argument"
    assert "recovery.invalid_query" in str(refused.value)


def test_budget_stop_is_a_receipt_never_a_verdict():
    with pytest.raises(CausalUnsupportedError) as refused:
        decide(max_operations=3)
    assert refused.value.reason_code == "transport_budget_cancel"
    assert "recovery.budget" in str(refused.value)


def test_python_values_are_validated():
    with pytest.raises(CausalValueError):
        transport.PartiallyObservedVariable("t", "t", "t_obs")
    with pytest.raises(CausalTypeError):
        transport.identify_observation_recovery(graph=graph(), query="q", catalog=catalog())
    with pytest.raises(CausalValueError):
        transport.identify_observation_recovery(
            graph=graph(), query=query(), catalog=catalog(), effect_outcomes=["y"]
        )


def test_a_budget_or_cancellation_stop_shows_its_receipt():
    from antecedent.errors import CausalError
    from antecedent.state import CancellationToken

    with pytest.raises(CausalError) as stopped:
        decide(max_operations=3)
    text = str(stopped.value)
    assert stopped.value.reason_code == "transport_budget_cancel"
    assert "receipt: stop search.operations" in text
    assert "unevaluated [" in text
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalError) as cancelled:
        decide(cancel=token)
    assert cancelled.value.reason_code == "transport_budget_cancel"
    assert "receipt: stop search.cancelled" in str(cancelled.value)
