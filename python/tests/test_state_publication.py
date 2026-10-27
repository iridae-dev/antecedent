"""Lineage-bound publication of recomputed results, and the claim effects of a state's
invalidation log.

``CausalState`` records what a data or premise event made stale; it never recomputes. A result is
published only through the identities of the prepared study that recomputed it, so a stale
version, a snapshot recorded before the latest data event, or a premise event answered by the
identical lineage is refused and leaves every freshness stamp as it was. Transported claims
resting on state queries recompute or withdraw from the same log.
"""

from __future__ import annotations

import re

import antecedent as ant
import numpy as np
import pytest
from antecedent import state as st
from antecedent.errors import CausalResourceError, CausalStateError, CausalTypeError
from antecedent.execution import CachePolicy, Execution
from antecedent.prepared import prepare, rebind
from antecedent.state import CausalState, publish_recomputed_results, result_lineage_fingerprint
from antecedent.transport import revalidation as rv

from _transport_fixtures import digest
from test_transport_revalidation import classes, scene


def _data(effect: float = 2.0) -> dict[str, np.ndarray]:
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
    return {"t": t, "z": z, "y": effect * t + z}


def _study(effect: float = 2.0):
    graph = ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    return prepare(_data(effect), graph=graph, query=ant.AverageEffect("t", "y"), refute=False)


def _identities(study) -> dict:
    """The identities of the result a study recomputes: the outcome's, with its data snapshot."""
    return dict(study.estimate().identities)


def _batch() -> tuple[list[str], list[np.ndarray]]:
    return ["x"], [np.arange(8, dtype=np.float64)]


def _state_with_query(cache_bytes: int = 1 << 20) -> tuple[CausalState, int]:
    state = CausalState(Execution(cache=CachePolicy.bounded(cache_bytes)))
    state.append_data(*_batch())
    _, query = state.register_average_effect(0, 1)
    return state, query


def test_lineage_fingerprint_covers_every_identity_layer():
    first = _identities(_study())
    again = _identities(_study())
    other_data = _identities(rebind(_study(), data=_data(4.0)))
    fingerprint = result_lineage_fingerprint(first)
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert result_lineage_fingerprint(again) == fingerprint
    assert first["data_snapshot_id"] != other_data["data_snapshot_id"]
    assert result_lineage_fingerprint(other_data) != fingerprint
    layers = {
        "target": first["target_id"],
        "identification": first["identification_id"],
        "identification_product": first["identification_product_id"],
        "program": first["program_id"],
        "inference_binding": first["inference_binding_id"],
        "observation": first["observation_id"],
        "data_snapshot": first["data_snapshot_id"],
    }
    assert result_lineage_fingerprint(layers) == fingerprint
    for layer in ("target", "inference_binding", "observation", "data_snapshot"):
        moved = {**layers, layer: digest(9)}
        assert result_lineage_fingerprint(moved) != fingerprint, layer
    # An absent optional layer is a different lineage, not a defaulted one.
    assert result_lineage_fingerprint({**layers, "program": None}) != fingerprint
    with pytest.raises(ValueError, match="required layer 'target'"):
        result_lineage_fingerprint({**layers, "target": None})
    with pytest.raises(ValueError, match="64 hexadecimal"):
        result_lineage_fingerprint({**layers, "observation": "not-a-digest"})


def test_publication_is_bound_to_the_lineage_that_recomputed_it():
    state, query = _state_with_query()
    first = _identities(_study())
    recomputed = _identities(rebind(_study(), data=_data(4.0)))
    assert state.stale_queries() == [query]
    publish_recomputed_results(state, state.version, [(query, first, 32)])
    assert state.stale_queries() == []

    state.append_data(*_batch())
    assert state.stale_queries() == [query]
    # A version that advanced during the recompute is refused and nothing is published.
    with pytest.raises(CausalStateError, match="stale commit"):
        publish_recomputed_results(state, state.version - 1, [(query, recomputed, 32)])
    assert state.stale_queries() == [query]
    # Republishing the snapshot recorded before the data event is not a recomputation.
    with pytest.raises(CausalStateError, match="before the latest data event"):
        publish_recomputed_results(state, state.version, [(query, first, 32)])
    assert state.stale_queries() == [query]
    publish_recomputed_results(state, state.version, [(query, recomputed, 32)])
    assert state.stale_queries() == []

    # The unchecked path cannot restamp a lineage-bound query.
    with pytest.raises(CausalStateError, match="published with a lineage"):
        state.refresh_results([(query, 1, 16)])

    # A premise event answered by the identical lineage is not a recomputation either.
    state.add_constraint("no-edge", 7)
    assert state.stale_queries() == [query]
    with pytest.raises(CausalStateError, match="republishes the"):
        publish_recomputed_results(state, state.version, [(query, recomputed, 32)])
    assert state.stale_queries() == [query]


def test_publication_refuses_unknown_queries_budget_overruns_and_bad_arguments():
    state, query = _state_with_query(cache_bytes=8)
    identities = _identities(_study())
    with pytest.raises(CausalStateError, match="query 99"):
        publish_recomputed_results(state, state.version, [(99, identities, 4)])
    with pytest.raises(CausalResourceError):
        publish_recomputed_results(state, state.version, [(query, identities, 1024)])
    assert state.stale_queries() == [query]
    publish_recomputed_results(state, state.version, [(query, identities, 4)])
    assert state.stale_queries() == []
    with pytest.raises(CausalTypeError):
        publish_recomputed_results(object(), 0, [])  # type: ignore[arg-type]
    assert st.publish_recomputed_results is publish_recomputed_results


def test_state_invalidations_recompute_or_withdraw_the_claims_resting_on_them():
    state, query = _state_with_query()
    _, other = state.register_average_effect(1, 0)
    publish_recomputed_results(
        state,
        state.version,
        [(query, _identities(_study()), 16), (other, _identities(_study(3.0)), 16)],
    )
    since = state.version
    bindings = [("effect_trial", "query", query), ("effect_registry", "query", other)]
    assert rv.invalidation_effects(state, since, bindings) == ()
    baseline = rv.revalidate(
        scene(1, 10), scene(1, 10), rv.invalidation_effects(state, since, bindings)
    )
    assert set(classes(baseline).values()) == {"still_valid"}

    state.add_constraint("no-edge", 7)
    effects = rv.invalidation_effects(state, since, bindings)
    assert {e.key for e in effects} == {
        rv.claim_node_key("effect_trial"),
        rv.claim_node_key("effect_registry"),
    }
    assert {e.kind for e in effects} == {"recomputed"}
    assert all(
        e.text.startswith(f"invalidation.at_version:{state.version}:graph:") for e in effects
    )
    report = rv.revalidate(scene(1, 10), scene(1, 10), effects)
    assert classes(report) == {
        "effect_registry": "recompute",
        "effect_trial": "recompute",
        "treat_or_wait": "recompute",
    }
    # Events at or before ``since`` are not effects of this observation.
    assert rv.invalidation_effects(state, state.version, bindings) == ()

    # Removing a premise withdraws the claim it rested on; a claim bound to one query only
    # follows the entries that name it or all results.
    since = state.version
    state.remove_constraint("no-edge")
    removed = rv.invalidation_effects(state, since, [("effect_trial", "query", query)])
    assert [(e.key, e.kind) for e in removed] == [(rv.claim_node_key("effect_trial"), "withdrawn")]
    withdrawn = rv.revalidate(scene(1, 10), scene(1, 10), removed)
    assert classes(withdrawn)["effect_trial"] == "withdrawn"
    assert classes(withdrawn)["effect_registry"] == "still_valid"

    # Applied through the claim state, the effects are recorded with the scene they observed.
    claims = rv.TransportClaimState()
    claims.observe(scene(1, 10))
    observed = claims.observe(scene(1, 10), rv.invalidation_effects(state, since, bindings))
    assert classes(observed)["effect_trial"] == "withdrawn"


def test_claim_effects_refuse_unknown_targets_and_a_future_version():
    state, query = _state_with_query()
    with pytest.raises(ValueError, match="after the state version"):
        rv.invalidation_effects(state, state.version + 1, [("c", "query", query)])
    with pytest.raises(ValueError, match="unknown invalidation target"):
        rv.invalidation_effects(state, 0, [("c", "plan", None)])
    with pytest.raises(ValueError, match="needs a query id"):
        rv.invalidation_effects(state, 0, [("c", "query", None)])
    with pytest.raises(CausalTypeError):
        rv.invalidation_effects(object(), 0, [])  # type: ignore[arg-type]
    state.update_assumption("positivity")
    kinds = ["lag_indexes", "graph_scores", "particle_filters", "mechanism_diagnostics"]
    bound = [(f"claim_{kind}", kind, None) for kind in kinds] + [("claim_stat", "suff_stat", "k")]
    effects = rv.invalidation_effects(state, 0, bound)
    # The all-results entry reaches every bound claim.
    assert {e.key for e in effects} == {rv.claim_node_key(c) for c, _, _ in bound}


def test_the_cache_budget_comes_from_the_execution_cache_policy() -> None:
    from antecedent.state import antecedent_state_append

    for policy in (CachePolicy.disabled(), CachePolicy.bounded(8)):
        state = CausalState(Execution(cache=policy))
        state.append_data(*_batch())
        _, query = state.register_average_effect(0, 1)
        with pytest.raises(CausalResourceError):
            publish_recomputed_results(state, state.version, [(query, _identities(_study()), 64)])
    roomy = CausalState(Execution(cache=CachePolicy.unbounded()))
    roomy.append_data(*_batch())
    _, query = roomy.register_average_effect(0, 1)
    publish_recomputed_results(roomy, roomy.version, [(query, _identities(_study()), 1 << 40)])
    assert antecedent_state_append(3, execution=Execution())[0] >= 3
    with pytest.raises(TypeError, match="unexpected keyword"):
        CausalState(cache_bytes=8)
    with pytest.raises(TypeError, match="unexpected keyword"):
        antecedent_state_append(2, cache_bytes=8)
    with pytest.raises(CausalTypeError):
        CausalState(execution={"cache": 8})


def test_particle_filter_stream_is_seeded_by_the_execution() -> None:
    def trace(seed: int) -> dict:
        state = CausalState()
        state.particle_filter_init("pf", 32, execution=Execution(seed=seed))
        for y in (0.2, -0.1, 0.4):
            state.particle_filter_step("pf", y)
        return state.particle_filter_get("pf")

    assert trace(2) == trace(2)
    assert trace(2) != trace(3)
    with pytest.raises(TypeError, match="unexpected keyword"):
        CausalState().particle_filter_init("pf", 32, seed=2)
