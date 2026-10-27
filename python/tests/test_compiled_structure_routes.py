"""Discovery, fitted-model attribution and simulation run through compiled routes.

Every ``discover_*`` binding, every ``FittedGcm`` attribution and simulation, and the
discover-then-fit helpers return results that keep the compiled route they ran on: its
contract (query kind, structure source, declared inference, and the structure provenance
of an accepted discovery), its claim, and its inspection record.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import _native


def _static_rows(seed: int = 4, n: int = 400) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    # Non-Gaussian noise so LiNGAM orients every edge.
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    y = 1.5 * t + 0.6 * z + rng.uniform(-1.0, 1.0, size=n)
    return ["z", "t", "y"], [z, t, y]


def _series_rows(seed: int = 7, n: int = 300) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.normal()
        y[i] = 0.6 * x[i - 1] + 0.3 * y[i - 1] + rng.normal()
    return ["x", "y"], [x, y]


def _assert_discover_contract(result, inference: str) -> None:
    contract = result.contract
    assert contract["query_kind"] == "Discover"
    assert contract["structure_source"] == "discovered"
    assert contract["inference"] == inference
    assert contract["lowered_program"] == result.route.program_identity
    claim = result.claim
    assert len(claim["claim_id"]) == 64
    assert claim["kind"]
    assert "calibration_status" in claim
    assert result.inspect()["query_kind"] == "Discover"


@pytest.mark.parametrize(
    "run",
    [
        lambda names, cols: _native.discover_pc(names, cols, alpha=0.05, seed=1),
        lambda names, cols: _native.discover_ges(names, cols, alpha=0.05, seed=1),
        lambda names, cols: _native.discover_lingam(names, cols, seed=1),
    ],
    ids=["pc", "ges", "direct_lingam"],
)
def test_static_discovery_states_its_contract(run):
    names, cols = _static_rows()
    first = run(names, cols)
    _assert_discover_contract(first, "frequentist")
    second = run(names, cols)
    assert first.claim["claim_id"] == second.claim["claim_id"]
    assert [(e.source, e.target, e.at_source, e.at_target) for e in first.graph_edges] == [
        (e.source, e.target, e.at_source, e.at_target) for e in second.graph_edges
    ]


def test_search_bounds_are_part_of_the_discovery_query_identity():
    names, cols = _static_rows()
    base = _native.discover_pc(names, cols, alpha=0.05, max_cond_size=1, seed=1)
    same = _native.discover_pc(names, cols, alpha=0.05, max_cond_size=1, seed=1)
    stricter = _native.discover_pc(names, cols, alpha=0.01, max_cond_size=1, seed=1)
    deeper = _native.discover_pc(names, cols, alpha=0.05, max_cond_size=2, seed=1)
    assert base.claim["claim_id"] == same.claim["claim_id"]
    claims = {r.claim["claim_id"] for r in (base, stricter, deeper)}
    assert len(claims) == 3, "alpha and the conditioning-set bound each change the query"
    series_names, series_cols = _series_rows()
    lag_one = _native.discover_pcmci_plus(series_names, series_cols, max_lag=1, seed=1)
    lag_two = _native.discover_pcmci_plus(series_names, series_cols, max_lag=2, seed=1)
    assert lag_one.claim["claim_id"] != lag_two.claim["claim_id"]


def test_pcmci_plus_states_its_contract():
    names, cols = _series_rows()
    first = _native.discover_pcmci_plus(names, cols, max_lag=1, seed=1)
    _assert_discover_contract(first, "frequentist")
    second = _native.discover_pcmci_plus(names, cols, max_lag=1, seed=1)
    assert first.claim["claim_id"] == second.claim["claim_id"]
    assert [(link.source, link.target, link.p_value) for link in first.links] == [
        (link.source, link.target, link.p_value) for link in second.links
    ]


def test_exact_posterior_states_a_bayesian_contract():
    names, cols = _static_rows(n=200)
    first = _native.discover_exact_dag_posterior(names, cols, seed=1)
    _assert_discover_contract(first, "bayesian")
    second = _native.discover_exact_dag_posterior(names, cols, seed=1)
    assert first.claim["claim_id"] == second.claim["claim_id"]
    assert first.weights == second.weights


def test_supplied_posterior_atoms_have_no_route():
    post = _native.GraphPosterior.from_atoms(["a", "b"], [1.0], [0])
    assert post.route is None
    with pytest.raises(ValueError, match="compiled route"):
        _ = post.contract


def test_discovered_fit_carries_structure_provenance():
    names, cols = _static_rows()
    data = dict(zip(names, cols, strict=True))
    result, edges = antecedent.gcm.attribute_paths_discovered(
        data,
        discovery=antecedent.discovery.LiNGAM(),
        sources=["z"],
        outcome="y",
        execution=antecedent.Execution(seed=1),
    )
    assert edges
    contract = result.contract
    assert contract["query_kind"] == "Attribute"
    assert contract["structure_provenance.method"] == "direct_lingam"
    assert contract["structure_provenance.inference"] == "frequentist"
    assert len(contract["structure_provenance.claim"]) == 64


def test_fit_gcm_accepted_refuses_a_result_with_marks_left():
    names, cols = _static_rows(seed=5, n=80)
    result = _native.discover_pc(names, cols, alpha=0.5, fdr=False, max_cond_size=0, seed=1)
    if result.cpdag_undirected_edges == 0:
        pytest.skip("this draw oriented every PC edge")
    with pytest.raises(Exception, match="cannot coerce"):
        _native.fit_gcm_accepted(names, cols, result)


def test_fitted_simulation_is_seeded_and_states_its_contract():
    names, cols = _static_rows()
    fitted = antecedent.model.fit_gcm(names, cols, [("z", "t"), ("t", "y"), ("z", "y")])

    first = fitted.sample_do({"t": 1.0}, 200, execution=antecedent.Execution(seed=3))
    second = fitted.sample_do({"t": 1.0}, 200, execution=antecedent.Execution(seed=3))
    np.testing.assert_array_equal(first.draws, second.draws)
    assert first.contract["query_kind"] == "Simulate"
    assert first.claim["claim_id"] == second.claim["claim_id"]

    ite_a = fitted.counterfactual_ite("t", "y", 1.0, 0.0, execution=antecedent.Execution(seed=3))
    ite_b = fitted.counterfactual_ite("t", "y", 1.0, 0.0, execution=antecedent.Execution(seed=3))
    np.testing.assert_array_equal(ite_a.unit_effects, ite_b.unit_effects)
    assert ite_a.mean_ite == ite_b.mean_ite
    assert ite_a.contract["query_kind"] == "Simulate"


def test_list_returning_methods_record_their_route():
    names, cols = _static_rows()
    fitted = _native.fit_gcm(names, cols, [("z", "t"), ("t", "y"), ("z", "y")])
    assert fitted.last_route is None
    # The anomaly route scores every row and refuses more rows than `max_units`;
    # `max_units=0` declares the cap as the row count.
    scores = fitted.anomaly_attribution(["y"], max_units=0)
    assert scores
    assert fitted.last_route.contract["query_kind"] == "Attribute"


def test_a_compiled_discovery_is_accepted_through_its_review_gate_with_its_provenance():
    names, cols = _static_rows()
    result = _native.discover_lingam(names, cols, seed=1)
    route = result.route
    graph, provenance = route.accepted_graph(accept_discovered=True)
    assert graph is not None and provenance is not None
    assert provenance["method"] == "direct_lingam"
    assert provenance["inference"] == "frequentist"
    # The accepted structure names the exact program and claim it was discovered under.
    assert provenance["program"] == route.program_identity
    assert provenance["claim"] == result.claim["claim_id"]
    assert provenance["review"] in {"accepted", "accepted_after_review"}
    # Declining to accept pending marks refuses at the review gate instead of guessing.
    with pytest.raises(antecedent.errors.CausalReviewError):
        route.accepted_graph(accept_discovered=False)
