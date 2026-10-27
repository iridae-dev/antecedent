"""Transport across a discovered equivalence class through the Python surface.

Oracles (derived by hand, not by the library):

* a chain ``x -> y -> z`` has no v-structure, so PC returns its Markov equivalence class: the
  three DAGs that create no collider, ``x -> y -> z``, ``x <- y -> z`` and ``x <- y <- z``;
* a collider ``x -> y <- z`` is compelled and is its own class;
* the structural verdict of each chain member for the mean response of ``z`` under ``do(x)``
  with ``y``'s mechanism allowed to differ: the recursive factorization for ``x -> y -> z``, the
  truncated factorization for ``x <- y -> z`` and direct transport for ``x <- y <- z`` (the
  varying mechanism is then not an ancestor of the outcome).

The data are large linear-Gaussian samples with a strict threshold so the discovered skeleton is
the true one. No test asserts that a discovered graph is the true graph in general.
"""

from __future__ import annotations

from unittest.mock import sentinel

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent._artifact_protocol import ExportedArtifact
from antecedent.errors import CausalValueError
from antecedent.transport import discovered

from _refusal import assert_registered_refusal

N = 4000


def chain(seed: int, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + 0.6 * rng.normal(size=n)
    z = 0.8 * y + 0.6 * rng.normal(size=n)
    return {"x": x, "y": y, "z": z}


def collider(seed: int, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.8 * x + 0.8 * z + 0.5 * rng.normal(size=n)
    return {"x": x, "y": y, "z": z}


def chain4(seed: int, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    x = 0.8 * w + 0.6 * rng.normal(size=n)
    y = 0.8 * x + 0.6 * rng.normal(size=n)
    z = 0.8 * y + 0.6 * rng.normal(size=n)
    return {"w": w, "x": x, "y": y, "z": z}


def contract(variables: tuple[str, ...] = ("x", "y", "z"), **kwargs: object) -> sd.StaticContract:
    return sd.StaticContract(variables=variables, alpha=0.001, fdr=None, seed=3, **kwargs)  # type: ignore[arg-type]


QUERY = {
    "outcome": "z",
    "treatment": "x",
    "grid": [0.0, 1.0],
    "selection_targets": ["y"],
    "source_population": "trial",
    "target_population": "clinic",
    "source_experiments": ["x"],
}


def test_a_chain_is_a_class_and_each_member_carries_its_structural_verdict() -> None:
    result = discovered.discover_and_transport(contract(), chain(1), **QUERY)  # type: ignore[arg-type]
    envelope = result.envelope
    assert envelope.kind == "structural_envelope"
    assert envelope.status == "identified_in_every_member"
    assert envelope.enumeration_complete
    assert len(envelope.members) == 3
    assert envelope.identified_members == 3
    assert not envelope.formulas_agree
    assert {(a, b) for a, b, _ in envelope.unresolved} == {("x", "y"), ("y", "z")}
    assert {reason for _, _, reason in envelope.unresolved} == {"undirected_in_equivalence_class"}
    assert envelope.selection_targets == ("y",)
    assert any(text.startswith("causal sufficiency") for text in envelope.assumptions)

    by_edges = {frozenset(m.directed_edges): m for m in envelope.members}
    forward = by_edges[frozenset({("x", "y"), ("y", "z")})]
    fork = by_edges[frozenset({("y", "x"), ("y", "z")})]
    backward = by_edges[frozenset({("y", "x"), ("z", "y")})]
    assert forward.verdict == "identified"
    assert forward.formula == "recursive_factorization"
    assert fork.verdict == "identified"
    assert (backward.formula, backward.rule) == ("direct", "transport.sid.direct")
    # The discovery that ran is returned with the envelope.
    assert result.outcome.structure_class == "cpdag"
    assert len(result.outcome.unresolved_edges) == 2


def test_a_compelled_structure_is_one_member_with_one_formula() -> None:
    # Outcome y is the collider, x is the treatment and z (a parent of y, not an ancestor of x)
    # is the varying mechanism: standardization over z. With outcome z the varying y would not
    # be an ancestor of the outcome and the formula would be direct.
    query = dict(QUERY, outcome="y", selection_targets=["z"])
    envelope = discovered.discover_and_transport(contract(), collider(2), **query).envelope  # type: ignore[arg-type]
    assert envelope.kind == "structural_envelope"
    assert len(envelope.members) == 1
    assert envelope.unresolved == ()
    assert envelope.status == "identified_in_every_member"
    assert envelope.formulas_agree
    assert envelope.members[0].formula == "standardize"


def test_the_envelope_follows_the_class_and_not_the_rows_at_an_unchanged_class() -> None:
    first = discovered.discover_and_transport(contract(), chain(1), **QUERY).envelope  # type: ignore[arg-type]
    again = discovered.discover_and_transport(contract(), chain(1), **QUERY).envelope  # type: ignore[arg-type]
    assert first.identity == again.identity
    other_rows = discovered.discover_and_transport(contract(), chain(7), **QUERY).envelope  # type: ignore[arg-type]
    assert other_rows.status == first.status
    assert [m.directed_edges for m in other_rows.members] == [
        m.directed_edges for m in first.members
    ]
    # Another class at the same shape and query is another envelope.
    other_class = discovered.discover_and_transport(contract(), collider(1), **QUERY).envelope  # type: ignore[arg-type]
    assert other_class.identity != first.identity


def test_a_retained_adapter_serves_the_request_without_rediscovery() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), chain(1))
    from_adapter = discovered.transport_across_class(adapter, **QUERY)  # type: ignore[arg-type]
    one_shot = discovered.discover_and_transport(contract(), chain(1), **QUERY).envelope  # type: ignore[arg-type]
    assert from_adapter.identity == one_shot.identity
    assert from_adapter.structure_identity == adapter.outcome.structure_identity  # type: ignore[union-attr]


def test_a_member_bound_that_stops_the_enumeration_makes_no_every_member_statement() -> None:
    query = dict(
        QUERY, outcome="z", treatment="w", source_experiments=["w"], selection_targets=["x"]
    )
    variables = ("w", "x", "y", "z")
    full = discovered.discover_and_transport(contract(variables), chain4(3), **query).envelope  # type: ignore[arg-type]
    assert len(full.members) == 4
    assert full.enumeration_complete
    bounded = discovered.discover_and_transport(
        contract(variables),
        chain4(3),
        member_bound=2,
        **query,  # type: ignore[arg-type]
    ).envelope
    assert bounded.status == "partial"
    assert not bounded.enumeration_complete
    assert len(bounded.members) == 2
    assert bounded.member_bound == 2
    assert bounded.identity != full.identity


def test_a_pag_chain_is_enumerated_as_its_five_markov_equivalent_mags() -> None:
    fci = contract(algorithm="fci")
    result = discovered.discover_and_transport(fci, chain(1), **QUERY)  # type: ignore[arg-type]
    envelope = result.envelope
    assert result.outcome.structure_class == "pag"
    assert envelope.enumeration_complete
    assert len(envelope.members) == 5
    assert sum(1 for m in envelope.members if m.bidirected_edges) == 2
    assert all(reason == "circle_endpoint" for _, _, reason in envelope.unresolved)


def test_a_temporal_structure_without_a_horizon_is_refused() -> None:
    rng = np.random.default_rng(11)
    n = 300
    a = np.zeros(n)
    b = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal()
        b[t] = 0.7 * a[t - 1] + 0.3 * b[t - 1] + rng.normal()
    adapter = sd.SelectiveDiscovery("temporal")
    adapter.run(sd.TemporalContract(variables=("a", "b"), seed=1), {"a": a, "b": b})
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.transport_across_class(
            adapter,
            outcome="b",
            treatment="a",
            grid=[0.0, 1.0],
            selection_targets=["b"],
            source_experiments=["a"],
        )
    assert_registered_refusal(info.value)
    assert info.value.detail == "discovered_transport.temporal_structure_unsupported"


def test_an_adapter_that_has_not_run_and_undeclared_names_are_refused() -> None:
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.transport_across_class(sd.SelectiveDiscovery("static"), **QUERY)  # type: ignore[arg-type]
    assert info.value.detail == "selective_discovery.nothing_retained"
    assert_registered_refusal(info.value)

    with pytest.raises(CausalValueError):
        discovered.discover_and_transport(
            contract(),
            chain(1),
            **dict(QUERY, selection_targets=["nope"]),  # type: ignore[arg-type]
        )


def test_an_estimate_over_columns_that_are_not_integer_codes_is_refused_precisely() -> None:
    data = chain(2)
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.estimate_discovered_transport(
            contract(),
            data,
            data,
            outcome="z",
            treatment="x",
            grid=[0.0, 1.0],
            selection_targets=["y"],
            source_population="trial",
            target_population="clinic",
        )
    assert_registered_refusal(info.value)
    assert info.value.detail == "discovered_transport.evidence_invalid"


def test_a_forged_estimate_artifact_does_not_replay() -> None:
    with pytest.raises(CausalValueError):
        discovered.verify_discovered_transport_estimate(b"not an artifact")


def test_estimate_factory_preserves_the_native_artifact_and_retained_context() -> None:
    """Wire adaptation retains opaque bytes; it does not certify or reconstruct execution."""
    wire = {
        "levels": [0, 1],
        "members": [],
        "answer": {"kind": "point", "levels": [], "expressions_agree": True},
        "report_identity": "original-report",
        "identity": "original-result",
    }
    artifact = ExportedArtifact(b"opaque native artifact", "discovered_transport_estimate")
    result = discovered._estimate_of(wire, artifact, sentinel.outcome, sentinel.report)
    assert result.export() is artifact
    assert bytes(result.export()) == b"opaque native artifact"
    assert result.outcome is sentinel.outcome
    assert result.report is sentinel.report
    assert result.identity == "original-result"
    assert result.report_identity == "original-report"


TEMPORAL_QUERY = {
    "outcome": "y",
    "treatment": "x",
    "grid": [0.0, 1.0],
    "selection_targets": ["z"],
    "source_population": "trial",
    "target_population": "clinic",
    "source_experiments": ["x"],
    "horizon_steps": 1,
}


def test_a_lagged_dag_is_identified_at_horizon_one_with_autoregressive_history() -> None:
    from antecedent.graph import TemporalDag

    template = TemporalDag.from_lagged_edges(
        ["x", "y", "z"],
        [("x", 1, "y", 0), ("z", 1, "y", 0), ("z", 1, "z", 0)],
    )
    envelope = discovered.transport_temporal_graph(
        template,
        ["x", "y", "z"],
        **TEMPORAL_QUERY,  # type: ignore[arg-type]
    )
    assert envelope.kind == "structural_envelope"
    assert envelope.status == "identified_in_every_member"
    assert envelope.horizon_steps == 1
    assert envelope.max_lag == 1
    assert envelope.members[0].formula == "standardize"
    assert ("x", "y", 1) in envelope.members[0].directed_edges


def test_a_temporal_pag_enumerates_circles_and_its_members_disagree() -> None:
    from antecedent.graph import TemporalPag

    pag = TemporalPag.from_marked_lagged_edges(
        ["x", "y", "z"],
        [("x", 1, "y", 0, "circle", "arrow"), ("z", 1, "y", 0, "circle", "arrow")],
    )
    envelope = discovered.transport_temporal_graph(
        pag,
        ["x", "y", "z"],
        **TEMPORAL_QUERY,  # type: ignore[arg-type]
    )
    assert envelope.status == "members_disagree"
    assert len(envelope.members) == 4
    assert any(m.bidirected_edges for m in envelope.members)
    assert all(reason == "circle_endpoint" for _, _, reason in envelope.unresolved)


def _lagged_continuous_series(seed: int = 31, n: int = 1500) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.6 * z[t - 1] + rng.normal()
    y = 0.3 * rng.normal(size=n)
    y[1:] += 0.9 * x[:-1] + 0.9 * z[:-1]
    return {"x": x, "y": y, "z": z}


def test_a_discovered_temporal_class_is_transported_at_its_declared_horizon() -> None:
    adapter = sd.SelectiveDiscovery("temporal")
    adapter.run(sd.TemporalContract(variables=("x", "y", "z"), seed=1), _lagged_continuous_series())
    envelope = discovered.transport_across_temporal_class(adapter, **TEMPORAL_QUERY)  # type: ignore[arg-type]
    assert envelope.kind == "structural_envelope"
    assert envelope.horizon_steps == 1
    assert envelope.selection_targets == ("z",)
    assert envelope.max_lag >= 1
    assert envelope.members
    # Strong lagged links are in the discovered structure, so every member carries them.
    for member in envelope.members:
        assert ("x", "y", 1) in member.directed_edges
        assert ("z", "y", 1) in member.directed_edges

    # A retained adapter that never ran has no temporal class to transport across.
    with pytest.raises(discovered.DiscoveredTransportRefusal) as unrun:
        discovered.transport_across_temporal_class(
            sd.SelectiveDiscovery("temporal"),
            **TEMPORAL_QUERY,  # type: ignore[arg-type]
        )
    assert unrun.value.detail == "selective_discovery.nothing_retained"
    assert_registered_refusal(unrun.value)


def _binary_lagged_series(steps: int, target: bool, seed: int) -> dict[str, np.ndarray]:
    """``y_t ~ Bernoulli(p(x_{t-1}, z_{t-1}))`` with an autoregressive selected ``z``."""
    rng = np.random.default_rng(seed)
    x = np.zeros(steps)
    y = np.zeros(steps)
    z = np.zeros(steps)
    for t in range(1, steps):
        z_next = 0.6 - 0.4 * z[t - 1] if target else 0.2 + 0.5 * z[t - 1]
        z[t] = float(rng.random() < z_next)
        x[t] = float(rng.random() < 0.5)
        y[t] = float(
            rng.random() < 0.1 + 0.4 * x[t - 1] + 0.3 * z[t - 1] + 0.1 * x[t - 1] * z[t - 1]
        )
    return {"x": x, "y": y, "z": z}


def test_a_temporal_estimate_recovers_the_stationary_target_response_of_a_discovered_class() -> (
    None
):
    source = _binary_lagged_series(30_000, target=False, seed=41)
    target = _binary_lagged_series(30_000, target=True, seed=42)
    estimate = discovered.estimate_discovered_temporal_transport(
        sd.TemporalContract(variables=("x", "y", "z"), seed=1),
        source,
        target,
        outcome="y",
        treatment="x",
        grid=[0.0, 1.0],
        selection_targets=["z"],
        horizon_steps=1,
        source_population="trial",
        target_population="clinic",
    )
    assert estimate.answer.kind == "point"
    # The target chain's stationary law solves pi = 0.6 - 0.4 pi, so by hand
    # E[y | do(x = a)] = (1 - pi) p(a, 0) + pi p(a, 1) with p(a, z) = 0.1 + 0.4a + 0.3z + 0.1az.
    pi = 0.6 / 1.4

    def truth(a: float) -> float:
        return (1.0 - pi) * (0.1 + 0.4 * a) + pi * (0.1 + 0.4 * a + 0.3 + 0.1 * a)

    for (level, low, high), expected in zip(
        estimate.answer.levels, (truth(0.0), truth(1.0)), strict=True
    ):
        assert low == pytest.approx(high)
        assert low == pytest.approx(expected, abs=0.04), level
    verified = discovered.verify_discovered_transport_estimate(bytes(estimate.export()))
    assert verified.answer.kind == "point"
    assert verified.report_identity == estimate.report_identity


def test_a_temporal_query_outside_the_structure_is_refused() -> None:
    from antecedent.graph import TemporalDag

    # Every declared target is in the structure, so the unknown outcome is the refusal's cause.
    template = TemporalDag.from_lagged_edges(["x", "y", "z"], [("x", 1, "y", 0), ("z", 1, "z", 0)])
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.transport_temporal_graph(
            template,
            ["x", "y", "z", "w"],
            **dict(TEMPORAL_QUERY, outcome="w"),  # type: ignore[arg-type]
        )
    assert_registered_refusal(info.value)
    assert info.value.detail == "discovered_transport.temporal_query_invalid"


def test_a_temporal_estimate_over_columns_that_are_not_integer_codes_is_refused() -> None:
    # Strong lagged links put all three variables in the discovered structure.
    rng = np.random.default_rng(5)
    x, z = rng.normal(size=600), rng.normal(size=600)
    y = 0.3 * rng.normal(size=600)
    y[1:] += 0.9 * x[:-1] + 0.9 * z[:-1]
    series = {"x": x, "y": y, "z": z}
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.estimate_discovered_temporal_transport(
            sd.TemporalContract(variables=("x", "y", "z"), seed=1),
            series,
            series,
            outcome="y",
            treatment="x",
            grid=[0.0, 1.0],
            selection_targets=["z"],
            horizon_steps=1,
            source_population="trial",
            target_population="clinic",
        )
    assert_registered_refusal(info.value)
    assert info.value.detail == "discovered_transport.evidence_invalid"


def test_a_zero_member_bound_is_refused_and_keeps_the_discovery_that_ran() -> None:
    with pytest.raises(discovered.DiscoveredTransportRefusal) as info:
        discovered.discover_and_transport(
            contract(),
            chain(1),
            member_bound=0,
            **QUERY,  # type: ignore[arg-type]
        )
    assert_registered_refusal(info.value)
    assert info.value.detail == "discovered_transport.member_bound_zero"
    assert info.value.remedy and "member bound" in info.value.remedy
    # Discovery is not discarded by the refusal of the transport request.
    assert info.value.outcome is not None and info.value.outcome.structure_class == "cpdag"
    assert info.value.report is not None
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), chain(1))
    with pytest.raises(discovered.DiscoveredTransportRefusal) as retained:
        discovered.transport_across_class(adapter, member_bound=0, **QUERY)  # type: ignore[arg-type]
    assert retained.value.detail == "discovered_transport.member_bound_zero"


def binary_collider(rows: int, z_rate: float, seed: int) -> dict[str, np.ndarray]:
    """``x -> y <- z`` with a randomized binary ``x`` and a ``z`` whose rate is the mechanism."""
    rng = np.random.default_rng(seed)
    x = (rng.random(rows) < 0.5).astype(np.float64)
    z = (rng.random(rows) < z_rate).astype(np.float64)
    y = (rng.random(rows) < 0.1 + 0.3 * x + 0.2 * z + 0.15 * x * z).astype(np.float64)
    return {"x": x, "y": y, "z": z}


def test_an_estimate_standardizes_the_source_response_over_the_target_mechanism_and_replays() -> (
    None
):
    source = binary_collider(20_000, 0.3, 5)
    target = binary_collider(20_000, 0.7, 6)
    declared = contract(ci_test="gsquared")
    estimate = discovered.estimate_discovered_transport(
        declared,
        source,
        target,
        outcome="y",
        treatment="x",
        grid=[0.0, 1.0],
        selection_targets=["z"],
        source_population="trial",
        target_population="clinic",
    )

    def response(level: float, z_rate: float) -> float:
        def p_y(z: float) -> float:
            return 0.1 + 0.3 * level + 0.2 * z + 0.15 * level * z

        return (1.0 - z_rate) * p_y(0.0) + z_rate * p_y(1.0)

    # The structural truth in the target: sum over z of P*(z) P(y | do(x), z) = 0.24 and 0.645.
    truth = [response(0.0, 0.7), response(1.0, 0.7)]
    assert truth == pytest.approx([0.24, 0.645], abs=1e-12)
    assert estimate.answer.kind == "point"
    assert len(estimate.members) == 1 and estimate.members[0].status == "estimated"
    assert estimate.answer.interval is None
    for (level, low, high), expected in zip(estimate.answer.levels, truth, strict=True):
        assert low == pytest.approx(expected, abs=0.03), level
        assert high == pytest.approx(low, abs=1e-9)
    assert estimate.answer.contrast == pytest.approx((truth[1] - truth[0],) * 2, abs=0.04)
    # Another target mechanism is another answer: the source response is not the target's.
    same_mechanism = discovered.estimate_discovered_transport(
        declared,
        source,
        binary_collider(20_000, 0.3, 7),
        outcome="y",
        treatment="x",
        grid=[0.0, 1.0],
        selection_targets=["z"],
        source_population="trial",
        target_population="clinic",
    )
    assert same_mechanism.identity != estimate.identity
    assert same_mechanism.answer.levels[0][1] == pytest.approx(response(0.0, 0.3), abs=0.03)

    artifact = bytes(estimate.export())
    verified = discovered.verify_discovered_transport_estimate(artifact)
    assert verified.members == (0,)
    assert verified.answer.kind == "point"
    assert verified.answer.levels == estimate.answer.levels
    assert verified.report_identity == estimate.report_identity
    flipped = bytearray(artifact)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises(CausalValueError):
        discovered.verify_discovered_transport_estimate(bytes(flipped))
    with pytest.raises(CausalValueError):
        discovered.verify_discovered_transport_estimate(artifact[:-1])
