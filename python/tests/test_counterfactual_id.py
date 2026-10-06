"""2.2B X8: the effect of treatment on the treated on a bounded ADMG, Python surface.

The Python layer calls the Rust route; these tests check it against an exact
enumerated latent-variable structural model written here, the artifact round
trip, and the reason-coded refusals.
"""

from __future__ import annotations

import itertools

import pytest
from antecedent.counterfactual_id import (
    EffectOnTreated,
    consume_counterfactual_id_artifact,
    prepare_effect_on_treated,
)
from antecedent.errors import CausalError
from antecedent.graph import Admg, Cpdag

LEVELS = {"x": [0.0, 1.0], "m": [0.0, 1.0], "y": [0.0, 1.0]}


def frontdoor() -> Admg:
    return Admg.from_edges(["x", "m", "y"], [("x", "m"), ("m", "y")], [("x", "y")])


def frontdoor_model(shift: float = 0.0):
    """Exogenous states of a front-door model: U on X <-> Y, a response type per variable."""
    u = [(0, 0.6 - shift), (1, 0.4 + shift)]
    x_types = [(0.5, lambda u: u), (0.3, lambda u: 0), (0.2, lambda u: 1)]
    m_types = [(0.2, lambda x: 1), (0.7, lambda x: x), (0.1, lambda x: 0)]
    y_types = [
        (0.3, lambda m, u: m ^ u),
        (0.3, lambda m, u: m & u),
        (0.2, lambda m, u: 1),
        (0.2, lambda m, u: u),
    ]
    for (uv, pu), (px, fx), (pm, fm), (py, fy) in itertools.product(u, x_types, m_types, y_types):
        yield pu * px * pm * py, uv, fx, fm, fy


def observational(model) -> list[float]:
    joint = [0.0] * 8
    for p, uv, fx, fm, fy in model:
        x = fx(uv)
        m = fm(x)
        y = fy(m, uv)
        joint[x * 4 + m * 2 + y] += p
    return joint


def ett_truth(model, active: int, observed: int) -> list[float]:
    """P(Y_{x=active} = y | X = observed) for y = 0, 1."""
    numerators = [0.0, 0.0]
    for p, uv, fx, fm, fy in model:
        if fx(uv) == observed:
            numerators[fy(fm(active), uv)] += p
    total = sum(numerators)
    return [n / total for n in numerators]


def prepare() -> object:
    return prepare_effect_on_treated(
        frontdoor(),
        LEVELS,
        treatment="x",
        active=1.0,
        observed=0.0,
        outcome="y",
        outcome_level=1.0,
    )


def test_frontdoor_ett_matches_the_enumerated_latent_model() -> None:
    prepared = prepare()
    for shift in (0.0, 0.15, -0.2):
        effect = prepared.evaluate(probabilities=observational(frontdoor_model(shift)))
        truth = ett_truth(list(frontdoor_model(shift)), 1, 0)
        assert effect.probability == pytest.approx(truth[1], abs=1e-12)
        for (level, p), t in zip(effect.outcome_distribution, truth, strict=True):
            assert p == pytest.approx(t, abs=1e-12), level
        observed = ett_truth(list(frontdoor_model(shift)), 0, 0)
        assert effect.effect == pytest.approx(truth[1] - observed[1], abs=1e-12)
    # The derivation is decided once: P_x(m) and P_m(x', y).
    assert "P[do(v0=1,)](v1=s0,)" in prepared.derivation
    assert prepared.search["operations_consumed"] > 0


def test_export_and_consume_replay_the_point_and_recompute_it_independently() -> None:
    prepared = prepare()
    effect = prepared.evaluate(probabilities=observational(frontdoor_model(0.1)))
    assert isinstance(effect, EffectOnTreated)
    assert not effect.independently_verified
    consumed = consume_counterfactual_id_artifact(effect.artifact)
    assert consumed.independently_verified
    assert consumed.probability == effect.probability
    assert consumed.effect == effect.effect
    assert consumed.derivation == effect.derivation
    assert consumed.data_digest == effect.data_digest
    assert consumed.search == effect.search
    # A counted law (empirical plug-in) round-trips as well.
    counted = prepared.evaluate(counts=[30, 12, 7, 21, 9, 14, 25, 18])
    again = consume_counterfactual_id_artifact(counted.artifact)
    assert again.probability == counted.probability
    assert again.independently_verified


def test_a_mutated_artifact_is_refused_with_its_reason_code() -> None:
    effect = prepare().evaluate(probabilities=observational(frontdoor_model()))
    artifact = bytearray(effect.artifact)
    refused = 0
    for position in range(0, len(artifact), max(1, len(artifact) // 40)):
        mutated = bytearray(artifact)
        mutated[position] ^= 0x5A
        try:
            consumed = consume_counterfactual_id_artifact(bytes(mutated))
        except CausalError as error:
            assert error.reason_code in {"invalid_argument", "route_not_supported"}
            assert "counterfactual_id." in str(error)
            refused += 1
        else:
            # A flip the decoder normalizes away must replay the same answer.
            assert consumed.probability == effect.probability
            assert consumed.derivation == effect.derivation
    assert refused >= 30


def test_refusals_carry_reason_codes() -> None:
    def refusal(call) -> CausalError:
        with pytest.raises(CausalError) as caught:
            call()
        return caught.value

    bow = Admg.from_edges(["x", "y"], [("x", "y")], [("x", "y")])
    error = refusal(
        lambda: prepare_effect_on_treated(
            bow,
            {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            treatment="x",
            active=1.0,
            observed=0.0,
            outcome="y",
            outcome_level=1.0,
        )
    )
    assert error.reason_code == "route_not_supported"
    assert "counterfactual_id.conflicting_subscripts" in str(error)
    assert "not a proof of non-identifiability" in str(error)

    def ett(graph=None, **overrides):
        arguments = dict(treatment="x", active=1.0, observed=0.0, outcome="y", outcome_level=1.0)
        arguments.update(overrides)
        levels = arguments.pop("levels", LEVELS)
        return lambda: prepare_effect_on_treated(graph or frontdoor(), levels, **arguments)

    cases = [
        (ett(uncertainty="bootstrap"), "estimator_inference_mismatch", "interval_requested"),
        (ett(treatment="nope"), "invalid_argument", "invalid_query"),
        (ett(observed=1.0), "invalid_argument", "invalid_query"),
        (ett(operations=100_001), "route_not_supported", "bounds_exceeded"),
        (ett(operations=2), "transport_budget_cancel", "budget"),
        (
            ett(levels={"x": [0.0, 1.0], "m": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0]}),
            "route_not_supported",
            "bounds_exceeded",
        ),
    ]
    for call, code, detail in cases:
        error = refusal(call)
        assert error.reason_code == code, (code, str(error))
        assert f"counterfactual_id.{detail}" in str(error)
    error = refusal(ett(Cpdag.from_edges(["x", "m", "y"], [])))
    assert error.reason_code == "cell_not_licensed"
    # A law without positivity for this functional.
    error = refusal(
        lambda: prepare().evaluate(probabilities=[0.2, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2])
    )
    assert error.reason_code == "invalid_argument"
    assert "counterfactual_id.positivity_violation" in str(error)


def test_the_binary_complement_answers_where_id_star_stops() -> None:
    """The audit's counterexample: ID* conflicts, P(y | do(x)) is identified, X is binary."""
    names = ["a", "b", "x", "y", "c"]
    graph = Admg.from_edges(
        names,
        [("a", "b"), ("b", "x"), ("b", "c"), ("x", "y"), ("x", "c"), ("y", "c")],
        [("a", "x"), ("a", "y")],
    )
    levels = {name: [0.0, 1.0] for name in names}
    prepared = prepare_effect_on_treated(
        graph, levels, treatment="x", active=1.0, observed=0.0, outcome="y", outcome_level=1.0
    )
    assert "complement(" in prepared.derivation
    # On a law with every variable independent, P_x(y) = P(y) = 0.8 and the
    # complement is 0.8 - P(y = 1, x = 1) = 0.8 * 0.7, so the ETT is 0.8.
    cells = [
        0.25 * (0.3 if x else 0.7) * (0.8 if y else 0.2) * 0.5
        for _a, _b, x, y, _c in itertools.product((0, 1), repeat=5)
    ]
    effect = prepared.evaluate(probabilities=cells)
    assert effect.probability == pytest.approx(0.8, abs=1e-12)
    consumed = consume_counterfactual_id_artifact(effect.artifact)
    assert consumed.independently_verified
    # With a three-level treatment the ID* refusal stands (not a verdict).
    levels["x"] = [0.0, 1.0, 2.0]
    with pytest.raises(CausalError, match="counterfactual_id.conflicting_subscripts") as caught:
        prepare_effect_on_treated(
            graph, levels, treatment="x", active=1.0, observed=0.0, outcome="y", outcome_level=1.0
        )
    assert caught.value.reason_code == "route_not_supported"


def test_prepare_decides_once_and_reports_the_search() -> None:
    prepared = prepare()
    assert prepared.names == ("x", "m", "y")
    search = prepared.search
    assert search["operations_limit"] == 20_000
    assert 0 < search["operations_consumed"] <= search["operations_limit"]
    assert search["depth_reached"] <= search["depth_limit"]
    # Evaluating twice reuses the one derivation.
    first = prepared.evaluate(probabilities=observational(frontdoor_model(0.0)))
    second = prepared.evaluate(probabilities=observational(frontdoor_model(0.2)))
    assert first.derivation == second.derivation == prepared.derivation
    assert first.probability != second.probability


def test_a_budget_stop_shows_its_receipt() -> None:
    with pytest.raises(CausalError) as stopped:
        prepare_effect_on_treated(
            frontdoor(),
            LEVELS,
            treatment="x",
            active=1.0,
            observed=0.0,
            outcome="y",
            outcome_level=1.0,
            operations=2,
        )
    text = str(stopped.value)
    assert stopped.value.reason_code == "transport_budget_cancel"
    assert "receipt: stop search.operations" in text
    assert "unevaluated [" in text


def test_a_cancelled_search_is_a_budget_stop_with_its_receipt() -> None:
    from antecedent.state import CancellationToken

    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalError) as stopped:
        prepare_effect_on_treated(
            frontdoor(),
            LEVELS,
            treatment="x",
            active=1.0,
            observed=0.0,
            outcome="y",
            outcome_level=1.0,
            cancel=token,
        )
    assert stopped.value.reason_code == "transport_budget_cancel"
    assert "counterfactual_id.budget" in str(stopped.value)
    assert "receipt: stop search.cancelled" in str(stopped.value)
