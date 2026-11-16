"""Per-edge inclusion priors and soft tiers on the exact and sampled structure posteriors."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.discovery import (
    BackgroundKnowledge,
    CiScreenedPosterior,
    DbnPosterior,
    EquivalenceClassPosterior,
    ExactDagPosterior,
    OrderMcmc,
    ScreenedExactMarginals,
    StructureMcmc,
    StructurePrior,
)
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success


def _chain(rows: int = 300, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    b = 0.7 * a + rng.normal(size=rows)
    c = 0.6 * b + rng.normal(size=rows)
    return {"a": a, "b": b, "c": c}


def _matrix(posterior: object, names: list[str]) -> np.ndarray:
    n = len(names)
    return np.asarray(posterior.edge_marginals).reshape(n, n)  # type: ignore[attr-defined]


def test_the_prior_is_validated_when_it_is_built() -> None:
    with pytest.raises(CausalValueError):
        StructurePrior(edge_probabilities={("a", "b"): 1.0})
    with pytest.raises(CausalValueError):
        StructurePrior(edge_probabilities={("a", "b"): 0.0})
    with pytest.raises(CausalValueError):
        StructurePrior(edge_probabilities={("a", "a"): 0.3})
    with pytest.raises(CausalValueError):
        StructurePrior(edge_probabilities=[("a", "b", 0.3), ("a", "b", 0.4)])
    with pytest.raises(CausalTypeError):
        StructurePrior(edge_probabilities={("a", "b"): "high"})  # type: ignore[dict-item]
    with pytest.raises(CausalValueError):
        StructurePrior(default_edge_probability=1.5)
    with pytest.raises(CausalValueError):
        StructurePrior(soft_tiers=(("a",), ("b",)))
    with pytest.raises(CausalValueError):
        StructurePrior(soft_tier_log_penalty=1.0)
    with pytest.raises(CausalValueError):
        StructurePrior(
            soft_tiers=(("a",), ("b",)), soft_tier_log_penalty=1.0, soft_tier_odds_factor=0.5
        )
    with pytest.raises(CausalValueError):
        StructurePrior(soft_tiers=(("a",), ("b",)), soft_tier_log_penalty=-1.0)


def test_an_odds_factor_is_the_log_penalty_of_minus_its_log() -> None:
    by_factor = StructurePrior(soft_tiers=(("a",), ("b",)), soft_tier_odds_factor=0.25)
    kwargs = by_factor._native_kwargs()
    assert kwargs["soft_tier_penalty"] == pytest.approx(math.log(4.0))
    assert StructurePrior()._native_kwargs() == {}


def test_a_mapping_and_the_equivalent_triples_state_one_prior() -> None:
    mapped = StructurePrior(edge_probabilities={("a", "b"): 0.3, ("b", "c"): 0.6})
    listed = StructurePrior(edge_probabilities=[("a", "b", 0.3), ("b", "c", 0.6)])
    assert mapped == listed
    assert hash(mapped) == hash(listed)


def test_a_per_edge_prior_moves_the_exact_marginals_toward_its_belief() -> None:
    data = _chain(rows=40)
    names = ["a", "b", "c"]
    plain = ExactDagPosterior().run(data)
    doubtful = ExactDagPosterior(
        prior=StructurePrior(edge_probabilities={("a", "b"): 0.02, ("b", "a"): 0.02})
    ).run(data)
    base, shifted = _matrix(plain, names), _matrix(doubtful, names)
    assert shifted[0, 1] + shifted[1, 0] < base[0, 1] + base[1, 0]


def test_an_unknown_variable_in_the_prior_is_refused() -> None:
    with pytest.raises(Exception, match="unknown variable"):
        ExactDagPosterior(prior=StructurePrior(edge_probabilities={("a", "zzz"): 0.3})).run(
            _chain()
        )


def test_a_soft_tier_shifts_mass_toward_the_declared_order_without_forbidding() -> None:
    data = _chain(rows=30)
    names = ["a", "b", "c"]
    free = _matrix(ExactDagPosterior().run(data), names)
    soft = _matrix(
        ExactDagPosterior(
            prior=StructurePrior(soft_tiers=(("a",), ("b",), ("c",)), soft_tier_log_penalty=4.0)
        ).run(data),
        names,
    )
    assert soft[1, 0] < free[1, 0]
    assert soft[1, 0] > 0.0


def test_samplers_and_class_posteriors_carry_the_prior() -> None:
    data = _chain()
    prior = StructurePrior(
        edge_probabilities={("a", "b"): 0.5},
        default_edge_probability=0.2,
        soft_tiers=(("a",), ("b", "c")),
        soft_tier_odds_factor=0.3,
    )
    names = ["a", "b", "c"]
    exact = _matrix(ExactDagPosterior(prior=prior).run(data), names)
    for sampler in (
        StructureMcmc(n_chains=4, n_warmup=500, n_draws=4000, prior=prior),
        OrderMcmc(
            n_chains=4,
            n_warmup=500,
            n_draws=4000,
            require_diagnostics_gate=False,
            prior=prior,
        ),
    ):
        got = _matrix(sampler.run(data), names)
        assert np.max(np.abs(got - exact)) < 0.08
    classes = EquivalenceClassPosterior(base=ExactDagPosterior(prior=prior)).run(data)
    assert classes.atom_kind == "Cpdag"


def test_hard_background_tiers_are_a_separate_statement() -> None:
    # The soft tiers state a prior; BackgroundKnowledge tiers forbid. They are different types.
    assert not hasattr(StructurePrior(), "tiers")
    assert BackgroundKnowledge(tiers=(("a",), ("b",))).tiers == (("a",), ("b",))


def test_the_wire_carries_the_prior_only_when_one_is_set() -> None:
    assert "prior" not in ExactDagPosterior()._wire()
    assert "prior" not in StructureMcmc()._wire()
    wire = StructureMcmc(prior=StructurePrior(default_edge_probability=0.2))._wire()
    assert wire["prior"] == {"default_edge_probability": 0.2}


def _two_chains(rows: int = 400, seed: int = 9) -> dict[str, np.ndarray]:
    """``a -> b -> c`` and ``d -> e`` beside three noise variables."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    b = 0.9 * a + rng.normal(size=rows)
    c = 0.9 * b + rng.normal(size=rows)
    d = rng.normal(size=rows)
    e = 0.9 * d + rng.normal(size=rows)
    noise = {f"n{i}": rng.normal(size=rows) for i in range(3)}
    return {"a": a, "b": b, "c": c, "d": d, "e": e, **noise}


def test_the_screened_route_selects_the_focus_neighbourhood_and_states_its_condition() -> None:
    data = _two_chains()
    result = ScreenedExactMarginals(focus=("b",), max_variables=3).run(data)
    assert result.names == ["a", "b", "c"]
    assert result.conditional_on_screen is not None
    assert "conditional on the CI screen" in result.conditional_on_screen
    matrix = np.asarray(result.edge_marginal_matrix())
    assert matrix[0, 1] + matrix[1, 0] > 0.9
    assert matrix[1, 2] + matrix[2, 1] > 0.9
    assert matrix[0, 2] + matrix[2, 0] < 0.5


def test_the_screened_route_refuses_what_the_exact_route_cannot_hold() -> None:
    with pytest.raises((CausalValueError, CausalTypeError)):
        ScreenedExactMarginals(focus=("b",), max_variables=16)
    with pytest.raises(CausalValueError):
        ScreenedExactMarginals(focus=())
    with pytest.raises(CausalValueError):
        ScreenedExactMarginals(focus=("b",), in_probability=1.0)
    with pytest.raises(Exception, match="unknown focus"):
        ScreenedExactMarginals(focus=("zzz",), max_variables=3).run(_two_chains())


def test_a_callers_edge_probability_is_kept_by_the_screened_route() -> None:
    data = _two_chains()
    plain = ScreenedExactMarginals(focus=("b",), max_variables=3).run(data)
    stated = ScreenedExactMarginals(
        focus=("b",),
        max_variables=3,
        prior=StructurePrior(edge_probabilities={("c", "b"): 0.001}),
    ).run(data)
    assert stated.names == plain.names
    assert stated.edge_marginal_matrix()[2][1] < plain.edge_marginal_matrix()[2][1]


def test_ci_screened_and_dbn_posteriors_carry_the_prior() -> None:
    prior = StructurePrior(default_edge_probability=0.1)
    assert CiScreenedPosterior(prior=prior)._wire()["prior"] == {"default_edge_probability": 0.1}
    assert DbnPosterior(prior=prior)._wire()["prior"] == {"default_edge_probability": 0.1}
    assert "prior" not in CiScreenedPosterior()._wire()
    assert "prior" not in DbnPosterior()._wire()
    data = _two_chains()
    sparse = CiScreenedPosterior(prior=StructurePrior(default_edge_probability=0.02)).run(data)
    assert sparse.n_vars == len(data)


def _pair(rows: int = 400, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=rows)
    return {"t": t, "y": 0.8 * t + rng.normal(size=rows)}


def _pair_prior_weights(plain: object, p_ty: float, p_yt: float) -> list[float]:
    """The uniform posterior's weights reweighted by hand: bit 0 is t -> y, bit 1 is y -> t."""
    raw = []
    for mask, weight in zip(plain.adjacency, plain.weights, strict=True):  # type: ignore[attr-defined]
        factor = (p_ty if mask & 1 else 1.0 - p_ty) * (p_yt if mask & 2 else 1.0 - p_yt)
        raw.append(weight * factor)
    total = sum(raw)
    return [weight / total for weight in raw]


def _effect(data: object, discovery: object) -> tuple[float, float]:
    result = require_success(
        antecedent.analyze(
            data,
            discovery=discovery,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256, prior_scale=1.0e6),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.posterior is not None
    return float(result.posterior.effect_mean), float(result.posterior.unidentified_mass)


def test_a_per_edge_prior_through_analyze_is_the_engine_posterior_and_moves_the_effect() -> None:
    data = _pair()
    plain = ExactDagPosterior().run(data)
    prior = StructurePrior(edge_probabilities={("t", "y"): 0.99, ("y", "t"): 0.001})
    searched = ExactDagPosterior(prior=prior)

    # The engine posterior is the uniform posterior reweighted by the hand-computed factors.
    direct = searched.run(data)
    expected = _pair_prior_weights(plain, 0.99, 0.001)
    assert list(direct.adjacency) == list(plain.adjacency)
    for got, want in zip(direct.weights, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)

    # Handing the search to analyze runs it under the prior: the answer is the answer on the
    # engine's own posterior.
    through_analyze = _effect(data, searched)
    on_direct = _effect(data, direct)
    assert through_analyze == pytest.approx(on_direct, abs=1e-9)

    # Nearly all the mass is now on t -> y, so the effect is the regression slope; the uniform
    # posterior splits it between the two score-equivalent orientations, which shows in the
    # effect or in the mass the study cannot identify.
    slope = float(np.polyfit(data["t"], data["y"], 1)[0])
    effect, unidentified = through_analyze
    assert effect == pytest.approx(slope, rel=0.05)
    assert unidentified < 1e-2
    uniform_effect, uniform_unidentified = _effect(data, ExactDagPosterior())
    shift = abs(uniform_effect - effect) / abs(slope) + abs(uniform_unidentified - unidentified)
    assert shift > 0.3


def test_a_soft_tier_through_analyze_is_the_engine_posterior() -> None:
    data = _pair()
    searched = ExactDagPosterior(
        prior=StructurePrior(soft_tiers=(("t",), ("y",)), soft_tier_log_penalty=6.0)
    )
    direct = searched.run(data)
    plain = ExactDagPosterior().run(data)
    # A present edge from the later tier into the earlier one is charged exp(-6).
    raw = [
        weight * (np.exp(-6.0) if mask & 2 else 1.0)
        for mask, weight in zip(plain.adjacency, plain.weights, strict=True)
    ]
    for got, want in zip(direct.weights, [w / sum(raw) for w in raw], strict=True):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)
    assert _effect(data, searched) == pytest.approx(_effect(data, direct), abs=1e-9)


def test_pooling_over_imputations_is_the_hand_mixture_of_the_per_imputation_posteriors() -> None:
    tables = [_chain(rows=120, seed=seed) for seed in (11, 12, 13)]
    search = ExactDagPosterior()
    singles = [search.run(table) for table in tables]
    pooled = search.run(imputations=tables)

    mixture: dict[int, float] = {}
    for posterior in singles:
        for mask, weight in zip(posterior.adjacency, posterior.weights, strict=True):
            mixture[mask] = mixture.get(mask, 0.0) + weight / 3.0
    assert pooled.n_graphs == len(mixture)
    for mask, weight in zip(pooled.adjacency, pooled.weights, strict=True):
        assert weight == pytest.approx(mixture[mask], abs=1e-12)
    names = ["a", "b", "c"]
    mean_marginals = sum(_matrix(post, names) for post in singles) / 3.0
    assert np.allclose(_matrix(pooled, names), mean_marginals, atol=1e-12)
    assert pooled.algorithm == "pooled_imputations(exact_dag_posterior)"

    # The pooled posterior is a graph posterior like any other: analyze mixes over it, and the
    # answer equals the one on the same atoms and weights supplied by hand.
    by_hand = antecedent.discovery.GraphPosterior.from_atoms(
        names, [mixture[mask] for mask in pooled.adjacency], list(pooled.adjacency)
    )
    query = antecedent.AverageEffect(treatment="b", outcome="c")

    def effect(discovery: object) -> float:
        result = require_success(
            antecedent.analyze(
                tables[0],
                discovery=discovery,
                query=query,
                inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
                refute=False,
                execution=antecedent.Execution(seed=1),
            )
        )
        assert result.posterior is not None
        return float(result.posterior.effect_mean)

    assert effect(pooled) == pytest.approx(effect(by_hand), abs=1e-6)


def test_imputation_uncertainty_widens_the_structure_posterior() -> None:
    # One imputation links a and b strongly, the other leaves them independent: the pooled
    # adjacency mass is the mean of the two, between what either posterior says alone.
    rng = np.random.default_rng(3)
    a = rng.normal(size=300)
    linked = {"a": a, "b": 0.9 * a + rng.normal(size=300)}
    unlinked = {"a": a, "b": rng.normal(size=300)}
    search = ExactDagPosterior()

    def adjacency(posterior: object) -> float:
        edge = _matrix(posterior, ["a", "b"])
        return float(edge[0, 1] + edge[1, 0])

    pooled = adjacency(search.run(imputations=[linked, unlinked]))
    high, low = adjacency(search.run(linked)), adjacency(search.run(unlinked))
    assert high > 0.95
    assert low < 0.5
    assert pooled == pytest.approx((high + low) / 2.0, abs=1e-12)
    assert low < pooled < high


def test_pooling_pools_exact_marginals_above_six_variables() -> None:
    def table(seed: int) -> dict[str, np.ndarray]:
        gen = np.random.default_rng(seed)
        cols = {"x0": gen.normal(size=150)}
        for i in range(1, 7):
            cols[f"x{i}"] = 0.7 * cols[f"x{i - 1}"] + gen.normal(size=150)
        return cols

    tables = [table(1), table(2)]
    search = ExactDagPosterior()
    singles = [search.run(item) for item in tables]
    pooled = search.run(imputations=tables)
    first, second = (np.array(single.edge_marginal_matrix()) for single in singles)
    assert np.allclose(pooled.edge_marginal_matrix(), (first + second) / 2.0, atol=1e-12)
    base = singles[0].log_evidence
    log_mean = np.log(np.mean([np.exp(single.log_evidence - base) for single in singles]))
    assert pooled.log_evidence == pytest.approx(base + log_mean, rel=1e-9)
    assert pooled.pooled_over_imputations == 2


def test_pooling_refuses_what_it_cannot_state() -> None:
    data = _chain()
    other = {"a": data["a"], "b": data["b"], "z": data["c"]}
    search = ExactDagPosterior()
    with pytest.raises(CausalValueError, match="same variables"):
        search.run(imputations=[data, other])
    with pytest.raises(CausalValueError, match="not both"):
        search.run(data, imputations=[data])
    with pytest.raises(CausalValueError, match="at least one"):
        search.run(imputations=[])
    with pytest.raises(CausalTypeError):
        search.run(imputations=data)
    with pytest.raises(CausalValueError):
        search.run()


def test_every_posterior_search_pools_and_analyze_takes_the_imputations() -> None:
    tables = [_pair(rows=300, seed=seed) for seed in (1, 2, 3)]
    sampled = StructureMcmc(n_chains=2, n_warmup=100, n_draws=300).run(imputations=tables)
    assert sampled.algorithm is not None
    assert sampled.algorithm.startswith("pooled_imputations")
    assert sum(sampled.weights) == pytest.approx(1.0, abs=1e-9)
    ordered = OrderMcmc(n_chains=2, n_warmup=100, n_draws=300, require_diagnostics_gate=False).run(
        imputations=tables
    )
    assert ordered.algorithm is not None
    assert ordered.algorithm.startswith("pooled_imputations")

    # The completed tables flow into the study with no manual step: the study searches each,
    # pools, and mixes the effect over the pooled posterior.
    through_analyze = require_success(
        antecedent.analyze(
            tables[0],
            discovery=ExactDagPosterior(),
            imputations=tables,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256, prior_scale=1.0e6),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert through_analyze.posterior is not None
    by_hand = _effect(tables[0], ExactDagPosterior().run(imputations=tables))
    got = (
        float(through_analyze.posterior.effect_mean),
        float(through_analyze.posterior.unidentified_mass),
    )
    assert got == pytest.approx(by_hand, abs=1e-9)
