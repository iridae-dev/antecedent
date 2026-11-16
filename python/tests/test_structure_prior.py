"""Per-edge inclusion priors and soft tiers on the exact and sampled structure posteriors."""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent.discovery import (
    BackgroundKnowledge,
    EquivalenceClassPosterior,
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    StructurePrior,
)
from antecedent.errors import CausalTypeError, CausalValueError


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
