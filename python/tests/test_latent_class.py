"""B4 latent-class (finite mixture) regime effects through the Python facade.

Oracle (the same dataset as ``crates/antecedent-estimate/tests/latent_class_effects.rs``): a
deterministic, exactly balanced two-class design whose classes are so far apart that every
posterior responsibility is exactly 0 or 1 at the generating parameters. The four noise points of
every ``(a, x)`` cell are ``{+0.5, -0.5, +0.25, -0.25}`` (zero sum inside the cell), so the
within-class regression returns the generating coefficients exactly, with residual variance
``0.15625``.

* class ``P`` (24 rows): ``a`` in {0, 1}, ``x`` in {-1, 0, 1}, ``y = a + 0.5 x + e``;
* class ``Q`` (16 rows): ``a`` in {0, 1}, ``x`` in {-1, 1}, ``y = 20 - 2 a - 0.5 x + e``.

So ``pi = (0.6, 0.4)``, ``tau = (1, -2)``, the canonical (ascending ``tau``) order is ``Q, P``
and the mixture-average effect is ``0.6 * 1 + 0.4 * (-2) = -0.2``. Bootstrap calibration is
deliberately NOT measured here.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalUnsupportedError
from antecedent.latent_class import (
    LatentClassRefusal,
    consume_latent_class_artifact,
    latent_class_effects,
)

from _refusal import assert_registered_refusal

NOISE = (0.5, -0.5, 0.25, -0.25)
NOISE_VARIANCE = 0.15625


def build() -> dict[str, np.ndarray]:
    y: list[float] = []
    a: list[float] = []
    x: list[float] = []
    cls: list[int] = []

    def push(arm: float, xv: float, mean: float, label: int) -> None:
        for e in NOISE:
            y.append(mean + e)
            a.append(arm)
            x.append(xv)
            cls.append(label)

    for arm in (0.0, 1.0):
        for xv in (-1.0, 0.0, 1.0):
            push(arm, xv, arm + 0.5 * xv, 0)
    for arm in (0.0, 1.0):
        for xv in (-1.0, 1.0):
            push(arm, xv, 20.0 - 2.0 * arm - 0.5 * xv, 1)
    return {"y": np.array(y), "a": np.array(a), "x": np.array(x), "cls": np.array(cls)}


def fit(d: dict[str, np.ndarray] | None = None, **kwargs):
    d = build() if d is None else d
    kwargs.setdefault("classes", 2)
    kwargs.setdefault("within_class_randomization", True)
    kwargs.setdefault("seed", 11)
    kwargs.setdefault("bootstrap_replicates", 0)
    return latent_class_effects(d["y"], d["a"], {"x": d["x"]}, **kwargs)


def test_b4_latent_recovers_hand_derived_two_class_fixed_point() -> None:
    result = fit()
    q, p = result.classes
    assert q.effect == pytest.approx(-2.0, abs=1e-8)
    assert q.weight == pytest.approx(0.4, abs=1e-12)
    assert q.intercept == pytest.approx(20.0, abs=1e-8)
    assert q.covariate_coefficients["x"] == pytest.approx(-0.5, abs=1e-8)
    assert q.residual_variance == pytest.approx(NOISE_VARIANCE, abs=1e-8)
    assert p.effect == pytest.approx(1.0, abs=1e-8)
    assert p.weight == pytest.approx(0.6, abs=1e-12)
    assert p.intercept == pytest.approx(0.0, abs=1e-8)
    assert p.covariate_coefficients["x"] == pytest.approx(0.5, abs=1e-8)
    assert result.mixture_effect == pytest.approx(-0.2, abs=1e-8)
    assert q.effective_n == pytest.approx(16.0, abs=1e-8)
    assert p.effective_n == pytest.approx(24.0, abs=1e-8)
    assert result.min_effect_gap == pytest.approx(3.0, abs=1e-8)
    assert result.separation == pytest.approx(1.0, abs=1e-12)
    assert result.mixture_effect_se is None and q.effect_se is None and result.bootstrap is None
    assert sorted(result.class_order) == [0, 1]
    assert all(c.raw_index == result.class_order[c.index] for c in result.classes)


def test_b4_latent_responsibilities_are_the_exact_class_indicators() -> None:
    d = build()
    result = fit(d)
    assert result.responsibilities.shape == (len(d["y"]), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0, atol=1e-12)
    # true class P (0) is canonical 1; true class Q (1) is canonical 0.
    canonical = 1 - d["cls"]
    np.testing.assert_array_equal(result.hard_assignment, canonical)
    np.testing.assert_allclose(
        result.responsibilities[np.arange(len(canonical)), canonical], 1.0, atol=1e-12
    )
    assert result.responsibilities_summary.hard_counts == (16, 24)
    assert result.responsibilities_summary.class_means == pytest.approx((0.4, 0.6), abs=1e-12)


def test_b4_latent_log_likelihood_is_non_decreasing_from_a_contaminated_start() -> None:
    d = build()
    labels = [int(1 - c) if i % 4 == 3 else int(c) for i, c in enumerate(d["cls"])]
    result = fit(d, seed=5, initial_labels=labels)
    trace = result.log_likelihood_trace_head
    assert len(trace) >= 3
    for earlier, later in zip(trace, trace[1:], strict=False):
        assert later >= earlier - 1e-9 * (1.0 + abs(earlier))
    assert result.classes[0].effect == pytest.approx(-2.0, abs=1e-8)
    assert result.classes[1].effect == pytest.approx(1.0, abs=1e-8)
    assert result.classes[1].weight == pytest.approx(0.6, abs=1e-12)


def test_b4_latent_label_permutation_gives_identical_canonical_output() -> None:
    d = build()
    truth = [int(c) for c in d["cls"]]
    swapped = [1 - c for c in truth]
    a = fit(d, seed=1, initial_labels=truth)
    b = fit(d, seed=1, initial_labels=swapped)
    for x, y in zip(a.classes, b.classes, strict=True):
        assert x.effect == pytest.approx(y.effect, abs=1e-12)
        assert x.weight == pytest.approx(y.weight, abs=1e-12)
        assert x.intercept == pytest.approx(y.intercept, abs=1e-12)
        assert x.residual_variance == pytest.approx(y.residual_variance, abs=1e-12)
    assert a.mixture_effect == pytest.approx(b.mixture_effect, abs=1e-12)
    # The mapping is retained and records that the raw labels were swapped.
    assert a.class_order == (1, 0)
    assert b.class_order == (0, 1)
    np.testing.assert_allclose(a.responsibilities, b.responsibilities, atol=1e-12)
    np.testing.assert_array_equal(a.hard_assignment, b.hard_assignment)


def test_b4_latent_seeds_and_row_order_do_not_change_the_canonical_result() -> None:
    d = build()
    baseline = fit(d, seed=1)
    for seed in (2, 3, 7, 12345):
        other = fit(d, seed=seed)
        assert other.mixture_effect == pytest.approx(baseline.mixture_effect, abs=1e-9)
        for x, y in zip(baseline.classes, other.classes, strict=True):
            assert x.effect == pytest.approx(y.effect, abs=1e-9)
            assert x.weight == pytest.approx(y.weight, abs=1e-9)
    flipped = {name: column[::-1].copy() for name, column in d.items()}
    rev = fit(flipped, seed=1)
    assert rev.mixture_effect == pytest.approx(baseline.mixture_effect, abs=1e-9)
    np.testing.assert_allclose(baseline.responsibilities, rev.responsibilities[::-1], atol=1e-9)


def test_b4_latent_mixture_effect_is_the_weighted_sum_of_class_effects() -> None:
    result = fit(seed=2)
    weighted = sum(c.weight * c.effect for c in result.classes)
    assert result.mixture_effect == pytest.approx(weighted, abs=1e-12)
    assert sum(c.weight for c in result.classes) == pytest.approx(1.0, abs=1e-12)
    assert weighted == pytest.approx(0.6 - 0.4 * 2.0, abs=1e-8)


def test_b4_latent_bootstrap_is_seeded_label_aligned_and_unmeasured() -> None:
    first = fit(seed=7, bootstrap_replicates=60)
    second = fit(seed=7, bootstrap_replicates=60)
    assert first.to_dict() == second.to_dict(), "the same seed replays bit-identically"
    boot = first.bootstrap
    assert boot is not None and boot.requested == 60
    assert boot.succeeded + boot.failed == 60 and boot.succeeded >= 40
    # Closed-form OLS standard error of tau inside an exactly balanced class.
    closed_q = (NOISE_VARIANCE / (16.0 * 0.25)) ** 0.5
    closed_p = (NOISE_VARIANCE / (24.0 * 0.25)) ** 0.5
    se_q, se_p = first.classes[0].effect_se, first.classes[1].effect_se
    assert se_q is not None and 0.5 * closed_q < se_q < 2.0 * closed_q
    assert se_p is not None and 0.5 * closed_p < se_p < 2.0 * closed_p
    assert first.mixture_effect_se is not None and first.mixture_effect_se > 0.0
    other = fit(seed=8, bootstrap_replicates=60)
    assert other.classes[0].effect_se != se_q
    assert other.classes[0].effect == pytest.approx(first.classes[0].effect, abs=1e-9)
    assert first.calibration == "unmeasured"
    assert first.inference_claim == "point_with_bootstrap_se"
    assert "unmeasured" in first.caveat
    status = {p.name: p.status for p in first.premises}
    assert status["conditional_randomization_within_class"] == "declared"
    assert status["class_weight_and_separation"] == "checked"


def test_b4_latent_weak_class_weight_refuses() -> None:
    with pytest.raises(LatentClassRefusal) as caught:
        fit(min_class_weight=0.5)  # the smaller class has weight 0.4
    assert caught.value.detail == "latent_class.weak_class"
    assert caught.value.reason_code == "population_not_estimable"
    assert isinstance(caught.value, CausalUnsupportedError)
    assert_registered_refusal(caught.value)


def test_b4_latent_single_and_noise_free_regimes_are_degenerate() -> None:
    with pytest.raises(LatentClassRefusal) as caught:
        fit(classes=1)
    assert caught.value.detail == "latent_class.degenerate_class"
    a, x, y = [], [], []
    for arm in (0.0, 1.0):
        for xv in (-1.0, 0.0, 1.0):
            for _ in range(4):
                y.append(arm + 0.5 * xv)
                a.append(arm)
                x.append(xv)
    single = {"y": np.array(y), "a": np.array(a), "x": np.array(x)}
    with pytest.raises(LatentClassRefusal) as caught:
        fit(single, seed=1)
    assert caught.value.detail == "latent_class.degenerate_class"
    assert caught.value.reason_code == "population_not_estimable"


def test_b4_latent_premise_and_configuration_refusals() -> None:
    d = build()
    with pytest.raises(LatentClassRefusal) as caught:
        latent_class_effects(d["y"], d["a"], {"x": d["x"]}, classes=2, seed=1)
    assert caught.value.detail == "latent_class.randomization_not_declared"
    assert caught.value.reason_code == "required_option_missing"
    assert_registered_refusal(caught.value)
    with pytest.raises(LatentClassRefusal) as caught:
        fit(d, classes=5)
    assert caught.value.detail == "latent_class.too_many_classes"
    constant = {**d, "a": np.ones_like(d["a"])}
    with pytest.raises(LatentClassRefusal) as caught:
        fit(constant)
    assert caught.value.detail == "latent_class.treatment_constant"
    assert caught.value.reason_code == "effect_not_identified"
    labels = [int(1 - c) if i % 4 == 3 else int(c) for i, c in enumerate(d["cls"])]
    with pytest.raises(LatentClassRefusal) as caught:
        fit(d, seed=1, initial_labels=labels, max_iterations=1)
    assert caught.value.detail == "latent_class.not_converged"
    assert caught.value.reason_code == "mechanism_fit_not_converged"


def test_b4_latent_artifact_round_trips_with_a_bit_for_bit_refit() -> None:
    result = fit(seed=7, bootstrap_replicates=10)
    again = consume_latent_class_artifact(result.export(), expected=result.identity)
    assert again.to_dict() == result.to_dict()
    np.testing.assert_array_equal(again.responsibilities, result.responsibilities)
    assert again.identity == result.identity
    assert again.log_likelihood_trace_head == result.log_likelihood_trace_head
    assert result.responsibilities_summary.digest == again.responsibilities_summary.digest
    from_mapping = consume_latent_class_artifact(result.export(), expected=result.identity._wire())
    assert from_mapping.identity == result.identity


def test_b4_latent_artifact_refuses_a_resealed_change_against_retained_identity() -> None:
    original = fit()
    d = build()
    d["y"] = d["y"].copy()
    d["y"][0] += 0.125
    resealed = fit(d)
    assert consume_latent_class_artifact(resealed.export()).identity == resealed.identity
    with pytest.raises(LatentClassRefusal) as caught:
        consume_latent_class_artifact(resealed.export(), expected=original.identity)
    assert caught.value.detail == "latent_class.wrong_contract"
    assert caught.value.offending == "data"
    assert caught.value.reason_code == "route_not_supported"
    assert_registered_refusal(caught.value)


def test_b4_latent_artifact_corruption_and_type_errors() -> None:
    result = fit()
    corrupt = bytearray(result.export())
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_latent_class_artifact(bytes(corrupt))
    with pytest.raises(CausalTypeError):
        consume_latent_class_artifact("not bytes")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        fit(within_class_randomization="yes")  # type: ignore[arg-type]
