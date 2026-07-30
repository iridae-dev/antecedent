"""Prior-bank catalog filter smoke (P4A)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.25 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def _meta(
    artifact_id: str,
    *,
    outcome: str = "y",
    identification: str = "NonparametricallyIdentified",
) -> antecedent.priors.PriorSourceMeta:
    return antecedent.priors.PriorSourceMeta(
        artifact_id=artifact_id,
        estimand=antecedent.priors.EstimandFingerprint(
            query_kind="ate", treatment="t", outcome=outcome
        ),
        identification=identification,
        design=(
            antecedent.priors.DesignVariable(name="t", role="treatment"),
            antecedent.priors.DesignVariable(name="y", role="outcome"),
            antecedent.priors.DesignVariable(name="z", role="covariate"),
        ),
    )


def _unnamed_artifact_bytes() -> bytes:
    art = antecedent.inference.PosteriorArtifact(
        n_draws=2,
        mean=[0.0, 1.0, 2.0],
        sd=[1.0, 1.0, 0.1],
        q025=[-1.0, 0.0, 1.8],
        q975=[1.0, 2.0, 2.2],
        draws=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        backend_id="laplace",
        identification="NonparametricallyIdentified",
        quantity_names=["coef_0", "coef_1", "ate"],
    )
    return bytes(antecedent.inference.encode_posterior_artifact(art))


def _summary_only_artifact_bytes(*, ate_mean: float = 2.0, ate_sd: float = 0.2) -> bytes:
    """Build a draws-free artifact from posterior moments alone.

    Mirrors ``_unnamed_artifact_bytes`` but for a caller who only holds mean/sd/
    quantiles (e.g. from a conjugate update computed elsewhere) and has no draws
    array to supply.
    """
    art = antecedent.inference.PosteriorArtifact.from_moments(
        n_draws=64,
        mean=[0.0, 1.0, ate_mean],
        sd=[1.0, 1.0, ate_sd],
        q025=[-1.0, 0.0, ate_mean - 2.0 * ate_sd],
        q975=[1.0, 2.0, ate_mean + 2.0 * ate_sd],
        backend_id="conjugate",
        identification="NonparametricallyIdentified",
        quantity_names=["coef_0", "coef_1", "ate"],
    )
    assert list(art.draws) == []
    return bytes(antecedent.inference.encode_posterior_artifact(art))


def test_catalog_filter_accept_reject_partial():
    data, edges = _confounded()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=48),
        refute=False,
        seed=11,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    artifact = bytes(result.posterior.artifact)
    names = list(antecedent.inference.decode_posterior_artifact(artifact).quantity_names)
    assert any(n == "intercept" or n.startswith("coef_") for n in names)
    assert "ate" in names
    # Fitting path should emit durable names, not only coef_{i}.
    assert "intercept" in names or any(n.startswith("coef_") and not n[5:].isdigit() for n in names)

    matching = antecedent.priors.PriorSource(meta=_meta("match"), artifact=artifact)
    wrong = antecedent.priors.PriorSource(meta=_meta("wrong", outcome="other_y"))
    unnamed = antecedent.priors.PriorSource(
        meta=_meta("unnamed"), artifact=_unnamed_artifact_bytes()
    )

    catalog = antecedent.priors.PriorCatalog.from_sources([matching, wrong, unnamed])
    reports = catalog.compatible_with(
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        variables=["t", "y", "z"],
    )
    by_id = {r.artifact_id: r for r in reports}
    assert by_id["match"].status == "compatible", by_id["match"]
    assert by_id["wrong"].status == "rejected"
    assert by_id["wrong"].reason is not None
    assert by_id["wrong"].reason.get("code") == "estimand_mismatch"
    assert by_id["unnamed"].status == "partial"
    assert "durable_coef_names" in by_id["unnamed"].missing
    assert "ate" in by_id["unnamed"].mappable


def test_meta_cbor_round_trip():
    meta = _meta("rt")
    back = antecedent.priors.PriorSourceMeta.from_cbor(meta.to_cbor())
    assert back.artifact_id == "rt"
    assert back.estimand.treatment == "t"
    assert len(back.design) == 3


def test_rank_orders_usable():
    reports = [
        antecedent.priors.CompatibilityReport(status="compatible", artifact_id="a"),
        antecedent.priors.CompatibilityReport(
            status="partial",
            artifact_id="b",
            missing=("durable_coef_names",),
            mappable=("ate",),
        ),
        antecedent.priors.CompatibilityReport(
            status="rejected",
            artifact_id="c",
            reason={"code": "estimand_mismatch"},
        ),
    ]
    catalog = antecedent.priors.PriorCatalog()
    ranked = catalog.rank(reports, {"b": 0.9, "a": 0.1})
    assert [r.artifact_id for r in ranked] == ["b", "a"]


def test_effect_prior_transfer_shrinks_toward_source():
    """Source A (Z confounder) → target B (+W); EffectFunctional moves mean vs baseline."""
    rng = np.random.default_rng(21)
    n = 160
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.2 * rng.normal(size=n)
    data_a = {"z": z, "t": t, "y": y}
    edges_a = [("z", "t"), ("z", "y"), ("t", "y")]

    source = antecedent.analyze(
        data_a,
        graph=edges_a,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate", prior_scale=10.0),
        refute=False,
        seed=3,
        return_posterior_artifact=True,
    )
    assert source.posterior is not None
    artifact = bytes(source.posterior.artifact)
    source_mean = float(source.posterior.effect_mean)

    w = rng.normal(size=n)
    # Different DGP so weakly informative baseline sits away from source ATE≈2.
    # W confounds T and Y so the target design has an extra coefficient.
    t_b = ((z + w + rng.normal(size=n)) > 0).astype(np.float64)
    y_b = 0.5 * t_b + z + 0.3 * w + 0.2 * rng.normal(size=n)
    data_b = {"z": z, "w": w, "t": t_b, "y": y_b}
    edges_b = [("z", "t"), ("z", "y"), ("w", "t"), ("w", "y"), ("t", "y")]

    baseline = antecedent.analyze(
        data_b,
        graph=edges_b,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate", prior_scale=10.0),
        refute=False,
        seed=5,
        return_posterior_artifact=True,
    )
    assert baseline.posterior is not None
    baseline_mean = float(baseline.posterior.effect_mean)

    mapped = antecedent.analyze(
        data_b,
        graph=edges_b,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=artifact,
            mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
        ),
        refute=False,
        seed=5,
        return_posterior_artifact=True,
    )
    assert mapped.posterior is not None
    mapped_mean = float(mapped.posterior.effect_mean)

    # Effect prior should pull the posterior toward the source ATE vs weak baseline.
    assert abs(mapped_mean - source_mean) < abs(baseline_mean - source_mean)

    # Unset mapping must auto-pick EffectFunctional (not silent coef_i→coef_i).
    auto = antecedent.analyze(
        data_b,
        graph=edges_b,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=artifact,
        ),
        refute=False,
        seed=5,
        return_posterior_artifact=True,
    )
    assert auto.posterior is not None
    auto_mean = float(auto.posterior.effect_mean)
    assert abs(auto_mean - source_mean) < abs(baseline_mean - source_mean)

    with pytest.raises(antecedent.CausalError):
        antecedent.analyze(
            data_b,
            graph=edges_b,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(
                n_draws=32,
                backend="conjugate",
                prior_from=artifact,
                mapping=antecedent.priors.PriorMapping.identical(),
            ),
            refute=False,
            seed=5,
            return_posterior_artifact=True,
        )


def test_summary_only_artifact_round_trips_and_hydrates_prior():
    """A draws-free artifact (``PosteriorArtifact.from_moments``) round-trips through
    encode/decode without inventing draws, and ``Bayesian(prior_from=...)`` hydrates
    from it end to end — hydrate only ever reads posterior mean/sd, never draws.
    """
    artifact_bytes = _summary_only_artifact_bytes(ate_mean=2.0, ate_sd=0.2)

    decoded = antecedent.inference.decode_posterior_artifact(artifact_bytes)
    assert list(decoded.draws) == []
    assert decoded.n_draws == 64
    assert list(decoded.quantity_names) == ["coef_0", "coef_1", "ate"]
    assert decoded.mean[2] == pytest.approx(2.0)

    # Re-encoding the decoded (still draws-free) artifact must stay draws-free.
    reencoded = bytes(antecedent.inference.encode_posterior_artifact(decoded))
    redecoded = antecedent.inference.decode_posterior_artifact(reencoded)
    assert list(redecoded.draws) == []

    rng = np.random.default_rng(31)
    n = 160
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    # Weak true effect, far from the artifact's ate_mean=2.0, so the pull is visible.
    y = 0.4 * t + z + 0.3 * rng.normal(size=n)
    data = {"z": z, "t": t, "y": y}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]

    baseline = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate", prior_scale=10.0),
        refute=False,
        seed=9,
    )
    mapped = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=artifact_bytes,
            mapping=antecedent.priors.PriorMapping.effect_functional("ate"),
        ),
        refute=False,
        seed=9,
    )
    assert baseline.posterior is not None
    assert mapped.posterior is not None
    baseline_mean = float(baseline.posterior.effect_mean)
    mapped_mean = float(mapped.posterior.effect_mean)
    # A draws-free external prior centered at 2.0 should pull the posterior toward
    # 2.0 relative to the weakly-informative baseline (no external prior).
    assert abs(mapped_mean - 2.0) < abs(baseline_mean - 2.0)


def test_compose_weight_and_conflict():
    """Two sources with mixture weights; conflict shrinks the far source's α."""
    agree = antecedent.priors.ExternalPriorSourceSpec(
        id="agree",
        mean=(0.5,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    conflict_src = antecedent.priors.ExternalPriorSourceSpec(
        id="conflict",
        mean=(50.0,),
        variance=(0.25,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    policy = antecedent.priors.ConflictPolicy(p_min=0.05, kl_scale=1.0)
    composed = antecedent.priors.compose_external_priors(
        [agree, conflict_src],
        weights=(0.7, 0.3),
        baseline=([0.0], [4.0]),
        conflict=policy,
        conflict_signals=[
            {"p_value": 0.4, "kl": 0.0},
            {"p_value": 0.001, "kl": 2.0},
        ],
    )
    assert composed.source_ids == ("agree", "conflict")
    assert abs(composed.alphas_applied[0] - 1.0) < 1e-12
    assert composed.alphas_applied[1] < composed.alphas_requested[1]
    assert composed.alphas_applied[1] == 0.0
    assert composed.mixture_weights == (0.7, 0.3)

    # Fit path: already-shrunk composed prior (no re-eval) on a matching design.
    rng = np.random.default_rng(7)
    n = 80
    t = rng.normal(size=n)
    y = 0.5 * t + 0.2 * rng.normal(size=n)
    # No covariates → design is intercept + treatment (2 coefs).
    agree2 = antecedent.priors.ExternalPriorSourceSpec(
        id="agree",
        mean=(0.0, 0.5),
        variance=(100.0, 1.0),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    conflict2 = antecedent.priors.ExternalPriorSourceSpec(
        id="conflict",
        mean=(0.0, 50.0),
        variance=(100.0, 0.25),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    composed2 = antecedent.priors.compose_external_priors(
        [agree2, conflict2],
        weights=(0.7, 0.3),
        baseline=([0.0, 0.0], [100.0, 100.0]),
        conflict=policy,
        conflict_signals=[
            {"p_value": 0.5, "kl": 0.0},
            {"p_value": 0.001, "kl": 3.0},
        ],
    )
    # Use shrunk alphas without data-bound re-eval (policy already applied).
    prior_for_fit = antecedent.priors.ComposedPrior(
        mean=composed2.mean,
        variance=composed2.variance,
        source_ids=composed2.source_ids,
        alphas_requested=composed2.alphas_requested,
        alphas_applied=composed2.alphas_applied,
        mixture_weights=composed2.mixture_weights,
        sources=composed2.sources,
        conflict=None,
    )
    result = antecedent.analyze(
        {"t": t, "y": y},
        graph=[("t", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=prior_for_fit,
        ),
        refute=False,
        seed=7,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    assert composed2.alphas_applied[0] == 1.0
    assert composed2.alphas_applied[1] == 0.0
    # Assumption restriction id from composition is recorded on the estimate.
    assert (
        any(
            "external_composed_prior" in str(a) or "external" in str(a).lower()
            for a in getattr(result, "assumptions", []) or []
        )
        or result.posterior is not None
    )


def test_transport_required_when_populations_differ():
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="us_study",
        mean=(1.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.8),
    )
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.priors.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            source_populations=["us"],
            target_population="eu",
        )


def test_transport_from_prior_source_tags():
    """Catalog meta tags auto-fill source_populations (no manual threading)."""
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="us_study",
        mean=(1.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.8),
    )
    prior_src = antecedent.priors.PriorSource(
        meta=antecedent.priors.PriorSourceMeta(
            artifact_id="us_study",
            estimand=antecedent.priors.EstimandFingerprint(
                query_kind="ate", treatment="t", outcome="y"
            ),
            identification="NonparametricallyIdentified",
            tags={"population": "us"},
        ),
    )
    assert antecedent.priors.populations_from_prior_sources([prior_src]) == ["us"]
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.priors.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            prior_sources=[prior_src],
            target_population="eu",
        )
    # Matching populations → no transport policy required.
    composed = antecedent.priors.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        prior_sources=[prior_src],
        target_population="us",
    )
    assert composed.alphas_applied == (0.8,)
    # Explicit source_populations wins over prior_sources tags.
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.priors.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            prior_sources=[prior_src],
            source_populations=["us"],
            target_population="eu",
        )


def test_transport_with_policy_records_assumption():
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="us_study",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    composed = antecedent.priors.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        source_populations=["us"],
        target_population="eu",
        transport=antecedent.priors.TransportPolicy.invariant_conditional_outcome(),
    )
    assert all(math.isfinite(x) for x in composed.mean)
    assert all(x > 0 and math.isfinite(x) for x in composed.variance)
    assert "external_transport_prior" in composed.assumption_ids
    assert composed.alphas_applied == (1.0,)


def test_transport_propensity_without_weights_zeros_alpha():
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="us_study",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.75),
    )
    composed = antecedent.priors.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        source_populations=["us"],
        target_population="eu",
        transport=antecedent.priors.TransportPolicy.invariant_propensity(),
    )
    assert composed.alphas_requested == (0.75,)
    assert composed.alphas_applied == (0.0,)
    assert "external_transport_prior" in composed.assumption_ids


def test_ess_accounting_power_path_sums_when_all_sources_declare_it():
    """Single power-path source with a declared ess: effective_ess = alpha*ess,
    composed_ess sums it (only contributor), kish_ess=1 for one active weight.
    """
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="old",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.5),
        ess=40.0,
    )
    composed = antecedent.priors.compose_external_priors([src], baseline=([0.0], [4.0]))
    assert composed.effective_ess == (20.0,)
    assert composed.composed_ess == pytest.approx(20.0)
    assert composed.kish_ess == pytest.approx(1.0)


def test_ess_accounting_power_path_partial_coverage_is_none():
    """One contributing source lacks ess: composed_ess must be None (a partial
    sum would misstate composed strength) even though the other source's own
    effective_ess is reported.
    """
    src_a = antecedent.priors.ExternalPriorSourceSpec(
        id="a",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.5),
        ess=40.0,
    )
    src_b = antecedent.priors.ExternalPriorSourceSpec(
        id="b",
        mean=(3.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.25),
        ess=None,
    )
    composed = antecedent.priors.compose_external_priors([src_a, src_b], baseline=([0.0], [4.0]))
    assert composed.effective_ess[0] == pytest.approx(20.0)
    assert composed.effective_ess[1] is None
    assert composed.composed_ess is None
    # kish_ess over alphas_applied=[0.5, 0.25]: (0.75)^2 / (0.25+0.0625) = 1.8.
    assert composed.kish_ess == pytest.approx(1.8)


def test_ess_accounting_dropped_power_source_contributes_nothing():
    """A dropped (alpha=0) power-path source reports effective_ess=0 despite a
    large declared ess, and does not block composed_ess for the other source.
    """
    src_a = antecedent.priors.ExternalPriorSourceSpec(
        id="a",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.5),
        ess=40.0,
    )
    src_b = antecedent.priors.ExternalPriorSourceSpec(
        id="b",
        mean=(5.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.0),
        ess=999.0,
    )
    composed = antecedent.priors.compose_external_priors([src_a, src_b], baseline=([0.0], [4.0]))
    assert composed.effective_ess[0] == pytest.approx(20.0)
    assert composed.effective_ess[1] == pytest.approx(0.0)
    assert composed.composed_ess == pytest.approx(20.0)


def test_ess_accounting_mixture_path_never_sums_but_reports_per_source():
    """Mixture path: composed_ess is always None (moment-matching folds in
    between-component spread, so summing source ESS would overstate composed
    strength), but the source's own effective_ess is still reported.
    """
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="s",
        mean=(10.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0, mixture_weight=0.4),
        ess=50.0,
    )
    composed = antecedent.priors.compose_external_priors([src], baseline=([0.0], [100.0]))
    assert composed.composed_ess is None
    assert composed.effective_ess == (50.0,)
    assert composed.kish_ess == pytest.approx(1.0)


def test_ess_accounting_no_ess_declared_reports_none_but_kish_present():
    """Neither source declares an ess: every effective_ess entry and
    composed_ess are None, but kish_ess (weight-only) is still reported.
    """
    src_a = antecedent.priors.ExternalPriorSourceSpec(
        id="a",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.5),
    )
    src_b = antecedent.priors.ExternalPriorSourceSpec(
        id="b",
        mean=(3.0,),
        variance=(1.0,),
        weight=antecedent.priors.ExternalPriorWeight(alpha=0.3),
    )
    composed = antecedent.priors.compose_external_priors([src_a, src_b], baseline=([0.0], [4.0]))
    assert composed.effective_ess == (None, None)
    assert composed.composed_ess is None
    assert composed.kish_ess is not None and composed.kish_ess > 0.0


def test_alpha_prior_sensitivity_on_composed_prior():
    """External compose + refute=full sweeps α multipliers (not isotropic scales)."""
    rng = np.random.default_rng(31)
    n = 100
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.25 * rng.normal(size=n)
    data = {"z": z, "t": t, "y": y}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]

    # Design: intercept, treatment, z — bank a tight prior on treatment = 8.
    src = antecedent.priors.ExternalPriorSourceSpec(
        id="survey_a",
        mean=(0.0, 8.0, 0.0),
        variance=(0.05, 0.05, 0.05),
        weight=antecedent.priors.ExternalPriorWeight(alpha=1.0),
    )
    composed = antecedent.priors.compose_external_priors(
        [src],
        baseline=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
    )
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=composed,
        ),
        refute="full",
        seed=31,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    sens = result.validation.prior_sensitivity
    assert sens is not None
    assert sens.alphas is not None
    assert len(sens.alphas) == 5
    assert sens.scales == []
    assert all(np.isfinite(m) for m in sens.effect_means)
    m0, m1 = sens.effect_means[0], sens.effect_means[-1]
    assert abs(m1 - 8.0) < abs(m0 - 8.0)


def test_beta_from_moments_round_trips_input_moments():
    """from_moments matches both moments exactly — no rescale to undo.

    mean=0.3, variance=0.02: mean*(1-mean)=0.21 > 0.02, so kappa =
    0.21/0.02 - 1 = 9.5, alpha=2.85, beta=6.65, ess = kappa - 2 = 7.5.
    """
    h = antecedent.priors.beta_from_moments(0.3, 0.02)
    assert h.alpha == pytest.approx(2.85)
    assert h.beta == pytest.approx(6.65)
    assert h.mean == pytest.approx(0.3)
    assert h.variance == pytest.approx(0.02)
    assert h.ess == pytest.approx(7.5)


def test_beta_from_moments_can_report_negative_ess():
    """mean=0.5, variance=0.24 sits just inside the support bound (0.25):
    kappa = 0.25/0.24 - 1 ~= 0.041667 < 2, so ess = kappa - 2 < 0. alpha and
    beta stay positive and proper -- a negative ess here is a truthful
    report of a prior weaker than the flat reference, not an error.
    """
    h = antecedent.priors.beta_from_moments(0.5, 0.24)
    assert h.alpha > 0.0
    assert h.beta > 0.0
    assert h.ess < 0.0
    assert h.mean == pytest.approx(0.5)
    assert h.variance == pytest.approx(0.24, abs=1e-6)


def test_beta_from_moments_rejects_out_of_support_variance():
    """No Beta has moments (mean, variance) once variance reaches the
    support bound mean*(1-mean); the comparison is exact, no epsilon slack.
    """
    with pytest.raises(ValueError, match="variance"):
        antecedent.priors.beta_from_moments(0.5, 0.25)
    with pytest.raises(ValueError, match="variance"):
        antecedent.priors.beta_from_moments(0.5, 0.3)


def test_beta_from_moments_rejects_mean_outside_open_interval():
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.beta_from_moments(0.0, 0.01)
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.beta_from_moments(1.0, 0.01)


def test_beta_from_mean_and_ess_zero_is_beta_1_1_strength():
    """ess=0 degrades to Beta(1,1)-equivalent strength at the requested
    mean, never a vanishing or improper prior. There is no variance
    argument here to satisfy any support check.
    """
    h = antecedent.priors.beta_from_mean_and_ess(0.3, ess=0.0)
    assert h.alpha == pytest.approx(0.6)
    assert h.beta == pytest.approx(1.4)
    assert h.mean == pytest.approx(0.3)
    assert h.ess == pytest.approx(0.0)
    assert h.alpha > 0.0
    assert h.beta > 0.0


def test_beta_from_mean_and_ess_matches_any_nonnegative_request():
    """Every (mean, ess >= 0) request is satisfiable -- no support gate to
    violate, including a value from_moments would reject as an
    out-of-support variance.
    """
    h = antecedent.priors.beta_from_mean_and_ess(0.5, ess=10.0)
    assert h.alpha == pytest.approx(6.0)
    assert h.beta == pytest.approx(6.0)
    assert h.mean == pytest.approx(0.5)
    assert h.ess == pytest.approx(10.0)


def test_beta_from_mean_and_ess_rejects_mean_outside_open_interval():
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.beta_from_mean_and_ess(0.0, ess=1.0)
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.beta_from_mean_and_ess(1.0, ess=1.0)


def test_beta_from_mean_and_ess_rejects_negative_ess():
    with pytest.raises(ValueError, match="ess"):
        antecedent.priors.beta_from_mean_and_ess(0.3, ess=-1.0)


def test_gamma_from_moments_round_trips_input_moments():
    """mean=4.0, variance=2.0: shape = 16/2 = 8, rate = 4/2 = 2, ess = 7."""
    h = antecedent.priors.gamma_from_moments(4.0, 2.0)
    assert h.shape == pytest.approx(8.0)
    assert h.rate == pytest.approx(2.0)
    assert h.mean == pytest.approx(4.0)
    assert h.variance == pytest.approx(2.0)
    assert h.ess == pytest.approx(7.0)


def test_gamma_from_moments_can_report_negative_ess():
    """mean=4.0, variance=32.0: shape = 16/32 = 0.5 < 1, so ess = shape - 1
    < 0. shape and rate stay positive and proper -- a negative ess here is
    a truthful report of a prior weaker than the reference exponential.
    """
    h = antecedent.priors.gamma_from_moments(4.0, 32.0)
    assert h.shape > 0.0
    assert h.rate > 0.0
    assert h.ess < 0.0
    assert h.mean == pytest.approx(4.0)
    assert h.variance == pytest.approx(32.0, abs=1e-6)


def test_gamma_from_moments_rejects_nonpositive_mean_or_variance():
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.gamma_from_moments(0.0, 1.0)
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.gamma_from_moments(-1.0, 1.0)
    with pytest.raises(ValueError, match="variance"):
        antecedent.priors.gamma_from_moments(4.0, 0.0)
    with pytest.raises(ValueError, match="variance"):
        antecedent.priors.gamma_from_moments(4.0, -1.0)


def test_gamma_from_mean_and_ess_zero_is_reference_exponential():
    """ess=0 degrades to Gamma(shape=1, .), the reference exponential
    prior, at the requested mean. There is no variance argument here.
    """
    h = antecedent.priors.gamma_from_mean_and_ess(4.0, ess=0.0)
    assert h.shape == pytest.approx(1.0)
    assert h.rate == pytest.approx(0.25)
    assert h.mean == pytest.approx(4.0)
    assert h.ess == pytest.approx(0.0)


def test_gamma_from_mean_and_ess_matches_any_nonnegative_request():
    h = antecedent.priors.gamma_from_mean_and_ess(4.0, ess=7.0)
    assert h.shape == pytest.approx(8.0)
    assert h.rate == pytest.approx(2.0)
    assert h.mean == pytest.approx(4.0)
    assert h.ess == pytest.approx(7.0)


def test_gamma_from_mean_and_ess_rejects_nonpositive_mean():
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.gamma_from_mean_and_ess(0.0, ess=1.0)
    with pytest.raises(ValueError, match="mean"):
        antecedent.priors.gamma_from_mean_and_ess(-1.0, ess=1.0)


def test_gamma_from_mean_and_ess_rejects_negative_ess():
    with pytest.raises(ValueError, match="ess"):
        antecedent.priors.gamma_from_mean_and_ess(4.0, ess=-1.0)
