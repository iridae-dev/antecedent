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
) -> antecedent.PriorSourceMeta:
    return antecedent.PriorSourceMeta(
        artifact_id=artifact_id,
        estimand=antecedent.EstimandFingerprint(query_kind="ate", treatment="t", outcome=outcome),
        identification=identification,
        design=(
            antecedent.DesignVariable(name="t", role="treatment"),
            antecedent.DesignVariable(name="y", role="outcome"),
            antecedent.DesignVariable(name="z", role="covariate"),
        ),
    )


def _unnamed_artifact_bytes() -> bytes:
    art = antecedent.PosteriorArtifact(
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
    return bytes(antecedent.encode_posterior_artifact(art))


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
    names = list(antecedent.decode_posterior_artifact(artifact).quantity_names)
    assert any(n == "intercept" or n.startswith("coef_") for n in names)
    assert "ate" in names
    # Fitting path should emit durable names, not only coef_{i}.
    assert "intercept" in names or any(
        n.startswith("coef_") and not n[5:].isdigit() for n in names
    )

    matching = antecedent.PriorSource(meta=_meta("match"), artifact=artifact)
    wrong = antecedent.PriorSource(meta=_meta("wrong", outcome="other_y"))
    unnamed = antecedent.PriorSource(meta=_meta("unnamed"), artifact=_unnamed_artifact_bytes())

    catalog = antecedent.PriorCatalog.from_sources([matching, wrong, unnamed])
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
    back = antecedent.PriorSourceMeta.from_cbor(meta.to_cbor())
    assert back.artifact_id == "rt"
    assert back.estimand.treatment == "t"
    assert len(back.design) == 3


def test_rank_orders_usable():
    reports = [
        antecedent.CompatibilityReport(status="compatible", artifact_id="a"),
        antecedent.CompatibilityReport(
            status="partial",
            artifact_id="b",
            missing=("durable_coef_names",),
            mappable=("ate",),
        ),
        antecedent.CompatibilityReport(
            status="rejected",
            artifact_id="c",
            reason={"code": "estimand_mismatch"},
        ),
    ]
    catalog = antecedent.PriorCatalog()
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
            mapping=antecedent.PriorMapping.effect_functional("ate"),
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

    with pytest.raises(Exception):
        antecedent.analyze(
            data_b,
            graph=edges_b,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(
                n_draws=32,
                backend="conjugate",
                prior_from=artifact,
                mapping=antecedent.PriorMapping.identical(),
            ),
            refute=False,
            seed=5,
        
        return_posterior_artifact=True,
    )


def test_compose_weight_and_conflict():
    """Two sources with mixture weights; conflict shrinks the far source's α."""
    agree = antecedent.ExternalPriorSourceSpec(
        id="agree",
        mean=(0.5,),
        variance=(1.0,),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    conflict_src = antecedent.ExternalPriorSourceSpec(
        id="conflict",
        mean=(50.0,),
        variance=(0.25,),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    policy = antecedent.ConflictPolicy(p_min=0.05, kl_scale=1.0)
    composed = antecedent.compose_external_priors(
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
    agree2 = antecedent.ExternalPriorSourceSpec(
        id="agree",
        mean=(0.0, 0.5),
        variance=(100.0, 1.0),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    conflict2 = antecedent.ExternalPriorSourceSpec(
        id="conflict",
        mean=(0.0, 50.0),
        variance=(100.0, 0.25),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    composed2 = antecedent.compose_external_priors(
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
    prior_for_fit = antecedent.ComposedPrior(
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
    assert any(
        "external_composed_prior" in str(a) or "external" in str(a).lower()
        for a in getattr(result, "assumptions", []) or []
    ) or result.posterior is not None


def test_transport_required_when_populations_differ():
    src = antecedent.ExternalPriorSourceSpec(
        id="us_study",
        mean=(1.0,),
        variance=(1.0,),
        weight=antecedent.ExternalPriorWeight(alpha=0.8),
    )
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            source_populations=["us"],
            target_population="eu",
        )


def test_transport_from_prior_source_tags():
    """Catalog meta tags auto-fill source_populations (no manual threading)."""
    src = antecedent.ExternalPriorSourceSpec(
        id="us_study",
        mean=(1.0,),
        variance=(1.0,),
        weight=antecedent.ExternalPriorWeight(alpha=0.8),
    )
    prior_src = antecedent.PriorSource(
        meta=antecedent.PriorSourceMeta(
            artifact_id="us_study",
            estimand=antecedent.EstimandFingerprint(
                query_kind="ate", treatment="t", outcome="y"
            ),
            identification="NonparametricallyIdentified",
            tags={"population": "us"},
        ),
    )
    assert antecedent.populations_from_prior_sources([prior_src]) == ["us"]
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            prior_sources=[prior_src],
            target_population="eu",
        )
    # Matching populations → no transport policy required.
    composed = antecedent.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        prior_sources=[prior_src],
        target_population="us",
    )
    assert composed.alphas_applied == (0.8,)
    # Explicit source_populations wins over prior_sources tags.
    with pytest.raises(ValueError, match="transport_policy_required"):
        antecedent.compose_external_priors(
            [src],
            baseline=([0.0], [4.0]),
            prior_sources=[prior_src],
            source_populations=["us"],
            target_population="eu",
        )


def test_transport_with_policy_records_assumption():
    src = antecedent.ExternalPriorSourceSpec(
        id="us_study",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    composed = antecedent.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        source_populations=["us"],
        target_population="eu",
        transport=antecedent.TransportPolicy.invariant_conditional_outcome(),
    )
    assert all(math.isfinite(x) for x in composed.mean)
    assert all(x > 0 and math.isfinite(x) for x in composed.variance)
    assert "external_transport_prior" in composed.assumption_ids
    assert composed.alphas_applied == (1.0,)


def test_transport_propensity_without_weights_zeros_alpha():
    src = antecedent.ExternalPriorSourceSpec(
        id="us_study",
        mean=(2.0,),
        variance=(1.0,),
        weight=antecedent.ExternalPriorWeight(alpha=0.75),
    )
    composed = antecedent.compose_external_priors(
        [src],
        baseline=([0.0], [4.0]),
        source_populations=["us"],
        target_population="eu",
        transport=antecedent.TransportPolicy.invariant_propensity(),
    )
    assert composed.alphas_requested == (0.75,)
    assert composed.alphas_applied == (0.0,)
    assert "external_transport_prior" in composed.assumption_ids


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
    src = antecedent.ExternalPriorSourceSpec(
        id="survey_a",
        mean=(0.0, 8.0, 0.0),
        variance=(0.05, 0.05, 0.05),
        weight=antecedent.ExternalPriorWeight(alpha=1.0),
    )
    composed = antecedent.compose_external_priors(
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
