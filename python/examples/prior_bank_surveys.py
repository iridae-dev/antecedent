#!/usr/bin/env python3
"""Survey prior bank: catalog → compose → analyze target (P4E facade demo).

Illustrative domain only — two fake survey posteriors tagged by product/context,
ranked by caller-supplied similarity, composed with power-prior weights, then
transferred into a new target survey. Requires a built extension
(``maturin develop`` in ``python/``).
"""

from __future__ import annotations

import numpy as np

import antecedent


def _survey(
    n: int,
    seed: int,
    *,
    ate: float,
    product: str,
    context: str,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], dict[str, str]]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = ate * t + z + 0.35 * rng.normal(size=n)
    tags = {"product": product, "context": context, "population": "field"}
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")], tags


def _fit_artifact(
    data: dict[str, np.ndarray],
    edges: list[tuple[str, str]],
    *,
    seed: int,
) -> tuple[bytes, float]:
    result = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=96, backend="conjugate"),
        refute=False,
        seed=seed,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    return bytes(result.posterior.artifact), float(result.posterior.effect_mean)


def main() -> None:
    query = causal.AverageEffect(treatment="t", outcome="y")
    edges = [("z", "t"), ("z", "y"), ("t", "y")]

    data_a, _, tags_a = _survey(160, 1, ate=2.0, product="widget", context="launch")
    data_b, _, tags_b = _survey(160, 2, ate=1.5, product="widget", context="retention")
    data_t, _, tags_t = _survey(180, 3, ate=1.8, product="widget", context="expansion")

    art_a, mean_a = _fit_artifact(data_a, edges, seed=11)
    art_b, mean_b = _fit_artifact(data_b, edges, seed=12)

    sources = [
        causal.PriorSource(
            meta=causal.PriorSourceMeta(
                artifact_id="survey_launch",
                estimand=causal.EstimandFingerprint(
                    query_kind="ate", treatment="t", outcome="y"
                ),
                identification="NonparametricallyIdentified",
                tags=tags_a,
                design=(
                    causal.DesignVariable(name="t", role="treatment"),
                    causal.DesignVariable(name="y", role="outcome"),
                    causal.DesignVariable(name="z", role="covariate"),
                ),
            ),
            artifact=art_a,
        ),
        causal.PriorSource(
            meta=causal.PriorSourceMeta(
                artifact_id="survey_retention",
                estimand=causal.EstimandFingerprint(
                    query_kind="ate", treatment="t", outcome="y"
                ),
                identification="NonparametricallyIdentified",
                tags=tags_b,
                design=(
                    causal.DesignVariable(name="t", role="treatment"),
                    causal.DesignVariable(name="y", role="outcome"),
                    causal.DesignVariable(name="z", role="covariate"),
                ),
            ),
            artifact=art_b,
        ),
    ]
    catalog = causal.PriorCatalog.from_sources(sources)
    # Tags that must match exactly (caller similarity handles soft context scores).
    reports = catalog.compatible_with(
        query=query,
        variables=["z", "t", "y"],
        tags={"product": "widget", "population": "field"},
    )
    # Caller-owned similarity (library does not invent domain scores).
    similarity = {"survey_launch": 0.85, "survey_retention": 0.55}
    ranked = catalog.rank(reports, similarity)
    accepted = [r for r in ranked if r.status in ("compatible", "partial")]
    assert accepted, "expected at least one compatible prior source"

    # Hydrate coefficient priors from effect summaries via compose specs.
    # Matching design (same T/Y/Z) → power-prior on 3-coef Gaussian approx.
    specs = [
        causal.ExternalPriorSourceSpec(
            id="survey_launch",
            mean=(0.0, mean_a, 0.0),
            variance=(1.0, 0.25, 1.0),
            weight=causal.ExternalPriorWeight(alpha=1.0),
        ),
        causal.ExternalPriorSourceSpec(
            id="survey_retention",
            mean=(0.0, mean_b, 0.0),
            variance=(1.0, 0.25, 1.0),
            weight=causal.ExternalPriorWeight(alpha=1.0),
        ),
    ]
    # Similarity → mixture weights (normalized leftover stays on baseline).
    w_launch = similarity["survey_launch"]
    w_ret = similarity["survey_retention"]
    w_sum = w_launch + w_ret
    composed = causal.compose_external_priors(
        specs,
        weights=(0.6 * w_launch / w_sum, 0.6 * w_ret / w_sum),
        baseline=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        conflict=causal.ConflictPolicy(p_min=0.05, kl_scale=1.0),
        conflict_signals=[
            {"p_value": 0.4, "kl": 0.05},
            {"p_value": 0.3, "kl": 0.1},
        ],
        # Population tags come from catalog meta (same "field" as target → no transport).
        prior_sources=sources,
        target_population=tags_t["population"],
    )
    # Clear offline conflict for fit; α' already applied.
    prior_for_fit = causal.ComposedPrior(
        mean=composed.mean,
        variance=composed.variance,
        source_ids=composed.source_ids,
        alphas_requested=composed.alphas_requested,
        alphas_applied=composed.alphas_applied,
        mixture_weights=composed.mixture_weights,
        sources=composed.sources,
        conflict=None,
    )

    target = causal.analyze(
        data_t,
        graph=edges,
        query=query,
        inference=causal.Bayesian(
            n_draws=96,
            backend="conjugate",
            prior_from=prior_for_fit,
        ),
        refute="full",
        seed=13,
    )
    assert target.posterior is not None
    assert np.isfinite(target.posterior.effect_mean)
    sens = target.validation.prior_sensitivity
    assert sens is not None and sens.alphas is not None
    assert target.validation.prior_predictive is not None

    print("accepted sources:", [r.artifact_id for r in accepted])
    print(
        "alphas_requested=",
        composed.alphas_requested,
        "alphas_applied=",
        composed.alphas_applied,
    )
    ppc = target.validation.prior_predictive
    print(f"prior_ppc p={ppc.p_value:.3f} observed={ppc.observed:.3f}")
    print(
        f"target effect_mean={target.posterior.effect_mean:.4f} "
        f"sd={target.posterior.effect_sd:.4f}"
    )
    print(f"alpha_sensitivity alphas={sens.alphas} means={sens.effect_means}")


if __name__ == "__main__":
    main()
