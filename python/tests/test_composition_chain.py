"""The whole composition chain on one planted dataset, through ``antecedent``.

Messy evidence -> uncertain measurements -> plausible causal structures -> evidence-weighted
estimation -> population-specific conclusions:

1. the exposure ``x`` is observed only through a misclassified coding, with the error rates
   estimated from a validation subsample (``exposure_measurement.MisclassifiedExposure``); its
   ``realize`` gives exposure draws and ``completed_tables`` one table per draw;
2. ``ExactDagPosterior(prior=StructurePrior(...))`` searches ``(z, x, y)`` on those tables and the
   engine pools the posteriors (``run(imputations=...)``);
3. the effect is estimated by AIPW whose outcome nuisance is regularised by a named composed
   prior (``compose_external_priors`` -> ``Aipw(prior_from=...)``), in every structure the pooled
   posterior supports, and the structure-wise effects are mixed by the pooled weights;
4. the estimate is for a kernel-weighted target population over ``z`` (``target_kernel``), whose
   truth is the closed-form kernel-weighted mean of the planted modified effect
   ``tau(z) = 1.5 + 0.5 z`` on the planted rows.

The stages hand each other only objects the library produced. ``analyze(discovery=...,
imputations=...)`` mixes the effect over the pooled posterior but for all rows only; a weighted
target needs one study per supported structure, so that mixture is composed here from public
pieces (the pooled ``adjacency`` / ``weights``, one ``analyze`` per structure, Rubin pooling with
``exposure_measurement.rubin_pool``). The naive pipeline (the recorded column, one assumed
structure, no prior, every row) misses the truth by more than the tolerance.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
from antecedent import exposure_measurement as em
from antecedent import measurement_exposure as mx
from antecedent.discovery import ExactDagPosterior, StructurePrior
from antecedent.estimators import Aipw
from antecedent.outcomes import require_success
from antecedent.population import target_kernel
from antecedent.priors import (
    ExternalPriorSourceSpec,
    ExternalPriorWeight,
    TransportKernel,
    compose_external_priors,
)

ROWS = 4000
VALIDATION_ROWS = 800
DRAWS = 8
SENSITIVITY, SPECIFICITY = 0.85, 0.9
TOLERANCE = 0.35
CENTRE, BANDWIDTH = 1.0, 0.6
NEGLIGIBLE_MASS = 1e-4
NAMES = ("z", "x", "y")


def expit(v):
    return 1.0 / (1.0 + np.exp(-v))


def tau(z):
    return 1.5 + 0.5 * z


def plant(seed=2026):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=ROWS)
    x = (rng.random(ROWS) < expit(0.6 * z)).astype(float)
    y = 1.0 + 0.5 * z + x * tau(z) + 0.5 * rng.normal(size=ROWS)
    w = np.where(x == 1, rng.random(ROWS) < SENSITIVITY, rng.random(ROWS) >= SPECIFICITY)
    truly = np.arange(VALIDATION_ROWS) % 2 == 0
    coded = np.where(
        truly,
        rng.random(VALIDATION_ROWS) < SENSITIVITY,
        rng.random(VALIDATION_ROWS) >= SPECIFICITY,
    )
    counts = mx.ValidationCounts(
        int(np.sum(truly & coded)),
        int(np.sum(truly & ~coded)),
        int(np.sum(~truly & ~coded)),
        int(np.sum(~truly & coded)),
    )
    return {"z": z, "x": w.astype(float), "y": y}, counts


def closed_form_truth(z):
    weights = np.exp(-0.5 * ((z - CENTRE) / BANDWIDTH) ** 2)
    return float(np.sum(weights * tau(z)) / np.sum(weights))


def structure_prior():
    return StructurePrior(
        edge_probabilities={("x", "y"): 0.5, ("y", "x"): 0.02, ("x", "z"): 0.02, ("y", "z"): 0.02},
        soft_tiers=(("z",), ("x",), ("y",)),
        soft_tier_log_penalty=2.0,
    )


def composed_prior():
    names = ("intercept", "coef_z")
    return compose_external_priors(
        [
            ExternalPriorSourceSpec(
                "registry-a",
                (1.6, 0.7),
                (0.6, 0.5),
                ExternalPriorWeight(1.0),
                coefficient_names=names,
            ),
            ExternalPriorSourceSpec(
                "trial-b", (1.9, 0.8), (0.8, 0.6), ExternalPriorWeight(0.5), coefficient_names=names
            ),
        ],
        baseline=([0.0, 0.0], [100.0, 100.0]),
    )


def decode(mask, n=3):
    """The directed edges of an adjacency mask: bit ``from * (n - 1) + (to if to < from else to - 1)``."""
    return [
        (a, b)
        for a in range(n)
        for b in range(n)
        if a != b and (int(mask) >> (a * (n - 1) + (b if b < a else b - 1))) & 1
    ]


def reaches(edges, start, goal):
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b)
                stack.append(b)
    return goal in seen


def edge_mass(posterior, a, b):
    return sum(
        float(w)
        for mask, w in zip(posterior.adjacency, posterior.weights, strict=True)
        if (a, b) in decode(mask)
    )


def estimate_structure(table, edges, composed, target):
    result = require_success(
        ant.analyze(
            table,
            graph=[(NAMES[a], NAMES[b]) for a, b in edges],
            query=ant.AverageEffect(treatment="x", outcome="y", target_population=target),
            estimator=Aipw(prior_from=composed, bootstrap=0),
            refute=False,
            execution=ant.Execution(seed=11),
        )
    )
    return result.ate, result.estimate.se_analytic, result


def composed_effect(tables, posterior, composed, target):
    """Mix the structure-wise effects of every table by the pooled weights; pool by Rubin."""
    points, variances, sample, unanswered = [], [], None, 0.0
    atoms = [
        (decode(mask), float(w))
        for mask, w in zip(posterior.adjacency, posterior.weights, strict=True)
    ]
    for table in tables:
        parts, dropped = [], 0.0
        for edges, weight in atoms:
            if weight < NEGLIGIBLE_MASS:
                dropped += weight
            elif not reaches(edges, 1, 2):
                parts.append((weight, 0.0, 0.0))
            else:
                try:
                    ate, se, result = estimate_structure(table, edges, composed, target)
                except Exception:
                    dropped += weight
                    continue
                sample = sample or result
                parts.append((weight, ate, se))
        unanswered = dropped
        answered = sum(p[0] for p in parts)
        mixed = sum(w * a for w, a, _ in parts) / answered
        within = sum(w * s for w, _, s in parts) / answered
        between = sum(w * (a - mixed) ** 2 for w, a, _ in parts) / answered
        points.append(mixed)
        variances.append(within**2 + between)
    return em.rubin_pool(points, variances), unanswered, sample


def naive_effect(data):
    result = require_success(
        ant.analyze(
            data,
            graph=[("z", "x"), ("z", "y"), ("x", "y")],
            query=ant.AverageEffect(treatment="x", outcome="y"),
            estimator=Aipw(bootstrap=0),
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    return result.ate


def test_the_chain_recovers_the_closed_form_population_effect_and_names_every_stage():
    data, counts = plant()
    truth = closed_form_truth(data["z"])

    # Stage 1: the measured exposure, declared with a validation subsample.
    declaration = em.MisclassifiedExposure("x", mx.ExposureError.shared(counts), covariates=("z",))
    draws = declaration.realize(data, outcome="y", covariates=("z",), draws=DRAWS, seed=31)
    assert draws.draws.shape == (DRAWS, ROWS) and set(np.unique(draws.draws)) <= {0.0, 1.0}
    assert draws.declaration_identity and draws.data_identity
    tables = draws.completed_tables(data)
    assert len(tables) == DRAWS and list(tables[0]) == list(NAMES)

    # Stage 2: the structure search on the completed tables, pooled in the engine.
    posterior = ExactDagPosterior(prior=structure_prior()).run(imputations=tables)
    assert posterior.algorithm == "pooled_imputations(exact_dag_posterior)"
    assert math.isclose(sum(posterior.weights), 1.0, abs_tol=1e-9)
    uniform = ExactDagPosterior().run(imputations=tables)
    with_prior, without_prior = edge_mass(posterior, 1, 2), edge_mass(uniform, 1, 2)
    assert with_prior > 0.9, with_prior
    assert with_prior > without_prior + 0.1, (with_prior, without_prior)

    # Stages 3 and 4: the evidence-weighted, population-targeted effect over the posterior.
    composed = composed_prior()
    assert composed.coefficient_names == ("intercept", "coef_z")
    target = target_kernel(TransportKernel("gaussian", BANDWIDTH), center={"z": CENTRE})
    pooled, unanswered, sample = composed_effect(tables, posterior, composed, target)
    assert unanswered < 0.05, unanswered
    assert abs(pooled.estimate - truth) < TOLERANCE, (pooled.estimate, pooled.standard_error, truth)

    # The chain matters: the naive pipeline misses the truth by more than the tolerance.
    naive = naive_effect({**data})
    assert abs(naive - truth) > TOLERANCE, (naive, truth)
    assert abs(pooled.estimate - truth) < abs(naive - truth)

    # Provenance: the estimate's result carries the composed prior and the weighted target; the
    # measurement and the structure prior are named by the objects that produced the tables
    # and the posterior.
    text = " ".join(str(a) for a in (sample.assumptions or []))
    assert "aipw_outcome_nuisance_prior" in text or "outcome-regression nuisance" in text
    assert sample.weighted_target is not None
    assert sample.weighted_target.declaration["family"] == "gaussian"
    assert sample.weighted_target.identity in text
    assert sample.provenance["weighted_target"]["identity"] == sample.weighted_target.identity
    assert posterior.n_graphs >= 2
