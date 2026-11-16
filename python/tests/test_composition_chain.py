"""The whole composition chain on one planted dataset, through ``antecedent``.

Messy evidence -> uncertain measurements -> plausible causal structures -> evidence-weighted
estimation -> population-specific conclusions:

1. the exposure ``x`` is observed only through a misclassified coding, with the error rates
   estimated from a validation subsample (``exposure_measurement.MisclassifiedExposure``);
2. ``ExactDagPosterior(prior=StructurePrior(...))`` searches ``(z, x, y)`` on every completed
   table; the engine pools the posteriors once and hands the accepted posterior, with its
   provenance, to every table's study;
3. on every completed table the effect is estimated by AIPW in every structure the pooled
   posterior supports, with the outcome nuisance regularised by external sources whose
   power-prior discount is selected from the table itself (``Aipw(prior_from=SelectAlpha(...))``),
   and the structure-wise effects are mixed by the pooled weights with the between-structure
   variance;
4. the estimate is for a kernel-weighted target population over ``z`` (``target_kernel``), whose
   truth is the closed-form kernel-weighted mean of the planted modified effect
   ``tau(z) = 1.5 + 0.5 z`` on the planted rows; the per-table mixtures are pooled by Rubin's
   rules.

One ``ant.prepare(..., measurement=..., discovery=...)`` and one ``ant.estimate`` run the chain:
no atom loop, no pooling and no provenance is composed by the caller, and the pooled result names
every stage it consumed. The naive pipeline (the recorded column, one assumed structure, no
prior, every row) misses the truth by more than the tolerance.
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
    SelectAlpha,
    TransportKernel,
)

ROWS = 4000
VALIDATION_ROWS = 800
DRAWS = 8
SENSITIVITY, SPECIFICITY = 0.85, 0.9
TOLERANCE = 0.35
CENTRE, BANDWIDTH = 1.0, 0.6


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


def prior_selection():
    names = ("intercept", "coef_z")
    return SelectAlpha(
        sources=[
            ExternalPriorSourceSpec(
                "registry-a",
                (1.6, 0.7),
                (0.6, 0.5),
                ExternalPriorWeight(1.0),
                coefficient_names=names,
            ),
            ExternalPriorSourceSpec(
                "trial-b", (1.9, 0.8), (0.8, 0.6), ExternalPriorWeight(1.0), coefficient_names=names
            ),
        ],
        baseline=([0.0, 0.0], [100.0, 100.0]),
        coefficient_names=names,
    )


def edge_mass(posterior, a, b):
    """Posterior mass of the edge ``a -> b`` (``edge_marginals`` is row-major over ``names``)."""
    return float(posterior.edge_marginals[a * posterior.n_vars + b])


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
    declaration = em.MisclassifiedExposure("x", mx.ExposureError.shared(counts), covariates=("z",))
    target = target_kernel(TransportKernel("gaussian", BANDWIDTH), center={"z": CENTRE})

    # Stages 1 to 4 in one declaration: the measurement imputes the exposure, the search runs on
    # the completed tables and is pooled in the engine, the study mixes the population-targeted
    # effect over the pooled posterior on every table, and the tables are pooled by Rubin.
    prepared = ant.prepare(
        data,
        query=ant.AverageEffect(treatment="x", outcome="y", target_population=target),
        measurement=declaration,
        measurement_draws=DRAWS,
        discovery=ExactDagPosterior(prior=structure_prior()),
        estimator=Aipw(prior_from=prior_selection(), bootstrap=0),
        refute=False,
        execution=ant.Execution(seed=11),
    )
    assert isinstance(prepared, em.PreparedMeasuredAnalysis)
    assert prepared.refusal is None and len(prepared.studies) == DRAWS
    chain = require_success(ant.estimate(prepared))
    assert isinstance(chain, em.MeasuredAnalysis)
    assert abs(chain.effect - truth) < TOLERANCE, (chain.effect, chain.standard_error, truth)

    # The chain matters: the naive pipeline misses the truth by more than the tolerance.
    naive = naive_effect({**data})
    assert abs(naive - truth) > TOLERANCE, (naive, truth)
    assert abs(chain.effect - truth) < abs(naive - truth)

    # The pooled posterior carries the declared structure prior: x -> y holds most of the mass
    # and its provenance states the prior, the pooling and the measurement that produced the
    # tables.
    structure = prepared.structure
    assert structure is not None
    assert structure["method"] == "exact_dag_posterior"
    assert "x->y" in structure["structure_prior"] or "v1->v2" in structure["structure_prior"]
    assert structure["pooled_imputations"] == str(DRAWS)
    assert structure["pooling_method"] == "equal_weight_mixture_of_per_imputation_posteriors"
    assert structure["digest"]
    pooled = ExactDagPosterior(prior=structure_prior()).run(
        imputations=prepared.exposure.completed_tables(data)
    )
    assert pooled.algorithm == "pooled_imputations(exact_dag_posterior)"
    assert math.isclose(sum(pooled.weights), 1.0, abs_tol=1e-9)
    uniform = ExactDagPosterior().run(imputations=prepared.exposure.completed_tables(data))
    assert edge_mass(pooled, 1, 2) > 0.9
    assert edge_mass(pooled, 1, 2) > edge_mass(uniform, 1, 2) + 0.1
    assert pooled.provenance is not None and pooled.provenance["pooled_imputations"] == str(DRAWS)

    # The pooled result names every stage it consumed.
    stages = chain.provenance
    assert stages["measurement"]["declaration_identity"] == prepared.exposure.declaration_identity
    assert stages["measurement"]["data_identity"] == prepared.exposure.data_identity
    assert stages["structure"]["pooled_imputations"] == str(DRAWS)
    text = " ".join(stages["declarations"])
    assert "aipw_outcome_nuisance_prior" in text or "outcome-regression nuisance" in text
    assert "power_prior_alpha_selected" in text and "selected from the target data" in text
    assert "weighted target" in text and "kernel neighbourhood" in text
    assert "structure posterior" in text and f"pooled over {DRAWS} imputations" in text
    target_stage = stages["weighted_target"]
    assert target_stage["declaration"]["family"] == "gaussian"
    assert target_stage["identity"] in text
