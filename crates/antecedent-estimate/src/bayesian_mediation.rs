//! Gaussian posterior mechanism products for linear temporal mediation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, IdentificationStatus, Lag, MediationContrast, MediationQuery,
};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_prob::{BayesLikelihood, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use antecedent_stats::CompiledDesign;

use crate::serial_dependence::{DEPENDENCE_ASSUMPTION_ID, DependenceScope, SerialDependence};
use crate::{
    BayesianGComputationAte, CausalPosterior, EstimationError, OverlapPolicy,
    PreparedBayesianProblem,
};

/// Prepare the mediator and outcome mechanisms on the same lag-aligned rows.
/// Innovations independent across the two mechanisms and independent coefficient
/// priors give a posterior product; no prior changes the supplied identification
/// status.
///
/// Rows are time-ordered, so each mechanism's Gaussian likelihood is tempered by
/// the long-run-variance ratio of its path combinations
/// ([`crate::serial_dependence`], [`DependenceScope::MediationPaths`]): the
/// treatment slope `a` of the mediator mechanism, and the direct `c'`, mediated
/// `b` and total `c' + â b` gradients of the outcome mechanism (`â` the OLS
/// mediator slope).
pub fn prepare_temporal_mediation(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    ctx: &ExecutionContext,
) -> Result<[PreparedBayesianProblem; 2], EstimationError> {
    prepare_temporal_mediation_adjusted(data, estimand, query, &[], ctx)
}

/// Prepare both mechanisms with the same graph-derived baseline covariates.
pub fn prepare_temporal_mediation_adjusted(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    adjustment: &[LaggedColumn],
    ctx: &ExecutionContext,
) -> Result<[PreparedBayesianProblem; 2], EstimationError> {
    query.validate()?;
    if !estimand.method_kind().is_ok_and(antecedent_expr::EstimandMethod::is_temporal_mediation)
        || estimand.mediators.len() != 1
    {
        return Err(EstimationError::unsupported(
            "Bayesian temporal mediation requires one identified mediator",
        ));
    }
    let mediator = estimand.mediators[0];
    let treatment_lag = query.horizons.first().copied().filter(|&h| h >= 1).unwrap_or(1);
    let mut columns = vec![
        LaggedColumn { variable: query.treatment, lag: Lag::from_raw(treatment_lag) },
        LaggedColumn { variable: mediator, lag: Lag::CONTEMPORANEOUS },
        LaggedColumn { variable: query.outcome, lag: Lag::CONTEMPORANEOUS },
    ];
    columns.extend_from_slice(adjustment);
    let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(1);
    let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
    let mut workspace = LaggedSampleWorkspace::default();
    let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
    let active = crate::adjustment::intervention_f64(&query.active)?;
    let control = crate::adjustment::intervention_f64(&query.control)?;
    if active.to_bits() == control.to_bits() || (active == 0.0 && control == 0.0) {
        return Err(EstimationError::unsupported("mediation contrast must have distinct levels"));
    }
    let build = |outcome: &[f64],
                 covs: &[(antecedent_core::VariableId, &[f64])]|
     -> Result<_, EstimationError> {
        Ok(PreparedBayesianProblem {
            design: CompiledDesign::linear_adjustment(sample.column(0), covs, outcome, &[])?,
            method: estimand.method.clone(),
            adjustment_set: Arc::from([]),
            active,
            control,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        })
    };
    let covs: Vec<_> = adjustment
        .iter()
        .enumerate()
        .map(|(i, column)| (column.variable, sample.column(3 + i)))
        .collect();
    let mut outcome_covs = vec![(mediator, sample.column(1))];
    outcome_covs.extend_from_slice(&covs);
    let mediator_mechanism = build(sample.column(1), &covs)?;
    let outcome_mechanism = build(sample.column(2), &outcome_covs)?;
    // Mediator design [1 | T | Z…]: the path slope `a` is column 1. Outcome design
    // [1 | T | M | Z…]: `c'` is column 1 and `b` column 2.
    let unit = |p: usize, entries: &[(usize, f64)]| -> Arc<[f64]> {
        let mut c = vec![0.0; p];
        for &(i, v) in entries {
            c[i] = v;
        }
        Arc::from(c)
    };
    let a_hat = ols_coefficient(&mediator_mechanism.design, 1)?;
    let pm = mediator_mechanism.design.ncols;
    let po = outcome_mechanism.design.ncols;
    let mediator_paths = DependenceScope::MediationPaths(Arc::from([unit(pm, &[(1, 1.0)])]));
    let outcome_paths = DependenceScope::MediationPaths(Arc::from([
        unit(po, &[(1, 1.0)]),
        unit(po, &[(2, 1.0)]),
        unit(po, &[(1, 1.0), (2, a_hat)]),
    ]));
    Ok([
        mediator_mechanism
            .with_serial_dependence(SerialDependence::LongRunTempering(mediator_paths)),
        outcome_mechanism.with_serial_dependence(SerialDependence::LongRunTempering(outcome_paths)),
    ])
}

/// OLS coefficient `index` of `design` (the plug-in path slope for a tempering gradient).
fn ols_coefficient(design: &CompiledDesign, index: usize) -> Result<f64, EstimationError> {
    use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
    let fit = FaerBackend.least_squares(
        &design.matrix[..design.nrows * design.ncols],
        design.nrows,
        design.ncols,
        &design.outcome,
        &mut LeastSquaresWorkspace::default(),
    )?;
    fit.coefficients
        .get(index)
        .copied()
        .filter(|v| v.is_finite())
        .ok_or_else(|| EstimationError::stats_msg("mediation path slope is not estimable"))
}

/// Compose independent mediator/outcome mechanism draws into a posterior over
/// the requested contrast, retaining all three decomposition quantities.
pub fn compose_temporal_mediation(
    mediator: &CausalPosterior,
    outcome: &CausalPosterior,
    query: &MediationQuery,
    status: IdentificationStatus,
) -> Result<CausalPosterior, EstimationError> {
    let coefficient = |post: &CausalPosterior, index| -> Result<Vec<f64>, EstimationError> {
        let col = post
            .draws
            .schema
            .quantities
            .iter()
            .position(
                |q| matches!(q, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index),
            )
            .ok_or_else(|| EstimationError::stats_msg("mediation posterior missing coefficient"))?;
        Ok(post.draws.column(col)?.to_vec())
    };
    let a = coefficient(mediator, 1)?;
    let cp = coefficient(outcome, 1)?;
    let b = coefficient(outcome, 2)?;
    let delta = crate::adjustment::intervention_f64(&query.active)?
        - crate::adjustment::intervention_f64(&query.control)?;
    let n = a.len().min(b.len());
    let direct: Vec<_> = cp.iter().take(n).map(|v| v * delta).collect();
    let indirect: Vec<_> = a.iter().zip(&b).map(|(a, b)| a * b * delta).collect();
    let total: Vec<_> = direct.iter().zip(&indirect).map(|(d, i)| d + i).collect();
    let effect = match query.contrast {
        MediationContrast::Total => &total,
        MediationContrast::Direct | MediationContrast::NaturalDirect => &direct,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => &indirect,
    };
    let values: Vec<_> =
        effect.iter().chain(&total).chain(&direct).chain(&indirect).copied().collect();
    let schema = PosteriorSchema {
        quantities: Arc::from([
            PosteriorQuantityKind::Effect { name: Arc::from("mediation") },
            PosteriorQuantityKind::Scalar { name: Arc::from("total") },
            PosteriorQuantityKind::Scalar { name: Arc::from("direct") },
            PosteriorQuantityKind::Scalar { name: Arc::from("mediated") },
        ]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n, values)?;
    let tempered = [mediator, outcome].iter().any(|post| {
        post.assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == DEPENDENCE_ASSUMPTION_ID)
        })
    });
    let mut post = outcome.clone();
    post.identification = status;
    post.summaries = draws.summarize();
    post.draws = draws;
    // Both mechanisms' tempering notes stay on the composed diagnostics.
    post.diagnostics.notes.extend(mediator.diagnostics.notes.iter().cloned());
    post.assumptions.entries.extend(mediator.assumptions.entries.iter().cloned());
    post.assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("temporal.mediation.gaussian_product"),
            description: Arc::from(if tempered {
                "linear additive mediator and outcome mechanisms with Gaussian innovations independent across the two mechanisms and independent coefficient priors; each mechanism's likelihood is tempered by the long-run-variance ratio of its path coefficients for serial dependence (a generalized posterior, not a correlated-innovation likelihood); no treatment-mediator interaction; empirical complete rows fixed"
            } else {
                "linear additive mediator and outcome mechanisms with independent Gaussian innovations and independent coefficient priors; no treatment-mediator interaction; empirical complete rows fixed"
            }),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("temporal.mediation.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    Ok(post)
}

/// Refuse likelihoods that do not implement the linear mechanism product.
pub fn require_gaussian_mediation(
    estimator: &BayesianGComputationAte,
) -> Result<(), EstimationError> {
    if estimator.likelihood != BayesLikelihood::GaussianIdentity {
        return Err(EstimationError::unsupported("Bayesian mediation requires GaussianIdentity"));
    }
    if estimator.backend == crate::BayesianBackendKind::Hmc {
        return Err(EstimationError::unsupported(
            "composed mediation HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    Ok(())
}
