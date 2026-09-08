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

use crate::{
    BayesianGComputationAte, CausalPosterior, EstimationError, OverlapPolicy,
    PreparedBayesianProblem,
};

/// Prepare the mediator and outcome mechanisms on the same lag-aligned rows.
/// Independent Gaussian innovations and independent coefficient priors identify
/// a posterior product; no prior changes the supplied identification status.
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
    let mut columns = vec![
        LaggedColumn { variable: query.treatment, lag: Lag::from_raw(1) },
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
        })
    };
    let covs: Vec<_> = adjustment
        .iter()
        .enumerate()
        .map(|(i, column)| (column.variable, sample.column(3 + i)))
        .collect();
    let mut outcome_covs = vec![(mediator, sample.column(1))];
    outcome_covs.extend_from_slice(&covs);
    Ok([build(sample.column(1), &covs)?, build(sample.column(2), &outcome_covs)?])
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
    let mut post = outcome.clone();
    post.identification = status;
    post.summaries = draws.summarize();
    post.draws = draws;
    post.assumptions.entries.extend(mediator.assumptions.entries.iter().cloned());
    post.assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("temporal.mediation.gaussian_product"),
            description: Arc::from("linear additive mediator and outcome mechanisms with independent Gaussian innovations and independent coefficient priors; no treatment-mediator interaction; empirical complete rows fixed"),
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
