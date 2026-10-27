//! Multiple imputation on the routes whose answer has several coordinates, and on a single time
//! series.
//!
//! The same impute-fit-pool path as the average-effect route (`missing_imputation`), pooled
//! jointly: every completed table's fit contributes its coordinates (the cells of a conditional
//! effect, the points of a response curve, the atoms of a distribution, the components of a
//! mediation decomposition, the arms of a randomized design, the horizons of a series) with the
//! covariance the route published, and `multiple_imputation::run_vector` pools them by Rubin's
//! rules (frequentist) or stacks their joint posterior draws (Bayesian). The pooled answer is
//! written back into the result's own fields, and the report carries the within-, between- and
//! total covariance, each coordinate's degrees of freedom and fraction of missing information,
//! the joint Wald test and the digest of the assumptions the pooled answer rests on.
//!
//! A quantity of the answer with no published variance (a mediation component whose bootstrap
//! reports only the requested contrast) is pooled by its mean over the completed tables with
//! its between-imputation variance beside it; it never enters the covariance.
//!
//! A single series is imputed by a lagged model (`Imputer::with_dynamics`): each variable is
//! drawn given the others at the same step and every variable at the neighbouring steps, with
//! the parameter draw resampling contiguous blocks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    CausalQuery, CausalResponse, CredibleDraws, ExecutionContext, IntervalInterpretation,
    MediationContrast, ResponseIdentification, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::column::OwnedColumn;
use antecedent_data::{
    ImputationSpec, MissingDeclaration, RowAccounting, TableView, TabularData, TimeSeriesData,
    complete_float_columns,
};
use antecedent_estimate::bayesian::CausalPosterior;
use antecedent_estimate::multiple_imputation::{
    self, ImputationAssumptions, ImputationColumn, Imputer, MultipleImputationFailure,
    MultipleImputationVectorRun, PooledVectorEstimate, RubinVectorPooled, StackedQuantities,
    VectorImputationFit,
};
use antecedent_estimate::{
    AtomUncertainty, BootstrapAccounting, EffectEstimate, InterventionalDistributionEstimate,
    JointCovariance, ProbabilityInterval, TemporalMediationEstimate, TemporalMediationGrid,
    TemporalMediationUncertainty, logit_probability_interval,
};
use antecedent_prob::PosteriorQuantityKind;

use super::builder::DataInput;
use super::execute::Study;
use super::missing_data::{self, MissingRoute};
use super::missing_imputation::{
    analysed_variables, build_imputer_with, check_analysed_variables, declared_imputation, refused,
    stacked_posterior,
};
use super::missing_support::route_missing_support;
use crate::error::CausalError;
use crate::result::{
    ImputationExtra, ImputationReport, ImputationVectorReport, ImputationWald, ResultPayload,
    StudyResult,
};

/// Name of the complete assignment column a randomized design adds to the imputation model: the
/// arm each row was assigned, which the table does not hold.
const ASSIGNMENT_COLUMN: &str = "__assignment";

fn add(ids: &mut Vec<VariableId>, id: VariableId) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

/// The variables a query names, in the order the imputation model anchors on them: the
/// treatment and outcome (or outcomes) first, then the modifiers, mediators and conditioning
/// variables it reads.
pub(super) fn query_variables(query: &CausalQuery) -> Vec<VariableId> {
    let mut ids: Vec<VariableId> = Vec::new();
    match query {
        CausalQuery::AverageEffect(q) => {
            add(&mut ids, q.treatment);
            add(&mut ids, q.outcome);
            q.effect_modifiers.iter().for_each(|&id| add(&mut ids, id));
        }
        CausalQuery::ConditionalEffect(q) => {
            add(&mut ids, q.inner.treatment);
            add(&mut ids, q.inner.outcome);
            q.inner.effect_modifiers.iter().for_each(|&id| add(&mut ids, id));
        }
        CausalQuery::Mediation(q) => {
            add(&mut ids, q.treatment);
            add(&mut ids, q.outcome);
            q.mediators.iter().for_each(|&id| add(&mut ids, id));
        }
        CausalQuery::Distribution(q) => {
            q.interventions
                .iter()
                .filter_map(antecedent_core::Intervention::primary_variable)
                .for_each(|id| add(&mut ids, id));
            q.outcomes.iter().for_each(|&id| add(&mut ids, id));
            q.conditioning.iter().for_each(|&id| add(&mut ids, id));
        }
        CausalQuery::Response(q) => {
            if let Some((treatment, outcome)) = q.functional.primary_pair() {
                add(&mut ids, treatment);
                add(&mut ids, outcome);
            }
        }
        CausalQuery::RandomizedEffect(q) => {
            add(&mut ids, q.outcome);
            if let Some((covariate, _)) = q.fixed_cuped {
                add(&mut ids, covariate);
            }
            q.ancova_covariates.iter().for_each(|&id| add(&mut ids, id));
        }
        CausalQuery::TemporalEffect(q) => {
            add(&mut ids, q.treatment);
            add(&mut ids, q.outcome);
            q.effect_modifiers.iter().for_each(|&id| add(&mut ids, id));
        }
        _ => {}
    }
    ids
}

// ---------------------------------------------------------------------------------------------
// What one completed table's fit reports.
// ---------------------------------------------------------------------------------------------

/// A frequentist fit's coordinates and the covariance the route published for them.
struct Extracted {
    labels: Vec<Arc<str>>,
    point: Vec<f64>,
    /// Row-major `p x p`.
    covariance: Vec<f64>,
    /// How `covariance` was formed.
    basis: &'static str,
    /// Complete-data degrees of freedom, when the estimator has a finite one.
    df: Option<f64>,
    /// Quantities with no published variance, pooled by their mean.
    extras: Vec<(Arc<str>, f64)>,
}

/// A Bayesian fit's joint posterior draws.
struct ExtractedDraws {
    labels: Vec<Arc<str>>,
    /// `draws[q]` holds quantity `q`'s draws.
    draws: Vec<Vec<f64>>,
    /// Quantity of the primary effect, when the posterior has one.
    effect_index: Option<usize>,
    extras: Vec<(Arc<str>, f64)>,
    /// Whether the draws are the result's posterior (else a response's retained credible draws).
    uses_posterior: bool,
}

fn label(text: impl Into<Arc<str>>) -> Arc<str> {
    text.into()
}

fn variance_of(se: f64) -> Option<f64> {
    (se.is_finite() && se >= 0.0).then_some(se * se)
}

/// Variance behind an effect's published standard error: the analytic one, else the bootstrap's.
fn effect_variance(effect: &EffectEstimate) -> Option<f64> {
    variance_of(effect.uncertainty.analytic.se)
        .or_else(|| effect.uncertainty.bootstrap.se.and_then(variance_of))
}

fn diagonal(variances: &[f64]) -> Vec<f64> {
    let p = variances.len();
    let mut matrix = vec![0.0; p * p];
    for (i, variance) in variances.iter().enumerate() {
        matrix[i * p + i] = *variance;
    }
    matrix
}

fn square(covariance: &JointCovariance, p: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            matrix[i * p + j] = covariance.get(i, j);
        }
    }
    matrix
}

fn no_variance(what: &str) -> CausalError {
    refused(format!(
        "{what} publishes no finite standard error to pool; request the route's resampled \
         uncertainty (a bootstrap replicate count) or declare complete_case"
    ))
}

fn extract_frequentist(result: &StudyResult) -> Result<Extracted, CausalError> {
    match &result.payload {
        ResultPayload::Response(response) => response_coordinates(result, response),
        ResultPayload::Distribution(distribution) => distribution_coordinates(distribution),
        ResultPayload::Mediation { decomposition, grid } => {
            mediation_coordinates(decomposition.as_ref(), grid.as_ref())
        }
        ResultPayload::RandomizedEffect(randomized) => randomized_coordinates(randomized),
        ResultPayload::Scalar => scalar_coordinates(result),
        _ => Err(refused("the route's answer is not one multiple imputation pools")),
    }
}

/// A scalar effect, or the cells of a conditional effect's grid with their joint covariance.
fn scalar_coordinates(result: &StudyResult) -> Result<Extracted, CausalError> {
    let effect = result
        .estimate
        .as_effect()
        .ok_or_else(|| refused("the route's answer is not a scalar effect"))?;
    if let (Some(covariance), Some(cdf)) = (&effect.joint.covariance, &effect.joint.exceedance_cdf)
    {
        let p = cdf.len();
        if p > 0 && covariance.dim == p {
            let mut extras = Vec::new();
            if effect.ate.is_finite() {
                extras.push((label("effect"), effect.ate));
            }
            return Ok(Extracted {
                labels: (0..p).map(|i| label(format!("cell[{i}]"))).collect(),
                point: cdf.to_vec(),
                covariance: square(covariance, p),
                basis: "joint_covariance",
                df: None,
                extras,
            });
        }
    }
    if !effect.ate.is_finite() {
        return Err(refused("the route reports no finite scalar effect to pool"));
    }
    let variance =
        effect_variance(effect).ok_or_else(|| no_variance("the fit of a completed table"))?;
    Ok(Extracted {
        labels: vec![label("effect")],
        point: vec![effect.ate],
        covariance: vec![variance],
        basis: "published_variance",
        df: effect.uncertainty.analytic.reference_df.filter(|df| *df > 0.0),
        extras: Vec::new(),
    })
}

fn response_vector(value: &ResponseValue) -> Result<Vec<f64>, CausalError> {
    let point: Vec<f64> = match value {
        ResponseValue::Scalar(x) => vec![*x],
        ResponseValue::Surface { mean: values, .. }
        | ResponseValue::Vector(values)
        | ResponseValue::Jacobian { values, .. } => values.to_vec(),
        ResponseValue::Envelope(_) => {
            return Err(refused(
                "a partially identified response envelope is an identified set, not a point to pool",
            ));
        }
    };
    if point.is_empty() || point.iter().any(|x| !x.is_finite()) {
        return Err(refused("the response holds no finite point to pool"));
    }
    Ok(point)
}

fn response_variances(
    uncertainty: &ResponseUncertainty,
    p: usize,
) -> Result<Vec<f64>, CausalError> {
    match uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } if p == 1 => {
            variance_of(*standard_error)
                .map(|variance| vec![variance])
                .ok_or_else(|| no_variance("the response"))
        }
        ResponseUncertainty::PointwiseBand { level, lower, upper, .. }
            if lower.len() == p && upper.len() == p =>
        {
            let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
            (0..p)
                .map(|i| variance_of((upper[i] - lower[i]) * 0.5 / z))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| no_variance("the response band"))
        }
        _ => Err(refused(
            "the response carries no joint covariance, standard error or pointwise band to pool",
        )),
    }
}

/// A response's values with the joint influence covariance the estimator retained, else each
/// coordinate's published variance.
fn response_coordinates(
    result: &StudyResult,
    response: &CausalResponse,
) -> Result<Extracted, CausalError> {
    let ResponseIdentification::PointIdentified(value) = &response.estimate else {
        return Err(refused("only a point-identified response is pooled"));
    };
    let point = response_vector(value)?;
    let p = point.len();
    let joint = result
        .estimate
        .as_effect()
        .and_then(|effect| effect.joint.covariance.as_ref())
        .filter(|covariance| covariance.dim == p);
    let (covariance, basis) = match joint {
        Some(covariance) => (square(covariance, p), "joint_covariance"),
        None => (diagonal(&response_variances(&response.uncertainty, p)?), "marginal_variances"),
    };
    Ok(Extracted {
        labels: (0..p).map(|i| label(format!("response[{i}]"))).collect(),
        point,
        covariance,
        basis,
        df: None,
        extras: Vec::new(),
    })
}

/// Atom probabilities (and the mean) with their bootstrap variances.
#[allow(clippy::float_cmp, reason = "a boundary probability is exactly 0 or 1")]
fn distribution_coordinates(
    distribution: &InterventionalDistributionEstimate,
) -> Result<Extracted, CausalError> {
    if distribution.atoms.is_empty()
        || distribution.atom_uncertainty.len() != distribution.atoms.len()
    {
        return Err(no_variance("the interventional distribution"));
    }
    let mut labels = Vec::new();
    let mut point = Vec::new();
    let mut variances = Vec::new();
    for (i, (atom, uncertainty)) in
        distribution.atoms.iter().zip(distribution.atom_uncertainty.iter()).enumerate()
    {
        let boundary = atom.probability == 0.0 || atom.probability == 1.0;
        let variance = match uncertainty.se_bootstrap.and_then(variance_of) {
            Some(variance) => variance,
            None if boundary => 0.0,
            None => return Err(no_variance("an atom of the interventional distribution")),
        };
        labels.push(label(format!("atom[{i}]")));
        point.push(atom.probability);
        variances.push(variance);
    }
    if distribution.mean.is_finite() {
        if let Some(variance) = distribution.se_bootstrap.and_then(variance_of) {
            labels.push(label("mean"));
            point.push(distribution.mean);
            variances.push(variance);
        }
    }
    Ok(Extracted {
        labels,
        point,
        covariance: diagonal(&variances),
        basis: "marginal_variances",
        df: None,
        extras: Vec::new(),
    })
}

fn components(estimate: &TemporalMediationEstimate, suffix: &str) -> Vec<(Arc<str>, f64)> {
    [("total", estimate.total), ("direct", estimate.direct), ("mediated", estimate.mediated)]
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (label(format!("{name}{suffix}")), value)))
        .collect()
}

/// A mediation decomposition: the requested contrast with its variance, and the total, direct
/// and mediated effects at each horizon whose block bootstrap reports all three.
fn mediation_coordinates(
    decomposition: Option<&TemporalMediationEstimate>,
    grid: Option<&TemporalMediationGrid>,
) -> Result<Extracted, CausalError> {
    let mut labels = Vec::new();
    let mut point = Vec::new();
    let mut variances = Vec::new();
    let mut extras = Vec::new();
    if let Some(grid) = grid.filter(|grid| !grid.slices.is_empty()) {
        for slice in grid.slices.iter() {
            let horizon = slice.horizon;
            match &slice.uncertainty {
                TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } => {
                    let parts = [
                        ("total", slice.estimate.total, block.total),
                        ("direct", slice.estimate.direct, block.direct),
                        ("mediated", slice.estimate.mediated, block.mediated),
                    ];
                    for (name, value, se) in parts {
                        let (Some(value), Some(variance)) = (value, se.and_then(variance_of))
                        else {
                            return Err(no_variance(&format!(
                                "the {name} effect at horizon {horizon}"
                            )));
                        };
                        labels.push(label(format!("{name}@{horizon}")));
                        point.push(value);
                        variances.push(variance);
                    }
                    extras.push((label(format!("effect@{horizon}")), slice.estimate.effect.ate));
                }
                TemporalMediationUncertainty::FrequentistPointwise { standard_error } => {
                    let variance = standard_error
                        .and_then(variance_of)
                        .ok_or_else(|| no_variance(&format!("the effect at horizon {horizon}")))?;
                    labels.push(label(format!("effect@{horizon}")));
                    point.push(slice.estimate.effect.ate);
                    variances.push(variance);
                    extras.extend(components(&slice.estimate, &format!("@{horizon}")));
                }
                _ => {
                    return Err(refused(format!(
                        "horizon {horizon} reports no frequentist standard error to pool"
                    )));
                }
            }
        }
    } else if let Some(decomposition) = decomposition {
        let variance = effect_variance(&decomposition.effect)
            .ok_or_else(|| no_variance("the requested mediation contrast"))?;
        labels.push(label("effect"));
        point.push(decomposition.effect.ate);
        variances.push(variance);
        extras.extend(components(decomposition, ""));
    } else {
        return Err(refused("the mediation result holds no decomposition"));
    }
    if point.iter().any(|x| !x.is_finite()) {
        return Err(refused("the mediation result holds a non-finite point"));
    }
    Ok(Extracted {
        labels,
        point,
        covariance: diagonal(&variances),
        basis: "marginal_variances",
        df: None,
        extras,
    })
}

/// A randomized design's effect, its factorial contrasts, or its arms, with the design
/// variances the fit published.
fn randomized_coordinates(
    randomized: &crate::result::RandomizedEffectEstimate,
) -> Result<Extracted, CausalError> {
    let mut labels = Vec::new();
    let mut point = Vec::new();
    let mut variances = Vec::new();
    if randomized.multi_arm_values.is_empty() {
        let variance = variance_of(randomized.variance_upper_bound.sqrt())
            .ok_or_else(|| no_variance("the randomized effect"))?;
        labels.push(label("effect"));
        point.push(randomized.effect);
        variances.push(variance);
        if let (Some(value), Some(variance)) = (
            randomized.second_factor_effect,
            randomized.second_factor_variance.and_then(|v| variance_of(v.sqrt())),
        ) {
            labels.push(label("second_factor"));
            point.push(value);
            variances.push(variance);
        }
        if let (Some(value), Some(variance)) = (
            randomized.factorial_interaction,
            randomized.factorial_interaction_variance.and_then(|v| variance_of(v.sqrt())),
        ) {
            labels.push(label("interaction"));
            point.push(value);
            variances.push(variance);
        }
    } else {
        for (arm, value, variance, _) in randomized.multi_arm_values.iter() {
            let variance =
                variance_of(variance.sqrt()).ok_or_else(|| no_variance("a randomized arm"))?;
            labels.push(label(format!("arm:{arm}")));
            point.push(*value);
            variances.push(variance);
        }
    }
    if point.iter().any(|x| !x.is_finite()) {
        return Err(refused("the randomized effect is not finite"));
    }
    Ok(Extracted {
        labels,
        point,
        covariance: diagonal(&variances),
        basis: "marginal_variances",
        df: None,
        extras: Vec::new(),
    })
}

/// A Bayesian fit's draws: the effect and scalar quantities of its posterior, else the credible
/// draws a response retained.
fn extract_bayesian(result: &StudyResult) -> Result<ExtractedDraws, CausalError> {
    if let ResultPayload::Mediation { grid: Some(grid), .. } = &result.payload {
        if grid.slices.len() > 1 {
            return Err(refused(
                "a multi-horizon Bayesian mediation reports a separate posterior per horizon and \
                 defines no joint posterior over horizons to stack; request one horizon or declare \
                 complete_case",
            ));
        }
    }
    if let ResultPayload::Response(response) = &result.payload {
        if let Some(credible) = response.uncertainty.credible_draws() {
            let p = credible.n_coordinates();
            let draws: Vec<Vec<f64>> =
                (0..p).filter_map(|j| credible.column(j).map(<[f64]>::to_vec)).collect();
            if draws.len() == p && p > 0 {
                return Ok(ExtractedDraws {
                    labels: (0..p).map(|i| label(format!("response[{i}]"))).collect(),
                    draws,
                    effect_index: None,
                    extras: Vec::new(),
                    uses_posterior: false,
                });
            }
        }
    }
    if let Some(posterior) = &result.posterior {
        let n = posterior.draws.n_draws;
        let mut labels = Vec::new();
        let mut draws = Vec::new();
        let mut effect_index = None;
        for (q, kind) in posterior.draws.schema.quantities.iter().enumerate() {
            let name = match kind {
                PosteriorQuantityKind::Effect { name } => {
                    if effect_index.is_none() {
                        effect_index = Some(draws.len());
                    }
                    name
                }
                PosteriorQuantityKind::Scalar { name } => name,
                _ => continue,
            };
            labels.push(Arc::clone(name));
            draws.push(posterior.draws.values[q * n..(q + 1) * n].to_vec());
        }
        if draws.is_empty() {
            return Err(refused("the posterior of a completed table has no effect quantity"));
        }
        let extras = match &result.payload {
            ResultPayload::Mediation { decomposition: Some(decomposition), grid: None } => {
                components(decomposition, "")
            }
            _ => Vec::new(),
        };
        return Ok(ExtractedDraws { labels, draws, effect_index, extras, uses_posterior: true });
    }
    Err(refused("the Bayesian fit of a completed table returned no posterior draws to pool"))
}

// ---------------------------------------------------------------------------------------------
// Writing the pooled answer into the first completed table's result.
// ---------------------------------------------------------------------------------------------

fn set_effect(effect: &mut EffectEstimate, point: f64, total_variance: f64, df: Option<f64>) {
    effect.ate = point;
    effect.uncertainty.analytic.se = total_variance.sqrt();
    effect.uncertainty.analytic.kind = None;
    effect.uncertainty.analytic.reference_df = df;
    effect.uncertainty.bootstrap = BootstrapAccounting::default();
}

/// The interval, simultaneous bands and per-row scores of the first completed table do not
/// describe the pooled answer.
fn clear_first_table_inference(effect: &mut EffectEstimate) {
    effect.scores.inference = None;
    effect.scores.influence = None;
    effect.family.simultaneous_interval = None;
    effect.family.contrast_interval = None;
    effect.family.adjusted_p_values = None;
}

fn finite_df(pooled: &RubinVectorPooled, i: usize) -> Option<f64> {
    pooled.coordinates.get(i).map(|c| c.df).filter(|df| df.is_finite())
}

fn pooled_variance(pooled: &RubinVectorPooled, i: usize) -> f64 {
    pooled.total[i * pooled.dim + i]
}

fn apply_frequentist(
    result: &mut StudyResult,
    query: &CausalQuery,
    pooled: &RubinVectorPooled,
    extras: &[f64],
) -> Result<(), CausalError> {
    match &result.payload {
        ResultPayload::Response(_) => apply_response(result, pooled),
        ResultPayload::Distribution(_) => apply_distribution(result, pooled),
        ResultPayload::Mediation { .. } => apply_mediation(result, query, pooled, extras),
        ResultPayload::RandomizedEffect(_) => apply_randomized(result, pooled),
        ResultPayload::Scalar => apply_scalar(result, pooled, extras),
        _ => Err(refused("the route's answer is not one multiple imputation pools")),
    }
}

fn apply_scalar(
    result: &mut StudyResult,
    pooled: &RubinVectorPooled,
    extras: &[f64],
) -> Result<(), CausalError> {
    let effect = result
        .estimate
        .as_effect_mut()
        .ok_or_else(|| refused("the route's answer is not a scalar effect"))?;
    let grid = effect.joint.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() == pooled.dim)
        && effect.joint.covariance.as_ref().is_some_and(|c| c.dim == pooled.dim);
    if grid {
        effect.joint.exceedance_cdf = Some(Arc::from(pooled.point.clone()));
        effect.joint.covariance =
            Some(JointCovariance { dim: pooled.dim, values: Arc::from(pooled.total.clone()) });
        if effect.ate.is_finite() {
            effect.ate = extras.first().copied().unwrap_or(effect.ate);
            effect.uncertainty.analytic.se = f64::NAN;
            effect.uncertainty.bootstrap = BootstrapAccounting::default();
        }
    } else {
        set_effect(effect, pooled.point[0], pooled_variance(pooled, 0), finite_df(pooled, 0));
    }
    clear_first_table_inference(effect);
    Ok(())
}

fn apply_response(result: &mut StudyResult, pooled: &RubinVectorPooled) -> Result<(), CausalError> {
    let p = pooled.dim;
    let response = result
        .payload
        .response_mut()
        .ok_or_else(|| refused("the route's answer is not a response"))?;
    let level = match &response.uncertainty {
        ResponseUncertainty::Scalar { level, .. }
        | ResponseUncertainty::PointwiseBand { level, .. }
        | ResponseUncertainty::SimultaneousBand { level, .. }
        | ResponseUncertainty::IdentifiedEnvelopeBand { level, .. } => *level,
        _ => 0.95,
    };
    let ResponseIdentification::PointIdentified(value) = &mut response.estimate else {
        return Err(refused("only a point-identified response is pooled"));
    };
    let scalar = matches!(value, ResponseValue::Scalar(_));
    match value {
        ResponseValue::Scalar(x) => *x = pooled.point[0],
        ResponseValue::Surface { mean: values, .. }
        | ResponseValue::Vector(values)
        | ResponseValue::Jacobian { values, .. } => *values = Arc::from(pooled.point.clone()),
        ResponseValue::Envelope(_) => {
            return Err(refused("a response envelope is not pooled"));
        }
    }
    let (mut lower, mut upper) = (Vec::with_capacity(p), Vec::with_capacity(p));
    for coordinate in &pooled.coordinates {
        let (lo, hi) = coordinate.interval(level);
        lower.push(lo);
        upper.push(hi);
    }
    response.uncertainty = if scalar {
        ResponseUncertainty::Scalar {
            standard_error: pooled.total[0].sqrt(),
            level,
            lower: lower[0],
            upper: upper[0],
            interpretation: IntervalInterpretation::Confidence,
            draws: None,
        }
    } else {
        ResponseUncertainty::PointwiseBand {
            level,
            lower: Arc::from(lower),
            upper: Arc::from(upper),
            interpretation: IntervalInterpretation::Confidence,
            draws: None,
        }
    };
    if let Some(effect) = result.estimate.as_effect_mut() {
        if scalar {
            set_effect(effect, pooled.point[0], pooled_variance(pooled, 0), finite_df(pooled, 0));
        }
        effect.joint.covariance =
            Some(JointCovariance { dim: p, values: Arc::from(pooled.total.clone()) });
        clear_first_table_inference(effect);
    }
    Ok(())
}

fn level_of(interval: &ProbabilityInterval) -> Option<f64> {
    match interval {
        ProbabilityInterval::Bounded { level, .. } => Some(*level),
        ProbabilityInterval::Unavailable(_) => None,
    }
}

fn apply_distribution(
    result: &mut StudyResult,
    pooled: &RubinVectorPooled,
) -> Result<(), CausalError> {
    let ResultPayload::Distribution(distribution) = &mut result.payload else {
        return Err(refused("the route's answer is not an interventional distribution"));
    };
    let atoms = distribution.atoms.len();
    if pooled.dim < atoms || distribution.atom_uncertainty.len() != atoms {
        return Err(refused("the pooled coordinates do not match the distribution's atoms"));
    }
    let level = distribution
        .atom_uncertainty
        .iter()
        .find_map(|u| level_of(&u.interval))
        .or_else(|| distribution.mean_interval.as_ref().and_then(level_of))
        .unwrap_or(0.95);
    let mut uncertainty = Vec::with_capacity(atoms);
    for (i, slot) in Arc::make_mut(&mut distribution.atoms).iter_mut().enumerate() {
        slot.probability = pooled.point[i];
        let se = pooled_variance(pooled, i).sqrt();
        uncertainty.push(AtomUncertainty {
            se_bootstrap: Some(se),
            replicates_ok: distribution.atom_uncertainty[i].replicates_ok,
            interval: logit_probability_interval(pooled.point[i], Some(se), level),
        });
    }
    distribution.atom_uncertainty = Arc::from(uncertainty);
    if pooled.dim > atoms {
        let i = atoms;
        let se = pooled_variance(pooled, i).sqrt();
        distribution.mean = pooled.point[i];
        distribution.se_bootstrap = Some(se);
        if distribution.mean_interval.is_some() {
            distribution.mean_interval =
                Some(logit_probability_interval(pooled.point[i], Some(se), level));
        }
    }
    let (mean, se) = (distribution.mean, distribution.se_bootstrap);
    if let Some(effect) = result.estimate.as_effect_mut() {
        if let Some(se) = se {
            effect.ate = mean;
            effect.uncertainty.bootstrap.se = Some(se);
        }
        clear_first_table_inference(effect);
    }
    Ok(())
}

/// The index, among the total / direct / mediated coordinates, of the contrast a query asks.
fn requested_component(query: &CausalQuery) -> usize {
    match query {
        CausalQuery::Mediation(q) => match q.contrast {
            MediationContrast::Total => 0,
            MediationContrast::Direct | MediationContrast::NaturalDirect => 1,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => 2,
        },
        _ => 0,
    }
}

/// Take the next `n` means of the pooled extras.
fn take_extras(extras: &mut std::slice::Iter<'_, f64>) -> Option<f64> {
    extras.next().copied()
}

fn apply_mediation(
    result: &mut StudyResult,
    query: &CausalQuery,
    pooled: &RubinVectorPooled,
    extras: &[f64],
) -> Result<(), CausalError> {
    let requested = requested_component(query);
    let mut extras = extras.iter();
    let mut cursor = 0_usize;
    let mut first_requested: Option<(f64, f64, f64)> = None;
    let ResultPayload::Mediation { decomposition, grid } = &mut result.payload else {
        return Err(refused("the route's answer is not a mediation decomposition"));
    };
    if let Some(grid) = grid.as_mut().filter(|grid| !grid.slices.is_empty()) {
        let slices = Arc::make_mut(&mut grid.slices);
        for (index, slice) in slices.iter_mut().enumerate() {
            let original = slice.estimate.effect.ate;
            match &mut slice.uncertainty {
                TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: se,
                    block,
                    ..
                } => {
                    let c = cursor;
                    cursor += 3;
                    slice.estimate.total = Some(pooled.point[c]);
                    slice.estimate.direct = Some(pooled.point[c + 1]);
                    slice.estimate.mediated = Some(pooled.point[c + 2]);
                    block.total = Some(pooled_variance(pooled, c).sqrt());
                    block.direct = Some(pooled_variance(pooled, c + 1).sqrt());
                    block.mediated = Some(pooled_variance(pooled, c + 2).sqrt());
                    let effect_mean =
                        take_extras(&mut extras).unwrap_or(pooled.point[c + requested]);
                    let variance = pooled_variance(pooled, c + requested);
                    *se = Some(variance.sqrt());
                    set_effect(&mut slice.estimate.effect, effect_mean, variance, None);
                    if index == 0 {
                        first_requested = Some((original, effect_mean, variance));
                    }
                }
                TemporalMediationUncertainty::FrequentistPointwise { standard_error, .. } => {
                    let c = cursor;
                    cursor += 1;
                    let variance = pooled_variance(pooled, c);
                    *standard_error = Some(variance.sqrt());
                    set_effect(
                        &mut slice.estimate.effect,
                        pooled.point[c],
                        variance,
                        finite_df(pooled, c),
                    );
                    for slot in [
                        &mut slice.estimate.total,
                        &mut slice.estimate.direct,
                        &mut slice.estimate.mediated,
                    ] {
                        if slot.is_some() {
                            *slot = take_extras(&mut extras);
                        }
                    }
                    if index == 0 {
                        first_requested = Some((original, pooled.point[c], variance));
                    }
                }
                _ => {
                    return Err(refused("a horizon reports no frequentist standard error to pool"));
                }
            }
        }
        if let (Some(decomposition), [only]) = (decomposition.as_mut(), grid.slices.as_ref()) {
            *decomposition = only.estimate.clone();
        }
    } else if let Some(decomposition) = decomposition.as_mut() {
        let variance = pooled_variance(pooled, 0);
        first_requested = Some((decomposition.effect.ate, pooled.point[0], variance));
        set_effect(&mut decomposition.effect, pooled.point[0], variance, finite_df(pooled, 0));
        for slot in
            [&mut decomposition.total, &mut decomposition.direct, &mut decomposition.mediated]
        {
            if slot.is_some() {
                *slot = take_extras(&mut extras);
            }
        }
    } else {
        return Err(refused("the mediation result holds no decomposition"));
    }
    if let (Some((original, point, variance)), Some(effect)) =
        (first_requested, result.estimate.as_effect_mut())
    {
        if effect.ate.to_bits() == original.to_bits() {
            set_effect(effect, point, variance, None);
        }
        clear_first_table_inference(effect);
    }
    Ok(())
}

fn apply_randomized(
    result: &mut StudyResult,
    pooled: &RubinVectorPooled,
) -> Result<(), CausalError> {
    let ResultPayload::RandomizedEffect(randomized) = &mut result.payload else {
        return Err(refused("the route's answer is not a randomized effect"));
    };
    let variance = |i: usize| pooled_variance(pooled, i);
    if randomized.multi_arm_values.is_empty() {
        randomized.effect = pooled.point[0];
        randomized.variance_upper_bound = variance(0);
        let mut next = 1;
        if randomized.second_factor_effect.is_some() && randomized.second_factor_variance.is_some()
        {
            randomized.second_factor_effect = Some(pooled.point[next]);
            randomized.second_factor_variance = Some(variance(next));
            next += 1;
        }
        if randomized.factorial_interaction.is_some()
            && randomized.factorial_interaction_variance.is_some()
        {
            randomized.factorial_interaction = Some(pooled.point[next]);
            randomized.factorial_interaction_variance = Some(variance(next));
        }
    } else {
        let arms: Vec<_> = randomized
            .multi_arm_values
            .iter()
            .enumerate()
            .map(|(i, (arm, _, _, count))| (Arc::clone(arm), pooled.point[i], variance(i), *count))
            .collect();
        if arms.len() >= 2 {
            randomized.effect = arms[1].1 - arms[0].1;
            randomized.variance_upper_bound = 2.0 * (arms[1].2 + arms[0].2);
        }
        randomized.multi_arm_values = Arc::from(arms);
    }
    // The design interval of the first completed table does not describe the pooled answer.
    randomized.standard_error =
        randomized.standard_error.map(|_| randomized.variance_upper_bound.sqrt());
    randomized.interval_95 = None;
    randomized.second_factor_interval_95 = None;
    randomized.factorial_interaction_interval_95 = None;
    randomized.multi_arm_intervals_95 = Arc::from(Vec::new());
    randomized.uncertainty =
        Arc::from("multiple_imputation_rubin_pooled_design_variance_no_interval");
    let (effect, bound) = (randomized.effect, randomized.variance_upper_bound);
    if let Some(slot) = result.estimate.as_effect_mut() {
        let finite = slot.uncertainty.analytic.se.is_finite();
        slot.ate = effect;
        slot.uncertainty.analytic.se = if finite { bound.sqrt() } else { f64::NAN };
        slot.uncertainty.analytic.kind = None;
        slot.uncertainty.bootstrap = BootstrapAccounting::default();
        clear_first_table_inference(slot);
    }
    Ok(())
}

/// The stacked posterior's effect and, for the routes that read their payload from posterior
/// means, the payload's fields.
fn apply_bayesian(
    result: &mut StudyResult,
    stacked: &StackedQuantities,
    effect_index: Option<usize>,
    posteriors: &[CausalPosterior],
    extras: &[f64],
) -> Result<(), CausalError> {
    if posteriors.is_empty() {
        return apply_credible_response(result, stacked);
    }
    result.posterior = Some(stacked_posterior(posteriors)?);
    if let (Some(index), Some(effect)) = (effect_index, result.estimate.as_effect_mut()) {
        let quantity = &stacked.quantities[index];
        effect.ate = quantity.mean;
        effect.uncertainty.analytic.se = quantity.variance.sqrt();
        effect.uncertainty.analytic.kind = None;
        effect.uncertainty.analytic.reference_df = None;
        clear_first_table_inference(effect);
    }
    if let (Some(posterior), ResultPayload::Distribution(distribution)) =
        (result.posterior.as_ref(), &mut result.payload)
    {
        let mean_first = matches!(
            posterior.draws.schema.quantities.first(),
            Some(PosteriorQuantityKind::Effect { .. })
        );
        let offset = usize::from(mean_first);
        if posterior.summaries.mean.len() >= offset + distribution.atoms.len() {
            for (i, atom) in Arc::make_mut(&mut distribution.atoms).iter_mut().enumerate() {
                atom.probability = posterior.summaries.mean[offset + i];
            }
            if mean_first {
                distribution.mean = posterior.summaries.mean[0];
                distribution.se_analytic = posterior.summaries.sd[0];
            }
        }
    }
    if let (Some(posterior), ResultPayload::Mediation { decomposition, grid }) =
        (result.posterior.as_ref(), &mut result.payload)
    {
        apply_bayesian_mediation(posterior, decomposition, grid, extras);
    }
    Ok(())
}

/// A mediation result's decomposition from the stacked posterior, whose columns are the
/// requested contrast, total, direct and mediated effects; without those four, the components
/// are the means of the completed tables'.
fn apply_bayesian_mediation(
    posterior: &CausalPosterior,
    decomposition: &mut Option<TemporalMediationEstimate>,
    grid: &mut Option<TemporalMediationGrid>,
    extras: &[f64],
) {
    let summaries = &posterior.summaries;
    let four = summaries.mean.len() >= 4;
    let summary = |column: usize| antecedent_estimate::MediationPosteriorSummary {
        mean: summaries.mean[column],
        standard_deviation: summaries.sd[column],
        q025: summaries.q025[column],
        q975: summaries.q975[column],
    };
    let update = |estimate: &mut TemporalMediationEstimate| {
        if four {
            estimate.effect.ate = summaries.mean[0];
            estimate.effect.uncertainty.analytic.se = summaries.sd[0];
            estimate.total = Some(summaries.mean[1]);
            estimate.direct = Some(summaries.mean[2]);
            estimate.mediated = Some(summaries.mean[3]);
        } else {
            let mut means = extras.iter();
            for slot in [&mut estimate.total, &mut estimate.direct, &mut estimate.mediated] {
                if slot.is_some() {
                    *slot = take_extras(&mut means);
                }
            }
        }
    };
    if let Some(decomposition) = decomposition.as_mut() {
        update(decomposition);
    }
    if let Some(grid) = grid.as_mut() {
        for slice in Arc::make_mut(&mut grid.slices).iter_mut() {
            update(&mut slice.estimate);
            if let (
                true,
                TemporalMediationUncertainty::BayesianPointwise {
                    requested,
                    total,
                    direct,
                    mediated,
                    n_draws,
                    ..
                },
            ) = (four, &mut slice.uncertainty)
            {
                *requested = summary(0);
                *total = summary(1);
                *direct = summary(2);
                *mediated = summary(3);
                *n_draws = posterior.draws.n_draws;
            }
        }
    }
}

fn apply_credible_response(
    result: &mut StudyResult,
    stacked: &StackedQuantities,
) -> Result<(), CausalError> {
    let p = stacked.quantities.len();
    let response = result
        .payload
        .response_mut()
        .ok_or_else(|| refused("the Bayesian fit returned neither a posterior nor a response"))?;
    let n_total = stacked.quantities.first().map_or(0, |q| q.draws.len());
    let columns: Vec<Vec<f64>> = stacked.quantities.iter().map(|q| q.draws.clone()).collect();
    let level = match &response.uncertainty {
        ResponseUncertainty::Scalar { level, .. }
        | ResponseUncertainty::PointwiseBand { level, .. } => *level,
        _ => 0.95,
    };
    let ResponseIdentification::PointIdentified(value) = &mut response.estimate else {
        return Err(refused("only a point-identified response is pooled"));
    };
    let means: Vec<f64> = stacked.quantities.iter().map(|q| q.mean).collect();
    let scalar = matches!(value, ResponseValue::Scalar(_));
    match value {
        ResponseValue::Scalar(x) => *x = means[0],
        ResponseValue::Surface { mean: values, .. }
        | ResponseValue::Vector(values)
        | ResponseValue::Jacobian { values, .. } => *values = Arc::from(means.clone()),
        ResponseValue::Envelope(_) => return Err(refused("a response envelope is not pooled")),
    }
    let tail = 0.5 * (1.0 - level);
    let (mut lower, mut upper) = (Vec::with_capacity(p), Vec::with_capacity(p));
    for column in &columns {
        let mut sorted = column.clone();
        sorted.sort_by(f64::total_cmp);
        lower.push(antecedent_kernels::quantile_type7_sorted(&sorted, tail));
        upper.push(antecedent_kernels::quantile_type7_sorted(&sorted, 1.0 - tail));
    }
    let draws = CredibleDraws::columns(n_total, &columns);
    response.uncertainty = if scalar {
        ResponseUncertainty::Scalar {
            standard_error: stacked.quantities[0].variance.sqrt(),
            level,
            lower: lower[0],
            upper: upper[0],
            interpretation: IntervalInterpretation::Credible,
            draws: Some(draws),
        }
    } else {
        ResponseUncertainty::PointwiseBand {
            level,
            lower: Arc::from(lower),
            upper: Arc::from(upper),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(draws),
        }
    };
    if let Some(effect) = result.estimate.as_effect_mut() {
        if scalar {
            effect.ate = means[0];
            effect.uncertainty.analytic.se = stacked.quantities[0].variance.sqrt();
            effect.uncertainty.analytic.kind = None;
        }
        clear_first_table_inference(effect);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The pooled report.
// ---------------------------------------------------------------------------------------------

/// Everything the closure over completed tables collects.
#[derive(Default)]
struct Collected {
    base: Option<StudyResult>,
    labels: Vec<Arc<str>>,
    basis: &'static str,
    extra_labels: Vec<Arc<str>>,
    /// Pooled-by-mean quantities, one vector per completed table.
    extras: Vec<Vec<f64>>,
    posteriors: Vec<CausalPosterior>,
    effect_index: Option<usize>,
}

impl Collected {
    /// Read `result`'s fit, keep what the write-back needs, and return the fit.
    fn record(
        &mut self,
        result: StudyResult,
        bayesian: bool,
        first_check: impl FnOnce(&StudyResult) -> Result<(), CausalError>,
    ) -> Result<VectorImputationFit, CausalError> {
        let mut keep_posterior = false;
        let (fit, labels, extras, effect_index, basis) = if bayesian {
            let extracted = extract_bayesian(&result)?;
            keep_posterior = extracted.uses_posterior;
            let fit = VectorImputationFit::bayesian(extracted.draws.clone());
            (fit, extracted.labels, extracted.extras, extracted.effect_index, "posterior_draws")
        } else {
            let extracted = extract_frequentist(&result)?;
            let fit = VectorImputationFit::frequentist(
                extracted.point.clone(),
                extracted.covariance.clone(),
                extracted.df,
            );
            (fit, extracted.labels, extracted.extras, None, extracted.basis)
        };
        if keep_posterior {
            if let Some(posterior) = &result.posterior {
                self.posteriors.push(posterior.clone());
            }
        }
        if self.base.is_none() {
            first_check(&result)?;
            self.labels = labels;
            self.basis = basis;
            self.extra_labels = extras.iter().map(|(name, _)| Arc::clone(name)).collect();
            self.effect_index = effect_index;
            self.base = Some(result);
        } else if labels != self.labels || extras.len() != self.extra_labels.len() {
            return Err(refused(
                "the completed tables' answers differ in their coordinates, so they cannot be \
                 pooled",
            ));
        }
        self.extras.push(extras.into_iter().map(|(_, value)| value).collect());
        Ok(fit)
    }
}

fn mean_of(values: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "imputation counts are far below 2^52")]
    let n = values.len().max(1) as f64;
    values.iter().sum::<f64>() / n
}

/// Between-imputation variance of one pooled-by-mean quantity.
fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean_of(values);
    #[allow(clippy::cast_precision_loss, reason = "imputation counts are far below 2^52")]
    let df = (values.len() - 1) as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / df
}

fn extras_report(collected: &Collected) -> (Vec<f64>, Arc<[ImputationExtra]>) {
    let mut means = Vec::with_capacity(collected.extra_labels.len());
    let mut report = Vec::with_capacity(collected.extra_labels.len());
    for (k, name) in collected.extra_labels.iter().enumerate() {
        let points: Vec<f64> = collected.extras.iter().map(|row| row[k]).collect();
        let point = mean_of(&points);
        means.push(point);
        report.push(ImputationExtra {
            label: Arc::clone(name),
            between: sample_variance(&points),
            points: Arc::from(points),
            point,
        });
    }
    (means, Arc::from(report))
}

struct Pooling {
    unit: Arc<str>,
    units: usize,
    imputed_cells: Arc<[(Arc<str>, usize)]>,
    estimator: Arc<str>,
    lags: usize,
    spec: ImputationSpec,
    digest: Arc<str>,
}

/// Publish the multivariate Rubin moments and joint Wald summary without changing
/// the separate scalar coordinate reported by the route.
fn rubin_vector_report(
    pooled: &RubinVectorPooled,
    labels: Arc<[Arc<str>]>,
    points: Arc<[Arc<[f64]>]>,
    covariance_basis: &str,
    extras: Arc<[ImputationExtra]>,
) -> ImputationVectorReport {
    let p = pooled.dim;
    let wald = pooled.wald(&vec![0.0; p]).ok().map(|wald| ImputationWald {
        null: Arc::from(vec![0.0; p]),
        statistic: wald.statistic,
        df_numerator: wald.df_numerator,
        df_denominator: wald.df_denominator,
    });
    ImputationVectorReport {
        labels,
        covariance_basis: Arc::from(covariance_basis),
        points,
        point: Arc::from(pooled.point.clone()),
        within: Arc::from(pooled.within.clone()),
        between: Arc::from(pooled.between.clone()),
        total: Arc::from(pooled.total.clone()),
        coordinate_df: pooled
            .coordinates
            .iter()
            .map(|c| c.df.is_finite().then_some(c.df))
            .collect(),
        coordinate_fraction_missing_information: pooled
            .coordinates
            .iter()
            .map(|c| Some(c.fraction_missing_information))
            .collect(),
        average_relative_increase: pooled
            .average_relative_increase
            .is_finite()
            .then_some(pooled.average_relative_increase),
        wald,
        extras,
    }
}

/// Pool the collected fits into the first table's result and build the report.
fn pool_into_result(
    study: &Study,
    run: &MultipleImputationVectorRun,
    mut collected: Collected,
    meta: &Pooling,
) -> Result<(StudyResult, ImputationReport), CausalError> {
    let (extra_means, extras_view) = extras_report(&collected);
    let mut result =
        collected.base.take().ok_or_else(|| refused("no completed table was fitted"))?;
    let labels: Arc<[Arc<str>]> = Arc::from(collected.labels.clone());
    let m = run.fits.len();
    let points: Arc<[Arc<[f64]>]> =
        run.fits.iter().map(|fit| Arc::<[f64]>::from(fit.point.clone())).collect();
    let first_points: Arc<[f64]> = run.fits.iter().map(|fit| fit.point[0]).collect();
    let primary = collected.effect_index.unwrap_or(0);
    let primary_points: Arc<[f64]> = run.fits.iter().map(|fit| fit.point[primary]).collect();
    let report = match &run.pooled {
        PooledVectorEstimate::Rubin(pooled) => {
            apply_frequentist(&mut result, &study.query, pooled, &extra_means)?;
            let vector = rubin_vector_report(pooled, labels, points, collected.basis, extras_view);
            let first = &pooled.coordinates[0];
            ImputationReport {
                spec: meta.spec,
                unit: Arc::clone(&meta.unit),
                units: meta.units,
                imputed_cells: Arc::clone(&meta.imputed_cells),
                estimator: Arc::clone(&meta.estimator),
                pooling: Arc::from("rubin"),
                points: first_points,
                within: first.within,
                between: first.between,
                total: first.total,
                df: first.df.is_finite().then_some(first.df),
                df_complete: first.df_complete,
                fraction_missing_information: Some(first.fraction_missing_information),
                relative_increase: first
                    .relative_increase
                    .is_finite()
                    .then_some(first.relative_increase),
                assumptions_digest: Arc::clone(&meta.digest),
                lags: meta.lags,
                vector: Some(vector),
            }
        }
        PooledVectorEstimate::Stacked(stacked) => {
            apply_bayesian(
                &mut result,
                stacked,
                collected.effect_index,
                &collected.posteriors,
                &extra_means,
            )?;
            let p = stacked.quantities.len();
            let between_matrix = stacked_between(stacked);
            let mut within = vec![0.0; p * p];
            for (i, quantity) in stacked.quantities.iter().enumerate() {
                within[i * p + i] = quantity.within;
            }
            let lead = &stacked.quantities[primary];
            let vector = ImputationVectorReport {
                labels,
                covariance_basis: Arc::from(collected.basis),
                points,
                point: stacked.quantities.iter().map(|q| q.mean).collect(),
                within: Arc::from(within),
                between: Arc::from(between_matrix),
                total: Arc::from(stacked.covariance.clone()),
                coordinate_df: vec![None; p].into(),
                coordinate_fraction_missing_information: vec![None; p].into(),
                average_relative_increase: None,
                wald: None,
                extras: extras_view,
            };
            ImputationReport {
                spec: meta.spec,
                unit: Arc::clone(&meta.unit),
                units: meta.units,
                imputed_cells: Arc::clone(&meta.imputed_cells),
                estimator: Arc::clone(&meta.estimator),
                pooling: Arc::from("stacked_posterior"),
                points: primary_points,
                within: lead.within,
                between: lead.between,
                total: lead.variance,
                df: None,
                df_complete: None,
                fraction_missing_information: None,
                relative_increase: None,
                assumptions_digest: Arc::clone(&meta.digest),
                lags: meta.lags,
                vector: Some(vector),
            }
        }
        _ => return Err(refused("the pooling rule is not one this route reports")),
    };
    debug_assert_eq!(report.vector.as_ref().map(|v| v.points.len()), Some(m));
    // The interval of the first completed table does not describe the pooled answer, and the
    // pooled construction has no coverage record yet.
    result.interval = None;
    result.executed_contract = None;
    Ok((result, report))
}

/// Full between-imputation covariance of a stacked posterior's coordinates, from the per-table
/// posterior means.
fn stacked_between(stacked: &StackedQuantities) -> Vec<f64> {
    let p = stacked.quantities.len();
    let mut matrix = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            let (qa, qb) = (&stacked.quantities[a], &stacked.quantities[b]);
            let m = qa.imputation_means.len();
            if m < 2 {
                continue;
            }
            #[allow(clippy::cast_precision_loss, reason = "imputation counts are far below 2^52")]
            let df = (m - 1) as f64;
            matrix[a * p + b] = qa
                .imputation_means
                .iter()
                .zip(&qb.imputation_means)
                .map(|(x, y)| (x - qa.mean) * (y - qb.mean))
                .sum::<f64>()
                / df;
        }
    }
    matrix
}

fn failure(failure: MultipleImputationFailure<CausalError>) -> CausalError {
    match failure {
        MultipleImputationFailure::Imputation(error) => CausalError::Estimate(error.into()),
        MultipleImputationFailure::Fit { error, .. } => error,
    }
}

fn route_of(study: &Study) -> MissingRoute {
    MissingRoute::of(&study.query, &study.data).unwrap_or(MissingRoute::Static)
}

// ---------------------------------------------------------------------------------------------
// Tabular routes.
// ---------------------------------------------------------------------------------------------

/// Dense group index per row from first-seen labels.
fn dense_groups(labels: &[Arc<str>]) -> Arc<[u32]> {
    let mut seen: BTreeMap<&str, u32> = BTreeMap::new();
    let mut groups = Vec::with_capacity(labels.len());
    for name in labels {
        let next = u32::try_from(seen.len()).unwrap_or(u32::MAX);
        groups.push(*seen.entry(name.as_ref()).or_insert(next));
    }
    Arc::from(groups)
}

/// Imputation must retain both the realized assignment and its sampling units.
struct RandomizedImputationDesign {
    columns: Vec<ImputationColumn>,
    groups: Option<Arc<[u32]>>,
}

/// The complete columns and sampling groups a randomized design adds to the imputation model:
/// the arm each row was assigned (which the table does not hold, and without which an outcome
/// would be imputed as if it were unaffected by assignment) and, under a declared clustered
/// design, the clusters the assignment units define.
fn randomized_design(
    study: &Study,
    rows: usize,
) -> Result<RandomizedImputationDesign, CausalError> {
    let CausalQuery::RandomizedEffect(query) = &study.query else {
        return Ok(RandomizedImputationDesign { columns: Vec::new(), groups: None });
    };
    if query.realized_assignment.len() != rows {
        return Err(refused("the randomized assignment does not align with the table rows"));
    }
    let assigned: Vec<f64> =
        query.realized_assignment.iter().map(|&arm| f64::from(u8::from(arm))).collect();
    let column = ImputationColumn::new(label(ASSIGNMENT_COLUMN), assigned);
    let groups = match study.declared.sampling_design {
        Some(antecedent_core::DeclaredSamplingDesign::ClusteredPanel) => {
            if query.outcome_units.len() != rows {
                return Err(refused(
                    "the randomized outcome units do not align with the table rows",
                ));
            }
            Some(dense_groups(&query.outcome_units))
        }
        _ => None,
    };
    Ok(RandomizedImputationDesign { columns: vec![column], groups })
}

/// Impute, fit with `click` on every completed table, and pool the coordinates jointly.
///
/// # Errors
///
/// The imputation model's refusals, a fit's error, an estimator multiple imputation does not
/// pool on this route cell, a fit whose answer has no variance to pool, or an analysed variable
/// outside the imputation model that is itself incomplete.
#[allow(
    clippy::too_many_lines,
    reason = "impute, fit and pool share the run's borrowed state in one sequence"
)]
pub(super) fn impute_fit_pool_vector<F>(
    study: &Study,
    data: &TabularData,
    ctx: &ExecutionContext,
    mut click: F,
) -> Result<StudyResult, CausalError>
where
    F: FnMut(&TabularData, &ExecutionContext) -> Result<StudyResult, CausalError>,
{
    let (declaration, spec) =
        declared_imputation(study).ok_or_else(|| refused("no imputation is declared"))?;
    let cell = missing_data::route_cell(&study.query, &study.data, &study.settings.inference);
    let row = route_missing_support(cell)
        .filter(|row| row.supports(declaration.policy))
        .ok_or_else(|| refused("this route cell does not execute multiple imputation"))?;
    let bayesian = study.settings.inference.is_bayesian();
    let RandomizedImputationDesign { columns: extra, groups } =
        randomized_design(study, data.row_count())?;
    let (imputer, ids) = build_imputer_with(study, declaration, spec, data, extra, groups)?;
    let imputed: Vec<usize> = imputer
        .variables()
        .iter()
        .enumerate()
        .filter(|(_, v)| v.missing > 0)
        .map(|(j, _)| j)
        .collect();
    let mut collected = Collected::default();
    let run = multiple_imputation::run_vector(&imputer, ctx, |table, ctx| {
        let replacements: Vec<(VariableId, Arc<[f64]>)> =
            imputed.iter().map(|&j| (ids[j], Arc::clone(&table.columns()[j]))).collect();
        let completed = complete_float_columns(data, &replacements)?;
        let result = click(&completed, ctx)?;
        let estimator = result.logical_plan.estimator.as_deref().unwrap_or("");
        if !row.imputes_with(estimator) {
            return Err(refused(format!(
                "multiple imputation pools the estimators {:?} on this route cell; the route ran \
                 `{estimator}`; next attempt: choose one of them or declare complete_case",
                row.imputation_estimators
            )));
        }
        collected.record(result, bayesian, |first| check_analysed_variables(first, data, &ids))
    })
    .map_err(failure)?;
    let mut assumptions = run.assumptions.clone();
    assumptions.design = study.declared.sampling_design;
    let estimator: Arc<str> = Arc::from(
        collected
            .base
            .as_ref()
            .and_then(|base| base.logical_plan.estimator.as_deref())
            .unwrap_or(""),
    );
    let meta = Pooling {
        unit: Arc::from(run.assumptions.unit),
        units: run.assumptions.units,
        imputed_cells: imputer.imputed_cells().into(),
        estimator,
        lags: 0,
        spec,
        digest: Arc::from(assumptions.digest()),
    };
    let (mut result, report) = pool_into_result(study, &run, collected, &meta)?;
    let variables: Vec<VariableId> = analysed_variables(&result);
    let accounting = RowAccounting::imputed(
        data,
        data.storage().analysis_mask(),
        declaration.policy,
        &variables,
    )?;
    let covariates: Arc<[Arc<str>]> = declaration
        .covariates
        .iter()
        .filter_map(|&id| data.schema().get(id).ok().map(|v| Arc::clone(&v.name)))
        .collect();
    let mut missing = missing_data::report(route_of(study), true, covariates, accounting);
    missing.imputation = Some(report);
    result.missing_data = Some(missing);
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// A single time series.
// ---------------------------------------------------------------------------------------------

/// Neighbouring steps on each side the series' imputation model conditions on: the deepest lag
/// the query reads. A series too short to identify a model that wide is refused by the imputer.
fn series_lags(query: &CausalQuery) -> Result<usize, CausalError> {
    let wanted = match query {
        CausalQuery::TemporalEffect(q) => {
            q.horizon_steps.max(q.max_history_lag.unwrap_or(1)).max(1)
        }
        CausalQuery::Mediation(q) => q.horizons.iter().copied().max().unwrap_or(1).max(1),
        _ => 1,
    };
    usize::try_from(wanted).map_err(|_| refused("the query reads more history than fits"))
}

/// Contiguous block length of the series' parameter draw: the cube root of the length, and at
/// least the span of one conditioning neighbourhood.
fn series_block(rows: usize, lags: usize) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a cube root of a row count is a small positive integer"
    )]
    let cube_root = (rows as f64).cbrt().ceil() as usize;
    cube_root.max(2 * lags + 1)
}

fn series_float_ids(series: &TimeSeriesData) -> Vec<VariableId> {
    series
        .storage()
        .columns()
        .iter()
        .filter_map(|column| match column {
            OwnedColumn::Float64(float) => Some(float.id),
            _ => None,
        })
        .collect()
}

fn series_model_variables(
    study: &Study,
    declaration: &MissingDeclaration,
    series: &TimeSeriesData,
) -> Result<Vec<VariableId>, CausalError> {
    let float_ids = series_float_ids(series);
    if declaration.covariates.is_empty() {
        if float_ids.len() > multiple_imputation::MAX_IMPUTATION_VARIABLES {
            return Err(refused(format!(
                "the series has {} float columns and an imputation model reads at most {}; \
                 declare the auxiliary variables the model should read",
                float_ids.len(),
                multiple_imputation::MAX_IMPUTATION_VARIABLES
            )));
        }
        return Ok(float_ids);
    }
    let mut ids = query_variables(&study.query);
    if ids.is_empty() {
        return Err(refused("this query names no variable the imputation model can anchor on"));
    }
    for &id in declaration.covariates.iter() {
        add(&mut ids, id);
    }
    for &id in &ids {
        if !float_ids.contains(&id) {
            let name = series.schema().get(id).map_or_else(|_| "?".into(), |v| v.name.to_string());
            return Err(refused(format!(
                "variable `{name}` of the imputation model is not a float column"
            )));
        }
    }
    Ok(ids)
}

/// The series with the imputed columns replaced by their completed values.
fn complete_series(
    series: &TimeSeriesData,
    replacements: &[(VariableId, Arc<[f64]>)],
) -> Result<TimeSeriesData, CausalError> {
    let table = TabularData::new(series.storage().clone());
    let completed = complete_float_columns(&table, replacements)?;
    Ok(TimeSeriesData::try_new(completed.storage().clone(), series.time_index().clone())?)
}

/// Refuse when a variable the estimator reads is incomplete and outside the imputation model.
fn check_series_variables(
    result: &StudyResult,
    series: &TimeSeriesData,
    model: &[VariableId],
) -> Result<(), CausalError> {
    for id in analysed_variables(result) {
        if model.contains(&id) {
            continue;
        }
        if !series.column(id)?.validity().is_all_valid() {
            let name = Arc::clone(&series.schema().get(id)?.name);
            return Err(refused(format!(
                "the estimator reads `{name}`, which has missing values and is not in the \
                 imputation model; declare it as an auxiliary variable of the imputation"
            )));
        }
    }
    Ok(())
}

/// The lagged imputer for `series` under the study's declaration, the model's variables and
/// the neighbouring steps it conditions on.
fn series_imputer(
    study: &Study,
    declaration: &MissingDeclaration,
    spec: ImputationSpec,
    series: &TimeSeriesData,
) -> Result<(Imputer, Vec<VariableId>, usize), CausalError> {
    let lags = series_lags(&study.query)?;
    let ids = series_model_variables(study, declaration, series)?;
    let mut columns = Vec::with_capacity(ids.len());
    for &id in &ids {
        let name = Arc::clone(&series.schema().get(id)?.name);
        columns.push(ImputationColumn::new(name, Arc::<[f64]>::from(series.float64_values(id)?)));
    }
    let rows = series.row_count();
    let units = multiple_imputation::bind_series_units(
        study.declared.sampling_design,
        rows,
        series_block(rows, lags),
    )
    .map_err(|error| CausalError::Estimate(error.into()))?;
    let imputer = Imputer::new(columns, spec, units)
        .and_then(|imputer| imputer.with_dynamics(lags))
        .map_err(|error| CausalError::Estimate(error.into()))?;
    Ok((imputer, ids, lags))
}

/// Digest of the assumptions a study's declared imputation rests on: the digest its result's
/// report carries, known before any table is fitted because it depends only on the declaration
/// and the data's shape and missing cells. `None` when no imputation is declared or the data
/// cannot be imputed.
pub(crate) fn declared_assumptions_digest(study: &Study) -> Option<String> {
    let (declaration, spec) = declared_imputation(study)?;
    let imputer = match &study.data {
        DataInput::Tabular(data) => {
            let RandomizedImputationDesign { columns: extra, groups } =
                randomized_design(study, data.row_count()).ok()?;
            build_imputer_with(study, declaration, spec, data, extra, groups).ok()?.0
        }
        DataInput::Temporal(series) => series_imputer(study, declaration, spec, series).ok()?.0,
        _ => return None,
    };
    Some(ImputationAssumptions::of(&imputer, study.declared.sampling_design).digest())
}

/// Impute a series by a lagged model, fit with `click` on every completed series, and pool.
///
/// # Errors
///
/// As [`impute_fit_pool_vector`], plus a sampling design other than the dependent time series
/// and a query that reads more history than the lagged model conditions on.
#[allow(
    clippy::too_many_lines,
    reason = "impute, fit and pool share the run's borrowed state in one sequence"
)]
pub(crate) fn impute_fit_pool_series<F>(
    study: &Study,
    series: &TimeSeriesData,
    ctx: &ExecutionContext,
    mut click: F,
) -> Result<StudyResult, CausalError>
where
    F: FnMut(&TimeSeriesData, &ExecutionContext) -> Result<StudyResult, CausalError>,
{
    let (declaration, spec) =
        declared_imputation(study).ok_or_else(|| refused("no imputation is declared"))?;
    let cell = missing_data::route_cell(&study.query, &study.data, &study.settings.inference);
    let row = route_missing_support(cell)
        .filter(|row| row.supports(declaration.policy))
        .ok_or_else(|| refused("this route cell does not execute multiple imputation"))?;
    let bayesian = study.settings.inference.is_bayesian();
    let (imputer, ids, lags) = series_imputer(study, declaration, spec, series)?;
    let imputed: Vec<usize> = imputer
        .variables()
        .iter()
        .enumerate()
        .filter(|(_, v)| v.missing > 0)
        .map(|(j, _)| j)
        .collect();
    let mut collected = Collected::default();
    let run = multiple_imputation::run_vector(&imputer, ctx, |table, ctx| {
        let replacements: Vec<(VariableId, Arc<[f64]>)> =
            imputed.iter().map(|&j| (ids[j], Arc::clone(&table.columns()[j]))).collect();
        let completed = complete_series(series, &replacements)?;
        let result = click(&completed, ctx)?;
        let estimator = result.logical_plan.estimator.as_deref().unwrap_or("");
        if !row.imputes_with(estimator) {
            return Err(refused(format!(
                "multiple imputation pools the estimators {:?} on this route cell; the route ran \
                 `{estimator}`; next attempt: choose one of them or declare complete_case",
                row.imputation_estimators
            )));
        }
        collected.record(result, bayesian, |first| check_series_variables(first, series, &ids))
    })
    .map_err(failure)?;
    let mut assumptions = run.assumptions.clone();
    assumptions.design = study.declared.sampling_design;
    let estimator: Arc<str> = Arc::from(
        collected
            .base
            .as_ref()
            .and_then(|base| base.logical_plan.estimator.as_deref())
            .unwrap_or(""),
    );
    let meta = Pooling {
        unit: Arc::from(run.assumptions.unit),
        units: run.assumptions.units,
        imputed_cells: imputer.imputed_cells().into(),
        estimator,
        lags,
        spec,
        digest: Arc::from(assumptions.digest()),
    };
    let (mut result, report) = pool_into_result(study, &run, collected, &meta)?;
    let variables: Vec<VariableId> = ids.clone();
    let accounting = RowAccounting::imputed(
        series,
        series.storage().analysis_mask(),
        declaration.policy,
        &variables,
    )?;
    let covariates: Arc<[Arc<str>]> = declaration
        .covariates
        .iter()
        .filter_map(|&id| series.schema().get(id).ok().map(|v| Arc::clone(&v.name)))
        .collect();
    let mut missing = missing_data::report(MissingRoute::Temporal, true, covariates, accounting);
    missing.imputation = Some(report);
    result.missing_data = Some(missing);
    Ok(result)
}
