// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a static derivative, elasticity or Jacobian.
//!
//! The ATE-shaped contrast refuters do not apply to a derivative, so its `cheap` and
//! `full` suites are checks on the derivative itself:
//!
//! * `derivative.support` (cheap, full): whether the published derivative's support
//!   report places the evaluation point (or, for an average derivative, the observed
//!   treatment range) on observed support.
//! * `derivative.finite_difference` (cheap, full): the central finite difference of
//!   the identified intervention levels around the evaluation point, on the reported
//!   scale, against the published derivative (a point derivative of order one or two;
//!   an observed-law average derivative through ± shift interventions).
//! * `derivative.bootstrap_stability` (full): a whole-procedure row bootstrap of the
//!   derivative under independent subjects; it checks that the published value lies in
//!   the replicates' 95% percentile range. Failed replicates are kept as failed mass.
//! * Bayesian (cheap, full): the prior and posterior predictive checks of the outcome
//!   regression, and under full prior sensitivity refitting the whole derivative at
//!   each prior-grid point and MCMC diagnostics.
//!
//! The state executes and is unlicensed: its verdicts are checks, not a calibration
//! claim, and no coverage record names it.

use super::{
    Arc, CausalError, CausalResponse, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, IdentificationStatus, IdentifiedEstimand, InferenceMode,
    Intervention, LatencyMode, RefuteSuite, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, TableView, TabularData, VariableId,
};
use antecedent_core::{DerivativeScale, DerivativeWeighting, Value};
use antecedent_validate::RefutationReport;

/// Replicates of the derivative bootstrap when the study requested none.
const DEFAULT_STABILITY_REPLICATES: u32 = 49;
/// Upper bound on the derivative bootstrap replicates.
const MAX_STABILITY_REPLICATES: u32 = 199;
/// Finite-difference step as a share of the treatment's sample standard deviation.
const STEP_SHARE: f64 = 0.1;
/// Relative tolerance of the finite-difference check (plus two standard errors).
const FD_RELATIVE_TOLERANCE: f64 = 0.25;

/// Borrowed inputs of one derivative refutation.
pub(super) struct DerivativeRefutationInput<'a> {
    pub(super) data: &'a TabularData,
    pub(super) query: &'a ResponseQuery,
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) fitter: &'a ContinuousResponseEstimator,
    pub(super) inference: &'a InferenceMode,
    /// Bayesian estimator with the prior in force (`None` under Frequentist inference).
    pub(super) bayesian: Option<&'a antecedent_estimate::BayesianGComputationAte>,
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
    pub(super) response: &'a CausalResponse,
    pub(super) suite: RefuteSuite,
    pub(super) latency: Option<LatencyMode>,
    /// The study's replicate count (zero: the default stability budget).
    pub(super) replicates: u32,
}

/// Whether `functional` is a derivative this state refutes.
pub(super) const fn is_derivative(functional: &ResponseFunctional) -> bool {
    matches!(
        functional,
        ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::AverageDerivative { .. }
            | ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. }
    )
}

/// First component of a derivative response and its standard error / posterior SD.
pub(super) fn derivative_scalar(response: &CausalResponse) -> (f64, f64) {
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => match value {
            ResponseValue::Scalar(value) => *value,
            ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => {
                values.first().copied().unwrap_or(f64::NAN)
            }
            ResponseValue::Surface { .. } | ResponseValue::Envelope(_) => f64::NAN,
        },
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            f64::NAN
        }
    };
    let se = match &response.uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } => *standard_error,
        _ => f64::NAN,
    };
    (value, se)
}

/// Refutation reports, posterior predictive checks and diagnostics of one derivative refuter run.
pub(super) type DerivativeRefutationOutput =
    (Vec<RefutationReport>, Vec<antecedent_validate::PredictiveCheckReport>, Vec<Diagnostic>);

/// Run the derivative refuter state.
///
/// # Errors
///
/// Cancellation, admission, or a failure of the Bayesian posterior checks. A failed
/// finite-difference level or bootstrap replicate is reported, not raised.
pub(super) fn refute_derivative(
    input: &DerivativeRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<DerivativeRefutationOutput, CausalError> {
    let mut reports = Vec::new();
    let mut checks = Vec::new();
    let mut diagnostics = Vec::new();
    if matches!(input.suite, RefuteSuite::None) {
        return Ok((reports, checks, diagnostics));
    }
    let (value, se) = derivative_scalar(input.response);
    reports.push(support_report(input.response, value));
    match finite_difference(input, ctx)? {
        Some(fd) => reports.push(finite_difference_report(value, se, fd)),
        None => diagnostics.push(Diagnostic::new(
            "refute.derivative.finite_difference_not_applicable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the finite-difference consistency check covers a point derivative of order one \
             (any scale) or two (identity scale) and an observed-law average derivative; \
             this derivative has no such plug-in difference",
        )),
    }
    if matches!(input.suite, RefuteSuite::Full) {
        match bootstrap_stability(input, value, ctx)? {
            Ok(report) => reports.push(report),
            Err(reason) => diagnostics.push(Diagnostic::new(
                "refute.derivative.bootstrap_stability_not_applicable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                reason,
            )),
        }
    }
    if let (InferenceMode::Bayesian(_), Some(estimator)) = (input.inference, input.bayesian) {
        let state = super::bayesian_level_refutation::refute_bayesian_intervention_level(
            &super::bayesian_level_refutation::BayesianLevelRefutationInput {
                data: input.data,
                query: input.query,
                estimand: input.estimand,
                fitter: input.fitter,
                estimator,
                status: input.status,
                assumptions: input.assumptions,
                level: value,
                level_sd: se,
                treatment: primary_treatment(&input.query.functional),
                outcome: primary_outcome(&input.query.functional),
                suite: input.suite,
                latency: input.latency,
                plugin_level: false,
                summarize: derivative_scalar,
                target_label: "derivative",
            },
            ctx,
        )?;
        reports.extend(state.refutations);
        checks.extend(state.predictive_checks);
        diagnostics.extend(state.diagnostics);
    }
    diagnostics.push(Diagnostic::new(
        "refute.derivative.state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "derivative refuter state: support at the evaluation point, finite-difference \
         consistency of the identified levels, and under full a whole-procedure bootstrap \
         stability check (Bayesian: also predictive checks and prior sensitivity of the \
         derivative). ATE-shaped contrast refuters do not apply. It executes unlicensed: no \
         coverage or evidence record names it, and the verdicts are checks, not a \
         calibration claim.",
    ));
    Ok((reports, checks, diagnostics))
}

fn primary_treatment(functional: &ResponseFunctional) -> VariableId {
    match functional {
        ResponseFunctional::PointDerivative { treatment, .. }
        | ResponseFunctional::AverageDerivative { treatment, .. } => *treatment,
        ResponseFunctional::DirectionalDerivative { treatments, .. }
        | ResponseFunctional::Jacobian { treatments, .. } => {
            treatments.first().copied().unwrap_or(VariableId::from_raw(0))
        }
        ResponseFunctional::MeanCurve { treatment, .. } => treatment.variable,
        ResponseFunctional::InterventionResponse { interventions, .. } => interventions
            .first()
            .and_then(Intervention::primary_variable)
            .unwrap_or(VariableId::from_raw(0)),
    }
}

fn primary_outcome(functional: &ResponseFunctional) -> VariableId {
    match functional {
        ResponseFunctional::PointDerivative { outcome, .. }
        | ResponseFunctional::AverageDerivative { outcome, .. }
        | ResponseFunctional::MeanCurve { outcome, .. }
        | ResponseFunctional::InterventionResponse { outcome, .. } => *outcome,
        ResponseFunctional::DirectionalDerivative { outcomes, .. }
        | ResponseFunctional::Jacobian { outcomes, .. } => {
            outcomes.first().copied().unwrap_or(VariableId::from_raw(0))
        }
    }
}

pub(super) fn support_report(response: &CausalResponse, value: f64) -> RefutationReport {
    let supported = matches!(response.support.status, antecedent_core::SupportStatus::Supported);
    RefutationReport::new(
        "derivative.support",
        value,
        value,
        if supported { 0.0 } else { 1.0 },
        true,
        supported,
        (!supported).then(|| {
            Arc::from(format!(
                "the derivative's support report is {:?}: the evaluation point is not on \
                 observed treatment support",
                response.support.status
            ))
        }),
        0,
    )
}

pub(super) fn finite_difference_report(value: f64, se: f64, fd: f64) -> RefutationReport {
    let se = if se.is_finite() { se } else { 0.0 };
    let gap = (fd - value).abs();
    let tolerance = FD_RELATIVE_TOLERANCE * value.abs().max(fd.abs()) + 2.0 * se;
    let informative = value.is_finite() && fd.is_finite();
    let passed = informative && gap <= tolerance;
    RefutationReport::new(
        "derivative.finite_difference",
        value,
        fd,
        if tolerance > 0.0 { gap / tolerance } else { gap },
        informative,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "finite difference {fd} of the identified levels differs from the derivative \
                 {value} by {gap}, above the tolerance {tolerance}"
            ))
        }),
        0,
    )
}

/// One identified intervention level `E[Y(do(interventions))]` under the study's inference.
fn level(
    input: &DerivativeRefutationInput<'_>,
    outcome: VariableId,
    interventions: Vec<Intervention>,
    ctx: &ExecutionContext,
) -> Option<f64> {
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from(interventions),
    });
    let response = match input.bayesian {
        Some(estimator) => input.fitter.estimate_bayesian(
            input.data,
            &query,
            input.status,
            input.assumptions.clone(),
            estimator,
            ctx,
        ),
        None => input.fitter.estimate_identified(
            input.data,
            &query,
            input.status,
            input.assumptions.clone(),
        ),
    }
    .ok()?;
    let (value, _) = super::response_path::response_scalar_summary(&response);
    value.is_finite().then_some(value)
}

fn treatment_step(data: &TabularData, treatment: VariableId) -> Option<f64> {
    let values = data.float64_values(treatment).ok()?;
    let finite: Vec<f64> = values.into_iter().filter(|value| value.is_finite()).collect();
    if finite.len() < 2 {
        return None;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let sd = (finite.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (sd > 0.0 && sd.is_finite()).then_some(STEP_SHARE * sd)
}

/// The central finite difference on the reported scale, or `None` when it does not apply.
fn finite_difference(
    input: &DerivativeRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<Option<f64>, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
    }
    Ok(match &input.query.functional {
        ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
            let Some(h) = treatment_step(input.data, *treatment) else { return Ok(None) };
            let at_level = |point: f64| {
                level(input, *outcome, vec![Intervention::set(*treatment, Value::f64(point))], ctx)
            };
            let (Some(up), Some(down)) = (at_level(at + h), at_level(at - h)) else {
                return Ok(None);
            };
            match (order, scale) {
                (1, _) => {
                    let slope = (up - down) / (2.0 * h);
                    match scale {
                        DerivativeScale::Identity => Some(slope),
                        DerivativeScale::LogTreatment => Some(at * slope),
                        DerivativeScale::LogOutcome | DerivativeScale::LogLog => {
                            let Some(mid) = at_level(*at) else { return Ok(None) };
                            if mid == 0.0 {
                                return Ok(None);
                            }
                            Some(if matches!(scale, DerivativeScale::LogLog) {
                                at * slope / mid
                            } else {
                                slope / mid
                            })
                        }
                    }
                }
                (2, DerivativeScale::Identity) => {
                    let Some(mid) = at_level(*at) else { return Ok(None) };
                    Some((up - 2.0 * mid + down) / (h * h))
                }
                _ => None,
            }
        }
        ResponseFunctional::AverageDerivative {
            outcome,
            treatment,
            weighting: DerivativeWeighting::Observed,
        } => {
            let Some(h) = treatment_step(input.data, *treatment) else { return Ok(None) };
            let shifted = |delta: f64| {
                level(
                    input,
                    *outcome,
                    vec![Intervention::shift(*treatment, Value::f64(delta))],
                    ctx,
                )
            };
            match (shifted(h), shifted(-h)) {
                (Some(up), Some(down)) => Some((up - down) / (2.0 * h)),
                _ => None,
            }
        }
        _ => None,
    })
}

/// Whole-procedure row bootstrap of the derivative's first component. The outer error
/// is a hard failure (cancellation, admission); the inner one names why the check does
/// not apply.
#[allow(
    clippy::too_many_lines,
    reason = "one whole-procedure bootstrap: the admission, the replicate loop and the stability verdict share the replicate state"
)]
fn bootstrap_stability(
    input: &DerivativeRefutationInput<'_>,
    value: f64,
    ctx: &ExecutionContext,
) -> Result<Result<RefutationReport, String>, CausalError> {
    if !value.is_finite() {
        return Ok(Err("the derivative has no finite first component to resample".to_owned()));
    }
    let replicates = match input.replicates {
        0 => DEFAULT_STABILITY_REPLICATES,
        requested => requested.min(MAX_STABILITY_REPLICATES),
    };
    let n = input.data.row_count();
    let ids: Vec<VariableId> =
        input.data.schema().variables().iter().map(|variable| variable.id).collect();
    let columns: Vec<Vec<f64>> =
        match ids.iter().map(|&id| input.data.float64_values(id)).collect::<Result<_, _>>() {
            Ok(columns) => columns,
            Err(_) => {
                return Ok(Err(
                    "a column cannot be read as floats, so the table cannot be resampled"
                        .to_owned(),
                ));
            }
        };
    let width = ids.len().checked_add(1).ok_or_else(|| CausalError::Resource {
        message: "derivative bootstrap width overflow".to_owned(),
    })?;
    let rows = n.checked_mul(width).ok_or_else(|| CausalError::Resource {
        message: "derivative bootstrap size overflow".to_owned(),
    })?;
    let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, rows)?;
    let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut rng = ctx.rng.stream_for(
            antecedent_core::StreamDomain::Resample,
            0xD3E1_B007_u64.wrapping_add(u64::from(replicate)),
        );
        let rows: Vec<usize> = (0..n)
            .map(|_| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a uniform in [0, 1) times n floors to a row index below n"
                )]
                let row = ((rng.next_f64() * n as f64) as usize).min(n.saturating_sub(1));
                row
            })
            .collect();
        let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
            .iter()
            .zip(&columns)
            .map(|(&id, column)| (id, rows.iter().map(|&row| column[row]).collect()))
            .collect();
        let Ok(resampled) = input.data.with_replaced_floats(&replacements) else {
            failed += 1;
            continue;
        };
        let response = match input.bayesian {
            Some(estimator) => input.fitter.estimate_bayesian(
                &resampled,
                input.query,
                input.status,
                input.assumptions.clone(),
                estimator,
                ctx,
            ),
            None => input.fitter.estimate_identified(
                &resampled,
                input.query,
                input.status,
                input.assumptions.clone(),
            ),
        };
        match response {
            Ok(response) => values.push(derivative_scalar(&response).0),
            Err(_) => failed += 1,
        }
    }
    let mut finite: Vec<f64> = values.into_iter().filter(|value| value.is_finite()).collect();
    let ok = u32::try_from(finite.len()).unwrap_or(u32::MAX);
    if finite.len() < 2 {
        return Ok(Err(format!(
            "only {ok} of {replicates} bootstrap replicates produced a derivative \
             ({failed} failed)"
        )));
    }
    finite.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "p in [0, 1] times the last index is a valid index"
        )]
        let index = (p * (finite.len() - 1) as f64).round() as usize;
        finite[index.min(finite.len() - 1)]
    };
    let (lower, upper) = (quantile(0.025), quantile(0.975));
    let passed = (lower..=upper).contains(&value);
    let width = upper - lower;
    Ok(Ok(RefutationReport::new(
        "derivative.bootstrap_stability",
        value,
        finite.iter().sum::<f64>() / finite.len() as f64,
        if width > 0.0 { (value - (lower + upper) / 2.0).abs() / width } else { 0.0 },
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "derivative {value} lies outside the bootstrap 95% range [{lower}, {upper}] \
                 ({ok} replicates, {failed} failed)"
            ))
        }),
        ok,
    )))
}
