// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a front-door-identified derivative.
//!
//! A derivative identified through a front door is the slope of an additive-outcome
//! plug-in (`estimate.response.front_door_derivative`), not a regression the
//! intervention-level finite difference can re-identify, so its `cheap` and `full`
//! suites check the plug-in itself:
//!
//! * `derivative.support` (cheap, full): whether the published derivative's support
//!   report places the evaluation point on observed treatment support.
//! * `derivative.front_door_smoothness_sensitivity` (cheap, full): the additive nuisance
//!   smooths of both stages are refit under a markedly stiffer penalty and under a
//!   richer basis with a looser penalty; the published slope must stay within the
//!   finite-difference tolerance (relative share plus two standard errors) of each
//!   refit. A slope that moves with the smoother is driven by the functional-form
//!   restriction (additivity in the mediators and the treatment), not by the data.
//! * `derivative.bootstrap_stability` (full): a whole-procedure resampling check. Rows
//!   (or circular blocks of consecutive windows, under a dependent series) are resampled,
//!   both stages are refit on each replicate as a point estimate, and the published
//!   value must lie in the replicates' 95% percentile range. Failed replicates are kept
//!   as failed mass.
//!
//! The state executes and is unlicensed: its verdicts are checks, not a calibration
//! claim, and no coverage record names it.

use super::{
    Arc, CausalError, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, IdentificationStatus, RefuteSuite, ResponseQuery, TabularData, VariableId,
};
use antecedent_data::TableView;
use antecedent_estimate::{
    ContinuousResponseEstimator, ContinuousResponseOptions, FrontDoorDerivativeInference,
};
use antecedent_validate::RefutationReport;

/// Replicates of the resampling check when the study requested none.
const DEFAULT_STABILITY_REPLICATES: u32 = 49;
/// Relative tolerance of the smoothness-sensitivity check (plus two standard errors).
const SENSITIVITY_RELATIVE_TOLERANCE: f64 = 0.25;
/// Penalty multiplier of the stiff refit and divisor of the flexible refit.
const PENALTY_FACTOR: f64 = 8.0;
/// Extra basis functions of the flexible refit.
const EXTRA_BASIS: usize = 4;

/// Borrowed inputs of one front-door derivative refutation.
pub(super) struct FrontDoorDerivativeRefutationInput<'a> {
    /// Table the derivative was estimated on (the window table of a temporal ADMG).
    pub(super) data: &'a TabularData,
    /// The derivative query in `data`'s own column ids.
    pub(super) query: &'a ResponseQuery,
    /// Covariates the front-door formula conditions on.
    pub(super) adjustment_set: &'a Arc<[VariableId]>,
    /// The study's response options, when it set any.
    pub(super) options: Option<&'a ContinuousResponseOptions>,
    pub(super) mediators: &'a [VariableId],
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
    /// The published response.
    pub(super) response: &'a CausalResponse,
    pub(super) suite: RefuteSuite,
    /// Rows resampled together (one for independent rows).
    pub(super) block: usize,
    /// The study's replicate count (zero: the default stability budget).
    pub(super) replicates: u32,
}

/// Reports and notes of one front-door derivative refuter run.
pub(super) type FrontDoorDerivativeRefutationOutput = (Vec<RefutationReport>, Vec<Diagnostic>);

/// Run the front-door derivative refuter state.
///
/// # Errors
///
/// Cancellation or admission. A failed refit or replicate is reported, not raised.
pub(super) fn refute_front_door_derivative(
    input: &FrontDoorDerivativeRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<FrontDoorDerivativeRefutationOutput, CausalError> {
    let mut reports = Vec::new();
    let mut diagnostics = Vec::new();
    if matches!(input.suite, RefuteSuite::None) {
        return Ok((reports, diagnostics));
    }
    let (value, se) = super::derivative_refutation::derivative_scalar(input.response);
    reports.push(super::derivative_refutation::support_report(input.response, value));
    reports.push(smoothness_sensitivity(input, value, se, ctx)?);
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
    diagnostics.push(Diagnostic::new(
        "refute.derivative.front_door_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "front-door derivative refuter state: support at the evaluation point, sensitivity of \
         the slope to the additive smoothers of both stages, and under full a whole-procedure \
         resampling stability check. The intervention-level finite difference does not apply \
         to a front-door plug-in. It executes unlicensed: no coverage or evidence record \
         names it, and the verdicts are checks, not a calibration claim.",
    ));
    Ok((reports, diagnostics))
}

/// The estimator with the study's options and the given smoother settings.
fn estimator_with(
    input: &FrontDoorDerivativeRefutationInput<'_>,
    basis: usize,
    lambda: f64,
) -> ContinuousResponseEstimator {
    let mut estimator = ContinuousResponseEstimator::new(Arc::clone(input.adjustment_set));
    if let Some(options) = input.options {
        estimator.options = options.clone();
    }
    estimator.options.nuisance_basis = basis;
    estimator.options.nuisance_lambda = lambda;
    estimator
}

/// Point-only front-door derivative of `data` under `estimator`.
fn refit(
    input: &FrontDoorDerivativeRefutationInput<'_>,
    estimator: &ContinuousResponseEstimator,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Option<f64> {
    let response = estimator
        .front_door_derivative(
            data,
            &input.query.functional,
            input.mediators,
            input.status,
            input.assumptions.clone(),
            FrontDoorDerivativeInference::Frequentist { replicates: 0, block: input.block },
            ctx,
        )
        .ok()?;
    let value = super::derivative_refutation::derivative_scalar(&response).0;
    value.is_finite().then_some(value)
}

fn smoothness_sensitivity(
    input: &FrontDoorDerivativeRefutationInput<'_>,
    value: f64,
    se: f64,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
    }
    let anchor = input.options.cloned().unwrap_or_default();
    let grid = [
        (anchor.nuisance_basis, anchor.nuisance_lambda * PENALTY_FACTOR),
        (anchor.nuisance_basis + EXTRA_BASIS, anchor.nuisance_lambda / PENALTY_FACTOR),
    ];
    let se = if se.is_finite() { se } else { 0.0 };
    let mut worst: Option<(f64, f64)> = None;
    let mut failed = 0_u32;
    let mut evaluated = 0_u32;
    for (basis, lambda) in grid {
        let estimator = estimator_with(input, basis, lambda);
        match refit(input, &estimator, input.data, ctx) {
            Some(alt) => {
                evaluated += 1;
                let tolerance =
                    SENSITIVITY_RELATIVE_TOLERANCE * value.abs().max(alt.abs()) + 2.0 * se;
                let ratio = if tolerance > 0.0 {
                    (alt - value).abs() / tolerance
                } else {
                    (alt - value).abs()
                };
                if worst.is_none_or(|(_, current)| ratio > current) {
                    worst = Some((alt, ratio));
                }
            }
            None => failed += 1,
        }
    }
    let Some((alt, ratio)) = worst else {
        return Ok(RefutationReport::new(
            "derivative.front_door_smoothness_sensitivity",
            value,
            f64::NAN,
            f64::NAN,
            false,
            false,
            Some(Arc::from("no smoother refit produced a finite front-door derivative")),
            0,
        ));
    };
    let passed = value.is_finite() && ratio <= 1.0 && failed == 0;
    Ok(RefutationReport::new(
        "derivative.front_door_smoothness_sensitivity",
        value,
        alt,
        ratio,
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "the front-door slope {value} moves to {alt} under a stiffer or richer additive \
                 smoother (gap over tolerance {ratio}; {failed} refit(s) failed)"
            ))
        }),
        evaluated,
    ))
}

/// Circular-block row indices of one resample of `n` rows.
fn resample_rows(n: usize, block: usize, rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
    let block = block.clamp(1, n.max(1));
    let blocks = n.div_ceil(block);
    let mut rows = Vec::with_capacity(blocks * block);
    for _ in 0..blocks {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the high word of a 64-bit uniform times n is below n"
        )]
        let start = ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize;
        rows.extend((0..block).map(|offset| (start + offset) % n));
    }
    rows.truncate(n);
    rows
}

fn bootstrap_stability(
    input: &FrontDoorDerivativeRefutationInput<'_>,
    value: f64,
    ctx: &ExecutionContext,
) -> Result<Result<RefutationReport, String>, CausalError> {
    if !value.is_finite() {
        return Ok(Err("the derivative has no finite first component to resample".to_owned()));
    }
    let replicates =
        if input.replicates == 0 { DEFAULT_STABILITY_REPLICATES } else { input.replicates };
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
        message: "front-door derivative bootstrap width overflow".to_owned(),
    })?;
    let rows = n.checked_mul(width).ok_or_else(|| CausalError::Resource {
        message: "front-door derivative bootstrap size overflow".to_owned(),
    })?;
    let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, rows)?;
    let anchor = input.options.cloned().unwrap_or_default();
    let estimator = estimator_with(input, anchor.nuisance_basis, anchor.nuisance_lambda);
    let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut rng = ctx.rng.stream_for(
            antecedent_core::StreamDomain::Resample,
            0xFD5B_0070_u64.wrapping_add(u64::from(replicate)),
        );
        let picked = resample_rows(n, input.block, &mut rng);
        let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
            .iter()
            .zip(&columns)
            .map(|(&id, column)| (id, picked.iter().map(|&row| column[row]).collect()))
            .collect();
        let Ok(resampled) = input.data.with_replaced_floats(&replacements) else {
            failed += 1;
            continue;
        };
        match refit(input, &estimator, &resampled, ctx) {
            Some(replicate_value) => values.push(replicate_value),
            None => failed += 1,
        }
    }
    let mut finite = values;
    let ok = u32::try_from(finite.len()).unwrap_or(u32::MAX);
    if finite.len() < 2 {
        return Ok(Err(format!(
            "only {ok} of {replicates} resampled front-door derivatives were produced \
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
    let spread = upper - lower;
    Ok(Ok(RefutationReport::new(
        "derivative.bootstrap_stability",
        value,
        finite.iter().sum::<f64>() / finite.len() as f64,
        if spread > 0.0 { (value - (lower + upper) / 2.0).abs() / spread } else { 0.0 },
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "front-door derivative {value} lies outside the resampled 95% range \
                 [{lower}, {upper}] ({ok} replicates, {failed} failed)"
            ))
        }),
        ok,
    )))
}

impl super::Study {
    /// Run the front-door derivative refuter state of this study on a published response.
    ///
    /// `data` and `query` are the table and derivative the response was estimated on (the
    /// window table and window-coordinate query of a temporal ADMG), `block` the rows
    /// resampled together.
    ///
    /// # Errors
    ///
    /// As [`refute_front_door_derivative`].
    pub(super) fn refute_front_door_derivative_response(
        &self,
        (data, query): (&TabularData, &ResponseQuery),
        claim: &super::response_path::FrontDoorClaim<'_>,
        response: &CausalResponse,
        block: usize,
        ctx: &ExecutionContext,
    ) -> Result<FrontDoorDerivativeRefutationOutput, CausalError> {
        refute_front_door_derivative(
            &FrontDoorDerivativeRefutationInput {
                data,
                query,
                adjustment_set: &claim.estimand.adjustment_set,
                options: self.response_options.as_ref(),
                mediators: &claim.estimand.mediators,
                status: claim.status,
                assumptions: &claim.assumptions,
                response,
                suite: self.refute,
                block,
                replicates: self.bootstrap_replicates,
            },
            ctx,
        )
    }
}
