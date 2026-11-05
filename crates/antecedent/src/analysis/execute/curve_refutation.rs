// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a static mean response curve.
//!
//! The ATE-shaped scalar refuters do not apply to a function-valued curve, so the `cheap` and
//! `full` suites of a `MeanCurve` re-run the whole curve procedure under the same three
//! perturbations the average-effect suites use and judge the **whole curve** on its grid:
//!
//! * `curve.support` (cheap, full): the published support report of the curve.
//! * `curve.placebo_treatment` (cheap, full): the treatment column is permuted, which keeps
//!   its marginal and removes every relation to the outcome and the adjustment set, and the
//!   curve is refit. A procedure that identifies the effect refits a flat curve, so the
//!   flatness `c_r(x) - mean_x c_r` of each placebo curve must be typical around zero at every
//!   grid point.
//! * `curve.random_common_cause` (cheap, full): an independent standard-normal covariate is
//!   added to the adjustment set and the curve is refit; the shift `c_r(x) - c(x)` from the
//!   published curve must be typical around zero at every grid point.
//! * `curve.data_subset` (full): a random 80% of the rows is kept (a Bernoulli row mask) and
//!   the curve is refit; the full-sample curve must be a typical draw from the subset curves,
//!   that is the shift `c_r(x) - c(x)` must be typical around zero at every grid point.
//!
//! Every refit evaluates the published grid (the query's grid replaced by the grid's values),
//! so a data-dependent grid does not move between replicates.
//!
//! The common verdict is a sup-norm typicality statistic. For grid point `x` let `m_x` and
//! `s_x` be the mean and sample standard deviation over the replicates of the deviation being
//! tested. The check passes when `max_x |m_x| / s_x <= z`, where `z` is the two-sided normal
//! quantile at level `0.05 / G` for `G` grid points (a Bonferroni bound over the grid, the
//! curve analogue of the scalar refuters' typicality test of one value under `N(mean, sd)`).
//! A deviation with zero spread passes only when its mean is zero to rounding. Failed refits
//! are kept as failed mass: more than a fifth failing is itself a failed check.
//!
//! The verdicts are checks of this route, not a calibration claim: no coverage record measures
//! them, and a pass does not certify identification (like their scalar counterparts they speak
//! to the estimator's behaviour under perturbation of the data it was given).

use super::{
    Arc, CausalError, CausalResponse, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, IdentificationStatus, RefuteSuite, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TableView, TabularData, VariableId,
};
use antecedent_core::StreamDomain;
use antecedent_data::ValidityBitmap;
use antecedent_validate::RefutationReport;

/// Replicates per perturbation under `cheap` when the study requested none.
const CHEAP_REPLICATES: u32 = 24;
/// Replicates per perturbation under `full` when the study requested none.
const FULL_REPLICATES: u32 = 48;
/// Fewest finite replicates a perturbation needs for a verdict.
const MIN_REPLICATES: usize = 8;
/// Largest share of failed refits a verdict tolerates.
const MAX_FAILED_SHARE: f64 = 0.2;
/// Share of rows the data-subset check keeps.
const SUBSET_FRACTION: f64 = 0.8;
/// Family-wise level of each sup-norm typicality check.
const FAMILY_ALPHA: f64 = 0.05;
/// Stream tags of the three perturbations.
const PLACEBO_STREAM: u64 = 0xC0DE_0001_0000_0000;
const RCC_STREAM: u64 = 0xC0DE_0002_0000_0000;
const SUBSET_STREAM: u64 = 0xC0DE_0003_0000_0000;

/// Refits the curve on perturbed data with `extra` added to the adjustment set, under the
/// replicate's own execution context (an independent Monte-Carlo stream: replicates must not
/// share the draws of a Bayesian refit, or their common Monte-Carlo error reads as a shift
/// with no spread); `None` when the refit fails or does not evaluate the published grid.
pub(super) type CurveFit<'a, D> =
    dyn Fn(&D, &[VariableId], &ExecutionContext) -> Option<Vec<f64>> + 'a;
/// Builds the replicate-`seed` perturbation of the data (a permuted placebo treatment or a row
/// subset); `None` when the data cannot be perturbed that way.
pub(super) type Perturb<'a, D> = dyn Fn(&D, u64) -> Option<D> + 'a;
/// Appends an independent covariate to the data and returns its id.
pub(super) type AddCause<'a, D> = dyn Fn(&D, u64) -> Option<(D, VariableId)> + 'a;

/// Everything one curve perturbation needs, over data of kind `D` (a table or a series).
pub(super) struct CurveProblem<'a, D> {
    pub(super) data: &'a D,
    /// The published curve on its grid.
    pub(super) published: &'a [f64],
    pub(super) fit: &'a CurveFit<'a, D>,
    pub(super) placebo: &'a Perturb<'a, D>,
    pub(super) subset: &'a Perturb<'a, D>,
    pub(super) common_cause: Option<&'a AddCause<'a, D>>,
    pub(super) replicates: u32,
}

/// Borrowed inputs of one curve refutation whose curve is fitted by the adjustment-based
/// continuous response estimator.
pub(super) struct CurveRefutationInput<'a> {
    pub(super) data: &'a TabularData,
    pub(super) query: &'a ResponseQuery,
    pub(super) fitter: &'a ContinuousResponseEstimator,
    /// Bayesian estimator with the prior in force (`None` under Frequentist inference): the
    /// refits then publish the posterior mean curve exactly as the cell does.
    pub(super) bayesian: Option<&'a antecedent_estimate::BayesianGComputationAte>,
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
    pub(super) response: &'a CausalResponse,
    pub(super) suite: RefuteSuite,
    /// The study's replicate count (zero: the suite default).
    pub(super) replicates: u32,
}

/// The treatment variable of a `MeanCurve` query.
pub(super) fn curve_treatment(query: &ResponseQuery) -> Option<VariableId> {
    match &query.functional {
        ResponseFunctional::MeanCurve { treatment, .. } => Some(treatment.variable),
        _ => None,
    }
}

/// Whether `functional` is a curve this state refutes.
pub(super) const fn is_mean_curve(functional: &ResponseFunctional) -> bool {
    matches!(functional, ResponseFunctional::MeanCurve { .. })
}

/// The published curve (or dose-by-horizon surface): its flattened grid and mean values.
pub(super) fn published_curve(response: &CausalResponse) -> Option<(Vec<f64>, Vec<f64>)> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => surface_of(value),
        _ => None,
    }
}

fn surface_of(value: &ResponseValue) -> Option<(Vec<f64>, Vec<f64>)> {
    match value {
        ResponseValue::Surface { grid, dimension, mean }
            if *dimension >= 1 && !mean.is_empty() && grid.len() == mean.len() * *dimension =>
        {
            Some((grid.to_vec(), mean.to_vec()))
        }
        // An identified set of curves (the completions of an equivalence class) is refuted
        // through its midpoint curve; the refit publishes its own envelope and midpoint.
        ResponseValue::Envelope(envelope)
            if envelope.dimension >= 1
                && !envelope.lower.is_empty()
                && envelope.lower.len() == envelope.upper.len()
                && envelope.grid.len() == envelope.lower.len() * envelope.dimension =>
        {
            let mid = envelope.lower.iter().zip(envelope.upper.iter()).map(|(l, u)| 0.5 * (l + u));
            Some((envelope.grid.to_vec(), mid.collect()))
        }
        _ => None,
    }
}

/// The curve a route result publishes: the point-identified surface, or for a graph posterior
/// that loses mass to atoms the identifier refuses, the conditional-on-identified surface.
pub(super) fn result_curve(result: &crate::StudyResult) -> Option<(Vec<f64>, Vec<f64>)> {
    result.response().and_then(published_curve).or_else(|| {
        result
            .structural_response
            .as_ref()
            .and_then(|mixture| mixture.conditional_on_identified.as_ref())
            .and_then(surface_of)
    })
}

/// `query` with its curve grid replaced by the published grid values.
pub(super) fn fixed_grid_query(query: &ResponseQuery, grid: &[f64]) -> Option<ResponseQuery> {
    let ResponseFunctional::MeanCurve { outcome, treatment } = &query.functional else {
        return None;
    };
    let mut fixed = query.clone();
    fixed.functional = ResponseFunctional::MeanCurve {
        outcome: *outcome,
        treatment: antecedent_core::ContinuousDomain::new(
            treatment.variable,
            antecedent_core::GridSpec::Values(grid.to_vec().into()),
        ),
    };
    Some(fixed)
}

/// Refutation reports and diagnostics of one curve refuter run.
pub(super) type CurveRefutationOutput = (Vec<RefutationReport>, Vec<Diagnostic>);

/// One curve refuter run over an arbitrary refit of data of kind `D`.
pub(super) struct CurveRun<'a, D> {
    pub(super) data: &'a D,
    /// The response whose support report and published curve are refuted.
    pub(super) response: &'a CausalResponse,
    /// The curve to refute (the response's own, or a posterior's conditional curve).
    pub(super) published: &'a [f64],
    pub(super) suite: RefuteSuite,
    pub(super) replicates: u32,
    pub(super) fit: &'a CurveFit<'a, D>,
    pub(super) placebo: &'a Perturb<'a, D>,
    pub(super) subset: &'a Perturb<'a, D>,
    /// `Err(reason)` when the route has no covariate to add (the random common cause is then
    /// reported as not applicable with that reason).
    pub(super) common_cause: Result<&'a AddCause<'a, D>, &'static str>,
}

/// Run the curve refuter state for a curve fitted by the continuous response estimator.
///
/// # Errors
///
/// Cancellation or admission. A failed refit is reported, not raised.
pub(super) fn refute_curve(
    input: &CurveRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<CurveRefutationOutput, CausalError> {
    let ResponseFunctional::MeanCurve { treatment, .. } = &input.query.functional else {
        return Ok((Vec::new(), Vec::new()));
    };
    let Some((grid, published)) = published_curve(input.response) else {
        return Ok((
            Vec::new(),
            vec![Diagnostic::new(
                "refute.curve.not_applicable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the curve refuter state needs a point-identified curve; this response is not \
                 one, so no curve check ran",
            )],
        ));
    };
    let Some(fixed) = fixed_grid_query(input.query, &grid) else {
        return Ok((Vec::new(), Vec::new()));
    };
    let width = grid.len();
    let fit = |data: &TabularData,
               extra: &[VariableId],
               rctx: &ExecutionContext|
     -> Option<Vec<f64>> {
        let mut adjustment: Vec<VariableId> = input.fitter.adjustment_set.to_vec();
        adjustment.extend_from_slice(extra);
        let mut fitter = ContinuousResponseEstimator::new(adjustment);
        fitter.options = input.fitter.options.clone();
        let response = match input.bayesian {
            Some(estimator) => fitter.estimate_bayesian(
                data,
                &fixed,
                input.status,
                input.assumptions.clone(),
                estimator,
                rctx,
            ),
            None => {
                fitter.estimate_identified(data, &fixed, input.status, input.assumptions.clone())
            }
        }
        .ok()?;
        let (refit_grid, curve) = published_curve(&response)?;
        let same_grid = refit_grid.len() == width
            && refit_grid.iter().zip(&grid).all(|(a, b)| (a - b).abs() <= 1e-9 * b.abs().max(1.0));
        (same_grid && curve.iter().all(|value| value.is_finite())).then_some(curve)
    };
    let placebo = tabular_placebo(treatment.variable, ctx);
    let subset = tabular_subset(ctx);
    let cause = tabular_cause(ctx);
    refute_curve_run(
        &CurveRun {
            data: input.data,
            response: input.response,
            published: &published,
            suite: input.suite,
            replicates: input.replicates,
            fit: &fit,
            placebo: &placebo,
            subset: &subset,
            common_cause: Ok(&cause),
        },
        ctx,
    )
}

/// Run the curve refuter state over `run.fit`.
///
/// # Errors
///
/// Cancellation or admission. A failed refit is reported, not raised.
pub(super) fn refute_curve_run<D>(
    run: &CurveRun<'_, D>,
    ctx: &ExecutionContext,
) -> Result<CurveRefutationOutput, CausalError> {
    let mut reports = Vec::new();
    let mut diagnostics = Vec::new();
    if matches!(run.suite, RefuteSuite::None) {
        return Ok((reports, diagnostics));
    }
    reports.push(support_report(run.response, run.published));
    let replicates = match (run.replicates, run.suite) {
        (0, RefuteSuite::Full) => FULL_REPLICATES,
        (0, _) => CHEAP_REPLICATES,
        (requested, _) => requested,
    };
    let problem = CurveProblem {
        data: run.data,
        published: run.published,
        fit: run.fit,
        placebo: run.placebo,
        subset: run.subset,
        common_cause: run.common_cause.ok(),
        replicates,
    };
    let _admitted =
        super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, run.published.len())?;
    let mut checks = vec![placebo_treatment(&problem, ctx)?];
    match run.common_cause {
        Ok(_) => checks.push(random_common_cause(&problem, ctx)?),
        Err(reason) => checks.push(Err(format!("curve random common cause: {reason}"))),
    }
    if matches!(run.suite, RefuteSuite::Full) {
        checks.push(data_subset(&problem, ctx)?);
    }
    for check in checks {
        match check {
            Ok(report) => reports.push(report),
            Err(reason) => diagnostics.push(Diagnostic::new(
                "refute.curve.check_not_applicable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                reason,
            )),
        }
    }
    diagnostics.push(Diagnostic::new(
        "refute.curve.state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "curve refuter state: support of the published curve, and the whole-curve refits \
         under a permuted placebo treatment (flat curve expected) and an added random common \
         cause (curve unchanged expected) and, under full, a random 80% data subset (a \
         circular window for a time series; the full-sample curve typical among the subset \
         curves), each judged by a Bonferroni sup-norm typicality statistic over the grid. \
         ATE-shaped contrast refuters do not apply. The verdicts are checks of this route, not \
         a calibration claim.",
    ));
    Ok((reports, diagnostics))
}

/// Relative tolerance under which completion curves count as one curve.
const SAME_CURVE_TOLERANCE: f64 = 1e-9;

impl super::Study {
    /// Run the per-completion curve refuter state of a class-aware (Cpdag / Pag) curve.
    ///
    /// Each evaluated completion is an atom of the identified set. An adjustment-identified
    /// completion runs the curve state against its own adjustment estimand and its own
    /// published curve; a completion identified through a front door or by the general ID
    /// algorithm has no adjustment fit to perturb and is reported as unchecked mass.
    /// Reports are mixed by completion weight only when every completion publishes the same
    /// curve; otherwise each completion's reports are kept and a disagreement diagnostic
    /// states how far the completions spread.
    ///
    /// # Errors
    ///
    /// Cancellation, admission, or a failure of the Bayesian prior resolution.
    pub(super) fn refute_class_curve_atoms<G>(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        weighted: &[(u64, f64, CausalResponse)],
        ctx: &ExecutionContext,
    ) -> Result<CurveRefutationOutput, CausalError> {
        use super::atom_mix::{ScalarRefuterMix, finish_atom_refutations};
        let mut per_atom = Vec::new();
        let mut diagnostics = Vec::new();
        let mut curves: Vec<Vec<f64>> = Vec::new();
        let mut unchecked = 0_usize;
        for (key, weight, response) in weighted {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_VALIDATE,
                });
            }
            let Some(case) = usize::try_from(*key).ok().and_then(|index| envelope.cases.get(index))
            else {
                continue;
            };
            let Some(estimand) = case.result.estimands.first() else { continue };
            let Some((_, curve)) = published_curve(response) else { continue };
            if *weight <= 0.0 {
                continue;
            }
            curves.push(curve);
            if matches!(
                estimand.method_kind().ok(),
                Some(super::EstimandMethod::FrontDoor | super::EstimandMethod::GeneralId)
            ) {
                unchecked += 1;
                continue;
            }
            let mut fitter = ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
            if let Some(options) = &self.response_options {
                fitter.options = options.clone();
            }
            let bayesian = match &self.settings.inference {
                super::InferenceMode::Bayesian(config) => {
                    Some(super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                        &fitter, data, query, config, ctx,
                    )?)
                }
                super::InferenceMode::Frequentist => None,
            };
            let (reports, notes) = refute_curve(
                &CurveRefutationInput {
                    data,
                    query,
                    fitter: &fitter,
                    bayesian: bayesian.as_ref(),
                    status: case.result.status,
                    assumptions: &case.result.required_assumptions,
                    response,
                    suite: self.settings.refute,
                    replicates: self.settings.bootstrap_replicates,
                },
                ctx,
            )?;
            diagnostics.extend(notes);
            per_atom.push((*weight, reports));
        }
        let same_curve = class_curve_diagnostics(&curves, unchecked, &mut diagnostics);
        Ok(finish_atom_refutations(
            per_atom,
            diagnostics,
            same_curve,
            ScalarRefuterMix::Effect,
            None,
            false,
        ))
    }
}

fn class_curve_diagnostics(
    curves: &[Vec<f64>],
    unchecked: usize,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let spread = match curves.split_first() {
        None => f64::NAN,
        Some((first, rest)) => rest
            .iter()
            .flat_map(|curve| curve.iter().zip(first).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, f64::max),
    };
    let scale = curves.iter().flatten().fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let same_curve = spread.is_finite() && spread <= SAME_CURVE_TOLERANCE * scale;
    diagnostics.push(Diagnostic::new(
        "refute.response.class_envelope_disagreement",
        DiagnosticKind::Scientific,
        if same_curve { DiagnosticSeverity::Info } else { DiagnosticSeverity::Warning },
        format!(
            "{} evaluated completion(s); the completions' curves differ by at most {spread} \
             (sup norm). {}",
            curves.len(),
            if same_curve {
                "Every completion publishes the same curve, so the per-completion refuter \
                 reports are mixed by completion weight."
            } else {
                "The completions disagree, so each completion's refuter reports are kept on \
                 their own; no mixed verdict is formed over an identified set."
            }
        ),
    ));
    if unchecked > 0 {
        diagnostics.push(Diagnostic::new(
            "refute.curve.unchecked_completions",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "{unchecked} completion(s) are identified through a front door or by the \
                 general ID algorithm, so no adjustment fit exists to perturb; their mass is \
                 not covered by the curve refuter state"
            ),
        ));
    }
    same_curve
}

fn support_report(response: &CausalResponse, published: &[f64]) -> RefutationReport {
    let supported = matches!(response.support.status, antecedent_core::SupportStatus::Supported);
    let sup = published.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    RefutationReport::new(
        "curve.support",
        sup,
        sup,
        if supported { 0.0 } else { 1.0 },
        true,
        supported,
        (!supported).then(|| {
            Arc::from(format!(
                "the curve's support report is {:?}: part of the grid is not on observed \
                 treatment support",
                response.support.status
            ))
        }),
        0,
    )
}

/// The replicate curves of one perturbation: `one(r)` builds replicate `r`'s curve.
fn collect(
    ctx: &ExecutionContext,
    replicates: u32,
    mut one: impl FnMut(u32) -> Option<Vec<f64>>,
) -> Result<(Vec<Vec<f64>>, u32), CausalError> {
    let mut curves = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        match one(replicate) {
            Some(curve) => curves.push(curve),
            None => failed += 1,
        }
    }
    Ok((curves, failed))
}

/// Why a perturbation yields no verdict, or the worst grid point of its typicality statistic.
fn enough(curves: &[Vec<f64>], failed: u32, replicates: u32) -> Result<(), String> {
    if curves.len() < MIN_REPLICATES {
        return Err(format!(
            "only {} of {replicates} refits produced a curve ({failed} failed)",
            curves.len()
        ));
    }
    Ok(())
}

/// Outcome of the sup-norm typicality statistic over a grid.
#[derive(Clone, Copy, Debug)]
pub(super) struct Typicality {
    /// `max_x |m_x| / s_x`.
    pub(super) worst_ratio: f64,
    /// Grid index attaining it.
    pub(super) worst_point: usize,
    /// `max_x |m_x|`.
    pub(super) max_abs_mean: f64,
    /// The Bonferroni quantile the ratio is judged against.
    pub(super) quantile: f64,
}

impl Typicality {
    pub(super) fn passed(&self) -> bool {
        self.worst_ratio.is_finite() && self.worst_ratio <= self.quantile
    }
}

/// The sup-norm typicality of replicate `deviations` (one vector per replicate, all `width`
/// long) around zero.
pub(super) fn typicality(deviations: &[Vec<f64>], width: usize, scale: f64) -> Typicality {
    let quantile = antecedent_kernels::norm_inv(1.0 - FAMILY_ALPHA / (2.0 * width.max(1) as f64));
    let floor = 1e-9 * scale.max(1.0);
    let n = deviations.len() as f64;
    let mut out = Typicality { worst_ratio: 0.0, worst_point: 0, max_abs_mean: 0.0, quantile };
    for point in 0..width {
        let mean = deviations.iter().map(|d| d[point]).sum::<f64>() / n;
        let variance =
            deviations.iter().map(|d| (d[point] - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let spread = variance.sqrt();
        let ratio = if spread <= floor {
            if mean.abs() <= floor { 0.0 } else { f64::INFINITY }
        } else {
            mean.abs() / spread
        };
        out.max_abs_mean = out.max_abs_mean.max(mean.abs());
        if ratio > out.worst_ratio {
            out.worst_ratio = ratio;
            out.worst_point = point;
        }
    }
    out
}

/// The execution context of replicate `seed`: the same context with its own random stream.
fn replicate_context(ctx: &ExecutionContext, seed: u64) -> ExecutionContext {
    let mut replicate = ctx.clone();
    let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed ^ 0x5EED_0000_0000_0000);
    replicate.rng = antecedent_core::RngFactory::from_seed(rng.next_u64());
    replicate
}

fn sup_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn finish(
    name: &'static str,
    published: &[f64],
    replicates: u32,
    statistic: Typicality,
    used: usize,
    failed: u32,
    what: &str,
) -> RefutationReport {
    let failed_share = f64::from(failed) / f64::from(replicates.max(1));
    let too_many_failed = failed_share > MAX_FAILED_SHARE;
    let passed = statistic.passed() && !too_many_failed;
    let message = (!passed).then(|| {
        Arc::from(if too_many_failed {
            format!(
                "{failed} of {replicates} refits failed, above the {:.0}% share a verdict tolerates",
                MAX_FAILED_SHARE * 100.0
            )
        } else {
            format!(
                "{what}: the deviation at grid point {} is {:.3} standard deviations from \
                 zero, above the {:.3} family-wise bound (largest mean deviation {})",
                statistic.worst_point,
                statistic.worst_ratio,
                statistic.quantile,
                statistic.max_abs_mean
            )
        })
    });
    RefutationReport::new(
        name,
        sup_norm(published),
        statistic.max_abs_mean,
        statistic.worst_ratio / statistic.quantile,
        true,
        passed,
        message,
        u32::try_from(used).unwrap_or(u32::MAX),
    )
}

/// The placebo-treatment check. The outer error is a hard failure; the inner one names why the
/// check does not apply.
pub(super) fn placebo_treatment<D>(
    problem: &CurveProblem<'_, D>,
    ctx: &ExecutionContext,
) -> Result<Result<RefutationReport, String>, CausalError> {
    let (curves, failed) = collect(ctx, problem.replicates, |replicate| {
        let placed =
            (problem.placebo)(problem.data, PLACEBO_STREAM.wrapping_add(u64::from(replicate)))?;
        let curve = (problem.fit)(
            &placed,
            &[],
            &replicate_context(ctx, PLACEBO_STREAM.wrapping_add(u64::from(replicate))),
        )?;
        let centre = curve.iter().sum::<f64>() / curve.len() as f64;
        Some(curve.into_iter().map(|value| value - centre).collect())
    })?;
    if let Err(reason) = enough(&curves, failed, problem.replicates) {
        return Ok(Err(format!("curve placebo treatment: {reason}")));
    }
    let published_centre = problem.published.iter().sum::<f64>() / problem.published.len() as f64;
    let scale = problem.published.iter().fold(0.0_f64, |a, v| a.max((v - published_centre).abs()));
    let statistic = typicality(&curves, problem.published.len(), scale);
    Ok(Ok(finish(
        "curve.placebo_treatment",
        problem.published,
        problem.replicates,
        statistic,
        curves.len(),
        failed,
        "the curve refit on a permuted treatment is not flat",
    )))
}

/// The random-common-cause check.
pub(super) fn random_common_cause<D>(
    problem: &CurveProblem<'_, D>,
    ctx: &ExecutionContext,
) -> Result<Result<RefutationReport, String>, CausalError> {
    let Some(add) = problem.common_cause else {
        return Ok(Err("curve random common cause: no covariate can be added".to_owned()));
    };
    let (curves, failed) = collect(ctx, problem.replicates, |replicate| {
        let (extended, id) = add(problem.data, RCC_STREAM.wrapping_add(u64::from(replicate)))?;
        let curve = (problem.fit)(
            &extended,
            &[id],
            &replicate_context(ctx, RCC_STREAM.wrapping_add(u64::from(replicate))),
        )?;
        Some(curve.iter().zip(problem.published).map(|(a, b)| a - b).collect())
    })?;
    if let Err(reason) = enough(&curves, failed, problem.replicates) {
        return Ok(Err(format!("curve random common cause: {reason}")));
    }
    let statistic = typicality(&curves, problem.published.len(), sup_norm(problem.published));
    Ok(Ok(finish(
        "curve.random_common_cause",
        problem.published,
        problem.replicates,
        statistic,
        curves.len(),
        failed,
        "adding an independent random covariate moves the curve",
    )))
}

/// The data-subset check.
pub(super) fn data_subset<D>(
    problem: &CurveProblem<'_, D>,
    ctx: &ExecutionContext,
) -> Result<Result<RefutationReport, String>, CausalError> {
    let (curves, failed) = collect(ctx, problem.replicates, |replicate| {
        let subset =
            (problem.subset)(problem.data, SUBSET_STREAM.wrapping_add(u64::from(replicate)))?;
        let curve = (problem.fit)(
            &subset,
            &[],
            &replicate_context(ctx, SUBSET_STREAM.wrapping_add(u64::from(replicate))),
        )?;
        Some(curve.iter().zip(problem.published).map(|(a, b)| a - b).collect())
    })?;
    if let Err(reason) = enough(&curves, failed, problem.replicates) {
        return Ok(Err(format!("curve data subset: {reason}")));
    }
    let statistic = typicality(&curves, problem.published.len(), sup_norm(problem.published));
    Ok(Ok(finish(
        "curve.data_subset",
        problem.published,
        problem.replicates,
        statistic,
        curves.len(),
        failed,
        "the full-sample curve is atypical among the subset curves",
    )))
}

/// Fisher-Yates shuffle driven by a uniform stream in `[0, 1)`.
fn permute<T>(items: &mut [T], mut uniform: impl FnMut() -> f64) {
    for i in (1..items.len()).rev() {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a uniform in [0, 1) times i + 1 floors to an index at most i"
        )]
        let j = ((uniform() * (i + 1) as f64) as usize).min(i);
        items.swap(i, j);
    }
}

fn gaussian(mut uniform: impl FnMut() -> f64) -> f64 {
    let u1 = uniform().max(f64::MIN_POSITIVE);
    let u2 = uniform();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Placebo of a table: the treatment column permuted (its marginal kept, every relation removed).
pub(super) fn tabular_placebo(
    treatment: VariableId,
    ctx: &ExecutionContext,
) -> impl Fn(&TabularData, u64) -> Option<TabularData> + '_ {
    move |data, seed| {
        let mut column = data.float64_values(treatment).ok()?;
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        permute(&mut column, || rng.next_f64());
        data.with_replaced_floats(&[(treatment, Arc::from(column))]).ok()
    }
}

/// Bernoulli 80% row subset of a table, as an analysis mask.
pub(super) fn tabular_subset(
    ctx: &ExecutionContext,
) -> impl Fn(&TabularData, u64) -> Option<TabularData> + '_ {
    move |data, seed| {
        let n = data.row_count();
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        let mut bytes = vec![0_u8; n.div_ceil(8)];
        for row in 0..n {
            if rng.next_f64() < SUBSET_FRACTION {
                bytes[row / 8] |= 1 << (row % 8);
            }
        }
        let mask = ValidityBitmap::from_bytes(bytes, n).ok()?;
        data.with_analysis_mask(mask).ok()
    }
}

/// An independent standard-normal covariate appended to a table.
pub(super) fn tabular_cause(
    ctx: &ExecutionContext,
) -> impl Fn(&TabularData, u64) -> Option<(TabularData, VariableId)> + '_ {
    move |data, seed| {
        let n = data.row_count();
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        let noise: Vec<f64> = (0..n).map(|_| gaussian(|| rng.next_f64())).collect();
        data.with_appended_float("curve_random_common_cause", Arc::from(noise)).ok()
    }
}

/// Placebo of a series: the treatment column with its time-ordered blocks shuffled. Within a
/// block of about `n^(1/3)` steps the serial structure of the treatment is kept; every link to
/// the outcome and the adjustment lags is removed.
pub(super) fn series_placebo(
    treatment: VariableId,
    ctx: &ExecutionContext,
) -> impl Fn(&antecedent_data::TimeSeriesData, u64) -> Option<antecedent_data::TimeSeriesData> + '_
{
    move |data, seed| {
        let column = data.float64_values(treatment).ok()?;
        let n = column.len();
        if n < 4 {
            return None;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a cube root of a row count is a small positive integer"
        )]
        let block = ((n as f64).cbrt().ceil() as usize).clamp(2, n / 2);
        let mut order: Vec<usize> = (0..n.div_ceil(block)).collect();
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        permute(&mut order, || rng.next_f64());
        let mut shuffled = Vec::with_capacity(n);
        for index in order {
            let start = index * block;
            shuffled.extend_from_slice(&column[start..(start + block).min(n)]);
        }
        data.with_replaced_float(treatment, Arc::from(shuffled)).ok()
    }
}

/// Circular 80% window of a series: a start drawn uniformly over all steps and the next
/// `0.8 n` steps read around the end. Every step is kept with the same probability, so the
/// window curves are centred on the full-sample curve (a non-circular window covers the middle
/// of the series more often than its ends and shifts a time-varying target). Every retained step
/// keeps its true predecessor except the one step after the wrap, the same single seam a
/// circular block bootstrap has.
pub(super) fn series_subset(
    ctx: &ExecutionContext,
) -> impl Fn(&antecedent_data::TimeSeriesData, u64) -> Option<antecedent_data::TimeSeriesData> + '_
{
    move |data, seed| {
        use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TimeIndex};
        let storage = data.storage();
        let n = storage.row_count();
        if n < 4 {
            return None;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "the floored product is at most the row count and is clamped to 1..=n-1"
        )]
        let len = (((n as f64) * SUBSET_FRACTION).floor() as usize).clamp(1, n - 1);
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a uniform in [0, 1) times n floors to a start below n"
        )]
        let start = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        let at = |i: usize| (start + i) % n;
        let window = |source: &ValidityBitmap| -> Option<ValidityBitmap> {
            let mut bytes = vec![0_u8; len.div_ceil(8)];
            for i in 0..len {
                if source.is_valid(at(i)) {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            ValidityBitmap::from_bytes(bytes, len).ok()
        };
        let mut columns = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(c) = column else { return None };
            let source = c.values.as_slice();
            let values: Arc<[f64]> = (0..len).map(|i| source[at(i)]).collect();
            columns.push(OwnedColumn::Float64(
                Float64Column::new(c.id, values, window(&c.validity)?).ok()?,
            ));
        }
        let mask = match storage.analysis_mask() {
            Some(mask) => Some(window(mask)?),
            None => None,
        };
        let weights = storage.weights().map(|w| (0..len).map(|i| w[at(i)]).collect::<Arc<[f64]>>());
        let windowed =
            OwnedColumnarStorage::try_new(storage.schema().clone(), columns, mask, weights).ok()?;
        antecedent_data::TimeSeriesData::try_new(
            windowed,
            TimeIndex { regularity: data.time_index().regularity.clone(), length: len },
        )
        .ok()
    }
}

/// Placebo of a panel: each unit's treatment series gets its own time-ordered block shuffle
/// (the within-unit serial structure survives inside a block; every link to the outcome and the
/// adjustment lags, within and across units, is removed).
pub(super) fn panel_placebo(
    treatment: VariableId,
    ctx: &ExecutionContext,
) -> impl Fn(&antecedent_data::PanelData, u64) -> Option<antecedent_data::PanelData> + '_ {
    move |panel, seed| {
        let placebo = series_placebo(treatment, ctx);
        let units: Vec<antecedent_data::PanelUnit> = panel
            .units()
            .iter()
            .enumerate()
            .map(|(index, unit)| {
                Some(antecedent_data::PanelUnit {
                    unit_id: unit.unit_id,
                    series: placebo(&unit.series, seed.wrapping_add((index as u64) << 20))?,
                })
            })
            .collect::<Option<_>>()?;
        antecedent_data::PanelData::try_new(Arc::<[antecedent_data::PanelUnit]>::from(units)).ok()
    }
}

/// Subset of a panel: a random 80% of the whole units (a unit is the sampling cluster, so no
/// unit is ever split). At least two units are kept.
pub(super) fn panel_subset(
    ctx: &ExecutionContext,
) -> impl Fn(&antecedent_data::PanelData, u64) -> Option<antecedent_data::PanelData> + '_ {
    move |panel, seed| {
        let units = panel.units();
        if units.len() < 3 {
            return None;
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, seed);
        let mut order: Vec<usize> = (0..units.len()).collect();
        permute(&mut order, || rng.next_f64());
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "the floored product is at most the unit count and is clamped to 2..=n-1"
        )]
        let keep =
            (((units.len() as f64) * SUBSET_FRACTION).floor() as usize).clamp(2, units.len() - 1);
        order.truncate(keep);
        order.sort_unstable();
        let kept: Vec<antecedent_data::PanelUnit> =
            order.into_iter().map(|index| units[index].clone()).collect();
        antecedent_data::PanelData::try_new(Arc::<[antecedent_data::PanelUnit]>::from(kept)).ok()
    }
}

impl super::Study {
    /// Attach the whole-procedure curve refuter state to the result a route produced for a
    /// `MeanCurve` under a cheap or full suite.
    ///
    /// `route` runs the same executor on perturbed data; it is called on a copy of the study
    /// whose suite is `none` and whose bootstrap is off, so a refit costs one point estimate
    /// and never refutes recursively. This is the refuter of every route whose curve is
    /// produced by an executor of its own (temporal series, temporal classes, graph
    /// posteriors of any atom kind): the executor's whole procedure is re-run, every atom
    /// refit and mixed by posterior weight exactly as the cell does, and the mixed curve is
    /// judged. A result that already carries curve reports is returned unchanged.
    ///
    /// # Errors
    ///
    /// Cancellation or admission. A failed refit is reported, not raised.
    pub(super) fn attach_curve_refutation<D>(
        &self,
        mut result: crate::StudyResult,
        data: &D,
        query: &ResponseQuery,
        perturbations: (&Perturb<'_, D>, &Perturb<'_, D>),
        route: &dyn Fn(
            &super::Study,
            &D,
            &ExecutionContext,
        ) -> Result<crate::StudyResult, CausalError>,
        ctx: &ExecutionContext,
    ) -> Result<crate::StudyResult, CausalError> {
        if matches!(self.settings.refute, RefuteSuite::None)
            || !is_mean_curve(&query.functional)
            || result.refutations.iter().any(|report| report.refuter.starts_with("curve."))
        {
            return Ok(result);
        }
        let Some((grid, published)) = result_curve(&result) else {
            result.diagnostics.push(Diagnostic::new(
                "refute.curve.not_applicable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the curve refuter state needs a point-identified (or conditional-on-identified) \
                 curve; this result has none, so no curve check ran",
            ));
            return Ok(result);
        };
        let mut quiet = self.clone();
        quiet.settings.refute = RefuteSuite::None;
        quiet.settings.bootstrap_replicates = 0;
        quiet.declared.custom_validators.clear();
        let fit =
            |perturbed: &D, extra: &[VariableId], rctx: &ExecutionContext| -> Option<Vec<f64>> {
                if !extra.is_empty() {
                    return None;
                }
                let refit = route(&quiet, perturbed, rctx).ok()?;
                let (refit_grid, curve) = result_curve(&refit)?;
                let same = refit_grid.len() == grid.len()
                    && refit_grid
                        .iter()
                        .zip(&grid)
                        .all(|(a, b)| (a - b).abs() <= 1e-9 * b.abs().max(1.0));
                (same && curve.iter().all(|value| value.is_finite())).then_some(curve)
            };
        let Some(response) = result.response().cloned() else { return Ok(result) };
        let (reports, notes) = refute_curve_run(
            &CurveRun {
                data,
                response: &response,
                published: &published,
                suite: self.settings.refute,
                replicates: self.settings.bootstrap_replicates,
                fit: &fit,
                placebo: perturbations.0,
                subset: perturbations.1,
                common_cause: Err(
                    "the covariates of this route are fixed by the certified graph (its \
                     adjustment, lag or atom structure), so an independent series is not a \
                     node of any certified adjustment and there is no set to which it could be \
                     added",
                ),
            },
            ctx,
        )?;
        result.refutations.extend(reports);
        result.diagnostics.retain(|diagnostic| {
            !matches!(
                diagnostic.code.as_ref(),
                "refute.response.skipped" | "refute.temporal_response.skipped"
            )
        });
        result.diagnostics.extend(notes);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CurveProblem, MIN_REPLICATES, data_subset, placebo_treatment, random_common_cause,
        series_placebo, series_subset, tabular_cause, tabular_placebo, tabular_subset, typicality,
    };
    use antecedent_core::{ExecutionContext, VariableId};
    use antecedent_data::{TableView, TabularData, TimeSeriesData};

    const GRID: usize = 5;

    /// 200 rows: `t`, `y`, `z` and a stale copy `t_stale` of the treatment.
    fn table() -> TabularData {
        let n = 200_usize;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 / 7.0).sin()).collect();
        let t: Vec<f64> = (0..n).map(|i| 0.5 * z[i] + (i as f64 / 3.0).cos()).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
            ("t_stale", t.as_slice()),
        ])
        .unwrap()
    }

    fn curve() -> Vec<f64> {
        (0..GRID).map(|k| 2.0 * (k as f64 - 2.0)).collect()
    }

    fn kept_share(data: &TabularData) -> f64 {
        let n = data.row_count();
        data.storage().analysis_mask().map_or(1.0, |mask| mask.count_valid() as f64 / n as f64)
    }

    #[test]
    fn typicality_passes_noise_around_zero_and_fails_a_shift() {
        let noise: Vec<Vec<f64>> = (0..20)
            .map(|r| (0..GRID).map(|k| ((r * 7 + k * 3) % 11) as f64 / 11.0 - 0.45).collect())
            .collect();
        let quiet = typicality(&noise, GRID, 1.0);
        assert!(quiet.passed(), "{quiet:?}");
        let shifted: Vec<Vec<f64>> =
            noise.iter().map(|d| d.iter().map(|v| v + 3.0).collect()).collect();
        let loud = typicality(&shifted, GRID, 1.0);
        assert!(!loud.passed(), "{loud:?}");
        let constant: Vec<Vec<f64>> = (0..20).map(|_| vec![0.5; GRID]).collect();
        assert!(!typicality(&constant, GRID, 1.0).passed(), "a constant shift has no spread");
        let zero: Vec<Vec<f64>> = (0..20).map(|_| vec![0.0; GRID]).collect();
        assert!(typicality(&zero, GRID, 1.0).passed());
        assert!(MIN_REPLICATES >= 2);
    }

    /// A procedure whose curve ignores the treatment it is given is flat under the placebo; the
    /// same procedure reading a stale copy of the treatment (a planted break) survives the
    /// permutation and is detected.
    #[test]
    fn placebo_passes_a_flat_refit_and_detects_a_curve_read_from_a_stale_treatment() {
        let data = table();
        let published = curve();
        let ctx = ExecutionContext::for_tests(11);
        let placebo = tabular_placebo(VariableId::from_raw(0), &ctx);
        let subset = tabular_subset(&ctx);
        let cause = tabular_cause(&ctx);
        let honest =
            |d: &TabularData, _: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
                let t = d.float64_values(VariableId::from_raw(0)).ok()?;
                let jitter = 1e-3
                    * t.iter().enumerate().map(|(i, v)| v * (i as f64 - 100.0)).sum::<f64>()
                    / 1e4;
                Some((0..GRID).map(|k| 1.0 + jitter * (k as f64 - 2.0)).collect())
            };
        let problem = CurveProblem {
            data: &data,
            published: &published,
            fit: &honest,
            placebo: &placebo,
            subset: &subset,
            common_cause: Some(&cause),
            replicates: 24,
        };
        let report = placebo_treatment(&problem, &ctx).unwrap().expect("applies");
        assert!(report.passed, "{report:?}");

        let leaky = |d: &TabularData, _: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
            let stale = d.float64_values(VariableId::from_raw(3)).ok()?;
            let mean = stale.iter().sum::<f64>() / stale.len() as f64;
            let slope = stale
                .iter()
                .zip(d.float64_values(VariableId::from_raw(1)).ok()?)
                .map(|(t, y)| (t - mean) * y)
                .sum::<f64>()
                / stale.iter().map(|t| (t - mean).powi(2)).sum::<f64>();
            Some((0..GRID).map(|k| slope * (k as f64 - 2.0)).collect())
        };
        let broken = CurveProblem { fit: &leaky, ..problem };
        let report = placebo_treatment(&broken, &ctx).unwrap().expect("applies");
        assert!(
            !report.passed,
            "a curve read from an unpermuted copy of the treatment: {report:?}"
        );
    }

    #[test]
    fn random_common_cause_passes_an_insensitive_refit_and_detects_a_covariate_sensitive_one() {
        let data = table();
        let published = curve();
        let ctx = ExecutionContext::for_tests(12);
        let placebo = tabular_placebo(VariableId::from_raw(0), &ctx);
        let subset = tabular_subset(&ctx);
        let cause = tabular_cause(&ctx);
        let insensitive =
            |d: &TabularData, extra: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
                let mean = match extra.first() {
                    Some(&id) => {
                        let v = d.float64_values(id).ok()?;
                        v.iter().sum::<f64>() / v.len() as f64
                    }
                    None => 0.0,
                };
                Some(curve().iter().map(|c| c + 0.01 * mean).collect())
            };
        let problem = CurveProblem {
            data: &data,
            published: &published,
            fit: &insensitive,
            placebo: &placebo,
            subset: &subset,
            common_cause: Some(&cause),
            replicates: 24,
        };
        let report = random_common_cause(&problem, &ctx).unwrap().expect("applies");
        assert!(report.passed, "{report:?}");

        let sensitive =
            |_: &TabularData, extra: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
                let shift = if extra.is_empty() { 0.0 } else { 0.5 };
                Some(curve().iter().map(|c| c + shift).collect())
            };
        let broken = CurveProblem { fit: &sensitive, ..problem };
        let report = random_common_cause(&broken, &ctx).unwrap().expect("applies");
        assert!(!report.passed, "a curve that shifts when any covariate is added: {report:?}");
    }

    #[test]
    fn data_subset_passes_a_stable_refit_and_detects_a_row_count_dependent_one() {
        let data = table();
        let published = curve();
        let ctx = ExecutionContext::for_tests(13);
        let placebo = tabular_placebo(VariableId::from_raw(0), &ctx);
        let subset = tabular_subset(&ctx);
        let stable =
            |d: &TabularData, _: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
                Some(curve().iter().map(|c| c + 0.01 * (kept_share(d) - 0.8)).collect())
            };
        let problem = CurveProblem {
            data: &data,
            published: &published,
            fit: &stable,
            placebo: &placebo,
            subset: &subset,
            common_cause: None,
            replicates: 24,
        };
        let report = data_subset(&problem, &ctx).unwrap().expect("applies");
        assert!(report.passed, "{report:?}");

        let scaled =
            |d: &TabularData, _: &[VariableId], _: &ExecutionContext| -> Option<Vec<f64>> {
                Some(curve().iter().map(|c| c * kept_share(d)).collect())
            };
        let broken = CurveProblem { fit: &scaled, ..problem };
        let report = data_subset(&broken, &ctx).unwrap().expect("applies");
        assert!(!report.passed, "a curve that scales with the number of rows: {report:?}");
    }

    fn series() -> TimeSeriesData {
        let n = 120_usize;
        let t: Vec<f64> = (0..n).map(|i| (i as f64 / 5.0).sin() + 0.1 * (i % 3) as f64).collect();
        let y: Vec<f64> = t.iter().map(|v| 2.0 * v).collect();
        TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1_000).unwrap()
    }

    /// The series perturbations keep what a lagged design needs: the placebo keeps the treatment
    /// multiset (blocks reordered, serial structure inside a block kept) and the subset is a
    /// circular window of 80% of the steps whose consecutive values are the original ones.
    #[test]
    fn series_perturbations_shuffle_blocks_and_take_a_circular_window() {
        let data = series();
        let ctx = ExecutionContext::for_tests(14);
        let original = data.float64_values(VariableId::from_raw(0)).unwrap();
        let placed = series_placebo(VariableId::from_raw(0), &ctx)(&data, 3).expect("placebo");
        let shuffled = placed.float64_values(VariableId::from_raw(0)).unwrap();
        assert_ne!(shuffled, original);
        let (mut a, mut b) = (shuffled.clone(), original.clone());
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a, b, "the placebo keeps the treatment marginal");
        assert_eq!(
            placed.float64_values(VariableId::from_raw(1)).unwrap(),
            data.float64_values(VariableId::from_raw(1)).unwrap(),
            "the outcome is untouched"
        );
        let window = series_subset(&ctx)(&data, 5).expect("window");
        let kept = window.float64_values(VariableId::from_raw(0)).unwrap();
        assert_eq!(kept.len(), 96);
        assert_eq!(window.time_index().length, 96);
        assert!(
            (0..original.len()).any(|start| {
                kept.iter().enumerate().all(|(i, v)| *v == original[(start + i) % original.len()])
            }),
            "the subset is one circular window of the series"
        );
    }
}
