//! Target-native validation of a compiled result, through the compiler.
//!
//! A result's refutation target is read from the result itself and bound to the identity of
//! the lowered program that produced it:
//!
//! * [`result_refutation_target`] names the quantity the result reports — its
//!   [`TargetFamily`], its quadrant (the lowered program's data classification and uncertainty
//!   reading) and one label per coordinate — with the lowered-program identity as the target's
//!   defining source, so a refutation computed under another program is another target.
//! * [`LoweredProgram::target_refutation_plan`] attaches the target-native refuter catalogue
//!   ([`TargetRefutationPlan::for_target`]) to the lowered program.
//! * [`refute_prepared_target`] executes that plan through the prepared handle: every refit
//!   perturbs the data and re-executes the handle's compiled program in full, then reads the
//!   whole target vector back from the refitted result. Refuters the generic data perturbation
//!   cannot construct for a route are returned as typed refusals naming the next attempt.
//! * [`certify_result_refutations`] binds target-native reports to the result's claim and
//!   refuses any report on another target; the claim itself refuses a refutation its own
//!   program did not produce when its target is not the scalar average effect
//!   ([`refuse_scalar_refutations_for_result`], enforced by [`StudyResult::claim`]). Route
//!   reports are stamped with the producing lowered program where an execution returns its
//!   result, so a route's own reports always pass.
//!
//! Every report is a diagnostic: thresholds are conventions, and calibration of the
//! refuters is not measured here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    CausalRng, ExecutionContext, ResponseFunctional, ResponseIdentification, ResponseValue,
    VariableId,
};
use antecedent_data::{OwnedColumn, TableView, TabularData, TimeSeriesData, ValidityBitmap};
use antecedent_kernels::{shuffle, unbiased_index};
use antecedent_validate::RefutationReport;
pub use antecedent_validate::target_native::{
    JointStatistic, JointStatisticKind, RefutationTarget, TargetCertificate,
    TargetCertificationError, TargetFamily, TargetPerturbation, TargetQuadrant, TargetRefit,
    TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefutationReport,
    TargetRefutationRun, TargetRefuterKind, TargetRefuterRefusal, TargetSuite,
};
use antecedent_validate::target_native::{
    certify_target_refutations, refuse_scalar_reports_for_target, run_target_plan,
};
use antecedent_validate::{ValidationError, mediation_sensitivity_fit};

use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::planner::lowered::{LoweredProgram, UncertaintyReading};
use crate::result::StudyResult;

/// Stamp every refutation `result` carries with the lowered program that produced it.
///
/// Called where an execution of a compiled program returns its result (dispatch, the prepared
/// click, the prepared contract binding and the prepared second-click refute): the reports in
/// the result at that point were produced by that program for its own target, whatever
/// suite produced them. A report already stamped keeps its producer.
pub(crate) fn stamp_route_refutations(result: &mut StudyResult) {
    if let Some(program) = result.lowered_identity {
        let program = *program.as_bytes();
        for report in &mut result.refutations {
            report.stamp_producer(program);
        }
    }
}

impl LoweredProgram {
    /// Quadrant of this program: temporal when the scientific problem reads series, panel or
    /// event data; Bayesian when the claim reads a posterior.
    #[must_use]
    pub fn refutation_quadrant(&self) -> TargetQuadrant {
        TargetQuadrant {
            temporal: !matches!(
                self.scientific().data(),
                DataClassification::Tabular | DataClassification::MultiEnvironment
            ),
            bayesian: self.claim().uncertainty() == UncertaintyReading::Posterior,
        }
    }

    /// The refutation target of `family` with `labels`, defined by this program's identity.
    ///
    /// # Errors
    ///
    /// Labels the target refuses (none, or several for a scalar average effect).
    pub fn refutation_target(
        &self,
        family: TargetFamily,
        labels: Vec<Arc<str>>,
    ) -> Result<RefutationTarget, CausalError> {
        RefutationTarget::new(
            family,
            self.refutation_quadrant(),
            labels,
            *self.identity().as_bytes(),
        )
        .map_err(CausalError::from)
    }

    /// The target-native refuter plan of this program for `family` and `labels`.
    ///
    /// # Errors
    ///
    /// As [`Self::refutation_target`].
    pub fn target_refutation_plan(
        &self,
        family: TargetFamily,
        labels: Vec<Arc<str>>,
        suite: TargetSuite,
    ) -> Result<TargetRefutationPlan, CausalError> {
        Ok(TargetRefutationPlan::for_target(self.refutation_target(family, labels)?, suite))
    }
}

/// The family of the quantity `result` reports.
#[must_use]
pub fn result_target_family(result: &StudyResult) -> TargetFamily {
    if result.mediation().is_some() || result.mediation_grid().is_some() {
        return TargetFamily::Mediation;
    }
    if let Some(response) = result.response() {
        return match &response.estimand {
            ResponseFunctional::AverageDerivative { .. }
            | ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => TargetFamily::ResponseDerivative,
            ResponseFunctional::MeanCurve { .. } => TargetFamily::WorldValued,
            ResponseFunctional::InterventionResponse { .. } => match &response.estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(_)) => {
                    TargetFamily::ScalarAverageEffect
                }
                _ if response.horizon_identification.is_some() => TargetFamily::TemporalEffect,
                _ => TargetFamily::WorldValued,
            },
        };
    }
    if result.distribution().is_some() {
        return TargetFamily::WorldValued;
    }
    if result.counterfactual().is_some() {
        return TargetFamily::ConditionalEffect;
    }
    TargetFamily::ScalarAverageEffect
}

fn labelled(prefix: &str, values: &[f64]) -> (Vec<Arc<str>>, Vec<f64>) {
    let labels = (0..values.len()).map(|i| Arc::from(format!("{prefix}[{i}]"))).collect();
    (labels, values.to_vec())
}

/// Labels and values of the quantity `result` reports, in a stable coordinate order.
///
/// # Errors
///
/// A result whose target has no finite point value (a partially identified envelope, a
/// graph-dependent response, a distribution, a mediation decomposition with a missing
/// component, or no scalar estimate).
pub fn result_target_values(
    result: &StudyResult,
) -> Result<(TargetFamily, Vec<Arc<str>>, Vec<f64>), CausalError> {
    let family = result_target_family(result);
    let refuse = |message: &'static str| CausalError::Unsupported { message };
    let (labels, values) = match family {
        TargetFamily::Mediation => {
            let m = result.mediation().ok_or_else(|| {
                refuse(
                    "a multi-horizon mediation grid is refuted one horizon at a time; next \
                     attempt: prepare one horizon per study",
                )
            })?;
            let total = m.total.ok_or_else(|| refuse("mediation total effect is unavailable"))?;
            let direct =
                m.direct.ok_or_else(|| refuse("mediation direct effect is unavailable"))?;
            let mediated =
                m.mediated.ok_or_else(|| refuse("mediation indirect effect is unavailable"))?;
            (
                vec![Arc::<str>::from("total"), Arc::from("direct"), Arc::from("indirect")],
                vec![total, direct, mediated],
            )
        }
        TargetFamily::ResponseDerivative
        | TargetFamily::WorldValued
        | TargetFamily::TemporalEffect
            if result.response().is_some() =>
        {
            let response = result.response().ok_or_else(|| refuse("no response payload"))?;
            match &response.estimate {
                ResponseIdentification::PointIdentified(value) => match value {
                    ResponseValue::Scalar(v) => (vec![Arc::<str>::from("value")], vec![*v]),
                    ResponseValue::Surface { mean, .. } => labelled("grid", mean),
                    ResponseValue::Vector(values) => labelled("coordinate", values),
                    ResponseValue::Jacobian { values, .. } => labelled("jacobian", values),
                    ResponseValue::Envelope(_) => {
                        return Err(refuse(
                            "an envelope is a set, not a point target; next attempt: the \
                             assumption-range sensitivity analyses of the bounds",
                        ));
                    }
                },
                _ => {
                    return Err(refuse(
                        "only a point-identified response has a target vector to refute; next \
                         attempt: refute each identified graph's response separately",
                    ));
                }
            }
        }
        TargetFamily::WorldValued => {
            return Err(refuse(
                "an interventional distribution is refuted through its mean response; next \
                 attempt: query the interventional mean as an InterventionResponse",
            ));
        }
        TargetFamily::ConditionalEffect => {
            let ite = result
                .counterfactual()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            labelled("unit", &ite.unit_effects)
        }
        _ => {
            let effect = result
                .estimate
                .as_effect()
                .ok_or_else(|| refuse("the result reports no scalar effect to refute"))?;
            (vec![Arc::<str>::from("ate")], vec![effect.ate])
        }
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(refuse("the result's target vector is not finite"));
    }
    Ok((family, labels, values))
}

/// Quadrant of `result`: its data classification and whether it carries a posterior.
#[must_use]
pub fn result_quadrant(result: &StudyResult) -> TargetQuadrant {
    TargetQuadrant {
        temporal: !matches!(
            result.logical_plan.data_classification,
            DataClassification::Tabular | DataClassification::MultiEnvironment
        ),
        bayesian: result.posterior.is_some(),
    }
}

/// The refutation target of `result`, defined by the lowered program that produced it.
///
/// # Errors
///
/// A result no lowered dispatch produced, or one whose target has no point value
/// ([`result_target_values`]).
pub fn result_refutation_target(result: &StudyResult) -> Result<RefutationTarget, CausalError> {
    let source = result.lowered_program_identity().ok_or(CausalError::Unsupported {
        message: "a refutation target is defined by the lowered program that produced the \
                  result; next attempt: execute through Study::compile or a prepared handle",
    })?;
    let (family, labels, _) = result_target_values(result)?;
    RefutationTarget::new(family, result_quadrant(result), labels, *source.as_bytes())
        .map_err(CausalError::from)
}

/// Refuse refutations `result` carries that its own program did not produce, when its target
/// is not the scalar average effect.
///
/// Provenance decides, not the report's name: a report the result's lowered program produced
/// during its own execution (stamped by [`stamp_route_refutations`]) speaks to that program's
/// target and passes, whatever suite produced it. Any other report (built outside a compiled
/// execution, or produced by another program) can only be a scalar average-effect refutation,
/// and it does not certify another target. A result no lowered program produced defines no
/// target to compare with and is left to the claim's own identity checks.
///
/// # Errors
///
/// [`TargetCertificationError::ScalarRefuterForOtherTarget`] for the first such report on a
/// result whose target is not the scalar average effect.
pub fn refuse_scalar_refutations_for_result(
    result: &StudyResult,
) -> Result<(), TargetCertificationError> {
    let family = result_target_family(result);
    if family == TargetFamily::ScalarAverageEffect {
        return Ok(());
    }
    let Some(program) = result.lowered_identity else {
        return Ok(());
    };
    let program = *program.as_bytes();
    let foreign: Vec<RefutationReport> =
        result.refutations.iter().filter(|report| !report.stamped_by(&program)).cloned().collect();
    // The family alone decides; the labels and source only name the claim in the error.
    let claim = RefutationTarget::new(family, result_quadrant(result), ["claim"], program)
        .map_err(|_| TargetCertificationError::ScalarRefuterForOtherTarget {
            refuter: Arc::from("unknown"),
            claim_family: family,
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        })?;
    refuse_scalar_reports_for_target(&claim, &foreign)
}

/// Bind target-native `reports` to `result`'s claim.
///
/// # Errors
///
/// [`TargetCertificationError::TargetMismatch`] for a report computed on another target
/// (another lowered program, family, quadrant or coordinate set);
/// [`TargetCertificationError::ScalarRefuterForOtherTarget`] when the result carries scalar
/// average-effect refutations for a target that is not that scalar.
pub fn certify_result_refutations(
    result: &StudyResult,
    reports: &[TargetRefutationReport],
) -> Result<TargetCertificate, TargetCertificationError> {
    refuse_scalar_refutations_for_result(result)?;
    let claim = result_refutation_target(result).map_err(|error| {
        TargetCertificationError::TargetMismatch {
            refuter: Arc::from(reports.first().map_or("none", TargetRefutationReport::refuter)),
            claim_target: format!("undefined ({error})"),
            report_target: reports
                .first()
                .map_or_else(String::new, |report| report.target.identity_hex()),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        }
    })?;
    certify_target_refutations(&claim, reports)
}

/// Declared controls and settings of a prepared target refutation.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefitOptions {
    /// A variable known not to cause the outcome; swapped in for the treatment.
    pub negative_control_exposure: Option<VariableId>,
    /// A variable the treatment is known not to cause; swapped in for the outcome.
    pub negative_control_outcome: Option<VariableId>,
    /// The single mediator of a mediation target, for the sequential-ignorability range.
    pub mediator: Option<VariableId>,
    /// Smallest absolute break-even mediator-outcome error correlation that passes.
    pub mediation_rho_bar: f64,
}

impl Default for TargetRefitOptions {
    fn default() -> Self {
        Self {
            negative_control_exposure: None,
            negative_control_outcome: None,
            mediator: None,
            mediation_rho_bar: 0.1,
        }
    }
}

/// Data a prepared target refutation perturbs.
#[derive(Clone, Copy, Debug)]
pub enum PreparedTargetData<'a> {
    /// Independent rows.
    Tabular(&'a TabularData),
    /// One regularly sampled series.
    Series(&'a TimeSeriesData),
}

/// Plan, reports, refusals and certificate of one prepared target refutation.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefutationOutcome {
    /// The plan attached to the result's lowered program.
    pub plan: TargetRefutationPlan,
    /// Reports and refusals.
    pub run: TargetRefutationRun,
    /// The reports, bound to the result's target.
    pub certificate: TargetCertificate,
}

/// Run the target-native suite of `result` through `prepared`.
///
/// `data` must be the data `result` was computed on (the retained data of the handle). Each
/// refit perturbs it and calls the handle's own estimate entry, which re-executes the
/// compiled program in full, so a prepared handle whose study requests a scalar refutation
/// suite pays for that suite on every refit: prepare with no scalar suite for target
/// refutation.
///
/// # Errors
///
/// The result's target has no point value, the data modality does not match the handle,
/// cancellation, an admission refusal, a failed refit, or (impossible by construction) a
/// report on another target.
pub fn refute_prepared_target(
    prepared: &PreparedStudy,
    result: &StudyResult,
    data: PreparedTargetData<'_>,
    suite: TargetSuite,
    config: &TargetRefutationConfig,
    options: &TargetRefitOptions,
    ctx: &ExecutionContext,
) -> Result<TargetRefutationOutcome, CausalError> {
    let target = result_refutation_target(result)?;
    let (_, _, original) = result_target_values(result)?;
    let treatment = result.treatment.ok_or(CausalError::Unsupported {
        message: "target refutation needs a result with a treatment",
    })?;
    if let Some(lowered) = prepared.plan().lowered() {
        if Some(lowered.identity()) != result.lowered_program_identity() {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "the result was not produced by this prepared handle's lowered program",
            });
        }
    }
    let plan = TargetRefutationPlan::for_target(target.clone(), suite);
    let refit = PreparedRefit {
        prepared,
        result,
        data,
        target: target.clone(),
        original,
        treatment,
        outcome: result.outcome,
        options: options.clone(),
    };
    let run = run_target_plan(&plan, &refit, config, ctx)?;
    let certificate = certify_target_refutations(&target, &run.reports)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(TargetRefutationOutcome { plan, run, certificate })
}

struct PreparedRefit<'a> {
    prepared: &'a PreparedStudy,
    result: &'a StudyResult,
    data: PreparedTargetData<'a>,
    target: RefutationTarget,
    original: Vec<f64>,
    treatment: VariableId,
    outcome: VariableId,
    options: TargetRefitOptions,
}

fn unsupported(reason: &str, next_attempt: &str) -> TargetRefitError {
    TargetRefitError::Unsupported {
        reason: Arc::from(reason),
        next_attempt: Arc::from(next_attempt),
    }
}

fn failed(error: &CausalError) -> TargetRefitError {
    TargetRefitError::Failed(ValidationError::estimation_msg(error.to_string()))
}

fn float_column(storage: &antecedent_data::OwnedColumnarStorage, id: VariableId) -> Option<&[f64]> {
    match storage.columns().get(id.as_usize()) {
        Some(OwnedColumn::Float64(column)) => Some(&column.values[..]),
        _ => None,
    }
}

fn mask_from(keep: &[bool]) -> Result<ValidityBitmap, TargetRefitError> {
    let mut bytes = vec![0_u8; keep.len().div_ceil(8)];
    for (i, &k) in keep.iter().enumerate() {
        if k {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    ValidityBitmap::from_bytes(bytes, keep.len())
        .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))
}

impl PreparedRefit<'_> {
    fn reestimate_tabular(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let refitted = self.prepared.estimate(data, ctx).map_err(|error| failed(&error))?;
        self.read_back(&refitted)
    }

    fn reestimate_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let refitted = self.prepared.estimate_series(data, ctx).map_err(|error| failed(&error))?;
        self.read_back(&refitted)
    }

    /// The refitted target vector, in the original coordinate order.
    fn read_back(&self, refitted: &StudyResult) -> Result<Vec<f64>, TargetRefitError> {
        let (family, labels, values) =
            result_target_values(refitted).map_err(|error| failed(&error))?;
        if family != self.target.family() || labels.as_slice() != self.target.labels() {
            return Err(TargetRefitError::Failed(ValidationError::estimation_msg(
                "the refitted result reports a different target than the original",
            )));
        }
        Ok(values)
    }

    fn permuted(values: &[f64], rng: &mut CausalRng, circular: bool) -> Arc<[f64]> {
        let n = values.len();
        if circular && n > 3 {
            // A circular shift by between a quarter and three quarters of the series keeps the
            // series' own dependence and breaks its alignment with the other columns.
            let span = n / 2;
            let offset = n / 4 + unbiased_index(rng, span.max(1));
            (0..n).map(|i| values[(i + offset) % n]).collect()
        } else {
            let mut out = values.to_vec();
            shuffle(rng, &mut out);
            out.into()
        }
    }

    fn tabular_refit(
        &self,
        data: &TabularData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let storage = data.storage();
        let replace = |id: VariableId, values: Arc<[f64]>| {
            data.with_replaced_float(id, values)
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))
        };
        let column = |id: VariableId| {
            float_column(storage, id).ok_or_else(|| {
                unsupported(
                    "the perturbed column is not a float64 column",
                    "encode the treatment, outcome and controls as float64 columns",
                )
            })
        };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(column(self.treatment)?, rng, false);
                self.reestimate_tabular(&replace(self.treatment, values)?, ctx)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(column(self.outcome)?, rng, false);
                self.reestimate_tabular(&replace(self.outcome, values)?, ctx)
            }
            TargetPerturbation::RandomCommonCause { .. } => Err(unsupported(
                "a prepared handle executes a fixed schema and graph, so a simulated common \
                 cause cannot be added to its adjustment",
                "declare the simulated cause as a parent of treatment and outcome and prepare \
                 that study, or run the scalar RandomCommonCause refuter on a linear \
                 average-effect route",
            )),
            TargetPerturbation::Subset { fraction, .. } => {
                let n = data.row_count();
                if n < 2 {
                    return Err(unsupported("a subset needs at least two rows", "more data"));
                }
                let mut rows: Vec<usize> = (0..n).collect();
                shuffle(rng, &mut rows);
                let kept = ((fraction * n as f64).round() as usize).clamp(1, n);
                let mut keep = vec![false; n];
                for &row in &rows[..kept] {
                    keep[row] = true;
                }
                let masked = data
                    .with_analysis_mask(mask_from(&keep)?)
                    .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
                self.reestimate_tabular(&masked, ctx)
            }
            TargetPerturbation::Bootstrap { .. } => {
                self.reestimate_tabular(&Self::bootstrap(data, rng)?, ctx)
            }
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let Some(control) = self.options.negative_control_exposure else {
                    return Err(unsupported(
                        "no negative-control exposure is declared",
                        "declare a negative-control exposure (a variable known not to cause the \
                         outcome) in TargetRefitOptions",
                    ));
                };
                let swapped = replace(self.treatment, Arc::from(column(control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate_tabular(&data, ctx)
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let Some(control) = self.options.negative_control_outcome else {
                    return Err(unsupported(
                        "no negative-control outcome is declared",
                        "declare a negative-control outcome (a variable the treatment is known \
                         not to cause) in TargetRefitOptions",
                    ));
                };
                let swapped = replace(self.outcome, Arc::from(column(control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate_tabular(&data, ctx)
            }
            TargetPerturbation::PreTrend { .. } | TargetPerturbation::Anticipation { .. } => {
                Err(unsupported(
                    "pre-trend and anticipation checks need time-ordered data",
                    "prepare the study on series or panel data",
                ))
            }
        }
    }

    /// Row bootstrap of a table whose columns are all complete float64 columns.
    fn bootstrap(data: &TabularData, rng: &mut CausalRng) -> Result<TabularData, TargetRefitError> {
        let storage = data.storage();
        if storage.analysis_mask().is_some() || storage.weights().is_some() {
            return Err(unsupported(
                "a row bootstrap of a masked or weighted table would change its population",
                "refute the unmasked, unweighted table, or use subset stability",
            ));
        }
        let n = data.row_count();
        let rows: Vec<usize> = (0..n).map(|_| unbiased_index(rng, n)).collect();
        let mut replacements = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "the row bootstrap gathers float64 columns only",
                    "encode every column as float64, or use subset stability",
                ));
            };
            if !column.validity.is_all_valid() {
                return Err(unsupported(
                    "the row bootstrap needs complete columns",
                    "resolve missing values first (a missing-data policy), or use subset \
                     stability",
                ));
            }
            let gathered: Arc<[f64]> = rows.iter().map(|&i| column.values[i]).collect();
            replacements.push((column.id, gathered));
        }
        data.with_replaced_floats(&replacements)
            .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))
    }

    fn series_refit(
        &self,
        data: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let storage = data.storage();
        let column = |id: VariableId| {
            float_column(storage, id).ok_or_else(|| {
                unsupported(
                    "the perturbed column is not a float64 column",
                    "encode the treatment and outcome as float64 columns",
                )
            })
        };
        let replace = |id: VariableId, values: Arc<[f64]>| {
            data.with_replaced_float(id, values)
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))
        };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(column(self.treatment)?, rng, true);
                self.reestimate_series(&replace(self.treatment, values)?, ctx)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(column(self.outcome)?, rng, true);
                self.reestimate_series(&replace(self.outcome, values)?, ctx)
            }
            TargetPerturbation::Subset { fraction, .. } => {
                // A contiguous window keeps every lag the procedure reads inside the window.
                let n = data.row_count();
                if n < 2 {
                    return Err(unsupported("a window needs at least two periods", "more data"));
                }
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n - width + 1);
                let keep: Vec<bool> = (0..n).map(|i| i >= start && i < start + width).collect();
                let masked = data
                    .with_analysis_mask(mask_from(&keep)?)
                    .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
                self.reestimate_series(&masked, ctx)
            }
            TargetPerturbation::Bootstrap { .. } => Err(unsupported(
                "rebuilding lags on a block-resampled raw series pairs values across block \
                 joins, which is not the series' own law",
                "the route's block bootstrap of its lagged design (the temporal path of the \
                 scalar bootstrap refuter), or subset stability",
            )),
            TargetPerturbation::RandomCommonCause { .. } => Err(unsupported(
                "a prepared handle executes a fixed schema and graph, so a simulated common \
                 cause cannot be added to its adjustment",
                "declare the simulated cause in the temporal graph and prepare that study",
            )),
            TargetPerturbation::NegativeControlExposure { .. }
            | TargetPerturbation::NegativeControlOutcome { .. } => Err(unsupported(
                "a negative control on one series has no resampling law for its null spread",
                "prepare the negative control on panel data, or test it on independent rows",
            )),
            TargetPerturbation::PreTrend { .. } | TargetPerturbation::Anticipation { .. } => {
                Err(unsupported(
                    "a single series has no treatment onset to place leads against, and under \
                     outcome-to-treatment feedback a lead is not a placebo",
                    "the event-study leads of a panel design (the quasi-experimental routes), \
                     which carry their own pre-trend and anticipation assumptions",
                ))
            }
        }
    }

    fn mediation_check(&self) -> Result<TargetRefutationReport, TargetRefitError> {
        let Some(mediator) = self.options.mediator else {
            return Err(unsupported(
                "no mediator is declared for the sequential-ignorability range",
                "declare the single mediator in TargetRefitOptions",
            ));
        };
        let PreparedTargetData::Tabular(data) = self.data else {
            return Err(unsupported(
                "the closed-form sequential-ignorability range is for independent rows",
                "on series, the temporal mediation suite's placebo-mediator and contiguous-window \
                 checks",
            ));
        };
        let storage = data.storage();
        let read = |id: VariableId| {
            float_column(storage, id).ok_or_else(|| {
                unsupported(
                    "mediation sensitivity reads float64 columns",
                    "encode treatment, mediator, outcome and adjustment as float64 columns",
                )
            })
        };
        let mut ids = vec![self.treatment, mediator, self.outcome];
        ids.extend(self.result.estimand.adjustment_set.iter().copied());
        let columns: Vec<&[f64]> = ids.iter().map(|&id| read(id)).collect::<Result<_, _>>()?;
        let n = data.row_count();
        let mask = storage.analysis_mask();
        let complete: Vec<usize> = (0..n)
            .filter(|&i| {
                mask.is_none_or(|m| m.is_valid(i)) && columns.iter().all(|c| c[i].is_finite())
            })
            .collect();
        let take = |c: &[f64]| complete.iter().map(|&i| c[i]).collect::<Vec<f64>>();
        let covariates: Vec<Vec<f64>> = columns[3..].iter().map(|c| take(c)).collect();
        let summary = mediation_sensitivity_fit(
            &take(columns[0]),
            &take(columns[1]),
            &take(columns[2]),
            &covariates,
        )?;
        let break_even = summary.rho_tilde.abs();
        let passed = break_even >= self.options.mediation_rho_bar;
        let mut refuted = self.original.clone();
        if let Some(last) = refuted.last_mut() {
            *last = summary.acme(0.0)?;
        }
        Ok(TargetRefutationReport {
            target: self.target.clone(),
            kind: TargetRefuterKind::SequentialIgnorability,
            original: self.original.clone().into(),
            refuted: refuted.into(),
            statistic: JointStatistic {
                kind: JointStatisticKind::BreakEven,
                value: break_even,
                df: 1,
                p_value: None,
            },
            passed,
            informative: true,
            replicates: 0,
            failure: (!passed).then(|| {
                Arc::from(format!(
                    "a mediator-outcome error correlation of {:.3} explains the indirect effect \
                     away (bar {})",
                    summary.rho_tilde, self.options.mediation_rho_bar
                ))
            }),
        })
    }
}

impl TargetRefit for PreparedRefit<'_> {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        match self.data {
            PreparedTargetData::Tabular(data) => self.tabular_refit(data, perturbation, rng, ctx),
            PreparedTargetData::Series(data) => self.series_refit(data, perturbation, rng, ctx),
        }
    }

    fn native_check(
        &self,
        kind: TargetRefuterKind,
        _ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        match kind {
            TargetRefuterKind::SequentialIgnorability => self.mediation_check(),
            TargetRefuterKind::FiniteDifferenceConsistency | TargetRefuterKind::LocalLinearity => {
                Err(unsupported(
                    "a prepared response reports its derivative, not the fitted response the \
                     finite difference reads",
                    "finite_difference_consistency and local_linearity on the fitted response \
                     (antecedent.validation.derivative_consistency in Python)",
                ))
            }
            TargetRefuterKind::PriorSensitivity => Err(unsupported(
                "a prepared handle executes one prior",
                "prepare the study under each alternative prior and pass the posterior draws of \
                 the target to prior_sensitivity_target_check \
                 (antecedent.validation.prior_sensitivity_target in Python)",
            )),
            TargetRefuterKind::PosteriorPredictive => Err(unsupported(
                "a prepared handle does not simulate replicated data from its posterior",
                "pass posterior draws of the target and targets refitted on posterior-replicated \
                 data to posterior_predictive_target_check \
                 (antecedent.validation.posterior_predictive_target in Python)",
            )),
            _ => Err(unsupported("no native check for this refuter", "a replicate refuter")),
        }
    }
}
