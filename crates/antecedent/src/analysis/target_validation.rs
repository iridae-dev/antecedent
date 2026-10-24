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

use std::cmp::Ordering;
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    CausalQuery, CausalRng, ContinuousDomain, DerivativeScale, DerivativeWeighting,
    ExecutionContext, GridSpec, Intervention, Lag, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, Value, VariableId,
};
use antecedent_data::{
    OwnedColumn, PanelData, PanelUnit, TableView, TabularData, TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_kernels::{sample_gamma, shuffle, unbiased_index};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::RefutationReport;
use antecedent_validate::target_native::{
    DerivativeCheckPoint, DerivativeConsistency, LocalLinearity, PosteriorPredictiveTarget,
    PriorSensitivityTarget, certify_target_refutations, finite_difference_consistency,
    local_linearity, posterior_predictive_target_check, prior_sensitivity_target_check,
    refuse_scalar_reports_for_target, run_target_plan, target_refuter_stream,
};
pub use antecedent_validate::target_native::{
    JointStatistic, JointStatisticKind, RefutationTarget, TargetCertificate,
    TargetCertificationError, TargetFamily, TargetPerturbation, TargetQuadrant, TargetRefit,
    TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefutationReport,
    TargetRefutationRun, TargetRefuterKind, TargetRefuterRefusal, TargetSuite,
};
use antecedent_validate::{ValidationError, mediation_sensitivity_fit};

use super::builder::DataInput;
use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::lowered::{LoweredProgram, UncertaintyReading};
use crate::result::StudyResult;

/// Float columns to replace, each with its new values.
type ColumnReplacements = Vec<(VariableId, Arc<[f64]>)>;

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
        return TargetFamily::InterventionalLaw;
    }
    if result.anomaly().is_some() || result.change_attribution().is_some() {
        return TargetFamily::Attribution;
    }
    // A counterfactual reports its population mean: the per-unit effects number as many as the
    // scored units, far more coordinates than any replicate law can estimate a covariance for.
    TargetFamily::ScalarAverageEffect
}

/// The per-component vector of an anomaly or change attribution: for each scored target the
/// mean score and the mean Shapley contribution of every noise component; for a change
/// attribution the total change and every component's contribution.
fn attribution_values(result: &StudyResult) -> Result<(Vec<Arc<str>>, Vec<f64>), CausalError> {
    let mut labels: Vec<Arc<str>> = Vec::new();
    let mut values = Vec::new();
    if let Some(scores) = result.anomaly() {
        for score in scores {
            let rows = score.rows.len();
            let width = score.noise_components.len();
            if rows == 0 || score.noise_contributions.len() != rows * width {
                return Err(CausalError::Unsupported {
                    message: "an anomaly attribution without scored rows has no vector to refute",
                });
            }
            let target = score.target.raw();
            labels.push(Arc::from(format!("anomaly[{target}].score")));
            values.push(score.scores.iter().sum::<f64>() / rows as f64);
            for (j, component) in score.noise_components.iter().enumerate() {
                labels.push(Arc::from(format!(
                    "anomaly[{target}].component[{}]",
                    component.variable().raw()
                )));
                values.push(
                    (0..rows).map(|row| score.noise_contributions[row * width + j]).sum::<f64>()
                        / rows as f64,
                );
            }
        }
    } else if let Some(change) = result.change_attribution() {
        labels.push(Arc::from("change.total"));
        values.push(change.total_change);
        for contribution in change.contributions.iter() {
            labels.push(Arc::from(format!(
                "change.component[{}]",
                contribution.component.variable().raw()
            )));
            values.push(contribution.contribution);
        }
    }
    if values.is_empty() {
        return Err(CausalError::Unsupported {
            message: "the attribution result carries no component vector to refute",
        });
    }
    Ok((labels, values))
}

/// The probability atoms of an interventional distribution, all but the last (the others
/// determine it, and a vector that sums to one has a singular replicate covariance). A
/// distribution with a single atom, or a mean only, is read through its mean.
fn law_values(result: &StudyResult) -> Result<(Vec<Arc<str>>, Vec<f64>), CausalError> {
    let distribution = result.distribution().ok_or(CausalError::Unsupported {
        message: "no interventional distribution payload to refute",
    })?;
    let atoms = &distribution.atoms;
    if atoms.len() >= 2 {
        let kept = atoms.len() - 1;
        let labels = (0..kept).map(|i| Arc::from(format!("atom[{i}]"))).collect();
        let values = atoms[..kept].iter().map(|atom| atom.probability).collect();
        return Ok((labels, values));
    }
    if distribution.mean.is_finite() {
        return Ok((vec![Arc::from("mean")], vec![distribution.mean]));
    }
    Err(CausalError::Unsupported {
        message: "the interventional distribution has neither two atoms nor a finite mean to refute",
    })
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
#[allow(
    clippy::type_complexity,
    reason = "the public signature returns the family, labels and values as one tuple"
)]
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
            let value = match &response.estimate {
                ResponseIdentification::PointIdentified(value) => value,
                // A graph posterior that loses mass to atoms the identifier refuses publishes
                // its value conditional on the atoms that identify; the refits recompute that
                // same conditional value.
                ResponseIdentification::GraphDependent(_) => result
                    .structural_response
                    .as_ref()
                    .and_then(|mixture| mixture.conditional_on_identified.as_ref())
                    .ok_or_else(|| {
                        refuse(
                            "only a point-identified response has a target vector to refute; \
                             next attempt: refute each identified graph's response separately",
                        )
                    })?,
                _ => {
                    return Err(refuse(
                        "only a point-identified response has a target vector to refute; next \
                         attempt: refute each identified graph's response separately",
                    ));
                }
            };
            match value {
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
            }
        }
        TargetFamily::WorldValued => {
            return Err(refuse(
                "an interventional distribution is refuted through its mean response; next \
                 attempt: query the interventional mean as an InterventionResponse",
            ));
        }
        TargetFamily::Attribution => attribution_values(result)?,
        TargetFamily::InterventionalLaw => law_values(result)?,
        TargetFamily::ConditionalEffect => {
            let ite = result
                .counterfactual()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            labelled("unit", &ite.unit_effects)
        }
        _ if result.counterfactual().is_some() => {
            let ite = result
                .counterfactual()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            (vec![Arc::<str>::from("mean_ite")], vec![ite.mean_ite])
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
    /// A variable known not to cause the outcome, swapped in for the treatment. When `None`,
    /// the graph supplies one: a float variable outside the adjustment set that is not a
    /// descendant of the treatment and is d-separated from the outcome given the adjustment
    /// set (static graphs), or none of whose nodes reaches the outcome (temporal graphs).
    pub negative_control_exposure: Option<VariableId>,
    /// A variable the treatment is known not to cause, swapped in for the outcome. When
    /// `None`, the graph supplies one on a static graph (not a descendant of the treatment,
    /// d-separated from it given the adjustment set); on series and panels the default is the
    /// lagged placebo: the outcome moved back past the longest lag of the graph, which no
    /// later treatment can affect.
    pub negative_control_outcome: Option<VariableId>,
    /// The single mediator of a mediation target, for the sequential-ignorability range.
    pub mediator: Option<VariableId>,
    /// Smallest absolute break-even mediator-outcome error correlation that passes.
    pub mediation_rho_bar: f64,
    /// Strength with which the simulated common cause enters the outcome (and a non-binary
    /// treatment). Zero simulates an independent cause; either way the cause is declared in
    /// the graph as a parent of treatment and outcome and the program adjusts for it.
    pub common_cause_strength: f64,
    /// Alternative isotropic prior scales, as multiples of the study's prior scale.
    pub prior_scale_factors: Vec<f64>,
    /// Largest acceptable shift of the target posterior, in posterior standard deviations.
    pub prior_shift_tolerance: f64,
    /// Step of the derivative diagnostics; `None` takes a quarter of the treatment's sample
    /// standard deviation.
    pub derivative_step: Option<f64>,
    /// Relative tolerance of the finite-difference consistency check.
    pub derivative_tolerance: f64,
    /// Largest acceptable relative nonlinearity over the step.
    pub linearity_tolerance: f64,
    /// Draw the series bootstrap, pre-trend, anticipation and lagged-placebo refits as
    /// explicit gathered circular-block series and cut windows, as a route that reads no row
    /// weights requires, instead of as row multiplicities. A route that reads row weights
    /// keeps the multiplicities unless this is set.
    pub gathered_resamples: bool,
    /// The declared division of a class's mass among its DAG members, which a common cause
    /// declared in every atom of a class posterior needs. `None` refuses such a posterior.
    pub within_class: Option<WithinClassDistribution>,
}

/// How the mass of a Markov equivalence class divides among its DAG members, an assumption the
/// posterior does not state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WithinClassDistribution {
    /// Every DAG member of the class has the same share.
    UniformOverMembers,
}

/// One report on the whole target vector from the per-coordinate derivative checks: the
/// coordinates (horizons, or the outcomes and directions of a vector derivative) are the
/// evaluation points of a single check, in target order.
fn merge_derivative_checks(
    kind: TargetRefuterKind,
    coordinates: Vec<DerivativeCoordinate>,
    target: &RefutationTarget,
) -> Result<TargetRefutationReport, TargetRefitError> {
    if kind == TargetRefuterKind::FiniteDifferenceConsistency {
        let mut merged: Option<DerivativeConsistency> = None;
        for coordinate in coordinates {
            let DerivativeCoordinate::FiniteDifference(check) = coordinate else { continue };
            match &mut merged {
                None => merged = Some(check),
                Some(all) => {
                    all.points.extend(check.points);
                    all.reported.extend(check.reported);
                    all.finite_difference.extend(check.finite_difference);
                    all.truncation_error.extend(check.truncation_error);
                    all.standardized.extend(check.standardized);
                }
            }
        }
        let check = merged.ok_or_else(|| unsupported("no value", "a target"))?;
        Ok(check.to_report(target)?)
    } else {
        let mut merged: Option<LocalLinearity> = None;
        for coordinate in coordinates {
            let DerivativeCoordinate::Linearity(check) = coordinate else { continue };
            match &mut merged {
                None => merged = Some(check),
                Some(all) => {
                    all.points.extend(check.points);
                    all.slope.extend(check.slope);
                    all.curvature.extend(check.curvature);
                    all.nonlinearity.extend(check.nonlinearity);
                }
            }
        }
        let check = merged.ok_or_else(|| unsupported("no value", "a target"))?;
        Ok(check.to_report(target)?)
    }
}

/// A lagged-template mask repacked from `n` to `m` nodes: the bit of the edge `from -> to` at
/// lag `lag + 1` moves from `lag * n * n + from * n + to` to `lag * m * m + from * m + to`.
fn repack_lag_mask(old: u64, n: usize, m: usize, max_lag: usize) -> u64 {
    let mut packed = 0_u64;
    for bit in 0..max_lag * n * n {
        if (old >> bit) & 1 == 1 {
            let (lag, from, to) = (bit / (n * n), (bit % (n * n)) / n, bit % n);
            packed |= 1_u64 << (lag * m * m + from * m + to);
        }
    }
    packed
}

#[cfg(test)]
mod repack_tests {
    use antecedent_validate::target_native::{TargetFamily, TargetQuadrant};

    use super::*;

    /// A target with one coordinate per horizon.
    fn horizon_target(horizons: usize) -> RefutationTarget {
        RefutationTarget::new(
            TargetFamily::ResponseDerivative,
            TargetQuadrant::TEMPORAL_FREQUENTIST,
            (1..=horizons).map(|h| format!("horizon[{h}]")),
            [7; 32],
        )
        .unwrap()
    }

    /// One coordinate per horizon: a linear response with that horizon's slope.
    fn horizon_checks(
        kind: TargetRefuterKind,
        slopes: &[f64],
        reported: &[f64],
    ) -> Vec<DerivativeCoordinate> {
        slopes
            .iter()
            .zip(reported)
            .map(|(&slope, &value)| {
                if kind == TargetRefuterKind::FiniteDifferenceConsistency {
                    DerivativeCoordinate::FiniteDifference(
                        finite_difference_consistency(
                            |offset| Ok(1.0 + slope * offset),
                            &[DerivativeCheckPoint { x: 0.0, reported: value }],
                            0.5,
                            0.05,
                        )
                        .unwrap(),
                    )
                } else {
                    DerivativeCoordinate::Linearity(
                        local_linearity(|offset| Ok(1.0 + slope * offset), &[0.0], 0.5, 0.25, 1e-9)
                            .unwrap(),
                    )
                }
            })
            .collect()
    }

    #[test]
    fn horizons_with_distinct_slopes_merge_into_one_report_on_the_horizon_target() {
        let target = horizon_target(2);
        let kind = TargetRefuterKind::FiniteDifferenceConsistency;
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.0]);
        let sound = merge_derivative_checks(kind, checks, &target).unwrap();
        assert!(sound.passed, "{:?}", sound.failure);
        assert!((sound.refuted[0] - 2.0).abs() < 1e-9 && (sound.refuted[1] - 3.0).abs() < 1e-9);
        assert_eq!(&*sound.original, &[2.0, 3.0]);
        // One wrong horizon fails the whole vector.
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.6]);
        assert!(!merge_derivative_checks(kind, checks, &target).unwrap().passed);
        // A report with another number of coordinates than the target is refused.
        let checks = horizon_checks(kind, &[2.0], &[2.0]);
        assert!(merge_derivative_checks(kind, checks, &target).is_err());
        // Local linearity merges the same way: a linear response is linear at every horizon.
        let kind = TargetRefuterKind::LocalLinearity;
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.0]);
        let linear = merge_derivative_checks(kind, checks, &target).unwrap();
        assert!(linear.passed);
        assert_eq!(linear.original.len(), 2);
    }

    #[test]
    fn a_lagged_edge_keeps_its_lag_source_and_target_when_a_node_is_added() {
        // Two nodes, two lags: bit (lag, from, to) = lag * 4 + from * 2 + to.
        let old = (1_u64 << 1) | (1_u64 << (4 + 2)); // lag 1: 0 -> 1; lag 2: 1 -> 0
        let packed = repack_lag_mask(old, 2, 3, 2);
        assert_eq!(packed, (1_u64 << 1) | (1_u64 << (9 + 3)));
        assert_eq!(repack_lag_mask(0, 2, 3, 2), 0);
    }
}

/// Most DAG members of one class the common-cause refuter enumerates.
const WITHIN_CLASS_MEMBER_CAP: usize = 512;

impl Default for TargetRefitOptions {
    fn default() -> Self {
        Self {
            negative_control_exposure: None,
            negative_control_outcome: None,
            mediator: None,
            mediation_rho_bar: 0.1,
            common_cause_strength: 0.0,
            prior_scale_factors: vec![0.25, 4.0],
            prior_shift_tolerance: 0.5,
            derivative_step: None,
            derivative_tolerance: 0.1,
            linearity_tolerance: 0.25,
            gathered_resamples: false,
            within_class: None,
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
    /// A panel of unit series.
    Panel(&'a PanelData),
    /// Independent environments, each one series of a shared schema.
    MultiEnv(&'a antecedent_data::MultiEnvironmentData),
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
/// `data` must be the data `result` was computed on (the retained data of the handle). Every
/// refit re-executes a compiled program in full: the handle's own program on perturbed data,
/// or, where the refuter changes the scientific problem (a simulated common cause declared in
/// the graph, an alternative prior, the fitted response curve of a derivative), a study
/// compiled and prepared from the handle's study with that one change. A prepared handle whose
/// study requests a scalar refutation suite pays for it on every refit of its own program:
/// prepare with no scalar suite for target refutation.
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
    // An attribution has no treatment; its stability refuters never read one, so the
    // outcome stands in for the unused field.
    let treatment = match result.treatment {
        Some(treatment) => treatment,
        None if matches!(
            target.family(),
            TargetFamily::Attribution | TargetFamily::InterventionalLaw
        ) =>
        {
            result.outcome
        }
        None => {
            return Err(CausalError::Unsupported {
                message: "target refutation needs a result with a treatment",
            });
        }
    };
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
        replicates: config.replicates,
        alpha: config.alpha,
        curve: std::cell::OnceCell::new(),
    };
    let run = run_target_plan(&plan, &refit, config, ctx)?;
    let certificate = certify_target_refutations(&target, &run.reports)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(TargetRefutationOutcome { plan, run, certificate })
}

/// Data of one refit, owned.
enum Modal {
    Tabular(TabularData),
    Series(TimeSeriesData),
    Panel(PanelData),
    MultiEnv(antecedent_data::MultiEnvironmentData),
}

/// One coordinate's derivative check, before the coordinates are merged into one report.
enum DerivativeCoordinate {
    FiniteDifference(DerivativeConsistency),
    Linearity(LocalLinearity),
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
    replicates: u32,
    alpha: f64,
    /// Fitted response curve of a derivative, `(point, value per horizon)`, computed once.
    curve: std::cell::OnceCell<Vec<(f64, Vec<f64>)>>,
}

fn unsupported(reason: impl Into<Arc<str>>, next_attempt: &str) -> TargetRefitError {
    TargetRefitError::Unsupported { reason: reason.into(), next_attempt: Arc::from(next_attempt) }
}

/// Copy of an explicit coefficient prior with every Gaussian coefficient standard deviation
/// multiplied by `factor`; means, residual-variance and restriction specs are kept.
fn scale_prior_deviations(
    prior: &antecedent_prob::PriorSet,
    factor: f64,
) -> Result<antecedent_prob::PriorSet, TargetRefitError> {
    let mut scaled = prior.clone();
    let mut found = false;
    for spec in &mut scaled.specs {
        if let antecedent_prob::PriorSpec::GaussianCoefficients(coefficients) = spec {
            coefficients.variance =
                coefficients.variance.iter().map(|variance| variance * factor * factor).collect();
            found = true;
        }
    }
    if found {
        Ok(scaled)
    } else {
        Err(unsupported(
            "the explicit prior has no Gaussian coefficient block to scale",
            "declare a Gaussian coefficient prior, or leave the isotropic prior scale",
        ))
    }
}

fn failed(error: &CausalError) -> TargetRefitError {
    TargetRefitError::Failed(ValidationError::estimation_msg(error.to_string()))
}

fn data_failed(error: antecedent_data::DataError) -> TargetRefitError {
    TargetRefitError::Failed(ValidationError::from(error))
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
    ValidityBitmap::from_bytes(bytes, keep.len()).map_err(data_failed)
}

/// `values` moved by `shift` periods (`out[t] = values[t + shift]`) and the rows that exist.
fn shifted(values: &[f64], shift: isize) -> (Arc<[f64]>, Vec<bool>) {
    let n = values.len();
    let mut out = vec![0.0; n];
    let mut keep = vec![false; n];
    for t in 0..n {
        let Some(source) = t.checked_add_signed(shift).filter(|s| *s < n) else { continue };
        out[t] = values[source];
        keep[t] = true;
    }
    (out.into(), keep)
}

/// Row weights that drop the design rows ending outside `[from, to)`: a shifted column has
/// no value there, and a zero weight removes the window without opening a gap in the series.
fn window_weights(n: usize, from: usize, to: usize) -> Arc<[f64]> {
    (0..n).map(|t| if t >= from && t < to { 1.0 } else { 0.0 }).collect()
}

fn times(a: &Arc<[f64]>, b: Option<Arc<[f64]>>) -> Arc<[f64]> {
    match b {
        Some(b) => a.iter().zip(b.iter()).map(|(x, y)| x * y).collect(),
        None => Arc::clone(a),
    }
}

/// Circular-block bootstrap of a series as row multiplicities: the design row ending at each
/// period keeps its own lag window, so no lag is read across a block join.
fn block_weights(
    n: usize,
    span: usize,
    rng: &mut CausalRng,
) -> Result<Arc<[f64]>, TargetRefitError> {
    let length = antecedent_data::circular_block_length(span, n);
    let mut index = vec![0_usize; n];
    antecedent_data::fill_circular_block_indexes(n, length, rng, &mut index)
        .map_err(data_failed)?;
    let mut weights = vec![0.0; n];
    for i in index {
        weights[i] += 1.0;
    }
    Ok(weights.into())
}

fn series_with(
    series: &TimeSeriesData,
    replacements: &[(VariableId, Arc<[f64]>)],
    keep: Option<&[bool]>,
    weights: Option<Arc<[f64]>>,
) -> Result<TimeSeriesData, TargetRefitError> {
    let mut out = series.clone();
    if !replacements.is_empty() {
        let table = TabularData::new(series.storage().clone())
            .with_replaced_floats(replacements)
            .map_err(data_failed)?;
        out = TimeSeriesData::try_new(table.storage().clone(), series.time_index().clone())
            .map_err(data_failed)?;
    }
    if let Some(keep) = keep {
        out = out.with_analysis_mask(mask_from(keep)?).map_err(data_failed)?;
    }
    if let Some(weights) = weights {
        let combined: Arc<[f64]> = match out.storage().weights() {
            Some(existing) => existing.iter().zip(weights.iter()).map(|(a, b)| a * b).collect(),
            None => weights,
        };
        out = out.with_weights(combined).map_err(data_failed)?;
    }
    Ok(out)
}

/// The periods `[start, end)` of a series as a complete, shorter series: every column is cut
/// to the same window, so the series stays lag-aligned and no analysis mask hides a row
/// (lagged temporal routes refuse an incomplete series). `None` when a column is not float64.
fn truncated_series(
    series: &TimeSeriesData,
    start: usize,
    end: usize,
) -> Result<Option<TimeSeriesData>, TargetRefitError> {
    let storage = series.storage();
    let n = series.row_count();
    if !(start < end && end <= n) {
        return Err(unsupported("the window holds no period", "a longer series"));
    }
    let rows: Vec<usize> = (start..end).collect();
    let mut columns = Vec::with_capacity(storage.columns().len());
    for column in storage.columns() {
        let OwnedColumn::Float64(column) = column else { return Ok(None) };
        let values: Arc<[f64]> = Arc::from(&column.values[start..end]);
        let validity = column.validity.gather_rows(&rows).map_err(data_failed)?;
        columns.push(OwnedColumn::Float64(
            antecedent_data::Float64Column::new(column.id, values, validity)
                .map_err(data_failed)?,
        ));
    }
    let mask =
        storage.analysis_mask().map(|m| m.gather_rows(&rows)).transpose().map_err(data_failed)?;
    let weights = storage.weights().map(|w| Arc::<[f64]>::from(&w[start..end]));
    let storage = antecedent_data::OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        columns,
        mask,
        weights,
    )
    .map_err(data_failed)?;
    let index = antecedent_data::TimeIndex {
        regularity: series.time_index().regularity.clone(),
        length: end - start,
    };
    TimeSeriesData::try_new(storage, index).map(Some).map_err(data_failed)
}

/// Longest lag of a temporal graph (zero for a static one).
fn max_graph_lag(study: &crate::analysis::execute::Study) -> usize {
    study.graph.as_temporal_dag().map_or(0, |graph| {
        graph
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { lag, .. } => Some(lag.raw() as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    })
}

/// Dense node of a static variable in a DAG.
fn static_node(dag: &Dag, variable: VariableId) -> Option<DenseNodeId> {
    dag.nodes()
        .iter()
        .position(|node| *node == antecedent_core::NodeRef::Static(variable))
        .and_then(|index| u32::try_from(index).ok().map(DenseNodeId::from_raw))
}

/// Nodes of a variable in a temporal DAG.
fn lagged_nodes(graph: &TemporalDag, variable: VariableId) -> Vec<(DenseNodeId, usize)> {
    graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match node {
            antecedent_core::NodeRef::Lagged { variable: v, lag } if *v == variable => {
                u32::try_from(index)
                    .ok()
                    .map(|raw| (DenseNodeId::from_raw(raw), lag.raw() as usize))
            }
            _ => None,
        })
        .collect()
}

impl PreparedRefit<'_> {
    fn study(&self) -> &crate::analysis::execute::Study {
        self.prepared.study()
    }

    fn adjustment(&self) -> Vec<VariableId> {
        self.result.estimand.adjustment_set.iter().copied().collect()
    }

    fn storage(&self) -> &antecedent_data::OwnedColumnarStorage {
        match self.data {
            PreparedTargetData::Tabular(data) => data.storage(),
            PreparedTargetData::Series(data) => data.storage(),
            PreparedTargetData::Panel(data) => data.units()[0].series.storage(),
            PreparedTargetData::MultiEnv(data) => data.environments()[0].storage(),
        }
    }

    fn is_float(&self, id: VariableId) -> bool {
        float_column(self.storage(), id).is_some()
    }

    /// Re-execute this handle's program on `data`.
    fn reestimate(
        &self,
        data: &Modal,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let refitted = match data {
            Modal::Tabular(data) => self.prepared.estimate(data, ctx),
            Modal::Series(data) => self.prepared.estimate_series(data, ctx),
            Modal::Panel(data) => self.prepared.estimate_panel(data, ctx),
            Modal::MultiEnv(data) => self.prepared.estimate_multi_env(data, ctx),
        }
        .map_err(|error| failed(&error))?;
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

    /// Prepare `study` (a one-change copy of this handle's study) on `data` and execute it.
    fn compile_and_run(
        mut study: crate::analysis::execute::Study,
        data: &Modal,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        study.refute = crate::analysis::builder::RefuteSuite::None;
        study.data = match data {
            Modal::Tabular(d) => DataInput::Tabular(d.clone()),
            Modal::Series(d) => match &study.data {
                DataInput::Event(_) => DataInput::Event(d.clone()),
                _ => DataInput::Temporal(d.clone()),
            },
            Modal::Panel(d) => DataInput::Panel(d.clone()),
            Modal::MultiEnv(d) => DataInput::MultiEnv(d.clone()),
        };
        let prepared = study.prepare(ctx)?;
        match data {
            Modal::Tabular(d) => prepared.estimate(d, ctx),
            Modal::Series(d) => prepared.estimate_series(d, ctx),
            Modal::Panel(d) => prepared.estimate_panel(d, ctx),
            Modal::MultiEnv(d) => prepared.estimate_multi_env(d, ctx),
        }
    }

    fn permuted(values: &[f64], rng: &mut CausalRng, circular: bool) -> Arc<[f64]> {
        let n = values.len();
        if circular && n > 3 {
            // A circular shift by between a quarter and three quarters of the series keeps the
            // series' own dependence and breaks its alignment with the other columns.
            let offset = n / 4 + unbiased_index(rng, (n / 2).max(1));
            (0..n).map(|i| values[(i + offset) % n]).collect()
        } else {
            let mut out = values.to_vec();
            shuffle(rng, &mut out);
            out.into()
        }
    }

    fn column_of(
        storage: &antecedent_data::OwnedColumnarStorage,
        id: VariableId,
    ) -> Result<&[f64], TargetRefitError> {
        float_column(storage, id).ok_or_else(|| {
            unsupported(
                "the perturbed column is not a float64 column",
                "encode the treatment, outcome and controls as float64 columns",
            )
        })
    }

    /// Negative-control exposure: declared, or the graph's first candidate.
    fn control_exposure(&self) -> Result<VariableId, TargetRefitError> {
        if let Some(id) = self.options.negative_control_exposure {
            return Ok(id);
        }
        let adjustment = self.adjustment();
        let n = self.storage().columns().len();
        let candidates = (0..n)
            .filter_map(|i| u32::try_from(i).ok().map(VariableId::from_raw))
            .filter(|v| *v != self.treatment && *v != self.outcome && !adjustment.contains(v))
            .filter(|v| self.is_float(*v));
        let graph = &self.study().graph;
        for v in candidates {
            let ok = if let Some(dag) = graph.as_dag() {
                let (Some(t), Some(y), Some(x)) = (
                    static_node(dag, self.treatment),
                    static_node(dag, self.outcome),
                    static_node(dag, v),
                ) else {
                    continue;
                };
                let z: Vec<DenseNodeId> =
                    adjustment.iter().filter_map(|a| static_node(dag, *a)).collect();
                let mut ws = DSeparationWorkspace::default();
                !dag.reaches(t, x)
                    && !dag.reaches(x, y)
                    && dag.is_d_separated(x, y, &z, &mut ws).unwrap_or(false)
            } else if let Some(tdag) = graph.as_temporal_dag() {
                let ys = lagged_nodes(tdag, self.outcome);
                let ts = lagged_nodes(tdag, self.treatment);
                let xs = lagged_nodes(tdag, v);
                !xs.is_empty()
                    && xs.iter().all(|(x, _)| ys.iter().all(|(y, _)| !tdag.reaches(*x, *y)))
                    && ts.iter().all(|(t, _)| xs.iter().all(|(x, _)| !tdag.reaches(*t, *x)))
            } else {
                false
            };
            if ok {
                return Ok(v);
            }
        }
        Err(unsupported(
            "the graph supplies no negative-control exposure: no float variable outside the \
             adjustment set is separated from the outcome and unaffected by the treatment",
            "declare a negative-control exposure in TargetRefitOptions",
        ))
    }

    /// Negative-control outcome on a static graph: declared, or the graph's first candidate.
    fn control_outcome(&self) -> Result<VariableId, TargetRefitError> {
        if let Some(id) = self.options.negative_control_outcome {
            return Ok(id);
        }
        let adjustment = self.adjustment();
        let Some(dag) = self.study().graph.as_dag() else {
            return Err(unsupported(
                "the graph is not a DAG, so no negative-control outcome can be read from it",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
        };
        let n = self.storage().columns().len();
        let (Some(t), Some(_)) = (static_node(dag, self.treatment), static_node(dag, self.outcome))
        else {
            return Err(unsupported(
                "the treatment or outcome is not a node of the graph",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
        };
        let z: Vec<DenseNodeId> = adjustment.iter().filter_map(|a| static_node(dag, *a)).collect();
        for i in 0..n {
            let Ok(raw) = u32::try_from(i) else { continue };
            let v = VariableId::from_raw(raw);
            if v == self.treatment
                || v == self.outcome
                || adjustment.contains(&v)
                || !self.is_float(v)
            {
                continue;
            }
            let Some(w) = static_node(dag, v) else { continue };
            let mut ws = DSeparationWorkspace::default();
            if !dag.reaches(t, w) && dag.is_d_separated(t, w, &z, &mut ws).unwrap_or(false) {
                return Ok(v);
            }
        }
        Err(unsupported(
            "the graph supplies no negative-control outcome: no float variable outside the \
             adjustment set is unaffected by and separated from the treatment",
            "declare a negative-control outcome in TargetRefitOptions",
        ))
    }

    /// The graph posterior with a simulated common cause `u` declared in every atom.
    ///
    /// A DAG atom, static or a lagged template, is repacked onto one more node, `u`, which is
    /// made a contemporaneous parent of the treatment and of the outcome; the atom weights are
    /// unchanged, so the posterior is the same mixture over structures with one root cause
    /// added to each.
    ///
    /// A CPDAG atom is a class, and the class of the DAG with `u` added depends on which member
    /// of the class is meant: an added parent can create a collider with a parent the treatment
    /// already has. The posterior does not say how a class's mass divides among its members, so
    /// this needs the declared assumption `WithinClassDistribution::UniformOverMembers`: every
    /// atom's DAG members are enumerated (at most [`WITHIN_CLASS_MEMBER_CAP`], else the class is
    /// refused), each member gets an equal share of the atom's mass, the common cause is added to
    /// each, and the shares are collapsed to the classes of the augmented DAGs. The assumption is
    /// written into the posterior's diagnostics. A PAG atom or a lagged class posterior is
    /// refused.
    #[allow(
        clippy::too_many_lines,
        reason = "one repacking of the atoms in three shapes: DAG, lagged template and class"
    )]
    fn posterior_with_common_cause(
        &self,
        posterior: &antecedent_discovery::GraphPosterior,
        u: VariableId,
    ) -> Result<antecedent_discovery::GraphPosterior, TargetRefitError> {
        use antecedent_discovery::GraphPosteriorAtomKind as Kind;
        use antecedent_discovery::adjacency_mask::{has_edge, set_edge};
        let n = posterior.n_vars;
        let m = n + 1;
        let refuse = |reason: &'static str| {
            unsupported(reason, "refute one accepted structure of the posterior")
        };
        if u.as_usize() != n || m > antecedent_discovery::MAX_POSTERIOR_NODES {
            return Err(refuse(
                "the posterior's nodes are not the study's columns, or the added node does not fit \
                 the atom packing",
            ));
        }
        let (t, y) = (self.treatment.as_usize(), self.outcome.as_usize());
        if t >= n || y >= n {
            return Err(unsupported(
                "the treatment or outcome is not a node of the posterior",
                "a posterior that names both",
            ));
        }
        let lags = posterior.max_lag.map(|lag| lag as usize);
        // Repack a contemporaneous atom onto `m` nodes and add the common cause.
        let repack = |atom: antecedent_discovery::AdjacencyMask| {
            let mut packed = antecedent_discovery::AdjacencyMask::EMPTY;
            for i in 0..n {
                for j in 0..n {
                    if i != j && has_edge(atom, n, i, j) {
                        packed = set_edge(packed, m, i, j, true);
                    }
                }
            }
            packed = set_edge(packed, m, n, t, true);
            set_edge(packed, m, n, y, true)
        };
        let mut atoms: Vec<antecedent_discovery::AdjacencyMask> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut note: Option<&'static str> = None;
        match posterior.atom_kind {
            Kind::Dag => {
                atoms = posterior.atoms.iter().map(|&atom| repack(atom)).collect();
                weights = posterior.weights.to_vec();
            }
            Kind::Cpdag if lags.is_none() => {
                if self.options.within_class != Some(WithinClassDistribution::UniformOverMembers) {
                    return Err(refuse(
                        "a class atom does not say how its mass divides among its DAG members, \
                         and the class of an added common cause depends on the member; declare \
                         WithinClassDistribution::UniformOverMembers to assume an even division",
                    ));
                }
                for (&atom, &weight) in posterior.atoms.iter().zip(posterior.weights.iter()) {
                    let cpdag =
                        antecedent_discovery::adjacency_mask::cpdag_from_adjacency_mask(atom, n)
                            .map_err(|error| {
                                TargetRefitError::Failed(ValidationError::from(error))
                            })?;
                    let mut members = antecedent_graph::CpdagCompletionSampler::new(
                        cpdag,
                        WITHIN_CLASS_MEMBER_CAP + 1,
                    )
                    .map_err(|error| {
                        TargetRefitError::Failed(ValidationError::estimation_msg(error.to_string()))
                    })?;
                    let dags: Vec<antecedent_graph::Dag> =
                        members.by_ref().map(|completion| completion.graph).collect();
                    if dags.is_empty() || dags.len() > WITHIN_CLASS_MEMBER_CAP {
                        return Err(refuse(
                            "a class has no enumerated DAG members or more than the cap, so an \
                             even division of its mass is not defined",
                        ));
                    }
                    let share = weight / dags.len() as f64;
                    for dag in &dags {
                        let mut mask = antecedent_discovery::AdjacencyMask::EMPTY;
                        for edge in dag.edges() {
                            if let Some((from, to)) = edge.parent_child() {
                                mask = set_edge(mask, n, from.as_usize(), to.as_usize(), true);
                            }
                        }
                        atoms.push(repack(mask));
                        weights.push(share);
                    }
                }
                note = Some(
                    "declared assumption: the mass of each class divides evenly among its DAG \
                     members before the common cause is added",
                );
            }
            _ => {
                return Err(refuse(
                    "a PAG posterior or a lagged class posterior has no DAG members to add the \
                     common cause to",
                ));
            }
        }
        let total: f64 = weights.iter().sum();
        let mut edge = vec![0.0; m * m];
        for (&atom, &weight) in atoms.iter().zip(&weights) {
            for i in 0..m {
                for j in 0..m {
                    if i != j && has_edge(atom, m, i, j) {
                        edge[i * m + j] += weight / total;
                    }
                }
            }
        }
        let weights: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let mut diagnostics = posterior.diagnostics.clone();
        if let Some(note) = note {
            diagnostics.notes.push(Arc::from(note));
        }
        let mut rebuilt = antecedent_discovery::GraphPosterior::from_atoms(
            m,
            weights.clone(),
            atoms,
            edge.clone(),
            edge,
            antecedent_discovery::SamplingSummary {
                ess: posterior.ess,
                diagnostics,
                rejected_invalid: posterior.rejected_invalid,
            },
        )
        .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        if let (Some(max_lag), Some(lag_masks)) = (lags, posterior.lag_masks.as_ref()) {
            // Lagged edges keep their sources and targets; the added node has none, and the bit
            // of an edge moves with the node count.
            if max_lag * m * m > 64 {
                return Err(refuse(
                    "the lagged template does not fit its packing with one more node",
                ));
            }
            let packed: Vec<u64> =
                lag_masks.iter().map(|&old| repack_lag_mask(old, n, m, max_lag)).collect();
            let mut lagged = vec![0.0; max_lag * m * m];
            for (&mask, &weight) in packed.iter().zip(&weights) {
                for (bit, cell) in lagged.iter_mut().enumerate() {
                    if (mask >> bit) & 1 == 1 {
                        *cell += weight;
                    }
                }
            }
            rebuilt = rebuilt
                .with_lagged_marginals(u32::try_from(max_lag).unwrap_or(u32::MAX), lagged)
                .and_then(|post| post.with_lag_masks(packed))
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        }
        if posterior.atom_kind == Kind::Cpdag {
            rebuilt = rebuilt
                .to_equivalence_classes()
                .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        }
        Ok(match &posterior.algorithm {
            Some(algorithm) => rebuilt.with_algorithm(Arc::clone(algorithm)),
            None => rebuilt,
        })
    }

    /// The graph with a simulated common cause `u` of treatment and outcome declared.
    fn graph_with_common_cause(
        &self,
        u: VariableId,
    ) -> Result<crate::accepted::AcceptedGraph, TargetRefitError> {
        let graph = &self.study().graph;
        let declare = |error: antecedent_graph::GraphError| {
            TargetRefitError::Failed(ValidationError::estimation_msg(format!(
                "declaring the simulated common cause: {error}"
            )))
        };
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior is a mixture of structures; declaring one common cause \
                 in every atom changes the posterior",
                "refute one accepted structure of the posterior",
            ));
        }
        if let Some(dag) = graph.as_dag() {
            let mut dag = dag.clone();
            let node = dag.add_node(antecedent_core::NodeRef::Static(u)).map_err(declare)?;
            let (Some(t), Some(y)) =
                (static_node(&dag, self.treatment), static_node(&dag, self.outcome))
            else {
                return Err(unsupported(
                    "the treatment or outcome is not a node of the graph",
                    "a graph that names both",
                ));
            };
            dag.insert_directed(node, t).map_err(declare)?;
            dag.insert_directed(node, y).map_err(declare)?;
            return Ok(crate::accepted::AcceptedGraph::from(dag));
        }
        if let Some(tdag) = graph.as_temporal_dag() {
            let mut tdag = tdag.clone();
            let y0 =
                ensure_lagged(&mut tdag, self.outcome, Lag::CONTEMPORANEOUS).map_err(declare)?;
            let treatment_nodes = lagged_nodes(&tdag, self.treatment);
            if treatment_nodes.is_empty() {
                return Err(unsupported(
                    "the treatment is not a node of the temporal graph",
                    "a graph that names it",
                ));
            }
            for (t, lag) in treatment_nodes {
                let raw = u32::try_from(lag).unwrap_or(u32::MAX);
                let u_at = ensure_lagged(&mut tdag, u, Lag::from_raw(raw)).map_err(declare)?;
                tdag.insert_directed(u_at, t).map_err(declare)?;
                if u_at != y0 && !tdag.children(u_at).contains(&y0) {
                    tdag.insert_directed(u_at, y0).map_err(declare)?;
                }
            }
            return Ok(crate::accepted::AcceptedGraph::temporal_dag(tdag));
        }
        Err(unsupported(
            "a simulated common cause is declared on a DAG or a temporal DAG; this graph class \
             carries marks a declared parent does not resolve",
            "accept a DAG or temporal DAG of the structure",
        ))
    }

    /// Random common cause through the compiler: append a simulated cause, declare it as a
    /// parent of treatment and outcome, prepare the study on the augmented data and run it.
    fn random_common_cause(
        &self,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let strength = self.options.common_cause_strength;
        let draw = |n: usize, rng: &mut CausalRng| -> Arc<[f64]> {
            (0..n).map(|_| antecedent_kernels::standard_normal(rng)).collect()
        };
        let plant = |storage: &antecedent_data::OwnedColumnarStorage,
                     u: &[f64]|
         -> Result<ColumnReplacements, TargetRefitError> {
            if strength == 0.0 {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            let y = Self::column_of(storage, self.outcome)?;
            out.push((self.outcome, y.iter().zip(u).map(|(y, u)| y + strength * u).collect()));
            let t = Self::column_of(storage, self.treatment)?;
            #[allow(
                clippy::float_cmp,
                reason = "a binary treatment is coded exactly 0.0 or 1.0, so exact equality is the test"
            )]
            let binary = t.iter().all(|v| *v == 0.0 || *v == 1.0);
            if !binary {
                out.push((
                    self.treatment,
                    t.iter().zip(u).map(|(t, u)| t + strength * u).collect(),
                ));
            }
            Ok(out)
        };
        let name = |storage: &antecedent_data::OwnedColumnarStorage| {
            let names: Vec<&str> =
                storage.schema().variables().iter().map(|v| v.name.as_ref()).collect();
            // Among `names.len() + 1` candidates at least one is free.
            (0..=names.len())
                .map(|i| format!("simulated_common_cause_{i}"))
                .find(|candidate| !names.contains(&candidate.as_str()))
                .unwrap_or_default()
        };
        let append = |table: TabularData, u: Arc<[f64]>, label: &str| {
            table.with_appended_float(label, u).map_err(data_failed)
        };
        let (data, u_id) = match self.data {
            PreparedTargetData::Tabular(data) => {
                let u = draw(data.row_count(), rng);
                let planted = plant(data.storage(), &u)?;
                let base = if planted.is_empty() {
                    data.clone()
                } else {
                    data.with_replaced_floats(&planted).map_err(data_failed)?
                };
                let (table, id) = append(base, u, &name(data.storage()))?;
                (Modal::Tabular(table), id)
            }
            PreparedTargetData::Series(series) => {
                let u = draw(series.row_count(), rng);
                let planted = plant(series.storage(), &u)?;
                let base = series_with(series, &planted, None, None)?;
                let (table, id) =
                    append(TabularData::new(base.storage().clone()), u, &name(series.storage()))?;
                let out =
                    TimeSeriesData::try_new(table.storage().clone(), base.time_index().clone())
                        .map_err(data_failed)?;
                (Modal::Series(out), id)
            }
            PreparedTargetData::Panel(panel) => {
                let label = name(panel.units()[0].series.storage());
                let mut units = Vec::with_capacity(panel.unit_count());
                let mut id = None;
                for unit in panel.units() {
                    let u = draw(unit.series.row_count(), rng);
                    let planted = plant(unit.series.storage(), &u)?;
                    let base = series_with(&unit.series, &planted, None, None)?;
                    let (table, uid) = append(TabularData::new(base.storage().clone()), u, &label)?;
                    id = Some(uid);
                    units.push(PanelUnit {
                        unit_id: unit.unit_id,
                        series: TimeSeriesData::try_new(
                            table.storage().clone(),
                            base.time_index().clone(),
                        )
                        .map_err(data_failed)?,
                    });
                }
                let panel = PanelData::try_new(units).map_err(data_failed)?;
                (Modal::Panel(panel), id.ok_or_else(|| unsupported("an empty panel", "data"))?)
            }
            PreparedTargetData::MultiEnv(environments) => {
                let label = name(environments.environments()[0].storage());
                let mut members = Vec::with_capacity(environments.env_count());
                let mut id = None;
                for environment in environments.environments() {
                    let u = draw(environment.row_count(), rng);
                    let planted = plant(environment.storage(), &u)?;
                    let base = series_with(environment, &planted, None, None)?;
                    let (table, uid) = append(TabularData::new(base.storage().clone()), u, &label)?;
                    id = Some(uid);
                    members.push(
                        TimeSeriesData::try_new(table.storage().clone(), base.time_index().clone())
                            .map_err(data_failed)?,
                    );
                }
                let rebuilt =
                    antecedent_data::MultiEnvironmentData::try_new(members).map_err(data_failed)?;
                (Modal::MultiEnv(rebuilt), id.ok_or_else(|| unsupported("no environment", "data"))?)
            }
        };
        let mut study = self.study().clone();
        if let Some(posterior) = &self.study().graph_posterior {
            study.graph_posterior = Some(self.posterior_with_common_cause(posterior, u_id)?);
        } else {
            study.graph = Arc::new(self.graph_with_common_cause(u_id)?);
        }
        let refitted = Self::compile_and_run(study, &data, ctx).map_err(|error| {
            unsupported(
                format!("the study with the simulated common cause declared did not run: {error}"),
                "a route that admits an extra declared covariate",
            )
        })?;
        self.read_back(&refitted)
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
            data.with_replaced_float(id, values).map_err(data_failed)
        };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.treatment)?, rng, false);
                self.reestimate(&Modal::Tabular(replace(self.treatment, values)?), ctx)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.outcome)?, rng, false);
                self.reestimate(&Modal::Tabular(replace(self.outcome, values)?), ctx)
            }
            TargetPerturbation::RandomCommonCause { .. } => self.random_common_cause(rng, ctx),
            TargetPerturbation::Subset { fraction, .. } => {
                let n = data.row_count();
                if n < 2 {
                    return Err(unsupported("a subset needs at least two rows", "more data"));
                }
                let mut rows: Vec<usize> = (0..n).collect();
                shuffle(rng, &mut rows);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let kept = ((fraction * n as f64).round() as usize).clamp(1, n);
                let mut keep = vec![false; n];
                for &row in &rows[..kept] {
                    keep[row] = true;
                }
                let masked = data.with_analysis_mask(mask_from(&keep)?).map_err(data_failed)?;
                self.reestimate(&Modal::Tabular(masked), ctx)
            }
            TargetPerturbation::Bootstrap { .. } => {
                self.reestimate(&Modal::Tabular(Self::bootstrap(data, rng)?), ctx)
            }
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let control = self.control_exposure()?;
                let swapped =
                    replace(self.treatment, Arc::from(Self::column_of(storage, control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate(&Modal::Tabular(data), ctx)
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let control = self.control_outcome()?;
                let swapped = replace(self.outcome, Arc::from(Self::column_of(storage, control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate(&Modal::Tabular(data), ctx)
            }
            TargetPerturbation::PreTrend { .. } | TargetPerturbation::Anticipation { .. } => {
                Err(unsupported(
                    "independent rows have no time order, so a lead or a pre-period does not exist",
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
        data.with_replaced_floats(&replacements).map_err(data_failed)
    }

    /// One series perturbed for `perturbation`; `None` when the perturbation is not a column
    /// or weight change of the series (common cause, handled by the compiler).
    fn perturb_series(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        exposure: Option<VariableId>,
        outcome_control: Option<VariableId>,
    ) -> Result<TimeSeriesData, TargetRefitError> {
        let storage = series.storage();
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        // A design-respecting resample: replicate 0 is the series itself.
        let resample =
            |replicate: u32, rng: &mut CausalRng| -> Result<Option<Arc<[f64]>>, TargetRefitError> {
                if replicate == 0 { Ok(None) } else { block_weights(n, span, rng).map(Some) }
            };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.treatment)?, rng, true);
                series_with(series, &[(self.treatment, values)], None, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.outcome)?, rng, true);
                series_with(series, &[(self.outcome, values)], None, None)
            }
            TargetPerturbation::Subset { fraction, .. } => {
                if n < 2 {
                    return Err(unsupported("a window needs at least two periods", "more data"));
                }
                // A contiguous window keeps every lag the procedure reads inside the window.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n - width + 1);
                match truncated_series(series, start, start + width)? {
                    Some(window) => Ok(window),
                    // A non-float column cannot be cut here: the window is carried as zero
                    // weights outside it, which also leaves the series complete.
                    None => series_with(
                        series,
                        &[],
                        None,
                        Some(window_weights(n, start + span.min(width), start + width)),
                    ),
                }
            }
            TargetPerturbation::Bootstrap { .. } => {
                series_with(series, &[], None, Some(block_weights(n, span, rng)?))
            }
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let control = exposure.ok_or_else(|| unsupported("no exposure", "declare one"))?;
                let values: Arc<[f64]> = Arc::from(Self::column_of(storage, control)?);
                let weights = resample(replicate, rng)?;
                series_with(series, &[(self.treatment, values)], None, weights)
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let weights = resample(replicate, rng)?;
                if let Some(control) = outcome_control {
                    let values: Arc<[f64]> = Arc::from(Self::column_of(storage, control)?);
                    series_with(series, &[(self.outcome, values)], None, weights)
                } else {
                    // Lagged placebo: the outcome moved back past the longest lag, which no
                    // treatment the target reads can have caused.
                    let back = isize::try_from(span).unwrap_or(isize::MAX);
                    let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                    let rows = window_weights(n, 2 * span, n);
                    series_with(
                        series,
                        &[(self.outcome, values)],
                        None,
                        Some(times(&rows, weights)),
                    )
                }
            }
            TargetPerturbation::PreTrend { replicate, lead } => {
                // Placebo timing: the treatment `lead` periods later in the role of the
                // treatment. Its lags in the design are the actual treatment path, so the
                // lead's coefficient is the effect of a future treatment on the present
                // outcome given the past: zero without anticipation or a pre-trend.
                let lead_rows = lead as usize;
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.treatment)?, lead);
                let weights = resample(replicate, rng)?;
                let rows = window_weights(n, 0, n.saturating_sub(lead_rows));
                series_with(series, &[(self.treatment, values)], None, Some(times(&rows, weights)))
            }
            TargetPerturbation::Anticipation { replicate } => {
                // Pre-period event study: the outcome from before every treatment period the
                // target reads in the role of the outcome.
                let back = isize::try_from(span).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                let weights = resample(replicate, rng)?;
                let rows = window_weights(n, 2 * span, n);
                series_with(series, &[(self.outcome, values)], None, Some(times(&rows, weights)))
            }
            TargetPerturbation::RandomCommonCause { .. } => Err(unsupported(
                "the common cause is constructed by the compiler",
                "random_common_cause",
            )),
        }
    }

    /// The pre-trend, anticipation and lagged-placebo series as an explicit window: the shifted
    /// column cut to the design rows that have a value, with the lag history kept in front of
    /// them, and a gathered circular-block resample of that window for replicates after the
    /// first. This is what the weighted construction means for a route that reads no row
    /// weights; `None` for any other perturbation.
    fn materialised_window(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
    ) -> Result<Option<TimeSeriesData>, TargetRefitError> {
        let storage = series.storage();
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        let back = isize::try_from(span).unwrap_or(isize::MAX);
        let (replacement, from, to, replicate) = match perturbation {
            TargetPerturbation::PreTrend { replicate, lead } => {
                let lead_rows = lead as usize;
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.treatment)?, lead);
                ((self.treatment, values), 0, n.saturating_sub(lead_rows), replicate)
            }
            TargetPerturbation::Anticipation { replicate } => {
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                ((self.outcome, values), 2 * span, n, replicate)
            }
            TargetPerturbation::NegativeControlOutcome { replicate }
                if self.options.negative_control_outcome.is_none() =>
            {
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                ((self.outcome, values), 2 * span, n, replicate)
            }
            _ => return Ok(None),
        };
        let base = series_with(series, &[replacement], None, None)?;
        let Some(window) = truncated_series(&base, from.saturating_sub(span), to)? else {
            return Ok(None);
        };
        if replicate == 0 {
            return Ok(Some(window));
        }
        let plan = antecedent_data::ResamplingPlan::CircularBlock {
            length: antecedent_data::circular_block_length(span, window.row_count()),
        };
        let mut scratch = Vec::new();
        antecedent_data::resample_timeseries(&window, plan, rng, &mut scratch)
            .map(Some)
            .map_err(data_failed)
    }

    /// The bootstrap of a series drawn as a gathered circular-block series, with a zero row
    /// weight on every design row whose lag window crosses a block join (such a row pairs values
    /// observed apart and would attenuate a lagged effect). A route that reads row weights
    /// recovers the replicate the multiplicities define; a route that reads none keeps the
    /// attenuation, of order `(span - 1) / block length`.
    fn gathered_bootstrap(
        &self,
        series: &TimeSeriesData,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        let length = antecedent_data::circular_block_length(span, n);
        let mut map = vec![0_usize; n];
        antecedent_data::fill_circular_block_indexes(n, length, rng, &mut map)
            .map_err(data_failed)?;
        let storage = series.storage();
        let mut columns = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "the gathered bootstrap gathers float64 columns only",
                    "encode every column as float64",
                ));
            };
            let values: Arc<[f64]> = map.iter().map(|&row| column.values[row]).collect();
            // Missing values and the analysis mask travel with the rows they belong to.
            let validity = column.validity.gather_rows(&map).map_err(data_failed)?;
            columns.push(OwnedColumn::Float64(
                antecedent_data::Float64Column::new(column.id, values, validity)
                    .map_err(data_failed)?,
            ));
        }
        let mask = storage
            .analysis_mask()
            .map(|mask| mask.gather_rows(&map))
            .transpose()
            .map_err(data_failed)?;
        // A design row whose lag window crosses a block join pairs values observed apart, which
        // attenuates a lagged effect by about the share of such rows. Their weight is zero, so
        // a route that reads row weights recovers the replicate the multiplicities define; a
        // route that reads none keeps the attenuation, of order `(span - 1) / length`. A
        // sampling weight of the original sample is carried to the row it was gathered from, so
        // the replicate is the block resample of the weighted sample.
        let weights: Arc<[f64]> = (0..n)
            .map(|row| {
                let aligned = row + 1 >= span && (1..span).all(|k| map[row - k] + k == map[row]);
                let sampling = storage.weights().map_or(1.0, |weights| weights[map[row]]);
                if aligned { sampling } else { 0.0 }
            })
            .collect();
        let gathered = antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            mask,
            Some(weights),
        )
        .and_then(|gathered| TimeSeriesData::try_new(gathered, series.time_index().clone()))
        .map_err(data_failed)?;
        self.reestimate(&Modal::Series(gathered), ctx)
    }

    fn series_refit(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        if self.options.gathered_resamples
            && !matches!(perturbation, TargetPerturbation::RandomCommonCause { .. })
        {
            if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) {
                return self.gathered_bootstrap(series, rng, ctx);
            }
            if let Some(window) = self.materialised_window(series, perturbation, rng)? {
                return self.reestimate(&Modal::Series(window), ctx);
            }
        }
        if matches!(perturbation, TargetPerturbation::RandomCommonCause { .. }) {
            return self.random_common_cause(rng, ctx);
        }
        let exposure = match perturbation {
            TargetPerturbation::NegativeControlExposure { .. } => Some(self.control_exposure()?),
            _ => None,
        };
        let weighted = match perturbation {
            TargetPerturbation::Bootstrap { .. }
            | TargetPerturbation::PreTrend { .. }
            | TargetPerturbation::Anticipation { .. } => true,
            TargetPerturbation::NegativeControlOutcome { .. } => {
                self.options.negative_control_outcome.is_none()
                    || !matches!(
                        perturbation,
                        TargetPerturbation::NegativeControlOutcome { replicate: 0 }
                    )
            }
            TargetPerturbation::NegativeControlExposure { replicate } => replicate > 0,
            _ => false,
        };
        let perturbed = self.perturb_series(
            series,
            perturbation,
            rng,
            exposure,
            self.options.negative_control_outcome,
        )?;
        let refit = self.reestimate(&Modal::Series(perturbed), ctx);
        // A route that reads no row weights cannot take the block resample as multiplicities;
        // the bootstrap is then drawn as a gathered circular-block series, whose lag windows
        // that straddle a block junction pair values observed apart (the block length keeps
        // their share small).
        if refit.is_err() {
            if let Some(window) = self.materialised_window(series, perturbation, rng)? {
                if let Ok(values) = self.reestimate(&Modal::Series(window), ctx) {
                    return Ok(values);
                }
            }
        }
        if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) && refit.is_err() {
            if let Ok(values) = self.gathered_bootstrap(series, rng, ctx) {
                return Ok(values);
            }
        }
        match refit {
            // The circular-block resample is carried as row multiplicities; a route without a
            // weighted construction refuses them, and the refuter is recorded with its reason.
            Err(TargetRefitError::Failed(error)) if weighted => Err(unsupported(
                format!("the route has no weighted construction for the block resample: {error}"),
                "the frequentist single-window temporal linear adjustment, which reads row \
                 weights, or subset stability",
            )),
            other => other,
        }
    }

    /// Periods `[start, end)` of a unit series, complete; refused when a column is not float64.
    fn cut(
        series: &TimeSeriesData,
        start: usize,
        end: usize,
    ) -> Result<TimeSeriesData, TargetRefitError> {
        truncated_series(series, start, end)?.ok_or_else(|| {
            unsupported(
                "a shifted panel series is cut to its complete periods, which needs float64 \
                 columns",
                "encode every column as float64",
            )
        })
    }

    /// Independent environments: every environment is perturbed on its own by the series rules
    /// (a circular block resample carried as row multiplicities on top of the environment's own
    /// sampling weights, a circular shift of a placebo column, a window of the environment), and
    /// the study is re-run on the perturbed environments together.
    fn multi_env_refit(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        if matches!(perturbation, TargetPerturbation::RandomCommonCause { .. }) {
            return self.random_common_cause(rng, ctx);
        }
        let exposure = match perturbation {
            TargetPerturbation::NegativeControlExposure { .. } => Some(self.control_exposure()?),
            _ => None,
        };
        let mut perturbed = Vec::with_capacity(data.env_count());
        for environment in data.environments() {
            perturbed.push(self.perturb_series(
                environment,
                perturbation,
                rng,
                exposure,
                self.options.negative_control_outcome,
            )?);
        }
        let rebuilt =
            antecedent_data::MultiEnvironmentData::try_new(perturbed).map_err(data_failed)?;
        self.reestimate(&Modal::MultiEnv(rebuilt), ctx).map_err(|error| match error {
            TargetRefitError::Failed(error) => unsupported(
                format!("the route has no construction for the perturbed environments: {error}"),
                "a route that reads row weights and environment windows, or subset stability",
            ),
            other @ TargetRefitError::Unsupported { .. } => other,
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one exhaustive match over the perturbations of a panel"
    )]
    fn panel_refit(
        &self,
        panel: &PanelData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let units = panel.units();
        let rebuild = |units: Vec<PanelUnit>| PanelData::try_new(units).map_err(data_failed);
        let resample_units = |rng: &mut CausalRng, units: &[PanelUnit]| -> Vec<PanelUnit> {
            (0..units.len())
                .map(|slot| {
                    let source = &units[unbiased_index(rng, units.len())];
                    PanelUnit {
                        unit_id: u32::try_from(slot).unwrap_or(u32::MAX),
                        series: source.series.clone(),
                    }
                })
                .collect()
        };
        let swap_all =
            |id: VariableId, from: VariableId| -> Result<Vec<PanelUnit>, TargetRefitError> {
                units
                    .iter()
                    .map(|unit| {
                        let values: Arc<[f64]> =
                            Arc::from(Self::column_of(unit.series.storage(), from)?);
                        Ok(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        })
                    })
                    .collect()
            };
        let panel = match perturbation {
            TargetPerturbation::RandomCommonCause { .. } => {
                return self.random_common_cause(rng, ctx);
            }
            TargetPerturbation::PlaceboTreatment { .. }
            | TargetPerturbation::PlaceboOutcome { .. } => {
                let id = if matches!(perturbation, TargetPerturbation::PlaceboTreatment { .. }) {
                    self.treatment
                } else {
                    self.outcome
                };
                let equal =
                    units.iter().all(|u| u.series.row_count() == units[0].series.row_count());
                let mut out = Vec::with_capacity(units.len());
                if equal && units.len() > 1 {
                    // Whole paths permuted across units keep each path's own dependence.
                    let mut order: Vec<usize> = (0..units.len()).collect();
                    shuffle(rng, &mut order);
                    for (unit, &source) in units.iter().zip(&order) {
                        let values: Arc<[f64]> =
                            Arc::from(Self::column_of(units[source].series.storage(), id)?);
                        out.push(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        });
                    }
                } else {
                    for unit in units {
                        let values =
                            Self::permuted(Self::column_of(unit.series.storage(), id)?, rng, true);
                        out.push(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        });
                    }
                }
                rebuild(out)?
            }
            TargetPerturbation::Subset { fraction, .. } => {
                let mut order: Vec<usize> = (0..units.len()).collect();
                shuffle(rng, &mut order);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, units]"
                )]
                let kept = ((fraction * units.len() as f64).round() as usize).clamp(1, units.len());
                let mut chosen: Vec<usize> = order[..kept].to_vec();
                chosen.sort_unstable();
                rebuild(chosen.into_iter().map(|i| units[i].clone()).collect())?
            }
            TargetPerturbation::Bootstrap { .. } => rebuild(resample_units(rng, units))?,
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let swapped = swap_all(self.treatment, self.control_exposure()?)?;
                if replicate == 0 {
                    rebuild(swapped)?
                } else {
                    rebuild(resample_units(rng, &swapped))?
                }
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let swapped = if let Some(control) = self.options.negative_control_outcome {
                    swap_all(self.outcome, control)?
                } else {
                    let span =
                        isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                    units
                        .iter()
                        .map(|unit| {
                            let (values, _) = shifted(
                                Self::column_of(unit.series.storage(), self.outcome)?,
                                -span,
                            );
                            let rows = unit.series.row_count();
                            let skip = usize::try_from(span).unwrap_or(rows);
                            Ok(PanelUnit {
                                unit_id: unit.unit_id,
                                series: Self::cut(
                                    &series_with(
                                        &unit.series,
                                        &[(self.outcome, values)],
                                        None,
                                        None,
                                    )?,
                                    skip,
                                    rows,
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>, TargetRefitError>>()?
                };
                if replicate == 0 {
                    rebuild(swapped)?
                } else {
                    rebuild(resample_units(rng, &swapped))?
                }
            }
            TargetPerturbation::PreTrend { replicate, lead } => {
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let shifted_units = units
                    .iter()
                    .map(|unit| {
                        let (values, _) =
                            shifted(Self::column_of(unit.series.storage(), self.treatment)?, lead);
                        let rows = unit.series.row_count();
                        let keep = rows.saturating_sub(usize::try_from(lead).unwrap_or(rows));
                        Ok(PanelUnit {
                            unit_id: unit.unit_id,
                            series: Self::cut(
                                &series_with(
                                    &unit.series,
                                    &[(self.treatment, values)],
                                    None,
                                    None,
                                )?,
                                0,
                                keep,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, TargetRefitError>>()?;
                if replicate == 0 {
                    rebuild(shifted_units)?
                } else {
                    rebuild(resample_units(rng, &shifted_units))?
                }
            }
            TargetPerturbation::Anticipation { replicate } => {
                let back = isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                let shifted_units = units
                    .iter()
                    .map(|unit| {
                        let (values, _) =
                            shifted(Self::column_of(unit.series.storage(), self.outcome)?, -back);
                        let rows = unit.series.row_count();
                        let skip = usize::try_from(back).unwrap_or(rows);
                        Ok(PanelUnit {
                            unit_id: unit.unit_id,
                            series: Self::cut(
                                &series_with(&unit.series, &[(self.outcome, values)], None, None)?,
                                skip,
                                rows,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, TargetRefitError>>()?;
                if replicate == 0 {
                    rebuild(shifted_units)?
                } else {
                    rebuild(resample_units(rng, &shifted_units))?
                }
            }
        };
        self.reestimate(&Modal::Panel(panel), ctx)
    }

    fn modal_data(&self) -> Modal {
        match self.data {
            PreparedTargetData::Tabular(d) => Modal::Tabular(d.clone()),
            PreparedTargetData::Series(d) => Modal::Series(d.clone()),
            PreparedTargetData::Panel(d) => Modal::Panel(d.clone()),
            PreparedTargetData::MultiEnv(d) => Modal::MultiEnv((*d).clone()),
        }
    }

    /// Posterior draws of the whole target (`draws × coordinates`, row-major).
    fn target_draws(&self, result: &StudyResult) -> Option<Vec<f64>> {
        let k = self.target.dimension();
        if let Some(response) = result.response() {
            let draws = match &response.uncertainty {
                ResponseUncertainty::Scalar { draws: Some(draws), .. }
                | ResponseUncertainty::PointwiseBand { draws: Some(draws), .. } => Some(draws),
                _ => None,
            };
            if let Some(draws) = draws.filter(|d| d.n_coordinates() == k && d.n_draws > 0) {
                let mut rows = Vec::with_capacity(draws.n_draws * k);
                for s in 0..draws.n_draws {
                    for j in 0..k {
                        rows.push(draws.column(j)?[s]);
                    }
                }
                return Some(rows);
            }
        }
        if k == 1 {
            let posterior = result.posterior.as_ref()?;
            let n = posterior.draws.n_draws;
            return posterior.draws.values.get(..n).map(<[f64]>::to_vec);
        }
        None
    }

    /// Prior sensitivity through the compiler: the study prepared under each alternative
    /// isotropic prior scale, the posterior of the whole target read back from each.
    fn prior_sensitivity(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let InferenceMode::Bayesian(config) = &self.study().inference else {
            return Err(unsupported(
                "prior sensitivity reads a posterior; the study is frequentist",
                "prepare the study with Bayesian inference",
            ));
        };
        let baseline = self.target_draws(self.result).ok_or_else(|| {
            unsupported(
                "the result retains no posterior draws of its whole target",
                "a Bayesian route that retains target draws",
            )
        })?;
        let mut alternatives: Vec<(Arc<str>, Vec<f64>)> = Vec::new();
        for &factor in &self.options.prior_scale_factors {
            if !(factor.is_finite() && factor > 0.0) {
                continue;
            }
            let mut study = self.study().clone();
            let mut alternative = config.clone();
            // An explicit coefficient prior has no single scale: its Gaussian coefficient
            // standard deviations are multiplied by `factor` around their own means (variances
            // by `factor^2`), the same perturbation the isotropic scale undergoes.
            // A prior hydrated from a posterior artifact, or composed from external sources,
            // exists only once the design is bound, so its variances are scaled where the route
            // resolves it (`prior_variance_multiplier`), not here.
            match &config.prior {
                Some(prior) => alternative.prior = Some(scale_prior_deviations(prior, factor)?),
                None if config.prior_artifact.is_some() || config.external_compose.is_some() => {
                    alternative.prior_variance_multiplier = Some(factor * factor);
                }
                None => alternative.prior_scale *= factor,
            }
            study.inference = InferenceMode::Bayesian(alternative);
            let refitted =
                Self::compile_and_run(study, &self.modal_data(), ctx).map_err(|error| {
                    unsupported(
                        format!("the study under prior scale x{factor} did not run: {error}"),
                        "a Bayesian route whose prior scale can be varied",
                    )
                })?;
            let draws = self.target_draws(&refitted).ok_or_else(|| {
                unsupported(
                    "the refit under the alternative prior retains no target draws",
                    "a Bayesian route that retains target draws",
                )
            })?;
            alternatives.push((Arc::from(format!("prior_scale x{factor}")), draws));
        }
        if alternatives.is_empty() {
            return Err(unsupported("no alternative prior scale was declared", "declare one"));
        }
        prior_sensitivity_target_check(&PriorSensitivityTarget {
            target: &self.target,
            baseline: &baseline,
            alternatives: &alternatives,
            tolerance: self.options.prior_shift_tolerance,
        })
        .map_err(TargetRefitError::Failed)
    }

    /// Posterior-predictive check of the target through the compiler, under the reference
    /// linear-Gaussian outcome model of the treatment and adjustment set: for each draw
    /// `(β_s, σ_s)` of its reference posterior the program is run on the noiseless outcome
    /// `μ_s = Xβ_s` (the target of the draw) and on a replicated outcome `μ_s + σ_s ε`; the
    /// observed target is compared with the replicated ones.
    #[allow(
        clippy::too_many_lines,
        reason = "the fit, its Cholesky factor and the draws share one reference design in sequence"
    )]
    fn posterior_predictive(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let PreparedTargetData::Tabular(data) = self.data else {
            return self.posterior_predictive_lagged(ctx);
        };
        let storage = data.storage();
        if storage.analysis_mask().is_some() || storage.weights().is_some() {
            return Err(unsupported(
                "the reference outcome model replicates an unmasked, unweighted table",
                "refute the unmasked table",
            ));
        }
        let mut ids = vec![self.treatment];
        ids.extend(self.adjustment());
        let mut columns = Vec::with_capacity(ids.len());
        for id in &ids {
            columns.push(Self::column_of(storage, *id)?);
        }
        let y = Self::column_of(storage, self.outcome)?;
        let n = y.len();
        let p = ids.len() + 1;
        if n <= p + 2 || y.iter().chain(columns.iter().copied().flatten()).any(|v| !v.is_finite()) {
            return Err(unsupported(
                "the reference outcome model needs complete float columns and more rows than \
                 parameters",
                "resolve missing values first",
            ));
        }
        let mut design = vec![1.0; n];
        for column in &columns {
            design.extend_from_slice(column);
        }
        let fit = FaerBackend
            .least_squares(&design, n, p, y, &mut LeastSquaresWorkspace::default())
            .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        // Cholesky factor of X'X.
        let mut xtx = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..=i {
                let s: f64 = (0..n).map(|r| design[i * n + r] * design[j * n + r]).sum();
                xtx[i * p + j] = s;
                xtx[j * p + i] = s;
            }
        }
        let mut l = xtx;
        for i in 0..p {
            for j in 0..=i {
                let mut s = l[i * p + j];
                for q in 0..j {
                    s -= l[i * p + q] * l[j * p + q];
                }
                if i == j {
                    if s.partial_cmp(&0.0) != Some(Ordering::Greater) {
                        return Err(unsupported(
                            "the treatment and adjustment design is rank deficient",
                            "a full-rank adjustment",
                        ));
                    }
                    l[i * p + i] = s.sqrt();
                } else {
                    l[i * p + j] = s / l[j * p + j];
                }
            }
        }
        let k = self.target.dimension();
        let draws_wanted = (self.replicates as usize).max(k + 2);
        let _lease = admit_target_cells(ctx, draws_wanted.saturating_mul(2), n)?;
        let mut thetas = Vec::with_capacity(draws_wanted * k);
        let mut replicated = Vec::with_capacity(draws_wanted * k);
        let dof = (n - p) as f64;
        for s in 0..draws_wanted {
            if ctx.cancellation.is_cancelled() {
                return Err(TargetRefitError::Failed(ValidationError::Cancelled));
            }
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PosteriorPredictive,
                u32::try_from(s).unwrap_or(u32::MAX),
                ctx,
            );
            let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
            let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
            // β = β̂ + σ L⁻ᵀ z.
            let z: Vec<f64> =
                (0..p).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
            let mut w = z;
            for i in (0..p).rev() {
                let mut acc = w[i];
                for q in (i + 1)..p {
                    acc -= l[q * p + i] * w[q];
                }
                w[i] = acc / l[i * p + i];
            }
            let beta: Vec<f64> =
                fit.coefficients.iter().zip(&w).map(|(b, w)| b + sigma * w).collect();
            let mean: Vec<f64> =
                (0..n).map(|r| (0..p).map(|c| design[c * n + r] * beta[c]).sum()).collect();
            let noisy: Arc<[f64]> = mean
                .iter()
                .map(|m| m + sigma * antecedent_kernels::standard_normal(&mut rng))
                .collect();
            let run = |outcome: Arc<[f64]>| -> Result<Vec<f64>, TargetRefitError> {
                let table = data.with_replaced_float(self.outcome, outcome).map_err(data_failed)?;
                self.reestimate(&Modal::Tabular(table), ctx).map_err(|error| match error {
                    TargetRefitError::Failed(error) => unsupported(
                        format!("the program did not run on a reference-model replicate: {error}"),
                        "a route that admits replicated outcomes",
                    ),
                    other @ TargetRefitError::Unsupported { .. } => other,
                })
            };
            thetas.extend(run(mean.into())?);
            replicated.extend(run(noisy)?);
        }
        posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &self.target,
            observed: &self.original,
            draws: &thetas,
            replicated: &replicated,
            alpha: self.alpha,
        })
        .map_err(TargetRefitError::Failed)
    }

    /// Posterior-predictive check on a series or the units of a panel, under a reference outcome
    /// model that carries the lag structure the route's graph declares: the outcome regressed on
    /// the treatment, the adjustment set and its own `L` previous values (`L` the longest lag of
    /// the graph, at least one), pooled over units. Each draw of its reference posterior replaces
    /// every unit's outcome by the model's recursion with no noise (the target of the draw) and
    /// with simulated noise (a replicate), the first `L` periods of a unit staying as observed,
    /// and the compiled program is re-run on both.
    #[allow(
        clippy::too_many_lines,
        clippy::needless_range_loop,
        clippy::cast_precision_loss,
        reason = "the pooled lagged design, its Cholesky factor and the recursive draws share one reference model in sequence"
    )]
    fn posterior_predictive_lagged(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let units: Vec<&TimeSeriesData> = match self.data {
            PreparedTargetData::Series(series) => vec![series],
            PreparedTargetData::Panel(panel) => panel.units().iter().map(|u| &u.series).collect(),
            PreparedTargetData::MultiEnv(environments) => {
                environments.environments().iter().collect()
            }
            PreparedTargetData::Tabular(_) => {
                return Err(unsupported("a table has no lag structure", "a series or panel"));
            }
        };
        let lags = max_graph_lag(self.study()).max(1);
        let mut ids = vec![self.treatment];
        ids.extend(self.adjustment());
        let p = ids.len() + 1 + lags;
        let mut outcomes: Vec<&[f64]> = Vec::with_capacity(units.len());
        let mut covariates: Vec<Vec<&[f64]>> = Vec::with_capacity(units.len());
        let mut rows = 0_usize;
        for unit in &units {
            let storage = unit.storage();
            if storage.analysis_mask().is_some() || storage.weights().is_some() {
                return Err(unsupported(
                    "the reference outcome model replicates unmasked, unweighted series",
                    "refute the unmasked series",
                ));
            }
            let y = Self::column_of(storage, self.outcome)?;
            let mut columns = Vec::with_capacity(ids.len());
            for id in &ids {
                columns.push(Self::column_of(storage, *id)?);
            }
            if y.len() <= lags
                || y.iter().chain(columns.iter().copied().flatten()).any(|v| !v.is_finite())
            {
                return Err(unsupported(
                    "the reference outcome model needs complete float columns longer than the lag",
                    "resolve missing values first",
                ));
            }
            rows += y.len() - lags;
            outcomes.push(y);
            covariates.push(columns);
        }
        if rows <= p + 2 {
            return Err(unsupported(
                "the reference outcome model needs more lag-aligned rows than parameters",
                "a longer series",
            ));
        }
        // Column-major design over the lag-aligned rows of every unit.
        let mut design = vec![0.0; rows * p];
        let mut response = Vec::with_capacity(rows);
        let mut row = 0;
        for (u, y) in outcomes.iter().enumerate() {
            for t in lags..y.len() {
                design[row] = 1.0;
                for (c, column) in covariates[u].iter().enumerate() {
                    design[(1 + c) * rows + row] = column[t];
                }
                for j in 1..=lags {
                    design[(ids.len() + j) * rows + row] = y[t - j];
                }
                response.push(y[t]);
                row += 1;
            }
        }
        let fit = FaerBackend
            .least_squares(&design, rows, p, &response, &mut LeastSquaresWorkspace::default())
            .map_err(|error| TargetRefitError::Failed(ValidationError::from(error)))?;
        let mut l = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..=i {
                let mut acc: f64 =
                    (0..rows).map(|r| design[i * rows + r] * design[j * rows + r]).sum();
                for q in 0..j {
                    acc -= l[i * p + q] * l[j * p + q];
                }
                if i == j {
                    if acc.partial_cmp(&0.0) != Some(Ordering::Greater) {
                        return Err(unsupported(
                            "the lagged reference design is rank deficient",
                            "a full-rank adjustment",
                        ));
                    }
                    l[i * p + i] = acc.sqrt();
                } else {
                    l[i * p + j] = acc / l[j * p + j];
                }
            }
        }
        let k = self.target.dimension();
        let draws_wanted = (self.replicates as usize).max(k + 2);
        let _lease = admit_target_cells(ctx, draws_wanted.saturating_mul(2), rows)?;
        let mut thetas = Vec::with_capacity(draws_wanted * k);
        let mut replicated = Vec::with_capacity(draws_wanted * k);
        let dof = (rows - p) as f64;
        for s in 0..draws_wanted {
            if ctx.cancellation.is_cancelled() {
                return Err(TargetRefitError::Failed(ValidationError::Cancelled));
            }
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PosteriorPredictive,
                u32::try_from(s).unwrap_or(u32::MAX),
                ctx,
            );
            let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
            let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
            let mut w: Vec<f64> =
                (0..p).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
            for i in (0..p).rev() {
                let mut acc = w[i];
                for q in (i + 1)..p {
                    acc -= l[q * p + i] * w[q];
                }
                w[i] = acc / l[i * p + i];
            }
            let beta: Vec<f64> =
                fit.coefficients.iter().zip(&w).map(|(b, w)| b + sigma * w).collect();
            let simulate = |noisy: bool, rng: &mut CausalRng| -> Vec<Arc<[f64]>> {
                outcomes
                    .iter()
                    .enumerate()
                    .map(|(u, y)| {
                        let mut path = y.to_vec();
                        for t in lags..y.len() {
                            let mut mean = beta[0];
                            for (c, column) in covariates[u].iter().enumerate() {
                                mean += beta[1 + c] * column[t];
                            }
                            for j in 1..=lags {
                                mean += beta[ids.len() + j] * path[t - j];
                            }
                            path[t] = if noisy {
                                mean + sigma * antecedent_kernels::standard_normal(rng)
                            } else {
                                mean
                            };
                        }
                        Arc::from(path)
                    })
                    .collect()
            };
            let mean_paths = simulate(false, &mut rng);
            let noisy_paths = simulate(true, &mut rng);
            let run = |paths: Vec<Arc<[f64]>>| -> Result<Vec<f64>, TargetRefitError> {
                let mut replaced = Vec::with_capacity(units.len());
                for (unit, path) in units.iter().zip(paths) {
                    replaced.push(series_with(unit, &[(self.outcome, path)], None, None)?);
                }
                let modal = match self.data {
                    PreparedTargetData::Panel(panel) => {
                        let mut members = Vec::with_capacity(replaced.len());
                        for (unit, series) in panel.units().iter().zip(replaced) {
                            members.push(PanelUnit { unit_id: unit.unit_id, series });
                        }
                        Modal::Panel(PanelData::try_new(members).map_err(data_failed)?)
                    }
                    PreparedTargetData::MultiEnv(_) => Modal::MultiEnv(
                        antecedent_data::MultiEnvironmentData::try_new(replaced)
                            .map_err(data_failed)?,
                    ),
                    _ => Modal::Series(replaced.remove(0)),
                };
                self.reestimate(&modal, ctx).map_err(|error| match error {
                    TargetRefitError::Failed(error) => unsupported(
                        format!("the program did not run on a reference-model replicate: {error}"),
                        "a route that admits replicated outcomes",
                    ),
                    other @ TargetRefitError::Unsupported { .. } => other,
                })
            };
            thetas.extend(run(mean_paths)?);
            replicated.extend(run(noisy_paths)?);
        }
        posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &self.target,
            observed: &self.original,
            draws: &thetas,
            replicated: &replicated,
            alpha: self.alpha,
        })
        .map_err(TargetRefitError::Failed)
    }

    /// The fitted response curve of a derivative at the points the derivative diagnostics read,
    /// through the compiler: the study's mean-curve query, under its own temporal
    /// specification, on the same data.
    ///
    /// A point derivative is read at its evaluation point. An average derivative is read at
    /// seven nodes of its weighting law (quantiles of the observed treatment, or midpoints of
    /// the uniform interval), so its finite difference is the mean of the nodes' point
    /// differences. Each node contributes the five grid points `x`, `x +- step / 2` and
    /// `x +- step`. A multi-horizon specification reads one curve column per horizon.
    ///
    /// Returns the nodes, the reported value of each horizon and the step.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "one check that classifies the derivative, picks its nodes and step and fits its curve once"
    )]
    fn response_curve(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), TargetRefitError> {
        const NODES: usize = 7;
        let Some(response) = self.result.response() else {
            return Err(unsupported(
                "the derivative diagnostics read a response result",
                "a derivative response query",
            ));
        };
        let treatment_column = || -> Vec<f64> {
            let mut column: Vec<f64> = match self.data {
                PreparedTargetData::Panel(panel) => panel
                    .units()
                    .iter()
                    .filter_map(|unit| float_column(unit.series.storage(), self.treatment))
                    .flatten()
                    .copied()
                    .collect(),
                PreparedTargetData::MultiEnv(environments) => environments
                    .environments()
                    .iter()
                    .filter_map(|series| float_column(series.storage(), self.treatment))
                    .flatten()
                    .copied()
                    .collect(),
                _ => float_column(self.storage(), self.treatment)
                    .map(<[f64]>::to_vec)
                    .unwrap_or_default(),
            };
            column.retain(|value| value.is_finite());
            column
        };
        let (outcome, treatment, nodes) = match &response.estimand {
            ResponseFunctional::PointDerivative {
                outcome,
                treatment,
                at,
                order: 1,
                scale: DerivativeScale::Identity,
            } => (*outcome, *treatment, vec![*at]),
            ResponseFunctional::PointDerivative { .. } => {
                return Err(unsupported(
                    "the consistency check compares a first derivative on the identity scale",
                    "query the first derivative on the identity scale",
                ));
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                let nodes: Vec<f64> = match weighting {
                    DerivativeWeighting::Observed => {
                        let mut column = treatment_column();
                        column.sort_by(f64::total_cmp);
                        if column.len() < NODES {
                            return Err(unsupported(
                                "the observed treatment has too few values to place the nodes \
                                 of the weighting law",
                                "more data",
                            ));
                        }
                        (0..NODES)
                            .map(|k| {
                                let rank = ((2 * k + 1) * column.len()) / (2 * NODES);
                                column[rank.min(column.len() - 1)]
                            })
                            .collect()
                    }
                    DerivativeWeighting::Uniform { lower, upper } => (0..NODES)
                        .map(|k| lower + (upper - lower) * (k as f64 + 0.5) / NODES as f64)
                        .collect(),
                    DerivativeWeighting::Custom(_) => {
                        return Err(unsupported(
                            "caller-supplied row weights have no support points to place nodes at",
                            "check point derivatives at chosen points",
                        ));
                    }
                };
                (*outcome, *treatment, nodes)
            }
            _ => {
                return Err(unsupported(
                    "a directional derivative or Jacobian has several treatments; its finite \
                     difference needs the joint response surface",
                    "check each treatment's point derivative",
                ));
            }
        };
        let horizons = match &self.study().query {
            CausalQuery::Response(original) => {
                original.temporal.as_ref().map_or(1, |spec| spec.horizons.len())
            }
            _ => 1,
        };
        if self.original.len() != horizons {
            return Err(unsupported(
                "the reported derivative does not have one value per horizon of the specification",
                "a derivative query with one value per horizon",
            ));
        }
        let reported = self.original.clone();
        let step = if let Some(step) = self.options.derivative_step {
            step
        } else {
            0.25 * sample_sd(&treatment_column())
        };
        if !(step.is_finite() && step > 0.0) {
            return Err(unsupported(
                "the treatment has no spread to take a derivative step from",
                "declare derivative_step",
            ));
        }
        if self.curve.get().is_none() {
            let half = 0.5 * step;
            // The grid of a mean curve is strictly increasing, and the neighbourhoods of the
            // nodes overlap.
            let mut points: Vec<f64> = nodes
                .iter()
                .flat_map(|&node| [-step, -half, 0.0, half, step].map(|offset| node + offset))
                .collect();
            points.sort_by(f64::total_cmp);
            points.dedup_by(|a, b| a.to_bits() == b.to_bits());
            let mut curve_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from(points.clone())),
                ),
            });
            // A temporal derivative's curve is fit under the same temporal specification.
            if let CausalQuery::Response(original) = &self.study().query {
                curve_query.temporal.clone_from(&original.temporal);
            }
            let query = CausalQuery::Response(curve_query);
            let mut study = self.study().clone();
            study.query = query;
            let refitted =
                if let Ok(result) = Self::compile_and_run(study.clone(), &self.modal_data(), ctx) {
                    result
                } else {
                    // The derivative's own estimator may not serve a mean curve; let the
                    // compiler choose the curve estimator of the same data and graph.
                    study.estimator = None;
                    study.estimator_spec = None;
                    Self::compile_and_run(study, &self.modal_data(), ctx).map_err(|error| {
                        unsupported(
                            format!("the study's mean curve did not run: {error}"),
                            "a graph and data that identify the dose-response curve",
                        )
                    })?
                };
            let curve = refitted
                .response()
                .and_then(|r| match &r.estimate {
                    ResponseIdentification::PointIdentified(ResponseValue::Surface {
                        mean,
                        ..
                    }) if mean.len() == points.len() * horizons => Some(
                        points
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(dose, point)| {
                                (point, mean[dose * horizons..(dose + 1) * horizons].to_vec())
                            })
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .ok_or_else(|| {
                    unsupported(
                        "the mean curve is not point identified on the check grid",
                        "a point-identified dose-response curve",
                    )
                })?;
            let _ = self.curve.set(curve);
        }
        Ok((nodes, reported, step))
    }

    fn curve_value(&self, x: f64, horizon: usize) -> Result<f64, ValidationError> {
        self.curve
            .get()
            .and_then(|curve| curve.iter().find(|(p, _)| p.to_bits() == x.to_bits()))
            .and_then(|(_, values)| values.get(horizon).copied())
            .ok_or_else(|| ValidationError::estimation_msg("response curve read off its grid"))
    }

    /// The mean over the nodes of the curve shifted by `offset`: its derivative at zero is the
    /// mean of the nodes' derivatives, which for an average derivative is the finite
    /// difference of the weighted average.
    fn node_mean(
        &self,
        nodes: &[f64],
        offset: f64,
        horizon: usize,
    ) -> Result<f64, ValidationError> {
        let mut total = 0.0;
        for &node in nodes {
            total += self.curve_value(node + offset, horizon)?;
        }
        Ok(total / nodes.len() as f64)
    }

    fn derivative_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        if let Some(response) = self.result.response() {
            if matches!(
                response.estimand,
                ResponseFunctional::DirectionalDerivative { .. }
                    | ResponseFunctional::Jacobian { .. }
            ) {
                return self.vector_derivative_check(kind, ctx);
            }
        }
        let (nodes, reported, step) = self.response_curve(ctx)?;
        let mut coordinates = Vec::with_capacity(reported.len());
        for (horizon, &value) in reported.iter().enumerate() {
            coordinates.push(self.check_coordinate(
                kind,
                |offset| self.node_mean(&nodes, offset, horizon),
                value,
                step,
            )?);
        }
        self.merge_coordinates(kind, coordinates)
    }

    /// One coordinate's check along a response function of the offset from the evaluation
    /// point.
    fn check_coordinate(
        &self,
        kind: TargetRefuterKind,
        response: impl FnMut(f64) -> Result<f64, ValidationError>,
        reported: f64,
        step: f64,
    ) -> Result<DerivativeCoordinate, TargetRefitError> {
        Ok(if kind == TargetRefuterKind::FiniteDifferenceConsistency {
            DerivativeCoordinate::FiniteDifference(finite_difference_consistency(
                response,
                &[DerivativeCheckPoint { x: 0.0, reported }],
                step,
                self.options.derivative_tolerance,
            )?)
        } else {
            DerivativeCoordinate::Linearity(local_linearity(
                response,
                &[0.0],
                step,
                self.options.linearity_tolerance,
                1e-9,
            )?)
        })
    }

    /// One report on the whole target vector from the per-coordinate checks.
    fn merge_coordinates(
        &self,
        kind: TargetRefuterKind,
        coordinates: Vec<DerivativeCoordinate>,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        merge_derivative_checks(kind, coordinates, &self.target)
    }

    /// The first-order check of a directional derivative or a Jacobian.
    ///
    /// Every coordinate (an outcome, and for a Jacobian a treatment) is the derivative of the
    /// outcome's interventional mean along a direction at the evaluation point, so its
    /// finite difference is read off the study's own `InterventionResponse` query at the
    /// shifted points `at + t * direction`: a Jacobian entry along the treatment's axis, a
    /// directional derivative along the declared direction.
    #[allow(
        clippy::too_many_lines,
        reason = "one check that classifies the derivative, places its points and fits each mean"
    )]
    fn vector_derivative_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        const MAX_COORDINATES: usize = 6;
        let Some(response) = self.result.response() else {
            return Err(unsupported(
                "the derivative diagnostics read a response result",
                "a derivative response query",
            ));
        };
        let (outcomes, treatments, at, directions): (
            Vec<VariableId>,
            Vec<VariableId>,
            Vec<f64>,
            Vec<Vec<f64>>,
        ) = match &response.estimand {
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                (outcomes.to_vec(), treatments.to_vec(), at.to_vec(), vec![direction.to_vec()])
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if *scale != DerivativeScale::Identity {
                    return Err(unsupported(
                        "the consistency check compares derivatives on the identity scale",
                        "query the Jacobian on the identity scale",
                    ));
                }
                let axes = (0..treatments.len())
                    .map(|k| (0..treatments.len()).map(|j| f64::from(u8::from(j == k))).collect())
                    .collect();
                (outcomes.to_vec(), treatments.to_vec(), at.to_vec(), axes)
            }
            _ => {
                return Err(unsupported("not a vector derivative", "a derivative query"));
            }
        };
        let coordinates: Vec<(usize, usize)> =
            (0..outcomes.len()).flat_map(|o| (0..directions.len()).map(move |d| (o, d))).collect();
        if coordinates.len() != self.original.len() || coordinates.len() > MAX_COORDINATES {
            return Err(unsupported(
                "the reported derivative vector does not match its outcomes and directions, or \
                 has more coordinates than the check refits",
                "check a smaller set of outcomes and treatments",
            ));
        }
        // The step along a direction: a share of the smallest treatment spread it crosses.
        let spread = |variable: VariableId| -> f64 {
            let column: Vec<f64> = match self.data {
                PreparedTargetData::Panel(panel) => panel
                    .units()
                    .iter()
                    .filter_map(|unit| float_column(unit.series.storage(), variable))
                    .flatten()
                    .copied()
                    .collect(),
                PreparedTargetData::MultiEnv(environments) => environments
                    .environments()
                    .iter()
                    .filter_map(|series| float_column(series.storage(), variable))
                    .flatten()
                    .copied()
                    .collect(),
                _ => {
                    float_column(self.storage(), variable).map(<[f64]>::to_vec).unwrap_or_default()
                }
            };
            sample_sd(&column)
        };
        let step = if let Some(step) = self.options.derivative_step {
            step
        } else {
            directions
                .iter()
                .flat_map(|direction| {
                    treatments
                        .iter()
                        .zip(direction)
                        .filter(|(_, d)| **d != 0.0)
                        .map(|(&variable, d)| 0.25 * spread(variable) / d.abs())
                        .collect::<Vec<_>>()
                })
                .fold(f64::INFINITY, f64::min)
        };
        if !(step.is_finite() && step > 0.0) {
            return Err(unsupported(
                "the treatments have no spread to take a derivative step from",
                "declare derivative_step",
            ));
        }
        let cache: std::cell::RefCell<std::collections::HashMap<(usize, Vec<u64>), f64>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
        let mean_at = |outcome_index: usize, point: &[f64]| -> Result<f64, ValidationError> {
            let key = (outcome_index, point.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            if let Some(found) = cache.borrow().get(&key) {
                return Ok(*found);
            }
            let interventions: Vec<Intervention> = treatments
                .iter()
                .zip(point)
                .map(|(&variable, &level)| Intervention::set(variable, Value::f64(level)))
                .collect();
            let mut study = self.study().clone();
            study.query = CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::InterventionResponse {
                    outcome: outcomes[outcome_index],
                    interventions: Arc::from(interventions),
                },
            ));
            let run = |study: crate::analysis::execute::Study| {
                Self::compile_and_run(study, &self.modal_data(), ctx)
            };
            let result = if let Ok(result) = run(study.clone()) {
                result
            } else {
                study.estimator = None;
                study.estimator_spec = None;
                run(study).map_err(|error| {
                    ValidationError::estimation_msg(format!(
                        "the interventional mean did not run: {error}"
                    ))
                })?
            };
            let value = result
                .response()
                .and_then(|r| match &r.estimate {
                    ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => Some(*v),
                    _ => None,
                })
                .ok_or_else(|| {
                    ValidationError::estimation_msg(
                        "the interventional mean is not point identified at a check point",
                    )
                })?;
            cache.borrow_mut().insert(key, value);
            Ok(value)
        };
        let mut checks = Vec::with_capacity(coordinates.len());
        for (index, &(outcome_index, direction_index)) in coordinates.iter().enumerate() {
            let direction = &directions[direction_index];
            let reported = self.original[index];
            checks.push(self.check_coordinate(
                kind,
                |offset| {
                    let point: Vec<f64> =
                        at.iter().zip(direction).map(|(a, d)| a + offset * d).collect();
                    mean_at(outcome_index, &point)
                },
                reported,
                step,
            )?);
        }
        self.merge_coordinates(kind, checks)
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
        let mut ids = vec![self.treatment, mediator, self.outcome];
        ids.extend(self.adjustment());
        let columns: Vec<&[f64]> =
            ids.iter().map(|&id| Self::column_of(storage, id)).collect::<Result<_, _>>()?;
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

/// Lease `count × width` cells for a native check's buffers.
fn admit_target_cells(
    ctx: &ExecutionContext,
    count: usize,
    width: usize,
) -> Result<antecedent_core::execution::AdmissionLease, TargetRefitError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            TargetRefitError::Failed(ValidationError::estimation_msg("check buffer size overflow"))
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            TargetRefitError::Failed(ValidationError::Cancelled)
        }
        other => TargetRefitError::Failed(ValidationError::estimation_msg(format!(
            "admission pool cannot fund the check buffers: {other}"
        ))),
    })
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
            PreparedTargetData::Panel(data) => self.panel_refit(data, perturbation, rng, ctx),
            PreparedTargetData::MultiEnv(data) => {
                self.multi_env_refit(data, perturbation, rng, ctx)
            }
        }
    }

    fn native_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        match kind {
            TargetRefuterKind::SequentialIgnorability => self.mediation_check(),
            TargetRefuterKind::FiniteDifferenceConsistency | TargetRefuterKind::LocalLinearity => {
                self.derivative_check(kind, ctx)
            }
            TargetRefuterKind::PriorSensitivity => self.prior_sensitivity(ctx),
            TargetRefuterKind::PosteriorPredictive => self.posterior_predictive(ctx),
            _ => Err(unsupported("no native check for this refuter", "a replicate refuter")),
        }
    }
}

/// Sample standard deviation of the finite values (zero for fewer than two).
fn sample_sd(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}
