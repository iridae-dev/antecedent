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
//!   (`refuse_scalar_refutations_for_result`, enforced by [`StudyResult::claim`] and
//!   [`certify_result_refutations`]). Route
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
    if !matches!(&result.authority, crate::result::RuntimeRefutationAuthority::Fresh) {
        return;
    }
    if let Some(program) = result.lowered_identity {
        let program = *program.as_bytes();
        for report in &mut result.refutations {
            report.stamp_producer(program);
        }
        // Repeated binding is common, but must never bless edits to a previously
        // issued result. Only a checked follow-up operation can issue anew.
        let _ = seal_runtime_refutations(result, crate::result::RuntimeRefutationOrigin::Route);
    }
}

impl LoweredProgram {
    /// Quadrant of this program: temporal when the scientific problem reads series, panel or
    /// event data, or multiple environments under a temporal structure; Bayesian when the
    /// claim reads a posterior.
    #[must_use]
    pub fn refutation_quadrant(&self) -> TargetQuadrant {
        TargetQuadrant {
            temporal: temporal_refutation_data(
                self.scientific().data(),
                Some(self.scientific().graph_class()),
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
        let identity = self.semantic_identity().ok_or(CausalError::Unsupported {
            message: "a runtime callback declaration cannot identify an executable refutation target",
        })?;
        RefutationTarget::new(family, self.refutation_quadrant(), labels, *identity.as_bytes())
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
            // A response level remains a world value even when it has one coordinate or
            // is evaluated at a temporal horizon. It is not an average-effect contrast.
            ResponseFunctional::MeanCurve { .. }
            | ResponseFunctional::InterventionResponse { .. } => TargetFamily::WorldValued,
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
            // One row-major pass accumulates every component's sum (same per-component
            // addition order as a column walk, without the strided access).
            let mut component_sums = vec![0.0_f64; width];
            for row in score.noise_contributions.chunks_exact(width.max(1)) {
                for (sum, value) in component_sums.iter_mut().zip(row) {
                    *sum += value;
                }
            }
            for (component, sum) in score.noise_components.iter().zip(&component_sums) {
                labels.push(Arc::from(format!(
                    "anomaly[{target}].component[{}]",
                    component.variable().raw()
                )));
                values.push(sum / rows as f64);
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

/// The atoms of an interventional distribution in a coordinate order that does not depend on
/// the rows: an estimate lists outcome levels in order of first appearance, so a resample or a
/// permutation that changes the first row would otherwise exchange the coordinates of the
/// vector. Atoms are ordered by conditioning assignment, then by outcome value (numerically
/// where the values are numbers).
pub(crate) fn canonical_law_atoms(
    atoms: &[antecedent_estimate::DistributionAtom],
) -> Vec<antecedent_estimate::DistributionAtom> {
    use std::cmp::Ordering;
    let compare = |left: &[(VariableId, antecedent_core::Value)],
                   right: &[(VariableId, antecedent_core::Value)]| {
        for ((_, a), (_, b)) in left.iter().zip(right) {
            let order = match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => format!("{a:?}").cmp(&format!("{b:?}")),
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    };
    let mut sorted = atoms.to_vec();
    sorted.sort_by(|a, b| {
        compare(&a.conditioning, &b.conditioning).then_with(|| compare(&a.outcomes, &b.outcomes))
    });
    sorted
}

/// The probability atoms of an interventional distribution, all but the last (the others
/// determine it, and a vector that sums to one has a singular replicate covariance). A
/// distribution with a single atom, or a mean only, is read through its mean.
fn law_values(result: &StudyResult) -> Result<(Vec<Arc<str>>, Vec<f64>), CausalError> {
    let distribution = result.distribution().ok_or(CausalError::Unsupported {
        message: "no interventional distribution payload to refute",
    })?;
    let atoms = &canonical_law_atoms(&distribution.atoms);
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
#[allow(
    clippy::too_many_lines,
    reason = "the target-specific branches preserve a single checked operation"
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
        _ if result.counterfactual_contrasts().is_some() => {
            let set = result
                .counterfactual_contrasts()
                .ok_or_else(|| refuse("no counterfactual payload to refute"))?;
            (
                set.contrasts
                    .iter()
                    .map(|contrast| {
                        Arc::<str>::from(format!("mean_ite[{}]", contrast.outcome.raw()))
                    })
                    .collect(),
                set.contrasts.iter().map(|contrast| contrast.mean_ite).collect(),
            )
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

/// Internal stable selector for one actual point-producing temporal class member.
/// Enumeration position is deliberately not part of the identity.
#[derive(Clone, Debug)]
struct TemporalResponseMemberSelector {
    outer_atom: u64,
    horizon_index: usize,
    horizon: u32,
    completion_digest: [u8; 32],
}

mod contributors;
#[cfg(test)]
use contributors::compare_static_class_witnesses;
use contributors::{
    check_static_response_contributors, check_temporal_response_contributors,
    exact_admg_contributors, same_admg_posterior_source, same_static_class_witnesses,
    temporal_response_member_values,
};

/// Quadrant of `result`: its data classification, retained structure for multiple
/// environments, and whether it carries a posterior.
#[must_use]
pub fn result_quadrant(result: &StudyResult) -> TargetQuadrant {
    if let Some(quadrant) = result.lowered_quadrant {
        return quadrant;
    }
    TargetQuadrant {
        temporal: temporal_refutation_data(
            result.logical_plan.data_classification,
            result.certificate.as_ref().map(|certificate| certificate.graph_class.as_str()),
        ),
        bayesian: result.posterior.is_some(),
    }
}

// Multi-environment is a partitioning declaration, not a declaration of iid rows:
// both static populations and lagged systems use it. Read the retained structure
// class rather than labeling every environment collection static (or temporal).
fn temporal_refutation_data(data: DataClassification, graph_class: Option<&str>) -> bool {
    match data {
        DataClassification::Tabular => false,
        DataClassification::Temporal | DataClassification::Panel | DataClassification::Event => {
            true
        }
        DataClassification::MultiEnvironment => [
            crate::GraphClass::TemporalDag,
            crate::GraphClass::TemporalCpdag,
            crate::GraphClass::TemporalPag,
            crate::GraphClass::TemporalAdmg,
        ]
        .iter()
        .any(|class| Some(class.as_str()) == graph_class),
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

/// A pooled result is a new target, not the first completed-table result.
/// Do not carry its first-table checks forward as evidence for that target.
pub(crate) fn reset_pooled_refutations(result: &mut StudyResult) {
    if !result.refutations.is_empty() {
        result.view.refutations.clear();
        result.view.diagnostics.push(antecedent_core::Diagnostic::new(
            "refute.imputation.withheld",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "completed-table refutations are withheld for the pooled estimate",
        ));
    }
    result.authority = crate::result::RuntimeRefutationAuthority::Fresh;
}

// This is an in-process owner state, never reconstructed from serialized fields.
// The result view invalidates it before giving a mutable reference to public
// target/identification/report state; no recursive caller-owned query is hashed.
#[allow(
    clippy::unnecessary_wraps,
    reason = "issuance callers use a shared fallible sealing interface"
)]
pub(super) fn seal_runtime_refutations(
    result: &mut StudyResult,
    origin: crate::result::RuntimeRefutationOrigin,
) -> Result<(), CausalError> {
    if result.lowered_program_identity().is_some() {
        result.authority =
            crate::result::RuntimeRefutationAuthority::Sealed { origin, owner: Arc::new(()) };
    }
    Ok(())
}

pub(super) fn validate_runtime_refutation_proof(result: &StudyResult) -> Result<(), CausalError> {
    match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { .. } => Ok(()),
        crate::result::RuntimeRefutationAuthority::Fresh if result.refutations.is_empty() => Ok(()),
        _ => Err(CausalError::Conflict {
            what: "refutations",
            detail: "checked result was modified or reports lack an issued binding",
        }),
    }
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
pub(crate) fn refuse_scalar_refutations_for_result(
    result: &StudyResult,
) -> Result<(), TargetCertificationError> {
    use crate::result::{RuntimeRefutationAuthority, RuntimeRefutationOrigin};
    match &result.authority {
        RuntimeRefutationAuthority::Invalidated => {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("modified result"),
                claim_target: "executed result".to_owned(),
                report_target: "modified target or report vector".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        RuntimeRefutationAuthority::Fresh if !result.refutations.is_empty() => {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("unissued refutation"),
                claim_target: "executed result".to_owned(),
                report_target: "caller-declared report".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        RuntimeRefutationAuthority::Sealed { origin: RuntimeRefutationOrigin::Route, .. }
            if result.response().is_some()
                && result.structural_response.is_some()
                && !result.refutations.is_empty() =>
        {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("class response"),
                claim_target: "whole-result target".to_owned(),
                report_target: "per-completion route checks".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        _ => {}
    }
    // Scope metadata is descriptive until the checked-operation authority is
    // retained and verified. A same-program stamp cannot make it authoritative.
    if let Some(report) = result.refutations.iter().find(|report| report.scope.is_some()) {
        return Err(TargetCertificationError::TargetMismatch {
            refuter: Arc::clone(&report.refuter),
            claim_target: "whole-result target".to_owned(),
            report_target: "unverified descriptive scope".to_owned(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        });
    }
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

/// Identify engine-issued structural route reports that may be exported only
/// as descriptive evidence with an unresolved replay dependency.
pub(super) fn descriptive_route_report_source(result: &StudyResult) -> Option<[u8; 32]> {
    use crate::result::{RuntimeRefutationAuthority, RuntimeRefutationOrigin};
    if !matches!(
        &result.authority,
        RuntimeRefutationAuthority::Sealed {
            origin: RuntimeRefutationOrigin::Route | RuntimeRefutationOrigin::ScopedCompletion,
            ..
        }
    ) || result.response().is_none()
        || result.structural_response.is_none()
        || result.refutations.is_empty()
    {
        return None;
    }
    let program = *result.lowered_identity?.as_bytes();
    result.refutations.iter().all(|report| report.stamped_by(&program)).then_some(program)
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
    outcome: &TargetRefutationOutcome,
) -> Result<TargetCertificate, TargetCertificationError> {
    refuse_scalar_refutations_for_result(result)?;
    let owner = match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } => Some(owner),
        _ => None,
    };
    if !owner.zip(outcome.owner.as_ref()).is_some_and(|(a, b)| Arc::ptr_eq(a, b)) {
        return Err(TargetCertificationError::TargetMismatch {
            refuter: Arc::from("target operation"),
            claim_target: "executed result".to_owned(),
            report_target: "unissued, modified, or foreign target outcome".to_owned(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        });
    }
    let claim = result_refutation_target(result).map_err(|error| {
        TargetCertificationError::TargetMismatch {
            refuter: Arc::from("target operation"),
            claim_target: format!("undefined ({error})"),
            report_target: String::new(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        }
    })?;
    certify_target_refutations(&claim, &outcome.run.reports)
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
            let DerivativeCoordinate::FiniteDifference(check) = coordinate else {
                continue;
            };
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
            let DerivativeCoordinate::Linearity(check) = coordinate else {
                continue;
            };
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
fn repack_lag_mask(
    old: &antecedent_discovery::LagMask,
    n: usize,
    m: usize,
    max_lag: usize,
) -> antecedent_discovery::LagMask {
    let mut packed = antecedent_discovery::LagMask::EMPTY;
    for bit in old.set_bits().map(|bit| bit as usize).filter(|&bit| bit < max_lag * n * n) {
        let (lag, from, to) = (bit / (n * n), (bit % (n * n)) / n, bit % n);
        packed.set_bit(u32::try_from(lag * m * m + from * m + to).unwrap_or(u32::MAX), true);
    }
    packed
}

#[cfg(test)]
mod repack_tests {
    use antecedent_validate::target_native::{TargetFamily, TargetQuadrant};

    use super::*;

    fn test_join_rows(map: &[usize], span: usize, block: Option<usize>) -> JoinFreeRows {
        let ctx = ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        join_free_rows(map, span, block, &mut budget, &ctx).unwrap()
    }

    #[test]
    fn block_boundary_cannot_bridge_even_successive_source_ticks() {
        let ticks = [3, 4, 5, 6, 7, 8];
        assert_eq!(*test_join_rows(&ticks, 2, Some(3)), [false, true, true, false, true, true]);
        assert_eq!(*test_join_rows(&ticks, 3, Some(3)), [false, false, true, false, false, true]);
        assert_eq!(*test_join_rows(&ticks, 2, None), [false, true, true, true, true, true]);
    }

    /// A gathered circular-block series, with a series whose every value names its own period:
    /// the lag pair a design row reads is `(source of row - k, source of row)`, and a pair is
    /// an observed one exactly when its sources are `k` periods apart. Built here from the
    /// block definition, not from the product's alignment test.
    #[test]
    fn a_gathered_block_bootstrap_hides_every_design_row_whose_lag_window_crosses_a_join() {
        let n = 60;
        for (seed, span) in [(1_u64, 2_usize), (2, 3), (3, 2), (4, 4)] {
            let mut rng = antecedent_core::RngFactory::from_seed(seed)
                .stream_for(antecedent_core::StreamDomain::Test, 9);
            let length = antecedent_data::circular_block_length(span, n);
            let mut map = vec![0_usize; n];
            antecedent_data::fill_circular_block_indexes(n, length, &mut rng, &mut map).unwrap();
            let free = test_join_rows(&map, span, Some(length));
            let mut joins = 0;
            for row in 0..n {
                // Brute force from the definition: the window of `row` is the original's when
                // each earlier member is the source period exactly its lag before (the wrap of
                // a block past the last period joins the last period to the first, so it is a
                // join); the first rows have no earlier member to cross a join with.
                let observed = row + 1 >= span
                    && row % length + 1 >= span
                    && (1..span).all(|k| map[row] >= k && map[row - k] == map[row] - k);
                assert_eq!(free[row], observed, "row {row}, span {span}");
                joins += usize::from(!free[row]);
            }
            assert!(joins > 0, "the construction must contain joins to be a test");
            // The mask carries it: the rows an attribution fits are the join-free ones.
            let storage = TimeSeriesData::from_f64_columns(
                [("x", vec![0.0; n].as_slice()), ("y", vec![0.0; n].as_slice())],
                1,
            )
            .unwrap();
            let mask = join_free_mask(
                storage.storage(),
                &map,
                span,
                Some(length),
                &mut antecedent_core::TemporalBudget::new(
                    ExecutionContext::for_tests(0).temporal_materialization_limits(),
                ),
                &ExecutionContext::for_tests(0),
            )
            .unwrap();
            for row in 0..n {
                assert_eq!(mask.is_valid(row), free[row]);
            }
        }
    }

    /// A contiguous sub-window has no join: its source map is consecutive, so every design row
    /// reads the lag window the original reads (judged by the same brute-force definition as
    /// the block bootstrap above), for any window length and span.
    #[test]
    fn a_contiguous_subsample_window_has_no_crossing_design_row() {
        let mut rng = antecedent_core::RngFactory::from_seed(3)
            .stream_for(antecedent_core::StreamDomain::Test, 4);
        for (rows, span) in [(12_usize, 2_usize), (30, 3), (7, 4), (100, 2)] {
            for _ in 0..50 {
                let (start, width) = subsample_bounds(rows, span, &mut rng);
                assert!(width >= (span + 2).min(rows) && start + width <= rows);
                let map: Vec<usize> = (start..start + width).collect();
                let free = test_join_rows(&map, span, None);
                for row in 0..width {
                    let observed = row + 1 >= span
                        && (1..span).all(|k| map[row] >= k && map[row - k] == map[row] - k);
                    assert_eq!(observed, free[row], "rows {rows}, span {span}, row {row}");
                }
            }
        }
    }

    /// The multiplicity form of the block bootstrap (read by the routes that take row weights)
    /// puts every unit of weight on a source design row, which keeps its own lag window, so a
    /// join cannot appear: the weights are exactly the counts of the source rows the gathered
    /// form draws, with the same stream.
    #[test]
    fn block_multiplicities_are_the_counts_of_source_design_rows() {
        for (n, span) in [(40_usize, 2_usize), (57, 3)] {
            let draw = || {
                antecedent_core::RngFactory::from_seed(8)
                    .stream_for(antecedent_core::StreamDomain::Test, 1)
            };
            let weights = block_weights(n, span, &mut draw()).unwrap();
            let length = antecedent_data::circular_block_length(span, n);
            let mut map = vec![0_usize; n];
            antecedent_data::fill_circular_block_indexes(n, length, &mut draw(), &mut map).unwrap();
            let mut counts = vec![0.0_f64; n];
            for &source in &map {
                counts[source] += 1.0;
            }
            assert_eq!(&weights[..], &counts[..]);
            assert!((weights.iter().sum::<f64>() - n as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn multi_environment_refutation_quadrant_preserves_the_retained_structure_kind() {
        let data = DataClassification::MultiEnvironment;
        for class in [
            crate::GraphClass::TemporalDag,
            crate::GraphClass::TemporalCpdag,
            crate::GraphClass::TemporalPag,
            crate::GraphClass::TemporalAdmg,
        ] {
            assert!(temporal_refutation_data(data, Some(class.as_str())));
        }
        for class in [
            crate::GraphClass::Dag,
            crate::GraphClass::Admg,
            crate::GraphClass::Cpdag,
            crate::GraphClass::Pag,
        ] {
            assert!(!temporal_refutation_data(data, Some(class.as_str())));
        }
        assert!(!temporal_refutation_data(data, Some("cyclic")));
        assert!(!temporal_refutation_data(data, None));
        assert!(temporal_refutation_data(DataClassification::Temporal, None));
    }

    #[test]
    fn retained_multi_environment_pulse_and_its_lowered_plan_are_both_temporal() {
        let environments = (0..3)
            .map(|environment| {
                let x: Vec<_> = (0..160)
                    .map(|row| (f64::from(row) * 1.37 + f64::from(environment)).sin())
                    .collect();
                let y: Vec<_> = (0..160)
                    .map(|row| {
                        let past = if row == 0 { 0.0 } else { x[row - 1] };
                        let time = f64::from(u32::try_from(row).unwrap());
                        1.0 + 2.0 * past + 0.1 * (time * 2.17).cos()
                    })
                    .collect();
                TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let data = antecedent_data::MultiEnvironmentData::try_new(environments).unwrap();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let x1 = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        let query = antecedent_core::TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1);
        let ctx = ExecutionContext::for_tests(38);
        let prepared = crate::Study::series_multi(data.clone())
            .graph(graph)
            .temporal_query(query)
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let plan = prepared.plan().lowered().unwrap();
        assert_eq!(plan.refutation_quadrant(), TargetQuadrant::TEMPORAL_FREQUENTIST);
        let result = prepared.estimate_multi_env(&data, &ctx).unwrap();
        assert_eq!(result_quadrant(&result), plan.refutation_quadrant());
        let target = result_refutation_target(&result).unwrap();
        assert_eq!(target.source(), result.lowered_program_identity().unwrap().as_bytes());
        assert_eq!(target.quadrant(), TargetQuadrant::TEMPORAL_FREQUENTIST);
    }

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
    fn a_class_within_the_limit_keeps_every_member_and_a_larger_one_a_stable_subsample() {
        let member = |i: usize| {
            antecedent_discovery::adjacency_mask::set_edge(
                antecedent_discovery::AdjacencyMask::EMPTY,
                16,
                0,
                1 + i % 15,
                true,
            )
        };
        let mut all = ClassSample::new(8);
        for i in 0..8 {
            all.offer(i, member(i));
        }
        assert_eq!(all.seen, 8);
        assert_eq!(all.into_members(), (0..8).map(member).collect::<Vec<_>>());
        let sample = |limit: usize| {
            let mut kept = ClassSample::new(limit);
            for i in 0..1000 {
                kept.offer(i, member(i));
            }
            kept
        };
        let kept = sample(16);
        assert_eq!((kept.seen, kept.members.len()), (1000, 16));
        // The subsample is the 16 positions of smallest rank, whatever the order they arrive in.
        let mut expected: Vec<usize> = (0..1000).collect();
        expected.sort_by_key(|&i| (ClassSample::rank(i), i));
        expected.truncate(16);
        expected.sort_unstable();
        let positions: Vec<usize> = kept.members.iter().map(|&(_, p, _)| p).collect();
        let mut positions_sorted = positions;
        positions_sorted.sort_unstable();
        assert_eq!(positions_sorted, expected);
    }

    #[test]
    fn a_lagged_edge_keeps_its_lag_source_and_target_when_a_node_is_added() {
        // Two nodes, two lags: bit (lag, from, to) = lag * 4 + from * 2 + to.
        let old = antecedent_discovery::LagMask::from_word((1_u64 << 1) | (1_u64 << (4 + 2)));
        // lag 1: 0 -> 1; lag 2: 1 -> 0
        let packed = repack_lag_mask(&old, 2, 3, 2);
        assert_eq!(packed, (1_u64 << 1) | (1_u64 << (9 + 3)));
        assert!(repack_lag_mask(&antecedent_discovery::LagMask::EMPTY, 2, 3, 2).is_empty());
        // A template wider than a machine word repacks too.
        let wide = antecedent_discovery::LagMask::EMPTY.with_bit(7 * 49 + 3 * 7 + 5, true);
        let moved = repack_lag_mask(&wide, 7, 8, 8);
        assert!(moved.bit(7 * 64 + 3 * 8 + 5) && moved.count_ones() == 1);
    }
}

/// Interventional means one derivative coordinate reads: the evaluation point and one step each way.
const DERIVATIVE_POINTS_PER_COORDINATE: usize = 3;

/// DAG members of one class the common-cause refuter retains when the context declares no memory
/// limit; a larger class is reduced to a deterministic subsample of this size.
const DEFAULT_WITHIN_CLASS_MEMBERS: usize = 512;

/// Bytes budgeted per enumerated DAG member of an `n`-node class: its edge set, the augmented
/// copy with the common cause, and the bookkeeping of both.
const fn within_class_member_bytes(n: usize) -> u64 {
    16 * (n as u64 * n as u64 + 64)
}

/// Most DAG members of one `n`-node class the common-cause refuter retains: as many as the
/// context's memory limit holds, or [`DEFAULT_WITHIN_CLASS_MEMBERS`] when it declares none. A
/// class with more members is subsampled to this many (see [`ClassSample`]).
fn within_class_member_cap(ctx: &ExecutionContext, n: usize) -> usize {
    ctx.memory_limit_bytes().map_or(DEFAULT_WITHIN_CLASS_MEMBERS, |limit| {
        usize::try_from(limit / within_class_member_bytes(n)).unwrap_or(usize::MAX).max(1)
    })
}

/// The members of one equivalence class a bounded retention keeps, with the number enumerated.
///
/// Each member is ranked by a fixed hash of its position in the enumeration, and the `limit`
/// members of smallest rank are kept. The enumeration order is deterministic, so the sample is
/// too, and because the hash does not depend on the member's content or on `limit`, the kept
/// members are a uniform draw without replacement from the class. A class within `limit` keeps
/// every member, in enumeration order.
struct ClassSample {
    limit: usize,
    /// Members kept, as (rank, position in the enumeration, member).
    members: Vec<(u64, usize, antecedent_discovery::AdjacencyMask)>,
    /// Members offered, kept or not.
    seen: usize,
}

impl ClassSample {
    fn new(limit: usize) -> Self {
        Self { limit: limit.max(1), members: Vec::new(), seen: 0 }
    }

    /// The splitmix64 finalizer of the position: a fixed, well mixed rank.
    fn rank(position: usize) -> u64 {
        let mut z = (position as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn offer(&mut self, position: usize, member: antecedent_discovery::AdjacencyMask) {
        self.seen += 1;
        let rank = Self::rank(position);
        if self.members.len() < self.limit {
            self.members.push((rank, position, member));
            return;
        }
        // A position never repeats, so the ranks of two members differ unless the hash
        // collides; the position breaks such a tie.
        let Some((worst, _)) = self
            .members
            .iter()
            .enumerate()
            .max_by_key(|(_, (rank, position, _))| (*rank, *position))
        else {
            return;
        };
        let (worst_rank, worst_position, _) = self.members[worst];
        if (rank, position) < (worst_rank, worst_position) {
            self.members[worst] = (rank, position, member);
        }
    }

    /// The kept members in enumeration order.
    fn into_members(mut self) -> Vec<antecedent_discovery::AdjacencyMask> {
        self.members.sort_unstable_by_key(|&(_, position, _)| position);
        self.members.into_iter().map(|(_, _, member)| member).collect()
    }
}

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
/// Readable target-native outputs. Mutable access invalidates checked authority.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefutationOutcomeView {
    /// The plan attached to the result's lowered program.
    pub plan: TargetRefutationPlan,
    /// Reports and refusals.
    pub run: TargetRefutationRun,
    /// The reports, bound to the result's target.
    pub certificate: TargetCertificate,
}

/// A checked prepared-operation outcome. Its private owner is not deserializable.
#[derive(Clone, Debug)]
pub struct TargetRefutationOutcome {
    view: TargetRefutationOutcomeView,
    owner: Option<Arc<()>>,
}
impl std::ops::Deref for TargetRefutationOutcome {
    type Target = TargetRefutationOutcomeView;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}
impl std::ops::DerefMut for TargetRefutationOutcome {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.owner = None;
        &mut self.view
    }
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
    let checked = checked_refutation_input(result, data, ctx)?;
    let owner = Arc::clone(&checked.owner);
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
        if lowered.semantic_identity().is_none()
            || lowered.semantic_identity() != result.lowered_program_identity()
        {
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
        member: None,
        temporal_budget: prepared.study().temporal_validation_budget.clone().unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        }),
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
    Ok(TargetRefutationOutcome {
        view: TargetRefutationOutcomeView { plan, run, certificate },
        owner: Some(owner),
    })
}

// Run a scoped member's actual level target through full-program refits. The
// selector comes from the point-time witness retained by this same operation.
#[allow(
    clippy::too_many_arguments,
    reason = "the checked refit binds its member, data and execution context"
)]
pub(super) fn refute_temporal_response_member(
    prepared: &PreparedStudy,
    result: &StudyResult,
    data: PreparedTargetData<'_>,
    outer_atom: u64,
    witness: &crate::result::TemporalResponseWitness,
    suite: TargetSuite,
    config: &TargetRefutationConfig,
    checked: &CheckedRefutationInput<'_>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(antecedent_validate::RefutationScope, TargetRefutationRun), CausalError> {
    if !matches!(&result.authority, crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } if Arc::ptr_eq(owner, &checked.owner))
    {
        return Err(CausalError::Conflict {
            what: "result",
            detail: "the checked refutation input belongs to another result",
        });
    }
    let source = result.lowered_program_identity().ok_or(CausalError::Unsupported {
        message: "scoped temporal refutation requires the producing lowered program",
    })?;
    let lowered = prepared.plan().lowered().ok_or(CausalError::Unsupported {
        message: "scoped temporal refutation requires a lowered prepared program",
    })?;
    if lowered.semantic_identity() != Some(source) {
        return Err(CausalError::Conflict {
            what: "program",
            detail: "the result and scoped refit have different lowered programs",
        });
    }
    let selector = TemporalResponseMemberSelector {
        outer_atom,
        horizon_index: witness.horizon_index,
        horizon: witness.horizon,
        completion_digest: witness.completion_digest,
    };
    let original = temporal_response_member_values(result, &selector, budget, ctx)?;
    let prefix = format!(
        "scope:v1:temporal_class_completion:{}:{}:{}:{}",
        outer_atom,
        witness.horizon_index,
        witness.horizon,
        antecedent_core::identity::to_hex(&witness.completion_digest)
    );
    let labels = (0..original.len()).map(|i| format!("{prefix}|coordinate[{i}]"));
    let target = RefutationTarget::new(
        TargetFamily::WorldValued,
        result_quadrant(result),
        labels,
        *source.as_bytes(),
    )
    .map_err(CausalError::from)?;
    let treatment = result.treatment.ok_or(CausalError::Unsupported {
        message: "scoped temporal response requires a treatment",
    })?;
    let plan = TargetRefutationPlan::for_target(target.clone(), suite);
    let refit = PreparedRefit {
        prepared,
        result,
        data,
        target: target.clone(),
        member: Some(selector),
        temporal_budget: budget.clone(),
        original,
        treatment,
        outcome: result.outcome,
        options: TargetRefitOptions::default(),
        replicates: config.replicates,
        alpha: config.alpha,
        curve: std::cell::OnceCell::new(),
    };
    let run = run_target_plan(&plan, &refit, config, ctx)?;
    // Checked target certification also rejects any report naming another target.
    certify_target_refutations(&target, &run.reports)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    let scope = antecedent_validate::RefutationScope::TemporalClassCompletion {
        outer_atom,
        horizon_index: witness.horizon_index,
        horizon: witness.horizon,
        completion_digest: witness.completion_digest,
        target,
    };
    Ok((scope, run))
}

impl PreparedStudy {
    /// Refute the supplied result on this preparation's exact retained observations.
    ///
    /// The caller must supply the result produced by this handle. The existing target
    /// certification checks bind the result to the lowered program; no data is re-ingested.
    ///
    /// # Errors
    /// The same target, modality, execution and admission errors as [`refute_prepared_target`].
    pub fn refute_retained_target(
        &self,
        result: &StudyResult,
        suite: TargetSuite,
        config: &TargetRefutationConfig,
        options: &TargetRefitOptions,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationOutcome, CausalError> {
        let data = match &self.study().data {
            DataInput::Tabular(data) => PreparedTargetData::Tabular(data),
            DataInput::Temporal(data) | DataInput::Event(data) => PreparedTargetData::Series(data),
            DataInput::Panel(data) => PreparedTargetData::Panel(data),
            DataInput::MultiEnv(data) => PreparedTargetData::MultiEnv(data),
        };
        refute_prepared_target(self, result, data, suite, config, options, ctx)
    }
}

// Compare caller-supplied observations against the immutable input actually
// retained by this execution. Content scanning is chunked, cancellable and
// admitted; schema strings are compared in bounded chunks.
fn same_bytes(a: &[u8], b: &[u8], ctx: &ExecutionContext) -> Result<bool, CausalError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.chunks(4096).zip(b.chunks(4096)) {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "refutation input identity" });
        }
        if a != b {
            return Ok(false);
        }
    }
    Ok(true)
}

fn same_schema(
    a: &antecedent_core::CausalSchema,
    b: &antecedent_core::CausalSchema,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (x, y) in a.variables().iter().zip(b.variables()) {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "refutation input identity" });
        }
        if x.id != y.id
            || x.value_type != y.value_type
            || x.role_hints != y.role_hints
            || x.category_domain != y.category_domain
            || x.measurement.noisy != y.measurement.noisy
            || !same_bytes(x.name.as_bytes(), y.name.as_bytes(), ctx)?
        {
            return Ok(false);
        }
        for (left, right) in
            [(&x.unit, &y.unit), (&x.measurement.description, &y.measurement.description)]
        {
            match (left, right) {
                (None, None) => {}
                (Some(left), Some(right))
                    if same_bytes(left.as_bytes(), right.as_bytes(), ctx)? => {}
                _ => return Ok(false),
            }
        }
    }
    Ok(true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutedInputKind {
    Tabular,
    Temporal,
    Event,
    Panel,
    MultiEnv,
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutedInputPart {
    schema: antecedent_core::CausalSchema,
    digest: [u8; 32],
    time: Option<antecedent_data::TimeIndex>,
    unit_id: Option<u32>,
}

#[derive(Debug)]
struct ExecutedInputParts {
    parts: Vec<ExecutedInputPart>,
    _lease: antecedent_core::execution::AdmissionLease,
}

#[derive(Clone, Debug)]
enum ExecutedInputStorage {
    Single(ExecutedInputPart),
    Segmented(Arc<ExecutedInputParts>),
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutedInputIdentity {
    kind: ExecutedInputKind,
    storage: ExecutedInputStorage,
}
impl ExecutedInputIdentity {
    fn parts(&self) -> &[ExecutedInputPart] {
        match &self.storage {
            ExecutedInputStorage::Single(part) => std::slice::from_ref(part),
            ExecutedInputStorage::Segmented(parts) => &parts.parts,
        }
    }
}

fn input_digest(
    storage: &antecedent_data::OwnedColumnarStorage,
    ctx: &ExecutionContext,
) -> Result<[u8; 32], CausalError> {
    storage.content_digest_with_context(ctx).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "refutation input identity" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "the target-specific branches preserve a single checked operation"
)]
pub(crate) fn executed_input_identity(
    data: &DataInput,
    ctx: &ExecutionContext,
) -> Result<ExecutedInputIdentity, CausalError> {
    // A run cancelled before it was finalised (a bootstrap stopped part-way) still returns the
    // partial result it reached, and that result is bound to its input. Only a cancellation
    // arriving while the input is being identified stops the identification.
    let finalised;
    let ctx = if ctx.cancellation.is_cancelled() {
        let mut fresh = ctx.clone();
        fresh.cancellation = antecedent_core::CancellationToken::new();
        finalised = fresh;
        &finalised
    } else {
        ctx
    };
    let part = |storage: &antecedent_data::OwnedColumnarStorage,
                time: Option<antecedent_data::TimeIndex>,
                unit_id: Option<u32>|
     -> Result<ExecutedInputPart, CausalError> {
        Ok(ExecutedInputPart {
            schema: storage.schema().clone(),
            digest: input_digest(storage, ctx)?,
            time,
            unit_id,
        })
    };
    let (kind, storage) = match data {
        DataInput::Tabular(table) => (
            ExecutedInputKind::Tabular,
            ExecutedInputStorage::Single(part(table.storage(), None, None)?),
        ),
        DataInput::Temporal(series) => (
            ExecutedInputKind::Temporal,
            ExecutedInputStorage::Single(part(
                series.storage(),
                Some(series.time_index().clone()),
                None,
            )?),
        ),
        DataInput::Event(series) => (
            ExecutedInputKind::Event,
            ExecutedInputStorage::Single(part(
                series.storage(),
                Some(series.time_index().clone()),
                None,
            )?),
        ),
        DataInput::Panel(panel) => {
            let count = panel.units().len();
            let bytes = u64::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(256))
                .and_then(|n| n.checked_add(512))
                .ok_or_else(|| CausalError::Resource {
                    message: "input identity size overflow".into(),
                })?;
            let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "refutation input identity" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
            let mut parts = Vec::new();
            parts
                .try_reserve_exact(count)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?;
            for unit in panel.units() {
                parts.push(part(
                    unit.series.storage(),
                    Some(unit.series.time_index().clone()),
                    Some(unit.unit_id),
                )?);
            }
            (
                ExecutedInputKind::Panel,
                ExecutedInputStorage::Segmented(Arc::new(ExecutedInputParts {
                    parts,
                    _lease: lease,
                })),
            )
        }
        DataInput::MultiEnv(multi) => {
            let count = multi.environments().len();
            let bytes = u64::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(256))
                .and_then(|n| n.checked_add(512))
                .ok_or_else(|| CausalError::Resource {
                    message: "input identity size overflow".into(),
                })?;
            let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "refutation input identity" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
            let mut parts = Vec::new();
            parts
                .try_reserve_exact(count)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?;
            for series in multi.environments() {
                parts.push(part(series.storage(), Some(series.time_index().clone()), None)?);
            }
            (
                ExecutedInputKind::MultiEnv,
                ExecutedInputStorage::Segmented(Arc::new(ExecutedInputParts {
                    parts,
                    _lease: lease,
                })),
            )
        }
    };
    Ok(ExecutedInputIdentity { kind, storage })
}

fn same_part(
    part: &ExecutedInputPart,
    storage: &antecedent_data::OwnedColumnarStorage,
    time: Option<&antecedent_data::TimeIndex>,
    unit_id: Option<u32>,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    Ok(part.unit_id == unit_id
        && part.time.as_ref() == time
        && same_schema(&part.schema, storage.schema(), ctx)?
        && part.digest == input_digest(storage, ctx)?)
}

fn same_executed_input(
    original: &ExecutedInputIdentity,
    supplied: PreparedTargetData<'_>,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "refutation input identity" });
    }
    match (original.kind, supplied) {
        (ExecutedInputKind::Tabular, PreparedTargetData::Tabular(table)) => {
            Ok(original.parts().len() == 1
                && same_part(&original.parts()[0], table.storage(), None, None, ctx)?)
        }
        (
            ExecutedInputKind::Temporal | ExecutedInputKind::Event,
            PreparedTargetData::Series(series),
        ) => Ok(original.parts().len() == 1
            && same_part(
                &original.parts()[0],
                series.storage(),
                Some(series.time_index()),
                None,
                ctx,
            )?),
        (ExecutedInputKind::Panel, PreparedTargetData::Panel(panel)) => {
            if original.parts().len() != panel.units().len() {
                return Ok(false);
            }
            for (part, unit) in original.parts().iter().zip(panel.units()) {
                if !same_part(
                    part,
                    unit.series.storage(),
                    Some(unit.series.time_index()),
                    Some(unit.unit_id),
                    ctx,
                )? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (ExecutedInputKind::MultiEnv, PreparedTargetData::MultiEnv(multi)) => {
            if original.parts().len() != multi.environments().len() {
                return Ok(false);
            }
            for (part, series) in original.parts().iter().zip(multi.environments()) {
                if !same_part(part, series.storage(), Some(series.time_index()), None, ctx)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Borrow-scoped capability issued only after comparing actual execution input.
pub(super) struct CheckedRefutationInput<'a> {
    owner: Arc<()>,
    _result: std::marker::PhantomData<&'a StudyResult>,
}

pub(super) fn checked_refutation_input<'a>(
    result: &'a StudyResult,
    data: PreparedTargetData<'_>,
    ctx: &ExecutionContext,
) -> Result<CheckedRefutationInput<'a>, CausalError> {
    validate_runtime_refutation_proof(result)?;
    let owner = match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } => Arc::clone(owner),
        _ => {
            return Err(CausalError::Conflict {
                what: "result",
                detail: "target refutation requires an unmodified engine-produced result",
            });
        }
    };
    let actual = result.executed_input.as_ref().ok_or(CausalError::Conflict {
        what: "data",
        detail: "the executed observations were not retained for target refutation",
    })?;
    if !same_executed_input(actual, data, ctx)? {
        return Err(CausalError::Conflict {
            what: "data",
            detail: "the refit observations differ from those that produced the result",
        });
    }
    Ok(CheckedRefutationInput { owner, _result: std::marker::PhantomData })
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
    member: Option<TemporalResponseMemberSelector>,
    temporal_budget: antecedent_core::TemporalBudget,
    original: Vec<f64>,
    treatment: VariableId,
    outcome: VariableId,
    options: TargetRefitOptions,
    replicates: u32,
    alpha: f64,
    /// Fitted response curve of a derivative, `(point, value per horizon)`, computed once.
    curve: std::cell::OnceCell<Vec<(f64, Vec<f64>)>>,
}

/// [`antecedent_kernels::design_gram_cholesky`] with a rank-deficient design reported as the
/// refit refusal `rank_deficient`.
fn gram_cholesky_factor(
    design: &[f64],
    n: usize,
    p: usize,
    rank_deficient: &'static str,
) -> Result<Vec<f64>, TargetRefitError> {
    antecedent_kernels::design_gram_cholesky(design, n, p)
        .ok_or_else(|| unsupported(rank_deficient, "a full-rank adjustment"))
}

fn unsupported(reason: impl Into<Arc<str>>, next_attempt: &str) -> TargetRefitError {
    TargetRefitError::Unsupported { reason: reason.into(), next_attempt: Arc::from(next_attempt) }
}

/// A refuter that cannot speak to this target (a degenerate replicate law, too few draws) is
/// recorded as not run with its reason and next attempt; any other failure stays an error.
fn refuter_outcome(error: ValidationError) -> TargetRefitError {
    match error {
        ValidationError::NotApplicable { message } => unsupported(
            message,
            "raise the replicate count, or check that the perturbation moves the target",
        ),
        other => TargetRefitError::Failed(other),
    }
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
    // A perturbed table that leaves a cell the functional reads empty is a table the
    // refuter cannot refit, not a failed refit: the refuter is refused with its reason.
    let message = error.to_string();
    if message.contains("missing table cell") || message.contains("missing probability table entry")
    {
        return unsupported(
            format!("a perturbed table leaves a required probability cell empty: {message}"),
            "supply more rows per cell, or a table whose resamples keep every cell the \
             functional reads populated",
        );
    }
    TargetRefitError::Failed(match error {
        CausalError::Cancelled { .. } => ValidationError::Cancelled,
        CausalError::Resource { message } => ValidationError::Resource { message: message.clone() },
        _ => ValidationError::estimation_msg(error.to_string()),
    })
}

fn operational_refit_error(error: &TargetRefitError) -> bool {
    matches!(
        error,
        TargetRefitError::Failed(ValidationError::Cancelled | ValidationError::Resource { .. })
    )
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
        let Some(source) = t.checked_add_signed(shift).filter(|s| *s < n) else {
            continue;
        };
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

/// Fraction of a window a contiguous subsample keeps. Windows this short vary about as much
/// as the full-window estimate would across samples (half-length windows vary too little and
/// reject a clean series on a quarter to a third of seeds; see the lead and control sweep in
/// `temporal_conditional_validation_truth`), so a null check against them is not over-eager.
const SUBSAMPLE_FRACTION: f64 = 1.0 / 3.0;

/// Start and width of a contiguous sub-window of `rows` periods for a route whose lag windows
/// may not cross a join: `SUBSAMPLE_FRACTION` of the periods, at least `span + 2` of them (or
/// all when fewer exist), starting uniformly among the positions that fit.
fn subsample_bounds(rows: usize, span: usize, rng: &mut CausalRng) -> (usize, usize) {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the fraction lies in (0, 1), and the saturating cast is clamped into [1, rows]"
    )]
    let kept = (SUBSAMPLE_FRACTION * rows as f64).round() as usize;
    let width = kept.max(span + 2).min(rows);
    (unbiased_index(rng, rows - width + 1), width)
}

/// Eligible windows use observed consecutive source ticks and do not cross a
/// draw boundary, even if two blocks happen to have adjacent tick values.
struct JoinFreeRows {
    values: Vec<bool>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for JoinFreeRows {
    type Target = [bool];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

fn join_free_rows(
    map: &[usize],
    span: usize,
    block_length: Option<usize>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<JoinFreeRows, TargetRefitError> {
    if ctx.cancellation.is_cancelled() {
        return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
    }
    if block_length == Some(0) {
        return Err(failed(&CausalError::Compile {
            message: "refit block length must be positive".into(),
        }));
    }
    let n = u64::try_from(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask work overflow".into() })
    })?;
    budget.charge_work(n).map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
    let bytes = n.checked_mul(20).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
        failed(&CausalError::Resource { message: "refit mask bytes overflow".into() })
    })?;
    if bytes > budget.limits().max_bytes {
        return Err(failed(&CausalError::Resource { message: "refit mask byte limit".into() }));
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| {
        failed(&match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask allocation refused".into() })
    })?;
    let mut length = 0_usize;
    for (row, &tick) in map.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
        }
        let continues = row > 0
            && block_length.is_none_or(|block| row % block != 0)
            && map[row - 1].checked_add(1) == Some(tick);
        length = if continues { length.saturating_add(1) } else { 1 };
        values.push(length >= span.max(1));
    }
    Ok(JoinFreeRows { values, _lease: lease })
}

/// The analysis mask of a gathered series for a route that fits row-wise lag-aligned tables
/// (an attribution): the gathered original mask with every row whose lag window crosses a join
/// hidden, because such a route reads no row weights and would otherwise fit and score the
/// artificial lag pairs of the join.
fn join_free_mask(
    storage: &antecedent_data::OwnedColumnarStorage,
    map: &[usize],
    span: usize,
    block_length: Option<usize>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<ValidityBitmap, TargetRefitError> {
    let free = join_free_rows(map, span, block_length, budget, ctx)?;
    let n = u64::try_from(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask work overflow".into() })
    })?;
    budget.charge_work(n).map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
    let mut keep = Vec::new();
    keep.try_reserve_exact(map.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "refit mask allocation refused".into() })
    })?;
    for (&eligible, &source) in free.iter().zip(map) {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
        }
        keep.push(eligible && storage.analysis_mask().is_none_or(|mask| mask.is_valid(source)));
    }
    mask_from(&keep)
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
        let OwnedColumn::Float64(column) = column else {
            return Ok(None);
        };
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

mod common_cause;
mod derivative;
mod predictive;

impl PreparedRefit<'_> {
    fn study(&self) -> &crate::analysis::execute::Study {
        self.prepared.study()
    }

    fn adjustment(&self) -> Vec<VariableId> {
        self.result.estimand.adjustment_set.iter().copied().collect()
    }

    /// Covariates of the reference outcome model besides the treatment: the estimand's
    /// adjustment set, or, for a structural-model query (a counterfactual or a nested natural
    /// effect on a DAG), which carries no adjustment set, the other float parents of the
    /// outcome, so the reference model does not fold a confounder into the treatment slope.
    fn reference_covariates(&self) -> Vec<VariableId> {
        let adjustment = self.adjustment();
        if !adjustment.is_empty()
            || !matches!(
                self.study().query,
                CausalQuery::Counterfactual(_)
                    | CausalQuery::NestedCounterfactual(_)
                    | CausalQuery::PathSpecific(_)
                    | CausalQuery::Distribution(_)
            )
        {
            return adjustment;
        }
        let mut parents: Vec<VariableId> = Vec::new();
        if let Some(posterior) = &self.study().graph_posterior {
            // A posterior carries no single structure (its placeholder graph has no edges): the
            // reference model takes the outcome's parents over every DAG atom, so it does not
            // fold an atom's confounder into the treatment slope.
            if posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag {
                for atom in posterior.atoms().iter() {
                    let Ok(dag) = antecedent_discovery::adjacency_mask::dag_from_adjacency_mask(
                        atom.clone(),
                        posterior.n_vars(),
                    ) else {
                        continue;
                    };
                    if let Some(outcome) = static_node(&dag, self.outcome) {
                        parents.extend(dag.parents(outcome).iter().map(|parent| {
                            VariableId::from_raw(
                                u32::try_from(parent.as_usize()).unwrap_or(u32::MAX),
                            )
                        }));
                    }
                }
            }
        } else if let Some(dag) = self.study().graph.as_dag() {
            if let Some(outcome) = static_node(dag, self.outcome) {
                parents.extend(dag.parents(outcome).iter().filter_map(|parent| {
                    match dag.nodes().get(parent.as_usize())? {
                        antecedent_core::NodeRef::Static(variable) => Some(*variable),
                        _ => None,
                    }
                }));
            }
        } else if let Some(tdag) = self.study().graph.as_temporal_dag() {
            for (outcome, lag) in lagged_nodes(tdag, self.outcome) {
                if lag != 0 {
                    continue;
                }
                parents.extend(tdag.parents(outcome).iter().filter_map(|parent| {
                    match tdag.nodes().get(parent.as_usize())? {
                        antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                        _ => None,
                    }
                }));
            }
        }
        parents.sort_unstable();
        parents.dedup();
        parents.retain(|variable| {
            *variable != self.treatment && *variable != self.outcome && self.is_float(*variable)
        });
        if parents.is_empty() { adjustment } else { parents }
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
        self.read_back(&refitted, true, ctx)
    }

    fn reestimate_materialised_series(
        &self,
        data: &TimeSeriesData,
        provenance: Option<&crate::analysis::execute::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let result = self
            .prepared
            .estimate_series_with_provenance(data, provenance, ctx)
            .map_err(|error| failed(&error))?;
        self.read_back(&result, true, ctx)
    }

    fn reestimate_gathered_series(
        &self,
        data: &TimeSeriesData,
        source_ticks: &[usize],
        block_length: Option<usize>,
        user_sampling_weights: bool,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let provenance = crate::analysis::execute::TemporalRowProvenance::admit_fixed_blocks(
            source_ticks,
            block_length,
            user_sampling_weights,
            budget,
            ctx,
        )
        .map_err(|error| failed(&error))?;
        let result = self
            .prepared
            .estimate_series_with_provenance(data, Some(&provenance), ctx)
            .map_err(|error| failed(&error))?;
        self.read_back(&result, true, ctx)
    }

    /// The refitted target vector, in the original coordinate order.
    #[allow(
        clippy::too_many_lines,
        reason = "source, contributor and coordinate checks share one bounded refit"
    )]
    fn read_back(
        &self,
        refitted: &StudyResult,
        same_prepared: bool,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let mut budget = self.temporal_budget.clone();
        if matches!(self.data, PreparedTargetData::Tabular(_)) && self.result.response().is_some() {
            match (&self.result.structural_response, &refitted.structural_response) {
                (Some(before), Some(after)) => {
                    // Graph keys are opaque, not globally collision-resistant. They
                    // identify masks only within this exact retained DAG posterior:
                    // construction rejects collisions between distinct adjacency masks.
                    let class_witnesses_match =
                        same_static_class_witnesses(self.result, refitted, &mut budget, ctx)?;
                    let program =
                        self.prepared.plan().lowered().and_then(LoweredProgram::semantic_identity);
                    let same_program = program.is_some()
                        && program == self.result.lowered_program_identity()
                        && program == refitted.lowered_program_identity();
                    let mut admg_source_matches = false;
                    let admg_match = if same_prepared && same_program {
                        if let Some(posterior) = self.study().graph_posterior.as_ref() {
                            admg_source_matches = same_admg_posterior_source(
                                posterior,
                                self.result.static_posterior_source.as_deref(),
                                refitted.static_posterior_source.as_deref(),
                            );
                            exact_admg_contributors(
                                posterior,
                                before,
                                after,
                                admg_source_matches,
                                &mut budget,
                                ctx,
                            )?
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    let posterior = self.study().graph_posterior.as_ref().filter(|posterior| {
                        same_prepared && (posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag
                            || admg_match
                            || (same_program && matches!(posterior.atom_kind(), antecedent_discovery::GraphPosteriorAtomKind::Cpdag | antecedent_discovery::GraphPosteriorAtomKind::Pag) && class_witnesses_match))
                    }).ok_or_else(|| unsupported(
                        "structural contributor identities cannot be verified for this refit",
                        "retain full completion identities for the refitted structure",
                    ))?;
                    let class_proven = same_prepared
                        && same_program
                        && class_witnesses_match
                        && matches!(
                            posterior.atom_kind(),
                            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                                | antecedent_discovery::GraphPosteriorAtomKind::Pag
                        );
                    let source_bound_ordered = admg_match && admg_source_matches;
                    if !source_bound_ordered {
                        let lookup_work = u64::try_from(before.atoms.len())
                            .ok()
                            .and_then(|n| {
                                u64::try_from(posterior.graph_keys().len())
                                    .ok()
                                    .and_then(|m| n.checked_mul(m))
                            })
                            .ok_or_else(|| {
                                failed(&CausalError::Resource {
                                    message: "static contributor lookup overflow".into(),
                                })
                            })?;
                        budget.charge_work(lookup_work).map_err(|error| {
                            failed(&CausalError::Resource { message: error.to_string() })
                        })?;
                        if !class_proven && before.atoms.len() != posterior.graph_keys().len() {
                            return Err(unsupported(
                                "structural contributor key is absent or ambiguous in the retained posterior",
                                "retain unique semantic graph identities",
                            ));
                        }
                        for atom in &before.atoms {
                            let mut matches = 0usize;
                            for key in posterior.graph_keys().iter() {
                                if ctx.cancellation.is_cancelled() {
                                    return Err(failed(&CausalError::Cancelled {
                                        stage: "static contributor lookup",
                                    }));
                                }
                                if *key == atom.graph_key {
                                    matches += 1;
                                }
                            }
                            if matches == 0 || (!class_proven && matches != 1) {
                                return Err(unsupported(
                                    "structural contributor key is absent or ambiguous in the retained posterior",
                                    "retain unique semantic graph identities",
                                ));
                            }
                        }
                    }
                    check_static_response_contributors(
                        before,
                        after,
                        source_bound_ordered,
                        &mut budget,
                        ctx,
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(unsupported(
                        "refit changed structural response availability",
                        "refit the same identified structural contributors",
                    ));
                }
            }
        }
        check_temporal_response_contributors(
            self.result,
            refitted,
            &mut budget,
            self.member.is_some(),
            ctx,
        )?;
        if let Some(selector) = &self.member {
            let values = temporal_response_member_values(refitted, selector, &mut budget, ctx)
                .map_err(|error| match error {
                    CausalError::Cancelled { .. } | CausalError::Resource { .. } => failed(&error),
                    _ => unsupported(error.to_string(), "refit the original completion"),
                })?;
            if values.len() != self.target.labels().len() {
                return Err(TargetRefitError::Failed(ValidationError::estimation_msg(
                    "the refitted completion reports a different target dimension",
                )));
            }
            return Ok(values);
        }
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
        study.settings.refute = crate::analysis::builder::RefuteSuite::None;
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
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior has no single structure to read a negative-control exposure \
                 from (its placeholder graph has no edges, so every column would read as \
                 unrelated to the treatment and the outcome)",
                "declare a negative-control exposure in TargetRefitOptions",
            ));
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
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior has no single structure to read a negative-control outcome \
                 from (its placeholder graph has no edges)",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
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
            let Some(w) = static_node(dag, v) else {
                continue;
            };
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
                let strata = self.change_populations(data.row_count());
                self.reestimate(
                    &Modal::Tabular(Self::bootstrap_within(data, rng, strata.as_deref())?),
                    ctx,
                )
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
        Self::bootstrap_within(data, rng, None)
    }

    /// The row positions of the baseline and the comparison population of a change attribution,
    /// when the study is one whose populations are row selections; `None` otherwise.
    fn change_populations(&self, rows: usize) -> Option<Vec<Vec<usize>>> {
        use antecedent_core::PopulationSelector;
        let CausalQuery::ChangeAttribution(query) = &self.study().query else {
            return None;
        };
        let positions = |selector: &PopulationSelector| -> Option<Vec<usize>> {
            match selector {
                PopulationSelector::All => Some((0..rows).collect()),
                PopulationSelector::Rows(rows_) => {
                    Some(rows_.iter().copied().filter(|&row| row < rows).collect())
                }
                PopulationSelector::TimeRange { start, end } => {
                    Some((*start..(*end).min(rows)).collect())
                }
                _ => None,
            }
        };
        Some(vec![positions(&query.baseline)?, positions(&query.comparison)?])
    }

    /// A row bootstrap that draws each position's source row from the stratum the position lies
    /// in (a population of a change attribution), so a resample never moves a row from one
    /// population into the other; positions outside every stratum keep their own row.
    fn bootstrap_within(
        data: &TabularData,
        rng: &mut CausalRng,
        strata: Option<&[Vec<usize>]>,
    ) -> Result<TabularData, TargetRefitError> {
        let storage = data.storage();
        if storage.analysis_mask().is_some() || storage.weights().is_some() {
            return Err(unsupported(
                "a row bootstrap of a masked or weighted table would change its population",
                "refute the unmasked, unweighted table, or use subset stability",
            ));
        }
        let n = data.row_count();
        let rows: Vec<usize> = match strata {
            None => (0..n).map(|_| unbiased_index(rng, n)).collect(),
            Some(strata) => {
                let mut rows: Vec<usize> = (0..n).collect();
                for stratum in strata.iter().filter(|stratum| !stratum.is_empty()) {
                    for &position in stratum {
                        rows[position] = stratum[unbiased_index(rng, stratum.len())];
                    }
                }
                rows
            }
        };
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
                // A circular window of design rows: the row ending at each period keeps its own
                // lag window, and every period is in a window with the same probability
                // (`width / n`), so the window estimates centre on the full-series estimate
                // instead of over-weighting the middle of the series.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n);
                let weights: Arc<[f64]> =
                    (0..n).map(|t| if (t + n - start) % n < width { 1.0 } else { 0.0 }).collect();
                series_with(series, &[], None, Some(weights))
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
    #[allow(
        clippy::type_complexity,
        reason = "the returned series retains its provenance and row selection"
    )]
    fn materialised_window(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<
        Option<(
            TimeSeriesData,
            Option<crate::analysis::execute::TemporalRowProvenance>,
            Option<JoinFreeRows>,
        )>,
        TargetRefitError,
    > {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
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
        let window_start = from.saturating_sub(span);
        let Some(window) = truncated_series(&base, window_start, to)? else {
            return Ok(None);
        };
        if replicate == 0 {
            let provenance = if self.target.family() == TargetFamily::Attribution {
                Some(
                    crate::analysis::execute::TemporalRowProvenance::admit_contiguous_range(
                        window_start,
                        window.row_count(),
                        series.storage().weights().is_some(),
                        &mut budget,
                        ctx,
                    )
                    .map_err(|error| failed(&error))?,
                )
            } else {
                None
            };
            return Ok(Some((window, provenance, None)));
        }
        let rows = window.row_count();
        if self.target.family() == TargetFamily::Attribution {
            // An attribution fits row-wise lag-aligned tables and applies the analysis mask, so
            // the circular block resample is gathered with its join-crossing rows hidden.
            let length = antecedent_data::circular_block_length(span, rows);
            let mut map = vec![0_usize; rows];
            antecedent_data::fill_circular_block_indexes(rows, length, rng, &mut map)
                .map_err(data_failed)?;
            let (gathered, guard) =
                self.gathered_rows(&window, &map, Some(length), &mut budget, ctx)?;
            budget
                .charge_work(u64::try_from(map.len()).map_err(|_| {
                    failed(&CausalError::Resource {
                        message: "refit coordinate work overflow".into(),
                    })
                })?)
                .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
            for tick in &mut map {
                if ctx.cancellation.is_cancelled() {
                    return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
                }
                *tick = tick.checked_add(window_start).ok_or_else(|| {
                    failed(&CausalError::Resource {
                        message: "refit source coordinate overflow".into(),
                    })
                })?;
            }
            let provenance = crate::analysis::execute::TemporalRowProvenance::admit_fixed_blocks(
                &map,
                Some(length),
                series.storage().weights().is_some(),
                &mut budget,
                ctx,
            )
            .map_err(|error| failed(&error))?;
            return Ok(Some((gathered, Some(provenance), Some(guard))));
        }
        // Any other route that reads neither row weights nor an analysis mask cannot skip the
        // design rows whose lag window crosses a join of a block resample, and a join pairs
        // values observed apart. Its replicate is a contiguous sub-window instead, which has no
        // join: a subsample of a dependent series (its spread is wider than the full series',
        // so the check is conservative where the window is short).
        let (start, width) = subsample_bounds(rows, span, rng);
        Ok(truncated_series(&window, start, start + width)?.map(|window| (window, None, None)))
    }

    /// The gathered series of the rows `map` (positions of `series`), with a design row whose lag
    /// window crosses a join of the map weighted zero and the sampling weight carried over.
    fn gathered_rows(
        &self,
        series: &TimeSeriesData,
        map: &[usize],
        block_length: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<(TimeSeriesData, JoinFreeRows), TargetRefitError> {
        let n = map.len();
        let span = max_graph_lag(self.study()) + 1;
        let time_index = match truncated_series(series, 0, n)? {
            Some(head) => head.time_index().clone(),
            None => series.time_index().clone(),
        };
        let storage = series.storage();
        let mut columns = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "a gathered window gathers float64 columns only",
                    "encode every column as float64",
                ));
            };
            let values: Arc<[f64]> = map.iter().map(|&row| column.values[row]).collect();
            let validity = column.validity.gather_rows(map).map_err(data_failed)?;
            columns.push(OwnedColumn::Float64(
                antecedent_data::Float64Column::new(column.id, values, validity)
                    .map_err(data_failed)?,
            ));
        }
        let block_boundaries =
            (self.target.family() == TargetFamily::Attribution).then_some(block_length).flatten();
        let weight_free = join_free_rows(map, span, block_boundaries, budget, ctx)?;
        let mask = if self.target.family() == TargetFamily::Attribution {
            Some(join_free_mask(storage, map, span, block_length, budget, ctx)?)
        } else {
            storage
                .analysis_mask()
                .map(|mask| mask.gather_rows(map))
                .transpose()
                .map_err(data_failed)?
        };
        budget
            .charge_work(u64::try_from(n).map_err(|_| {
                failed(&CausalError::Resource { message: "refit weight work overflow".into() })
            })?)
            .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(|_| {
            failed(&CausalError::Resource { message: "refit weight allocation refused".into() })
        })?;
        for (row, &source) in map.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
            }
            let sampling = storage.weights().map_or(1.0, |weights| weights[source]);
            values.push(if weight_free[row] { sampling } else { 0.0 });
        }
        let weights: Arc<[f64]> = values.into();
        let gathered = antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            mask,
            Some(weights),
        )
        .and_then(|gathered| TimeSeriesData::try_new(gathered, time_index))
        .map_err(data_failed)?;
        Ok((gathered, weight_free))
    }

    /// The bootstrap of a series drawn as a gathered circular-block series, with a zero row
    /// weight on every design row whose lag window crosses a block join (such a row pairs values
    /// observed apart and would attenuate a lagged effect). A route that reads row weights
    /// recovers the replicate the multiplicities define. An attribution reads none, so those
    /// rows are hidden by the analysis mask, which it applies by deleting them before fitting
    /// and scoring; any other route that reads neither keeps the attenuation, of order
    /// `(span - 1) / block length`.
    fn gathered_bootstrap(
        &self,
        series: &TimeSeriesData,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
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
        let block_boundaries =
            (self.target.family() == TargetFamily::Attribution).then_some(length);
        let weight_free = join_free_rows(&map, span, block_boundaries, &mut budget, ctx)?;
        let mask = if self.target.family() == TargetFamily::Attribution {
            Some(join_free_mask(storage, &map, span, Some(length), &mut budget, ctx)?)
        } else {
            storage
                .analysis_mask()
                .map(|mask| mask.gather_rows(&map))
                .transpose()
                .map_err(data_failed)?
        };
        // A design row whose lag window crosses a block join pairs values observed apart, which
        // attenuates a lagged effect by about the share of such rows. Their weight is zero, so
        // a route that reads row weights recovers the replicate the multiplicities define; a
        // route that reads none keeps the attenuation, of order `(span - 1) / length`. A
        // sampling weight of the original sample is carried to the row it was gathered from, so
        // the replicate is the block resample of the weighted sample.
        budget
            .charge_work(u64::try_from(n).map_err(|_| {
                failed(&CausalError::Resource { message: "refit weight work overflow".into() })
            })?)
            .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(|_| {
            failed(&CausalError::Resource { message: "refit weight allocation refused".into() })
        })?;
        for (row, &source) in map.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
            }
            let sampling = storage.weights().map_or(1.0, |weights| weights[source]);
            values.push(if weight_free[row] { sampling } else { 0.0 });
        }
        let weights: Arc<[f64]> = values.into();
        let gathered = antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            mask,
            Some(weights),
        )
        .and_then(|gathered| TimeSeriesData::try_new(gathered, series.time_index().clone()))
        .map_err(data_failed)?;
        if self.target.family() == TargetFamily::Attribution {
            self.reestimate_gathered_series(
                &gathered,
                &map,
                Some(length),
                storage.weights().is_some(),
                &mut budget,
                ctx,
            )
        } else {
            self.reestimate(&Modal::Series(gathered), ctx)
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the target-specific branches preserve a single checked operation"
    )]
    fn series_refit(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        // An attribution fits its mechanisms on the rows it is given and reads no row weights,
        // so its block resample must be drawn as a gathered series, not as multiplicities.
        if (self.options.gathered_resamples || self.target.family() == TargetFamily::Attribution)
            && !matches!(perturbation, TargetPerturbation::RandomCommonCause { .. })
        {
            if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) {
                return self.gathered_bootstrap(series, rng, ctx);
            }
            // Negative controls are substituted before drawing the same
            // circular block map. Gathering preserves the sampled multiplicities
            // and their source/block provenance without treating generated
            // bookkeeping weights as user-declared weights.
            let replacement = match perturbation {
                TargetPerturbation::NegativeControlExposure { replicate } if replicate > 0 => {
                    let control = self.control_exposure()?;
                    Some((self.treatment, Arc::from(Self::column_of(series.storage(), control)?)))
                }
                TargetPerturbation::NegativeControlOutcome { replicate } if replicate > 0 => self
                    .options
                    .negative_control_outcome
                    .map(|control| {
                        Self::column_of(series.storage(), control)
                            .map(|values| (self.outcome, Arc::from(values)))
                    })
                    .transpose()?,
                _ => None,
            };
            if let Some(replacement) = replacement {
                let substituted = series_with(series, &[replacement], None, None)?;
                return self.gathered_bootstrap(&substituted, rng, ctx);
            }
            if let Some((window, provenance, _guard)) =
                self.materialised_window(series, perturbation, rng, ctx)?
            {
                return self.reestimate_materialised_series(&window, provenance.as_ref(), ctx);
            }
        }
        if matches!(perturbation, TargetPerturbation::RandomCommonCause { .. }) {
            return self.random_common_cause(rng, ctx);
        }
        // The window of an attribution subset is drawn circularly: a window that must lie inside
        // the series includes the first and last periods far less often than the middle ones,
        // so the replicate mean of a statistic that depends on which periods are present
        // (a planted outlier at the edge) is not the original even for a clean series. A
        // circular window includes every period with the same probability.
        if let (TargetPerturbation::Subset { fraction, .. }, TargetFamily::Attribution) =
            (perturbation, self.target.family())
        {
            let n = series.row_count();
            if n < 2 {
                return Err(unsupported("a window needs at least two periods", "more data"));
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
            )]
            let width = ((fraction * n as f64).round() as usize).clamp(1, n);
            let start = unbiased_index(rng, n);
            let map: Vec<usize> = (0..width).map(|i| (start + i) % n).collect();
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            let (gathered, _guard) = self.gathered_rows(series, &map, None, &mut budget, ctx)?;
            return self.reestimate_gathered_series(
                &gathered,
                &map,
                None,
                series.storage().weights().is_some(),
                &mut budget,
                ctx,
            );
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
        if refit.as_ref().err().is_some_and(operational_refit_error) {
            return refit;
        }
        // A route that reads no row weights (a conditional effect) cannot take the circular
        // window as zero weights; the same subset is then cut as a contiguous window, which keeps
        // every lag the procedure reads inside it.
        if refit.is_err() {
            if let TargetPerturbation::Subset { fraction, .. } = perturbation {
                let n = series.row_count();
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n - width + 1);
                if let Some(window) = truncated_series(series, start, start + width)? {
                    match self.reestimate(&Modal::Series(window), ctx) {
                        Ok(values) => return Ok(values),
                        Err(error) if operational_refit_error(&error) => return Err(error),
                        Err(_) => {}
                    }
                }
            }
        }
        // Routes that do not consume row weights need a gathered circular-block series.
        // Attribution refits retain the block boundaries so complete windows do not cross joins.
        if refit.is_err() {
            if let Some((window, provenance, _guard)) =
                self.materialised_window(series, perturbation, rng, ctx)?
            {
                match self.reestimate_materialised_series(&window, provenance.as_ref(), ctx) {
                    Ok(values) => return Ok(values),
                    Err(error) if operational_refit_error(&error) => return Err(error),
                    Err(_) => {}
                }
            }
        }
        if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) && refit.is_err() {
            match self.gathered_bootstrap(series, rng, ctx) {
                Ok(values) => return Ok(values),
                Err(error) if operational_refit_error(&error) => return Err(error),
                Err(_) => {}
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

    /// Shift a panel column and retain the complete window of each unit.
    fn shifted_panel_units(
        units: &[PanelUnit],
        id: VariableId,
        offset: isize,
    ) -> Result<Vec<PanelUnit>, TargetRefitError> {
        units
            .iter()
            .map(|unit| {
                let (values, _) = shifted(Self::column_of(unit.series.storage(), id)?, offset);
                let rows = unit.series.row_count();
                let trim = offset.unsigned_abs();
                let (start, end) =
                    if offset < 0 { (trim, rows) } else { (0, rows.saturating_sub(trim)) };
                Ok(PanelUnit {
                    unit_id: unit.unit_id,
                    series: Self::cut(
                        &series_with(&unit.series, &[(id, values)], None, None)?,
                        start,
                        end,
                    )?,
                })
            })
            .collect()
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
        let rebuild_replicate = |rng: &mut CausalRng, units: Vec<PanelUnit>, replicate: u32| {
            if replicate == 0 { rebuild(units) } else { rebuild(resample_units(rng, &units)) }
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
                rebuild_replicate(rng, swapped, replicate)?
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let swapped = if let Some(control) = self.options.negative_control_outcome {
                    swap_all(self.outcome, control)?
                } else {
                    let span =
                        isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                    Self::shifted_panel_units(units, self.outcome, -span)?
                };
                rebuild_replicate(rng, swapped, replicate)?
            }
            TargetPerturbation::PreTrend { replicate, lead } => {
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let shifted_units = Self::shifted_panel_units(units, self.treatment, lead)?;
                rebuild_replicate(rng, shifted_units, replicate)?
            }
            TargetPerturbation::Anticipation { replicate } => {
                let back = isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                let shifted_units = Self::shifted_panel_units(units, self.outcome, -back)?;
                rebuild_replicate(rng, shifted_units, replicate)?
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
            // Posterior columns may put nuisance coefficients before the effect.
            // Read the declared effect, never the first column by position.
            let effect = posterior.effect_column()?;
            return posterior.draws.column(effect).ok().map(<[f64]>::to_vec);
        }
        None
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
            standardised_bias: Arc::from([]),
            standardised_bias_norm: None,
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
        if self.result.temporal_posterior_witnesses.is_some()
            && matches!(perturbation, TargetPerturbation::RandomCommonCause { .. })
        {
            return Err(unsupported(
                "the added common cause changes completion identity",
                "a scoped projection that maps the augmented graph to the original completion",
            ));
        }
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

#[cfg(test)]
mod panel_shift_tests {
    use super::*;

    #[test]
    fn shifted_panel_windows_keep_alignment_and_refuse_empty_units() {
        let make_unit = |id, values: &[f64], other: &[f64]| PanelUnit {
            unit_id: id,
            series: TimeSeriesData::from_f64_columns([("t", values), ("y", other)], 1).unwrap(),
        };
        let units = vec![
            make_unit(7, &[1., 2., 3., 4.], &[10., 20., 30., 40.]),
            make_unit(9, &[5., 6., 7.], &[50., 60., 70.]),
        ];
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let forward = PreparedRefit::shifted_panel_units(&units, treatment, 1).unwrap();
        assert_eq!(forward.iter().map(|u| u.unit_id).collect::<Vec<_>>(), [7, 9]);
        assert_eq!(float_column(forward[0].series.storage(), treatment).unwrap(), [2., 3., 4.]);
        assert_eq!(float_column(forward[0].series.storage(), outcome).unwrap(), [10., 20., 30.]);
        assert_eq!(float_column(forward[1].series.storage(), treatment).unwrap(), [6., 7.]);
        let backward = PreparedRefit::shifted_panel_units(&units, outcome, -1).unwrap();
        assert_eq!(float_column(backward[0].series.storage(), treatment).unwrap(), [2., 3., 4.]);
        assert_eq!(float_column(backward[0].series.storage(), outcome).unwrap(), [10., 20., 30.]);
        assert_eq!(float_column(backward[1].series.storage(), outcome).unwrap(), [50., 60.]);
        assert!(PreparedRefit::shifted_panel_units(&units, outcome, -3).is_err());
        assert!(PreparedRefit::shifted_panel_units(&units, treatment, 3).is_err());
    }
}

#[cfg(test)]
mod temporal_contributor_order_tests {
    use super::*;
    use crate::{RefuteSuite, Study};
    use antecedent_core::{
        CausalQuery, Intervention, ResponseFunctional, ResponseQuery, TemporalPolicy,
        TemporalResponseSpec, Value, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_discovery::adjacency_mask::set_edge;
    use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
    use antecedent_prob::InferenceDiagnostics;
    use std::sync::Arc;

    fn result(ctx: &ExecutionContext) -> StudyResult {
        let n = 300;
        let (mut t, mut y, mut z) = (vec![0.; n], vec![0.; n], vec![0.; n]);
        for i in 0..n {
            let s = i as f64;
            let r = (s * 0.29).sin();
            z[i] = 0.4 * r + (s * 0.13).cos() + 0.3 * (s * 1.7).sin();
            t[i] = 0.3 + 0.5 * r + 0.2 * z[i] + 0.4 * (s * 2.3).cos();
            if i > 0 {
                y[i] = 1. + 2. * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (s * 0.77).sin();
            }
        }
        let data = TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let adj = set_edge(0, 3, 2, 0, true).to_word().unwrap();
        let posterior = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.5],
            vec![adj, 0],
            vec![0.; 9],
            vec![0.; 9],
            SamplingSummary {
                ess: 1.,
                diagnostics: InferenceDiagnostics::analytic("contributor_order"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
        .with_lagged_marginals(1, vec![0.; 9])
        .unwrap()
        .with_lag_masks(vec![(1 << 1) | (1 << 7), 1 << 1])
        .unwrap();
        let query = CausalQuery::Response(
            ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(1),
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(1.),
                )]),
            })
            .with_temporal(
                TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
            ),
        );
        Study::series(data)
            .graph_posterior(posterior)
            .query(query)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(ctx)
            .unwrap()
    }

    #[allow(
        clippy::used_underscore_binding,
        reason = "the copied member must retain its graph admission leases"
    )]
    fn copied_member(
        m: &crate::result::TemporalPosteriorWitnessMember,
    ) -> crate::result::TemporalPosteriorWitnessMember {
        crate::result::TemporalPosteriorWitnessMember {
            outer_atom: m.outer_atom,
            outer_weight: m.outer_weight,
            witnesses: Arc::clone(&m.witnesses),
            structural: m.structural.clone(),
            _resources: m._resources.clone(),
        }
    }

    #[test]
    fn contributor_matching_is_keyed_not_enumeration_order() {
        let ctx = ExecutionContext::for_tests(1);
        let original = result(&ctx);
        let source = original.temporal_posterior_witnesses.as_ref().unwrap();
        assert!(source.members.len() >= 2);
        let mut reordered = original.clone();
        let mut members: Vec<_> = source.members.iter().map(copied_member).collect();
        members.reverse();
        for member in &mut members {
            let mut entries = member.witnesses.entries.clone();
            entries.reverse();
            member.witnesses = Arc::new(crate::result::TemporalResponseWitnessSet {
                entries,
                _leases: Vec::new(),
            });
        }
        reordered.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members,
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        let compare = |other: &StudyResult| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_temporal_response_contributors(&original, other, &mut budget, true, &ctx)
        };
        assert!(compare(&reordered).is_ok());
        let mut missing = original.clone();
        missing.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members: source.members.iter().skip(1).map(copied_member).collect(),
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        assert!(compare(&missing).is_err());
        let mut changed = original.clone();
        let mut members: Vec<_> = source.members.iter().map(copied_member).collect();
        let mut entries = members[0].witnesses.entries.clone();
        assert!(!entries.is_empty());
        entries[0].completion_digest[0] ^= 1;
        members[0].witnesses =
            Arc::new(crate::result::TemporalResponseWitnessSet { entries, _leases: Vec::new() });
        changed.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members,
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        assert!(compare(&changed).is_err());
    }
}

#[cfg(test)]
mod static_contributor_tests {
    use super::*;
    use crate::result::{
        StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
        StructuralWeightBasis,
    };
    use antecedent_core::IdentificationStatus;

    fn mixture() -> StructuralResponseMixture {
        let atom = |graph_key, weight, value| StructuralResponseAtom {
            graph_key,
            weight,
            status: IdentificationStatus::NonparametricallyIdentified,
            value: Some(value),
            posterior: None,
            response: None,
        };
        StructuralResponseMixture {
            weight_basis: StructuralWeightBasis::PosteriorProbability,
            atoms: vec![
                atom(7, 0.4, ResponseValue::Scalar(1.0)),
                atom(9, 0.6, ResponseValue::Vector(Arc::from([2.0, 3.0]))),
            ],
            identified_mass: 1.0,
            unidentified_mass: 0.0,
            unevaluable_mass: 0.0,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: None,
            conditional_on_identified: Some(ResponseValue::Scalar(2.0)),
            full_mass_scope: true,
            truncated_atoms: 0,
            aggregation_policy: Some(StructuralAggregationPolicy::SameEstimandWeightedMean),
        }
    }

    #[test]
    fn wide_admg_sources_require_the_exact_immutable_posterior_allocation() {
        use antecedent_discovery::{
            AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
        };
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::from_atoms(
            9,
            vec![0.4, 0.6],
            vec![AdjacencyMask::from_words(&[7, 1]), AdjacencyMask::from_words(&[9, 1])],
            vec![0.0; 81],
            vec![0.0; 81],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("wide_admg_source"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let witness = |source: &Arc<[AdjacencyMask]>| crate::result::StaticPosteriorSourceWitness {
            source: Arc::downgrade(source),
            _lease: ctx.acquire_admission(0, 1024).unwrap(),
        };
        let first = witness(posterior.atoms());
        let second = witness(posterior.atoms());
        assert!(same_admg_posterior_source(&posterior, Some(&first), Some(&second)));
        let foreign: Arc<[AdjacencyMask]> = Arc::from(posterior.atoms().to_vec());
        let foreign_witness = witness(&foreign);
        assert!(!same_admg_posterior_source(&posterior, Some(&first), Some(&foreign_witness)));
        assert!(!same_admg_posterior_source(&posterior, Some(&first), None));
        let mut original = mixture();
        for (atom, key) in original.atoms.iter_mut().zip(posterior.graph_keys().iter()) {
            atom.graph_key = *key;
        }
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            exact_admg_contributors(&posterior, &original, &original, true, &mut budget, &ctx)
                .unwrap()
        );
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            !exact_admg_contributors(&posterior, &original, &original, false, &mut budget, &ctx)
                .unwrap()
        );
    }

    #[test]
    fn unbound_small_admg_refuses_oversized_mask_storage() {
        use antecedent_discovery::{
            AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
        };
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::from_atoms(
            3,
            vec![0.4, 0.6],
            vec![AdjacencyMask::from_words(&[0, 0, 0, 0, 1]), AdjacencyMask::from_word(9)],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("oversized_admg_mask"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let mut original = mixture();
        for (atom, key) in original.atoms.iter_mut().zip(posterior.graph_keys().iter()) {
            atom.graph_key = *key;
        }
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            !exact_admg_contributors(&posterior, &original, &original, false, &mut budget, &ctx)
                .unwrap()
        );
    }

    #[test]
    fn admg_contributors_bind_every_ordered_word_weight_and_disposition() {
        use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::new(
            3,
            vec![0.4, 0.6],
            vec![7, 9],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic(
                    "exact_admg_contributors",
                ),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let mut original = mixture();
        original.atoms[1].status = IdentificationStatus::NotIdentified;
        let check = |candidate: &StructuralResponseMixture| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            exact_admg_contributors(&posterior, &original, candidate, false, &mut budget, &ctx)
                .unwrap()
        };
        assert!(check(&original));
        let mut changed = original.clone();
        changed.atoms[1].graph_key = 10;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms[1].weight = 0.5;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms[1].status = IdentificationStatus::NonparametricallyIdentified;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms.reverse();
        assert!(!check(&changed));
    }

    #[test]
    fn source_bound_ordered_samples_allow_repeats_but_preserve_each_sample() {
        let ctx = ExecutionContext::for_tests(1);
        let mut original = mixture();
        original.atoms[1].graph_key = original.atoms[0].graph_key;
        let check = |candidate: &StructuralResponseMixture, ordered| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_static_response_contributors(&original, candidate, ordered, &mut budget, &ctx)
        };
        assert!(check(&original, true).is_ok());
        assert!(check(&original, false).is_err());
        let mut changed = original.clone();
        changed.atoms.swap(0, 1);
        assert!(check(&changed, true).is_err());
        changed = original.clone();
        changed.atoms[1].status = IdentificationStatus::NotIdentified;
        assert!(check(&changed, true).is_err());
        changed = original.clone();
        changed.atoms[1].value = None;
        assert!(check(&changed, true).is_err());
    }

    #[test]
    fn static_contributors_require_unique_stable_identities_mass_and_coordinates() {
        let ctx = ExecutionContext::for_tests(1);
        let original = mixture();
        let check = |candidate: &StructuralResponseMixture| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_static_response_contributors(&original, candidate, false, &mut budget, &ctx)
        };
        let mut reordered = original.clone();
        reordered.atoms.reverse();
        reordered.atoms[0].value = Some(ResponseValue::Vector(Arc::from([4.0, 5.0])));
        assert!(check(&reordered).is_ok());
        let mut missing = original.clone();
        missing.atoms.pop();
        assert!(check(&missing).is_err());
        let mut changed_key = original.clone();
        changed_key.atoms[0].graph_key = 8;
        assert!(check(&changed_key).is_err());
        let mut ambiguous = original.clone();
        ambiguous.atoms[1].graph_key = ambiguous.atoms[0].graph_key;
        assert!(check(&ambiguous).is_err());
        let mut changed_weight = original.clone();
        changed_weight.atoms[0].weight = 0.5;
        assert!(check(&changed_weight).is_err());
        let mut changed_mass = original.clone();
        changed_mass.unidentified_mass = 0.1;
        assert!(check(&changed_mass).is_err());
        let mut changed_coordinates = original.clone();
        changed_coordinates.atoms[1].value = Some(ResponseValue::Vector(Arc::from([4.0])));
        assert!(check(&changed_coordinates).is_err());
        let mut changed_status = original.clone();
        changed_status.atoms[0].status = IdentificationStatus::NotIdentified;
        assert!(check(&changed_status).is_err());
    }
    #[test]
    fn class_witness_refuses_changed_source_completion_and_mass() {
        let ctx = ExecutionContext::for_tests(1);
        let cache: Arc<dyn std::fmt::Debug + Send + Sync + std::panic::RefUnwindSafe> =
            Arc::new(());
        let atoms: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([1_u64.into()]);
        let marks: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([0_u64.into()]);
        let make = || crate::result::StaticClassWitnessSet {
            atom_kind: antecedent_discovery::GraphPosteriorAtomKind::Pag,
            n_vars: 3,
            identification_cache: Arc::downgrade(&cache),
            source_atoms: Arc::downgrade(&atoms),
            source_marks: Some(Arc::downgrade(&marks)),
            samples: vec![(1, 1.0_f64.to_bits())],
            atoms: vec![crate::result::StaticClassAtomWitness {
                graph_key: 1,
                outer_weight: 1.0_f64.to_bits(),
                completions: crate::result::StaticFittedCompletions {
                    entries: vec![(
                        [1; 32],
                        1.0_f64.to_bits(),
                        IdentificationStatus::NonparametricallyIdentified,
                    )],
                    _lease: ctx.acquire_admission(0, 256).unwrap(),
                },
            }],
            _lease: ctx.acquire_admission(0, 256).unwrap(),
        };
        let original = make();
        let mut candidate = make();
        let compare = |candidate: &crate::result::StaticClassWitnessSet| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            compare_static_class_witnesses(&original, candidate, &mut budget, &ctx).unwrap()
        };
        assert!(compare(&candidate));
        candidate.atoms[0].completions.entries[0].0 = [2; 32];
        assert!(!compare(&candidate));
        candidate.atoms[0].completions.entries[0].0 = [1; 32];
        candidate.atoms[0].outer_weight = 0.5_f64.to_bits();
        assert!(!compare(&candidate));
        candidate.atoms[0].outer_weight = 1.0_f64.to_bits();
        let foreign: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([1_u64.into()]);
        candidate.source_atoms = Arc::downgrade(&foreign);
        assert!(!compare(&candidate));
    }
}
