//! Identification of the joint panel response under a declared Gaussian restriction.
//!
//! General ID on a declared unrolled graph identifies the joint intervention as a
//! nonparametric functional of the observed law. The fitted panel realizes that functional only
//! if the observed conditional laws are the panel mechanism's own: a mean that is a saturated
//! lag-weighted sum of the channels (plus declared products, a covariate and a season effect)
//! and Gaussian group intercepts and stationary AR(1) errors that are independent of the
//! program given the context. That is a parametric restriction, and it is claimed here only
//! when the caller declares it on the target. The claim then carries the restriction, the lag
//! structure it was checked against, and the outcome of residual normality and linearity
//! checks on the fitted mechanism. It never states nonparametric identification.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    JointPanelDesignAttempt, JointPanelDesignOutcome, JointPanelError, JointPanelFit,
    JointPanelState, JointPanelWorldSupport, contracts,
};
use crate::PreparedStudy;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    ModelAssumptions, ParametricAssumption, ScientificTarget, TargetForm,
};
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::panel_unrolled::UnrolledPanelGraph;
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, IdentificationResult, IdentificationStatus, UnresolvedObligation,
};
use std::sync::Arc;

/// Stable id of the declared restriction.
pub(super) const GAUSSIAN_RESTRICTION: &str = "panel.gaussian_ar_mechanism";
/// Attempt id carried by an identification that rests on the restriction.
pub(super) const RESTRICTION_ATTEMPT: &str = "joint_adjustment.gaussian_restriction";
/// Family-wise level of the residual checks.
const CHECK_LEVEL: f64 = 0.01;

/// Declare that the fitted panel's conditional laws are the panel mechanism's own: a mean that
/// is a saturated lag-weighted sum of the channels, declared products, the covariate and the
/// season effect, plus Gaussian group intercepts and stationary Gaussian AR(1) errors that are
/// independent of the program given the context. This is a user assumption about the data, not
/// a test. It is consumed by `with_unrolled_graph_attempt`, which still requires that general ID
/// identify the joint intervention on the declared unrolled graph.
#[must_use]
pub fn declare_joint_panel_gaussian_restriction(target: &ScientificTarget) -> ScientificTarget {
    let mut records = target.assumptions().records().clone();
    records.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(GAUSSIAN_RESTRICTION),
            description: Arc::from(
                "Conditional on the observed context, the outcome is a saturated lag-weighted sum of the channels (and declared products) plus a covariate and season effect, with Gaussian group intercepts and stationary Gaussian AR(1) errors independent of the whole program: linear-Gaussian conditional laws and no unmeasured group-level or serial cause of assignment",
            ),
        }),
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    ScientificTarget::new(
        target.form().clone(),
        ModelAssumptions::new(target.assumptions().graph(), records),
        target.inference(),
    )
}

fn declared(target: &ScientificTarget) -> bool {
    target.assumptions().records().entries.iter().any(|record| {
        matches!(&record.assumption,
            Assumption::ParametricRestriction(premise) if premise.id.as_ref() == GAUSSIAN_RESTRICTION)
            && record.scope == AssumptionScope::Identification
            && record.status != AssumptionStatus::Contradicted
    })
}

impl PreparedStudy<JointPanelState> {
    /// Whether the target declares the Gaussian restriction.
    pub(super) fn gaussian_restriction_declared(&self) -> bool {
        declared(&self.request().target)
    }

    /// The lag structure the declared restriction is stated for must be the one the unrolled
    /// graph names, and the graph must not make the program respond to the outcome history.
    pub(super) fn check_restriction_structure(
        &self,
        graph: &UnrolledPanelGraph,
    ) -> Result<(), JointPanelError> {
        let spec = graph.spec();
        let lag_order = self.request().lag_order;
        if self.request().temporal.is_some() {
            return Err(JointPanelError::Unsupported(
                "the Gaussian restriction is checked on the invariant fitted mechanism; a selected time-varying mechanism has no residual check, so its declaration is refused".into(),
            ));
        }
        if spec.carryover_reach != lag_order {
            return Err(JointPanelError::Unsupported(format!(
                "the declared unrolled graph reaches {} periods of carryover but the fitted mechanism has lag order {lag_order}; the restriction is stated for the fitted lag structure",
                spec.carryover_reach
            )));
        }
        if spec.history_feedback {
            return Err(JointPanelError::Unsupported(
                "the fitted mechanism treats the program as independent of the outcome history and errors; time-varying confounding by past outcomes is not realized by it, so the restriction cannot identify this graph".into(),
            ));
        }
        if spec.outcome_persistence != (lag_order > 0) {
            return Err(JointPanelError::Unsupported(
                "serial outcome dependence in the unrolled graph must match the fitted AR(1) errors (present only with carryover)".into(),
            ));
        }
        Ok(())
    }

    /// Retain the restricted identification of the joint response.
    pub(super) fn with_gaussian_restricted_identification(
        &self,
        result: &IdentificationResult,
        graph: &UnrolledPanelGraph,
        outcome_periods: &[usize],
    ) -> Result<Self, JointPanelError> {
        let target = self.request().target.clone();
        let worlds = contracts::target_worlds(&target)?;
        let mut retained = result.clone();
        retained.status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
        retained.required_assumptions.extend_unique(&target.assumptions().records().entries);
        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        parts.world_obligations = worlds
            .iter()
            .map(|world| {
                IdentificationObligation::for_world(
                    world.id(),
                    IdentificationOutcome::Identified,
                    IdentificationClaimClass::IdentifiedClaim,
                    vec![],
                    None,
                )
            })
            .collect();
        if matches!(target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
            parts.assembled = Some(IdentificationObligation::assembled(
                IdentificationOutcome::Identified,
                IdentificationClaimClass::IdentifiedClaim,
                vec![],
                None,
            ));
        }
        let identification = CompiledIdentificationProduct::retain_for_target(&retained, parts, &target)
            .map_err(JointPanelError::compile)?;
        let spec = graph.spec();
        let mut next = self.clone();
        next.state.identification = Arc::new(identification);
        let mut attempts = self.state.attempts.to_vec();
        let at = attempts
            .iter()
            .position(|attempt| attempt.id == "joint_adjustment.observational")
            .map_or(attempts.len(), |index| index + 1);
        attempts.insert(at, JointPanelDesignAttempt {
            id: RESTRICTION_ATTEMPT.into(),
            outcome: JointPanelDesignOutcome::Identified,
            detail: format!(
                "identified under the declared Gaussian restriction ({GAUSSIAN_RESTRICTION}), not nonparametrically: general ID identifies the joint intervention on the declared unrolled graph ({} channel nodes over {} periods, outcome periods {outcome_periods:?}), and the fitted mechanism (lag order {}) realizes that functional only if the observed conditional laws are linear-Gaussian in the saturated lag-weighted channels with Gaussian group effects and AR(1) errors independent of the program; residual normality and linearity checks at family-wise level {CHECK_LEVEL} run on the fitted mechanism and withdraw the claim if they flag it",
                spec.channels * spec.periods,
                spec.periods,
                self.request().lag_order,
            ),
            next_attempt: None,
            target_identity: target.identity(),
            worlds: self
                .request()
                .worlds
                .iter()
                .map(|world| JointPanelWorldSupport {
                    world: world.world,
                    matched_groups: self
                        .model_data()
                        .groups
                        .iter()
                        .map(|group| group.id.clone())
                        .collect(),
                    identified: true,
                    reason: "general ID on the declared unrolled graph under the declared Gaussian restriction".into(),
                })
                .collect(),
            proof: None,
        });
        next.state.attempts = attempts.into();
        Ok(next)
    }

    /// Check the declared restriction against the fitted mechanism. A result whose claim rests
    /// on the restriction keeps it when the checks pass; a flagged check withdraws it.
    pub(super) fn checked_against_fit(&self, fit: &JointPanelFit) -> Result<Self, JointPanelError> {
        let Some(position) = self.state.attempts.iter().position(|attempt| {
            attempt.id == RESTRICTION_ATTEMPT && attempt.outcome == JointPanelDesignOutcome::Identified
        }) else {
            return Ok(self.clone());
        };
        let JointPanelFit::Invariant(invariant) = fit else {
            return self.withdrawn(
                position,
                "the fitted mechanism is time-varying and has no residual check".into(),
            );
        };
        let diagnostics = antecedent_estimate::joint_panel::restriction::panel_restriction_diagnostics(
            &self.state.data,
            &invariant.point,
            CHECK_LEVEL,
        )?;
        let summary = {
            let normality = diagnostics.normality.as_ref().map_or_else(
                || "normality not reported (too few innovations)".to_owned(),
                |check| {
                    format!(
                        "innovation normality Jarque-Bera {:.3} (p {:.4}, n {})",
                        check.jarque_bera, check.p_value, check.count
                    )
                },
            );
            let linearity = diagnostics
                .linearity
                .iter()
                .map(|check| format!("{} lack of fit p {:.4}", check.feature, check.p_value))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{normality}; {linearity}")
        };
        if diagnostics.flagged() {
            let mut reasons = Vec::new();
            if diagnostics.normality_flagged() {
                reasons.push("the whitened innovations are not Gaussian".to_owned());
            }
            for feature in diagnostics.linearity_flags() {
                reasons.push(format!("the mean is not represented by the fitted mechanism in {feature}"));
            }
            return self.withdrawn(position, format!("{}; {summary}", reasons.join("; ")));
        }
        let mut next = self.clone();
        let mut attempts = self.state.attempts.to_vec();
        attempts[position].detail = format!(
            "{} Checks on the fitted mechanism did not flag the restriction at family-wise level {CHECK_LEVEL} ({summary}); passing is not proof that it holds.",
            attempts[position].detail
        );
        next.state.attempts = attempts.into();
        Ok(next)
    }

    fn withdrawn(&self, position: usize, reason: String) -> Result<Self, JointPanelError> {
        let target = self.request().target.clone();
        let outcome = IdentificationOutcome::Unsupported;
        let class = IdentificationClaimClass::ModelConditional;
        let unresolved = vec![UnresolvedObligation::new(
            "gaussian_restriction_checks",
            format!("the declared Gaussian restriction is contradicted by the fitted panel: {reason}"),
        )];
        let mut parts = CompiledProductParts::new(outcome, class);
        parts.world_obligations = contracts::target_worlds(&target)?
            .iter()
            .map(|world| {
                IdentificationObligation::for_world(world.id(), outcome, class, unresolved.clone(), None)
            })
            .collect();
        parts.unresolved_obligations.clone_from(&unresolved);
        if matches!(target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
            parts.assembled = Some(IdentificationObligation::assembled(
                outcome,
                class,
                unresolved,
                None,
            ));
        }
        let identification = CompiledIdentificationProduct::new(
            vec![],
            antecedent_expr::CausalExprArena::new(),
            antecedent_identify::DerivationTrace::default(),
            target.assumptions().records().clone(),
            vec![],
            parts,
        )
        .and_then(|product| product.bind_target(&target))
        .map_err(JointPanelError::compile)?;
        let mut next = self.clone();
        next.state.identification = Arc::new(identification);
        let mut attempts = self.state.attempts.to_vec();
        attempts[position].outcome = JointPanelDesignOutcome::Rejected;
        attempts[position].detail = format!(
            "the declared Gaussian restriction is contradicted by the fitted panel: {reason}"
        );
        attempts[position].next_attempt = Some(
            "a mechanism family or declared products that represent the observed mean, or a whole-program randomized assignment".into(),
        );
        for support in &mut attempts[position].worlds {
            support.identified = false;
        }
        next.state.attempts = attempts.into();
        Ok(next)
    }
}
