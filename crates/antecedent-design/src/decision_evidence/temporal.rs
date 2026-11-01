//! Explicit assumptions connecting distinct temporal causal estimands.
//!
//! An excursion is not a whole-regime effect, a fixed schedule or a model response. A bridge
//! supplies a bounded effect relationship; it does not turn an assumption into a finding.
use super::{
    DecisionFrame, EvidenceRefusal, JointExperimentEvidence, ProxyBridge, ProxyDecision,
    ProxyMapping, Request, Target, decide_with_proxy,
};

/// Scientific temporal target, independent of the estimator used to study it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemporalEstimand {
    /// One randomized decision and the response at a later lag, with other decisions following the law.
    Excursion {
        /// Lag from decision to outcome.
        lag: u32,
        /// Sequential assignment-law identity.
        assignment_law_id: String,
    },
    /// A complete fixed intervention schedule over the declared horizon.
    FixedSchedule {
        /// Active schedule identity.
        active_schedule_id: String,
        /// Reference schedule identity.
        reference_schedule_id: String,
        /// Number of periods.
        horizon: u32,
    },
    /// A contrast between two static treatment regimes.
    StaticRegime {
        /// Active regime identity.
        active_regime_id: String,
        /// Reference regime identity.
        reference_regime_id: String,
        /// Number of decisions.
        horizon: u32,
        /// Sequential-exchangeability assumption supporting the source.
        exchangeability_assumption_id: String,
    },
    /// A history-dependent intervention rule over the declared horizon.
    AdaptiveRegime {
        /// Active rule identity and version.
        active_regime_id: String,
        /// Reference rule identity and version.
        reference_regime_id: String,
        /// Number of periods.
        horizon: u32,
    },
    /// A fitted temporal response evaluated under an intervention schedule.
    ModelResponse {
        /// Retained model and version identity.
        model_id: String,
        /// Active intervention schedule identity.
        active_schedule_id: String,
        /// Reference intervention schedule identity.
        reference_schedule_id: String,
        /// Number of periods.
        horizon: u32,
    },
}

impl TemporalEstimand {
    fn encode(&self, w: &mut antecedent_core::CanonicalWriter) -> Result<(), EvidenceRefusal> {
        match self {
            Self::Excursion { lag, assignment_law_id } if !assignment_law_id.trim().is_empty() => {
                w.word(0).word(u64::from(*lag)).text(assignment_law_id);
            }
            Self::FixedSchedule { active_schedule_id, reference_schedule_id, horizon }
                if !active_schedule_id.trim().is_empty() && !reference_schedule_id.trim().is_empty()
                    && active_schedule_id != reference_schedule_id && *horizon > 0 => {
                w.word(1).text(active_schedule_id).text(reference_schedule_id).word(u64::from(*horizon));
            }
            Self::StaticRegime { active_regime_id, reference_regime_id, horizon, exchangeability_assumption_id }
                if !active_regime_id.trim().is_empty() && !reference_regime_id.trim().is_empty()
                    && active_regime_id != reference_regime_id && !exchangeability_assumption_id.trim().is_empty()
                    && *horizon > 0 => {
                w.word(4).text(active_regime_id).text(reference_regime_id).word(u64::from(*horizon)).text(exchangeability_assumption_id);
            }
            Self::AdaptiveRegime { active_regime_id, reference_regime_id, horizon }
                if !active_regime_id.trim().is_empty() && !reference_regime_id.trim().is_empty()
                    && active_regime_id != reference_regime_id && *horizon > 0 => {
                w.word(2).text(active_regime_id).text(reference_regime_id).word(u64::from(*horizon));
            }
            Self::ModelResponse { model_id, active_schedule_id, reference_schedule_id, horizon }
                if !model_id.trim().is_empty() && !active_schedule_id.trim().is_empty()
                    && !reference_schedule_id.trim().is_empty() && active_schedule_id != reference_schedule_id
                    && *horizon > 0 => {
                w.word(3).text(model_id).text(active_schedule_id).text(reference_schedule_id).word(u64::from(*horizon));
            }
            _ => return Err(EvidenceRefusal::Invalid(
                "temporal estimands require identified laws, schedules or regimes and a positive horizon".into(),
            )),
        }
        Ok(())
    }
}

/// A bounded substantive relationship between a source temporal effect and another target.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalEffectBridge {
    /// Source excursion or static regime, which must match the evidence and request exactly.
    pub source: TemporalEstimand,
    /// Target schedule, regime, model response or another excursion.
    pub target: TemporalEstimand,
    /// Outcome component measured by the source.
    pub source_component: String,
    /// Target outcome identity.
    pub target_variable_id: String,
    /// Units of the target contrast and decision threshold.
    pub target_units: String,
    /// Bounded causal-effect mapping, its assumption, and the mapping evidence.
    /// A predictive association is explicitly refused.
    pub mapping: ProxyMapping,
}

/// Decision under an explicit temporal bridge, including evidence lineage and assumptions.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalBridgeDecision {
    /// Identity of source, target, mapping, evidence and decision rule.
    pub identity: String,
    /// Source temporal estimand.
    pub source: TemporalEstimand,
    /// Target temporal estimand.
    pub target: TemporalEstimand,
    /// Assumption-dependent decision and current evidence lineage.
    pub decision: ProxyDecision,
}

/// Apply a bounded temporal bridge without equating the source and target estimands.
///
/// The bridge only supports effects, not unobserved arm means or arbitrary nonlinear policy
/// values. Bounds are substantive assumptions, not calibrated uncertainty or identification.
///
/// # Errors
/// Refuses missing or mismatched excursion scope, predictive associations, malformed target
/// identities, missing mapping assumptions and evidence, or unsupported proxy uncertainty.
pub fn decide_with_temporal_bridge(
    evidence: &JointExperimentEvidence,
    request: &Request,
    bridge: &TemporalEffectBridge,
    frame: &DecisionFrame,
) -> Result<TemporalBridgeDecision, EvidenceRefusal> {
    let mut w =
        antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.temporal_bridge.v1");
    bridge.source.encode(&mut w)?;
    bridge.target.encode(&mut w)?;
    match (&bridge.source, &request.target) {
        (
            TemporalEstimand::Excursion { lag, assignment_law_id },
            Target::Excursion { lag: requested, assignment_law_id: law },
        ) if lag == requested && assignment_law_id == law => {}
        (
            TemporalEstimand::StaticRegime {
                active_regime_id,
                reference_regime_id,
                horizon,
                exchangeability_assumption_id,
            },
            Target::Regime { periods, exchangeability_assumption_id: assumption },
        ) if active_regime_id == &request.treated_regime_id
            && reference_regime_id == &request.control_regime_id
            && horizon == periods
            && exchangeability_assumption_id == assumption => {}
        _ => return Err(EvidenceRefusal::Invalid(
            "temporal bridge source must match the observed excursion or static-regime estimand"
                .into(),
        )),
    }
    let proxy = ProxyBridge {
        proxy_component: bridge.source_component.clone(),
        target_variable_id: bridge.target_variable_id.clone(),
        target_units: bridge.target_units.clone(),
        population_id: request.population_id.clone(),
        treated_regime_id: request.treated_regime_id.clone(),
        control_regime_id: request.control_regime_id.clone(),
        horizon: request.horizon,
        mapping: bridge.mapping.clone(),
    };
    let decision = decide_with_proxy(evidence, request, &proxy, frame)?;
    w.text(&decision.identity);
    Ok(TemporalBridgeDecision {
        identity: w.finish_hex(),
        source: bridge.source.clone(),
        target: bridge.target.clone(),
        decision,
    })
}
