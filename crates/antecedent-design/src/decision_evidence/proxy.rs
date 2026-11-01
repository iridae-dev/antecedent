//! Explicit, assumption-dependent bridges from a randomized proxy effect to a target effect.
//! A predictive association alone is never a causal-effect bridge.
use super::{
    DecisionFrame, DerivedQuantity, EvidenceRefusal, InputBinding, JointExperimentEvidence,
    Request, Target, UnresolvedReason, UtilityExpr, Verdict, derive,
};

/// A declared relationship between the proxy effect and the target effect.
#[derive(Clone, Debug, PartialEq)]
pub enum ProxyMapping {
    /// A predictive association is not sufficient to bridge causal effects.
    PredictiveAssociation,
    /// `target_effect = multiplier * proxy_effect + residual`, over declared ranges.
    /// Ranges are assumptions, not probability distributions or empirical confidence intervals.
    CausalEffectBounds {
        /// Inclusive multiplier range.
        multiplier: (f64, f64),
        /// Inclusive residual-effect range.
        residual: (f64, f64),
        /// Substantive assumption identifying the effect relationship.
        assumption_id: String,
        /// Evidence supporting the mapping, retained as lineage (not promoted to a license).
        mapping_evidence_id: String,
    },
}

/// A mapping whose scientific scope is explicitly declared.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyBridge {
    /// Name of the measured proxy component.
    pub proxy_component: String,
    /// Identity of the decision-relevant target outcome.
    pub target_variable_id: String,
    /// Units of the target effect and decision threshold.
    pub target_units: String,
    /// Population in which the effect relationship is assumed.
    pub population_id: String,
    /// Treated regime in the mapping.
    pub treated_regime_id: String,
    /// Control regime in the mapping.
    pub control_regime_id: String,
    /// Horizon at which the mapping is assumed.
    pub horizon: u32,
    /// Mapping and its evidence.
    pub mapping: ProxyMapping,
}

/// Partial evidence about the target effect and a decision made over its admissible range.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyDecision {
    /// Identity of the evidence, mapping, target scope and decision rule.
    pub identity: String,
    /// Range at the observed proxy effect, conditional on the mapping assumptions.
    pub assumption_range: Option<(f64, f64)>,
    /// Outer envelope obtained by applying the mapping bounds to the proxy sampling interval.
    /// This is a sensitivity envelope, not an identified set or a calibrated interval.
    pub sampling_envelope: Option<(f64, f64)>,
    /// Scientific decision classification.
    pub verdict: Verdict,
    /// Experimental and mapping evidence identities, in that order.
    pub evidence_ids: Vec<String>,
    /// Assumption that the effect mapping holds.
    pub assumption_id: String,
    /// The target outcome identity.
    pub target_variable_id: String,
    /// Units of the target effect.
    pub target_units: String,
}

impl ProxyDecision {
    /// Evidence that is withdrawn or unrecorded in the current ledger.
    /// Historical decision values remain inspectable after withdrawal.
    #[must_use]
    pub fn unsupported_evidence(&self, ledger: &super::EvidenceLedger) -> Vec<(String, String)> {
        self.evidence_ids.iter().flat_map(|id| ledger.unsupported_parts(id)).collect()
    }
}

fn multiply(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let v = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Bound a target causal effect using an explicitly declared proxy-effect relationship.
///
/// No independence between mapping and experimental evidence is assumed: declared bounds are
/// propagated as a set. The envelope inherits no calibrated standing from the proxy estimate.
///
/// # Errors
/// Refuses predictive associations, scope mismatch, malformed ranges and deployment extrapolation.
pub fn decide_with_proxy(
    evidence: &JointExperimentEvidence,
    request: &Request,
    bridge: &ProxyBridge,
    frame: &DecisionFrame,
) -> Result<ProxyDecision, EvidenceRefusal> {
    if !matches!(
        request.target,
        Target::ExperimentalContrast | Target::Excursion { .. } | Target::Regime { .. }
    ) {
        return Err(EvidenceRefusal::Invalid(
            "proxy effect mappings do not establish deployment extrapolation".into(),
        ));
    }
    if bridge.population_id != request.population_id
        || bridge.treated_regime_id != request.treated_regime_id
        || bridge.control_regime_id != request.control_regime_id
        || bridge.horizon != request.horizon
    {
        return Err(EvidenceRefusal::Invalid(
            "proxy mapping scope must match population, interventions and horizon".into(),
        ));
    }
    if bridge.target_variable_id.trim().is_empty()
        || bridge.target_units.trim().is_empty()
        || frame.treated_action.trim().is_empty()
        || frame.control_action.trim().is_empty()
        || frame.treated_action == frame.control_action
        || !frame.threshold.is_finite()
    {
        return Err(EvidenceRefusal::Invalid(
            "proxy target and a valid two-action decision are required".into(),
        ));
    }
    let (multiplier, residual, assumption_id, mapping_evidence_id) = match &bridge.mapping {
        ProxyMapping::PredictiveAssociation => {
            return Err(EvidenceRefusal::Invalid(
                "predictive association does not identify a causal effect relationship".into(),
            ));
        }
        ProxyMapping::CausalEffectBounds {
            multiplier,
            residual,
            assumption_id,
            mapping_evidence_id,
        } => (*multiplier, *residual, assumption_id, mapping_evidence_id),
    };
    if [multiplier.0, multiplier.1, residual.0, residual.1].iter().any(|x| !x.is_finite())
        || multiplier.0 > multiplier.1
        || residual.0 > residual.1
        || assumption_id.trim().is_empty()
        || mapping_evidence_id.trim().is_empty()
    {
        return Err(EvidenceRefusal::Invalid("proxy mapping requires finite ordered effect bounds and identified assumptions and evidence".into()));
    }
    let q = DerivedQuantity {
        name: "proxy_effect".into(),
        units: "proxy".into(),
        expr: UtilityExpr::Input(0),
        inputs: vec![InputBinding::Measured(bridge.proxy_component.clone())],
    };
    let result = derive(evidence, request, &q)?;
    if result.assumption_range.is_some() {
        return Err(EvidenceRefusal::Invalid(
            "proxy mapping does not yet compose with an additional spillover assumption range"
                .into(),
        ));
    }
    let bound = |x: (f64, f64)| {
        let m = multiply(multiplier, x);
        (m.0 + residual.0, m.1 + residual.1)
    };
    let assumption_range = result.contrast.map(|v| bound((v, v)));
    let proxy_sampling = result.sampling.as_ref().map(|s| (s.lower, s.upper));
    let sampling_envelope = proxy_sampling.map(bound);
    if assumption_range
        .into_iter()
        .chain(sampling_envelope)
        .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
    {
        return Err(EvidenceRefusal::Invalid("proxy effect bounds overflow".into()));
    }
    let verdict = if !result.unresolved.is_empty() {
        Verdict::Unresolved { reasons: result.unresolved }
    } else if let Some((lo, hi)) = sampling_envelope {
        if lo > frame.threshold {
            Verdict::Robust { action: frame.treated_action.clone() }
        } else if hi < frame.threshold {
            Verdict::Robust { action: frame.control_action.clone() }
        } else {
            let proxy_interval = proxy_sampling.expect("envelope requires proxy sampling");
            let mut lower_action = false;
            let mut upper_action = false;
            for factor in [multiplier.0, multiplier.1] {
                for offset in [residual.0, residual.1] {
                    let endpoint = multiply((factor, factor), proxy_interval);
                    lower_action |= endpoint.1 + offset < frame.threshold;
                    upper_action |= endpoint.0 + offset > frame.threshold;
                }
            }
            if lower_action && upper_action {
                Verdict::AssumptionDependent {
                    assumption_ids: vec![assumption_id.clone()],
                    at_lower: frame.control_action.clone(),
                    at_upper: frame.treated_action.clone(),
                }
            } else {
                Verdict::Unresolved { reasons: vec![UnresolvedReason::SamplingStraddlesThreshold] }
            }
        }
    } else {
        Verdict::Unresolved { reasons: vec![UnresolvedReason::SamplingUnavailable] }
    };
    let mut writer =
        antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.proxy_decision.v1");
    writer
        .text(&evidence.estimate_identity())
        .text(&bridge.proxy_component)
        .text(&bridge.target_variable_id)
        .text(&bridge.target_units)
        .text(&bridge.population_id)
        .text(&bridge.treated_regime_id)
        .text(&bridge.control_regime_id)
        .word(u64::from(bridge.horizon))
        .real(multiplier.0)
        .real(multiplier.1)
        .real(residual.0)
        .real(residual.1)
        .text(assumption_id)
        .text(mapping_evidence_id)
        .text(&frame.rule_identity());
    Ok(ProxyDecision {
        identity: writer.finish_hex(),
        assumption_range,
        sampling_envelope,
        verdict,
        evidence_ids: vec![evidence.evidence_id.clone(), mapping_evidence_id.clone()],
        assumption_id: assumption_id.clone(),
        target_variable_id: bridge.target_variable_id.clone(),
        target_units: bridge.target_units.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::multiply;

    #[test]
    fn product_bounds_cover_negative_and_positive_effects() {
        assert_eq!(multiply((-2.0, 3.0), (-4.0, 5.0)), (-12.0, 15.0));
        assert_eq!(multiply((-2.0, -1.0), (2.0, 4.0)), (-8.0, -2.0));
    }
}
