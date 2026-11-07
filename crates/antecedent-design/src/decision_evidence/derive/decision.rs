//! Decision classification and interval calibration from derived quantities.

use super::{
    Dependence, DerivedQuantity, DerivedResult, EvidenceRefusal, Exposure, InputBinding,
    JointExperimentEvidence, Request, SamplingInterval, Target, UnresolvedReason, derive,
};
use antecedent_core::CanonicalWriter;

/// The two actions a decision compares and the rule that separates them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionFrame {
    /// Action implemented by the treated regime.
    pub treated_action: String,
    /// Action implemented by the control regime.
    pub control_action: String,
    /// The treated action is preferred when the contrast exceeds this threshold.
    pub threshold: f64,
}

impl DecisionFrame {
    /// Identity of the decision rule (not of the evidence).
    #[must_use]
    pub fn rule_identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.decision_rule.v1");
        w.text(&self.treated_action).text(&self.control_action).real(self.threshold);
        w.finish_hex()
    }
}

/// Which action a decision favors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arm {
    /// The treated action.
    Treated,
    /// The control action.
    Control,
}

/// Classification of a decision.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// One action is preferred under sampling uncertainty and under every declared assumption.
    Robust {
        /// Preferred action name.
        action: String,
    },
    /// The sampled interval favors different actions at vertices of the declared range.
    /// Intermediate assumption values may still leave the decision unresolved.
    AssumptionDependent {
        /// Assumptions the preference depends on.
        assumption_ids: Vec<String>,
        /// Preferred action at the lowest contrast.
        at_lower: String,
        /// Preferred action at the highest contrast.
        at_upper: String,
    },
    /// The decision cannot be stated; the reasons say what is missing.
    Unresolved {
        /// Reasons.
        reasons: Vec<UnresolvedReason>,
    },
}

/// A decision and the derived quantity it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResult {
    /// The frame.
    pub frame: DecisionFrame,
    /// The evidence it rests on.
    pub derived: DerivedResult,
    /// The classification.
    pub verdict: Verdict,
    /// Statement that a classification is not a probability.
    pub statement: &'static str,
}

/// Classify a decision over a derived quantity.
///
/// # Errors
///
/// Those of [`derive`].
pub fn decide(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
    frame: &DecisionFrame,
) -> Result<DecisionResult, EvidenceRefusal> {
    if frame.treated_action == frame.control_action || !frame.threshold.is_finite() {
        return Err(EvidenceRefusal::FrameShape);
    }
    let derived = derive(evidence, request, quantity)?;
    let t = frame.threshold;
    let side = |lo: f64, hi: f64| -> Option<bool> {
        if lo > t {
            Some(true)
        } else if hi < t {
            Some(false)
        } else {
            None
        }
    };
    let name = |treated: bool| {
        if treated { frame.treated_action.clone() } else { frame.control_action.clone() }
    };
    let verdict = if !derived.unresolved.is_empty() {
        Verdict::Unresolved { reasons: derived.unresolved.clone() }
    } else {
        let intervals: Vec<&SamplingInterval> = if derived.sampling_at_vertices.is_empty() {
            derived.sampling.iter().collect()
        } else {
            derived.sampling_at_vertices.iter().collect()
        };
        let sides: Vec<Option<bool>> = intervals.iter().map(|s| side(s.lower, s.upper)).collect();
        if derived.assumption_range.as_ref().is_some_and(|r| !r.exact)
            || sides.is_empty()
            || sides.iter().any(Option::is_none)
        {
            Verdict::Unresolved { reasons: vec![UnresolvedReason::SamplingStraddlesThreshold] }
        } else {
            let all_true = sides.iter().all(|s| *s == Some(true));
            let all_false = sides.iter().all(|s| *s == Some(false));
            match &derived.assumption_range {
                Some(r) if !(all_true || all_false) || side(r.lower, r.upper).is_none() => {
                    Verdict::AssumptionDependent {
                        assumption_ids: r.assumption_ids.clone(),
                        at_lower: name(r.lower > t),
                        at_upper: name(r.upper > t),
                    }
                }
                _ => Verdict::Robust { action: name(all_true) },
            }
        }
    };
    Ok(DecisionResult {
        frame: frame.clone(),
        derived,
        verdict,
        statement: "a classification over declared assumptions; it is not a probability and no assumption range was turned into one",
    })
}

/// Describe the exact decision-evidence interval construction for calibration measurement.
/// This is a candidate basis only: it does not support or attest an interval.
/// The scope count is the number of independently randomized units, not outcome rows.
///
/// # Errors
/// Returns a refusal when no sampling interval is produced, or when an assumption range
/// would require a different calibration target.
pub fn decision_interval_calibration_basis(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
) -> Result<antecedent_core::CalibrationBasis, EvidenceRefusal> {
    let result = derive(evidence, request, quantity)?;
    if result.sampling.is_none()
        || result.assumption_range.is_some()
        || !matches!(
            (&*evidence.design.assignment_design, &request.target),
            ("cluster" | "complete" | "matched_pairs" | "switchback", Target::ExperimentalContrast)
                | ("temporal_excursion", Target::Excursion { .. })
                | ("temporal_regime", Target::Regime { .. })
                | ("adaptive_regime", Target::AdaptiveRegime { .. })
        )
        || !matches!(evidence.exposure, Exposure::NoBetweenUnitInterference { .. })
        || !matches!(
            (evidence.design.assignment_design.as_str(), &evidence.dependence),
            ("cluster" | "complete", Dependence::Joint { .. })
                | (
                    "matched_pairs"
                        | "switchback"
                        | "temporal_excursion"
                        | "temporal_regime"
                        | "adaptive_regime",
                    Dependence::JointArms { .. }
                )
        )
        || quantity.inputs.iter().any(|b| !matches!(b, InputBinding::Measured(_)))
    {
        return Err(EvidenceRefusal::Invalid(
            "calibration basis requires a sampling interval without an assumption range".into(),
        ));
    }
    let key = [
        "DecisionEvidence".to_owned(),
        if matches!(
            evidence.design.assignment_design.as_str(),
            "temporal_regime" | "adaptive_regime"
        ) {
            "Observational"
        } else {
            "RandomizedTrial"
        }
        .to_owned(),
        "design".to_owned(),
        "tabular".to_owned(),
        "Frequentist".to_owned(),
        format!("joint_{}_{}", evidence.design.assignment_design, evidence.adjustment_identity()),
        if matches!(evidence.design.assignment_design.as_str(), "matched_pairs" | "switchback") {
            "delta_method_student_95"
        } else {
            "delta_method_normal_95"
        }
        .to_owned(),
        match evidence.design.assignment_design.as_str() {
            "matched_pairs" => "matched_pair_delta",
            "switchback" => "switchback_sequence_delta",
            "temporal_excursion" => "temporal_cluster_ratio_delta",
            "temporal_regime" => "temporal_regime_bootstrap_delta",
            "adaptive_regime" => "adaptive_regime_jackknife_delta",
            _ => "joint_neyman_delta",
        }
        .to_owned(),
        if matches!(
            evidence.design.assignment_design.as_str(),
            "temporal_regime" | "adaptive_regime"
        ) {
            "independent_subjects"
        } else {
            "independent_assignment_units"
        }
        .to_owned(),
        String::new(),
        match &request.target {
            Target::Excursion { lag, assignment_law_id } => {
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.excursion_calibration_target.v1",
                );
                w.text(&result.quantity_identity).word(u64::from(*lag)).text(assignment_law_id);
                format!("excursion_{}", w.finish_hex())
            }
            Target::AdaptiveRegime { periods, exchangeability_assumption_id } => {
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.adaptive_regime_calibration_target.v1",
                );
                w.text(&result.quantity_identity)
                    .word(u64::from(*periods))
                    .text(exchangeability_assumption_id);
                format!("adaptive_regime_{}", w.finish_hex())
            }
            Target::Regime { periods, exchangeability_assumption_id } => {
                let Some(super::super::DesignDetails::TemporalRegime {
                    method,
                    bootstrap_replicates: Some(replicates),
                    ..
                }) = &evidence.design_details
                else {
                    return Err(EvidenceRefusal::Invalid(
                        "regime calibration requires aligned bootstrap metadata".into(),
                    ));
                };
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.regime_calibration_target.v1",
                );
                w.text(&result.quantity_identity)
                    .word(u64::from(*periods))
                    .text(exchangeability_assumption_id)
                    .text(method)
                    .word(*replicates as u64);
                format!("regime_{}", w.finish_hex())
            }
            _ => format!("derived_{}", result.quantity_identity),
        },
    ]
    .map(std::sync::Arc::<str>::from);
    Ok(antecedent_core::CalibrationBasis::new(
        key,
        0.95,
        "point".into(),
        (if matches!(
            evidence.design.assignment_design.as_str(),
            "matched_pairs"
                | "switchback"
                | "temporal_excursion"
                | "temporal_regime"
                | "adaptive_regime"
        ) {
            evidence.design.treated_units
        } else {
            evidence.design.treated_units + evidence.design.control_units
        }) as u64,
        None,
        None,
        0.0,
    )
    .with_validation("none"))
}
