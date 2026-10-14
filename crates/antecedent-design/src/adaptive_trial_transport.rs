//! An adaptive trial as a transport source.
//!
//! In an adaptive trial the arm a unit receives depends on the units before
//! it, so the assignment that selects who is observed under each arm is
//! driven by a policy and by the data. Its observations carry information
//! about a target population only through the contract that recorded that
//! policy. This module is one bounded route: arm means standardized to a
//! finite stratum law of the target, estimated from the trial by weighting
//! with the realized assignment probabilities.
//!
//! # Contract
//!
//! * **Represented design.** The trial is an [`AdaptiveDesignContract`]: the
//!   assignment-policy history, the realized assignment probabilities, the
//!   exposure history, outcome availability and delay, the interim analyses
//!   and the stopping rule. A trial without that contract is refused with the
//!   missing source contract named; it is never analysed as if it were fixed.
//! * **Licensed cell.** The contract's procedure cell must be licensed for
//!   adaptively weighted scores. A fixed design has its own transport route.
//! * **Selection diagram.** The caller declares the stratum law as shifted or
//!   invariant and the outcome-given-stratum-and-arm mechanism, which must be
//!   invariant; a shifted outcome mechanism is refused.
//! * **Support.** For every target stratum with positive mass and for both
//!   compared arms the trial must hold at least one analysed unit assigned
//!   that arm, each with realized assignment probability at or above the
//!   declared floor.
//!
//! # Estimator
//!
//! For stratum `s` and arm `a`, with analysed units `u` assigned `a` and
//! realized assignment probability `pi_u(a)`:
//!
//! ```text
//! mu(s, a) = sum_u y_u / pi_u(a)  /  sum_u 1 / pi_u(a)
//! theta(a) = sum_s q(s) * mu(s, a)
//! ```
//!
//! where `q` is the target stratum law and `y_u` is the mean of the unit's
//! measurements. The values are points; the estimator carries no interval
//! and no sampling distribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{AdaptiveDesignContract, InferenceParadigm};
use antecedent_core::CanonicalWriter;

use crate::sequential_evaluation::{SequentialEvaluationError, SequentialResult};
use crate::transported_policy::{MechanismStatus, TRANSPORTED_POLICY_INFERENCE, transport_refusal};

/// Version word written into every adaptive-transport identity.
pub const ADAPTIVE_TRANSPORT_IDENTITY_VERSION: u64 = 1;
/// Most strata a target law may name.
pub const MAX_TRANSPORT_STRATA: usize = 1024;
/// Tolerance for the target law summing to one.
const LAW_TOLERANCE: f64 = 1e-9;

/// A mechanism of the trial that a selection node may point to.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TrialMechanism {
    /// The law of the stratifying covariate in the population.
    StratumLaw,
    /// The outcome law given stratum and arm.
    OutcomeGivenStratumAndArm,
}

impl TrialMechanism {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StratumLaw => "stratum_law",
            Self::OutcomeGivenStratumAndArm => "outcome_given_stratum_and_arm",
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::StratumLaw => 1,
            Self::OutcomeGivenStratumAndArm => 2,
        }
    }
}

/// A declared selection diagram over the trial's mechanisms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialSelectionDiagram {
    /// Population the trial sampled.
    pub source_population: String,
    /// Population the arm means are transported to.
    pub target_population: String,
    /// The status of each mechanism: both mechanisms, once.
    pub mechanisms: Vec<(TrialMechanism, MechanismStatus)>,
}

impl TrialSelectionDiagram {
    /// Canonical identity, independent of the order the mechanisms are listed
    /// in.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.adaptive_trial_transport.diagram");
        writer
            .word(ADAPTIVE_TRANSPORT_IDENTITY_VERSION)
            .text(&self.source_population)
            .text(&self.target_population);
        let mut rows = self.mechanisms.clone();
        rows.sort();
        writer.word(rows.len() as u64);
        for (mechanism, status) in rows {
            writer.word(mechanism.tag()).word(status.tag());
        }
        writer.finish_hex()
    }
}

/// What is asked of an adaptive-trial transport.
///
/// `contract`, `diagram` and `target_stratum_law` are `Option` so that a
/// missing contract is representable and refused by name.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveTransportRequest<'a> {
    /// The retained analysis contract of the trial.
    pub contract: Option<&'a AdaptiveDesignContract>,
    /// The stratum index of each enrolled unit, in enrollment order.
    pub unit_strata: &'a [usize],
    /// The stratum labels.
    pub stratum_labels: &'a [String],
    /// The target population's stratum law, in stratum order.
    pub target_stratum_law: Option<&'a [f64]>,
    /// The declared selection diagram.
    pub diagram: Option<&'a TrialSelectionDiagram>,
    /// Index of the arm whose transported mean is `treatment_mean`.
    pub treatment_arm: usize,
    /// Index of the comparison arm.
    pub control_arm: usize,
}

/// The weighted mean of one stratum and arm.
#[derive(Clone, Debug, PartialEq)]
pub struct StratumArmCell {
    /// Stratum index.
    pub stratum: usize,
    /// Arm index.
    pub arm: usize,
    /// Analysed units assigned the arm in the stratum.
    pub units: usize,
    /// Sum of the inverse realized assignment probabilities.
    pub weight_sum: f64,
    /// The weighted mean outcome.
    pub mean: f64,
}

/// The premises a transported arm mean is conditional on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveTransportPremises {
    /// The declared selection diagram.
    pub selection: &'static str,
    /// Assignment depends only on what the contract recorded.
    pub sequential_ignorability: &'static str,
    /// The outcome mechanism is shared.
    pub invariance: &'static str,
    /// Positivity in the trial and overlap with the target.
    pub support: &'static str,
    /// Outcome availability does not depend on the outcome.
    pub availability: &'static str,
}

impl AdaptiveTransportPremises {
    /// The premises every value of this route is conditional on.
    pub const DECLARED: Self = Self {
        selection: "selection acts on the stratum law alone: no selection node points to the outcome mechanism given stratum and arm",
        sequential_ignorability: "each unit's assignment depends only on the history and covariates the contract recorded, and the recorded probabilities are the ones the policy used",
        invariance: "the outcome law given stratum and arm is the same in the trial's population and the target",
        support: "every target stratum with positive mass has analysed units under both compared arms, each at or above the declared probability floor",
        availability: "outcome availability and delay do not depend on the outcome, and units never observed are ignorable",
    };
}

/// The result of an adaptive-trial transport.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveTransportEstimate {
    /// Transported mean of the treatment arm.
    pub treatment_mean: f64,
    /// Transported mean of the control arm.
    pub control_mean: f64,
    /// `treatment_mean - control_mean`.
    pub contrast: f64,
    /// The cells used (target strata with positive mass, both arms), in
    /// stratum then arm order.
    pub cells: Vec<StratumArmCell>,
    /// Units analysed at the final look.
    pub analysed_units: usize,
    /// Identity of the trial contract.
    pub contract_identity: String,
    /// Identity of the declared selection diagram.
    pub diagram_identity: String,
    /// Canonical identity of the whole estimate.
    pub identity: String,
    /// The premises the estimate is conditional on.
    pub premises: AdaptiveTransportPremises,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

fn adaptive_refusal(
    code: &'static str,
    rule: &str,
    message: impl Into<String>,
    remedy: &'static str,
) -> SequentialEvaluationError {
    transport_refusal(code, format!("adaptive_transport.{rule}"), message, remedy)
}

fn check_diagram(diagram: &TrialSelectionDiagram) -> SequentialResult<()> {
    let mut stratum: Option<MechanismStatus> = None;
    let mut outcome: Option<MechanismStatus> = None;
    for &(mechanism, status) in &diagram.mechanisms {
        let slot = match mechanism {
            TrialMechanism::StratumLaw => &mut stratum,
            TrialMechanism::OutcomeGivenStratumAndArm => &mut outcome,
        };
        if slot.replace(status).is_some() {
            return Err(adaptive_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "mechanism_declared_twice",
                format!("the diagram gives `{}` a status twice", mechanism.as_str()),
                "declare each mechanism once",
            ));
        }
    }
    let (Some(_), Some(outcome)) = (stratum, outcome) else {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "mechanism_status_missing",
            "the diagram does not declare both the stratum law and the outcome mechanism",
            "declare the stratum law and the outcome-given-stratum-and-arm mechanism as invariant or shifted; invariance is never assumed",
        ));
    };
    if outcome == MechanismStatus::Shifted {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "outcome_mechanism_not_invariant",
            "the outcome law given stratum and arm is declared shifted; the trial cannot supply it for the target",
            "establish the outcome mechanism in the target population (a target trial), or restrict the shift to the stratum law",
        ));
    }
    // Either status of the stratum law is licensed: a shifted law is the
    // purpose of the route and an invariant one is a trivial transport.
    Ok(())
}

fn check_shape(
    contract: &AdaptiveDesignContract,
    request: &AdaptiveTransportRequest<'_>,
    law: &[f64],
) -> SequentialResult<()> {
    let strata = request.stratum_labels.len();
    let labels_ok = (1..=MAX_TRANSPORT_STRATA).contains(&strata)
        && request.stratum_labels.iter().all(|label| !label.trim().is_empty());
    let mut sorted: Vec<&str> = request.stratum_labels.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let distinct = sorted.windows(2).all(|pair| pair[0] != pair[1]);
    if !labels_ok || !distinct {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "strata",
            format!(
                "a transport needs 1 to {MAX_TRANSPORT_STRATA} distinct non-blank stratum labels"
            ),
            "name the strata the target law is declared over",
        ));
    }
    if request.unit_strata.len() != contract.units()
        || request.unit_strata.iter().any(|&s| s >= strata)
    {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "unit_strata",
            format!(
                "{} unit strata were supplied for a trial of {} units, or one is outside the {strata} strata",
                request.unit_strata.len(),
                contract.units()
            ),
            "give every enrolled unit one declared stratum",
        ));
    }
    let law_ok = law.len() == strata
        && law.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && (law.iter().sum::<f64>() - 1.0).abs() <= LAW_TOLERANCE;
    if !law_ok {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "target_stratum_law",
            "the target stratum law must be one probability per stratum summing to one",
            "supply the target population's law over the declared strata",
        ));
    }
    let arms = contract.arms();
    if request.treatment_arm >= arms
        || request.control_arm >= arms
        || request.treatment_arm == request.control_arm
    {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "arms",
            format!("the compared arms must be two distinct arms of the trial's {arms}"),
            "name two distinct declared arms",
        ));
    }
    Ok(())
}

/// Transport two arm means of an adaptive trial to a target stratum law.
///
/// # Errors
/// A missing trial contract, selection diagram or target stratum law; a
/// fixed-design trial; a procedure cell not licensed for adaptively weighted
/// scores; exposure that differs from assignment or an outcome-dependent
/// delay; an outcome mechanism marked shifted; an assigned arm below the
/// declared probability floor; a target stratum with no analysed unit under a
/// compared arm; and malformed strata, arms or target law.
///
/// # Panics
/// Never for a contract built by [`AdaptiveDesignContract::new`].
pub fn transport_adaptive_trial(
    request: &AdaptiveTransportRequest<'_>,
) -> SequentialResult<AdaptiveTransportEstimate> {
    let Some(contract) = request.contract else {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("adaptive_contract_incomplete"),
            "assignment_monitoring_contract_missing",
            "the trial's data were collected under an assignment that depends on history, but no retained assignment and monitoring contract was supplied",
            "supply the AdaptiveDesignContract of the trial: the policy history, realized assignment probabilities, exposure history, outcome availability and delay, interim analyses and stopping rule",
        ));
    };
    let Some(law) = request.target_stratum_law else {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "target_stratum_law_missing",
            "no stratum law was supplied for the target population",
            "supply the target population's law over the strata; the trial cannot provide it",
        ));
    };
    let Some(diagram) = request.diagram else {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "selection_diagram_missing",
            "no selection diagram relates the trial's population to the target",
            "supply a selection diagram marking the stratum law and the outcome mechanism invariant or shifted",
        ));
    };
    if !contract.is_adaptive() {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "design_not_adaptive",
            "the contract records a design fixed in advance, whose assignment is not history-driven",
            "transport a fixed-design trial through the fixed-design transport routes; this route is for data collected under an adaptive assignment",
        ));
    }
    check_shape(contract, request, law)?;
    check_diagram(diagram)?;
    contract
        .admit_paradigm(InferenceParadigm::FrequentistAdaptiveWeighting)
        .map_err(SequentialEvaluationError::from)?;
    let declaration = contract.declaration();
    let probabilities = declaration.assignment_probabilities.as_ref().ok_or_else(|| {
        adaptive_refusal(
            antecedent_core::reason_code!("adaptive_contract_incomplete"),
            "assignment_monitoring_contract_missing",
            "the contract holds no realized assignment probabilities",
            "supply the AdaptiveDesignContract of the trial",
        )
    })?;
    let last = contract.looks().saturating_sub(1);
    let analysed = contract.analysis_units(last);
    if analysed.is_empty() {
        return Err(adaptive_refusal(
            antecedent_core::reason_code!("transport_support_failure"),
            "no_analysed_units",
            "no outcome was available at the final look",
            "the trial holds no observed outcome to transport",
        ));
    }
    let arms = [request.treatment_arm, request.control_arm];
    let strata = request.stratum_labels.len();
    // numerator, denominator and count by (stratum, compared arm)
    let mut sums = vec![[(0.0_f64, 0.0_f64, 0_usize); 2]; strata];
    for &unit in &analysed {
        let arm = declaration.assigned_arm[unit] as usize;
        let Some(slot) = arms.iter().position(|&a| a == arm) else { continue };
        let probability = probabilities[unit][arm];
        if probability < declaration.probability_floor {
            return Err(adaptive_refusal(
                antecedent_core::reason_code!("adaptive_assumption_violated"),
                "assignment_below_floor",
                format!(
                    "unit {unit} was assigned arm {arm} at probability {probability}, below the declared floor {}",
                    declaration.probability_floor
                ),
                "restrict the compared arms to those the policy kept above the floor, or declare a lower floor with its justification; nothing is clipped",
            ));
        }
        let measurements = &declaration.outcomes[unit];
        if measurements.is_empty() {
            return Err(adaptive_refusal(
                antecedent_core::reason_code!("adaptive_contract_incomplete"),
                "outcome_missing",
                format!("analysed unit {unit} carries no measurement"),
                "retain the outcome of every unit whose availability is declared",
            ));
        }
        #[allow(clippy::cast_precision_loss, reason = "a unit holds at most 1024 measurements")]
        let outcome = measurements.iter().sum::<f64>() / measurements.len() as f64;
        let weight = 1.0 / probability;
        let cell = &mut sums[request.unit_strata[unit]][slot];
        cell.0 += weight * outcome;
        cell.1 += weight;
        cell.2 += 1;
    }
    let mut cells = Vec::new();
    let mut theta = [0.0_f64; 2];
    for (stratum, &mass) in law.iter().enumerate() {
        if mass <= 0.0 {
            continue;
        }
        for (slot, &arm) in arms.iter().enumerate() {
            let (numerator, denominator, units) = sums[stratum][slot];
            if units == 0 {
                return Err(adaptive_refusal(
                    antecedent_core::reason_code!("transport_support_failure"),
                    "stratum_arm_unobserved",
                    format!(
                        "target stratum `{}` has mass {mass} but no analysed unit was assigned arm {arm} there",
                        request.stratum_labels[stratum]
                    ),
                    "restrict the target law to strata the trial observed under both arms, coarsen the strata, or collect units there",
                ));
            }
            let mean = numerator / denominator;
            theta[slot] += mass * mean;
            cells.push(StratumArmCell { stratum, arm, units, weight_sum: denominator, mean });
        }
    }
    let diagram_identity = diagram.identity_hex();
    let contract_identity = contract.identity().to_hex();
    let mut writer = CanonicalWriter::new("antecedent.adaptive_trial_transport.estimate");
    writer
        .word(ADAPTIVE_TRANSPORT_IDENTITY_VERSION)
        .text(&contract_identity)
        .text(&diagram_identity)
        .texts(request.stratum_labels)
        .reals(law)
        .word(request.treatment_arm as u64)
        .word(request.control_arm as u64);
    writer.word(request.unit_strata.len() as u64);
    for &stratum in request.unit_strata {
        writer.word(stratum as u64);
    }
    writer.real(theta[0]).real(theta[1]);
    Ok(AdaptiveTransportEstimate {
        treatment_mean: theta[0],
        control_mean: theta[1],
        contrast: theta[0] - theta[1],
        cells,
        analysed_units: analysed.len(),
        contract_identity,
        diagram_identity,
        identity: writer.finish_hex(),
        premises: AdaptiveTransportPremises::DECLARED,
        inference: TRANSPORTED_POLICY_INFERENCE,
    })
}

/// Re-derive an estimate from its request and compare, rather than trusting
/// the stored values.
///
/// # Errors
/// Every refusal of [`transport_adaptive_trial`], and a refusal when the
/// stored identity or any of the three values differs bit for bit.
pub fn verify_adaptive_transport(
    estimate: &AdaptiveTransportEstimate,
    request: &AdaptiveTransportRequest<'_>,
) -> SequentialResult<()> {
    let fresh = transport_adaptive_trial(request)?;
    if fresh.identity != estimate.identity
        || fresh.treatment_mean.to_bits() != estimate.treatment_mean.to_bits()
        || fresh.control_mean.to_bits() != estimate.control_mean.to_bits()
        || fresh.contrast.to_bits() != estimate.contrast.to_bits()
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_transport.value_mismatch",
            "a stored transported arm mean does not replay from its contract, strata, diagram and target law",
            "regenerate the estimate from the request",
        ));
    }
    Ok(())
}
