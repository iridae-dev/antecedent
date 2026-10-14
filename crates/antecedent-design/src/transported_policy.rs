//! Transported evaluation of a supplied contingent policy.
//!
//! A [`ContingentPolicy`] is evaluated by the sequential g-formula in the
//! model's own population. This module evaluates the same fixed policy in a
//! *target* population, under a declared selection diagram, without changing
//! the policy and without learning anything from the target.
//!
//! # Contract
//!
//! * **Selection diagram.** The caller declares, for every mechanism of the
//!   model (the initial state law, each action's transition law, each
//!   observation channel's law, the state utility), whether a selection node
//!   points to it ([`MechanismStatus::Shifted`]) or it is invariant between
//!   source and target ([`MechanismStatus::Invariant`]). A mechanism left
//!   undeclared is refused by name; there is no default invariance.
//! * **Bounded class.** Only the initial-state law may be shifted. A shifted
//!   transition, observation or utility mechanism is not identified by the
//!   source and is refused with the mechanism named.
//! * **Target law.** The caller supplies the target initial-state law. Its
//!   support must lie inside the source initial-state support, so every
//!   history reachable in the target is reachable in the source and every
//!   mechanism used was learned where it is used.
//! * **Value.** The transported value is the sequential g-formula under the
//!   source's invariant mechanisms and the target's initial law:
//!   `V_T(pi) = sum over histories h of P_T(h | pi) * realized(h)`.
//!
//! The value is a point value. No interval, posterior or coverage statement
//! is made, and the only premises are the declared ones in
//! [`TransportPremises`].
//!
//! # Refresh
//!
//! Refresh with outcome-only data. The transported value is a function of
//! the source model, the policy, the declared selection diagram and the target
//! initial law. New outcome-only data refits the source model; the policy,
//! the diagram and the target law are kept as declarations and the value, its
//! identity and the source's own evaluation are recomputed. A refit that moves
//! mass outside the declared invariant mechanisms, a target law outside the
//! source's support or a model of another population is refused
//! (`diagram_binding_mismatch`, `initial_law_declared_invariant` and the
//! support refusal).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{CanonicalWriter, ExecutionContext};

use crate::sequential_evaluation::{
    ContingentPolicy, EvaluationLimits, FiniteSequentialModel, PolicyEvaluation,
    SEQUENTIAL_EVALUATION_INFERENCE, SequentialAssumptions, SequentialEvaluationError,
    SequentialResult, evaluate_policy,
};

/// Version word written into every transported-policy identity. Bump only
/// when a canonical encoding in this module changes.
pub const TRANSPORTED_POLICY_IDENTITY_VERSION: u64 = 1;
/// The only inferential claim the route carries.
pub const TRANSPORTED_POLICY_INFERENCE: &str = SEQUENTIAL_EVALUATION_INFERENCE;
/// Tolerance for the target law summing to one.
const LAW_TOLERANCE: f64 = 1e-9;
/// Largest difference at which two initial laws are one law.
const SAME_LAW_TOLERANCE: f64 = 1e-12;

/// A mechanism of a finite sequential model that a selection node may point
/// to.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ModelMechanism {
    /// The initial law over the latent states.
    InitialState,
    /// The transition law of one action.
    Transition {
        /// Action index.
        action: usize,
    },
    /// The observation (confusion) law of one action's channel.
    Observation {
        /// Action index.
        action: usize,
    },
    /// The utility assigned to the latent states.
    Utility,
}

impl ModelMechanism {
    /// Stable name, with the action index where one applies.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::InitialState => "initial_state".to_owned(),
            Self::Transition { action } => format!("transition[{action}]"),
            Self::Observation { action } => format!("observation[{action}]"),
            Self::Utility => "utility".to_owned(),
        }
    }

    pub(crate) fn write(self, writer: &mut CanonicalWriter) {
        match self {
            Self::InitialState => {
                writer.word(1);
            }
            Self::Transition { action } => {
                writer.word(2).word(action as u64);
            }
            Self::Observation { action } => {
                writer.word(3).word(action as u64);
            }
            Self::Utility => {
                writer.word(4);
            }
        }
    }
}

/// Whether a mechanism is the same in source and target.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum MechanismStatus {
    /// No selection node points to the mechanism: it is shared.
    Invariant,
    /// A selection node points to the mechanism: it may differ.
    Shifted,
}

impl MechanismStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Invariant => "invariant",
            Self::Shifted => "shifted",
        }
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Self::Invariant => 1,
            Self::Shifted => 2,
        }
    }
}

/// A declared selection diagram over the mechanisms of the model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionDiagram {
    /// Population the model's mechanisms were established for.
    pub source_population: String,
    /// Population the policy is evaluated in.
    pub target_population: String,
    /// The status of each mechanism: every mechanism of the model, once.
    pub mechanisms: Vec<(ModelMechanism, MechanismStatus)>,
}

impl SelectionDiagram {
    /// Canonical identity, independent of the order the mechanisms are listed
    /// in.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.transported_policy.selection_diagram");
        writer
            .word(TRANSPORTED_POLICY_IDENTITY_VERSION)
            .text(&self.source_population)
            .text(&self.target_population);
        let mut rows = self.mechanisms.clone();
        rows.sort();
        writer.word(rows.len() as u64);
        for (mechanism, status) in rows {
            mechanism.write(&mut writer);
            writer.word(status.tag());
        }
        writer.finish_hex()
    }
}

/// The premises a transported value is conditional on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransportPremises {
    /// The declared selection diagram.
    pub selection: &'static str,
    /// The invariant mechanisms.
    pub invariance: &'static str,
    /// Support overlap of the target initial law.
    pub overlap: &'static str,
    /// The premises of the sequential g-formula itself.
    pub sequential: SequentialAssumptions,
}

impl TransportPremises {
    /// The premises every transported value is conditional on.
    pub const DECLARED: Self = Self {
        selection: "the declared selection diagram is complete: selection acts only on the mechanisms marked shifted, here the initial-state law alone",
        invariance: "every mechanism marked invariant (transitions, observation channels, utility) is the same law in the source and the target, so the source's interventional laws are the target's",
        overlap: "the target initial-state law is absolutely continuous with respect to the source's, so every target history is reachable in the source",
        sequential: SequentialAssumptions::DECLARED,
    };
}

/// Which axis of difference a route transports across; it selects the detail
/// namespace and the remedies of the shared checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Axis {
    Population,
    Period,
}

impl Axis {
    pub(crate) const fn prefix(self) -> &'static str {
        match self {
            Self::Population => "transported_policy",
            Self::Period => "temporal_policy_transport",
        }
    }

    pub(crate) fn detail(self, rule: &str) -> String {
        format!("{}.{rule}", self.prefix())
    }
}

/// Build a typed refusal; each call site names its registered reason code.
pub(crate) fn transport_refusal(
    code: &'static str,
    detail: impl Into<String>,
    message: impl Into<String>,
    remedy: &'static str,
) -> SequentialEvaluationError {
    SequentialEvaluationError::new(code, detail, message, remedy)
}

/// Every mechanism of `model`, in canonical order: the initial state, each
/// action's transition and (when it has a channel) observation, the utility.
pub(crate) fn required_mechanisms(model: &FiniteSequentialModel) -> Vec<ModelMechanism> {
    let mut out = vec![ModelMechanism::InitialState];
    for (action, spec) in model.actions.iter().enumerate() {
        out.push(ModelMechanism::Transition { action });
        if spec.channel.is_some() {
            out.push(ModelMechanism::Observation { action });
        }
    }
    out.push(ModelMechanism::Utility);
    out
}

/// Check a declaration covers the model exactly and shifts only the initial
/// state law; return whether the initial state law is shifted.
pub(crate) fn shifted_initial_state(
    model: &FiniteSequentialModel,
    declared: &[(ModelMechanism, MechanismStatus)],
    axis: Axis,
) -> SequentialResult<bool> {
    let required = required_mechanisms(model);
    let mut seen: BTreeMap<ModelMechanism, MechanismStatus> = BTreeMap::new();
    for &(mechanism, status) in declared {
        if !required.contains(&mechanism) {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                axis.detail("mechanism_unknown"),
                format!(
                    "the declaration names `{}`, which the model does not have",
                    mechanism.name()
                ),
                "declare the status of the model's own mechanisms only: the initial state, each action's transition, each channel's observation law and the utility",
            ));
        }
        if seen.insert(mechanism, status).is_some() {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                axis.detail("mechanism_declared_twice"),
                format!("the declaration gives `{}` a status twice", mechanism.name()),
                "declare each mechanism once",
            ));
        }
    }
    for mechanism in &required {
        if !seen.contains_key(mechanism) {
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                axis.detail("mechanism_status_missing"),
                format!(
                    "the declaration does not say whether `{}` is invariant or shifted",
                    mechanism.name()
                ),
                "declare every mechanism of the model as invariant or shifted; invariance is never assumed for an undeclared mechanism",
            ));
        }
    }
    for (mechanism, status) in &seen {
        if *status == MechanismStatus::Shifted && *mechanism != ModelMechanism::InitialState {
            let (rule, remedy) = match axis {
                Axis::Population => (
                    "mechanism_not_invariant",
                    "the source cannot supply a transition, observation or utility mechanism that differs in the target: establish it for the target population (a target experiment, or a model native to the target), or restrict the shift to the initial-state law",
                ),
                Axis::Period => (
                    "mechanism_time_varying",
                    "a mechanism that varies with the period is not carried across periods by the source: establish it for the target period, or declare it time-invariant if that is the scientific premise",
                ),
            };
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                axis.detail(rule),
                format!(
                    "`{}` is declared shifted; only the initial-state law may differ between source and target",
                    mechanism.name()
                ),
                remedy,
            ));
        }
    }
    Ok(seen.get(&ModelMechanism::InitialState) == Some(&MechanismStatus::Shifted))
}

/// Check the target initial law: a distribution over the model's states, with
/// support inside the source's, and consistent with the declared status of
/// the initial-state mechanism.
pub(crate) fn check_target_initial(
    model: &FiniteSequentialModel,
    target: &[f64],
    initial_shifted: bool,
    axis: Axis,
) -> SequentialResult<()> {
    let states = model.states.len();
    let well_formed = target.len() == states
        && target.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && (target.iter().sum::<f64>() - 1.0).abs() <= LAW_TOLERANCE;
    if !well_formed {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("target_initial_law"),
            format!(
                "the target initial law must be one probability per state ({states}) summing to one"
            ),
            "supply the target's initial-state law over the model's latent states",
        ));
    }
    for (state, (&mass, &source)) in target.iter().zip(&model.initial).enumerate() {
        if mass > 0.0 && source == 0.0 {
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_support_failure"),
                axis.detail("target_support_outside_source"),
                format!(
                    "state `{}` has target mass {mass} but source mass zero, so no source evidence covers the mechanisms used there",
                    model.states[state]
                ),
                "restrict the target law to the source's support, or establish the mechanisms for the states the source never reached",
            ));
        }
    }
    let differs =
        target.iter().zip(&model.initial).any(|(t, s)| (t - s).abs() > SAME_LAW_TOLERANCE);
    if differs && !initial_shifted {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("initial_law_declared_invariant"),
            "the target initial law differs from the source's, but the declaration marks the initial-state law invariant",
            "mark the initial-state law shifted, or supply the source's initial law as the target's",
        ));
    }
    Ok(())
}

/// Evaluate `policy` under the source initial law and under `target_initial`,
/// keeping every other mechanism of `model`.
///
/// The channels stay bound to the source population: that binding is the
/// invariance premise the declaration has already been checked for.
pub(crate) fn evaluate_under_initial(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    target_initial: &[f64],
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<(PolicyEvaluation, PolicyEvaluation)> {
    let source = evaluate_policy(model, policy, limits, ctx)?;
    let mut shifted = model.clone();
    shifted.initial = target_initial.to_vec();
    let target = evaluate_policy(&shifted, policy, limits, ctx)?;
    Ok((source, target))
}

/// What is asked of a transported policy evaluation.
///
/// `diagram` and `target_initial` are `Option` so that a missing contract is
/// representable and refused by name, never defaulted.
#[derive(Clone, Copy, Debug)]
pub struct TransportedPolicyRequest<'a> {
    /// The model the policy is established in (the source population).
    pub source: &'a FiniteSequentialModel,
    /// The fixed policy to evaluate.
    pub policy: &'a ContingentPolicy,
    /// Population the policy is evaluated in.
    pub target_population: &'a str,
    /// The target's initial-state law.
    pub target_initial: Option<&'a [f64]>,
    /// The declared selection diagram.
    pub diagram: Option<&'a SelectionDiagram>,
    /// Bounds on each enumeration.
    pub limits: EvaluationLimits,
}

/// The result of a transported policy evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedPolicyEvaluation {
    /// The policy's value in the target population.
    pub target_value: f64,
    /// The policy's value in the source population (the model's own).
    pub source_value: f64,
    /// The source enumeration.
    pub source: PolicyEvaluation,
    /// The target enumeration.
    pub target: PolicyEvaluation,
    /// Source population label.
    pub source_population: String,
    /// Target population label.
    pub target_population: String,
    /// Mechanisms the diagram declares shifted.
    pub shifted_mechanisms: Vec<ModelMechanism>,
    /// Identity of the declared selection diagram.
    pub diagram_identity: String,
    /// The target initial law used.
    pub target_initial: Vec<f64>,
    /// Histories enumerated across both enumerations: the observed work.
    pub nodes_visited: u64,
    /// The premises the value is conditional on.
    pub premises: TransportPremises,
    /// Canonical identity of the whole evaluation.
    pub identity: String,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

fn evaluation_identity(
    request: &TransportedPolicyRequest<'_>,
    diagram_identity: &str,
    target_initial: &[f64],
    policy_identity: &str,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.transported_policy.evaluation");
    writer
        .word(TRANSPORTED_POLICY_IDENTITY_VERSION)
        .text(&request.source.identity_hex())
        .text(policy_identity)
        .text(request.target_population)
        .text(diagram_identity)
        .reals(target_initial);
    writer.finish_hex()
}

/// Evaluate a fixed contingent policy in a target population under a
/// declared selection diagram.
///
/// # Errors
/// A missing selection diagram or target initial law; a diagram for another
/// population pair; an undeclared, duplicated or unknown mechanism; a
/// transition, observation or utility mechanism marked shifted; a target law
/// that is malformed, outside the source's support or inconsistent with the
/// declared status of the initial-state mechanism; a target equal to the
/// source; and every refusal of the underlying sequential evaluation.
pub fn evaluate_transported_policy(
    request: &TransportedPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<TransportedPolicyEvaluation> {
    let axis = Axis::Population;
    let model = request.source;
    let Some(diagram) = request.diagram else {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("selection_diagram_missing"),
            format!(
                "the policy was requested for population `{}` but no selection diagram relates it to the model's `{}`",
                request.target_population, model.population
            ),
            "supply a selection diagram that marks every mechanism of the model invariant or shifted between the source and target populations",
        ));
    };
    let Some(target_initial) = request.target_initial else {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("target_initial_law_missing"),
            format!(
                "no initial-state law was supplied for target population `{}`",
                request.target_population
            ),
            "supply the target population's initial law over the model's latent states; it is the one input the source cannot provide",
        ));
    };
    if request.target_population.trim().is_empty() {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("target_population"),
            "the target population needs a non-blank label",
            "name the population the policy is evaluated in",
        ));
    }
    if request.target_population == model.population {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("target_is_source"),
            format!("the target population `{}` is the model's own", model.population),
            "evaluate the policy in the model's own population with the sequential g-formula; there is nothing to transport",
        ));
    }
    if diagram.source_population != model.population
        || diagram.target_population != request.target_population
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("diagram_binding_mismatch"),
            format!(
                "the diagram relates `{}` to `{}`; the request is `{}` to `{}`",
                diagram.source_population,
                diagram.target_population,
                model.population,
                request.target_population
            ),
            "declare the selection diagram for the source population of the model and the requested target population",
        ));
    }
    let initial_shifted = shifted_initial_state(model, &diagram.mechanisms, axis)?;
    check_target_initial(model, target_initial, initial_shifted, axis)?;
    let (source, target) =
        evaluate_under_initial(model, request.policy, target_initial, request.limits, ctx)?;
    let diagram_identity = diagram.identity_hex();
    let mut shifted_mechanisms: Vec<ModelMechanism> = diagram
        .mechanisms
        .iter()
        .filter(|(_, status)| *status == MechanismStatus::Shifted)
        .map(|(mechanism, _)| *mechanism)
        .collect();
    shifted_mechanisms.sort();
    let identity =
        evaluation_identity(request, &diagram_identity, target_initial, &target.policy_identity);
    Ok(TransportedPolicyEvaluation {
        target_value: target.value,
        source_value: source.value,
        nodes_visited: source.nodes_visited + target.nodes_visited,
        source,
        target,
        source_population: model.population.clone(),
        target_population: request.target_population.to_owned(),
        shifted_mechanisms,
        diagram_identity,
        target_initial: target_initial.to_vec(),
        premises: TransportPremises::DECLARED,
        identity,
        inference: TRANSPORTED_POLICY_INFERENCE,
    })
}

/// Re-derive a transported evaluation from its request and compare, rather
/// than trusting the stored value.
///
/// # Errors
/// Every refusal of [`evaluate_transported_policy`], and a refusal when the
/// stored identity or either value differs bit for bit from the re-derived
/// one.
pub fn verify_transported_policy(
    evaluation: &TransportedPolicyEvaluation,
    request: &TransportedPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<()> {
    let fresh = evaluate_transported_policy(request, ctx)?;
    if fresh.identity != evaluation.identity
        || fresh.target_value.to_bits() != evaluation.target_value.to_bits()
        || fresh.source_value.to_bits() != evaluation.source_value.to_bits()
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            Axis::Population.detail("value_mismatch"),
            "a stored transported policy value does not replay from its model, policy, diagram and target law",
            "regenerate the evaluation from the request",
        ));
    }
    Ok(())
}
