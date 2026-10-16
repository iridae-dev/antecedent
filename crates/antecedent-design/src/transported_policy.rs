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
//! * **Bounded class.** The initial-state law may be shifted, and so may the
//!   utility when the caller supplies the target's own ([`TargetUtility`]). A
//!   shifted transition or observation mechanism is not identified by the
//!   source and is refused with the mechanism named; a shifted utility does
//!   not relax that, nor the support rule below.
//! * **Target law.** The caller supplies the target initial-state law. Its
//!   support must lie inside the source initial-state support, so every
//!   history reachable in the target is reachable in the source and every
//!   mechanism used was learned where it is used.
//! * **Target utility.** What an outcome is worth is a valuation, not a
//!   causal mechanism, so the source cannot supply the target's. A utility
//!   declared shifted is read from a [`TargetUtility`] that covers exactly the
//!   source model's actions and states with finite values and names where it
//!   comes from (a source label and a content digest). A utility declared
//!   shifted with none supplied, or supplied while declared invariant, is
//!   refused.
//! * **Value.** The transported value is the sequential g-formula under the
//!   source's invariant mechanisms, the target's initial law and, when
//!   supplied, the target's utility:
//!   `V_T(pi) = sum over histories h of P_T(h | pi) * realized_T(h)`.
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
//! the diagram, the target law and the target utility are kept as declarations
//! and the value, its identity and the source's own evaluation are recomputed.
//! A refit that moves
//! mass outside the declared invariant mechanisms, a target law outside the
//! source's support or a model of another population is refused
//! (`diagram_binding_mismatch`, `initial_law_declared_invariant` and the
//! support refusal).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{CanonicalWriter, ExecutionContext};

use crate::sequential_evaluation::{
    ContingentPolicy, EvaluationLimits, FiniteSequentialModel, MAX_LABEL_BYTES, PolicyEvaluation,
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
        selection: "the declared selection diagram is complete: selection acts only on the mechanisms marked shifted, the initial-state law and, when a target utility is supplied, the utility",
        invariance: "every mechanism marked invariant (transitions, observation channels and, unless a target utility is supplied, the utility) is the same law in the source and the target, so the source's interventional laws are the target's",
        overlap: "the target initial-state law is absolutely continuous with respect to the source's, so every target history is reachable in the source",
        sequential: SequentialAssumptions::DECLARED,
    };
}

/// Where a target utility comes from: a label naming the target-side source of
/// the valuation and the digest of its content, both declared by the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UtilityProvenance {
    /// Label of the target-side source the valuation was read from.
    pub source: String,
    /// Lowercase hexadecimal content digest (64 characters) of that source.
    pub digest: String,
}

/// The target population's own utility of one action.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionUtility {
    /// Utility deducted when the action is taken (finite, non-negative).
    pub cost: f64,
    /// Utility per latent state realized when the action is taken.
    pub immediate: Vec<f64>,
    /// Utility per latent state of the action's delayed outcome; present
    /// exactly when the source action has a delayed outcome, whose timing
    /// stays the source's.
    pub delayed: Option<Vec<f64>>,
}

/// The target population's own utility mechanism: what each action and each
/// terminal state is worth there.
///
/// It replaces the source's utilities and nothing else. It must cover the
/// source model's actions and latent states exactly (so the actions supported
/// and the states reached are the source's), with finite values, and it
/// declares its provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetUtility {
    /// Where the valuation comes from.
    pub provenance: UtilityProvenance,
    /// Utility per state when the horizon is reached without a commit.
    pub terminal: Vec<f64>,
    /// One entry per source action, in the source's action order.
    pub actions: Vec<ActionUtility>,
}

impl TargetUtility {
    /// Canonical identity of the valuation and its provenance.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.transported_policy.target_utility");
        writer
            .word(TRANSPORTED_POLICY_IDENTITY_VERSION)
            .text(&self.provenance.source)
            .text(&self.provenance.digest)
            .reals(&self.terminal)
            .word(self.actions.len() as u64);
        for action in &self.actions {
            writer.real(action.cost).reals(&action.immediate);
            match &action.delayed {
                Some(delayed) => {
                    writer.word(1).reals(delayed);
                }
                None => {
                    writer.word(0);
                }
            }
        }
        writer.finish_hex()
    }

    /// The source `model` with this valuation in place of its utilities.
    ///
    /// # Errors
    /// Missing or malformed provenance, a different number of actions or
    /// states than the model, a delayed outcome present on one side only, a
    /// non-finite value or a negative cost.
    pub(crate) fn applied_to(
        &self,
        model: &FiniteSequentialModel,
    ) -> SequentialResult<FiniteSequentialModel> {
        let provenance = &self.provenance;
        if provenance.source.trim().is_empty() || provenance.source.len() > MAX_LABEL_BYTES {
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                "transported_policy.target_utility_provenance_missing",
                "the target utility names no source for the valuation",
                "declare where the target's utility comes from: a source label and the content digest of that source",
            ));
        }
        let hex = provenance.digest.len() == 64
            && provenance.digest.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        if !hex {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "transported_policy.target_utility_digest",
                "the target utility digest must be 64 lowercase hexadecimal characters",
                "declare the content digest of the target-side valuation source",
            ));
        }
        let states = model.states.len();
        let per_state = |values: &[f64]| values.len() == states;
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite());
        let shaped = self.actions.len() == model.actions.len()
            && per_state(&self.terminal)
            && self.actions.iter().zip(&model.actions).all(|(target, source)| {
                per_state(&target.immediate)
                    && target.delayed.as_deref().map(per_state)
                        == source.delayed.as_ref().map(|_| true)
            });
        if !shaped {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "transported_policy.target_utility_support",
                format!(
                    "the target utility must give one entry per source action ({}) with one value per source state ({states}) and a delayed outcome exactly where the source has one",
                    model.actions.len()
                ),
                "declare the target's utility over the source model's own actions and latent states; the target cannot add or drop actions or states",
            ));
        }
        let values_ok = finite(&self.terminal)
            && self.actions.iter().all(|a| {
                a.cost.is_finite()
                    && a.cost >= 0.0
                    && finite(&a.immediate)
                    && a.delayed.as_deref().is_none_or(finite)
            });
        if !values_ok {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "transported_policy.target_utility_value",
                "every target utility value must be finite and every action cost non-negative",
                "declare finite utilities and non-negative costs",
            ));
        }
        let mut target = model.clone();
        target.terminal_utility.clone_from(&self.terminal);
        for (action, utility) in target.actions.iter_mut().zip(&self.actions) {
            action.utility_cost = utility.cost;
            action.immediate_utility.clone_from(&utility.immediate);
            if let (Some(delayed), Some(values)) = (&mut action.delayed, &utility.delayed) {
                delayed.utility.clone_from(values);
            }
        }
        Ok(target)
    }
}

/// Which axis of difference a route transports across; it selects the detail
/// namespace and the remedies of the shared checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Axis {
    Population,
    Period,
}

impl Axis {
    /// Pick the refusal detail for this axis; both spellings are literals so
    /// every emitted detail is visible to the promotion gate.
    pub(crate) const fn detail(
        self,
        population: &'static str,
        period: &'static str,
    ) -> &'static str {
        match self {
            Self::Population => population,
            Self::Period => period,
        }
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

/// The mechanisms a checked declaration shifts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShiftedMechanisms {
    pub(crate) initial: bool,
    pub(crate) utility: bool,
}

/// Check a declaration covers the model exactly and shifts only the initial
/// state law; return whether the initial state law is shifted.
pub(crate) fn shifted_initial_state(
    model: &FiniteSequentialModel,
    declared: &[(ModelMechanism, MechanismStatus)],
    axis: Axis,
) -> SequentialResult<bool> {
    declared_shifts(model, declared, axis, false).map(|shifted| shifted.initial)
}

/// Check a declaration covers the model exactly and shifts nothing beyond the
/// initial state law, or the utility when `utility_may_shift`; return what it
/// shifts.
pub(crate) fn declared_shifts(
    model: &FiniteSequentialModel,
    declared: &[(ModelMechanism, MechanismStatus)],
    axis: Axis,
    utility_may_shift: bool,
) -> SequentialResult<ShiftedMechanisms> {
    let required = required_mechanisms(model);
    let mut seen: BTreeMap<ModelMechanism, MechanismStatus> = BTreeMap::new();
    for &(mechanism, status) in declared {
        if !required.contains(&mechanism) {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                axis.detail(
                    "transported_policy.mechanism_unknown",
                    "temporal_policy_transport.mechanism_unknown",
                ),
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
                axis.detail(
                    "transported_policy.mechanism_declared_twice",
                    "temporal_policy_transport.mechanism_declared_twice",
                ),
                format!("the declaration gives `{}` a status twice", mechanism.name()),
                "declare each mechanism once",
            ));
        }
    }
    for mechanism in &required {
        if !seen.contains_key(mechanism) {
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                axis.detail(
                    "transported_policy.mechanism_status_missing",
                    "temporal_policy_transport.mechanism_status_missing",
                ),
                format!(
                    "the declaration does not say whether `{}` is invariant or shifted",
                    mechanism.name()
                ),
                "declare every mechanism of the model as invariant or shifted; invariance is never assumed for an undeclared mechanism",
            ));
        }
    }
    for (mechanism, status) in &seen {
        let allowed = match mechanism {
            ModelMechanism::InitialState => true,
            ModelMechanism::Utility => utility_may_shift,
            _ => false,
        };
        if *status == MechanismStatus::Shifted && !allowed {
            let (detail, remedy) = match axis {
                Axis::Population => (
                    "transported_policy.mechanism_not_invariant",
                    "the source cannot supply a transition or observation mechanism that differs in the target: establish it for the target population (a target experiment, or a model native to the target), or restrict the shift to the initial-state law and a supplied target utility",
                ),
                Axis::Period => (
                    "temporal_policy_transport.mechanism_time_varying",
                    "a mechanism that varies with the period is not carried across periods by the source: establish it for the target period, or declare it time-invariant if that is the scientific premise",
                ),
            };
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                detail,
                format!(
                    "`{}` is declared shifted; only the initial-state law{} may differ between source and target",
                    mechanism.name(),
                    if utility_may_shift { " and the utility" } else { "" }
                ),
                remedy,
            ));
        }
    }
    let shifted = |mechanism| seen.get(&mechanism) == Some(&MechanismStatus::Shifted);
    Ok(ShiftedMechanisms {
        initial: shifted(ModelMechanism::InitialState),
        utility: shifted(ModelMechanism::Utility),
    })
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
            axis.detail(
                "transported_policy.target_initial_law",
                "temporal_policy_transport.target_initial_law",
            ),
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
                axis.detail(
                    "transported_policy.target_support_outside_source",
                    "temporal_policy_transport.target_support_outside_source",
                ),
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
            axis.detail(
                "transported_policy.initial_law_declared_invariant",
                "temporal_policy_transport.initial_law_declared_invariant",
            ),
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
    evaluate_under_target(model, policy, target_initial, None, limits, ctx)
}

/// As [`evaluate_under_initial`], with the target's own utility in place of
/// the source's when one is supplied.
fn evaluate_under_target(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    target_initial: &[f64],
    target_utility: Option<&TargetUtility>,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<(PolicyEvaluation, PolicyEvaluation)> {
    let source = evaluate_policy(model, policy, limits, ctx)?;
    let mut shifted = match target_utility {
        Some(utility) => utility.applied_to(model)?,
        None => model.clone(),
    };
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
    /// The target's own utility; required exactly when the diagram marks the
    /// utility shifted.
    pub target_utility: Option<&'a TargetUtility>,
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
    /// The target utility used, when the utility is shifted.
    pub target_utility: Option<TargetUtility>,
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
    // Appended only when present, so evaluations without a target utility keep
    // their identities.
    if let Some(utility) = request.target_utility {
        writer.text(&utility.identity_hex());
    }
    writer.finish_hex()
}

/// Evaluate a fixed contingent policy in a target population under a
/// declared selection diagram.
///
/// # Errors
/// A missing selection diagram or target initial law; a diagram for another
/// population pair; an undeclared, duplicated or unknown mechanism; a
/// transition or observation mechanism marked shifted; a utility marked shifted
/// without a target utility, or a target utility with the utility marked
/// invariant; a target utility that is unsourced, malformed or over other
/// actions or states than the model's; a target law that is malformed, outside
/// the source's support or inconsistent with the declared status of the
/// initial-state mechanism; a target equal to the source; and every refusal of
/// the underlying sequential evaluation.
pub fn evaluate_transported_policy(
    request: &TransportedPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<TransportedPolicyEvaluation> {
    let axis = Axis::Population;
    let model = request.source;
    let Some(diagram) = request.diagram else {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "transported_policy.selection_diagram_missing",
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
            "transported_policy.target_initial_law_missing",
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
            "transported_policy.target_population",
            "the target population needs a non-blank label",
            "name the population the policy is evaluated in",
        ));
    }
    if request.target_population == model.population {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "transported_policy.target_is_source",
            format!("the target population `{}` is the model's own", model.population),
            "evaluate the policy in the model's own population with the sequential g-formula; there is nothing to transport",
        ));
    }
    if diagram.source_population != model.population
        || diagram.target_population != request.target_population
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "transported_policy.diagram_binding_mismatch",
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
    let shifted = declared_shifts(model, &diagram.mechanisms, axis, true)?;
    check_target_initial(model, target_initial, shifted.initial, axis)?;
    check_target_utility_declaration(request, shifted.utility)?;
    let (source, target) = evaluate_under_target(
        model,
        request.policy,
        target_initial,
        request.target_utility,
        request.limits,
        ctx,
    )?;
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
        target_utility: request.target_utility.cloned(),
        premises: TransportPremises::DECLARED,
        identity,
        inference: TRANSPORTED_POLICY_INFERENCE,
    })
}

/// The target utility must be supplied exactly when the declaration marks the utility shifted.
fn check_target_utility_declaration(
    request: &TransportedPolicyRequest<'_>,
    shifted_utility: bool,
) -> SequentialResult<()> {
    match (shifted_utility, request.target_utility) {
        (true, None) => {
            return Err(transport_refusal(
                antecedent_core::reason_code!("transport_missing_evidence"),
                "transported_policy.target_utility_missing",
                format!(
                    "the utility is declared shifted for population `{}` but no target utility was supplied",
                    request.target_population
                ),
                "supply the target population's own utility with its provenance, or mark the utility invariant",
            ));
        }
        (false, Some(_)) => {
            return Err(transport_refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "transported_policy.utility_declared_invariant",
                "a target utility was supplied, but the declaration marks the utility invariant",
                "mark the utility shifted, or drop the target utility",
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Re-derive a transported evaluation from its request and compare, rather
/// than trusting the stored value.
///
/// # Errors
/// Every refusal of [`evaluate_transported_policy`], and a refusal when the
/// stored identity, target utility or either value differs bit for bit from
/// the re-derived one.
pub fn verify_transported_policy(
    evaluation: &TransportedPolicyEvaluation,
    request: &TransportedPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<()> {
    let fresh = evaluate_transported_policy(request, ctx)?;
    if fresh.identity != evaluation.identity
        || fresh.target_utility != evaluation.target_utility
        || fresh.target_value.to_bits() != evaluation.target_value.to_bits()
        || fresh.source_value.to_bits() != evaluation.source_value.to_bits()
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "transported_policy.value_mismatch",
            "a stored transported policy value does not replay from its model, policy, diagram and target law",
            "regenerate the evaluation from the request",
        ));
    }
    Ok(())
}
