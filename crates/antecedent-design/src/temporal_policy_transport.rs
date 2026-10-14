//! Transport of a supplied contingent policy across periods.
//!
//! A policy value established for one period is carried to another period
//! under one bounded premise: the dynamic mechanisms are time-invariant over a
//! fixed horizon, and only the state the system starts the period in has
//! moved (a period shift).
//!
//! # Contract
//!
//! * **Time-invariance declaration.** For every mechanism of the model the
//!   caller declares it invariant across the two periods or shifted
//!   ([`MechanismStatus`]). Only the initial-state law may be shifted; a
//!   transition, observation or utility mechanism marked shifted is
//!   time-varying and is refused with the mechanism named. A mechanism left
//!   undeclared is refused; time invariance is never assumed.
//! * **Fixed horizon.** The declaration, the source model and the target
//!   period state the same number of decision times. A different horizon is
//!   outside the bounded route.
//! * **Compatible histories.** The target period's action and observation
//!   alphabet must be the source's: the same actions, in order, with the same
//!   observation symbols. The policy is a table keyed by histories over that
//!   alphabet and is never re-keyed implicitly.
//! * **Target state law.** The caller supplies the initial-state law of the
//!   target period; its support must lie inside the source's.
//!
//! The value is a point value; it carries no interval and no period-drift
//! model beyond the supplied initial law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};

use crate::sequential_evaluation::{
    ContingentPolicy, EvaluationLimits, FiniteSequentialModel, PolicyEvaluation,
    SequentialAssumptions, SequentialResult,
};
use crate::transported_policy::{
    Axis, MechanismStatus, ModelMechanism, TRANSPORTED_POLICY_IDENTITY_VERSION,
    TRANSPORTED_POLICY_INFERENCE, TransportPremises, check_target_initial, evaluate_under_initial,
    shifted_initial_state, transport_refusal,
};

/// The premises every temporally transported value is conditional on.
const TEMPORAL_PREMISES: TransportPremises = TransportPremises {
    selection: "the declared invariance is complete: only the mechanisms marked shifted may differ between the periods, here the initial-state law alone",
    invariance: "every mechanism marked invariant (transitions, observation channels, utility) is time-invariant over the fixed horizon, so the source period's interventional laws are the target period's",
    overlap: "the target period's initial-state law is absolutely continuous with respect to the source period's, so every target history is reachable in the source period",
    sequential: SequentialAssumptions::DECLARED,
};

/// One action of a history alphabet and the observation symbols it can emit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionAlphabet {
    /// The action's label.
    pub label: String,
    /// The labels of the observation symbols, in symbol-index order.
    pub symbols: Vec<String>,
}

/// The alphabet of observed histories: the actions and the symbols each can
/// produce.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryAlphabet {
    /// Actions in action-index order.
    pub actions: Vec<ActionAlphabet>,
}

impl HistoryAlphabet {
    /// The alphabet of a model's own histories.
    ///
    /// # Errors
    /// A malformed model, as the sequential evaluation refuses it.
    pub fn of_model(model: &FiniteSequentialModel) -> SequentialResult<Self> {
        let compiled = model.compile()?;
        Ok(Self {
            actions: compiled
                .actions
                .iter()
                .map(|action| ActionAlphabet {
                    label: action.label.clone(),
                    symbols: action.symbol_labels.clone(),
                })
                .collect(),
        })
    }
}

/// The declared invariance of the mechanisms across two periods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeInvariance {
    /// Period the model's mechanisms were established for.
    pub source_period: String,
    /// Period the policy is evaluated in.
    pub target_period: String,
    /// Number of decision times the invariance is declared over.
    pub horizon: usize,
    /// The status of each mechanism: every mechanism of the model, once.
    pub mechanisms: Vec<(ModelMechanism, MechanismStatus)>,
}

impl TimeInvariance {
    /// Canonical identity, independent of the order the mechanisms are listed
    /// in.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.temporal_policy_transport.invariance");
        writer
            .word(TRANSPORTED_POLICY_IDENTITY_VERSION)
            .text(&self.source_period)
            .text(&self.target_period)
            .word(self.horizon as u64);
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

/// What is known of the target period.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalTarget {
    /// The target period's label.
    pub period: String,
    /// Number of decision times in the target period.
    pub horizon: usize,
    /// The action and observation alphabet of the target period.
    pub alphabet: HistoryAlphabet,
}

/// What is asked of a temporal policy transport.
///
/// `invariance` and `target_initial` are `Option` so that a missing contract
/// is representable and refused by name, never defaulted.
#[derive(Clone, Copy, Debug)]
pub struct TemporalPolicyRequest<'a> {
    /// The model the policy is established in (the source period).
    pub source: &'a FiniteSequentialModel,
    /// The fixed policy to evaluate.
    pub policy: &'a ContingentPolicy,
    /// The target period.
    pub target: &'a TemporalTarget,
    /// The target period's initial-state law.
    pub target_initial: Option<&'a [f64]>,
    /// The declared time invariance of the mechanisms.
    pub invariance: Option<&'a TimeInvariance>,
    /// Bounds on each enumeration.
    pub limits: EvaluationLimits,
}

/// The result of a temporal policy transport.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalPolicyEvaluation {
    /// The policy's value in the target period.
    pub target_value: f64,
    /// The policy's value in the model's own period.
    pub source_value: f64,
    /// The source enumeration.
    pub source: PolicyEvaluation,
    /// The target enumeration.
    pub target: PolicyEvaluation,
    /// Source period label.
    pub source_period: String,
    /// Target period label.
    pub target_period: String,
    /// The fixed horizon, in decision times.
    pub horizon: usize,
    /// Identity of the declared time invariance.
    pub invariance_identity: String,
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
    request: &TemporalPolicyRequest<'_>,
    invariance_identity: &str,
    target_initial: &[f64],
    policy_identity: &str,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.temporal_policy_transport.evaluation");
    writer
        .word(TRANSPORTED_POLICY_IDENTITY_VERSION)
        .text(&request.source.identity_hex())
        .text(policy_identity)
        .text(&request.target.period)
        .word(request.target.horizon as u64)
        .text(invariance_identity)
        .reals(target_initial);
    writer.finish_hex()
}

/// Evaluate a fixed contingent policy in a target period under declared
/// time-invariant mechanisms and a fixed horizon.
///
/// # Errors
/// A missing invariance declaration or target initial law; a declaration for
/// another pair of periods; a horizon that differs between the model, the
/// declaration and the target; a target alphabet that is not the model's; an
/// undeclared, duplicated or unknown mechanism; a transition, observation or
/// utility mechanism that is time-varying; a malformed target law or one
/// outside the source's support; a target period equal to the source's; and
/// every refusal of the underlying sequential evaluation.
pub fn evaluate_temporal_policy(
    request: &TemporalPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<TemporalPolicyEvaluation> {
    let axis = Axis::Period;
    let model = request.source;
    let target = request.target;
    let Some(invariance) = request.invariance else {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("time_invariance_missing"),
            format!(
                "the policy was requested for period `{}` but no time-invariance declaration relates it to the model's `{}`",
                target.period, model.period
            ),
            "declare, for every mechanism of the model, that it is invariant across the two periods or shifted, over a stated fixed horizon",
        ));
    };
    let Some(target_initial) = request.target_initial else {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("target_initial_law_missing"),
            format!("no initial-state law was supplied for target period `{}`", target.period),
            "supply the initial-state law of the target period over the model's latent states; it is the one input the source period cannot provide",
        ));
    };
    if target.period.trim().is_empty() {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("target_period"),
            "the target period needs a non-blank label",
            "name the period the policy is evaluated in",
        ));
    }
    if target.period == model.period {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            axis.detail("target_is_source"),
            format!("the target period `{}` is the model's own", model.period),
            "evaluate the policy in the model's own period with the sequential g-formula; there is nothing to transport",
        ));
    }
    if invariance.source_period != model.period || invariance.target_period != target.period {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("invariance_binding_mismatch"),
            format!(
                "the declaration relates period `{}` to `{}`; the request is `{}` to `{}`",
                invariance.source_period, invariance.target_period, model.period, target.period
            ),
            "declare the time invariance for the model's period and the requested target period",
        ));
    }
    if invariance.horizon != model.horizon || target.horizon != model.horizon {
        return Err(transport_refusal(
            antecedent_core::reason_code!("route_not_supported"),
            axis.detail("horizon_mismatch"),
            format!(
                "the model has {} decision times, the declaration {} and the target period {}",
                model.horizon, invariance.horizon, target.horizon
            ),
            "state one fixed horizon for the model, the declaration and the target period; a horizon change is not a bounded route here",
        ));
    }
    let source_alphabet = HistoryAlphabet::of_model(model)?;
    if source_alphabet != target.alphabet {
        return Err(transport_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            axis.detail("history_alphabet_incompatible"),
            format!(
                "the target period's actions and observation symbols differ from the model's ({} actions declared, {} supplied)",
                source_alphabet.actions.len(),
                target.alphabet.actions.len()
            ),
            "establish that the target period has the same actions and observation symbols with the same meanings; a policy table is not re-keyed implicitly",
        ));
    }
    let initial_shifted = shifted_initial_state(model, &invariance.mechanisms, axis)?;
    check_target_initial(model, target_initial, initial_shifted, axis)?;
    let (source, target_eval) =
        evaluate_under_initial(model, request.policy, target_initial, request.limits, ctx)?;
    let invariance_identity = invariance.identity_hex();
    let identity = evaluation_identity(
        request,
        &invariance_identity,
        target_initial,
        &target_eval.policy_identity,
    );
    Ok(TemporalPolicyEvaluation {
        target_value: target_eval.value,
        source_value: source.value,
        nodes_visited: source.nodes_visited + target_eval.nodes_visited,
        source,
        target: target_eval,
        source_period: model.period.clone(),
        target_period: target.period.clone(),
        horizon: model.horizon,
        invariance_identity,
        target_initial: target_initial.to_vec(),
        premises: TEMPORAL_PREMISES,
        identity,
        inference: TRANSPORTED_POLICY_INFERENCE,
    })
}

/// Re-derive a temporal evaluation from its request and compare, rather than
/// trusting the stored value.
///
/// # Errors
/// Every refusal of [`evaluate_temporal_policy`], and a refusal when the
/// stored identity or either value differs bit for bit from the re-derived
/// one.
pub fn verify_temporal_policy(
    evaluation: &TemporalPolicyEvaluation,
    request: &TemporalPolicyRequest<'_>,
    ctx: &ExecutionContext,
) -> SequentialResult<()> {
    let fresh = evaluate_temporal_policy(request, ctx)?;
    if fresh.identity != evaluation.identity
        || fresh.target_value.to_bits() != evaluation.target_value.to_bits()
        || fresh.source_value.to_bits() != evaluation.source_value.to_bits()
    {
        return Err(transport_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            Axis::Period.detail("value_mismatch"),
            "a stored temporal policy value does not replay from its model, policy, declaration and target law",
            "regenerate the evaluation from the request",
        ));
    }
    Ok(())
}
