//! The intervention a response query answers, as identification can state it.
//!
//! A single-treatment `InterventionResponse` names one intervention. Identification must
//! carry that intervention, not a stand-in for it:
//!
//! - a hard level (a `Set`, or anything that reduces to one) is the mean `E[Y | do(T=t)]`;
//! - a declared `Shift` or stochastic policy (Bernoulli, Categorical, Gaussian) is a
//!   functional of the dose-response surface `E[Y | do(T=t)]` under the same adjustment set, so
//!   the estimand is the symbolic-level surface, never a treated-versus-control contrast, and
//!   the derivation and the assumption set say which policy was answered;
//! - anything else (a `Transform` or `Assign` that reads other variables, a `Soft` family
//!   other than constant or additive shift, a `Sequence` over several variables) is refused
//!   with a next attempt, because no estimand here states it.
//!
//! One classification is shared by the DAG back-door route, the CPDAG and MAG response
//! routes, and the joint response route, so no route can answer a different question than the
//! one declared.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, Intervention, StochasticPolicy, Value, VariableId,
};
use antecedent_expr::{CausalExprArena, ExprId};

use crate::error::IdentificationError;
use crate::intervention_support::normalize_to_set;
use crate::result::{DerivationTrace, IdentificationResult};

/// Derivation rule naming the intervention a response result answers.
pub const RESPONSE_INTERVENTION_RULE: &str = "response.intervention";

/// Refusal of a response intervention that identification cannot state as a hard level, a
/// shift or a stochastic policy.
pub(crate) const RESPONSE_INTERVENTION_UNSUPPORTED: &str = "response identification states a hard Set level, a numeric Shift, or a Bernoulli / Categorical / Gaussian Stochastic policy; this intervention is none of them (a Transform or Assign reading other variables, an unsupported Soft family, a multi-variable Sequence, or a non-numeric or non-finite parameter); next attempt: declare the treatment as a Set level, a Shift or a Stochastic policy, or answer the program on a parametric SCM";

/// Refusal of a shift or stochastic policy whose treatment effect is not identified by
/// adjustment on this graph.
pub(crate) const RESPONSE_POLICY_NEEDS_ADJUSTMENT: &str = "a Shift or Stochastic policy response is a functional of the dose-response surface, which is identified here only by back-door adjustment, and no adjustment set identifies this treatment's effect; general ID states hard Set levels only; next attempt: ask for hard Set levels (do(T=t)) so general ID can decide them, or declare a graph whose back-door criterion holds";

/// What a single-treatment response intervention asks for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResponseIntervention {
    /// Hard level `do(T = level)`: the mean `E[Y | do(T=level)]`.
    Level(Value),
    /// A declared shift or stochastic policy; the text states it.
    Policy(String),
}

impl ResponseIntervention {
    /// Classify one declared intervention.
    ///
    /// # Errors
    ///
    /// [`RESPONSE_INTERVENTION_UNSUPPORTED`] when identification cannot state it.
    pub(crate) fn classify(intervention: &Intervention) -> Result<Self, IdentificationError> {
        if let Ok(Intervention::Set { value, .. }) = normalize_to_set(intervention) {
            return Ok(Self::Level(value));
        }
        let unsupported = || IdentificationError::unsupported(RESPONSE_INTERVENTION_UNSUPPORTED);
        match intervention {
            Intervention::Shift { delta, .. } => {
                let delta = delta.as_f64().filter(|d| d.is_finite()).ok_or_else(unsupported)?;
                Ok(Self::Policy(shift_text(delta)))
            }
            Intervention::Soft { mechanism, .. }
                if mechanism.family_id.as_ref() == "additive_shift" =>
            {
                match mechanism.parameters.as_ref() {
                    [delta] if delta.is_finite() => Ok(Self::Policy(shift_text(*delta))),
                    _ => Err(unsupported()),
                }
            }
            Intervention::Stochastic { policy, .. } => {
                if policy_is_valid(policy) {
                    Ok(Self::Policy(format!(
                        "Stochastic {policy:?}: E[Y | do(T ~ policy)], the declared law mixed \
                         over the levels of the dose-response surface E[Y | do(T=t)]"
                    )))
                } else {
                    Err(unsupported())
                }
            }
            _ => Err(unsupported()),
        }
    }

    /// Whether the target is a declared policy rather than a hard level.
    pub(crate) fn is_policy(&self) -> bool {
        matches!(self, Self::Policy(_))
    }

    /// The intervention actually answered, for a derivation step.
    pub(crate) fn describe(&self, treatment: VariableId) -> String {
        match self {
            Self::Level(value) => format!(
                "answered intervention on {treatment}: hard level do({treatment} = {value}), \
                 E[Y | do({treatment} = {value})]"
            ),
            Self::Policy(text) => format!("answered intervention on {treatment}: {text}"),
        }
    }

    /// The back-door functional of this target under `adjustment`: the mean at the hard level,
    /// or the symbolic-level surface a policy is a functional of.
    pub(crate) fn backdoor_functional(
        &self,
        arena: &mut CausalExprArena,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
    ) -> ExprId {
        match self {
            Self::Level(value) => {
                arena.backdoor_mean(treatment, outcome, adjustment, value.clone())
            }
            Self::Policy(_) => arena.backdoor_response_surface(treatment, outcome, adjustment),
        }
    }

    /// Label a policy result with the intervention it answers and the positivity premise the
    /// policy adds (its treatment values must have support given the adjustment set).
    ///
    /// A hard level is stated by its own functional (`E[Y | do(T=t)]`), so it needs no label.
    pub(crate) fn record(
        &self,
        treatment: VariableId,
        derivation: &mut DerivationTrace,
        assumptions: &mut AssumptionSet,
    ) {
        if !self.is_policy() {
            return;
        }
        derivation.push(RESPONSE_INTERVENTION_RULE, self.describe(treatment));
        let record = AssumptionRecord {
            assumption: Assumption::Positivity,
            source: AssumptionSource::AlgorithmDefault {
                algorithm: RESPONSE_INTERVENTION_RULE.into(),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        };
        if !assumptions.entries.contains(&record) {
            assumptions.push(record);
        }
    }
}

/// Classify every declared intervention of a response, refusing the first one identification
/// cannot state.
///
/// # Errors
///
/// [`RESPONSE_INTERVENTION_UNSUPPORTED`].
pub(crate) fn classify_all(
    interventions: &[Intervention],
) -> Result<Vec<(Option<VariableId>, ResponseIntervention)>, IdentificationError> {
    interventions
        .iter()
        .map(|intervention| {
            ResponseIntervention::classify(intervention)
                .map(|answered| (intervention.primary_variable(), answered))
        })
        .collect()
}

/// Rewrite the estimands an adjustment route identified for the pair `(treatment, outcome)`
/// so they state `answered`, and record the answered intervention.
///
/// Back-door-family estimands are rebuilt with [`ResponseIntervention::backdoor_functional`]
/// over their own adjustment set. A policy is a functional of the adjusted surface only, so a
/// policy over an estimand identified some other way is refused, never left as the stand-in
/// the route computed.
///
/// # Errors
///
/// [`RESPONSE_POLICY_NEEDS_ADJUSTMENT`] when a policy meets a non-adjustment estimand.
pub(crate) fn express_on_result(
    result: &mut IdentificationResult,
    treatment: VariableId,
    outcome: VariableId,
    answered: &ResponseIntervention,
) -> Result<(), IdentificationError> {
    for estimand in &mut result.estimands {
        if estimand.method_kind().is_ok_and(antecedent_expr::EstimandMethod::is_backdoor_family) {
            estimand.functional = answered.backdoor_functional(
                &mut result.arena,
                treatment,
                outcome,
                &estimand.adjustment_set,
            );
        } else if answered.is_policy() {
            return Err(IdentificationError::unsupported(RESPONSE_POLICY_NEEDS_ADJUSTMENT));
        }
    }
    answered.record(treatment, &mut result.derivation, &mut result.required_assumptions);
    Ok(())
}

fn shift_text(delta: f64) -> String {
    format!(
        "Shift by {delta}: E[Y | do(T := T + {delta})], the factual treatment moved by {delta} \
         (a modified treatment policy), read off the dose-response surface E[Y | do(T=t)]"
    )
}

fn policy_is_valid(policy: &StochasticPolicy) -> bool {
    match policy {
        StochasticPolicy::Bernoulli { p } => p.is_finite() && (0.0..=1.0).contains(p),
        StochasticPolicy::Gaussian { mean, variance } => {
            mean.is_finite() && variance.is_finite() && *variance >= 0.0
        }
        StochasticPolicy::Categorical { probs } => {
            !probs.is_empty()
                && probs.iter().all(|p| p.is_finite() && *p >= 0.0)
                && probs.iter().sum::<f64>() > 0.0
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::MechanismOverride;

    use super::*;

    fn t() -> VariableId {
        VariableId::from_raw(0)
    }

    #[test]
    fn hard_levels_and_their_reductions_are_levels() {
        for intervention in [
            Intervention::set(t(), Value::f64(2.0)),
            Intervention::soft(t(), MechanismOverride::named("constant", [2.0])),
            Intervention::stochastic(t(), StochasticPolicy::bernoulli(1.0)),
        ] {
            assert!(
                matches!(
                    ResponseIntervention::classify(&intervention),
                    Ok(ResponseIntervention::Level(_))
                ),
                "{intervention:?}"
            );
        }
    }

    #[test]
    fn shifts_and_non_degenerate_laws_are_policies_that_say_what_they_are() {
        let shift =
            ResponseIntervention::classify(&Intervention::shift(t(), Value::f64(0.5))).unwrap();
        assert!(shift.is_policy());
        assert!(shift.describe(t()).contains("T + 0.5"), "{}", shift.describe(t()));
        for policy in [
            StochasticPolicy::bernoulli(0.3),
            StochasticPolicy::gaussian(0.5, 0.25),
            StochasticPolicy::categorical([1.0, 1.0]),
        ] {
            let answered =
                ResponseIntervention::classify(&Intervention::stochastic(t(), policy)).unwrap();
            assert!(answered.is_policy());
            assert!(answered.describe(t()).contains("Stochastic"));
        }
        let soft = Intervention::soft(t(), MechanismOverride::named("additive_shift", [1.0]));
        assert!(ResponseIntervention::classify(&soft).unwrap().is_policy());
    }

    #[test]
    fn what_no_estimand_states_is_refused_with_a_next_attempt() {
        let reads_another = antecedent_core::ValueExpr::var(VariableId::from_raw(1));
        let program = Intervention::assign(t(), reads_another);
        for intervention in [
            program,
            Intervention::shift(t(), Value::symbolic_intervention()),
            Intervention::stochastic(t(), StochasticPolicy::bernoulli(1.5)),
            Intervention::soft(t(), MechanismOverride::named("linear_gaussian", [1.0])),
        ] {
            let error = ResponseIntervention::classify(&intervention).unwrap_err().to_string();
            assert!(error.contains("next attempt"), "{intervention:?}: {error}");
        }
    }
}
