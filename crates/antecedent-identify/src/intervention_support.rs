//! Intervention normalization and hard-Set extraction for identifiers.
//!
//! `Soft(constant)` and Sequence-of-Sets reduce to hard Sets for nonparametric ID.
//! Shift and `Soft(additive_shift)` are refused: X := X+δ is not do(X=δ).
//! Arbitrary Soft families and continuous Stochastic policies remain unsupported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use antecedent_core::{Intervention, InterventionSequence, StochasticPolicy, Value};

use crate::error::IdentificationError;

/// Human-readable name for an intervention variant.
#[must_use]
pub(crate) fn intervention_kind_name(intervention: &Intervention) -> &'static str {
    match intervention {
        Intervention::Set { .. } => "Set",
        Intervention::Shift { .. } => "Shift",
        Intervention::Stochastic { .. } => "Stochastic",
        Intervention::Soft { .. } => "Soft",
        Intervention::Sequence(_) => "Sequence",
        _ => "unknown",
    }
}

/// Static unsupported message naming the intervention kind.
#[must_use]
pub(crate) fn non_set_unsupported_message(kind: &'static str) -> &'static str {
    match kind {
        "Soft" => {
            "supports hard Set and Soft(constant) reductions only; \
             got Soft with an unsupported mechanism family"
        }
        "Shift" => "supports hard Set / Shift levels only; got an unsupported Shift form",
        "Stochastic" => {
            "supports hard Set and discrete Stochastic (Bernoulli / Categorical) only; \
             got an unsupported Stochastic policy (e.g. continuous Gaussian)"
        }
        "Sequence" => {
            "supports Sequence of hard Sets only; got a Sequence containing non-Set steps"
        }
        _ => {
            "supports hard Set interventions (plus Soft(constant), discrete Stochastic, \
             Sequence-of-Sets reductions) only"
        }
    }
}

/// Normalize an intervention to a hard [`Intervention::Set`] when possible.
///
/// Reductions:
/// - Soft(`constant`, `[v]`) → Set(v)
/// - Stochastic Bernoulli(0|1) → Set(0|1)
/// - Sequence of (reducible-to-)Sets on one variable → last Set
/// - Sequence of Sets on distinct variables → error here (use [`normalize_intervention_list`])
///
/// Refused (not a hard do-level):
/// - `Shift(δ)` and Soft(`additive_shift`, `[δ]`) — X := X+δ is not do(X=δ)
///
/// # Errors
///
/// Unsupported Soft family, additive shift, continuous Stochastic, empty/mixed Sequence, etc.
pub(crate) fn normalize_to_set(
    intervention: &Intervention,
) -> Result<Intervention, IdentificationError> {
    match intervention {
        Intervention::Set { .. } => Ok(intervention.clone()),
        Intervention::Soft { variable, mechanism } => {
            let family = mechanism.family_id.as_ref();
            let param = mechanism.parameters.first().copied().ok_or_else(|| {
                IdentificationError::unsupported(
                    "Soft intervention requires a non-empty parameter vector",
                )
            })?;
            if !param.is_finite() {
                return Err(IdentificationError::unsupported(
                    "Soft intervention parameter must be finite",
                ));
            }
            match family {
                "constant" => Ok(Intervention::set(*variable, Value::f64(param))),
                "additive_shift" => Err(IdentificationError::unsupported(
                    "Soft(additive_shift) is not a hard do-level; use Intervention::Set for \
                     do(X=x) or a parametric SCM for a true additive shift",
                )),
                _ => Err(IdentificationError::unsupported(non_set_unsupported_message("Soft"))),
            }
        }
        Intervention::Shift { .. } => Err(IdentificationError::unsupported(
            "Shift(X := X+δ) is not do(X=δ); nonparametric ID refuses the reduction. \
             Use Intervention::Set for a hard level, or a parametric SCM for an additive shift",
        )),
        Intervention::Stochastic { variable, policy } => match policy {
            StochasticPolicy::Bernoulli { p } => {
                if !p.is_finite() || !(0.0..=1.0).contains(p) {
                    return Err(IdentificationError::unsupported(
                        "Bernoulli p must be finite and in [0, 1]",
                    ));
                }
                // Degenerate policies collapse to Set; non-degenerate Bernoulli
                // ATE is identified via the hard unit contrast (see Auto / ID
                // callers that expand mixture diagnostics).
                if *p == 0.0 || *p == 1.0 {
                    Ok(Intervention::set(*variable, Value::f64(*p)))
                } else {
                    // Represent as Set(p) only for value extraction of a single
                    // "level" is wrong for mixture; callers should use
                    // `stochastic_bernoulli_levels` for ATE. Keep explicit error
                    // for require_set_value paths that don't expand.
                    Err(IdentificationError::unsupported(
                        "non-degenerate Stochastic Bernoulli requires mixture expansion \
                         (use AutoIdentifier / normalize_ate_pair)",
                    ))
                }
            }
            StochasticPolicy::Categorical { probs } => {
                let ones: Vec<usize> =
                    probs.iter().enumerate().filter(|(_, p)| **p == 1.0).map(|(i, _)| i).collect();
                if ones.len() == 1 && probs.iter().all(|p| *p == 0.0 || *p == 1.0) {
                    Ok(Intervention::set(*variable, Value::f64(ones[0] as f64)))
                } else {
                    Err(IdentificationError::unsupported(
                        "non-degenerate Stochastic Categorical requires mixture expansion",
                    ))
                }
            }
            StochasticPolicy::Gaussian { .. } => {
                Err(IdentificationError::unsupported(non_set_unsupported_message("Stochastic")))
            }
            _ => Err(IdentificationError::unsupported(non_set_unsupported_message("Stochastic"))),
        },
        Intervention::Sequence(seq) => normalize_sequence_to_set(seq),
        other => Err(IdentificationError::unsupported(non_set_unsupported_message(
            intervention_kind_name(other),
        ))),
    }
}

fn normalize_sequence_to_set(
    seq: &InterventionSequence,
) -> Result<Intervention, IdentificationError> {
    if seq.steps.is_empty() {
        return Err(IdentificationError::unsupported("empty Intervention::Sequence"));
    }
    let mut normalized = Vec::with_capacity(seq.steps.len());
    for step in seq.steps.iter() {
        normalized.push(normalize_to_set(&step.intervention)?);
    }
    let vars: Vec<_> = normalized
        .iter()
        .map(|iv| {
            iv.primary_variable().ok_or_else(|| {
                IdentificationError::unsupported("Sequence step missing primary variable")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = vars[0];
    if vars.iter().any(|v| *v != first) {
        return Err(IdentificationError::unsupported(
            "Sequence with multiple target variables cannot reduce to a single Set; \
             use a Distribution query with normalize_intervention_list",
        ));
    }
    // Last assignment wins (temporal overwrite on the same variable).
    Ok(normalized.pop().expect("non-empty after check"))
}

/// Normalize a list of interventions for Distribution queries (multi-do).
///
/// Sequence steps are flattened; each resulting intervention must reduce to Set.
///
/// # Errors
///
/// Any step that cannot reduce to a hard Set.
pub(crate) fn normalize_intervention_list(
    interventions: impl IntoIterator<Item = Intervention>,
) -> Result<Vec<Intervention>, IdentificationError> {
    let mut out = Vec::new();
    for intervention in interventions {
        match intervention {
            Intervention::Sequence(seq) => {
                for step in seq.steps.iter() {
                    out.push(normalize_to_set(&step.intervention)?);
                }
            }
            other => out.push(normalize_to_set(&other)?),
        }
    }
    Ok(out)
}

/// Require every intervention to reduce to a hard [`Intervention::Set`].
///
/// # Errors
///
/// First intervention that cannot be normalized.
pub(crate) fn require_hard_set_interventions<'a>(
    interventions: impl IntoIterator<Item = &'a Intervention>,
    _algorithm: &str,
) -> Result<(), IdentificationError> {
    for intervention in interventions {
        normalize_to_set(intervention)?;
    }
    Ok(())
}

/// Extract the value from a hard Set, or after reducing Soft(constant)/Sequence/degenerate Stochastic.
pub(crate) fn require_set_value(
    intervention: &Intervention,
    _algorithm: &str,
) -> Result<Value, IdentificationError> {
    match normalize_to_set(intervention)? {
        Intervention::Set { value, .. } => Ok(value),
        other => Err(IdentificationError::unsupported(non_set_unsupported_message(
            intervention_kind_name(&other),
        ))),
    }
}

/// Normalize an ATE active/control pair.
///
/// Soft(constant)/Sequence reduce to Sets. Non-degenerate Bernoulli sides
/// expand to a hard unit contrast `do(1) − do(0)` with scale `w_a − w_c` where
/// `w` is the Bernoulli success probability (or 0/1 for a hard Set level in {0,1}).
///
/// # Errors
///
/// Unsupported intervention forms.
pub(crate) fn normalize_ate_pair(
    active: &Intervention,
    control: &Intervention,
) -> Result<(Intervention, Intervention, Option<f64>), IdentificationError> {
    let a_bern = bernoulli_weight(active)?;
    let c_bern = bernoulli_weight(control)?;
    if a_bern.is_some() || c_bern.is_some() {
        let va =
            active.primary_variable().or_else(|| control.primary_variable()).ok_or_else(|| {
                IdentificationError::unsupported("Bernoulli ATE missing treatment variable")
            })?;
        let wa = match a_bern {
            Some(p) => p,
            None => set_binary_weight(active)?,
        };
        let wc = match c_bern {
            Some(p) => p,
            None => set_binary_weight(control)?,
        };
        return Ok((
            Intervention::set(va, Value::f64(1.0)),
            Intervention::set(va, Value::f64(0.0)),
            Some(wa - wc),
        ));
    }
    Ok((normalize_to_set(active)?, normalize_to_set(control)?, None))
}

fn bernoulli_weight(intervention: &Intervention) -> Result<Option<f64>, IdentificationError> {
    match intervention {
        Intervention::Stochastic { policy: StochasticPolicy::Bernoulli { p }, .. } => {
            if !p.is_finite() || !(0.0..=1.0).contains(p) {
                return Err(IdentificationError::unsupported(
                    "Bernoulli p must be finite and in [0, 1]",
                ));
            }
            Ok(Some(*p))
        }
        _ => Ok(None),
    }
}

fn set_binary_weight(intervention: &Intervention) -> Result<f64, IdentificationError> {
    let set = normalize_to_set(intervention)?;
    let Intervention::Set { value, .. } = set else {
        return Err(IdentificationError::unsupported(
            "Bernoulli mixture ATE requires the other side to be Set(0), Set(1), or Bernoulli",
        ));
    };
    let Some(v) = value.as_f64() else {
        return Err(IdentificationError::unsupported(
            "Bernoulli mixture ATE requires f64 Set levels",
        ));
    };
    if v == 0.0 || v == 1.0 {
        Ok(v)
    } else {
        Err(IdentificationError::unsupported(
            "Bernoulli mixture ATE requires Set levels in {0, 1} on the non-Bernoulli side",
        ))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Intervention, MechanismOverride, VariableId};

    use super::*;

    #[test]
    fn soft_constant_reduces_to_set() {
        let v = VariableId::from_raw(0);
        let iv = Intervention::soft(v, MechanismOverride::constant(1.5));
        let n = normalize_to_set(&iv).unwrap();
        assert_eq!(n, Intervention::set(v, Value::f64(1.5)));
    }

    #[test]
    fn shift_is_not_a_hard_set() {
        let v = VariableId::from_raw(0);
        let iv = Intervention::shift(v, Value::f64(0.25));
        let err = normalize_to_set(&iv).unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Shift"))
        );
    }

    #[test]
    fn soft_additive_shift_is_not_a_hard_set() {
        let v = VariableId::from_raw(0);
        let iv = Intervention::soft(v, MechanismOverride::named("additive_shift", vec![0.25]));
        let err = normalize_to_set(&iv).unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("additive_shift"))
        );
    }

    #[test]
    fn soft_linear_gaussian_still_rejected() {
        let v = VariableId::from_raw(0);
        let iv = Intervention::soft(v, MechanismOverride::named("linear_gaussian", vec![1.0, 0.0]));
        assert!(normalize_to_set(&iv).is_err());
    }
}
