//! Caller-supplied mass over incomplete-temporal class members.
//!
//! This is not a graph posterior and not completion enumeration. Mechanism
//! priors stay on [`crate::BayesianConfig`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_identify::TemporalClassEnvelope;

use crate::error::CausalError;

/// Structural mass over already-enumerated temporal class members.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassPrior {
    inner: ClassPriorInner,
}

#[derive(Clone, Debug, PartialEq)]
enum ClassPriorInner {
    Ordered(Arc<[f64]>),
    Pairs(Arc<[(u64, f64)]>),
}

impl ClassPrior {
    /// Masses keyed by [`antecedent_identify::TemporalCompletionGraph::fingerprint`].
    ///
    /// # Errors
    ///
    /// Empty, duplicate keys, non-finite or negative mass, or non-positive total.
    pub fn from_pairs(pairs: impl Into<Arc<[(u64, f64)]>>) -> Result<Self, CausalError> {
        let pairs = pairs.into();
        if pairs.is_empty() {
            return Err(CausalError::Compile {
                message: "class prior requires at least one completion mass".into(),
            });
        }
        let mut seen = HashMap::with_capacity(pairs.len());
        for &(key, mass) in pairs.iter() {
            validate_mass(mass)?;
            if seen.insert(key, mass).is_some() {
                return Err(CausalError::Compile {
                    message: format!("class prior has a duplicate completion key {key}"),
                });
            }
        }
        ensure_positive_total(pairs.iter().map(|(_, mass)| *mass))?;
        Ok(Self { inner: ClassPriorInner::Pairs(pairs) })
    }

    /// Masses aligned 1:1 with `envelope.cases` after identification.
    ///
    /// # Errors
    ///
    /// Empty, non-finite or negative mass, or non-positive total.
    pub fn from_ordered(masses: impl Into<Arc<[f64]>>) -> Result<Self, CausalError> {
        let masses = masses.into();
        if masses.is_empty() {
            return Err(CausalError::Compile {
                message: "class prior requires at least one completion mass".into(),
            });
        }
        for &mass in masses.iter() {
            validate_mass(mass)?;
        }
        ensure_positive_total(masses.iter().copied())?;
        Ok(Self { inner: ClassPriorInner::Ordered(masses) })
    }

    /// Canonical identity: exact mass bits, keyed pairs sorted by fingerprint.
    pub(crate) fn identity_wire(&self) -> antecedent_io::ClassPriorIdentityWire {
        match &self.inner {
            ClassPriorInner::Ordered(masses) => antecedent_io::ClassPriorIdentityWire::Ordered(
                masses.iter().map(|mass| mass.to_bits()).collect(),
            ),
            ClassPriorInner::Pairs(pairs) => {
                let mut pairs: Vec<(u64, u64)> =
                    pairs.iter().map(|&(key, mass)| (key, mass.to_bits())).collect();
                pairs.sort_unstable();
                antecedent_io::ClassPriorIdentityWire::Pairs(pairs)
            }
        }
    }

    /// Bind this prior to an identified envelope.
    ///
    /// Returned masses sum to one, avoiding scale-dependent arithmetic in consumers.
    /// Every case must have a mass. Extra pair keys refuse. Missing keys refuse;
    /// uniform fill is not invented. Enumeration weights are never returned.
    ///
    /// # Errors
    ///
    /// Length or key mismatch against the envelope.
    pub fn masses_for_envelope(
        &self,
        envelope: &TemporalClassEnvelope,
    ) -> Result<Vec<f64>, CausalError> {
        let cases = &envelope.envelope.cases;
        match &self.inner {
            ClassPriorInner::Ordered(masses) => {
                if masses.len() != cases.len() {
                    return Err(CausalError::Compile {
                        message: format!(
                            "class prior has {} masses but the temporal class envelope has {} \
                             completions; refuse missing or extra keys rather than filling uniform \
                             enumeration weights",
                            masses.len(),
                            cases.len()
                        ),
                    });
                }
                Ok(normalized(masses.to_vec()))
            }
            ClassPriorInner::Pairs(pairs) => {
                let mut by_key: HashMap<u64, f64> = pairs.iter().copied().collect();
                let mut out = Vec::with_capacity(cases.len());
                for case in cases {
                    let key = case.graph.fingerprint();
                    let Some(mass) = by_key.remove(&key) else {
                        return Err(CausalError::Compile {
                            message: format!(
                                "class prior is missing completion key {key}; refuse filling \
                                 enumeration weights"
                            ),
                        });
                    };
                    out.push(mass);
                }
                if !by_key.is_empty() {
                    let extra: Vec<u64> = by_key.keys().copied().collect();
                    return Err(CausalError::Compile {
                        message: format!(
                            "class prior has extra completion keys {extra:?} that are not in the \
                             identified envelope"
                        ),
                    });
                }
                Ok(normalized(out))
            }
        }
    }
}

/// A class prior bound to one temporal envelope and reused across horizons.
///
/// Multi-horizon analyses identify each horizon separately, and a PAG window
/// can retain a different completion set or order per horizon. Binding once by
/// completion fingerprint keeps ordered masses attached to the completions they
/// were declared for instead of re-reading them positionally.
#[derive(Clone, Debug)]
pub(crate) struct ClassPriorBinding {
    by_fingerprint: HashMap<u64, f64>,
}

impl ClassPriorBinding {
    /// Bind `prior` to the completions of `envelope`.
    ///
    /// # Errors
    ///
    /// Any [`ClassPrior::masses_for_envelope`] refusal, or duplicate fingerprints.
    pub(crate) fn bind(
        prior: &ClassPrior,
        envelope: &TemporalClassEnvelope,
    ) -> Result<Self, CausalError> {
        let masses = prior.masses_for_envelope(envelope)?;
        let mut by_fingerprint = HashMap::with_capacity(masses.len());
        for (case, mass) in envelope.envelope.cases.iter().zip(masses) {
            if by_fingerprint.insert(case.graph.fingerprint(), mass).is_some() {
                return Err(CausalError::Compile {
                    message: "temporal class envelope repeats a completion fingerprint; a \
                              class prior cannot be bound unambiguously"
                        .into(),
                });
            }
        }
        Ok(Self { by_fingerprint })
    }

    /// Masses in `envelope` case order, summing to one.
    ///
    /// # Errors
    ///
    /// The horizon enumerated a different completion set than the one bound.
    pub(crate) fn masses_for_envelope(
        &self,
        envelope: &TemporalClassEnvelope,
    ) -> Result<Vec<f64>, CausalError> {
        let cases = &envelope.envelope.cases;
        if cases.len() != self.by_fingerprint.len() {
            return Err(CausalError::Compile {
                message: format!(
                    "class prior was bound to {} completions but this horizon enumerated {}; \
                     class membership differs across horizons, so one class prior cannot be \
                     applied to every horizon",
                    self.by_fingerprint.len(),
                    cases.len()
                ),
            });
        }
        cases
            .iter()
            .map(|case| {
                let key = case.graph.fingerprint();
                self.by_fingerprint.get(&key).copied().ok_or_else(|| CausalError::Compile {
                    message: format!(
                        "class prior has no mass for completion {key} enumerated at this \
                         horizon; class membership differs across horizons"
                    ),
                })
            })
            .collect()
    }
}

fn normalized(mut masses: Vec<f64>) -> Vec<f64> {
    let total: f64 = masses.iter().sum();
    for mass in &mut masses {
        *mass /= total;
    }
    masses
}

fn validate_mass(mass: f64) -> Result<(), CausalError> {
    if !mass.is_finite() || mass < 0.0 {
        return Err(CausalError::Compile {
            message: format!("class prior mass must be finite and nonnegative, got {mass}"),
        });
    }
    Ok(())
}

fn ensure_positive_total(masses: impl IntoIterator<Item = f64>) -> Result<(), CausalError> {
    let total: f64 = masses.into_iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(CausalError::Compile {
            message: "class prior total mass must be finite and strictly positive".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_ordered_refuses_empty_and_negative() {
        assert!(ClassPrior::from_ordered([] as [f64; 0]).is_err());
        assert!(ClassPrior::from_ordered([1.0, -0.1]).is_err());
        assert!(ClassPrior::from_ordered([f64::NAN]).is_err());
        assert!(ClassPrior::from_ordered([0.0, 0.0]).is_err());
    }

    #[test]
    fn from_pairs_refuses_duplicates() {
        assert!(ClassPrior::from_pairs([(1, 0.5), (1, 0.5)]).is_err());
        assert!(ClassPrior::from_pairs([(1, 0.4), (2, 0.6)]).is_ok());
    }
}
