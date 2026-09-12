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
