//! Caller-declared mass over the members of a graph class.
//!
//! Two uses share one declaration:
//!
//! * over the completions of an incomplete temporal class (a temporal CPDAG or PAG), keyed by
//!   [`antecedent_identify::TemporalCompletionGraph::fingerprint`] or aligned with the
//!   enumerated cases;
//! * together with a graph posterior whose atoms are CPDAGs or PAGs, as the conditional measure
//!   over the members of every class atom: uniform under a named assumption
//!   ([`ClassPrior::uniform`]), or keyed by member fingerprint
//!   ([`antecedent_graph::dag_fingerprint`] of a DAG member, [`antecedent_graph::mag_fingerprint`]
//!   of a MAG member) and normalised within each class. A member's probability is then its
//!   atom's posterior weight times its conditional mass.
//!
//! Every member of a Markov equivalence class fits score-equivalent data equally well, so the
//! within-class measure is never learned from the data: it is a declaration and names its
//! assumption. It is not a graph posterior and not completion enumeration. Mechanism priors
//! stay on [`crate::BayesianConfig`].
//!
//! A class atom with at most [`ClassPrior::completion_limit`] members is enumerated exactly. A
//! larger one is refused unless a uniform measure declares a sample ([`ClassPrior::sample`]):
//! members are then drawn uniformly with replacement from a stream the seed fixes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_identify::TemporalClassEnvelope;

use crate::error::CausalError;

/// Most members of one class atom enumerated exactly when no other limit is declared.
pub const DEFAULT_CLASS_EXACT_LIMIT: usize = 32;

/// A declared draw of class members: how many members, with replacement, from which seed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassSamplingRequest {
    /// Members drawn per class atom above the exact limit, with replacement.
    pub members: usize,
    /// Seed of the draw; each class atom reads its own stream of it.
    pub seed: u64,
}

/// Structural mass over the members of a graph class.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassPrior {
    inner: ClassPriorInner,
    assumption: Option<Arc<str>>,
    exact_limit: Option<usize>,
    sampling: Option<ClassSamplingRequest>,
}

#[derive(Clone, Debug, PartialEq)]
enum ClassPriorInner {
    Ordered(Arc<[f64]>),
    Pairs(Arc<[(u64, f64)]>),
    Uniform,
}

/// Why declared class masses cannot be bound to the members of a class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ClassMassMismatch {
    /// Ordered masses name no member of a graph-posterior class.
    Ordered,
    /// A member has no declared mass.
    Missing(u64),
    /// The declared masses of one class sum to zero.
    ZeroTotal,
}

impl ClassPrior {
    fn declared(inner: ClassPriorInner) -> Self {
        Self { inner, assumption: None, exact_limit: None, sampling: None }
    }

    /// Masses keyed by member fingerprint: [`antecedent_identify::TemporalCompletionGraph::fingerprint`]
    /// for a temporal class, the member fingerprint of a graph-posterior class atom otherwise.
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
        Ok(Self::declared(ClassPriorInner::Pairs(pairs)))
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
        Ok(Self::declared(ClassPriorInner::Ordered(masses)))
    }

    /// Equal mass on every member of a class, declared under the named `assumption` (for
    /// example `"assumption.uniform_within_class"`). Within a Markov equivalence class the
    /// data cannot tell members apart, so the uniform measure is an assumption, not an
    /// estimate. For a PAG it is uniform over the maximal ancestral graphs the PAG represents,
    /// not over the latent-variable DAGs behind them.
    ///
    /// # Errors
    ///
    /// A blank assumption.
    pub fn uniform(assumption: &str) -> Result<Self, CausalError> {
        Self::declared(ClassPriorInner::Uniform).with_assumption(assumption)
    }

    /// Name the assumption the declared masses stand on.
    ///
    /// # Errors
    ///
    /// A blank assumption.
    pub fn with_assumption(mut self, assumption: &str) -> Result<Self, CausalError> {
        let assumption = assumption.trim();
        if assumption.is_empty() {
            return Err(CausalError::Compile {
                message: "class prior assumption must name the declaration it stands on".into(),
            });
        }
        self.assumption = Some(Arc::from(assumption));
        Ok(self)
    }

    /// Enumerate a class atom exactly when it has at most `limit` members (default
    /// [`DEFAULT_CLASS_EXACT_LIMIT`]).
    ///
    /// # Errors
    ///
    /// A zero limit.
    pub fn exact_limit(mut self, limit: usize) -> Result<Self, CausalError> {
        if limit == 0 {
            return Err(CausalError::Compile {
                message: "class prior exact limit must be at least one member".into(),
            });
        }
        self.exact_limit = Some(limit);
        Ok(self)
    }

    /// Above the exact limit, draw `members` members of each class atom uniformly with
    /// replacement from `seed`. Only a uniform measure can be sampled without every member's
    /// key; a keyed or ordered prior that declares a sample is refused at compile
    /// (`preference_compile.class_prior_sampling_needs_uniform`).
    ///
    /// # Errors
    ///
    /// Zero members.
    pub fn sample(mut self, members: usize, seed: u64) -> Result<Self, CausalError> {
        if members == 0 {
            return Err(CausalError::Compile {
                message: "class prior sample must draw at least one member".into(),
            });
        }
        self.sampling = Some(ClassSamplingRequest { members, seed });
        Ok(self)
    }

    /// The named assumption, when one was declared (always, for a uniform measure).
    #[must_use]
    pub fn assumption(&self) -> Option<&str> {
        self.assumption.as_deref()
    }

    /// Whether the measure is uniform within every class.
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        matches!(self.inner, ClassPriorInner::Uniform)
    }

    /// Most members of one class atom enumerated exactly.
    #[must_use]
    pub fn completion_limit(&self) -> usize {
        self.exact_limit.unwrap_or(DEFAULT_CLASS_EXACT_LIMIT)
    }

    /// The declared draw above the exact limit, if any.
    #[must_use]
    pub const fn sampling(&self) -> Option<ClassSamplingRequest> {
        self.sampling
    }

    /// Keys of a keyed prior; `None` for an ordered or uniform one.
    pub(crate) fn keys(&self) -> Option<Vec<u64>> {
        match &self.inner {
            ClassPriorInner::Pairs(pairs) => Some(pairs.iter().map(|(key, _)| *key).collect()),
            ClassPriorInner::Ordered(_) | ClassPriorInner::Uniform => None,
        }
    }

    /// Canonical identity: exact mass bits, keyed pairs sorted by fingerprint. A prior with no
    /// assumption, limit or sample has the identity it always had; otherwise the masses are
    /// wrapped with those declarations.
    pub(crate) fn identity_wire(&self) -> antecedent_io::ClassPriorIdentityWire {
        use antecedent_io::ClassPriorIdentityWire as Wire;
        let base = match &self.inner {
            ClassPriorInner::Ordered(masses) => {
                Wire::Ordered(masses.iter().map(|mass| mass.to_bits()).collect())
            }
            ClassPriorInner::Pairs(pairs) => {
                let mut pairs: Vec<(u64, u64)> =
                    pairs.iter().map(|&(key, mass)| (key, mass.to_bits())).collect();
                pairs.sort_unstable();
                Wire::Pairs(pairs)
            }
            ClassPriorInner::Uniform => Wire::Uniform,
        };
        if self.assumption.is_none() && self.exact_limit.is_none() && self.sampling.is_none() {
            return base;
        }
        Wire::Annotated {
            base: Box::new(base),
            assumption: self.assumption.as_deref().map(str::to_owned),
            exact_limit: self.exact_limit.map(|limit| limit as u64),
            sampling: self.sampling.map(|request| (request.members as u64, request.seed)),
        }
    }

    /// The conditional masses of the members of one class with the given fingerprints, in
    /// that order, summing to one: equal for a uniform measure, the declared masses normalised
    /// within the class for a keyed one. A member without a key refuses; no mass is invented.
    pub(crate) fn conditional_masses(
        &self,
        fingerprints: &[u64],
    ) -> Result<Vec<f64>, ClassMassMismatch> {
        match &self.inner {
            ClassPriorInner::Uniform => {
                let share = 1.0 / fingerprints.len().max(1) as f64;
                Ok(vec![share; fingerprints.len()])
            }
            ClassPriorInner::Ordered(_) => Err(ClassMassMismatch::Ordered),
            ClassPriorInner::Pairs(pairs) => {
                let by_key: HashMap<u64, f64> = pairs.iter().copied().collect();
                let masses = fingerprints
                    .iter()
                    .map(|key| by_key.get(key).copied().ok_or(ClassMassMismatch::Missing(*key)))
                    .collect::<Result<Vec<f64>, _>>()?;
                normalized(masses).map_err(|_| ClassMassMismatch::ZeroTotal)
            }
        }
    }

    /// Bind this prior to an identified envelope.
    ///
    /// Returned masses sum to one, avoiding scale-dependent arithmetic in consumers.
    /// Every case must have a mass. Extra pair keys refuse. Missing keys refuse;
    /// uniform fill is not invented: a uniform measure gives equal masses only because it was
    /// declared. Enumeration weights are never returned.
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
            ClassPriorInner::Uniform => {
                if cases.is_empty() {
                    return Err(non_positive_total());
                }
                Ok(vec![1.0 / cases.len() as f64; cases.len()])
            }
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
                normalized(masses.to_vec())
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
                    let mut extra: Vec<u64> = by_key.keys().copied().collect();
                    extra.sort_unstable();
                    return Err(CausalError::Compile {
                        message: format!(
                            "class prior has extra completion keys {extra:?} that are not in the \
                             identified envelope"
                        ),
                    });
                }
                normalized(out)
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

/// Masses divided through by their maximum, with the compensated sum of the scaled masses.
///
/// Scaling by the maximum keeps every term in `[0, 1]`, so neither the sum nor the later
/// division can overflow however large the finite masses are, and the compensated sum makes
/// the result independent of summation order to the last bit or two.
fn scaled_masses(masses: &[f64]) -> Option<(Vec<f64>, f64)> {
    let max = masses.iter().copied().fold(0.0_f64, f64::max);
    if !max.is_finite() || max <= 0.0 {
        return None;
    }
    let scaled: Vec<f64> = masses.iter().map(|mass| mass / max).collect();
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for &term in &scaled {
        let next = sum + term;
        compensation +=
            if sum.abs() >= term.abs() { (sum - next) + term } else { (term - next) + sum };
        sum = next;
    }
    Some((scaled, sum + compensation))
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the operation accepts owned input as part of its interface"
)]
fn normalized(masses: Vec<f64>) -> Result<Vec<f64>, CausalError> {
    let Some((mut scaled, total)) = scaled_masses(&masses) else {
        return Err(non_positive_total());
    };
    for mass in &mut scaled {
        *mass /= total;
    }
    Ok(scaled)
}

fn non_positive_total() -> CausalError {
    CausalError::Compile {
        message: "class prior total mass must be finite and strictly positive".into(),
    }
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
    let masses: Vec<f64> = masses.into_iter().collect();
    scaled_masses(&masses).map(|_| ()).ok_or_else(non_positive_total)
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
    fn huge_finite_masses_normalise_without_overflow() {
        let prior = ClassPrior::from_ordered([f64::MAX, f64::MAX]).unwrap();
        let ClassPriorInner::Ordered(masses) = &prior.inner else { unreachable!() };
        let out = normalized(masses.to_vec()).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-15 && (out[1] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn from_pairs_refuses_duplicates() {
        assert!(ClassPrior::from_pairs([(1, 0.5), (1, 0.5)]).is_err());
        assert!(ClassPrior::from_pairs([(1, 0.4), (2, 0.6)]).is_ok());
    }

    #[test]
    fn temporal_identities_are_unchanged() {
        use antecedent_io::ClassPriorIdentityWire as Wire;
        let ordered = ClassPrior::from_ordered([0.25, 0.75]).unwrap();
        assert_eq!(
            ordered.identity_wire(),
            Wire::Ordered(vec![0.25_f64.to_bits(), 0.75_f64.to_bits()])
        );
        let keyed = ClassPrior::from_pairs([(9, 0.5), (3, 0.5)]).unwrap();
        assert_eq!(
            keyed.identity_wire(),
            Wire::Pairs(vec![(3, 0.5_f64.to_bits()), (9, 0.5_f64.to_bits())])
        );
        // The encoded bytes of an unannotated prior are those of the bare masses.
        assert_eq!(
            antecedent_io::to_cbor(&keyed.identity_wire()).unwrap(),
            antecedent_io::to_cbor(&Wire::Pairs(vec![
                (3, 0.5_f64.to_bits()),
                (9, 0.5_f64.to_bits())
            ]))
            .unwrap()
        );
    }

    #[test]
    fn a_uniform_measure_names_its_assumption_and_its_draw() {
        assert!(ClassPrior::uniform("  ").is_err());
        let uniform = ClassPrior::uniform("assumption.uniform_within_class").unwrap();
        assert!(uniform.is_uniform());
        assert_eq!(uniform.assumption(), Some("assumption.uniform_within_class"));
        assert_eq!(uniform.completion_limit(), DEFAULT_CLASS_EXACT_LIMIT);
        assert!(uniform.clone().exact_limit(0).is_err());
        assert!(uniform.clone().sample(0, 1).is_err());
        let drawn = uniform.clone().exact_limit(8).unwrap().sample(200, 7).unwrap();
        assert_eq!(drawn.completion_limit(), 8);
        assert_eq!(drawn.sampling(), Some(ClassSamplingRequest { members: 200, seed: 7 }));
        assert_ne!(drawn.identity_wire(), uniform.identity_wire());
        let reseeded = uniform.exact_limit(8).unwrap().sample(200, 8).unwrap();
        assert_ne!(drawn.identity_wire(), reseeded.identity_wire());
    }

    #[test]
    fn conditional_masses_are_uniform_or_keyed_within_one_class() {
        let uniform = ClassPrior::uniform("assumption.uniform_within_class").unwrap();
        assert_eq!(uniform.conditional_masses(&[4, 5, 6]).unwrap(), vec![1.0 / 3.0; 3]);
        let keyed = ClassPrior::from_pairs([(4, 1.0), (5, 3.0), (7, 2.0)]).unwrap();
        assert_eq!(keyed.conditional_masses(&[4, 5]).unwrap(), vec![0.25, 0.75]);
        assert_eq!(keyed.conditional_masses(&[4, 6]), Err(ClassMassMismatch::Missing(6)));
        let zero = ClassPrior::from_pairs([(4, 0.0), (5, 1.0)]).unwrap();
        assert_eq!(zero.conditional_masses(&[4]), Err(ClassMassMismatch::ZeroTotal));
        let ordered = ClassPrior::from_ordered([1.0]).unwrap();
        assert_eq!(ordered.conditional_masses(&[4]), Err(ClassMassMismatch::Ordered));
    }
}
