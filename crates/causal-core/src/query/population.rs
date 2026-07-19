//! Target-population resolution against tabular rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::ids::DistributionRef;

use super::error::QueryError;
use super::target::{PredicateExpr, TargetPopulation};

/// Caller-supplied bindings for named predicates and custom target distributions.
#[derive(Clone, Debug, Default)]
pub struct PopulationRegistry {
    predicates: BTreeMap<Arc<str>, Arc<[usize]>>,
    distributions: BTreeMap<u32, Arc<[f64]>>,
}

impl PopulationRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a named predicate to explicit row indexes.
    pub fn insert_predicate(&mut self, name: impl Into<Arc<str>>, rows: impl Into<Arc<[usize]>>) {
        self.predicates.insert(name.into(), rows.into());
    }

    /// Bind a custom distribution handle to non-negative row weights (length = `n`).
    pub fn insert_distribution(&mut self, id: DistributionRef, weights: impl Into<Arc<[f64]>>) {
        self.distributions.insert(id.raw(), weights.into());
    }

    /// Look up a named predicate.
    #[must_use]
    pub fn predicate(&self, name: &str) -> Option<&[usize]> {
        self.predicates.get(name).map(|r| r.as_ref())
    }

    /// Look up distribution weights.
    #[must_use]
    pub fn distribution(&self, id: DistributionRef) -> Option<&[f64]> {
        self.distributions.get(&id.raw()).map(|w| w.as_ref())
    }
}

/// Resolved population as a keep-mask and optional observation weights.
#[derive(Clone, Debug)]
pub struct PopulationSelection {
    /// Length-`n` keep mask (`true` = include).
    pub keep: Arc<[bool]>,
    /// Optional length-`n` non-negative weights (CustomDistribution).
    pub weights: Option<Arc<[f64]>>,
}

impl TargetPopulation {
    /// Resolve this population over `n` rows.
    ///
    /// `treatment` is required for [`Self::Treated`] / [`Self::Untreated`] (binary 0/1 column,
    /// length `n`). Named predicates and custom distributions require `registry`.
    ///
    /// # Errors
    ///
    /// Missing registry bindings, shape mismatches, or unknown variants.
    pub fn resolve(
        &self,
        n: usize,
        treatment: Option<&[f64]>,
        registry: Option<&PopulationRegistry>,
    ) -> Result<PopulationSelection, QueryError> {
        match self {
            Self::AllObserved => Ok(PopulationSelection {
                keep: Arc::from(vec![true; n]),
                weights: None,
            }),
            Self::Treated | Self::Untreated => {
                let t = treatment.ok_or(QueryError::PopulationNeedsTreatment)?;
                if t.len() != n {
                    return Err(QueryError::PopulationLengthMismatch {
                        expected: n,
                        actual: t.len(),
                    });
                }
                let want_treated = matches!(self, Self::Treated);
                let mut keep = vec![false; n];
                for (i, &ti) in t.iter().enumerate() {
                    let is_t = (ti - 1.0).abs() <= 1e-12;
                    let is_c = ti.abs() <= 1e-12;
                    if !is_t && !is_c {
                        return Err(QueryError::PopulationNonBinaryTreatment);
                    }
                    keep[i] = if want_treated { is_t } else { is_c };
                }
                Ok(PopulationSelection { keep: Arc::from(keep), weights: None })
            }
            Self::Environment(_) => Err(QueryError::PopulationEnvironmentUnsupported),
            Self::Predicate(expr) => {
                let rows: &[usize] = match expr {
                    PredicateExpr::Rows(rows) => rows.as_ref(),
                    PredicateExpr::Named(name) => {
                        let reg = registry.ok_or(QueryError::PopulationRegistryRequired)?;
                        reg.predicate(name.as_ref()).ok_or_else(|| {
                            QueryError::UnknownPredicateName { name: Arc::clone(name) }
                        })?
                    }
                };
                let mut keep = vec![false; n];
                for &r in rows {
                    if r >= n {
                        return Err(QueryError::PopulationRowOutOfRange { row: r, n });
                    }
                    keep[r] = true;
                }
                Ok(PopulationSelection { keep: Arc::from(keep), weights: None })
            }
            Self::CustomDistribution(id) => {
                let reg = registry.ok_or(QueryError::PopulationRegistryRequired)?;
                let weights = reg.distribution(*id).ok_or(QueryError::UnknownDistributionRef {
                    id: id.raw(),
                })?;
                if weights.len() != n {
                    return Err(QueryError::PopulationLengthMismatch {
                        expected: n,
                        actual: weights.len(),
                    });
                }
                if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                    return Err(QueryError::InvalidPopulationWeights);
                }
                Ok(PopulationSelection {
                    keep: Arc::from(vec![true; n]),
                    weights: Some(Arc::from(weights.to_vec())),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::DistributionRef;

    #[test]
    fn resolves_rows_and_named_and_weights() {
        let mut reg = PopulationRegistry::new();
        reg.insert_predicate("cohort", [0usize, 2]);
        reg.insert_distribution(DistributionRef::from_raw(1), [0.5, 0.0, 1.5]);

        let rows = TargetPopulation::Predicate(PredicateExpr::rows([1usize]))
            .resolve(3, None, None)
            .unwrap();
        assert_eq!(rows.keep.as_ref(), &[false, true, false]);

        let named = TargetPopulation::Predicate(PredicateExpr::named("cohort"))
            .resolve(3, None, Some(&reg))
            .unwrap();
        assert_eq!(named.keep.as_ref(), &[true, false, true]);

        let w = TargetPopulation::CustomDistribution(DistributionRef::from_raw(1))
            .resolve(3, None, Some(&reg))
            .unwrap();
        assert_eq!(w.weights.as_ref().unwrap().as_ref(), &[0.5, 0.0, 1.5]);
    }
}
