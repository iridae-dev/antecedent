//! Target-population resolution against tabular rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::ids::DistributionRef;
use crate::row_mask::{RowMask, RowMaskError};

use super::error::QueryError;
use super::probability::BINARY_CODING_TOLERANCE;
use super::target::{PredicateExpr, TargetPopulation};

/// Caller-supplied bindings for named predicates and custom target distributions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PopulationRegistry {
    predicates: BTreeMap<Arc<str>, Arc<[usize]>>,
    distributions: BTreeMap<u32, Arc<[f64]>>,
    distribution_dependencies: BTreeMap<u32, Arc<[crate::VariableId]>>,
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
        self.distribution_dependencies.remove(&id.raw());
    }

    /// Bind known standardization weights and their caller-declared covariate parents.
    pub fn insert_distribution_with_dependence(
        &mut self,
        id: DistributionRef,
        weights: impl Into<Arc<[f64]>>,
        depends_on: impl Into<Arc<[crate::VariableId]>>,
    ) {
        self.distributions.insert(id.raw(), weights.into());
        self.distribution_dependencies.insert(id.raw(), depends_on.into());
    }

    /// Declared parents of known standardization weights (empty means constant weights).
    #[must_use]
    pub fn distribution_dependencies(&self, id: DistributionRef) -> Option<&[crate::VariableId]> {
        self.distribution_dependencies.get(&id.raw()).map(std::convert::AsRef::as_ref)
    }

    /// Look up a named predicate.
    #[must_use]
    pub fn predicate(&self, name: &str) -> Option<&[usize]> {
        self.predicates.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Look up distribution weights.
    #[must_use]
    pub fn distribution(&self, id: DistributionRef) -> Option<&[f64]> {
        self.distributions.get(&id.raw()).map(std::convert::AsRef::as_ref)
    }

    /// Named predicates in name order.
    pub fn predicates(&self) -> impl Iterator<Item = (&str, &[usize])> {
        self.predicates.iter().map(|(name, rows)| (name.as_ref(), rows.as_ref()))
    }

    /// Bound distributions in handle order with their declared parents
    /// (empty when none were declared).
    pub fn distributions(
        &self,
    ) -> impl Iterator<Item = (DistributionRef, &[f64], &[crate::VariableId])> {
        self.distributions.iter().map(|(&handle, weights)| {
            let id = DistributionRef::from_raw(handle);
            (id, weights.as_ref(), self.distribution_dependencies(id).unwrap_or(&[]))
        })
    }
}

/// Resolved population as a keep-mask and optional observation weights.
#[derive(Clone, Debug)]
pub struct PopulationSelection {
    /// Length-`n` bit-packed keep mask (set = include).
    pub keep: RowMask,
    /// Optional length-`n` non-negative weights (`CustomDistribution`).
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
            // A row-weight retarget keeps every row here: its weights live with
            // the retarget owner that holds the snapshot they were written for,
            // and this variant carries only their digest, so no weights can be
            // recovered from it.
            Self::AllObserved | Self::RowWeights { .. } => {
                Ok(PopulationSelection { keep: RowMask::all_set(n), weights: None })
            }
            Self::Treated | Self::Untreated => {
                let t = treatment.ok_or(QueryError::PopulationNeedsTreatment)?;
                if t.len() != n {
                    return Err(QueryError::PopulationLengthMismatch {
                        expected: n,
                        actual: t.len(),
                    });
                }
                let want_treated = matches!(self, Self::Treated);
                let is_treated = |ti: f64| (ti - 1.0).abs() <= BINARY_CODING_TOLERANCE;
                let is_control = |ti: f64| ti.abs() <= BINARY_CODING_TOLERANCE;
                if t.iter().any(|&ti| !is_treated(ti) && !is_control(ti)) {
                    return Err(QueryError::PopulationNonBinaryTreatment);
                }
                let keep = RowMask::from_fn(n, |i| {
                    if want_treated { is_treated(t[i]) } else { is_control(t[i]) }
                });
                Ok(PopulationSelection { keep, weights: None })
            }
            Self::Environment(_) => Err(QueryError::PopulationEnvironmentUnsupported),
            Self::LocalAtCutoff { .. } => Err(QueryError::PopulationLocalAtCutoffNotRows),
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
                let keep = RowMask::from_set_rows(n, rows.iter().copied()).map_err(|error| {
                    let RowMaskError::RowOutOfRange { row, .. } = error else {
                        unreachable!("building from listed rows can only fail on an out-of-range row")
                    };
                    QueryError::PopulationRowOutOfRange { row, n }
                })?;
                Ok(PopulationSelection { keep, weights: None })
            }
            Self::CustomDistribution(id) => {
                let reg = registry.ok_or(QueryError::PopulationRegistryRequired)?;
                let weights = reg
                    .distribution(*id)
                    .ok_or(QueryError::UnknownDistributionRef { id: id.raw() })?;
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
                    keep: RowMask::all_set(n),
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
    fn cutoff_population_is_a_limit_not_a_row_subset() {
        let local = TargetPopulation::local_at_cutoff(crate::VariableId::from_raw(2), -0.0);
        assert!(local.is_local_at_cutoff(crate::VariableId::from_raw(2), 0.0));
        assert!(!local.is_local_at_cutoff(crate::VariableId::from_raw(2), 0.5));
        assert!(!local.is_local_at_cutoff(crate::VariableId::from_raw(1), 0.0));
        assert!(local.validate().is_ok());
        assert!(matches!(
            local.resolve(3, None, None),
            Err(QueryError::PopulationLocalAtCutoffNotRows)
        ));
        let bad = TargetPopulation::local_at_cutoff(crate::VariableId::from_raw(2), f64::INFINITY);
        assert!(matches!(bad.validate(), Err(QueryError::NonFiniteCutoff)));
    }

    #[test]
    fn resolves_rows_and_named_and_weights() {
        let mut reg = PopulationRegistry::new();
        reg.insert_predicate("cohort", [0usize, 2]);
        reg.insert_distribution(DistributionRef::from_raw(1), [0.5, 0.0, 1.5]);

        let rows = TargetPopulation::Predicate(PredicateExpr::rows([1usize]))
            .resolve(3, None, None)
            .unwrap();
        assert_eq!(rows.keep.to_bools(), vec![false, true, false]);

        let named = TargetPopulation::Predicate(PredicateExpr::named("cohort"))
            .resolve(3, None, Some(&reg))
            .unwrap();
        assert_eq!(named.keep.to_bools(), vec![true, false, true]);

        let w = TargetPopulation::CustomDistribution(DistributionRef::from_raw(1))
            .resolve(3, None, Some(&reg))
            .unwrap();
        assert_eq!(w.weights.as_ref().unwrap().as_ref(), &[0.5, 0.0, 1.5]);
    }
}
