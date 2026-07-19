//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::{DistributionRef, EnvironmentId};

use super::error::QueryError;

/// Portable predicate over units/rows.
///
/// [`Self::Rows`] is evaluated directly; [`Self::Named`] resolves through
/// [`super::PopulationRegistry`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PredicateExpr {
    /// Registry-named predicate resolved by callers.
    Named(Arc<str>),
    /// Explicit row indices into the bound tabular view.
    Rows(Arc<[usize]>),
}

impl PredicateExpr {
    /// Named registry predicate.
    #[must_use]
    pub fn named(id: impl Into<Arc<str>>) -> Self {
        Self::Named(id.into())
    }

    /// Explicit row subset.
    #[must_use]
    pub fn rows(rows: impl Into<Arc<[usize]>>) -> Self {
        Self::Rows(rows.into())
    }

    /// Validate predicate geometry (non-empty name / rows).
    ///
    /// # Errors
    ///
    /// Empty name or empty row set.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::Named(name) => {
                if name.is_empty() {
                    Err(QueryError::EmptyPredicateName)
                } else {
                    Ok(())
                }
            }
            Self::Rows(rows) => {
                if rows.is_empty() {
                    Err(QueryError::EmptyPopulationRows)
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Target population for an effect query.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TargetPopulation {
    /// All observed units.
    AllObserved,
    /// Treated units only.
    Treated,
    /// Untreated units only.
    Untreated,
    /// Environment-restricted population.
    Environment(EnvironmentId),
    /// Predicate-selected units ([`PredicateExpr`]).
    Predicate(PredicateExpr),
    /// Custom target distribution handle (weights via [`super::PopulationRegistry`]).
    CustomDistribution(DistributionRef),
}

impl TargetPopulation {
    /// Validate population geometry for Planned / structured variants.
    ///
    /// # Errors
    ///
    /// Empty predicate name or empty row set.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::AllObserved | Self::Treated | Self::Untreated | Self::Environment(_) => Ok(()),
            Self::Predicate(expr) => expr.validate(),
            Self::CustomDistribution(_) => Ok(()),
        }
    }
}
