//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::{DistributionRef, EnvironmentId, VariableId};

use super::attribution::OrderedFloatBits;
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
    /// Row-weight retarget bound to one data snapshot.
    RowWeights {
        /// Target-weights identity (`antecedent.identity.target_weights.v1`):
        /// the exact weight bits in row order, their row count, the data
        /// snapshot, the score table they reweight, and `depends_on`.
        weights: [u8; 32],
        /// Covariates the weights are declared to depend on.
        depends_on: Arc<[VariableId]>,
    },
    /// Units at the threshold of a running variable: the limit population `R = c`.
    ///
    /// This is the population a sharp regression-discontinuity design speaks for. It is
    /// not a row subset: the effect over it is the difference of the one-sided limits of
    /// `E[Y | R = r]` at `c`, and it says nothing about units away from the cutoff.
    LocalAtCutoff {
        /// Running (assignment) variable `R`.
        running: VariableId,
        /// Cutoff `c` (finite; `-0.0` is stored as `0.0`).
        cutoff: OrderedFloatBits,
    },
}

impl TargetPopulation {
    /// Units at the cutoff `cutoff` of `running` ([`Self::LocalAtCutoff`]).
    #[must_use]
    pub fn local_at_cutoff(running: VariableId, cutoff: f64) -> Self {
        // One bit pattern per cutoff, so equal designs compare and hash equal.
        let cutoff = if cutoff == 0.0 { 0.0 } else { cutoff };
        Self::LocalAtCutoff { running, cutoff: OrderedFloatBits::from_f64(cutoff) }
    }

    /// Whether this is the cutoff population of exactly this design.
    #[must_use]
    pub fn is_local_at_cutoff(&self, running: VariableId, cutoff: f64) -> bool {
        *self == Self::local_at_cutoff(running, cutoff)
    }

    /// Validate population geometry for Planned / structured variants.
    ///
    /// # Errors
    ///
    /// Empty predicate name, empty row set, or a non-finite cutoff.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::Predicate(expr) => expr.validate(),
            Self::LocalAtCutoff { cutoff, .. } => {
                if cutoff.to_f64().is_finite() {
                    Ok(())
                } else {
                    Err(QueryError::NonFiniteCutoff)
                }
            }
            Self::AllObserved
            | Self::Treated
            | Self::Untreated
            | Self::Environment(_)
            | Self::CustomDistribution(_)
            | Self::RowWeights { .. } => Ok(()),
        }
    }
}
