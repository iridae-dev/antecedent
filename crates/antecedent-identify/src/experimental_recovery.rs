//! Minimum experimental recovery recipe for fully observed linear Gaussian systems.
//!
//! This is the Hyttinen–Eberhardt–Hoyer multi-experiment specification for a
//! declared set of randomized clamp experiments. The design judge accepts or
//! refuses the design. It does not read a sample or assemble a mechanism matrix.
//! Numeric direct effects come only from an assembled constraint block, and only
//! for directions that block's column rank fixes.
//!
//! The system is `x = Kx + ε` on a fully observed node set, with `K[into, from]`
//! the direct mechanism of `from` into `into` and a zero diagonal (canonical
//! model). A clamp experiment partitions the nodes into intervened `J` and
//! passive `U` and replaces each clamped equation with an exogenous assignment.
//!
//! The recipe, in order:
//!
//! - **Exogeneity.** The clamp is a randomized surgical intervention, uncorrelated
//!   with the disturbances. Conditioning on the factual disturbance is not a clamp.
//! - **Disturbances and regimes.** Regimes are independent, and the noise law is
//!   the fully observed independent Gaussian law (diagonal disturbance covariance).
//!   Correlated disturbances are a latent-variable expansion.
//! - **Weak stability.** `I − U K` is invertible on every experiment in the set.
//!   That invertibility is not physical dynamical stability (eigenvalues of `U K`
//!   inside the unit disk, or a contraction bound such as `ρ(|K|) < 1`).
//! - **Coverage.** The pair condition: for ordered pair `(from, into)`, some
//!   experiment clamps `from` while `into` stays passive. If any source into a
//!   target fails it, the whole incoming row is one unresolved direction.
//! - **Rank.** When every pair into a target holds, generic experimental effects
//!   give that row full column rank. A measure-zero cancellation of those effects
//!   can still drop the rank, so a closed report is generic identification, not a
//!   verified rank. A declared column rank below the row width names the unresolved
//!   directions and leaves the recovery goal open. The Moore-Penrose completion of
//!   that block is not applied. That declaration is not the column rank of an
//!   assembled constraint matrix. The matrix rank, its null directions, and any
//!   direct effects are read from the matrix. [`DeclaredEquationRank`] and
//!   [`WeakStability::InvertibilityDeclared`] do not by themselves return
//!   coefficients, and a caller-asserted full rank does not fill a free direction
//!   of a rank-deficient matrix. A rank declared on a target the pair condition
//!   does not cover is a warning on a partial report. It does not refuse the
//!   design and it does not solve that row.
//! - **Finite-sample conditioning.** A declared sample rank or finite condition
//!   number of the constraint matrix is not the population rank. Sampling error
//!   can make an underdetermined system look full rank (Hyttinen, Eberhardt, and
//!   Hoyer, JMLR 2012, Section 4.2). That declaration does not recover a
//!   coordinate, and it does not open a row the population rank already closes.
//! - **Uniqueness restrictions.** A minimum-norm completion, a background zero,
//!   or faithfulness selects one matrix in an open solution set (Section 4.1,
//!   step 4, and Section 4.2). The selection is not applied. It is not physical
//!   dynamical stability, and physical dynamical stability does not select the
//!   unresolved directions. Weak stability remains invertibility of `I − U K`
//!   (Definition 6), which is not a stability certificate either.
//!
//! Observational equilibrium regression and prior regularization are not this
//! recipe. A latent-variable expansion is an optional later attempt, recorded as
//! [`NotThisRecipe`], not as an identification verdict.
//!
//! [`RecoveredCoefficientUncertainty`] is the sampling law of a recovery, not a
//! point. Each draw keeps its coefficients and its disturbance together.
//! A failed or inadmissible draw stays in the law at its own weight.
//! Rank-aware blocks contribute [`UnresolvedDirection`]s. They do not complete `K`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_core::{NextAttempt, VariableId};

use crate::experiment_moments::{
    ConstraintColumnReport, TargetConstraintBlock, constraint_column_report,
};

/// Name of this clamp-experiment recipe.
///
/// A refused observational-equilibrium ordinary-least-squares attempt reports
/// this string as its next attempt.
pub const CLAMP_EXPERIMENT_RECIPE: &str =
    "hyttinen_eberhardt_hoyer.fully_observed_clamp_experiment";

/// Name of the optional later latent-variable expansion.
pub const LATENT_VARIABLE_EXPANSION_ATTEMPT: &str =
    "hyttinen_eberhardt_hoyer.latent_variable_expansion";

/// Name of prior or penalty regularization, which is not this recipe.
pub const PRIOR_REGULARIZATION_ATTEMPT: &str = "hyttinen_eberhardt_hoyer.prior_regularization";

mod decide;
mod draw;
#[cfg(test)]
mod tests;
mod types;

pub use decide::*;
pub use draw::*;
pub use types::*;
