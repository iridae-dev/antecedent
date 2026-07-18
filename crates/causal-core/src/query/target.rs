//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::EnvironmentId;

/// Target population for an effect query (DESIGN.md §8.2).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
/// Target population for an effect query (DESIGN.md §8.2).
pub enum TargetPopulation {
    /// All observed units.
    AllObserved,
    /// Treated units only.
    Treated,
    /// Untreated units only.
    Untreated,
    /// Environment-restricted population.
    Environment(EnvironmentId),
}

