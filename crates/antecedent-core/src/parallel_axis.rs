//! Dominant parallel axis of one physical operator.
//!
//! Worker ceilings live on [`crate::AdmissionPool`]. This module names the axis
//! and does not reserve workers, spawn threads, or keep a second cap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// Dominant parallel axis of one physical operator.
///
/// Study is concurrent queries. Fold is cross-fitting. Chain is a set of
/// Markov chains. Replicate is bootstrap or resampling. Graph test is
/// conditional-independence or graph-atom work. Numerical kernel is one large
/// matrix operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParallelAxis {
    /// Concurrent queries in one study.
    Study,
    /// Cross-fitting folds.
    Fold,
    /// Markov chains.
    Chain,
    /// Bootstrap or resampling replicates.
    Replicate,
    /// Conditional-independence tests or graph atoms.
    GraphTest,
    /// One large matrix kernel.
    NumericalKernel,
}

impl ParallelAxis {
    /// Study, fold, chain, replicate, graph test, and numerical kernel.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::Study,
            Self::Fold,
            Self::Chain,
            Self::Replicate,
            Self::GraphTest,
            Self::NumericalKernel,
        ]
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Study => "study",
            Self::Fold => "fold",
            Self::Chain => "chain",
            Self::Replicate => "replicate",
            Self::GraphTest => "graph_test",
            Self::NumericalKernel => "numerical_kernel",
        }
    }
}

impl fmt::Display for ParallelAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::ParallelAxis;

    #[test]
    fn each_axis_has_one_name_and_no_second_cap() {
        let mut names = std::collections::BTreeSet::new();
        for axis in ParallelAxis::all() {
            assert!(names.insert(axis.as_str()), "{}", axis.as_str());
            assert_eq!(axis.to_string(), axis.as_str());
        }
        assert_eq!(names.len(), ParallelAxis::all().len());
    }
}
