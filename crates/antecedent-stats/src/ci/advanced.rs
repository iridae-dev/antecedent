//! Oracle, kNN distance-dependence, symbolic CMI, and GPDC CI tests.
//!
//! One module per test; shared helpers are `pub(super)` in the module that owns them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod gpdc;
mod knn;
mod nystrom;
mod oracle;
mod symbolic;

pub use gpdc::{Gpdc, GpdcExactQuote};
pub use knn::{KnnDependence, MixedKnnDependence};
pub use nystrom::{GPDC_NYSTROM_MAX_INDUCING_POINTS, GpdcNystrom};
pub use oracle::OracleCi;
pub use symbolic::SymbolicCmi;

#[cfg(test)]
use self::{gpdc::*, knn::*, nystrom::*};
#[cfg(test)]
use crate::ci::block_shuffle::query_stream_salt;
#[cfg(test)]
use crate::ci::types::{ConditionalIndependenceTest, nonparametric_permutation_count};
#[cfg(test)]
use crate::error::StatsError;
#[cfg(test)]
use antecedent_core::search_budget::GpdcKernel;
#[cfg(test)]
use antecedent_core::{ExecutionContext, StreamDomain};

#[cfg(test)]
mod gpdc_tests;
#[cfg(test)]
mod knn_tests;
#[cfg(test)]
mod null_tests;
#[cfg(test)]
mod nystrom_reference_tests;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod symbolic_tests;
