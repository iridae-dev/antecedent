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
use self::{gpdc::*, knn::*, nystrom::*, oracle::*, symbolic::*};
#[cfg(test)]
use std::collections::{BTreeMap, HashMap};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use antecedent_core::search_budget::GpdcKernel;
#[cfg(test)]
use antecedent_core::{ExecutionContext, StreamDomain};
#[cfg(test)]
use antecedent_kernels::unbiased_index;
#[cfg(test)]
use crate::ci::block_shuffle::{add_one_pvalue, permutation_null_exceedances, query_stream_salt};
#[cfg(test)]
use crate::ci::gsquared::{
    discrete_strata, encode_categories, ensure_finite_categories, ensure_permutation_support,
};
#[cfg(test)]
use crate::ci::residualize::residual_is_uninformative;
#[cfg(test)]
use crate::ci::types::{
    CiResult, CiBatchResult, ConditionalIndependenceTest, KnnDependenceWorkspace, PreparedCiTest,
    nonparametric_permutation_count, reject_unsupported_block_size, requested_block_size,
};
#[cfg(test)]
use crate::dcor::distance_correlation;
#[cfg(test)]
use crate::error::StatsError;
#[cfg(test)]
use crate::matching::{MatchingDistance, MatchingIndex};

#[cfg(test)]
mod gpdc_tests;
#[cfg(test)]
mod knn_tests;
#[cfg(test)]
mod null_tests;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod symbolic_tests;
