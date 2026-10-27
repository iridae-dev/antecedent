//! Symbolic conditional mutual information test on discrete symbols.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ci::block_shuffle::{add_one_pvalue, permutation_null_exceedances, query_stream_salt};
use crate::ci::gsquared::{
    discrete_strata, encode_categories, ensure_finite_categories, ensure_permutation_support,
};
use crate::ci::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, nonparametric_permutation_count, reject_unsupported_block_size,
    requested_block_size,
};
use crate::error::StatsError;
use antecedent_core::{ExecutionContext, StreamDomain};
use std::collections::BTreeMap;

/// Symbolic CMI on already-binned symbols (G²-style on symbol pairs); each distinct finite
/// value is a symbol.
#[derive(Clone, Debug, Default)]
pub struct SymbolicCmi;

impl SymbolicCmi {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConditionalIndependenceTest for SymbolicCmi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let n = request.columns[q.x].len();
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            // NaN / ±inf would round to a finite symbol; refuse them.
            ensure_finite_categories(
                request.columns,
                [q.x, q.y].into_iter().chain(z.iter().copied()),
            )?;
            let xi = encode_categories(request.columns[q.x])?;
            let yi = encode_categories(request.columns[q.y])?;
            // Permutation p-value on Y, shuffled within Z strata so the Y–Z link is
            // preserved under H0 (an unconditional shuffle inflates type-I error when
            // Y depends on Z). Strata are invariant under the Y-only permutation.
            let strata = discrete_strata(request.columns, z, n)?;
            ensure_permutation_support(&strata, &xi, &yi)?;
            let mi = conditional_symbolic_mi(&xi, &yi, &strata);
            // The Y codes are permuted as the integer codes they are.
            let mut yi_perm = yi.clone();
            let mut rng = ctx.rng.stream_for(
                StreamDomain::StatsCi,
                0x51C_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
            );
            // As for `KnnDependence`: blocking is well defined only when the conditioning set
            // is empty, where `discrete_strata` yields a single time-ordered stratum. With
            // conditioning, the strata are Z-symbol groups scattered across time and blocking
            // is structurally impossible, so the request is refused.
            let block_size = requested_block_size(request.significance);
            if block_size > 1 && !z.is_empty() {
                reject_unsupported_block_size(request.significance, "SymbolicCmi")?;
            }
            let n_perm = nonparametric_permutation_count(request.significance);
            let null_ge = permutation_null_exceedances(
                n_perm,
                &mut yi_perm,
                &strata,
                block_size,
                &mut rng,
                |permuted| {
                    Ok::<_, StatsError>(conditional_symbolic_mi(&xi, permuted, &strata) >= mi)
                },
            )?;
            let p = add_one_pvalue(null_ge, n_perm);
            results.push(CiResult { statistic: mi, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> crate::ci::types::AttainableP {
        crate::ci::types::AttainableP::NonparametricPermutation
    }
}

/// Stratum-size-weighted mean of the per-stratum `MI(X;Y | Z = z)` over strata with at least
/// two rows. [`ensure_permutation_support`] has already guaranteed such a stratum exists.
fn conditional_symbolic_mi(xi: &[i32], yi: &[i32], strata: &[Vec<usize>]) -> f64 {
    let mut mi = 0.0;
    let mut weight = 0.0;
    for rows in strata {
        if rows.len() < 2 {
            continue;
        }
        let w = rows.len() as f64;
        mi += w * symbolic_mi_on_rows(xi, yi, rows);
        weight += w;
    }
    if weight > 0.0 { mi / weight } else { 0.0 }
}

fn symbolic_mi_on_rows(xi: &[i32], yi: &[i32], rows: &[usize]) -> f64 {
    // Ordered maps: the float sum below must not depend on the process hash seed, or the
    // `null >= mi` comparison of two mathematically equal statistics flips between runs.
    let mut joint: BTreeMap<(i32, i32), f64> = BTreeMap::new();
    let mut mx: BTreeMap<i32, f64> = BTreeMap::new();
    let mut my: BTreeMap<i32, f64> = BTreeMap::new();
    let nf = rows.len() as f64;
    for &r in rows {
        let (a, b) = (xi[r], yi[r]);
        *joint.entry((a, b)).or_default() += 1.0;
        *mx.entry(a).or_default() += 1.0;
        *my.entry(b).or_default() += 1.0;
    }
    let mut mi = 0.0;
    for ((a, b), c) in &joint {
        let pxy = c / nf;
        let px = mx[a] / nf;
        let py = my[b] / nf;
        if pxy > 0.0 && px > 0.0 && py > 0.0 {
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    mi
}
