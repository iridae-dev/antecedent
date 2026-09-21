//! Shared permutation machinery for the CI family: block permutations, query-keyed RNG
//! streams, and the residual block null used by every partial-correlation variant.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref
)]

use antecedent_core::{CausalRng, KernelPolicy};
use antecedent_kernels::shuffle;

use super::residualize::weighted_pearson;
use crate::error::StatsError;

/// Relative slack when comparing a permuted statistic to the observed one: two arrangements
/// with the same mathematical `|r|` differ in the last bits, and an exact `>=` would count
/// one of them and drop the other.
const TIE_REL_TOL: f64 = 1e-12;

/// A stable 64-bit key for one CI query, independent of its position in a batch.
///
/// The conditioning set is an unordered set, so it is sorted before hashing; `x` and `y` are
/// hashed in the given order because the null permutes one of them. Two calls with the same
/// data, seed and query therefore draw the same permutations wherever the query sits in a
/// batch.
pub(crate) fn query_stream_salt(x: &[usize], y: &[usize], z: &[usize]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for part in [x, y] {
        mix(part.len() as u64);
        for &c in part {
            mix(c as u64);
        }
    }
    let mut zs = z.to_vec();
    zs.sort_unstable();
    mix(zs.len() as u64);
    for c in zs {
        mix(c as u64);
    }
    h
}

/// Row permutation `perm[dst] = src` that reorders `n` rows by contiguous blocks of
/// `block_size` (the last block possibly short): the blocks are Fisher–Yates shuffled with an
/// unbiased index sampler and laid out in the shuffled order. Rows never move relative to
/// their neighbours inside a block, so serial dependence within a block survives.
/// `block_size <= 1` is an ordinary element-wise permutation.
pub(crate) fn block_row_permutation(
    n: usize,
    block_size: usize,
    rng: &mut CausalRng,
    perm: &mut Vec<usize>,
) {
    let bs = block_size.max(1).min(n.max(1));
    let n_blocks = n.div_ceil(bs);
    let mut order: Vec<usize> = (0..n_blocks).collect();
    shuffle(rng, &mut order);
    perm.clear();
    for b in order {
        let start = b * bs;
        perm.extend(start..(start + bs).min(n));
    }
}

/// Permute `y` in place by contiguous blocks; see [`block_row_permutation`].
///
/// This is the block-preserving primitive behind
/// [`SignificanceMethod::BlockShuffle`](super::types::SignificanceMethod).
pub(crate) fn block_permute_contiguous(y: &mut [f64], block_size: usize, rng: &mut CausalRng) {
    let mut perm = Vec::with_capacity(y.len());
    block_row_permutation(y.len(), block_size, rng, &mut perm);
    let original = y.to_vec();
    for (dst, &src) in perm.iter().enumerate() {
        y[dst] = original[src];
    }
}

/// Block-permutation p-value for the partial correlation, from residuals.
///
/// `rx` and `ry` are X and Y residualised on the conditioning set (with the same weights as
/// `weights`, when given). The null block-permutes the X residual and recomputes the
/// (weighted) correlation with the fixed Y residual, so it is the same construction for the
/// scalar, weighted and multivariate paths. Residualising *before* permuting is what makes a
/// block-preserving null valid with a non-empty Z: permuting raw X would break X's Z-driven
/// structure that the observed residual does not share, and the null would be too dispersed
/// or too tight depending on the sign of the mismatch. With an empty Z the residuals are the
/// centred columns and this is the plain block permutation of X.
///
/// Add-one p-value `(1 + #{|r*| ≥ |r|}) / (1 + replicates)`.
///
/// # Errors
///
/// [`StatsError::Shape`] when a residual has no variance.
pub(crate) fn residual_null_pvalue(
    policy: &KernelPolicy,
    rx: &[f64],
    ry: &[f64],
    weights: Option<&[f64]>,
    replicates: u32,
    block_size: usize,
    rng: &mut CausalRng,
) -> Result<f64, StatsError> {
    let n = rx.len();
    let ones;
    let w: &[f64] = if let Some(w) = weights {
        w
    } else {
        ones = vec![1.0; n];
        &ones
    };
    let degenerate = StatsError::Shape { message: "block-shuffle null: degenerate residual" };
    let observed = weighted_pearson(policy, rx, ry, w).ok_or(degenerate)?.abs();
    let mut perm = Vec::with_capacity(n);
    let mut shuffled = vec![0.0; n];
    let mut extreme = 0u32;
    for _ in 0..replicates {
        block_row_permutation(n, block_size, rng, &mut perm);
        for (dst, &src) in perm.iter().enumerate() {
            shuffled[dst] = rx[src];
        }
        let r = weighted_pearson(policy, &shuffled, ry, w)
            .ok_or(StatsError::Shape { message: "block-shuffle null: degenerate residual" })?;
        if r.abs() >= observed * (1.0 - TIE_REL_TOL) {
            extreme += 1;
        }
    }
    Ok((f64::from(extreme) + 1.0) / (f64::from(replicates) + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::ExecutionContext;

    #[test]
    fn block_row_permutation_keeps_blocks_contiguous_and_covers_every_row() {
        let ctx = ExecutionContext::for_tests(5);
        let mut rng = ctx.rng.stream(1);
        let mut perm = Vec::new();
        for (n, bs) in [(10usize, 3usize), (12, 4), (7, 1), (5, 9)] {
            block_row_permutation(n, bs, &mut rng, &mut perm);
            let mut seen = perm.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n} bs={bs}");
            let bs_eff = bs.max(1).min(n);
            // Every block-aligned source window appears as one consecutive run.
            for b in 0..n.div_ceil(bs_eff) {
                let start = b * bs_eff;
                let end = (start + bs_eff).min(n);
                let at = perm.iter().position(|&s| s == start).unwrap();
                let run: Vec<usize> = perm[at..at + (end - start)].to_vec();
                assert_eq!(run, (start..end).collect::<Vec<_>>(), "n={n} bs={bs} block {b}");
            }
        }
    }

    #[test]
    fn query_salt_ignores_conditioning_order_but_separates_queries() {
        let a = query_stream_salt(&[0], &[1], &[4, 2, 3]);
        assert_eq!(a, query_stream_salt(&[0], &[1], &[2, 3, 4]));
        assert_ne!(a, query_stream_salt(&[0], &[1], &[2, 3]));
        assert_ne!(a, query_stream_salt(&[0], &[2], &[2, 3, 4]));
        assert_ne!(a, query_stream_salt(&[1], &[0], &[2, 3, 4]));
    }

    #[test]
    fn residual_null_matches_exact_enumeration_of_block_arrangements() {
        // n = 6, block 2 => 3 blocks => 3! = 6 arrangements. With many replicates the
        // add-one p-value converges to the exact fraction of arrangements with |r*| >= |r|.
        let rx = [1.0, -2.0, 0.5, 3.0, -1.5, -1.0];
        let ry = [0.7, 0.1, -1.2, 2.0, -0.4, -1.2];
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let corr = |a: &[f64], b: &[f64]| {
            let (ma, mb) = (mean(a), mean(b));
            let sab: f64 = a.iter().zip(b).map(|(p, q)| (p - ma) * (q - mb)).sum();
            let saa: f64 = a.iter().map(|p| (p - ma) * (p - ma)).sum();
            let sbb: f64 = b.iter().map(|q| (q - mb) * (q - mb)).sum();
            sab / (saa * sbb).sqrt()
        };
        let observed = corr(&rx, &ry).abs();
        let mut extreme = 0usize;
        for order in [[0usize, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let mut x: Vec<f64> = Vec::new();
            for b in order {
                x.extend_from_slice(&rx[2 * b..2 * b + 2]);
            }
            if corr(&x, &ry).abs() >= observed * (1.0 - 1e-12) {
                extreme += 1;
            }
        }
        let exact = extreme as f64 / 6.0;
        let ctx = ExecutionContext::for_tests(11);
        let mut rng = ctx.rng.stream(3);
        let reps = 6000u32;
        let p =
            residual_null_pvalue(&ctx.kernel_policy, &rx, &ry, None, reps, 2, &mut rng).unwrap();
        // Add-one smoothing biases by <= 1/(reps+1); Monte Carlo SE <= 0.0065.
        assert!((p - exact).abs() < 0.03, "p={p} exact={exact}");
    }
}
