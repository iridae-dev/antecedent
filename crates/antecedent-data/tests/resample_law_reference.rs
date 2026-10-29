//! Resampling plans against their exact output laws.
//!
//! Each reference below is written from the definition of the scheme and enumerates the
//! complete output law of a small series (every sequence of block starts, every cluster
//! draw, every permutation, the Markov chain of the stationary bootstrap), or uses the exact
//! Dirichlet marginal for the Bayesian bootstrap. The shipped fills are then run over
//! thousands of per-replicate streams and compared with a chi-square (cells) or
//! Kolmogorov-Smirnov (continuous) criterion that rejects at a level near 1e-9, so a
//! seeded run is deterministic and a wrong law fails by a wide margin.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::resample::fill_resample_weight_replicate;
use antecedent_data::{
    PermutationScheme, ResamplingPlan, fill_resample_index_replicate,
};

fn draws(
    plan: ResamplingPlan,
    n: usize,
    clusters: Option<&[u32]>,
    replicates: u64,
    seed: u64,
) -> BTreeMap<Vec<u64>, u64> {
    let ctx = ExecutionContext::for_tests(seed);
    let mut counts = BTreeMap::new();
    let mut out = Vec::new();
    for r in 0..replicates {
        fill_resample_index_replicate(plan, n, clusters, &ctx, 0x5EED, r, &mut out).unwrap();
        let key: Vec<u64> = out.iter().map(|id| id.get()).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// Pearson chi-square of observed counts against `expected` (sequence -> probability);
/// every observed sequence must lie in the reference support. The cutoff is
/// `df + 7 sqrt(2 df) + 30`, far beyond the 1e-9 tail of chi-square for these degrees of freedom.
fn assert_law(
    label: &str,
    observed: &BTreeMap<Vec<u64>, u64>,
    expected: &BTreeMap<Vec<u64>, f64>,
    replicates: u64,
) {
    let total: f64 = expected.values().sum();
    assert!((total - 1.0).abs() < 1e-12, "{label}: reference law sums to {total}");
    for key in observed.keys() {
        assert!(expected.contains_key(key), "{label}: sequence {key:?} is outside the support");
    }
    let n = replicates as f64;
    // Cells expected below five draws are pooled into one bin so the chi-square is valid.
    let (mut chi, mut cells) = (0.0, 0usize);
    let (mut pooled_p, mut pooled_o) = (0.0, 0.0);
    for (key, p) in expected {
        let o = observed.get(key).copied().unwrap_or(0) as f64;
        if p * n < 5.0 {
            pooled_p += p;
            pooled_o += o;
        } else {
            chi += (o - p * n) * (o - p * n) / (p * n);
            cells += 1;
        }
    }
    if pooled_p > 0.0 {
        let e = pooled_p * n;
        if e >= 5.0 {
            chi += (pooled_o - e) * (pooled_o - e) / e;
            cells += 1;
        }
    }
    let df = (cells - 1) as f64;
    let cutoff = df + 7.0 * (2.0 * df).sqrt() + 30.0;
    assert!(chi < cutoff, "{label}: chi-square {chi} exceeds {cutoff} on {df} df");
}

/// Reference: concatenate blocks `[s, s+length)` (indices taken modulo `n` when circular),
/// one start per block drawn uniformly, truncating the last block at `n` rows.
fn block_law(n: usize, length: usize, circular: bool) -> BTreeMap<Vec<u64>, f64> {
    let starts = if circular { n } else { n - length + 1 };
    let blocks = n.div_ceil(length);
    let mut law = BTreeMap::new();
    let total = starts.pow(u32::try_from(blocks).unwrap());
    for code in 0..total {
        let mut rest = code;
        let mut seq = Vec::new();
        for _ in 0..blocks {
            let s = rest % starts;
            rest /= starts;
            for k in 0..length {
                if seq.len() < n {
                    seq.push(u64::try_from(if circular { (s + k) % n } else { s + k }).unwrap());
                }
            }
        }
        *law.entry(seq).or_insert(0.0) += 1.0 / total as f64;
    }
    law
}

#[test]
fn moving_and_circular_block_laws_match_the_enumerated_block_start_laws() {
    for (n, length, circular) in [
        (5usize, 2usize, false),
        (5, 3, false),
        (5, 2, true),
        (5, 3, true),
        (4, 4, false),
        (4, 4, true),
        (4, 1, false),
        (3, 2, true),
        (6, 5, false),
    ] {
        let plan = if circular {
            ResamplingPlan::CircularBlock { length }
        } else {
            ResamplingPlan::MovingBlock { length }
        };
        let law = block_law(n, length, circular);
        let replicates = (law.len() as u64 * 60).max(4000);
        let observed = draws(plan, n, None, replicates, 11);
        assert_law(&format!("n={n} length={length} circular={circular}"), &observed, &law, replicates);
    }
}

#[test]
fn a_moving_block_as_long_as_the_series_is_the_identity() {
    let observed = draws(ResamplingPlan::MovingBlock { length: 7 }, 7, None, 50, 3);
    assert_eq!(observed.len(), 1);
    assert_eq!(observed.keys().next().unwrap(), &(0..7).collect::<Vec<u64>>());
}

/// Reference Markov chain of the stationary bootstrap (Politis and Romano 1994): the first
/// index is uniform; each later index continues the current block with probability `1 - p`
/// (the next row, wrapping) and otherwise restarts uniformly, `p = 1 / expected_length`.
fn stationary_law(n: usize, expected_length: f64) -> BTreeMap<Vec<u64>, f64> {
    let p = 1.0 / expected_length;
    let mut law = BTreeMap::new();
    let total = n.pow(u32::try_from(n).unwrap());
    for code in 0..total {
        let mut rest = code;
        let mut seq = Vec::new();
        for _ in 0..n {
            seq.push(rest % n);
            rest /= n;
        }
        let mut prob = 1.0 / n as f64;
        for w in seq.windows(2) {
            let cont = if w[1] == (w[0] + 1) % n { 1.0 - p } else { 0.0 };
            prob *= cont + p / n as f64;
        }
        law.insert(seq.iter().map(|&i| u64::try_from(i).unwrap()).collect(), prob);
    }
    law
}

#[test]
fn stationary_bootstrap_follows_the_geometric_block_chain_exactly() {
    for (n, expected_length) in [(4usize, 2.0f64), (3, 3.0), (4, 100.0), (4, 1.0), (3, 1.25)] {
        let law = stationary_law(n, expected_length);
        let replicates = 400_000;
        let observed =
            draws(ResamplingPlan::StationaryBlock { expected_length }, n, None, replicates, 5);
        // A one-row expected block is an iid draw; cells must be thick enough for a chi-square.
        assert_law(&format!("stationary n={n} L={expected_length}"), &observed, &law, replicates);
    }
}

#[test]
fn stationary_bootstrap_refuses_an_expected_length_below_one_or_not_finite() {
    let ctx = ExecutionContext::for_tests(1);
    let mut out = Vec::new();
    for bad in [0.0, 0.999, -3.0, f64::NAN, f64::INFINITY] {
        let plan = ResamplingPlan::StationaryBlock { expected_length: bad };
        let text = fill_resample_index_replicate(plan, 5, None, &ctx, 1, 0, &mut out)
            .expect_err(&format!("expected_length {bad} was accepted"))
            .to_string();
        assert!(
            text.contains("stationary expected_length") && text.contains("at least 1"),
            "expected_length {bad}: {text}"
        );
    }
    // The boundary is admitted: an expected block of one row is an iid draw of full length.
    let plan = ResamplingPlan::StationaryBlock { expected_length: 1.0 };
    fill_resample_index_replicate(plan, 5, None, &ctx, 1, 0, &mut out).unwrap();
    assert_eq!(out.len(), 5);
}

#[test]
fn iid_and_full_permutation_laws_are_uniform() {
    let n = 3usize;
    let mut iid = BTreeMap::new();
    for code in 0..n.pow(3) {
        let seq: Vec<u64> = (0..3).map(|k| u64::try_from(code / n.pow(k) % n).unwrap()).collect();
        iid.insert(seq, 1.0 / 27.0);
    }
    assert_law("iid", &draws(ResamplingPlan::IidBootstrap, 3, None, 40_000, 2), &iid, 40_000);

    let mut perms = BTreeMap::new();
    for a in 0..4u64 {
        for b in 0..4u64 {
            for c in 0..4u64 {
                for d in 0..4u64 {
                    let mut s = [a, b, c, d];
                    s.sort_unstable();
                    if s == [0, 1, 2, 3] {
                        perms.insert(vec![a, b, c, d], 1.0 / 24.0);
                    }
                }
            }
        }
    }
    assert_eq!(perms.len(), 24);
    let plan = ResamplingPlan::Permutation(PermutationScheme::Full);
    assert_law("full permutation", &draws(plan, 4, None, 48_000, 9), &perms, 48_000);
    assert_eq!(draws(plan, 1, None, 5, 9).keys().next().unwrap(), &vec![0u64]);
}

#[test]
fn cluster_bootstrap_draws_g_whole_clusters_uniformly_with_label_order_irrelevant() {
    // Labels are unsorted and non-contiguous; the clusters are {3: rows 1,3}, {5: row 4},
    // {7: rows 0,2} and each of the G = 3 draws picks one of them with probability 1/3.
    let ids = [7u32, 3, 7, 3, 5];
    let clusters: [&[u64]; 3] = [&[1, 3], &[4], &[0, 2]];
    let mut law: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let mut seq = Vec::new();
                for k in [a, b, c] {
                    seq.extend_from_slice(clusters[k]);
                }
                *law.entry(seq).or_insert(0.0) += 1.0 / 27.0;
            }
        }
    }
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    let observed = draws(plan, 5, Some(&ids), 27_000, 4);
    assert_law("cluster bootstrap", &observed, &law, 27_000);
}

#[test]
fn a_single_cluster_bootstrap_returns_that_cluster() {
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    let observed = draws(plan, 4, Some(&[9, 9, 9, 9]), 20, 4);
    assert_eq!(observed.len(), 1);
    assert_eq!(observed.keys().next().unwrap(), &vec![0u64, 1, 2, 3]);
}

#[test]
fn within_cluster_permutation_shuffles_each_cluster_among_its_own_slots() {
    let ids = [7u32, 3, 7, 3, 5, 7];
    // Slots of 7 are {0, 2, 5}, of 3 are {1, 3}, of 5 is {4}: 3! * 2! equally likely outputs.
    let mut law: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
    let seven = [[0u64, 2, 5], [0, 5, 2], [2, 0, 5], [2, 5, 0], [5, 0, 2], [5, 2, 0]];
    let three = [[1u64, 3], [3, 1]];
    for s in seven {
        for t in three {
            law.insert(vec![s[0], t[0], s[1], t[1], 4, s[2]], 1.0 / 12.0);
        }
    }
    let plan = ResamplingPlan::Permutation(PermutationScheme::WithinCluster);
    assert_law("within-cluster permutation", &draws(plan, 6, Some(&ids), 24_000, 8), &law, 24_000);
}

fn ks_statistic(sorted: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    let n = sorted.len() as f64;
    let mut worst = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        worst = worst.max((f - i as f64 / n).abs()).max(((i + 1) as f64 / n - f).abs());
    }
    worst
}

#[test]
fn bayesian_bootstrap_weights_follow_the_scaled_dirichlet_marginals() {
    let ctx = ExecutionContext::for_tests(21);
    for n in [2usize, 3, 5, 40] {
        let replicates = 20_000u64;
        let mut first = Vec::new();
        let (mut m, mut v, mut cov) = (0.0, 0.0, 0.0);
        let mut out = Vec::new();
        for r in 0..replicates {
            fill_resample_weight_replicate(
                ResamplingPlan::BayesianBootstrap,
                n,
                &ctx,
                0xB00,
                r,
                &mut out,
            )
            .unwrap();
            let sum: f64 = out.iter().sum();
            assert!((sum - n as f64).abs() < 1e-9 * n as f64, "weights sum to {sum}");
            assert!(out.iter().all(|w| *w > 0.0));
            first.push(out[0] / n as f64);
            m += out[0];
            v += (out[0] - 1.0) * (out[0] - 1.0);
            cov += (out[0] - 1.0) * (out[1] - 1.0);
        }
        let r = replicates as f64;
        // Dirichlet(1,..,1) marginal of one coordinate is Beta(1, n - 1): cdf 1 - (1 - x)^(n-1).
        first.sort_by(f64::total_cmp);
        let ks = ks_statistic(&first, |x| 1.0 - (1.0 - x).powi(i32::try_from(n - 1).unwrap()));
        assert!(ks < 2.6 / r.sqrt(), "n={n}: KS {ks} against Beta(1, n-1)");
        let nf = n as f64;
        // n * Dirichlet: mean 1, variance (n - 1)/(n + 1), covariance -1/(n + 1).
        assert!((m / r - 1.0).abs() < 6.0 * ((nf - 1.0) / (nf + 1.0) / r).sqrt() + 1e-12);
        assert!((v / r - (nf - 1.0) / (nf + 1.0)).abs() < 0.05, "n={n}: variance {}", v / r);
        assert!((cov / r + 1.0 / (nf + 1.0)).abs() < 0.03, "n={n}: covariance {}", cov / r);
    }
    // One observation carries all the mass n = 1.
    let mut out = Vec::new();
    fill_resample_weight_replicate(ResamplingPlan::BayesianBootstrap, 1, &ctx, 1, 0, &mut out)
        .unwrap();
    assert_eq!(out, vec![1.0]);
}
