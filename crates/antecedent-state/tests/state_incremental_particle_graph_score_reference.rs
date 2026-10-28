//! Gaussian BIC graph score and the bootstrap particle filter against executing references.
//!
//! The score half compares the cached local score, the incremental `delta_score` and the running
//! total with `oracle_data/state_incremental_particle_graph_score_reference.py` (frozen in the
//! JSON beside it): residual sums of squares from exact rational least squares on centered dyadic
//! columns, the logarithms at 60 digits. Cases cover an outcome offset of 1e9, parents that differ
//! by 4e-6, a child fitted to noise 1e-5 of its spread, columns on scales 2^-40 and 2^40, three and
//! four rows, and a six-variable graph edited edge by edge; a repeated parent, a deterministic child
//! and a node that parents itself are refused.
//!
//! The filter half re-derives every step from the definition with the same random stream: the
//! predicted particle `a x + s e`, the log-weight update `-((y - x) / s_obs)^2 / 2`, the normalized
//! weights and Kish size, and the systematic resample (the smallest ancestor whose cumulative weight
//! reaches `(u + j) / n`) when the size falls below half the particles. The statistical agreement
//! of the filter with the exact Kalman moments is the separate `particle_filter_oracle.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;
use antecedent_state::{
    GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LgssmParams, LocalScoreCache,
    ParentSetOp, ParticleFilterState, StateError,
};
use serde_json::Value;

fn num(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn data_of(case: &Value) -> GraphScoreData {
    let columns: Vec<Vec<f64>> = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_array().unwrap().iter().map(num).collect())
        .collect();
    let (n_vars, n_rows) = (columns.len(), columns[0].len());
    let flat: Vec<f64> = columns.into_iter().flatten().collect();
    GraphScoreData::new(n_rows, n_vars, Arc::from(flat)).unwrap()
}

fn cache(data: &GraphScoreData) -> LocalScoreCache {
    LocalScoreCache::new(GraphScoreCacheKey {
        data_version: 1,
        family: GraphScoreFamily::GaussianBic,
        var_fingerprint: data.n_vars as u64,
        penalty_fingerprint: data.n_rows as u64,
    })
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * (1.0 + want.abs()),
        "{what}: got {got:.17e} want {want:.17e}"
    );
}

#[test]
fn local_scores_deltas_and_totals_match_the_exact_reference() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/state_incremental_particle_graph_score_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = data_of(case);
        let mut cache = cache(&data);
        let mut total = cache.score_graph(&data).unwrap();
        // The empty graph scores every column against an intercept alone.
        let steps = case["steps"].as_array().unwrap();
        for (index, step) in steps.iter().enumerate() {
            let node = u32::try_from(step["node"].as_u64().unwrap()).unwrap();
            let parents: Vec<u32> = step["parents"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            let (delta, new_total) = cache
                .delta_score(&data, ParentSetOp::SetParents { node, parents: Arc::from(parents) })
                .unwrap();
            let want_total = num(&step["total"]);
            let what = format!("{name} step {index}");
            close(new_total, want_total, 1e-9, &format!("{what} total"));
            close(total + delta, new_total, 1e-12, &format!("{what} delta"));
            total = new_total;
            for (j, want) in step["local"].as_object().unwrap() {
                let j: u32 = j.parse().unwrap();
                let got = cache.local_score(&data, j, &cache.parents_of(j)).unwrap();
                close(got, num(want), 1e-9, &format!("{what} local {j}"));
            }
        }
    }
}

#[test]
fn degenerate_parents_and_fits_are_refused_not_scored() {
    let n = 12usize;
    let x: Vec<f64> =
        (0..n).map(|i| f64::from(u32::try_from((i * 7) % 11).unwrap()) / 4.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let noisy: Vec<f64> = (0..n).map(|i| f64::from(u32::try_from((i * 5) % 7).unwrap())).collect();
    let mut flat = x.clone();
    flat.extend(&x); // a repeated column
    flat.extend(&y); // a deterministic child
    flat.extend(&noisy);
    let data = GraphScoreData::new(n, 4, Arc::from(flat)).unwrap();
    let mut cache = cache(&data);
    let local = |cache: &mut LocalScoreCache, node: u32, parents: &[u32]| {
        cache.local_score(&data, node, &Arc::from(parents))
    };
    assert!(local(&mut cache, 3, &[0, 1]).is_err(), "a repeated parent column is rank deficient");
    assert!(
        matches!(local(&mut cache, 2, &[0]), Err(StateError::DegenerateScore(_))),
        "a child that is an exact linear function of its parent has no finite likelihood"
    );
    assert!(local(&mut cache, 3, &[0]).is_ok());
    let before = cache.score_graph(&data).unwrap();
    assert!(
        cache
            .delta_score(&data, ParentSetOp::SetParents { node: 3, parents: Arc::from([3u32]) })
            .is_err(),
        "a node cannot be its own parent"
    );
    assert!(
        cache
            .delta_score(&data, ParentSetOp::SetParents { node: 2, parents: Arc::from([0u32]) })
            .is_err()
    );
    assert_eq!(
        cache.score_graph(&data).unwrap().to_bits(),
        before.to_bits(),
        "a refused edit leaves the total"
    );
    let one_row = GraphScoreData::new(1, 1, Arc::from([1.0])).unwrap();
    assert!(self::cache(&one_row).local_score(&one_row, 0, &Arc::from([])).is_err());
}

/// The definition of one bootstrap step, written against the random stream only.
struct Reference {
    particles: Vec<f64>,
    log_weights: Vec<f64>,
    rng: CausalRng,
}

impl Reference {
    fn new(n: usize, params: LgssmParams, seed: u64) -> Self {
        let mut rng = CausalRng::from_seed(seed);
        let particles = (0..n).map(|_| params.process_std * standard_normal(&mut rng)).collect();
        Self { particles, log_weights: vec![0.0; n], rng }
    }

    fn normalized(&self) -> Vec<f64> {
        let max = self.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let raw: Vec<f64> = self.log_weights.iter().map(|w| (w - max).exp()).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|w| w / sum).collect()
    }

    fn step(&mut self, y: f64, params: LgssmParams) -> bool {
        let n = self.particles.len();
        for i in 0..n {
            let predicted =
                params.a * self.particles[i] + params.process_std * standard_normal(&mut self.rng);
            let z = (y - predicted) / params.obs_std;
            self.particles[i] = predicted;
            self.log_weights[i] += -0.5 * z * z;
        }
        let w = self.normalized();
        let ess = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
        if ess >= 0.5 * n as f64 {
            return false;
        }
        let u0 = self.rng.next_f64() / n as f64;
        let mut cumulative = Vec::with_capacity(n);
        let mut running = 0.0;
        for v in &w {
            running += v;
            cumulative.push(running);
        }
        let mut next = Vec::with_capacity(n);
        for j in 0..n {
            let target = u0 + j as f64 / n as f64;
            // The smallest ancestor whose cumulative weight reaches the target (the last when
            // rounding leaves the cumulative sum a hair under one).
            let ancestor = cumulative.iter().position(|c| *c >= target).unwrap_or(n - 1);
            next.push(self.particles[ancestor]);
        }
        self.particles = next;
        self.log_weights.fill(0.0);
        true
    }
}

#[test]
fn every_filter_step_equals_the_definition_driven_by_the_same_stream() {
    let cases: [(LgssmParams, usize, u64, Vec<f64>); 4] = [
        (LgssmParams::default(), 64, 3, vec![0.2, -0.4, 0.9, 1.1, 0.3, -0.8, 0.0, 0.5]),
        // An observation 400 standard deviations out leaves one particle with all the weight.
        (LgssmParams { a: 0.5, process_std: 0.1, obs_std: 0.05 }, 40, 11, vec![0.0, 20.0, 20.0]),
        // One particle: nothing to resample, the weight is one.
        (LgssmParams::default(), 1, 5, vec![0.3, 0.1, -0.2]),
        // Observation units of 1e-150: the log-weights, not the weights, carry the scale.
        (
            LgssmParams { a: 0.9, process_std: 3e-151, obs_std: 5e-151 },
            128,
            8,
            vec![2e-151, -1e-151, 4e-151, 0.0],
        ),
    ];
    for (case, (params, n, seed, ys)) in cases.into_iter().enumerate() {
        let mut product = ParticleFilterState::init(n, params, 1, seed).unwrap();
        let mut reference = Reference::new(n, params, seed);
        assert_eq!(product.particles(), reference.particles.as_slice(), "case {case} init");
        let mut resampled = 0;
        for (t, y) in ys.into_iter().enumerate() {
            product.step(y).unwrap();
            resampled += usize::from(reference.step(y, params));
            let what = format!("case {case} step {t}");
            assert_eq!(product.particles(), reference.particles.as_slice(), "{what}: particles");
            let w = reference.normalized();
            let theirs: Vec<f64> = {
                let max = product.log_weights().iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let raw: Vec<f64> = product.log_weights().iter().map(|v| (v - max).exp()).collect();
                let sum: f64 = raw.iter().sum();
                raw.iter().map(|v| v / sum).collect()
            };
            for (a, b) in w.iter().zip(&theirs) {
                assert!((a - b).abs() <= 1e-13, "{what}: weights");
            }
            let mean: f64 = w.iter().zip(&reference.particles).map(|(a, b)| a * b).sum();
            let scale = params.process_std.max(params.obs_std);
            assert!(
                (product.weighted_mean() - mean).abs() <= 1e-12 * scale.max(mean.abs()),
                "{what}"
            );
            let ess = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
            assert!((product.ess() - ess).abs() <= 1e-9 * ess, "{what}: ess");
        }
        assert_eq!(product.n_obs, 3 + u64::from(case == 0) * 5 + u64::from(case == 3));
        if case == 1 {
            assert!(resampled >= 1, "a degenerate weight set must trigger the resample");
        }
    }
}
