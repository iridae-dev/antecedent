//! Frozen measurement design for the ORIGINAL continuous eleven-dimensional
//! posterior, separate from MLE/Fisher. Ordinary tests never run measurement.
//! See `docs/nested-markov-bayesian-calibration.md` for scope and acceptance.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "fixed three-functional simulation summaries"
)]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use std::io::{Read, Write};

use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_io::nested_markov_bayesian_artifact::{
    Artifact, BayesianFunctional, Expectation, Limits,
};
use antecedent_kernels::quantile_type7_sorted;
use antecedent_learn::nested_markov_bayesian::{Options, Prior};
use antecedent_prob::mcmc_stats::parameter_mcmc_diagnostics;
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};

const RAW: [[f64; 11]; 2] = [
    [0.43, 0.68, 0.31, 0.37, 0.61, 0.29, 0.64, 0.18, 0.28, 0.24, 0.46],
    [0.36, 0.59, 0.24, 0.28, 0.66, 0.38, 0.73, 0.17, 0.22, 0.27, 0.50],
];
fn law(p: &[f64; 11]) -> [f64; 16] {
    // Independent original-model enumeration, no production probability function.
    let mut cells = [0.0; 16];
    for a in 0..2 {
        for t in 0..2 {
            for m in 0..2 {
                for y in 0..2 {
                    let g = p[7 + 2 * a + m];
                    let table = [g, p[3 + a] - g, p[5 + m] - g, 1.0 - p[3 + a] - p[5 + m] + g];
                    cells[8 * a + 4 * t + 2 * m + y] = (if a == 0 { p[0] } else { 1.0 - p[0] })
                        * (if m == 0 { p[1 + t] } else { 1.0 - p[1 + t] })
                        * table[t * 2 + y];
                }
            }
        }
    }
    assert!((cells.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    assert!(cells.iter().all(|p| *p > 0.0));
    cells
}
fn sample(n: usize, seed: u64, coordinate: usize) -> NestedMarkovInput {
    let probabilities = law(&RAW[coordinate]);
    let mut rng = candidate::Generator::new(seed);
    let mut counts = vec![0.0; 16];
    for _ in 0..n {
        let u = rng.uniform();
        let mut sum = 0.0;
        let mut selected = 15;
        for (i, p) in probabilities.iter().enumerate() {
            sum += p;
            if u < sum {
                selected = i;
                break;
            }
        }
        counts[selected] += 1.0;
    }
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2; 4],
            cells: counts,
        }],
    }
}
fn truth(coordinate: usize) -> [f64; 3] {
    let p = RAW[coordinate];
    let m0 = p[1] * (1.0 - p[5]) + (1.0 - p[1]) * (1.0 - p[6]);
    let m1 = p[2] * (1.0 - p[5]) + (1.0 - p[2]) * (1.0 - p[6]);
    [m0, m1, m1 - m0]
}
fn endpoint_precision(artifact: &Artifact) -> Result<f64, &'static str> {
    let o = &artifact.options;
    let mut largest = 0.0_f64;
    for k in 11..14 {
        let mut sorted: Vec<_> = artifact.posterior.samples.iter().map(|row| row[k]).collect();
        sorted.sort_by(f64::total_cmp);
        let width = artifact.posterior.credible[k][1] - artifact.posterior.credible[k][0];
        if width <= 0.0 {
            return Err("zero interval width");
        }
        for (side, p) in [(0, 0.025), (1, 0.975)] {
            let endpoint = artifact.posterior.credible[k][side];
            let indicator: Vec<_> = artifact
                .posterior
                .samples
                .iter()
                .map(|r| if r[k] <= endpoint { 1.0 } else { 0.0 })
                .collect();
            // Rank normalization is affine on this two-valued indicator. Its bulk
            // Geyer ESS directly estimates local endpoint-CDF ESS, unlike the
            // production diagnostics' separate five/ninety-five tail indicators.
            let ess = parameter_mcmc_diagnostics(&indicator, o.chains, o.draws, 1)[0].ess_bulk;
            if !ess.is_finite() || ess < 400.0 {
                return Err("endpoint indicator ESS below400");
            }
            let radius = 1.959_963_984_540_054 * (p * (1.0 - p) / ess).sqrt();
            let low = quantile_type7_sorted(&sorted, (p - radius).max(0.0));
            let high = quantile_type7_sorted(&sorted, (p + radius).min(1.0));
            let relative = (high - low) / width;
            largest = largest.max(relative);
            if !relative.is_finite() || relative > 0.20 {
                return Err("estimated endpoint MC bracket exceeds20percent interval width");
            }
        }
    }
    Ok(largest)
}
struct Outcome {
    values: [f64; 3],
    covariance: [[f64; 3]; 3],
    credible: [[f64; 2]; 3],
    bases: [CalibrationBasis; 3],
    endpoint_fraction: f64,
    worst_rhat: f64,
    smallest_ess: f64,
}
// Fixed-size sufficient moments retain the exact replicate-order updates across
// the shared 400 -> 2000 extension; no posterior draws/artifacts are checkpointed.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Auxiliary {
    fixture: String,
    attempts: u32,
    successful: u32,
    sums: [f64; 3],
    cross: [[f64; 3]; 3],
    sum_cov: [[f64; 3]; 3],
    failures: std::collections::BTreeMap<String, u32>,
    largest_endpoint: f64,
    worst_rhat: f64,
    smallest_ess: Option<f64>,
}
impl Auxiliary {
    fn empty(fixture: String) -> Self {
        Self {
            fixture,
            attempts: 0,
            successful: 0,
            sums: [0.0; 3],
            cross: [[0.0; 3]; 3],
            sum_cov: [[0.0; 3]; 3],
            failures: std::collections::BTreeMap::default(),
            largest_endpoint: 0.0,
            worst_rhat: 0.0,
            smallest_ess: None,
        }
    }
    fn success(
        &mut self,
        values: [f64; 3],
        cov: [[f64; 3]; 3],
        endpoint: f64,
        rhat: f64,
        ess: f64,
    ) {
        self.attempts += 1;
        self.successful += 1;
        self.largest_endpoint = self.largest_endpoint.max(endpoint);
        self.worst_rhat = self.worst_rhat.max(rhat);
        self.smallest_ess = Some(self.smallest_ess.map_or(ess, |prior| prior.min(ess)));
        for i in 0..3 {
            for j in 0..3 {
                self.cross[i][j] += values[i] * values[j];
                self.sum_cov[i][j] += cov[i][j];
            }
            self.sums[i] += values[i];
        }
    }
    fn failure(&mut self, reason: &str) {
        self.attempts += 1;
        // Diagnostic category text is bounded independently of replicate count.
        let category: String = reason.chars().take(512).collect();
        *self.failures.entry(category).or_default() += 1;
        assert!(self.failures.len() <= 4096, "auxiliary failure category bound");
    }
    fn decode(bytes: &[u8], fixture: &str, start: u32) -> Result<Self, String> {
        if bytes.len() > 4 * 1024 * 1024 {
            return Err("auxiliary checkpoint byte bound".into());
        }
        let state: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if state.fixture != fixture
            || state.attempts != start
            || state.successful > start
            || u64::from(state.successful)
                + state.failures.values().map(|v| u64::from(*v)).sum::<u64>()
                != u64::from(start)
            || state.failures.len() > 4096
            || state.failures.keys().any(|k| k.chars().count() > 512)
            || !state
                .sums
                .iter()
                .chain(state.cross.iter().flatten())
                .chain(state.sum_cov.iter().flatten())
                .all(|v| v.is_finite())
            || !state.largest_endpoint.is_finite()
            || !state.worst_rhat.is_finite()
            || state.smallest_ess.is_some_and(|v| !v.is_finite() || v <= 0.0)
            || (state.successful > 0) != state.smallest_ess.is_some()
        {
            return Err("incompatible auxiliary checkpoint fixture, prefix or moments".into());
        }
        Ok(state)
    }
    fn path(tally: &str, test: &str, n: usize) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{tally}.{test}.n{n}.nested-bayes-aux-v1.json"))
    }
    fn restore(fixture: String, test: &str, n: usize) -> Self {
        let start = calibration::replicate_start();
        if start == 0 {
            return Self::empty(fixture);
        }
        let tally = std::env::var(calibration::PRIOR_TALLIES_ENV)
            .expect("extension needs original tally path");
        let path = Self::path(&tally, test, n);
        let file =
            std::fs::File::open(&path).expect("extension needs matching auxiliary checkpoint");
        let mut bytes = Vec::new();
        file.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes).expect("read auxiliary checkpoint");
        Self::decode(&bytes, &fixture, start).expect("validated auxiliary prefix")
    }
    fn persist(&self, test: &str, n: usize) {
        let Ok(tally) = std::env::var(calibration::TALLY_OUT_ENV) else {
            return;
        };
        let path = Self::path(&tally, test, n);
        let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
        let bytes = serde_json::to_vec(self).expect("finite auxiliary state");
        assert!(bytes.len() <= 4 * 1024 * 1024, "auxiliary checkpoint byte bound");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .expect("create auxiliary checkpoint");
        file.write_all(&bytes).expect("write auxiliary checkpoint");
        file.sync_all().expect("sync auxiliary checkpoint");
        std::fs::rename(&temporary, &path).expect("atomic auxiliary checkpoint");
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        std::fs::File::open(parent)
            .expect("open auxiliary checkpoint directory")
            .sync_all()
            .expect("sync auxiliary checkpoint directory");
    }
}
fn fixture(coordinate: usize, test: &str, n: usize) -> String {
    let declaration = serde_json::to_vec(&(
        "nested_bayes_aux_v1",
        test,
        n,
        RAW[coordinate],
        coordinate + 1,
        4,
        2048,
        4096,
        5_000_000,
        0.95,
        0.20,
        0.25,
        grid_seed(0x48be_a190 + coordinate as u64 * 0x100_0000),
        antecedent_learn::nested_markov_bayesian::METHOD,
        antecedent_learn::nested_markov_bayesian::RNG,
        calibration::measured_at(),
    ))
    .unwrap();
    blake3::hash(&declaration).to_hex().to_string()
}
fn measure(coordinate: usize, test: &'static str) {
    let n = grid_n(4000);
    let truths = truth(coordinate);
    let labels = ["mean0", "mean1", "contrast"];
    let mut tallies = labels.map(|label| {
        CoverageTally::for_record(
            RecordKey {
                test,
                dgp: if coordinate == 0 {
                    "iid_full11d_verma_uniform_beta"
                } else {
                    "iid_full11d_verma_beta2"
                },
                interval: "posterior_eti",
            },
            0.95,
        )
        .labelled(label.to_string())
    });
    let mut auxiliary = Auxiliary::restore(fixture(coordinate, test, n), test, n);
    let results = map_replicates(n_sim(), |rep| -> Result<Outcome, String> {
        let seed = grid_seed(0x48be_a190 + rep + coordinate as u64 * 0x100_0000);
        let input = sample(n, seed, coordinate);
        let prior = Prior {
            alpha: [if coordinate == 0 { 1.0 } else { 2.0 }; 11],
            beta: [if coordinate == 0 { 1.0 } else { 2.0 }; 11],
        };
        let options = Options {
            chains: 4,
            warmup: 2048,
            draws: 4096,
            max_proposals: 5_000_000,
            seed,
            credible_mass: 0.95,
        };
        let ctx = ExecutionContext::for_tests(seed);
        let artifact = Artifact::build(&input, &FitOptions::default(), prior, options, &ctx)
            .map_err(|e| format!("producer:{e}"))?;
        let expected = Expectation {
            premises_digest: Some(artifact.premises_digest.clone()),
            point: antecedent_io::nested_markov_artifact::NestedMarkovExpectation {
                premises_digest: Some(artifact.point.premises_digest.clone()),
                data_digest: Some(artifact.point.data_digest.clone()),
            },
        };
        let replay = Artifact::consume(
            &artifact.export().map_err(|e| e.to_string())?,
            &expected,
            Limits::default(),
            &ExecutionContext::for_tests(seed.wrapping_add(1)),
        )
        .map_err(|e| format!("consumer:{e}"))?;
        let endpoint_fraction =
            endpoint_precision(&replay).map_err(|e| format!("endpoint_precision:{e}"))?;
        // Keep only replayed summaries, not every replicate's 16384x14 draws.
        // Peak posterior memory follows worker count rather than total replicates.
        Ok(Outcome {
            values: std::array::from_fn(|i| replay.posterior.mean[11 + i]),
            covariance: std::array::from_fn(|i| {
                std::array::from_fn(|j| replay.posterior.covariance[(11 + i) * 14 + 11 + j])
            }),
            credible: std::array::from_fn(|i| replay.posterior.credible[11 + i]),
            bases: [
                BayesianFunctional::Mean0,
                BayesianFunctional::Mean1,
                BayesianFunctional::Contrast,
            ]
            .map(|f| replay.calibration_basis(f)),
            endpoint_fraction,
            worst_rhat: replay
                .posterior
                .diagnostics
                .iter()
                .map(|d| d.rank_rhat.max(d.folded_rhat))
                .fold(0.0_f64, f64::max),
            smallest_ess: replay
                .posterior
                .diagnostics
                .iter()
                .map(|d| d.bulk_ess.min(d.tail_ess))
                .fold(f64::INFINITY, f64::min),
        })
    });
    for result in results {
        match result {
            Ok(out) => {
                auxiliary.success(
                    out.values,
                    out.covariance,
                    out.endpoint_fraction,
                    out.worst_rhat,
                    out.smallest_ess,
                );
                for i in 0..3 {
                    candidate::bind(&mut tallies[i], &out.bases[i]);
                    let [low, high] = out.credible[i];
                    tallies[i].record(Some((low, high)), truths[i]);
                }
            }
            Err(reason) => {
                auxiliary.failure(&reason);
                for tally in &mut tallies {
                    tally.skip();
                }
            }
        }
    }
    assert_eq!(auxiliary.attempts, n_sim(), "complete auxiliary replicate prefix");
    auxiliary.persist(test, n); // Before any coverage assertion can trigger recheck.
    println!(
        "diagnostic-precision {test} n={n} attempts={} successful={} failed={:?} max_endpoint_mc_bracket_fraction={} max_rank_folded_rhat={} min_bulk_tail_ess={:?}; estimated_mc_precision_not_endpoint_guarantee",
        auxiliary.attempts,
        auxiliary.successful,
        auxiliary.failures,
        auxiliary.largest_endpoint,
        auxiliary.worst_rhat,
        auxiliary.smallest_ess
    );
    for tally in tallies {
        tally.assert();
    }
    assert!(auxiliary.successful > 1, "no successful joint covariance sample");
    let count = f64::from(auxiliary.successful);
    let mean = auxiliary.sums.map(|v| v / count);
    let empirical: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (auxiliary.cross[i][j] - auxiliary.sums[i] * auxiliary.sums[j] / count) / (count - 1.0)
        })
    });
    let mut posterior = [[0.0; 3]; 3];
    let mut error = 0.0;
    let mut norm = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            posterior[i][j] = auxiliary.sum_cov[i][j] / count;
            error += (posterior[i][j] - empirical[i][j]).powi(2);
            norm += empirical[i][j].powi(2);
        }
    }
    let relative_covariance_error = (error / norm).sqrt();
    println!(
        "diagnostic-precision {test} n={n} posterior_mean={mean:?} exact_truth={truths:?} average_joint_covariance={posterior:?} empirical_joint_covariance={empirical:?} relative_frobenius_error={relative_covariance_error}"
    );
    assert!(relative_covariance_error <= 0.25, "joint covariance precision gate");
    for i in 0..3 {
        assert!(
            (mean[i] - truths[i]).abs() <= 0.25 * empirical[i][i].sqrt(),
            "mean bias gate functional{i}"
        );
    }
}
#[test]
#[ignore = "calibration: final measurement only; continuous original11D posterior"]
fn nested_bayesian_full11d_uniform_l95() {
    measure(0, "nested_bayesian_full11d_uniform_l95");
}
#[test]
#[ignore = "calibration: final measurement only; continuous original11D posterior"]
fn nested_bayesian_full11d_beta2_l95() {
    measure(1, "nested_bayesian_full11d_beta2_l95");
}

#[test]
fn auxiliary_checkpoint_extends_exact_prefix_and_rejects_incompatible_state() {
    fn feed(state: &mut Auxiliary, rep: u32) {
        if rep % 7 == 0 {
            state.failure("synthetic:numerical");
        } else {
            let x = f64::from(rep) / 2048.0;
            state.success(
                [x, 1.0 - x, 1.0 - 2.0 * x],
                [[0.01; 3]; 3],
                x / 10.0,
                1.0 + x / 1000.0,
                1000.0 - x,
            );
        }
    }
    let identity = fixture(0, "synthetic", 4000);
    let mut whole = Auxiliary::empty(identity.clone());
    for rep in 0..2000 {
        feed(&mut whole, rep);
    }
    let mut first = Auxiliary::empty(identity.clone());
    for rep in 0..400 {
        feed(&mut first, rep);
    }
    let bytes = serde_json::to_vec(&first).unwrap();
    let mut restored = Auxiliary::decode(&bytes, &identity, 400).unwrap();
    for rep in 400..2000 {
        feed(&mut restored, rep);
    }
    assert_eq!(whole, restored, "all auxiliary moments and extrema include original400");
    assert!(Auxiliary::decode(&bytes, &fixture(1, "synthetic", 4000), 400).is_err());
    assert!(Auxiliary::decode(&bytes, &fixture(0, "synthetic", 8000), 400).is_err());
    assert!(Auxiliary::decode(&bytes, &identity, 399).is_err());
    let mut inconsistent = first.clone();
    inconsistent.successful += 1;
    assert!(
        Auxiliary::decode(&serde_json::to_vec(&inconsistent).unwrap(), &identity, 400).is_err()
    );
    assert!(Auxiliary::decode(&vec![0; 4 * 1024 * 1024 + 1], &identity, 400).is_err());
}
