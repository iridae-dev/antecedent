//! Frozen measurement design for the ORIGINAL continuous eleven-dimensional
//! posterior, separate from MLE/Fisher. Ordinary tests never run measurement.
//! See `conformance/transport/nested_bayesian/README.md` for scope and acceptance.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "fixed three-functional simulation summaries"
)]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::ExecutionContext;
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
    artifact: Artifact,
    endpoint_fraction: f64,
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
        Ok(Outcome { artifact: replay, endpoint_fraction })
    });
    let mut failures = std::collections::BTreeMap::<String, usize>::new();
    let mut means = Vec::new();
    let mut sum_cov = [[0.0; 3]; 3];
    let mut largest_endpoint = 0.0_f64;
    let mut worst_rhat = 0.0_f64;
    let mut smallest_ess = f64::INFINITY;
    for result in results {
        match result {
            Ok(out) => {
                largest_endpoint = largest_endpoint.max(out.endpoint_fraction);
                for d in &out.artifact.posterior.diagnostics {
                    worst_rhat = worst_rhat.max(d.rank_rhat.max(d.folded_rhat));
                    smallest_ess = smallest_ess.min(d.bulk_ess.min(d.tail_ess));
                }
                let values = [
                    out.artifact.posterior.mean[11],
                    out.artifact.posterior.mean[12],
                    out.artifact.posterior.mean[13],
                ];
                means.push(values);
                for i in 0..3 {
                    candidate::bind(
                        &mut tallies[i],
                        &out.artifact.calibration_basis(
                            [
                                BayesianFunctional::Mean0,
                                BayesianFunctional::Mean1,
                                BayesianFunctional::Contrast,
                            ][i],
                        ),
                    );
                    let [low, high] = out.artifact.posterior.credible[11 + i];
                    tallies[i].record(Some((low, high)), truths[i]);
                    for j in 0..3 {
                        sum_cov[i][j] += out.artifact.posterior.covariance[(11 + i) * 14 + 11 + j];
                    }
                }
            }
            Err(reason) => {
                *failures.entry(reason).or_default() += 1;
                for tally in &mut tallies {
                    tally.skip();
                }
            }
        }
    }
    println!(
        "diagnostic-precision {test} n={n} attempts={} successful={} failed={:?} max_endpoint_mc_bracket_fraction={largest_endpoint} max_rank_folded_rhat={worst_rhat} min_bulk_tail_ess={smallest_ess}; estimated_mc_precision_not_endpoint_guarantee",
        n_sim(),
        means.len(),
        failures
    );
    for tally in tallies {
        tally.assert();
    }
    assert!(means.len() > 1, "no successful joint covariance sample");
    let count = means.len() as f64;
    let mut mean = [0.0; 3];
    for row in &means {
        for i in 0..3 {
            mean[i] += row[i] / count;
        }
    }
    let mut empirical = [[0.0; 3]; 3];
    let mut posterior = [[0.0; 3]; 3];
    for row in &means {
        for i in 0..3 {
            for j in 0..3 {
                empirical[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]) / (count - 1.0);
            }
        }
    }
    let mut error = 0.0;
    let mut norm = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            posterior[i][j] = sum_cov[i][j] / count;
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
