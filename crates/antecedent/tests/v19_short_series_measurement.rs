//! Measurement behind the per-family short-series thresholds
//! (`antecedent_estimate::CircularBlockFamily::min_effective_rows`).
//!
//! Not a gate. The one ignored test sweeps AR(1) persistence
//! `ρ ∈ {0.5, 0.8, 0.9, 0.95}` and series length `n` over calibration DGPs of
//! every circular-block SE family and prints, per cell: nominal-90% coverage of
//! `estimate ± z·se_bootstrap`, bias and mean SE over the Monte-Carlo SD of the
//! estimate, the effective-rows statistic the runtime computed (parsed from the
//! `estimate.temporal.circular_block_se` / shared-block provenance), the median
//! block count `rows / block_length`, and the share of replicates carrying
//! `estimate.temporal.circular_block_se.short_series`. The output table is
//! committed in `docs/short-series-thresholds.md`:
//!
//! ```text
//! ANTECEDENT_CALIBRATION_NSIM=2000 cargo test --release -p antecedent \
//!   --test v19_short_series_measurement -- --ignored --nocapture
//! ```
//!
//! `ANTECEDENT_SHORT_SERIES_CSV=<path>` also writes one row per replicate;
//! `ANTECEDENT_SHORT_SERIES_DESIGNS=<family or name substring>,…` restricts the
//! sweep; `ANTECEDENT_SHORT_SERIES_SEED_OFFSET=<k>` (below `10000 − replicates`)
//! draws an independent replication.
//!
//! Designs (family: DGP). Each family is measured on a short-memory score and on
//! a persistent one, since the effective-row count alone does not tell them
//! apart:
//!
//! * single-window: `TemporalDag` Pulse h=1 on `common::driven_dgp` (MA(3)
//!   treatment, AR(1) residual) and on `common::persistent_dgp::chain_pulse_dag`
//!   over `fixtures::chain_pag_series` (AR(1) treatment and residual);
//! * mediation: Total / Direct / Mediated on `common::driven_dgp` and on
//!   `common::persistent_dgp::mediation_series`;
//! * sequential: `TemporalDag` multi-step Sustained over lags 2..1 on
//!   `fixtures::confounded_series` with `fixtures::confounded_dag` and on
//!   `common::persistent_dgp::sequential_series`;
//! * mixture: the six-completion `fixtures::chain_pag` Pulse envelope, the
//!   two-completion `fixtures::confounded_cpdag` Pulse envelope, and the
//!   heterogeneous DBN posterior's multi-step Sustained (sequential atoms).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

mod common;

use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent::estimate::TemporalMediationUncertainty;
use antecedent::{InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{CausalQuery, ExecutionContext, MediationContrast};
use antecedent_data::TimeSeriesData;
use common::calibration::{Z90, coverage_band, n_sim, normal_interval};
use common::driven_dgp::{self, Scenario};
use common::{fixtures, persistent_dgp};

const SHORT_SERIES: &str = "estimate.temporal.circular_block_se.short_series";
const RHOS: [f64; 4] = [0.5, 0.8, 0.9, 0.95];

/// One replicate: coverage of each reported contrast plus the runtime statistic.
struct Rep {
    covered: Vec<bool>,
    /// Headline estimate minus its truth.
    error: f64,
    /// Headline circular-block SE.
    se: f64,
    effective_rows: f64,
    rows: f64,
    block_length: f64,
    warned: bool,
}

struct Design {
    family: &'static str,
    name: &'static str,
    contrasts: &'static [&'static str],
    ns: &'static [usize],
    seed: u64,
    fit: fn(rho: f64, n: usize, seed: u64) -> Option<Rep>,
}

fn study(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
    boot: u32,
    seed: u64,
) -> Option<StudyResult> {
    Study::series(data)
        .graph(graph.into())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()
}

/// The number printed right after `key` in `message` (NaN when absent).
fn number_after(message: &str, key: &str) -> f64 {
    let Some(at) = message.find(key) else { return f64::NAN };
    let rest = &message[at + key.len()..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'N' | 'a')))
        .unwrap_or(rest.len());
    rest[..end].trim_end_matches('.').parse().unwrap_or(f64::NAN)
}

/// `(effective rows, rows, block length)` from the circular-block provenance.
fn block_stats(result: &StudyResult) -> (f64, f64, f64) {
    for d in &result.diagnostics {
        let m = d.message.as_ref();
        let (rows_key, block_key) = match d.code.as_ref() {
            "estimate.temporal.circular_block_se" => ("n=", "circular-block length "),
            "estimate.temporal_class.frequentist.shared_block"
            | "estimate.dbn_posterior.frequentist" => ("over the ", "blocks of "),
            _ => continue,
        };
        return (
            number_after(m, "score effective rows "),
            number_after(m, rows_key),
            number_after(m, block_key),
        );
    }
    (f64::NAN, f64::NAN, f64::NAN)
}

/// `contrasts`: `(point, se, truth)` per reported contrast; the first is the
/// headline used for the error and SE columns.
fn rep_from(result: &StudyResult, contrasts: &[(f64, Option<f64>, f64)]) -> Rep {
    let (point, se, truth) = contrasts[0];
    let (effective_rows, rows, block_length) = block_stats(result);
    Rep {
        covered: contrasts
            .iter()
            .map(|&(point, se, truth)| {
                normal_interval(point, se, Z90).is_some_and(|(lo, hi)| lo <= truth && truth <= hi)
            })
            .collect(),
        error: point - truth,
        se: se.unwrap_or(f64::NAN),
        effective_rows,
        rows,
        block_length,
        warned: result.diagnostics.iter().any(|d| d.code.as_ref() == SHORT_SERIES),
    }
}

fn headline(result: &StudyResult, truth: f64) -> Rep {
    let est = &result.estimate;
    rep_from(result, &[(est.ate, est.se_bootstrap, truth)])
}

fn mediation_rep(result: &StudyResult) -> Option<Rep> {
    use driven_dgp::{A, B, C};
    let slice = &result.mediation_grid.as_ref()?.slices[0];
    let TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } = &slice.uncertainty
    else {
        return None;
    };
    let p = &slice.estimate;
    Some(rep_from(
        result,
        &[
            (p.total?, block.total, C + A * B),
            (p.direct?, block.direct, C),
            (p.mediated?, block.mediated, A * B),
        ],
    ))
}

// ---------------------------------------------------------------------------
// Designs
// ---------------------------------------------------------------------------

fn scenario(rho: f64, n: usize, seed: u64) -> Scenario {
    Scenario { label: "measurement", rho, n, seed }
}

fn single_window_driven(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        driven_dgp::pulse_series(scenario(rho, n, seed), 0, false),
        driven_dgp::pulse_dag(false),
        CausalQuery::TemporalEffect(driven_dgp::pulse(1)),
        100,
        seed,
    )?;
    Some(headline(&result, driven_dgp::BETA))
}

fn single_window_persistent(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        fixtures::chain_pag_series(n, rho, seed),
        persistent_dgp::chain_pulse_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::pulse()),
        100,
        seed,
    )?;
    Some(headline(&result, fixtures::B1))
}

fn mediation_driven(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        driven_dgp::mediation_series(scenario(rho, n, seed), 0),
        driven_dgp::mediation_dag(),
        driven_dgp::mediation_query(MediationContrast::Mediated),
        100,
        seed,
    )?;
    mediation_rep(&result)
}

fn mediation_persistent(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        persistent_dgp::mediation_series(n, rho, seed),
        persistent_dgp::mediation_dag(),
        driven_dgp::mediation_query(MediationContrast::Mediated),
        100,
        seed,
    )?;
    mediation_rep(&result)
}

fn sequential_confounded(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        fixtures::confounded_series(n, fixtures::B2, rho, seed),
        fixtures::confounded_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::multi_sustained()),
        100,
        seed,
    )?;
    Some(headline(&result, fixtures::B1 + fixtures::B2))
}

fn sequential_persistent(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        persistent_dgp::sequential_series(n, rho, seed),
        fixtures::two_lag_dag(),
        CausalQuery::TemporalEffect(persistent_dgp::multi_sustained()),
        100,
        seed,
    )?;
    Some(headline(&result, fixtures::B1 + fixtures::B2))
}

fn mixture_chain_pag(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        fixtures::chain_pag_series(n, rho, seed),
        fixtures::chain_pag(),
        CausalQuery::TemporalEffect(persistent_dgp::pulse()),
        199,
        seed,
    )?;
    let (adjusted, unadjusted) = fixtures::chain_pag_effects();
    // Six identified completions in equal enumeration mass: four adjust `z`.
    Some(headline(&result, (4.0 * adjusted + 2.0 * unadjusted) / 6.0))
}

fn mixture_cpdag(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = study(
        fixtures::confounded_series(n, 0.0, rho, seed),
        fixtures::confounded_cpdag(),
        CausalQuery::TemporalEffect(persistent_dgp::pulse()),
        199,
        seed,
    )?;
    Some(headline(&result, fixtures::cpdag_completion_truths(rho).iter().sum::<f64>() / 2.0))
}

fn mixture_dbn_sequential(rho: f64, n: usize, seed: u64) -> Option<Rep> {
    let result = Study::series(fixtures::confounded_series(n, fixtures::B2, rho, seed))
        .graph_posterior(fixtures::heterogeneous_dbn())
        .temporal_query(persistent_dgp::multi_sustained())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(199)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()?;
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_multistep_atom_truths(rho));
    Some(headline(&result, truth))
}

const SHORT_NS: &[usize] = &[40, 60, 100, 160, 400];
const LONG_NS: &[usize] = &[60, 100, 160, 400, 800, 1600];
const MIXTURE_NS: &[usize] = &[40, 60, 100, 160, 400, 800, 1600];

fn designs() -> Vec<Design> {
    const ONE: &[&str] = &["headline"];
    const MEDIATION: &[&str] = &["Total", "Direct", "Mediated"];
    let design =
        |family, name, contrasts, ns, seed, fit| Design { family, name, contrasts, ns, seed, fit };
    vec![
        design(
            "single-window",
            "Pulse h=1, MA(3) treatment",
            ONE,
            SHORT_NS,
            1_000_000_000,
            single_window_driven,
        ),
        design(
            "single-window",
            "Pulse h=1, AR(1) treatment",
            ONE,
            SHORT_NS,
            2_000_000_000,
            single_window_persistent,
        ),
        design(
            "mediation",
            "mediation, MA(3) treatment",
            MEDIATION,
            SHORT_NS,
            3_000_000_000,
            mediation_driven,
        ),
        design(
            "mediation",
            "mediation, AR(1) treatment",
            MEDIATION,
            SHORT_NS,
            3_500_000_000,
            mediation_persistent,
        ),
        design(
            "sequential",
            "multi-step Sustained, confounded DAG",
            ONE,
            SHORT_NS,
            4_000_000_000,
            sequential_confounded,
        ),
        design(
            "sequential",
            "multi-step Sustained, AR(1) treatment",
            ONE,
            SHORT_NS,
            4_500_000_000,
            sequential_persistent,
        ),
        design(
            "mixture",
            "TemporalPag Pulse, six completions",
            ONE,
            LONG_NS,
            5_000_000_000,
            mixture_chain_pag,
        ),
        design(
            "mixture",
            "TemporalCpdag Pulse, two completions",
            ONE,
            MIXTURE_NS,
            6_000_000_000,
            mixture_cpdag,
        ),
        design(
            "mixture",
            "DBN multi-step Sustained, two atoms",
            ONE,
            MIXTURE_NS,
            7_000_000_000,
            mixture_dbn_sequential,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

struct Cell {
    design: usize,
    rho: f64,
    n: usize,
    reps: Vec<Rep>,
    skipped: u32,
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn run_cells(designs: &[Design]) -> Vec<Cell> {
    let only = std::env::var("ANTECEDENT_SHORT_SERIES_DESIGNS").ok();
    let jobs: Vec<(usize, usize, usize)> = designs
        .iter()
        .enumerate()
        .filter(|(_, design)| {
            only.as_deref().is_none_or(|only| {
                only.split(',').any(|f| design.name.contains(f) || design.family == f)
            })
        })
        .flat_map(|(d, design)| {
            (0..RHOS.len()).flat_map(move |r| design.ns.iter().map(move |&n| (d, r, n)))
        })
        .collect();
    let offset: u64 = std::env::var("ANTECEDENT_SHORT_SERIES_SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&o| o < 10_000 - u64::from(n_sim()))
        .unwrap_or(0);
    let next = AtomicUsize::new(0);
    let cells = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some(&(d, r, n)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let design = &designs[d];
                    let rho = RHOS[r];
                    let base = design.seed + r as u64 * 100_000_000 + n as u64 * 10_000 + offset;
                    let mut reps = Vec::new();
                    let mut skipped = 0;
                    for rep in 0..n_sim() {
                        match (design.fit)(rho, n, base + u64::from(rep)) {
                            Some(r) => reps.push(r),
                            None => skipped += 1,
                        }
                    }
                    cells.lock().unwrap().push(Cell { design: d, rho, n, reps, skipped });
                }
            });
        }
    });
    let mut cells = cells.into_inner().unwrap();
    cells.sort_by(|a, b| (a.design, a.n).cmp(&(b.design, b.n)).then(a.rho.total_cmp(&b.rho)));
    cells
}

#[test]
#[ignore = "measurement: short-series threshold provenance, not a gate"]
fn short_series_effective_rows_coverage_grid() {
    let designs = designs();
    let cells = run_cells(&designs);
    let (lo, hi) = coverage_band(n_sim(), 0.9);
    eprintln!(
        "short-series measurement: {} replicates per cell, nominal 0.90, band at this count \
         [{lo:.3}, {hi:.3}] (the 400-replicate gate band is [0.855, 0.945])",
        n_sim()
    );
    eprintln!(
        "| family | design | ρ | n | coverage | bias/SD | SE/SD | eff. rows q10 / median / q90 \
         | blocks | warned |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|---|---|");
    let mut csv = String::from(
        "family,design,rho,n,rep,error,se,effective_rows,rows,block_length,warned,covered\n",
    );
    for cell in &cells {
        let design = &designs[cell.design];
        let scored = cell.reps.len().max(1) as f64;
        let coverage: Vec<String> = (0..design.contrasts.len())
            .map(|i| {
                let rate = cell.reps.iter().filter(|r| r.covered[i]).count() as f64 / scored;
                if design.contrasts.len() == 1 {
                    format!("{rate:.3}")
                } else {
                    format!("{} {rate:.3}", design.contrasts[i])
                }
            })
            .collect();
        let eff: Vec<f64> = cell.reps.iter().map(|r| r.effective_rows).collect();
        let blocks: Vec<f64> = cell.reps.iter().map(|r| r.rows / r.block_length).collect();
        let warned = cell.reps.iter().filter(|r| r.warned).count() as f64 / scored;
        let mean_error = cell.reps.iter().map(|r| r.error).sum::<f64>() / scored;
        let sd = (cell.reps.iter().map(|r| (r.error - mean_error).powi(2)).sum::<f64>()
            / (scored - 1.0).max(1.0))
        .sqrt();
        let ses: Vec<f64> = cell.reps.iter().map(|r| r.se).filter(|s| s.is_finite()).collect();
        let mean_se = ses.iter().sum::<f64>() / ses.len().max(1) as f64;
        eprintln!(
            "| {} | {} | {:.2} | {} | {} | {:+.2} | {:.2} | {:.0} / {:.0} / {:.0} | {:.1} \
             | {:.2}{} |",
            design.family,
            design.name,
            cell.rho,
            cell.n,
            coverage.join(", "),
            mean_error / sd,
            mean_se / sd,
            quantile(&eff, 0.1),
            quantile(&eff, 0.5),
            quantile(&eff, 0.9),
            quantile(&blocks, 0.5),
            warned,
            if cell.skipped > 0 { format!(" ({} skipped)", cell.skipped) } else { String::new() },
        );
        for (rep, r) in cell.reps.iter().enumerate() {
            let covered: Vec<&str> = r.covered.iter().map(|&c| if c { "1" } else { "0" }).collect();
            writeln!(
                csv,
                "{},\"{}\",{},{},{rep},{},{},{},{},{},{},{}",
                design.family,
                design.name,
                cell.rho,
                cell.n,
                r.error,
                r.se,
                r.effective_rows,
                r.rows,
                r.block_length,
                u8::from(r.warned),
                covered.join("/"),
            )
            .unwrap();
        }
    }
    if let Ok(path) = std::env::var("ANTECEDENT_SHORT_SERIES_CSV") {
        std::fs::write(&path, csv).unwrap();
        eprintln!("per-replicate records written to {path}");
    }
}
