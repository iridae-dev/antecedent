//! Frequentist `TemporalDag` Pulse and multi-step Sustained without a bootstrap.
//!
//! The dependence-aware circular-block length (a refit plus one Politis–White
//! scan per normal-equation score) is only needed when replicates are drawn;
//! at zero replicates and in the Interactive tier the point estimate must stay
//! within a few times the plain least-squares cost. This bench keeps that
//! budget under a soft guard (`--test` runs it once and asserts) and prints
//! min-of-`ANTECEDENT_BENCH_REPEATS` wall times for `ANTECEDENT_BENCH_CELLS`
//! (comma-separated `pulse_n2000|B0`-style names; every cell by default).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]

use std::time::{Duration, Instant};

use antecedent::{LatencyMode, RefuteSuite, Study};
use antecedent_core::{ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

/// Soft budget for the guarded Interactive cells at n = 20 000 (release build,
/// Apple M1 Max): a few times the fixed values, and well under what an eager
/// score scan costs there (the O(n^1.5) scan grows faster than the fit, so the
/// guard sits where the two are far apart), so a quadratic scan regression
/// trips it while machine noise does not.
const INTERACTIVE_BUDGET: Duration = Duration::from_millis(5);
const GUARDED_ROWS: usize = 20_000;

fn series(n: usize) -> TimeSeriesData {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for t in 1..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0;
        x[t] = 0.6 * x[t - 1] + u;
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + 0.3 * u;
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn graph() -> TemporalDag {
    let x = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, x, Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, y, Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn sustained3_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::sustained(-3, -1))
        .with_horizon_steps(1)
}

/// Builds one cell's query.
type QueryFactory = fn() -> TemporalEffectQuery;

#[derive(Clone, Copy)]
enum Tier {
    /// Zero replicates, default refute suite.
    Zero,
    /// Zero replicates, `RefuteSuite::Full` (runs `bootstrap.ci_coverage`).
    ZeroFull,
    /// `LatencyMode::Interactive`.
    Interactive,
    /// 199 replicates, default refute suite.
    Standard,
}

impl Tier {
    fn label(self) -> &'static str {
        match self {
            Self::Zero => "B0",
            Self::ZeroFull => "B0full",
            Self::Interactive => "I",
            Self::Standard => "B199",
        }
    }
}

fn study(data: &TimeSeriesData, query: TemporalEffectQuery, tier: Tier) -> Study {
    let builder = Study::series(data.clone()).graph(graph()).temporal_query(query);
    match tier {
        Tier::Zero => builder.bootstrap_replicates(0),
        Tier::ZeroFull => builder.bootstrap_replicates(0).refute(RefuteSuite::Full),
        Tier::Interactive => builder.latency_mode(LatencyMode::Interactive),
        Tier::Standard => builder.bootstrap_replicates(199),
    }
    .build()
    .unwrap()
}

fn time_min(study: &Study, ctx: &ExecutionContext, repeats: usize) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..repeats {
        let started = Instant::now();
        let result = study.clone().run(ctx).unwrap();
        let elapsed = started.elapsed();
        assert!(result.estimate.ate.is_finite());
        best = best.min(elapsed);
    }
    best
}

fn main() {
    let ctx = ExecutionContext::for_tests(11);
    let repeats: usize =
        std::env::var("ANTECEDENT_BENCH_REPEATS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let selected = std::env::var("ANTECEDENT_BENCH_CELLS").ok();
    let guard_only = std::env::args().any(|a| a == "--test");
    let cells: Vec<(&str, usize, QueryFactory)> = vec![
        ("pulse_n2000", 2000, pulse_query),
        ("sustained3_n2000", 2000, sustained3_query),
        ("pulse_n20k", GUARDED_ROWS, pulse_query),
        ("sustained3_n20k", GUARDED_ROWS, sustained3_query),
        ("pulse_n100k", 100_000, pulse_query),
        ("sustained3_n100k", 100_000, sustained3_query),
    ];
    let tiers = [Tier::Zero, Tier::ZeroFull, Tier::Interactive, Tier::Standard];
    let mut data_cache: Vec<(usize, TimeSeriesData)> = Vec::new();
    for (name, n, query) in cells {
        for tier in tiers {
            let cell = format!("{name}|{}", tier.label());
            let wanted = selected.as_deref().is_none_or(|s| s.split(',').any(|c| c == cell));
            let guarded = n == GUARDED_ROWS && matches!(tier, Tier::Interactive);
            if guard_only && !guarded {
                continue;
            }
            if !guard_only && !wanted {
                continue;
            }
            if !data_cache.iter().any(|(rows, _)| *rows == n) {
                data_cache.push((n, series(n)));
            }
            let data = &data_cache.iter().find(|(rows, _)| *rows == n).unwrap().1;
            let study = study(data, query(), tier);
            let best = time_min(&study, &ctx, if guard_only { 3 } else { repeats });
            println!("{cell}\t{:.3} ms", best.as_secs_f64() * 1e3);
            assert!(
                !guarded || best < INTERACTIVE_BUDGET,
                "{cell} exceeded soft budget {INTERACTIVE_BUDGET:?}: {best:?}"
            );
        }
    }
}
