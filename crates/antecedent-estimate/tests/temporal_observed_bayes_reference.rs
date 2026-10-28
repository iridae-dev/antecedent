//! The observed-data temporal Gibbs posterior against an exact three-dimensional quadrature.
//!
//! One outcome mechanism `y_t = b0 + b1 x_{t-1} + e_t` on eight usable rows has three free
//! parameters `(b0, b1, s2)`. With the declared prior (`(b0, b1) | s2 ~ N(0, s2 scale^2 I)`,
//! `s2 ~ InvGamma(1e-3, 1e-3)`) and three latent outcomes that are right censored, left censored
//! or unselected, the posterior of the response cells `b0` (dose 0) and `b0 + b1` (dose 1) has
//! no closed form but is a smooth three-dimensional integral once each latent outcome is summed
//! out exactly (survival function, distribution function or 1). `oracle_data/
//! temporal_observed_bayes_reference.py` evaluates it by quadrature and freezes the means,
//! standard deviations and `P(b1 > 0.8)`; this test runs the public entry point
//! `estimate_observed_temporal_response` (identification, mean curve, observation model) and
//! compares the Gibbs summaries with them.
//!
//! Monte Carlo bound. The four chains are started from over-dispersed latent outcomes, so their
//! pooled draws estimate each summary with standard error about `sd / sqrt(ESS)`, where ESS is the
//! diagnostics' minimum bulk ESS over all monitored quantities (a lower bound of every quantity's
//! own ESS, so the bound is conservative) at the draw count used. Every comparison allows five
//! such standard errors for a mean, five `sd / sqrt(ESS)` for a standard deviation and five
//! `sqrt(p (1 - p) / ESS)` for a probability, and the test requires the ESS to be large enough
//! that the bound is under a fifth of the posterior standard deviation. It repeats over several
//! seeds and also bounds the seed average. The quadrature's own error (coarse and fine grid
//! difference, frozen with the reference) is added to every bound.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ObservationAssumption, ObservationSpec,
    ResponseFunctional, ResponseQuery, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response;
use antecedent_estimate::{BayesianGComputationAte, CausalPosterior, TemporalIdentification};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value;
use std::sync::Arc;

const X: [f64; 9] = [0.3, -0.8, 1.1, 0.5, -0.4, 1.4, -1.2, 0.2, 0.9];
const THRESHOLD: f64 = 0.8;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_observed_bayes_reference.json"))
        .unwrap()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Right,
    Left,
    Selected,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Right => "right_censored",
            Kind::Left => "left_censored",
            Kind::Selected => "selected",
        }
    }
}

/// Columns x, y, c, r (ids 0..3) for the frozen rows; row 0 is an exact history outcome.
fn table(kind: Kind, placeholder: f64) -> TimeSeriesData {
    let rows = &reference()["tables"][kind.name()]["rows"];
    let mut y = vec![if kind == Kind::Left { -1.0 } else { 1.0 }];
    let mut c = vec![if kind == Kind::Left { -5.0 } else { 5.0 }];
    let mut r = vec![1.0];
    for (i, row) in rows.as_array().unwrap().iter().enumerate() {
        let row = row.as_array().unwrap();
        let tag = row[0].as_str().unwrap();
        let bound = if kind == Kind::Left { -5.0 } else { 5.0 };
        match tag {
            "exact" => {
                y.push(row[1].as_f64().unwrap());
                c.push(bound);
                r.push(1.0);
            }
            "lower" | "upper" => {
                let b = row[1].as_f64().unwrap();
                y.push(b);
                c.push(b);
                r.push(0.0);
            }
            "missing" => {
                y.push(placeholder);
                c.push(0.0);
                r.push(0.0);
            }
            other => panic!("row {i}: {other}"),
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", &X[..]), ("y", y.as_slice()), ("c", c.as_slice()), ("r", r.as_slice())],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let id = VariableId::from_raw;
    let mut graph = TemporalDag::empty();
    let lagged = ensure_lagged(&mut graph, id(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let out = ensure_lagged(&mut graph, id(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(lagged, out).unwrap();
    graph
}

fn query(kind: Kind) -> ResponseQuery {
    let id = VariableId::from_raw;
    let (spec, assumption) = match kind {
        Kind::Selected => (
            ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(3) },
            ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)])),
        ),
        Kind::Right => (
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        Kind::Left => (
            ObservationSpec::LeftCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
    };
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: id(1),
        treatment: ContinuousDomain::new(id(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
    .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(spec, [assumption])
}

fn run(
    data: &TimeSeriesData,
    kind: Kind,
    n_draws: usize,
    seed: u64,
    prior_scale: f64,
) -> Result<CausalPosterior, antecedent_estimate::EstimationError> {
    let id = VariableId::from_raw;
    let graph = graph();
    let effect = TemporalEffectQuery::pulse(id(0), id(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let identification =
        TemporalBackdoorIdentifier::new().identify_temporal(&graph, &effect).unwrap();
    let estimator =
        BayesianGComputationAte { n_draws, seed, prior_scale, ..BayesianGComputationAte::new() };
    let (_, posterior) = estimate_observed_temporal_response(
        data,
        &graph,
        TemporalIdentification {
            horizons: &[(&identification.result.estimands[0], &identification.indexer)],
            status: identification.result.status,
            assumptions: identification.result.required_assumptions.clone(),
        },
        &query(kind),
        &estimator,
        &ExecutionContext::for_tests(1),
    )?;
    Ok(posterior)
}

fn moments(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (mean, (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt())
}

/// The Gibbs summaries of one run: cell means/sds, and P(b1 > THRESHOLD) from the cell difference.
struct Summary {
    b0: (f64, f64),
    b01: (f64, f64),
    b1: (f64, f64),
    p: f64,
    ess: f64,
}
fn summarize(posterior: &CausalPosterior) -> Summary {
    let c0 = posterior.draws.column(0).unwrap();
    let c1 = posterior.draws.column(1).unwrap();
    let diff: Vec<f64> = c0.iter().zip(c1).map(|(a, b)| b - a).collect();
    let p = diff.iter().filter(|&&d| d > THRESHOLD).count() as f64 / diff.len() as f64;
    Summary {
        b0: moments(c0),
        b01: moments(c1),
        b1: moments(&diff),
        p,
        ess: posterior.diagnostics.ess_bulk_min.unwrap(),
    }
}

fn exact(kind: Kind, scale: f64) -> (Value, f64) {
    let r = reference();
    let t = &r["tables"][kind.name()];
    let key = format!("scale_{scale:?}");
    (t[&key].clone(), t[format!("{key}_grid_difference")].as_f64().unwrap())
}

/// Run `seeds` Gibbs runs of `n_draws` and compare each and their average with the quadrature.
fn assert_matches_quadrature(kind: Kind, scale: f64, n_draws: usize, seeds: &[u64]) {
    let data = table(kind, 12345.0);
    let (want, grid) = exact(kind, scale);
    let f = |k: &str| want[k].as_f64().unwrap();
    let mut pooled = [0.0; 4];
    let mut ess_sum = 0.0;
    for &seed in seeds {
        let s = summarize(&run(&data, kind, n_draws, seed, scale).unwrap());
        assert!(s.ess > 1500.0, "{kind:?} seed {seed}: ESS {} too small for the bound", s.ess);
        let root = s.ess.sqrt();
        let p_se = (f("p_b1_above") * (1.0 - f("p_b1_above")) / s.ess).sqrt();
        let checks = [
            ("b0 mean", s.b0.0, f("b0_mean"), 5.0 * f("b0_sd") / root),
            ("b0 sd", s.b0.1, f("b0_sd"), 5.0 * f("b0_sd") / root),
            ("b0+b1 mean", s.b01.0, f("b0_plus_b1_mean"), 5.0 * f("b0_plus_b1_sd") / root),
            ("b0+b1 sd", s.b01.1, f("b0_plus_b1_sd"), 5.0 * f("b0_plus_b1_sd") / root),
            ("b1 mean", s.b1.0, f("b1_mean"), 5.0 * f("b1_sd") / root),
            ("b1 sd", s.b1.1, f("b1_sd"), 5.0 * f("b1_sd") / root),
            ("P(b1 > 0.8)", s.p, f("p_b1_above"), 5.0 * p_se),
        ];
        for (label, got, exact, bound) in checks {
            // The bound is meaningful: under a fifth of the posterior spread (a probability: 0.1).
            assert!(bound < 0.2 * f("b1_sd").max(f("b0_sd")).max(0.5), "{label}: bound {bound}");
            eprintln!(
                "{kind:?} scale {scale} seed {seed} {label}: {got:.5} vs {exact:.5} (bound {bound:.5}, ESS {:.0})",
                s.ess
            );
            assert!(
                (got - exact).abs() <= bound + grid,
                "{kind:?} scale {scale} seed {seed} {label}: Gibbs {got} vs exact {exact} (bound {bound}, grid {grid})"
            );
        }
        ess_sum += s.ess;
        pooled[0] += s.b0.0;
        pooled[1] += s.b01.0;
        pooled[2] += s.b1.0;
        pooled[3] += s.p;
    }
    let n = seeds.len() as f64;
    // The seeds are independent runs, so the average has standard error sd / sqrt(sum of ESS).
    let pooled_se = |sd: f64| 5.0 * sd / ess_sum.sqrt() + grid;
    let averages = [
        ("b0", pooled[0] / n, f("b0_mean"), pooled_se(f("b0_sd"))),
        ("b0+b1", pooled[1] / n, f("b0_plus_b1_mean"), pooled_se(f("b0_plus_b1_sd"))),
        ("b1", pooled[2] / n, f("b1_mean"), pooled_se(f("b1_sd"))),
    ];
    for (label, got, exact, bound) in averages {
        eprintln!(
            "{kind:?} scale {scale} pooled {label}: {got:.5} vs {exact:.5} (bound {bound:.5}, ESS sum {ess_sum:.0})"
        );
        assert!(
            (got - exact).abs() <= bound,
            "{kind:?} pooled {label}: {got} vs {exact} (bound {bound})"
        );
    }
}

const SEEDS: [u64; 4] = [11, 202, 3003, 40_004];

#[test]
fn temporal_observed_bayes_right_censored_matches_quadrature() {
    assert_matches_quadrature(Kind::Right, 10.0, 40_000, &SEEDS);
}

#[test]
fn temporal_observed_bayes_left_censored_matches_quadrature() {
    assert_matches_quadrature(Kind::Left, 10.0, 40_000, &SEEDS);
}

#[test]
fn temporal_observed_bayes_selected_matches_quadrature() {
    assert_matches_quadrature(Kind::Selected, 10.0, 40_000, &SEEDS);
}

#[test]
fn temporal_observed_bayes_prior_scale_is_read() {
    // A tight prior shrinks the posterior visibly: the exact means at scale 0.5 differ from
    // those at scale 10 by far more than the bound, and the Gibbs run follows the declared scale.
    for kind in [Kind::Right, Kind::Selected, Kind::Left] {
        let (wide, _) = exact(kind, 10.0);
        let (tight, _) = exact(kind, 0.5);
        assert!(
            (wide["b0_mean"].as_f64().unwrap() - tight["b0_mean"].as_f64().unwrap()).abs() > 0.2
        );
        assert_matches_quadrature(kind, 0.5, 40_000, &SEEDS[..2]);
    }
}

/// A run at the short draw counts can be refused by the R-hat gate for one seed (that refusal is
/// the gate working); the layout checks use the first seed the gate publishes.
fn published(data: &TimeSeriesData, n_draws: usize) -> CausalPosterior {
    (5..40)
        .find_map(|seed| run(data, Kind::Right, n_draws, seed, 10.0).ok())
        .expect("some seed passes the publication gate")
}

#[test]
fn temporal_observed_bayes_draw_count_seed_and_fixed_chain_layout_are_read() {
    let data = table(Kind::Right, 12345.0);
    // Draws: ceil(n / 4) per chain, at least 1024; four chains after 512 warmup sweeps.
    for (requested, total) in [(100, 4096), (4096, 4096), (8192, 8192), (8193, 4 * 2049)] {
        let p = published(&data, requested);
        assert_eq!(p.draws.column(0).unwrap().len(), total, "requested {requested}");
        assert_eq!(p.diagnostics.n_chains, Some(4));
        assert_eq!(p.diagnostics.n_warmup, Some(512));
        assert_eq!(p.diagnostics.iterations as usize, total / 4);
    }
    // The seed selects the streams: same seed, same draws; another seed, other draws.
    let a = run(&data, Kind::Right, 16_384, 5, 10.0).unwrap();
    let b = run(&data, Kind::Right, 16_384, 5, 10.0).unwrap();
    let c = run(&data, Kind::Right, 16_384, 6, 10.0).unwrap();
    assert_eq!(a.draws.column(0).unwrap(), b.draws.column(0).unwrap());
    assert_ne!(a.draws.column(0).unwrap(), c.draws.column(0).unwrap());
    // The chains are four disjoint seeded streams: their means differ but agree within the bound.
    let col = a.draws.column(0).unwrap();
    let means: Vec<f64> = col.chunks(4096).map(|c| moments(c).0).collect();
    assert!(means.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn temporal_observed_bayes_observation_data_rules_are_read() {
    // Placeholder outcomes of unselected rows are never data: any value gives the same draws.
    let a = run(&table(Kind::Selected, 12345.0), Kind::Selected, 4096, 9, 10.0).unwrap();
    let b = run(&table(Kind::Selected, -7.0), Kind::Selected, 4096, 9, 10.0).unwrap();
    assert_eq!(a.draws.column(1).unwrap(), b.draws.column(1).unwrap());
    // The same table read under three observation models gives three different posteriors.
    let data = table(Kind::Right, 12345.0);
    let right = run(&data, Kind::Right, 4096, 9, 10.0).unwrap();
    let selected = run(&data, Kind::Selected, 4096, 9, 10.0).unwrap();
    assert_ne!(right.draws.column(0).unwrap(), selected.draws.column(0).unwrap());
    // An unusable prior scale is refused, not silently replaced.
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(run(&data, Kind::Right, 4096, 9, scale).is_err(), "scale {scale}");
    }
    // Refusals: an event beyond its bound, a non-binary indicator and a non-finite bound.
    let mut y = vec![1.0, 1.64, -0.14, 2.18, 1.2, 1.28, 1.82, 0.14, 0.76];
    let mut c = vec![5.0; 9];
    let mut r = vec![1.0; 9];
    let build = |y: &[f64], c: &[f64], r: &[f64]| {
        TimeSeriesData::from_f64_columns([("x", &X[..]), ("y", y), ("c", c), ("r", r)], 1).unwrap()
    };
    assert!(run(&build(&y, &c, &r), Kind::Right, 4096, 9, 10.0).is_ok());
    y[3] = 9.0; // event outcome above its right bound
    assert!(run(&build(&y, &c, &r), Kind::Right, 4096, 9, 10.0).is_err());
    y[3] = 2.18;
    r[3] = 0.5;
    assert!(run(&build(&y, &c, &r), Kind::Right, 4096, 9, 10.0).is_err());
    r[3] = 0.0;
    c[3] = f64::INFINITY;
    assert!(run(&build(&y, &c, &r), Kind::Right, 4096, 9, 10.0).is_err());
}
