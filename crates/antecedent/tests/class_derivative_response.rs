//! Derivatives, elasticities, directional derivatives and Jacobians on a supplied Cpdag, Pag
//! and latent-confounded Admg.
//!
//! The data are a linear structural law with an observed confounder `z` of the treatment `a`
//! and the outcomes, so the structural derivative `d E[y | do(a)] / da` is the coefficient of
//! `a` in the regression of `y` on `(1, a, z)`. That coefficient is computed here by the normal
//! equations, independently of the library, and every functional is checked against the
//! reference it implies (identity, log-log, log-treatment and log-outcome scales, the average
//! derivative, a Jacobian and a directional derivative). The naive slope of `y` on `a` is
//! confounded and must not be what a route publishes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test scaffolding indexes small fixtures and one scenario is kept in one body"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, ResponseFunctional as F,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag};

const N: usize = 800;
/// Evaluation point of the point derivatives.
const AT: f64 = 2.0;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Columns `a, z, y, r, v, w` (indices 0..=5).
///
/// `r` is an exogenous cause of `a`; `w` is exogenous to `z`; a latent `l` moves both `w` and
/// `y`. `z = cos(.41 i) + .5 w`, `a = 2 + .6 z + .5 r + sin(.71 i) + .2 cos(.13 i)`,
/// `y = 5 + 2 a + 3 z + .7 l + e`, `v = 1 + .25 a - 1.5 z + e`.
struct Fixture {
    a: Vec<f64>,
    z: Vec<f64>,
    y: Vec<f64>,
    r: Vec<f64>,
    v: Vec<f64>,
    w: Vec<f64>,
}

impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            a: Vec::new(),
            z: Vec::new(),
            y: Vec::new(),
            r: Vec::new(),
            v: Vec::new(),
            w: Vec::new(),
        };
        for i in 0..N {
            let t = i as f64;
            let latent = (t * 0.77 + 2.0).cos();
            let w = (t * 1.7 + 0.3).sin() + 0.8 * latent;
            let r = (t * 0.29 + 1.0).sin();
            let z = (t * 0.41).cos() + 0.5 * w;
            let a = 2.0 + 0.6 * z + 0.5 * r + (t * 0.71).sin() + 0.2 * (t * 0.13).cos();
            let y = 5.0 + 2.0 * a + 3.0 * z + 0.7 * latent + 0.05 * (t * 3.1).cos();
            let v = 1.0 + 0.25 * a - 1.5 * z + 0.05 * (t * 2.7).sin();
            f.a.push(a);
            f.z.push(z);
            f.y.push(y);
            f.r.push(r);
            f.v.push(v);
            f.w.push(w);
        }
        f
    }

    fn data(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("a", self.a.as_slice()),
            ("z", self.z.as_slice()),
            ("y", self.y.as_slice()),
            ("r", self.r.as_slice()),
            ("v", self.v.as_slice()),
            ("w", self.w.as_slice()),
        ])
        .unwrap()
    }
}

/// Least-squares coefficients of `y` on an intercept and `regressors`, by the normal equations
/// and Gaussian elimination with partial pivoting.
fn ols(regressors: &[&[f64]], y: &[f64]) -> Vec<f64> {
    let k = regressors.len() + 1;
    let mut xtx = vec![vec![0.0; k + 1]; k];
    for i in 0..y.len() {
        let row: Vec<f64> =
            std::iter::once(1.0).chain(regressors.iter().map(|column| column[i])).collect();
        for p in 0..k {
            for q in 0..k {
                xtx[p][q] += row[p] * row[q];
            }
            xtx[p][k] += row[p] * y[i];
        }
    }
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&p, &q| xtx[p][col].abs().total_cmp(&xtx[q][col].abs()))
            .expect("non-empty pivot range");
        xtx.swap(col, pivot);
        for row in 0..k {
            if row != col {
                let factor = xtx[row][col] / xtx[col][col];
                for c in col..=k {
                    let value = xtx[col][c];
                    xtx[row][c] -= factor * value;
                }
            }
        }
    }
    (0..k).map(|p| xtx[p][k] / xtx[p][p]).collect()
}

/// The structural references every functional is compared with.
struct Reference {
    /// `d E[y | do(a)] / da` and `d E[v | do(a)] / da`.
    slope_y: f64,
    slope_v: f64,
    /// `E[y | do(a = AT)]` under the adjusted regression.
    level_y: f64,
    /// The confounded slope of `y` on `a` alone.
    naive_y: f64,
}

fn reference(f: &Fixture) -> Reference {
    let adjusted_y = ols(&[&f.a, &f.z], &f.y);
    let adjusted_v = ols(&[&f.a, &f.z], &f.v);
    let naive = ols(&[&f.a], &f.y);
    let mean_z = f.z.iter().sum::<f64>() / N as f64;
    Reference {
        slope_y: adjusted_y[1],
        slope_v: adjusted_v[1],
        level_y: adjusted_y[0] + adjusted_y[1] * AT + adjusted_y[2] * mean_z,
        naive_y: naive[1],
    }
}

/// Which structure carries the query.
#[derive(Clone, Copy, Debug)]
enum Structure {
    Cpdag,
    Pag,
    Admg,
}

/// `r -> a <- z`, `a -> y`, `z -> y`, `a -> v`, `z -> v` over columns `a, z, y, r, v` (a
/// v-structure, so every edge is compelled and the class has one completion).
fn cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

/// The same edges as a PAG with no circles: `a -> y` is visible because `r` points into `a`
/// and is not adjacent to `y`.
fn pag() -> Pag {
    let mut g = Pag::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

/// The same edges plus `w -> z` and a latent common cause of `w` and `y` (`w <-> y`). The
/// observed confounder `z` still separates `a` from that latent cause.
fn admg() -> Admg {
    let mut g = Admg::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4), (5, 1)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g.insert_bidirected(d(5), d(2)).unwrap();
    g
}

fn study(
    data: &TabularData,
    structure: Structure,
    accepted: bool,
    functional: &F,
    inference: InferenceMode,
) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = match (structure, accepted) {
        (Structure::Cpdag, false) => builder.graph(cpdag()),
        (Structure::Cpdag, true) => builder.graph(AcceptedGraph::from(cpdag())),
        (Structure::Pag, false) => builder.graph(pag()),
        (Structure::Pag, true) => builder.graph(AcceptedGraph::from(pag())),
        (Structure::Admg, false) => builder.graph(admg()),
        (Structure::Admg, true) => builder.graph(AcceptedGraph::from(admg())),
    };
    builder
        .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
        .response_options(ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..ContinuousResponseOptions::default()
        })
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The point value published by a result, as a flat vector.
fn published(result: &StudyResult, label: &str) -> Vec<f64> {
    let ResponseIdentification::PointIdentified(value) = &result.response().unwrap().estimate
    else {
        panic!("{label}: the response is not point identified");
    };
    match value {
        ResponseValue::Scalar(x) => vec![*x],
        ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => values.to_vec(),
        other => panic!("{label}: unexpected payload {other:?}"),
    }
}

/// The interval limits a result publishes, per coordinate.
fn interval(result: &StudyResult, label: &str) -> (Vec<f64>, Vec<f64>) {
    match &result.response().unwrap().uncertainty {
        ResponseUncertainty::Scalar { lower, upper, .. } => (vec![*lower], vec![*upper]),
        ResponseUncertainty::PointwiseBand { lower, upper, .. } => (lower.to_vec(), upper.to_vec()),
        other => panic!("{label}: expected a published interval, got {other:?}"),
    }
}

fn close(got: f64, want: f64, label: &str) {
    let tolerance = 0.12 * want.abs().max(0.5) + 0.1;
    assert!((got - want).abs() < tolerance, "{label}: {got} is not within {tolerance} of {want}");
}

fn point(scale: DerivativeScale) -> F {
    F::PointDerivative { outcome: v(2), treatment: v(0), at: AT, order: 1, scale }
}

fn ids(xs: &[u32]) -> Arc<[VariableId]> {
    xs.iter().map(|&x| v(x)).collect()
}

/// Every functional with the reference value its structural law implies.
fn cases(reference: &Reference) -> Vec<(&'static str, F, Vec<f64>)> {
    let slope = reference.slope_y;
    vec![
        ("point", point(DerivativeScale::Identity), vec![slope]),
        ("elasticity", point(DerivativeScale::LogLog), vec![slope * AT / reference.level_y]),
        ("semi_treatment", point(DerivativeScale::LogTreatment), vec![slope * AT]),
        ("semi_outcome", point(DerivativeScale::LogOutcome), vec![slope / reference.level_y]),
        (
            "average",
            F::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            },
            vec![slope],
        ),
        (
            "jacobian",
            F::Jacobian {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([AT]),
                scale: DerivativeScale::Identity,
            },
            vec![slope, reference.slope_v],
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([AT]),
                direction: Arc::from([1.0]),
            },
            vec![slope, reference.slope_v],
        ),
    ]
}

#[test]
fn the_reference_separates_the_structural_slope_from_the_confounded_one() {
    let reference = reference(&Fixture::new());
    assert!((reference.slope_y - 2.0).abs() < 0.05, "{}", reference.slope_y);
    assert!((reference.slope_v - 0.25).abs() < 0.05, "{}", reference.slope_v);
    assert!(
        (reference.naive_y - reference.slope_y).abs() > 0.5,
        "the confounder must bias the unadjusted slope: {} vs {}",
        reference.naive_y,
        reference.slope_y
    );
}

#[test]
fn frequentist_class_and_latent_confounded_derivatives_match_the_structural_reference() {
    let fixture = Fixture::new();
    let data = fixture.data();
    let reference = reference(&fixture);
    let ctx = ExecutionContext::for_tests(13);
    for structure in [Structure::Cpdag, Structure::Pag, Structure::Admg] {
        for (key, functional, want) in cases(&reference) {
            for accepted in [false, true] {
                let label = format!("{structure:?} {key} accepted={accepted}");
                let built =
                    study(&data, structure, accepted, &functional, InferenceMode::Frequentist);
                let fresh = built.run(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                let got = published(&fresh, &label);
                assert_eq!(got.len(), want.len(), "{label}");
                for (got, want) in got.iter().zip(&want) {
                    close(*got, *want, &label);
                }
                // The slope of an identity-scale first derivative is not the confounded one.
                if key == "point" || key == "average" {
                    assert!(
                        (got[0] - reference.naive_y).abs() > 0.3,
                        "{label}: {} is the confounded slope {}",
                        got[0],
                        reference.naive_y
                    );
                }
                // A single-model additive-GAM gradient (Jacobian, directional derivative) keeps
                // its frequentist band withheld until its coverage is measured: the point is
                // published and the withholding is disclosed.
                if matches!(structure, Structure::Admg) && matches!(key, "jacobian" | "directional")
                {
                    let response = fresh.response().unwrap();
                    assert!(
                        matches!(response.uncertainty, ResponseUncertainty::None),
                        "{label}: a withheld band is not published"
                    );
                    assert!(
                        response
                            .support
                            .warnings
                            .iter()
                            .any(|w| w.code.as_ref() == "response.derivative_interval_withheld"),
                        "{label}: the withheld interval is disclosed"
                    );
                    continue;
                }
                let (lower, upper) = interval(&fresh, &label);
                assert_eq!(lower.len(), got.len(), "{label}");
                for j in 0..got.len() {
                    assert!(
                        lower[j].is_nan() || upper[j].is_nan() || lower[j] <= upper[j],
                        "{label}"
                    );
                }
                // A class route is also served by the prepared handle; it must agree.
                if !matches!(structure, Structure::Admg) {
                    let prepared = built.prepare(&ctx).unwrap();
                    let click = prepared.estimate(&data, &ctx).unwrap();
                    assert_eq!(click.response().unwrap(), fresh.response().unwrap(), "{label}");
                    // An independent consumer reads the contracted result back and either
                    // verifies the program or names what it could not verify.
                    let artifact = prepared
                        .encode_contracted_result(&click, "class-derivative-response", &ctx)
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(
                        consumed.acceptance.accepts_as_verified_program()
                            || !consumed.acceptance.unresolved.is_empty(),
                        "{label}: the consumer must verify or explain"
                    );
                }
            }
        }
    }
}

#[test]
fn bayesian_class_and_latent_confounded_derivatives_publish_credible_intervals() {
    let fixture = Fixture::new();
    let data = fixture.data();
    let reference = reference(&fixture);
    let ctx = ExecutionContext::for_tests(19);
    for structure in [Structure::Cpdag, Structure::Pag, Structure::Admg] {
        for (key, functional, want) in cases(&reference) {
            let label = format!("bayesian {structure:?} {key}");
            let built = study(
                &data,
                structure,
                false,
                &functional,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)),
            );
            let result = built.run(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            let got = published(&result, &label);
            assert_eq!(got.len(), want.len(), "{label}");
            for (got, want) in got.iter().zip(&want) {
                close(*got, *want, &label);
            }
            let (lower, upper) = interval(&result, &label);
            for j in 0..got.len() {
                assert!(
                    lower[j].is_nan() || upper[j].is_nan() || lower[j] <= upper[j],
                    "{label}: [{}, {}]",
                    lower[j],
                    upper[j]
                );
            }
            let response = result.response().unwrap();
            let interpretation = match &response.uncertainty {
                ResponseUncertainty::Scalar { interpretation, .. }
                | ResponseUncertainty::PointwiseBand { interpretation, .. } => *interpretation,
                other => panic!("{label}: {other:?}"),
            };
            assert_eq!(
                interpretation,
                antecedent_core::IntervalInterpretation::Credible,
                "{label}: a posterior interval must be tagged credible"
            );
        }
    }
}

#[test]
fn a_derivative_with_latent_confounding_of_treatment_and_outcome_is_refused_with_its_boundary() {
    // `a -> y` with a latent common cause `a <-> y`: no adjustment set exists and the general
    // ID algorithm finds a hedge, so the derivative is not identified by any estimator
    // (Shpitser and Pearl 2006, Theorem 4). The route must refuse rather than publish the
    // confounded slope.
    let fixture = Fixture::new();
    let data = fixture.data();
    let mut graph = Admg::with_variables(6);
    graph.insert_directed(d(0), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let outcome = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(point(DerivativeScale::Identity))))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ctx));
    let error = outcome.expect_err("a hedge must not be estimated");
    assert!(error.is_identification_refusal(), "{error}");
}

/// `z -> t`, `z -> y`, `t -> y` with `z — t` undirected: completion `z -> t` adjusts `{z}`,
/// completion `t -> z` adjusts nothing. Columns `t, y, z`.
fn two_completion_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_directed(d(2), d(1)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_undirected(d(2), d(0)).unwrap();
    graph
}

fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            sum += (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        sum - 6.0
    }
}

#[test]
fn completions_that_disagree_publish_the_identified_set_of_the_independent_references() {
    // `z ~ N(0, 1)`, `t = .8 z + e`, `y = t + z + e`. Adjusting `{z}` gives slope 1; adjusting
    // nothing gives `1 + .8 / 1.64`. Both are computed from the data by the normal equations.
    let n = 4000;
    let mut g = gaussian(11);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    let adjusted = ols(&[&t, &z], &y)[1];
    let unadjusted = ols(&[&t], &y)[1];
    assert!(unadjusted - adjusted > 0.3, "the references must differ: {adjusted} {unadjusted}");
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let functional = F::PointDerivative {
        outcome: v(1),
        treatment: v(0),
        at: 0.0,
        order: 1,
        scale: DerivativeScale::Identity,
    };
    let result = Study::tabular(data)
        .graph(two_completion_cpdag())
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .response_options(ContinuousResponseOptions {
            bandwidth: Some(0.8),
            ..ContinuousResponseOptions::default()
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let mixture = result.structural_response.as_ref().expect("completion mixture");
    let mut values: Vec<f64> = mixture
        .atoms
        .iter()
        .filter_map(|atom| match atom.value.as_ref() {
            Some(ResponseValue::Scalar(x)) => Some(*x),
            _ => None,
        })
        .collect();
    values.sort_by(f64::total_cmp);
    assert_eq!(values.len(), 2, "{values:?}");
    close(values[0], adjusted, "adjusted completion");
    close(values[1], unadjusted, "unadjusted completion");
    let ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(set)) =
        &result.response().unwrap().estimate
    else {
        panic!("disagreeing completions must publish their identified set");
    };
    assert!((set.lower[0] - values[0]).abs() < 1e-9 && (set.upper[0] - values[1]).abs() < 1e-9);
    // The sampling band of the identified set: the Imbens-Manski interval of the completions'
    // slopes, from their joint row influence. It contains the set, and each independent
    // reference is inside it.
    let ResponseUncertainty::Scalar { lower, upper, level, interpretation, .. } =
        &result.response().unwrap().uncertainty
    else {
        panic!("an identified set of unequal derivative completions must publish a sampling band");
    };
    assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Confidence);
    assert!(*level > 0.0 && *level < 1.0);
    assert!(*lower <= values[0] && *upper >= values[1], "[{lower}, {upper}] vs {values:?}");
    assert!(
        *lower <= adjusted && adjusted <= *upper && *lower <= unadjusted && unadjusted <= *upper
    );
    assert!(
        *upper - *lower > values[1] - values[0],
        "the band must widen the set by sampling error"
    );
    let interval = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set_interval)
        .expect("the identified-set interval is retained");
    assert_eq!(
        interval.method,
        antecedent_estimate::IdentifiedSetIntervalMethod::ImbensManskiInfluence
    );
    assert_eq!(interval.completions, 2);
}

/// `x -> m -> y` with a latent common cause of `x` and `y`, so no adjustment set exists and
/// the mean is identified only through the front door.
///
/// `x = 1 + sin(.71 i) + .2 cos(.13 i) + .8 l`, `m = .5 + .9 x + .5 sin(1.9 i)`,
/// `y = 3 + 2 m + .7 l + e` with the latent `l = cos(.77 i + 2)`; the structural slope of
/// `E[y | do(x)]` is `.9 * 2 = 1.8`.
fn front_door_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 900;
    let mut x = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let latent = (t * 0.77 + 2.0).cos();
        let xi = 1.0 + (t * 0.71).sin() + 0.2 * (t * 0.13).cos() + 0.8 * latent;
        let mi = 0.5 + 0.9 * xi + 0.5 * (t * 1.9).sin();
        x.push(xi);
        m.push(mi);
        y.push(3.0 + 2.0 * mi + 0.7 * latent + 0.05 * (t * 3.1).cos());
    }
    (x, m, y)
}

fn front_door_study(
    inference: InferenceMode,
    functional: &F,
    replicates: u32,
) -> (Study, TabularData) {
    let (x, m, y) = front_door_columns();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap();
    (study, data)
}

#[test]
fn a_front_door_identified_derivative_is_estimated_not_refused() {
    // Independent reference: the front-door product of the slope of the mediator on the
    // treatment and the coefficient of the mediator in the outcome regression given the
    // treatment, each from the normal equations. The unadjusted slope is confounded.
    let (x, m, y) = front_door_columns();
    let mediator_on_treatment = ols(&[&x], &m)[1];
    let outcome_on_mediator = ols(&[&m, &x], &y)[1];
    let front_door = mediator_on_treatment * outcome_on_mediator;
    let naive = ols(&[&x], &y)[1];
    assert!((front_door - 1.8).abs() < 0.05, "{front_door}");
    assert!((naive - front_door).abs() > 0.2, "the latent cause must bias the naive slope");
    let at = x.iter().sum::<f64>() / x.len() as f64;
    let point = F::PointDerivative {
        outcome: v(2),
        treatment: v(0),
        at,
        order: 1,
        scale: DerivativeScale::Identity,
    };
    let average = F::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    };
    let ctx = ExecutionContext::for_tests(23);
    for (key, functional) in [("point", point), ("average", average)] {
        let (frequentist, _) = front_door_study(InferenceMode::Frequentist, &functional, 40);
        let result = frequentist.run(&ctx).unwrap();
        let got = published(&result, key)[0];
        assert!(
            (got - front_door).abs() < 0.15,
            "{key}: {got} vs front-door reference {front_door}"
        );
        assert!((got - naive).abs() > 0.1, "{key}: {got} is the confounded slope {naive}");
        let (lower, upper) = interval(&result, key);
        assert!(
            lower[0] <= got && got <= upper[0],
            "{key}: [{}, {}] around {got}",
            lower[0],
            upper[0]
        );
        assert!(upper[0] - lower[0] > 0.0);
        let (bayes, _) = front_door_study(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(40)),
            &functional,
            0,
        );
        let posterior = bayes.run(&ctx).unwrap();
        let mean = published(&posterior, key)[0];
        assert!((mean - front_door).abs() < 0.15, "{key}: posterior mean {mean}");
        let ResponseUncertainty::Scalar { interpretation, lower, upper, .. } =
            &posterior.response().unwrap().uncertainty
        else {
            panic!("{key}: a posterior interval is required");
        };
        assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Credible);
        assert!(lower <= upper);
    }
}

#[test]
fn a_front_door_derivative_can_be_prepared_and_refreshed() {
    let functional = F::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    };
    let (study, data) = front_door_study(InferenceMode::Frequentist, &functional, 0);
    let ctx = ExecutionContext::for_tests(29);
    let fresh = study.run(&ctx).unwrap();
    let click = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    assert_eq!(published(&click, "click"), published(&fresh, "fresh"));
}
