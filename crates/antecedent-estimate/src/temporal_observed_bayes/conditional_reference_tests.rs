//! The Gibbs full conditionals against their definitions.
//!
//! `sweep` must draw each latent outcome from its Markov-blanket conditional: the Gaussian
//! product of its own equation and every child equation, truncated to the side of its bound
//! (right censored: above, left censored: below) or untruncated when unselected. `Mechanism::draw`
//! must draw `(beta, s2)` from the normal/inverse-gamma conditional of the complete table.
//! `oracle_data/temporal_observed_bayes_reference.py` derives the conditional moments in closed
//! form; each check allows five Monte Carlo standard errors of independent draws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::Lag;
use antecedent_graph::ensure_lagged;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/oracle_data/temporal_observed_bayes_reference.json"
    ))
    .unwrap()
}

fn mechanism(
    variable: VariableId,
    parents: Vec<(VariableId, usize)>,
    beta: Vec<f64>,
    variance: f64,
) -> Mechanism {
    let history = parents.iter().map(|(_, lag)| *lag).max().unwrap_or(0);
    Mechanism {
        variable,
        parents,
        history,
        beta,
        variance,
        matrix: vec![],
        design_initialized: false,
        design_varies: true,
        y: vec![],
        prior: PriorSet::weakly_informative(2),
        workspace: LaplaceWorkspace::default(),
    }
}

fn observation(side: &str, bound: f64) -> Observation {
    match side {
        "lower" => Observation::Lower(bound),
        "upper" => Observation::Upper(bound),
        _ => Observation::Missing,
    }
}

/// Mean and second raw moment of `n` conditional draws of the single latent row, with the
/// five-standard-error half-widths of both (second moment: from the sample sd of y^2).
fn sampled(
    models: &[Mechanism],
    data: &mut BTreeMap<VariableId, Vec<f64>>,
    observations: &[Observation],
    row: usize,
    initial_variance: f64,
    seed: u64,
) -> [(f64, f64); 2] {
    let outcome = VariableId::from_raw(0);
    let n = 200_000usize;
    let mut rng = CausalRng::from_seed(seed);
    let (mut s1, mut s2, mut s4) = (0.0, 0.0, 0.0);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for _ in 0..n {
        sweep(data, outcome, observations, models, initial_variance, &mut rng).unwrap();
        let y = data[&outcome][row];
        lo = lo.min(y);
        hi = hi.max(y);
        s1 += y;
        s2 += y * y;
        s4 += y.powi(4);
    }
    let n = n as f64;
    let (m1, m2) = (s1 / n, s2 / n);
    let se1 = ((m2 - m1 * m1).max(0.0) / n).sqrt();
    let se2 = ((s4 / n - m2 * m2).max(0.0) / n).sqrt();
    // The draws lie on the admissible side of the bound.
    match observations[row] {
        Observation::Lower(b) => assert!(lo >= b, "draw {lo} below its lower bound {b}"),
        Observation::Upper(b) => assert!(hi <= b, "draw {hi} above its upper bound {b}"),
        _ => {}
    }
    [(m1, 5.0 * se1), (m2, 5.0 * se2)]
}

#[test]
fn latent_conditionals_match_the_closed_form_markov_blanket() {
    let outcome = VariableId::from_raw(0);
    let reference = reference();
    for (name, case) in reference["conditionals"].as_object().unwrap() {
        if name == "initial_history_lower" {
            continue;
        }
        let f = |k: &str| case[k].as_f64().unwrap();
        // Own equation: y ~ N(prior_mean, prior_var), no parents. Child k is a variable z_k
        // with z_k ~ N(a + b y, w) observed at the same row.
        let mut models = vec![mechanism(outcome, vec![], vec![f("prior_mean")], f("prior_var"))];
        let mut data = BTreeMap::from([(outcome, vec![0.0])]);
        for (k, child) in case["children"].as_array().unwrap().iter().enumerate() {
            let c: Vec<f64> =
                child.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let z = VariableId::from_raw(10 + k as u32);
            models.push(mechanism(z, vec![(outcome, 0)], vec![c[1], c[2]], c[3]));
            data.insert(z, vec![c[0]]);
        }
        let obs = [observation(case["side"].as_str().unwrap(), f("bound"))];
        let [(m1, b1), (m2, b2)] = sampled(&models, &mut data, &obs, 0, 1.0e9, 7);
        assert!((m1 - f("mean")).abs() <= b1, "{name}: mean {m1} vs {} (bound {b1})", f("mean"));
        assert!(
            (m2 - f("second")).abs() <= b2,
            "{name}: second moment {m2} vs {} (bound {b2})",
            f("second")
        );
        // The bounds are tight: under a tenth of the conditional's standard deviation.
        let sd = (f("second") - f("mean") * f("mean")).sqrt();
        assert!(
            b1 < 0.05 * sd + 1e-3 && b2 < 0.2 * sd.max(1.0) * (1.0 + f("mean").abs()),
            "{name}"
        );
    }
}

#[test]
fn latent_initial_history_uses_the_declared_prior_and_the_first_child() {
    // Row 0 lies before the outcome equation's history (one lag): its law is N(0, initial
    // variance) times the child equation of row 1, truncated below the bound.
    let outcome = VariableId::from_raw(0);
    let case = &reference()["conditionals"]["initial_history_lower"];
    let f = |k: &str| case[k].as_f64().unwrap();
    let child: Vec<f64> =
        case["children"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let model = mechanism(outcome, vec![(outcome, 1)], vec![child[1], child[2]], child[3]);
    let mut data = BTreeMap::from([(outcome, vec![0.0, child[0]])]);
    let obs = [Observation::Lower(f("bound")), Observation::Exact(child[0])];
    let [(m1, b1), (m2, b2)] =
        sampled(std::slice::from_ref(&model), &mut data, &obs, 0, f("prior_var"), 11);
    assert!((m1 - f("mean")).abs() <= b1, "mean {m1} vs {} (bound {b1})", f("mean"));
    assert!((m2 - f("second")).abs() <= b2, "second {m2} vs {} (bound {b2})", f("second"));
}

#[test]
fn coefficient_and_variance_draws_match_the_normal_inverse_gamma_conditional() {
    let id = VariableId::from_raw;
    let reference = reference();
    let x: Vec<f64> = vec![0.3, -0.8, 1.1, 0.5, -0.4, 1.4, -1.2, 0.2, 0.9];
    let mut y = vec![1.0];
    y.extend([1.64, -0.14, 2.18, 1.2, 1.28, 1.82, 0.14, 0.76]);
    let data =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let lagged = ensure_lagged(&mut graph, id(0), Lag::from_raw(1)).unwrap();
    let out = ensure_lagged(&mut graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(lagged, out).unwrap();
    let values = BTreeMap::from([(id(0), x), (id(1), y)]);
    for scale in [10.0, 0.5] {
        let want = &reference["conjugate"][format!("scale_{scale:?}")];
        let mut models = mechanisms(&data, &graph, id(1), scale).unwrap();
        assert_eq!(models.len(), 1);
        let mut rng = CausalRng::from_seed(5);
        let n = 100_000usize;
        let (mut b0, mut b1, mut s2) = (vec![], vec![], vec![]);
        for _ in 0..n {
            models[0].draw(&values, &mut rng).unwrap();
            b0.push(models[0].beta[0]);
            b1.push(models[0].beta[1]);
            s2.push(models[0].variance);
        }
        let stats = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / n as f64;
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            (m, var)
        };
        let mean = |k: &str| want["beta_mean"][k.parse::<usize>().unwrap()].as_f64().unwrap();
        for (label, draws, exact_mean, exact_var) in [
            ("b0", &b0, mean("0"), want["beta0_var"].as_f64().unwrap()),
            ("b1", &b1, mean("1"), want["beta1_var"].as_f64().unwrap()),
        ] {
            let (m, var) = stats(draws);
            assert!(
                (m - exact_mean).abs() <= 5.0 * (exact_var / n as f64).sqrt(),
                "scale {scale} {label}: mean {m} vs {exact_mean}"
            );
            // Student-t marginal with 8 degrees of freedom: kurtosis 4.5, so the sample variance
            // has relative standard error sqrt(3.5 / n).
            assert!(
                (var / exact_var - 1.0).abs() <= 5.0 * (3.5 / n as f64).sqrt(),
                "scale {scale} {label}: variance {var} vs {exact_var}"
            );
        }
        // Inverse-gamma with shape 4.001 has a finite mean and, barely, a finite variance; the
        // mean bound uses the sample sd.
        let (m, var) = stats(&s2);
        let exact = want["sigma2_mean"].as_f64().unwrap();
        assert!(
            (m - exact).abs() <= 5.0 * (var / n as f64).sqrt(),
            "scale {scale}: sigma2 mean {m} vs {exact}"
        );
    }
}
