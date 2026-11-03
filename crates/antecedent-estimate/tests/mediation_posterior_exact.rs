//! The nonlinear-mediation posterior means, from the route's own fitted design summaries,
//! against a ridge solution built here.
//!
//! The mediator model is `M ~ 1 + A + X` on the deterministic crossed grid of
//! `nonlinear_mediation.rs`: the expected posterior is a Gaussian-elimination ridge solution
//! built from the raw grid rows (intercept free, residual variance RSS/(n-3), penalty variance
//! over scale squared). The route's retained mediator summary must agree with those rows, and
//! the outcome posterior is a ridge solution of the route's retained outcome summary built by
//! the same elimination.

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::nonlinear_mediation::{
    NonlinearMediationConfig, NonlinearMediationEstimate, NonlinearMediationInput,
    NonlinearMediationPremises, estimate_nonlinear_mediation,
};
use antecedent_estimate::posterior_mean::{PosteriorRequest, mediation_coefficient_posterior};
use antecedent_estimate::vector_treatment::DesignSummary;

const XS: [f64; 5] = [-1.0, 0.0, 1.0, 2.0, 3.0];
const EX: f64 = 1.0;
const EX2: f64 = 3.0;
const EX3: f64 = 7.0;

#[derive(Clone, Copy)]
struct Scm {
    alpha: f64,
    gamma: f64,
    sigma2: f64,
    b0: f64,
    b1: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    delta: f64,
    bx: f64,
}

fn base_scm() -> Scm {
    Scm {
        alpha: 0.5,
        gamma: 0.5,
        sigma2: 0.64,
        b0: 1.0,
        b1: 0.8,
        c1: 0.6,
        c2: 0.4,
        c3: 0.0,
        delta: 0.3,
        bx: 0.7,
    }
}

struct Grid {
    a: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
}

fn grid(s: &Scm) -> Grid {
    let rows = 2.0 * 5.0 * 4.0;
    let v = s.sigma2 * (rows - 3.0) / rows;
    let lo = 0.4_f64;
    let hi = (2.0 * v - lo * lo).sqrt();
    let levels = [-hi, -lo, lo, hi];
    let mut g = Grid { a: Vec::new(), m: Vec::new(), y: Vec::new(), x: Vec::new() };
    for a in [0.0, 1.0] {
        for x in XS {
            for e in levels {
                let m = s.alpha * a + s.gamma * x + e;
                let y = s.b0
                    + s.b1 * a
                    + s.c1 * m
                    + s.c2 * m * m
                    + s.c3 * m * m * m
                    + s.delta * a * m
                    + s.bx * x;
                g.a.push(a);
                g.m.push(m);
                g.y.push(y);
                g.x.push(x);
            }
        }
    }
    g
}

fn config(nodes: usize) -> NonlinearMediationConfig {
    NonlinearMediationConfig {
        quadrature_nodes: nodes,
        bootstrap_replicates: 0,
        ..NonlinearMediationConfig::default()
    }
}

fn run(
    g: &Grid,
    premises: &NonlinearMediationPremises,
    cfg: &NonlinearMediationConfig,
) -> Result<NonlinearMediationEstimate, EstimationError> {
    let covs = [g.x.as_slice()];
    let input = NonlinearMediationInput {
        treatment: &g.a,
        mediator: &g.m,
        outcome: &g.y,
        covariates: &covs,
    };
    estimate_nonlinear_mediation(&input, premises, cfg, &ExecutionContext::for_tests(1))
}


fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                let delta = factor * a[col][k];
                a[row][k] -= delta;
            }
            let delta = factor * b[col];
            b[row] -= delta;
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Ridge from normal equations: residual variance `rss / (n - p)`, intercept free.
fn ridge_from_normal_equations(
    gram: &[f64],
    xty: &[f64],
    rss: f64,
    n: usize,
    scale: f64,
) -> Vec<f64> {
    let p = xty.len();
    let mut a: Vec<Vec<f64>> = (0..p).map(|i| gram[i * p..(i + 1) * p].to_vec()).collect();
    let penalty = rss / (n - p) as f64 / (scale * scale);
    for (j, row) in a.iter_mut().enumerate().skip(1) {
        row[j] += penalty;
    }
    solve(a, xty.to_vec())
}

fn normal_equations(rows: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let p = rows[0].len();
    let mut gram = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for (row, &yi) in rows.iter().zip(y) {
        for j in 0..p {
            xty[j] += row[j] * yi;
            for k in 0..p {
                gram[j * p + k] += row[j] * row[k];
            }
        }
    }
    let a: Vec<Vec<f64>> = (0..p).map(|i| gram[i * p..(i + 1) * p].to_vec()).collect();
    let beta = solve(a, xty.clone());
    let rss = rows
        .iter()
        .zip(y)
        .map(|(row, &yi)| {
            let fitted: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            (yi - fitted) * (yi - fitted)
        })
        .sum();
    (gram, xty, rss)
}

fn assert_summary_matches(summary: &DesignSummary, gram: &[f64], xty: &[f64], rss: f64) {
    assert_eq!(summary.gram.len(), gram.len());
    for (g, h) in summary.gram.iter().zip(gram) {
        assert!((g - h).abs() < 1e-8 * (1.0 + h.abs()), "gram {g} vs {h}");
    }
    for (g, h) in summary.xty.iter().zip(xty) {
        assert!((g - h).abs() < 1e-8 * (1.0 + h.abs()), "xty {g} vs {h}");
    }
    assert!((summary.rss - rss).abs() < 1e-8 * (1.0 + rss), "rss {} vs {rss}", summary.rss);
}

#[test]
fn mediation_posterior_means_equal_ridge_solutions_built_from_the_rows_and_summaries() {
    let s = base_scm();
    let g = grid(&s);
    let est = run(&g, &NonlinearMediationPremises::sequentially_ignorable(), &config(8))
        .expect("estimate");
    let n = g.a.len();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, g.a[i], g.x[i]]).collect();
    let (gram, xty, rss) = normal_equations(&rows, &g.m);
    // The mediator noise is nonzero, so the prior visibly moves the slopes.
    assert!(rss > 1.0);
    assert_summary_matches(&est.mediator_design, &gram, &xty, rss);
    for scale in [0.4, 1.0] {
        let (mediator, outcome) = mediation_coefficient_posterior(
            &est.mediator_design,
            &est.outcome_design,
            PosteriorRequest::IndependentNormal { scale },
        )
        .unwrap();
        let want_m = ridge_from_normal_equations(&gram, &xty, rss, n, scale);
        assert_eq!(mediator.len(), want_m.len());
        for (got, want) in mediator.iter().zip(&want_m) {
            assert!((got - want).abs() < 1e-9, "scale {scale}: mediator {got} vs {want}");
        }
        assert!((mediator[1] - s.alpha).abs() > 1e-3, "the prior must shrink alpha");
        let d = &est.outcome_design;
        let want_o = ridge_from_normal_equations(&d.gram, &d.xty, d.rss, d.n_rows, scale);
        assert_eq!(outcome.len(), want_o.len());
        for (got, want) in outcome.iter().zip(&want_o) {
            assert!((got - want).abs() < 1e-6 * (1.0 + want.abs()), "outcome {got} vs {want}");
        }
    }
}
