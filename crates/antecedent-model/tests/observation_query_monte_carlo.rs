//! Observation-conditioned queries and their parameter redraws against
//! independent Monte Carlo.
//!
//! The Gaussian conditionals of the Schur-complement route are checked against
//! regressions fitted on simulated draws of the structural equations (the
//! conditional mean of a jointly Gaussian outcome is the population regression
//! on the observed block, and the conditional variance is its residual
//! variance). The mixed observe-and-intervene redraw on the confounded finite
//! model is checked against Dirichlet draws and the truncated factorization
//! evaluated by hand in this file, with its own sampler.

#![allow(
    clippy::needless_range_loop,
    reason = "the oracle indexes small dense systems by position"
)]

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedLinearGaussian, ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
    TabularRow,
};
use antecedent_model::observation_query::condition_gaussian;
use antecedent_model::observation_uncertainty::{
    RedrawPrior, RedrawTarget, RetainedMechanism, TableCountRow, TableCounts, redraw_conditional,
};

/// A splitmix generator with Box-Muller normals and Marsaglia-Tsang gammas,
/// independent of the library's sampler.
struct Sampler(u64);

impl Sampler {
    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.bits() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// Gamma(shape >= 1, 1).
    fn gamma(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    fn dirichlet(&mut self, shapes: &[f64]) -> Vec<f64> {
        let g: Vec<f64> = shapes.iter().map(|s| self.gamma(*s)).collect();
        let total: f64 = g.iter().sum();
        g.iter().map(|v| v / total).collect()
    }
}

fn real_binding(var: &str) -> ParentBinding {
    ParentBinding {
        variable: var.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn context() -> MechanismContext {
    MechanismContext { population: "pop".into(), regime: "observational".into() }
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents.iter().map(|(p, _)| real_binding(p)).collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: context(),
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

const B: &[&str] = &["0", "1"];

fn table(var: &str, parents: &[&str], rows: &[(&[&str], [f64; 2])]) -> MechanismDescriptor {
    let levels: Vec<String> = B.iter().map(|s| (*s).to_owned()).collect();
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, probs)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(p, l)| ((*p).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: levels.iter().cloned().zip(probs.iter().copied()).collect(),
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|p| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(levels.clone()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|p| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(levels),
        context: context(),
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Declared,
            artifact_id: format!("declared:{var}"),
            evidence_id: format!("assumed:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: context(),
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: true,
    }
}

// ----------------------------------------------------------- Gaussian

/// X ~ N(1, 4); M = 0.5 + 2 X + e (sd 1); Y = -1 + 0.5 X + 3 M + 0.5 e.
fn gaussian_spec() -> ComposedModelSpec {
    spec_of(vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ])
}

/// One simulated row `(x, m, y)`; `set_x` replaces X's equation (an intervention).
fn simulate(rng: &mut Sampler, set_x: Option<f64>) -> [f64; 3] {
    let x = set_x.unwrap_or_else(|| 1.0 + 2.0 * rng.normal());
    let m = 0.5 + 2.0 * x + rng.normal();
    let y = -1.0 + 0.5 * x + 3.0 * m + 0.5 * rng.normal();
    [x, m, y]
}

/// Solve `a z = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let pivot = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, pivot);
        b.swap(c, pivot);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut z = vec![0.0; n];
    for r in (0..n).rev() {
        let tail: f64 = (r + 1..n).map(|k| a[r][k] * z[k]).sum();
        z[r] = (b[r] - tail) / a[r][r];
    }
    z
}

/// Least-squares regression of `target` on an intercept and `regressors`
/// evaluated at `at`: the fitted value there and the residual variance.
fn regression_at(rows: &[[f64; 3]], target: usize, regressors: &[usize], at: &[f64]) -> (f64, f64) {
    let k = regressors.len() + 1;
    let design = |row: &[f64; 3]| -> Vec<f64> {
        std::iter::once(1.0).chain(regressors.iter().map(|&j| row[j])).collect()
    };
    let mut gram = vec![vec![0.0; k]; k];
    let mut rhs = vec![0.0; k];
    for row in rows {
        let d = design(row);
        for a in 0..k {
            rhs[a] += d[a] * row[target];
            for b in 0..k {
                gram[a][b] += d[a] * d[b];
            }
        }
    }
    let beta = solve(gram, rhs);
    let ssr: f64 = rows
        .iter()
        .map(|row| {
            let fit: f64 = design(row).iter().zip(&beta).map(|(d, b)| d * b).sum();
            (row[target] - fit).powi(2)
        })
        .sum();
    let fitted = beta[0] + beta[1..].iter().zip(at).map(|(b, v)| b * v).sum::<f64>();
    (fitted, ssr / (rows.len() - k) as f64)
}

#[test]
fn gaussian_conditionals_match_regressions_on_simulated_structural_draws() {
    let ctx = ExecutionContext::for_tests(1);
    let model = ComposedLinearGaussian::compile(&gaussian_spec()).unwrap();
    let mut rng = Sampler(20_240_601);
    let observational: Vec<[f64; 3]> = (0..400_000).map(|_| simulate(&mut rng, None)).collect();

    // Y | M = 3: a one-regressor population regression.
    let exact = condition_gaussian(&model, "Y", &[], &[("M", 3.0)], &ctx).unwrap();
    let (mean, variance) = regression_at(&observational, 2, &[1], &[3.0]);
    assert!((mean - exact.mean).abs() < 0.05, "{mean} vs {}", exact.mean);
    assert!((variance / exact.variance - 1.0).abs() < 0.03, "{variance} vs {}", exact.variance);

    // M | X = 2, Y = 10: two observed coordinates, the 2x2 block of the Schur complement.
    let exact = condition_gaussian(&model, "M", &[], &[("X", 2.0), ("Y", 10.0)], &ctx).unwrap();
    let (mean, variance) = regression_at(&observational, 1, &[0, 2], &[2.0, 10.0]);
    assert!((mean - exact.mean).abs() < 0.01, "{mean} vs {}", exact.mean);
    assert!((variance / exact.variance - 1.0).abs() < 0.03, "{variance} vs {}", exact.variance);

    // do(X = 2) then observe M = 8: the cut graph, simulated with X's equation replaced.
    let cut: Vec<[f64; 3]> = (0..400_000).map(|_| simulate(&mut rng, Some(2.0))).collect();
    let exact = condition_gaussian(&model, "Y", &[("X", 2.0)], &[("M", 8.0)], &ctx).unwrap();
    let (mean, variance) = regression_at(&cut, 2, &[1], &[8.0]);
    assert!((mean - exact.mean).abs() < 0.02, "{mean} vs {}", exact.mean);
    assert!((variance / exact.variance - 1.0).abs() < 0.03, "{variance} vs {}", exact.variance);
    // Observing the intervened-on variable's descendant differs from conditioning on it
    // without the intervention: the two routes are not interchangeable.
    let plain = condition_gaussian(&model, "Y", &[], &[("M", 8.0)], &ctx).unwrap();
    assert!((plain.mean - exact.mean).abs() > 0.1);
}

// ------------------------------------------------------- mixed, finite

/// Fitted counts of the confounded model Z -> S, (S, Z) -> Y.
fn confounded_counts() -> Vec<RetainedMechanism> {
    let row = |parents: &[(&str, &str)], c0: u64, c1: u64| TableCountRow {
        parent_levels: parents.iter().map(|(p, l)| ((*p).into(), (*l).into())).collect(),
        counts: vec![("0".into(), c0), ("1".into(), c1)],
    };
    vec![
        RetainedMechanism::Table(TableCounts {
            variable: "Z".into(),
            rows: vec![row(&[], 40, 60)],
        }),
        RetainedMechanism::Table(TableCounts {
            variable: "S".into(),
            rows: vec![row(&[("Z", "0")], 45, 15), row(&[("Z", "1")], 12, 48)],
        }),
        RetainedMechanism::Table(TableCounts {
            variable: "Y".into(),
            rows: vec![
                row(&[("S", "0"), ("Z", "0")], 27, 3),
                row(&[("S", "0"), ("Z", "1")], 6, 6),
                row(&[("S", "1"), ("Z", "0")], 9, 6),
                row(&[("S", "1"), ("Z", "1")], 5, 43),
            ],
        }),
    ]
}

fn confounded_spec() -> ComposedModelSpec {
    spec_of(vec![
        table("Z", &[], &[(&[], [0.4, 0.6])]),
        table("S", &["Z"], &[(&["0"], [0.75, 0.25]), (&["1"], [0.2, 0.8])]),
        table(
            "Y",
            &["S", "Z"],
            &[
                (&["0", "0"], [0.9, 0.1]),
                (&["0", "1"], [0.5, 0.5]),
                (&["1", "0"], [0.6, 0.4]),
                (&["1", "1"], [0.1, 0.9]),
            ],
        ),
    ])
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

/// Posterior draws of `P(Z = 1 | do(S = 1), Y = 1)` by the truncated factorization over
/// independent Dirichlet rows, sampled and evaluated here.
fn monte_carlo_mixed(draws: usize, alpha: f64) -> Vec<f64> {
    let mut rng = Sampler(977);
    let shapes = |c0: u64, c1: u64| [c0 as f64 + alpha, c1 as f64 + alpha];
    (0..draws)
        .map(|_| {
            let z = rng.dirichlet(&shapes(40, 60));
            // Y's rows at S = 1, the value set by the intervention.
            let y_z0 = rng.dirichlet(&shapes(9, 6));
            let y_z1 = rng.dirichlet(&shapes(5, 43));
            let (w0, w1) = (z[0] * y_z0[1], z[1] * y_z1[1]);
            w1 / (w0 + w1)
        })
        .collect()
}

#[test]
fn mixed_observation_and_intervention_redraw_matches_independent_monte_carlo() {
    let ctx = ExecutionContext::for_tests(3);
    let prior = RedrawPrior { dirichlet_alpha: 1.0, ..RedrawPrior::default() };
    let target = RedrawTarget::Finite {
        outcome: "Z".into(),
        interventions: vec![("S".into(), "1".into())],
        observed: vec![("Y".into(), "1".into())],
    };
    let n = 20_000_u32;
    let draws =
        redraw_conditional(&confounded_spec(), &target, &confounded_counts(), &prior, n, 31, &ctx)
            .unwrap();
    assert_eq!(draws.quantities, ["probability:0", "probability:1"]);
    assert_eq!(draws.draws, n as usize);
    // The point value is the declared tables' hand enumeration:
    // weights 0.4 * 0.4 = 0.16 and 0.6 * 0.9 = 0.54.
    assert!((draws.point[1] - 0.54 / 0.70).abs() < 1e-12);
    // Each row is one distribution.
    for row in draws.values.chunks(2) {
        assert!((row[0] + row[1] - 1.0).abs() < 1e-12);
    }

    let reference = monte_carlo_mixed(n as usize, prior.dirichlet_alpha);
    let (ref_mean, ref_sd) = mean_sd(&reference);
    let (mean, sd) = mean_sd(&draws.column(1));
    let tolerance = 6.0 * ref_sd * (2.0 / f64::from(n)).sqrt();
    assert!((mean - ref_mean).abs() < tolerance, "{mean} vs {ref_mean} (+-{tolerance})");
    assert!((sd / ref_sd - 1.0).abs() < 0.05, "{sd} vs {ref_sd}");
    // The spread is real: the redraw is not the point value repeated.
    assert!(sd > 0.01);
}

#[test]
fn an_intervened_observation_redraw_differs_from_conditioning_without_the_intervention() {
    let ctx = ExecutionContext::for_tests(3);
    let prior = RedrawPrior::default();
    let run = |interventions: Vec<(String, String)>| {
        let target = RedrawTarget::Finite {
            outcome: "Z".into(),
            interventions,
            observed: vec![("Y".into(), "1".into())],
        };
        redraw_conditional(
            &confounded_spec(),
            &target,
            &confounded_counts(),
            &prior,
            4_000,
            2,
            &ctx,
        )
        .unwrap()
    };
    let set = run(vec![("S".into(), "1".into())]);
    let seen = run(vec![]);
    // Conditioning alone keeps the S-Z association, so P(Z = 1 | Y = 1) is not the
    // cut-graph value, in the point value and in the posterior mean.
    assert!((set.point[1] - seen.point[1]).abs() > 0.02);
    assert!((set.mean(1) - seen.mean(1)).abs() > 0.02);
}
