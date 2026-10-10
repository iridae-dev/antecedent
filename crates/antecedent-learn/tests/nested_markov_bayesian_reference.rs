//! Independent deterministic integration oracle: integrates each association g
//! analytically by polynomial-exact Gauss-Legendre quadrature, leaving FOUR
//! margins. No sampler, seed, proposal kernel, model probabilities, or production
//! summary code is called by this reference. Counts are one in every cell.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    reason = "independent small polynomial quadrature oracle"
)]
use antecedent_core::ExecutionContext;
use antecedent_learn::nested_markov_bayesian::{OUTPUTS, Options, Prior, fit};

fn gauss(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..64 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p;
            }
            derivative = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let step = p1 / derivative;
            z -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        out.push(((z + 1.0) / 2.0, 1.0 / ((1.0 - z * z) * derivative * derivative)));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}
#[derive(Clone)]
struct Term {
    coefficient: f64,
    factors: Vec<usize>,
}
fn terms(k: usize) -> Vec<Term> {
    let t = |coefficient, factors| Term { coefficient, factors };
    if k < 11 {
        return vec![t(1.0, vec![k])];
    }
    if k < 13 {
        let c = k - 10;
        return vec![t(1.0, vec![]), t(-1.0, vec![6]), t(1.0, vec![c, 6]), t(-1.0, vec![c, 5])];
    }
    vec![t(1.0, vec![2, 6]), t(-1.0, vec![2, 5]), t(-1.0, vec![1, 6]), t(1.0, vec![1, 5])]
}
fn moment(factors: &[usize], blockmean: &[f64; 8], blocksecond: &[[f64; 8]; 8]) -> f64 {
    let mut singleton = [0; 3];
    let mut block = Vec::new();
    for k in factors {
        if *k < 3 {
            singleton[*k] += 1;
        } else {
            block.push(*k - 3);
        }
    }
    let mut result = match block.len() {
        0 => 1.0,
        1 => blockmean[block[0]],
        2 => blocksecond[block[0]][block[1]],
        _ => panic!("at most quadratic margins"),
    };
    for (k, power) in singleton.iter().enumerate() {
        result *= match power {
            0 => 1.0,
            1 => 0.5,
            2 => {
                if k == 0 {
                    5.0 / 19.0
                } else {
                    3.0 / 11.0
                }
            }
            _ => panic!("quadratic singleton"),
        };
    }
    result
}
fn weighted_quantile(mut values: Vec<(f64, f64)>, p: f64) -> f64 {
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = values.iter().map(|v| v.1).sum::<f64>();
    let mut running = 0.0;
    for (x, w) in values {
        running += w;
        if running >= p * total {
            return x;
        }
    }
    panic!("positive mass")
}
struct Reference {
    mean: [f64; OUTPUTS],
    cov: Vec<f64>,
    effect_quantiles: [[f64; 2]; 3],
    margin_quantiles: [[f64; 2]; 2],
}
fn integrate(n: usize, q4_shape: f64) -> Reference {
    let nodes = gauss(n);
    let gn = gauss(4);
    let mut normalizer = 0.0;
    let mut first = [0.0; 8];
    let mut second = [[0.0; 8]; 8];
    let mut q4weights = vec![0.0; n * n];
    for &(a, wa) in &nodes {
        for &(b, wb) in &nodes {
            for (ci, &(c, wc)) in nodes.iter().enumerate() {
                for (di, &(d, wd)) in nodes.iter().enumerate() {
                    let margins = [a, b, c, d];
                    let mut z = [0.0; 4];
                    let mut m = [0.0; 4];
                    let mut s = [0.0; 4];
                    for i in 0..2 {
                        for j in 0..2 {
                            let k = i * 2 + j;
                            let low = (margins[i] + margins[2 + j] - 1.0).max(0.0);
                            let high = margins[i].min(margins[2 + j]);
                            for &(u, w) in &gn {
                                let g = low + (high - low) * u;
                                let likelihood = g
                                    * (margins[i] - g)
                                    * (margins[2 + j] - g)
                                    * (1.0 - margins[i] - margins[2 + j] + g);
                                let weight = (high - low) * w * likelihood;
                                z[k] += weight;
                                m[k] += weight * g;
                                s[k] += weight * g * g;
                            }
                        }
                    }
                    let w = wa
                        * wb
                        * wc
                        * wd
                        * c.powf(q4_shape - 1.0)
                        * d.powf(q4_shape - 1.0)
                        * z.iter().product::<f64>();
                    normalizer += w;
                    q4weights[ci * n + di] += w;
                    let mut bm = [0.0; 8];
                    bm[..4].copy_from_slice(&margins);
                    for k in 0..4 {
                        bm[4 + k] = m[k] / z[k];
                    }
                    for i in 0..8 {
                        first[i] += w * bm[i];
                        for j in 0..8 {
                            second[i][j] += w * if i == j && i >= 4 {
                                s[i - 4] / z[i - 4]
                            } else {
                                bm[i] * bm[j]
                            };
                        }
                    }
                }
            }
        }
    }
    for i in 0..8 {
        first[i] /= normalizer;
        for j in 0..8 {
            second[i][j] /= normalizer;
        }
    }
    let mut mean = [0.0; OUTPUTS];
    let mut cov = vec![0.0; OUTPUTS * OUTPUTS];
    for i in 0..OUTPUTS {
        for t in terms(i) {
            mean[i] += t.coefficient * moment(&t.factors, &first, &second);
        }
    }
    for i in 0..OUTPUTS {
        for j in 0..OUTPUTS {
            for t in terms(i) {
                for u in terms(j) {
                    let mut f = t.factors.clone();
                    f.extend(&u.factors);
                    cov[i * OUTPUTS + j] +=
                        t.coefficient * u.coefficient * moment(&f, &first, &second);
                }
            }
            cov[i * OUTPUTS + j] -= mean[i] * mean[j];
        }
    }
    let cn = gauss(16);
    let mut effects: [Vec<(f64, f64)>; 3] = std::array::from_fn(|_| Vec::new());
    let mut margins: [Vec<(f64, f64)>; 2] = std::array::from_fn(|_| Vec::new());
    for (i, &(q0, _)) in nodes.iter().enumerate() {
        for (j, &(q1, _)) in nodes.iter().enumerate() {
            let w = q4weights[i * n + j];
            margins[0].push((q0, w));
            margins[1].push((q1, w));
            for &(c0, w0) in &cn {
                for &(c1, w1) in &cn {
                    let weight = w * w0 * w1 * (c0 * (1.0 - c0) * c1 * (1.0 - c1)).powi(4);
                    let m0 = 1.0 - c0 * q0 - (1.0 - c0) * q1;
                    let m1 = 1.0 - c1 * q0 - (1.0 - c1) * q1;
                    for (k, v) in [m0, m1, m1 - m0].into_iter().enumerate() {
                        effects[k].push((v, weight));
                    }
                }
            }
        }
    }
    Reference {
        mean,
        cov,
        effect_quantiles: effects
            .map(|v| [weighted_quantile(v.clone(), 0.025), weighted_quantile(v, 0.975)]),
        margin_quantiles: margins
            .map(|v| [weighted_quantile(v.clone(), 0.025), weighted_quantile(v, 0.975)]),
    }
}
#[test]
fn low_count_full_joint_posterior_matches_independent_integration_and_prior_sensitivity() {
    let ctx = ExecutionContext::for_tests(0);
    let mut uniform_effect = 0.0;
    for shape in [1.0, 4.0] {
        let low = integrate(16, shape);
        let reference = integrate(24, shape);
        for k in 0..OUTPUTS {
            assert!((low.mean[k] - reference.mean[k]).abs() < 0.0005);
        }
        for k in 0..OUTPUTS * OUTPUTS {
            assert!((low.cov[k] - reference.cov[k]).abs() < 0.0003);
        }
        for k in 0..3 {
            for side in 0..2 {
                assert!(
                    (low.effect_quantiles[k][side] - reference.effect_quantiles[k][side]).abs()
                        < 0.008
                );
            }
        }
        let mut prior = Prior::default();
        prior.alpha[5] = shape;
        prior.alpha[6] = shape;
        let posterior = fit(
            &[1; 16],
            &prior,
            &Options { seed: 817, draws: 4096, warmup: 2048, ..Options::default() },
            &ctx,
        )
        .unwrap();
        for k in 0..OUTPUTS {
            assert!(
                (posterior.mean[k] - reference.mean[k]).abs() < 0.012,
                "shape={shape} param={k} sample={} oracle={}",
                posterior.mean[k],
                reference.mean[k]
            );
        }
        for k in 0..OUTPUTS * OUTPUTS {
            assert!((posterior.covariance[k] - reference.cov[k]).abs() < 0.002, "cov {k}");
        }
        for k in 0..3 {
            for side in 0..2 {
                assert!(
                    (posterior.credible[11 + k][side] - reference.effect_quantiles[k][side]).abs()
                        < 0.014,
                    "shape {shape} effect{k} side{side}: {:?} {:?}",
                    posterior.credible[11 + k],
                    reference.effect_quantiles[k]
                );
            }
        }
        for k in 0..2 {
            for side in 0..2 {
                assert!(
                    (posterior.credible[5 + k][side] - reference.margin_quantiles[k][side]).abs()
                        < 0.035
                );
            }
        }
        if shape == 1.0 {
            uniform_effect = posterior.mean[11];
        } else {
            assert!(
                uniform_effect - posterior.mean[11] > 0.04,
                "low-count prior must affect the actual posterior"
            );
        }
    }
}
