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
use antecedent_learn::nested_markov_bayesian::{OUTPUTS, Options, Prior, fit, probabilities};

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

/// Posterior means for an asymmetric count table, so that a swapped index (`X1`/`X3`,
/// `X2`/`X4`, or the cell order of `counts`) cannot cancel by symmetry as it does for the
/// all-ones table above. The reference values are from
/// `tests/oracle_data/nested_markov_asymmetric_means.py`: the singleton coordinates are
/// written down as Beta posteriors and the eight coupled coordinates are integrated by
/// quadrature (agreement between 32 and 48 margin nodes is about 1e-5). The `do(X2)` means
/// follow from independence of the `c` and `q4` blocks.
#[test]
fn asymmetric_counts_posterior_means_match_independent_quadrature() {
    let counts: [u64; 16] = [3, 1, 2, 5, 1, 4, 2, 1, 6, 2, 1, 3, 2, 1, 4, 2];
    let reference: [f64; OUTPUTS] = [
        0.476_190_476_190_476_16, // a
        0.52,                     // c0
        0.473_684_210_526_315_76, // c1
        0.579_274_240_452_427_9,  // q20
        0.552_023_413_415_480_8,  // q21
        0.585_038_393_290_0,      // q40
        0.450_696_311_434_962_5,  // q41
        0.439_015_118_629_108_4,  // g00
        0.193_930_740_050_160_05, // g01
        0.357_158_283_200_337_4,  // g10
        0.173_311_247_994_847_2,  // g11
        0.479_445_806_000_372_7,  // mean_do_x2_0
        0.485_667_965_581_031_2,  // mean_do_x2_1
        0.006_222_159_580_658_5,  // contrast
    ];
    let posterior = fit(
        &counts,
        &Prior::default(),
        &Options { seed: 4242, draws: 4096, warmup: 2048, ..Options::default() },
        &ExecutionContext::for_tests(0),
    )
    .unwrap();
    for k in 0..OUTPUTS {
        let tolerance = if k < 3 { 0.012 } else { 0.015 };
        assert!(
            (posterior.mean[k] - reference[k]).abs() < tolerance,
            "param {k}: sample {} oracle {}",
            posterior.mean[k],
            reference[k]
        );
    }
}

/// Every law the eleven coordinates can produce lies in the binary Verma nested model, with
/// cell index `i*8 + t*4 + j*2 + y` for `X1=i, X2=t, X3=j, X4=y`. The constraints are computed
/// here from the joint cells alone, by definition: `X3` is independent of `X1` given `X2`
/// (the ordinary conditional independence of the graph `1 -> 2 -> 3 -> 4`, `2 <-> 4`), and the
/// Verma equality says `sum_t p(t | i) p(y | i, t, j)` does not depend on `i`. The coordinates
/// also determine fifteen cell probabilities from eleven numbers, so the model is a strict
/// submodel: a generic table violates the Verma equality, which the last assertion confirms.
#[test]
fn coordinate_law_satisfies_the_verma_nested_markov_constraints() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let cell = |c: &[f64; 16], i: usize, t: usize, j: usize, y: usize| c[i * 8 + t * 4 + j * 2 + y];
    let mut accepted = 0;
    while accepted < 2000 {
        let mut p = [0.0; 11];
        for v in &mut p {
            *v = next();
        }
        let Some(c) = probabilities(&p) else { continue };
        accepted += 1;
        let p123 = |i: usize, t: usize, j: usize| (0..2).map(|y| cell(&c, i, t, j, y)).sum::<f64>();
        let p12 = |i: usize, t: usize| (0..2).map(|j| p123(i, t, j)).sum::<f64>();
        let p1 = |i: usize| (0..2).map(|t| p12(i, t)).sum::<f64>();
        for t in 0..2 {
            for j in 0..2 {
                let a = p123(0, t, j) / p12(0, t);
                let b = p123(1, t, j) / p12(1, t);
                assert!((a - b).abs() < 1e-12, "X3 depends on X1 given X2: {a} {b}");
            }
        }
        for j in 0..2 {
            for y in 0..2 {
                let verma = |i: usize| {
                    (0..2)
                        .map(|t| p12(i, t) / p1(i) * cell(&c, i, t, j, y) / p123(i, t, j))
                        .sum::<f64>()
                };
                assert!((verma(0) - verma(1)).abs() < 1e-12, "Verma equality fails at ({j},{y})");
            }
        }
    }
    // A generic positive table is not in the model.
    let mut generic = [0.0; 16];
    for (k, v) in generic.iter_mut().enumerate() {
        *v = 1.0 + ((k * 7) % 5) as f64;
    }
    let total: f64 = generic.iter().sum();
    for v in &mut generic {
        *v /= total;
    }
    let p123 = |i: usize, t: usize, j: usize| (0..2).map(|y| cell(&generic, i, t, j, y)).sum::<f64>();
    let p12 = |i: usize, t: usize| (0..2).map(|j| p123(i, t, j)).sum::<f64>();
    let a = p123(0, 0, 0) / p12(0, 0);
    let b = p123(1, 0, 0) / p12(1, 0);
    assert!((a - b).abs() > 1e-3, "the constraint check must be able to fail");
}
