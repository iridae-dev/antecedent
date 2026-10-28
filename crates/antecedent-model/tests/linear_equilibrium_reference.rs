// SPDX-License-Identifier: MIT OR Apache-2.0
//! The linear equilibrium `X = KX + c + ε` against fixed-point references.
//!
//! The shipped path factors each strongly connected free block by Gauss-Jordan elimination.
//! The reference here never factors or inverts anything: it sums the Neumann series
//! `(I - K_FF)^{-1} = Σ K_FF^k` by repeated squaring, which converges geometrically on the
//! models drawn (every row's absolute sum stays below one) and is an independent
//! construction of the same operator. Means, covariances, block structure, condition numbers
//! and the singular-block verdicts are compared on seeded random cyclic and acyclic models
//! under every kind of hard clamp.

use antecedent_model::linear_solution::{hard_clamp_mean_batch, revalidate_free_block};
use antecedent_model::scc_compile::{
    CompileBudget, GaussianDisturbance, LinearSupportRoute, compile_gaussian_equilibrium,
    compile_routed_gaussian_equilibrium, route_linear_support,
};
use antecedent_model::{
    CyclicComponent, HardClamp, LinearStructuralMap, Uniqueness, assess,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

fn matmul(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        for k in 0..m {
            for j in 0..m {
                out[i * m + j] += a[i * m + k] * b[k * m + j];
            }
        }
    }
    out
}

/// `Σ_{k < 2^rounds} K^k`, the Neumann series of `(I - K)^{-1}`.
fn neumann_inverse(k: &[f64], m: usize, rounds: usize) -> Vec<f64> {
    let mut sum = vec![0.0; m * m];
    for i in 0..m {
        sum[i * m + i] = 1.0;
    }
    let mut power = k.to_vec();
    for _ in 0..rounds {
        let step = matmul(&power, &sum, m);
        for (s, d) in sum.iter_mut().zip(&step) {
            *s += d;
        }
        power = matmul(&power, &power, m);
    }
    sum
}

fn inf_norm(a: &[f64], m: usize) -> f64 {
    (0..m).map(|i| a[i * m..(i + 1) * m].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
}

struct Case {
    n: usize,
    k: Vec<f64>,
    c: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    clamp: Vec<usize>,
    values: Vec<f64>,
}

fn random_case(rng: &mut Lcg, n: usize, density: f64, rho: f64) -> Case {
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next() < density {
                k[i * n + j] = rng.signed();
            }
        }
        let row: f64 = k[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum();
        if row > rho {
            for v in &mut k[i * n..(i + 1) * n] {
                *v *= rho / row;
            }
        }
    }
    let c: Vec<f64> = (0..n).map(|_| 3.0 * rng.signed()).collect();
    let mu: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
    let a: Vec<f64> = (0..n * n).map(|_| rng.signed()).collect();
    let mut sigma = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                sigma[i * n + j] += a[i * n + l] * a[j * n + l];
            }
        }
        sigma[i * n + i] += 0.1;
    }
    let clamp: Vec<usize> = (0..n).filter(|_| rng.next() < 0.35).collect();
    let values = clamp.iter().map(|_| 4.0 * rng.signed()).collect();
    Case { n, k, c, mu, sigma, clamp, values }
}

/// Reference mean (clamped coordinates hold their values) and covariance (zero rows and
/// columns for clamped coordinates).
fn reference_moments(case: &Case, rounds: usize) -> (Vec<f64>, Vec<f64>) {
    let n = case.n;
    let free: Vec<usize> = (0..n).filter(|i| !case.clamp.contains(i)).collect();
    let m = free.len();
    let mut kff = vec![0.0; m * m];
    let mut sff = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (p, &i) in free.iter().enumerate() {
        for (q, &j) in free.iter().enumerate() {
            kff[p * m + q] = case.k[i * n + j];
            sff[p * m + q] = case.sigma[i * n + j];
        }
        rhs[p] = case.c[i] + case.mu[i];
        for (h, &node) in case.clamp.iter().enumerate() {
            rhs[p] += case.k[i * n + node] * case.values[h];
        }
    }
    let inv = neumann_inverse(&kff, m, rounds);
    let mut mean = vec![0.0; n];
    for (h, &node) in case.clamp.iter().enumerate() {
        mean[node] = case.values[h];
    }
    let mut cov = vec![0.0; n * n];
    for (p, &i) in free.iter().enumerate() {
        mean[i] = (0..m).map(|q| inv[p * m + q] * rhs[q]).sum();
        for (q, &j) in free.iter().enumerate() {
            let mut value = 0.0;
            for a in 0..m {
                for b in 0..m {
                    value += inv[p * m + a] * sff[a * m + b] * inv[q * m + b];
                }
            }
            cov[i * n + j] = value;
        }
    }
    (mean, cov)
}

/// Strongly connected components of the free support by transitive closure.
fn reference_components(case: &Case) -> Vec<Vec<usize>> {
    let n = case.n;
    let free: Vec<usize> = (0..n).filter(|i| !case.clamp.contains(i)).collect();
    let edge = |from: usize, into: usize| from != into && case.k[into * n + from] != 0.0;
    let mut reach = vec![vec![false; n]; n];
    for &a in &free {
        reach[a][a] = true;
        for &b in &free {
            if edge(a, b) {
                reach[a][b] = true;
            }
        }
    }
    for &k in &free {
        for &i in &free {
            for &j in &free {
                if reach[i][k] && reach[k][j] {
                    reach[i][j] = true;
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for &a in &free {
        if seen[a] {
            continue;
        }
        let members: Vec<usize> =
            free.iter().copied().filter(|&b| reach[a][b] && reach[b][a]).collect();
        for &b in &members {
            seen[b] = true;
        }
        out.push(members);
    }
    out
}

fn has_cycle(case: &Case) -> bool {
    reference_components(case).iter().any(|c| {
        c.len() > 1
    })
}

fn component_for(model: &LinearStructuralMap) -> CyclicComponent {
    let family = assess(model, &HardClamp::none(model.dimension()).unwrap()).unwrap().family;
    CyclicComponent::try_new("feedback", "source", &["note"], model, &family).unwrap()
}

fn relative_close(a: f64, b: f64, tolerance: f64, what: &str) {
    let scale = 1.0 + a.abs().max(b.abs());
    assert!((a - b).abs() <= tolerance * scale, "{what}: {a} vs {b}");
}

#[test]
fn compiled_equilibria_match_the_neumann_series_on_random_models_and_clamps() {
    let mut rng = Lcg(0x5CC);
    let mut cyclic_seen = 0;
    let mut acyclic_seen = 0;
    let mut clamped_seen = 0;
    for trial in 0..600 {
        let n = 1 + trial % 6;
        let density = [0.25, 0.5, 0.9][trial % 3];
        let case = random_case(&mut rng, n, density, 0.9);
        if case.clamp.len() == n {
            continue;
        }
        let model = LinearStructuralMap::try_new(case.k.clone(), case.c.clone()).unwrap();
        let clamp = HardClamp::try_new(n, &case.clamp).unwrap();
        let disturbance = GaussianDisturbance::try_new(case.mu.clone(), case.sigma.clone()).unwrap();
        let component = component_for(&model);
        let direct = compile_gaussian_equilibrium(
            &component,
            &model,
            &disturbance,
            &clamp,
            &case.values,
            CompileBudget::unlimited(),
        )
        .unwrap();
        let routed = compile_routed_gaussian_equilibrium(
            &component,
            &model,
            &disturbance,
            &clamp,
            &case.values,
            CompileBudget::unlimited(),
        )
        .unwrap();
        let (mean, cov) = reference_moments(&case, 12);
        for i in 0..n {
            relative_close(direct.mean()[i], mean[i], 1e-9, &format!("trial {trial} mean {i}"));
            relative_close(routed.equilibrium().mean()[i], mean[i], 1e-9, "routed mean");
            for j in 0..n {
                relative_close(direct.covariance()[i * n + j], cov[i * n + j], 1e-9, "covariance");
                relative_close(
                    routed.equilibrium().covariance()[i * n + j],
                    cov[i * n + j],
                    1e-9,
                    "routed covariance",
                );
            }
        }
        // The batch path solves the same clamp for the same means.
        if !case.clamp.is_empty() {
            clamped_seen += 1;
            let batch = hard_clamp_mean_batch(&model, &clamp, &case.values, &case.mu).unwrap();
            for i in 0..n {
                relative_close(batch.means()[i], mean[i], 1e-9, "batch mean");
            }
        }
        // Block structure is the strongly connected components of the free support.
        let mut expected = reference_components(&case);
        let mut got: Vec<Vec<usize>> = direct.blocks().iter().map(|b| b.members.clone()).collect();
        expected.sort();
        got.sort();
        assert_eq!(got, expected, "trial {trial}: blocks");
        // The route is acyclic exactly when no free block has more than one member.
        let route = route_linear_support(&model, &clamp).unwrap();
        match route {
            LinearSupportRoute::Acyclic { ref order } => {
                acyclic_seen += 1;
                assert!(!has_cycle(&case), "trial {trial}: acyclic route on a cyclic support");
                let position = |node: usize| order.iter().position(|&x| x == node).unwrap();
                for &i in order {
                    for &j in order {
                        if i != j && case.k[i * n + j] != 0.0 {
                            assert!(position(j) < position(i), "trial {trial}: order");
                        }
                    }
                }
                assert_eq!(order.len(), n - case.clamp.len());
            }
            LinearSupportRoute::Cyclic { .. } => {
                cyclic_seen += 1;
                assert!(has_cycle(&case), "trial {trial}: cyclic route on an acyclic support");
            }
        }
        // Condition numbers: closed form for orders 1 and 2, a lower bound otherwise.
        for block in direct.blocks() {
            let m = block.members.len();
            let mut a = vec![0.0; m * m];
            let mut kss = vec![0.0; m * m];
            for (p, &i) in block.members.iter().enumerate() {
                for (q, &j) in block.members.iter().enumerate() {
                    kss[p * m + q] = case.k[i * n + j];
                    a[p * m + q] = if p == q { 1.0 - kss[p * m + q] } else { -kss[p * m + q] };
                }
            }
            let truth = inf_norm(&a, m) * inf_norm(&neumann_inverse(&kss, m, 12), m);
            if block.exact {
                relative_close(block.kappa_inf, truth, 1e-9, "closed-form kappa");
            } else {
                assert!(block.kappa_inf <= truth * (1.0 + 1e-9), "estimate above the truth");
                assert!(block.kappa_inf * (m as f64) >= truth, "estimate {} vs {truth}", block.kappa_inf);
            }
        }
    }
    assert!(cyclic_seen > 100 && acyclic_seen > 50 && clamped_seen > 100);
}

#[test]
fn a_nearly_critical_two_cycle_keeps_its_closed_form_to_the_conditioning() {
    for product in [0.5f64, 0.9, 0.99, 0.999, 0.999_999] {
        // |K01 K10| = product < 1 with each row sum below one.
        let (a, b) = (product.sqrt(), product.sqrt());
        let k = vec![0.0, a, b, 0.0];
        let model = LinearStructuralMap::try_new(k.clone(), vec![1.0, -2.0]).unwrap();
        let sigma = vec![1.0, 0.3, 0.3, 2.0];
        let disturbance = GaussianDisturbance::try_new(vec![0.0, 0.0], sigma.clone()).unwrap();
        let clamp = HardClamp::none(2).unwrap();
        let law = compile_gaussian_equilibrium(
            &component_for(&model),
            &model,
            &disturbance,
            &clamp,
            &[],
            CompileBudget::unlimited(),
        )
        .unwrap();
        // Exact: (I-K)^{-1} = [[1, a],[b, 1]] / (1 - ab).
        let det = 1.0 - a * b;
        let inv = [1.0 / det, a / det, b / det, 1.0 / det];
        let mean = [inv[0] * 1.0 + inv[1] * -2.0, inv[2] * 1.0 + inv[3] * -2.0];
        for i in 0..2 {
            // Error scales with the conditioning 1/(1 - ab).
            let tolerance = 1e-15 / det;
            relative_close(law.mean()[i], mean[i], tolerance.max(1e-15), "mean");
        }
        let kappa = (1.0 + a).max(1.0 + b) * (1.0 + a).max(1.0 + b) / det;
        let exact_kappa = (1.0 + a).max(1.0 + b) * (inv[0].abs() + inv[1].abs()).max(inv[2].abs() + inv[3].abs());
        relative_close(law.blocks()[0].kappa_inf, exact_kappa, 1e-9 * (1.0 + kappa), "kappa");
    }
}

#[test]
fn singular_and_nonsingular_cycles_are_decided_by_the_determinant() {
    // Two-node cycle: det(I - K) = 1 - K01 K10.
    let clamp = HardClamp::none(2).unwrap();
    for (a, b, singular) in [
        (1.0, 1.0, true),
        (2.0, 0.5, true),
        (-1.0, -1.0, true),
        (-1.0, 1.0, false),
        (2.0, -2.0, false),
        (0.5, 3.0, false),
    ] {
        let model = LinearStructuralMap::try_new(vec![0.0, a, b, 0.0], vec![0.0, 0.0]).unwrap();
        let block = assess(&model, &clamp).unwrap().original;
        assert_eq!(
            matches!(block.uniqueness, Uniqueness::Singular(_) | Uniqueness::NumericallySingular { .. }),
            singular,
            "K01={a} K10={b}: {:?}",
            block.uniqueness
        );
        assert_eq!(revalidate_free_block(&model, &clamp).is_err(), singular);
    }
    // Three-node ring: det(I - K) = 1 - K01 K12 K20.
    for (product, singular) in [(1.0, true), (-1.0, false), (8.0, false), (0.125, false)] {
        let cube = f64::cbrt(product);
        let k = vec![0.0, cube, 0.0, 0.0, 0.0, cube, cube, 0.0, 0.0];
        // K[0][1], K[1][2], K[2][0] close the ring 1 -> 0, 2 -> 1, 0 -> 2.
        let model = LinearStructuralMap::try_new(k, vec![0.0; 3]).unwrap();
        let block = assess(&model, &HardClamp::none(3).unwrap()).unwrap().original;
        assert_eq!(
            matches!(block.uniqueness, Uniqueness::Singular(_) | Uniqueness::NumericallySingular { .. }),
            singular,
            "ring product {product}: {:?}",
            block.uniqueness
        );
    }
    // Clamping one ring node breaks the cycle: the free block is then a chain and unique.
    let model =
        LinearStructuralMap::try_new(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![0.0; 3])
            .unwrap();
    let clamped = HardClamp::try_new(3, &[1]).unwrap();
    assert!(revalidate_free_block(&model, &clamped).is_ok());
    assert!(revalidate_free_block(&model, &HardClamp::none(3).unwrap()).is_err());
}
