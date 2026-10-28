//! `DirectLiNGAM` against structural models whose errors are exactly independent.
//!
//! Each error is a skewed or flat non-Gaussian marginal on a small grid and the data rows are
//! the full product of the grids, so the empirical joint distribution of the errors is exactly
//! the product of the marginals: the errors have zero sample covariance and zero sample distance
//! covariance with each other and with every variable that does not contain them. With such
//! data the exogeneity criterion is exact: a true source has residuals on every other variable
//! that are exactly independent of it, so its score is zero up to rounding while any other
//! candidate scores strictly positive. The search must return the generating causal order, and
//! because OLS on the true predecessors recovers the structural coefficients exactly, the
//! generating edges and their coefficients too. The distance correlation behind the score is
//! also compared with its O(n^2) definition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_stats::dcor::DistanceY;

use super::DirectLingam;
use crate::engine::DiscoveryWorkspace;
use crate::test_tables::continuous_table;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

/// Non-Gaussian marginal on `g` grid points, a different shape for each `shape`.
fn marginal(shape: usize, g: usize) -> Vec<f64> {
    let q = |k: usize| (k as f64 + 0.5) / g as f64;
    let raw: Vec<f64> = (0..g)
        .map(|k| match shape % 4 {
            0 => -(1.0 - q(k)).ln(),
            1 => q(k),
            2 => q(k).powi(3),
            _ => (q(k) / (1.0 - q(k))).ln().abs() * if k % 3 == 0 { -1.0 } else { 1.0 },
        })
        .collect();
    let mean = raw.iter().sum::<f64>() / g as f64;
    let sd = (raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / g as f64).sqrt();
    raw.iter().map(|v| (v - mean) / sd).collect()
}

struct Model {
    parents: Vec<Vec<(usize, f64)>>,
    order: Vec<usize>,
}

impl Model {
    fn columns(&self, g: usize, shapes: &[usize]) -> Vec<Vec<f64>> {
        let p = self.parents.len();
        let marginals: Vec<Vec<f64>> = (0..p).map(|v| marginal(shapes[v], g)).collect();
        let rows = g.pow(u32::try_from(p).unwrap());
        let mut cols: Vec<Vec<f64>> = vec![vec![0.0; rows]; p];
        for &v in &self.order {
            let stride = g.pow(u32::try_from(v).unwrap());
            for r in 0..rows {
                let mut x = marginals[v][(r / stride) % g];
                for &(par, b) in &self.parents[v] {
                    x += b * cols[par][r];
                }
                cols[v][r] = x;
            }
        }
        cols
    }

    fn random(rng: &mut Lcg, p: usize, density: u64) -> Self {
        let mut order: Vec<usize> = (0..p).collect();
        for i in (1..p).rev() {
            order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut parents = vec![Vec::new(); p];
        for i in 0..p {
            for j in (i + 1)..p {
                if rng.next() % 100 < density {
                    let sign = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
                    parents[order[j]].push((order[i], sign * (0.5 + 0.5 * rng.unit())));
                }
            }
        }
        Self { parents, order }
    }
}

#[test]
fn product_grid_errors_are_exactly_uncorrelated_and_independent() {
    let m = Model { parents: vec![vec![], vec![(0, 0.8)]], order: vec![0, 1] };
    let cols = m.columns(6, &[0, 1]);
    // The residual of x1 on x0 is the second error exactly.
    let n = cols[0].len() as f64;
    let (m0, m1) = (cols[0].iter().sum::<f64>() / n, cols[1].iter().sum::<f64>() / n);
    let sxy: f64 = cols[0].iter().zip(&cols[1]).map(|(a, b)| (a - m0) * (b - m1)).sum();
    let sxx: f64 = cols[0].iter().map(|a| (a - m0) * (a - m0)).sum();
    assert!((sxy / sxx - 0.8).abs() < 1e-12);
    let resid: Vec<f64> = cols[0].iter().zip(&cols[1]).map(|(a, b)| (b - m1) - 0.8 * (a - m0)).collect();
    let d = DistanceY::new(&cols[0]).correlate(&resid);
    assert!(d.abs() < 1e-7, "independent product errors have dcor {d}");
    // Dependence is detected in the wrong direction.
    let sxy_rev = sxy / cols[1].iter().map(|b| (b - m1) * (b - m1)).sum::<f64>();
    let rev: Vec<f64> = cols[0].iter().zip(&cols[1]).map(|(a, b)| (a - m0) - sxy_rev * (b - m1)).collect();
    assert!(DistanceY::new(&cols[1]).correlate(&rev) > 0.05);
}

/// Distance correlation from its definition: double-centred pairwise distance matrices.
fn dcor_definition(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let centred = |v: &[f64]| -> Vec<f64> {
        let d: Vec<f64> = (0..n * n).map(|k| (v[k / n] - v[k % n]).abs()).collect();
        let row: Vec<f64> = (0..n).map(|i| d[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
        let grand = row.iter().sum::<f64>() / n as f64;
        (0..n * n).map(|k| d[k] - row[k / n] - row[k % n] + grand).collect()
    };
    let (a, b) = (centred(x), centred(y));
    let mean = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(p, q)| p * q).sum::<f64>() / (n * n) as f64;
    let (vxy, vxx, vyy) = (mean(&a, &b), mean(&a, &a), mean(&b, &b));
    (vxy / (vxx * vyy).sqrt()).sqrt()
}

#[test]
fn distance_correlation_matches_its_definition() {
    let mut rng = Lcg(5);
    for (n, power) in [(40usize, 1.0), (57, 2.0), (80, 3.0)] {
        let x: Vec<f64> = (0..n).map(|_| rng.unit() - 0.4).collect();
        let y: Vec<f64> = x.iter().map(|v| v.powf(power).abs() + 0.2 * (rng.unit() - 0.5)).collect();
        let want = dcor_definition(&x, &y);
        let got = DistanceY::new(&x).correlate(&y);
        assert!((got - want).abs() < 1e-10, "n={n}: {got} vs {want}");
        // Ties (a column taking three values) exercise the rank-based fast path.
        let t: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let want = dcor_definition(&t, &y);
        let got = DistanceY::new(&t).correlate(&y);
        assert!((got - want).abs() < 1e-10, "ties n={n}: {got} vs {want}");
    }
}

fn run(cols: Vec<Vec<f64>>) -> crate::lingam::StaticDagDiscoveryResult {
    let p = cols.len();
    let names: Vec<String> = (0..p).map(|i| format!("x{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let data = continuous_table(&refs, cols);
    let vars: Vec<VariableId> = (0..p as u32).map(VariableId::from_raw).collect();
    DirectLingam::new()
        .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ExecutionContext::for_tests(1))
        .unwrap()
}

fn assert_recovers(model: &Model, g: usize, shapes: &[usize], label: &str) {
    let result = run(model.columns(g, shapes));
    let mut want: Vec<(u32, u32, f64)> = Vec::new();
    for (c, ps) in model.parents.iter().enumerate() {
        for &(p, b) in ps {
            want.push((p as u32, c as u32, b));
        }
    }
    want.sort_by_key(|e| (e.0, e.1));
    let mut got: Vec<(u32, u32, f64)> = result
        .evidence
        .edge_evidence
        .iter()
        .map(|e| (e.link.source.raw(), e.link.target.raw(), e.statistic.unwrap()))
        .collect();
    got.sort_by_key(|e| (e.0, e.1));
    assert_eq!(got.len(), want.len(), "{label}: edges {got:?} vs {want:?}");
    for (g_, w) in got.iter().zip(&want) {
        assert_eq!((g_.0, g_.1), (w.0, w.1), "{label}: edges {got:?} vs {want:?}");
        assert!((g_.2 - w.2).abs() < 1e-9, "{label}: coefficient {} vs {}", g_.2, w.2);
    }
    // Sources of the true graph that are unordered by any path may swap; the reported order must
    // still be a linear extension of the generating graph.
    let config = result.algorithm.config.to_string();
    let listed = &config[config.find("order=[").unwrap() + 7..];
    let found: Vec<usize> = listed[..listed.find(']').unwrap()]
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let position = |v: usize| found.iter().position(|&u| u == v).unwrap();
    for (c, ps) in model.parents.iter().enumerate() {
        for &(p, _) in ps {
            assert!(position(p) < position(c), "{label}: order {found:?} puts {c} before its parent {p}");
        }
    }
}

#[test]
fn known_structures_are_recovered_exactly() {
    let chain = Model { parents: vec![vec![], vec![(0, 0.9)], vec![(1, -0.8)]], order: vec![0, 1, 2] };
    assert_recovers(&chain, 7, &[0, 1, 2], "chain");
    let reversed_ids = Model { parents: vec![vec![(1, 0.7)], vec![(2, -0.9)], vec![]], order: vec![2, 1, 0] };
    assert_recovers(&reversed_ids, 7, &[2, 0, 1], "chain against id order");
    let collider = Model { parents: vec![vec![], vec![], vec![(0, 0.8), (1, 0.8)]], order: vec![0, 1, 2] };
    assert_recovers(&collider, 7, &[0, 3, 1], "collider");
    let triangle =
        Model { parents: vec![vec![], vec![(0, 0.6)], vec![(0, -0.5), (1, 0.7)]], order: vec![0, 1, 2] };
    assert_recovers(&triangle, 7, &[1, 2, 3], "triangle");
}

#[test]
fn random_models_recover_their_order_edges_and_coefficients() {
    let mut rng = Lcg(0x5EED_1234_ABCD);
    let mut checked = 0;
    for trial in 0..24usize {
        let p = 3 + trial % 2;
        let g = if p == 3 { 7 } else { 5 };
        let model = Model::random(&mut rng, p, 55);
        if model.parents.iter().all(Vec::is_empty) {
            continue;
        }
        let shapes: Vec<usize> = (0..p).map(|v| (trial + v) % 4).collect();
        assert_recovers(&model, g, &shapes, &format!("trial {trial}"));
        checked += 1;
    }
    assert!(checked >= 18, "only {checked} usable models");
}
