//! Temporal mediation identification against population regressions of a linear SEM.
//!
//! For every stationary template over treatment `T`, mediator `M`, outcome `Y` and covariates
//! `Z`, `W` built from the required edges `T[t-1] -> M[t]`, `M[t] -> Y[t]` and every subset of
//! eleven optional edges (lagged, contemporaneous, autoregressive, treatment-induced), at
//! horizons one and two, the identifier either refuses or returns the shared adjustment set `S`.
//! For each answer the test draws linear-Gaussian coefficients on the unfolded DAG (three draws),
//! forms the exact population covariance `(I - B)^-1 D (I - B)^-T`, and checks that the
//! regressions the estimand prescribes - `M ~ T + S`, `Y ~ T + M + S` and `Y ~ T + S` - give the
//! true total effect (the path-sum entry of `(I - B)^-1`), the controlled direct effect (the
//! same entry with the edges into the mediators removed) and the mediated effect
//! `sum_j a_j b_j`. A two-mediator family with ordered and parallel mediators is covered by a
//! seeded sample of its subsets. For a sample of the refusals the test reports whether a valid
//! set exists at all, so that a refusal that is only conservative is distinguished from a
//! graph that no regression set can identify.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, MediationContrast, MediationQuery, VariableId};
use antecedent_data::TemporalNodeKey;
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_identify::{IdentificationError, TemporalMediationIdentifier};

const T: u32 = 0;
const M: u32 = 1;
const Y: u32 = 2;
const Z: u32 = 3;
const W: u32 = 4;
const N2: u32 = 5;

type Edge = (u32, u32, u32, u32);

const OPTIONAL: [Edge; 11] = [
    (T, 1, Y, 0),
    (Z, 1, T, 0),
    (Z, 0, T, 0),
    (Z, 1, Y, 0),
    (Z, 0, M, 0),
    (Z, 0, Y, 0),
    (W, 1, Z, 0),
    (T, 1, T, 0),
    (Y, 1, Y, 0),
    (Y, 1, M, 0),
    (T, 1, Z, 0),
];

const REQUIRED: [Edge; 2] = [(T, 1, M, 0), (M, 0, Y, 0)];

fn template(edges: &[Edge]) -> TemporalDag {
    let mut g = TemporalDag::empty();
    let mut ids: Vec<((u32, u32), DenseNodeId)> = Vec::new();
    let mut node = |g: &mut TemporalDag, v: u32, lag: u32| {
        if let Some((_, id)) = ids.iter().find(|(k, _)| *k == (v, lag)) {
            return *id;
        }
        let id = g.add_lagged(VariableId::from_raw(v), Lag::from_raw(lag)).unwrap();
        ids.push(((v, lag), id));
        id
    };
    for &(sv, sl, tv, tl) in edges {
        let a = node(&mut g, sv, sl);
        let b = node(&mut g, tv, tl);
        g.insert_directed(a, b).unwrap();
    }
    g
}

/// Deterministic value in `[0, 1)` from a salt (splitmix64).
fn unit(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Solve `A x = b` by Gauss-Jordan with partial pivoting; `None` when singular.
fn solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let k = b[0].len();
    let mut m: Vec<Vec<f64>> =
        (0..n).map(|i| a[i].iter().copied().chain(b[i].iter().copied()).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        let pivot = m[c][c];
        for v in &mut m[c] {
            *v /= pivot;
        }
        for r in 0..n {
            if r != c {
                let f = m[r][c];
                if f != 0.0 {
                    let row_c = m[c].clone();
                    for (x, y) in m[r].iter_mut().zip(&row_c) {
                        *x -= f * y;
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n..n + k].to_vec()).collect())
}

struct Sem {
    /// `(I - B)^-1`, entry `[child][source]` is the total effect of source on child.
    total: Vec<Vec<f64>>,
    cov: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

fn sem(parents: &[Vec<usize>], seed: u64) -> Sem {
    let n = parents.len();
    let mut b = vec![vec![0.0; n]; n];
    for (child, ps) in parents.iter().enumerate() {
        for &p in ps {
            let u = unit(seed ^ ((p as u64) << 20) ^ ((child as u64) << 40) ^ 0xB5);
            let sign =
                if unit(seed ^ ((child as u64) << 8) ^ (p as u64)) < 0.5 { -1.0 } else { 1.0 };
            b[child][p] = sign * (0.3 + 0.8 * u);
        }
    }
    let total = inverse_i_minus(&b);
    let var: Vec<f64> = (0..n).map(|i| 0.5 + 1.5 * unit(seed ^ (i as u64) ^ 0xD1)).collect();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            cov[i][j] = (0..n).map(|k| total[i][k] * var[k] * total[j][k]).sum();
        }
    }
    Sem { total, cov, b }
}

fn inverse_i_minus(b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b.len();
    let a: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j)) - b[i][j]).collect()).collect();
    let id: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    solve(&a, &id).expect("I - B is unit triangular up to order")
}

/// Population coefficients of `response ~ regressors` (in the order given).
fn ols(cov: &[Vec<f64>], response: usize, regressors: &[usize]) -> Option<Vec<f64>> {
    let a: Vec<Vec<f64>> =
        regressors.iter().map(|&i| regressors.iter().map(|&j| cov[i][j]).collect()).collect();
    let rhs: Vec<Vec<f64>> = regressors.iter().map(|&i| vec![cov[i][response]]).collect();
    solve(&a, &rhs).map(|x| x.into_iter().map(|r| r[0]).collect())
}

struct Truth {
    total: f64,
    direct: f64,
}

/// Total and controlled direct effect of node `t` on node `y` with mediators `m` held fixed.
fn truth(sem: &Sem, t: usize, y: usize, mediators: &[usize]) -> Truth {
    let mut cut = sem.b.clone();
    for &m in mediators {
        cut[m].iter_mut().for_each(|v| *v = 0.0);
    }
    Truth { total: sem.total[y][t], direct: inverse_i_minus(&cut)[y][t] }
}

/// The three regression estimates for a node set `s`: (total, direct, mediated).
fn estimates(
    sem: &Sem,
    t: usize,
    y: usize,
    mediators: &[usize],
    s: &[usize],
) -> Option<(f64, f64, f64)> {
    let mut reduced = vec![t];
    reduced.extend_from_slice(s);
    let total = ols(&sem.cov, y, &reduced)?[0];
    let mut outcome_reg = vec![t];
    outcome_reg.extend_from_slice(mediators);
    outcome_reg.extend_from_slice(s);
    let beta = ols(&sem.cov, y, &outcome_reg)?;
    let direct = beta[0];
    let mut mediated = 0.0;
    for (j, &m) in mediators.iter().enumerate() {
        let mut reg = vec![t];
        reg.extend_from_slice(s);
        let a = ols(&sem.cov, m, &reg)?[0];
        mediated += a * beta[1 + j];
    }
    Some((total, direct, mediated))
}

fn matches_truth(sem: &Sem, t: usize, y: usize, mediators: &[usize], s: &[usize]) -> bool {
    let tr = truth(sem, t, y, mediators);
    let Some((total, direct, mediated)) = estimates(sem, t, y, mediators, s) else {
        return false;
    };
    let scale = 1.0 + tr.total.abs();
    (total - tr.total).abs() < 1e-7 * scale
        && (direct - tr.direct).abs() < 1e-7 * scale
        && (mediated - (tr.total - tr.direct)).abs() < 1e-7 * scale
}

struct Tally {
    answered: usize,
    refused_not_certified: usize,
    refused_other: usize,
    adjustment_refusals: usize,
    conservative_refusals: usize,
    unidentifiable_refusals: usize,
    sampled_refusals: usize,
}

fn run(
    edge_sets: impl Iterator<Item = Vec<Edge>>,
    mediators: &[u32],
    stride_check: usize,
) -> Tally {
    let mut tally = Tally {
        answered: 0,
        refused_not_certified: 0,
        refused_other: 0,
        adjustment_refusals: 0,
        conservative_refusals: 0,
        unidentifiable_refusals: 0,
        sampled_refusals: 0,
    };
    let ider = TemporalMediationIdentifier::new();
    let query = MediationQuery::binary(
        VariableId::from_raw(T),
        VariableId::from_raw(Y),
        mediators.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        MediationContrast::Mediated,
    );
    for (index, edges) in edge_sets.enumerate() {
        let g = template(&edges);
        for horizon in [1u32, 2] {
            match ider.identify_with_horizon(&g, &query, horizon) {
                Ok((id, temporal)) => {
                    tally.answered += 1;
                    let dag = &temporal.unfolded.dag;
                    let parents: Vec<Vec<usize>> = (0..dag.node_count())
                        .map(|c| {
                            dag.parents(DenseNodeId::from_raw(c as u32))
                                .iter()
                                .map(|p| p.as_usize())
                                .collect()
                        })
                        .collect();
                    let t = temporal.indexer.dense_id(temporal.treatment_key).unwrap() as usize;
                    let y = temporal.indexer.dense_id(temporal.outcome_key).unwrap() as usize;
                    let med: Vec<usize> = mediators
                        .iter()
                        .map(|&m| {
                            temporal
                                .indexer
                                .dense_id(TemporalNodeKey {
                                    variable: VariableId::from_raw(m),
                                    offset: temporal.outcome_key.offset,
                                })
                                .unwrap() as usize
                        })
                        .collect();
                    let s: Vec<usize> =
                        id.estimands[0].adjustment_set.iter().map(|v| v.raw() as usize).collect();
                    assert!(
                        !s.contains(&t) && !s.contains(&y) && s.iter().all(|x| !med.contains(x)),
                        "S contains T, Y or a mediator: edges {edges:?} h={horizon}"
                    );
                    for seed in [11u64, 977, 40_001] {
                        let sem = sem(&parents, seed ^ (index as u64) << 4);
                        assert!(
                            matches_truth(&sem, t, y, &med, &s),
                            "regressions on S={s:?} miss the truth: edges {edges:?} h={horizon} seed {seed}"
                        );
                    }
                }
                Err(error) => {
                    if matches!(error, IdentificationError::NotCertified { .. }) {
                        tally.refused_not_certified += 1;
                    } else {
                        tally.refused_other += 1;
                    }
                    // For a sample of the refusals whose unfolded graph exists, ask whether
                    // any valid set exists on that graph.
                    let Ok(temporal) =
                        ider.temporal.identify_temporal(&g, &temporal_query(&query, horizon))
                    else {
                        continue;
                    };
                    tally.adjustment_refusals += 1;
                    if stride_check > 0 && tally.adjustment_refusals % stride_check == 0 {
                        let dag = &temporal.unfolded.dag;
                        let n = dag.node_count();
                        let parents: Vec<Vec<usize>> = (0..n)
                            .map(|c| {
                                dag.parents(DenseNodeId::from_raw(c as u32))
                                    .iter()
                                    .map(|p| p.as_usize())
                                    .collect()
                            })
                            .collect();
                        let t = temporal.indexer.dense_id(temporal.treatment_key).unwrap() as usize;
                        let y = temporal.indexer.dense_id(temporal.outcome_key).unwrap() as usize;
                        let med: Vec<usize> = mediators
                            .iter()
                            .map(|&m| {
                                temporal
                                    .indexer
                                    .dense_id(TemporalNodeKey {
                                        variable: VariableId::from_raw(m),
                                        offset: temporal.outcome_key.offset,
                                    })
                                    .unwrap() as usize
                            })
                            .collect();
                        let mut sources = vec![t];
                        sources.extend_from_slice(&med);
                        let mut down = vec![false; n];
                        let mut stack = sources.clone();
                        while let Some(v) = stack.pop() {
                            for &c in dag.children(DenseNodeId::from_raw(v as u32)) {
                                if !down[c.as_usize()] {
                                    down[c.as_usize()] = true;
                                    stack.push(c.as_usize());
                                }
                            }
                        }
                        let candidates: Vec<usize> = (0..n)
                            .filter(|v| *v != t && *v != y && !med.contains(v) && !down[*v])
                            .collect();
                        if candidates.len() > 14 {
                            eprintln!("skip: {} candidates", candidates.len());
                            continue;
                        }
                        tally.sampled_refusals += 1;
                        let sems: Vec<Sem> =
                            [11u64, 977].iter().map(|s| sem(&parents, *s ^ index as u64)).collect();
                        let mut found = false;
                        for mask in 0u32..(1 << candidates.len()) {
                            let s: Vec<usize> = (0..candidates.len())
                                .filter(|i| mask >> i & 1 == 1)
                                .map(|i| candidates[i])
                                .collect();
                            if sems.iter().all(|m| matches_truth(m, t, y, &med, &s)) {
                                found = true;
                                break;
                            }
                        }
                        if found {
                            tally.conservative_refusals += 1;
                        } else {
                            tally.unidentifiable_refusals += 1;
                        }
                    }
                }
            }
        }
    }
    tally
}

fn temporal_query(
    mediation: &MediationQuery,
    horizon_steps: u32,
) -> antecedent_core::TemporalEffectQuery {
    antecedent_core::TemporalEffectQuery {
        treatment: mediation.treatment,
        outcome: mediation.outcome,
        policy: antecedent_core::TemporalPolicy::Pulse { at: -1 },
        control: mediation.control.clone(),
        active: mediation.active.clone(),
        horizon_steps,
        max_history_lag: None,
        target_population: mediation.target_population.clone(),
        effect_modifiers: std::sync::Arc::from([]),
    }
}

#[test]
fn every_answered_single_mediator_template_matches_the_true_effects() {
    let sets = (0u32..1 << OPTIONAL.len()).map(|mask| {
        let mut edges = REQUIRED.to_vec();
        edges.extend((0..OPTIONAL.len()).filter(|i| mask >> i & 1 == 1).map(|i| OPTIONAL[i]));
        edges
    });
    let tally = run(sets, &[M], 12);
    eprintln!(
        "single mediator: answered {}, refused {} (refusals with a graph {}, other {}), \
         sampled {}: valid set exists {}, none {}",
        tally.answered,
        tally.refused_not_certified,
        tally.adjustment_refusals,
        tally.refused_other,
        tally.sampled_refusals,
        tally.conservative_refusals,
        tally.unidentifiable_refusals
    );
    assert!(tally.answered > 100, "the sweep must exercise many answers: {}", tally.answered);
    assert!(tally.refused_not_certified > 500, "and many refusals");
    assert!(tally.sampled_refusals > 0, "refusals were sampled");
}

#[test]
fn ordered_and_parallel_mediator_sets_match_the_true_effects() {
    // Variables: T, M, N (second mediator, id 5), Y, Z.
    let optional: [Edge; 11] = [
        (T, 1, N2, 0),
        (N2, 0, Y, 0),
        (M, 0, N2, 0),
        (T, 1, Y, 0),
        (Z, 1, Y, 0),
        (Z, 0, M, 0),
        (Z, 0, N2, 0),
        (Z, 1, T, 0),
        (T, 1, T, 0),
        (N2, 1, N2, 0),
        (Z, 1, M, 0),
    ];
    let sets = (0u32..1 << optional.len()).map(|mask| {
        let mut edges = REQUIRED.to_vec();
        edges.extend((0..optional.len()).filter(|i| mask >> i & 1 == 1).map(|i| optional[i]));
        edges
    });
    let tally = run(sets, &[M, N2], 0);
    eprintln!(
        "two mediators: answered {}, refused {} (other {})",
        tally.answered, tally.refused_not_certified, tally.refused_other
    );
    assert!(tally.answered > 200, "{}", tally.answered);
}

#[test]
fn a_query_with_several_horizons_is_not_answered_for_the_first_alone() {
    let g = template(&REQUIRED);
    let mut q = MediationQuery::binary(
        VariableId::from_raw(T),
        VariableId::from_raw(Y),
        [VariableId::from_raw(M)],
        MediationContrast::Mediated,
    );
    q.horizons = std::sync::Arc::from([1u32, 2]);
    let ider = TemporalMediationIdentifier::new();
    assert!(matches!(ider.identify(&g, &q), Err(IdentificationError::UnsupportedQuery { .. })));
    // Each declared horizon is still answered through the per-horizon entry point.
    for h in [1u32, 2] {
        assert!(ider.identify_with_horizon(&g, &q, h).is_ok());
    }
}
