//! Attribution to the noise blocks of a latent-confounded SCM: which mixed graphs identify it,
//! the two-SCM witness for the ones that do not, and the two block games against known truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_attribution::{
    AttributionError, DifferenceMeasure, DistributionChangeOptions, HEDGE_WITNESS_ID,
    HEDGE_WITNESS_TEST, MechanismHedge, NoiseBlockIdentification, NoiseBlockPlan,
    distribution_change, distribution_change_blocks, plan_noise_blocks,
    score_anomalies_blocks_with, score_anomalies_with,
};
use antecedent_core::{
    AllocationMethod, AnomalyAttributionQuery, ChangeAttributionQuery, ComponentId,
    ExecutionContext, PopulationSelector, ShapleyConfig, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};

fn node(index: u32) -> DenseNodeId {
    DenseNodeId::from_raw(index)
}

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

/// A mixed graph over `n` variables.
fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(from, to) in directed {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(node(a), node(b)).unwrap();
    }
    graph
}

fn identified(verdict: NoiseBlockIdentification) -> NoiseBlockPlan {
    match verdict {
        NoiseBlockIdentification::Identified(plan) => plan,
        NoiseBlockIdentification::Hedged(hedge) => panic!("expected an identified plan: {hedge:?}"),
    }
}

fn hedged(verdict: NoiseBlockIdentification) -> MechanismHedge {
    match verdict {
        NoiseBlockIdentification::Hedged(hedge) => hedge,
        NoiseBlockIdentification::Identified(plan) => panic!("expected a hedge: {plan:?}"),
    }
}

fn raw_blocks(plan: &NoiseBlockPlan) -> Vec<Vec<u32>> {
    plan.blocks.iter().map(|block| block.iter().map(|v| v.raw()).collect()).collect()
}

#[test]
fn blocks_are_the_districts_of_the_ancestral_subgraph() {
    // a -> t <- b with a <-> b: the two causes of the target share a latent cause, and every
    // mechanism is identified (t has no district mate, a and b have no parent).
    let collider = admg(3, &[(0, 2), (1, 2)], &[(0, 1)]);
    let plan = identified(plan_noise_blocks(&collider, &[variable(2)]).unwrap());
    assert_eq!(raw_blocks(&plan), vec![vec![0, 1], vec![2]]);
    assert_eq!(plan.directed.directed_edge_pairs_sorted(), vec![(0, 2), (1, 2)]);

    // The same graph attributed to `a` alone: `b` is not an ancestor, so the latent edge does
    // not reach the ancestral sub-graph.
    let alone = identified(plan_noise_blocks(&collider, &[variable(0)]).unwrap());
    assert_eq!(raw_blocks(&alone), vec![vec![0]]);

    // A latent edge between the target and a descendant of its parent leaves the ancestors
    // singletons: w is not an ancestor of y.
    let descendant = admg(3, &[(0, 1), (0, 2)], &[(2, 1)]);
    let plan = identified(plan_noise_blocks(&descendant, &[variable(1)]).unwrap());
    assert_eq!(raw_blocks(&plan), vec![vec![0], vec![1]]);

    // d -> m -> i with d <-> i: d reaches i only through m, which is outside the district, so
    // the mechanism of i is the back-door adjustment for d and is identified; {d, i} is a block.
    let front_door = admg(3, &[(0, 1), (1, 2)], &[(0, 2)]);
    let plan = identified(plan_noise_blocks(&front_door, &[variable(2)]).unwrap());
    assert_eq!(raw_blocks(&plan), vec![vec![0, 2], vec![1]]);
}

#[test]
fn a_district_member_that_reaches_a_node_inside_the_district_is_a_hedge() {
    // p -> i with p <-> i: the textbook hedge, the structural function of i is confounded.
    let direct = admg(2, &[(0, 1)], &[(0, 1)]);
    let hedge = hedged(plan_noise_blocks(&direct, &[variable(1)]).unwrap());
    assert_eq!((hedge.node, hedge.ancestor), (variable(1), variable(0)));
    assert_eq!(hedge.path, vec![variable(0), variable(1)]);
    let text = hedge.describe();
    assert!(text.contains(HEDGE_WITNESS_ID) && text.contains(HEDGE_WITNESS_TEST), "{text}");

    // j -> k -> i with j <-> k and j <-> i: j reaches k inside the district {j, k, i}.
    let chain = admg(3, &[(0, 1), (1, 2)], &[(0, 1), (0, 2)]);
    let hedge = hedged(plan_noise_blocks(&chain, &[variable(2)]).unwrap());
    assert_eq!((hedge.node, hedge.ancestor), (variable(1), variable(0)));

    // A target that is not a node of the graph is an error, not a verdict.
    assert!(matches!(
        plan_noise_blocks(&direct, &[variable(9)]),
        Err(AttributionError::MissingVariable { .. })
    ));
}

/// One point of a finite probability space: the probability and the two noise terms.
type Point = (f64, f64, f64);

/// The noise of `p` and of `i` at six points; both terms have mean zero.
const OMEGA: [Point; 6] = [
    (0.125, -1.0, -1.0),
    (0.125, -1.0, 1.0),
    (0.25, 0.0, -1.0),
    (0.25, 0.0, 1.0),
    (0.125, 1.0, -1.0),
    (0.125, 1.0, 1.0),
];

/// `h(p) = p^2 - E[p^2]`: mean zero under the law of the noise of `p`.
fn confounded_effect(p: f64) -> f64 {
    p * p - 0.5
}

/// The observed law of `(p, i)` as a map from the pair (scaled by four, so exact) to its mass.
fn law(draw: impl Fn(f64, f64) -> (f64, f64)) -> BTreeMap<(i64, i64), f64> {
    let mut law = BTreeMap::new();
    for &(mass, noise_p, noise_i) in &OMEGA {
        let (p, i) = draw(noise_p, noise_i);
        *law.entry(((p * 4.0).round() as i64, (i * 4.0).round() as i64)).or_insert(0.0) += mass;
    }
    law
}

#[test]
fn two_scms_agree_on_the_observed_law_and_disagree_on_the_attribution() {
    // Graph: p -> i with p <-> i, one district {p, i}, so one noise block.
    //
    // SCM A: p = n_p and i = n_i (p has no effect).
    // SCM B: p = n_p and i = h(p) + n_i - h(n_p) (p has the effect h, and the noise of i is
    //        shifted by the same function of n_p, which the latent edge allows: the noise of i
    //        may depend on the noise of its district mate p, and has mean zero again).
    let law_a = law(|noise_p, noise_i| (noise_p, noise_i));
    let law_b = law(|noise_p, noise_i| {
        let noise_b = noise_i - confounded_effect(noise_p);
        (noise_p, confounded_effect(noise_p) + noise_b)
    });
    assert_eq!(law_a, law_b, "the two SCMs have the same observed law");
    let mean_noise_b: f64 = OMEGA
        .iter()
        .map(|&(mass, noise_p, noise_i)| mass * (noise_i - confounded_effect(noise_p)))
        .sum();
    assert!(mean_noise_b.abs() < 1e-15, "the noise of B keeps mean zero: {mean_noise_b}");

    // The attribution to the one block {p, i}: Shapley of a one-player game is the value at the
    // factual noise minus the value with every noise term at the reference (zero).
    let reference_a = 0.0;
    let reference_b = confounded_effect(0.0);
    assert!((reference_a - reference_b).abs() > 0.4, "the target at the reference noise differs");
    for &(_, noise_p, noise_i) in &OMEGA {
        let factual = noise_i; // i is the same random variable in both SCMs
        let phi_a = factual - reference_a;
        let phi_b = factual - reference_b;
        assert!(
            (phi_a - phi_b).abs() > 0.4,
            "unit ({noise_p}, {noise_i}) is attributed alike by two SCMs that differ"
        );
    }

    // The engine refuses exactly this structure, and cites this test.
    let graph = admg(2, &[(0, 1)], &[(0, 1)]);
    let hedge = hedged(plan_noise_blocks(&graph, &[variable(1)]).unwrap());
    assert!(hedge.describe().contains(HEDGE_WITNESS_ID));
    let witness =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(HEDGE_WITNESS_TEST);
    assert!(witness.is_file(), "the refusal cites {HEDGE_WITNESS_TEST}, which must exist");
}

fn multiply(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = (0..3).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    product
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[column][row] = matrix[row][column];
        }
    }
    out
}

/// Covariance of the linear SEM `x = B x + e` over (j, k, i) with `B` strictly lower triangular:
/// `Sigma = (I - B)^-1 Omega (I - B)^-T` with `(I - B)^-1 = I + B + B^2` for a nilpotent `B`.
fn covariance(a: f64, b: f64, omega: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let coefficients = [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [0.0, b, 0.0]];
    let square = multiply(coefficients, coefficients);
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            inverse[row][column] = f64::from(u8::from(row == column))
                + coefficients[row][column]
                + square[row][column];
        }
    }
    multiply(multiply(inverse, omega), transpose(inverse))
}

#[test]
fn a_hedge_that_is_not_a_direct_edge_is_also_not_identified_in_the_linear_gaussian_class() {
    // j -> k -> i with j <-> k and j <-> i (no k <-> i): the structural coefficient of k -> i is
    // not identified. Two parameter vectors with the same covariance matrix, one with the
    // coefficient 1 and one with 0.
    let omega_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let omega_b = [[1.0, -1.0, 1.0], [-1.0, 2.0, 0.0], [1.0, 0.0, 3.0]];
    // Both noise covariances respect the graph (no k <-> i) and are positive definite.
    assert!(f64::abs(omega_b[1][2]) < 1e-15);
    let minors = |m: [[f64; 3]; 3]| {
        let det3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        (m[0][0], m[0][0] * m[1][1] - m[0][1] * m[1][0], det3)
    };
    let (m1, m2, m3) = minors(omega_b);
    assert!(m1 > 0.0 && m2 > 0.0 && m3 > 0.0);

    let sigma_a = covariance(1.0, 1.0, omega_a);
    let sigma_b = covariance(2.0, 0.0, omega_b);
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (sigma_a[row][column] - sigma_b[row][column]).abs() < 1e-12,
                "covariance differs at ({row}, {column}): {} vs {}",
                sigma_a[row][column],
                sigma_b[row][column]
            );
        }
    }
    let graph = admg(3, &[(0, 1), (1, 2)], &[(0, 1), (0, 2)]);
    hedged(plan_noise_blocks(&graph, &[variable(2)]).unwrap());
}

/// Deterministic, nearly uncorrelated unit-scale sequences.
fn sines(rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sequence = |frequency: f64, phase: f64| -> Vec<f64> {
        (0..rows).map(|i| (frequency * i as f64 + phase).sin()).collect()
    };
    (sequence(0.37, 0.2), sequence(1.13, 0.7), sequence(2.9, 1.3))
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied()).unwrap()
}

fn fitted(graph: Dag, data: &TabularData) -> CompiledCausalModel {
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
        .unwrap();
    compiled.with_mechanisms(store)
}

fn collider_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph
}

/// `a`, `b` (correlated through a shared sine) and `t = a + 2b + noise`, with the last row's `b`
/// far out so that row is the anomaly.
fn anomaly_table() -> TabularData {
    let rows = 80;
    let (s1, s2, s3) = sines(rows);
    let a: Vec<f64> = s1.clone();
    let mut b: Vec<f64> = (0..rows).map(|i| -0.6 * s1[i] + 0.8 * s2[i]).collect();
    b[rows - 1] += 6.0;
    let t: Vec<f64> = (0..rows).map(|i| a[i] + 2.0 * b[i] + 0.3 * s3[i]).collect();
    table(&[("a", a.as_slice()), ("b", b.as_slice()), ("t", t.as_slice())])
}

#[test]
fn singleton_blocks_reproduce_the_ungrouped_anomaly_scores() {
    let data = anomaly_table();
    let model = fitted(collider_dag(), &data);
    let query = AnomalyAttributionQuery::new([variable(2)], 1000);
    let ctx = ExecutionContext::for_tests(7);
    let plain = score_anomalies_with(&model, &data, &query, &ctx).unwrap();
    // One block per player, in the order the ungrouped scores list them.
    let blocks: Vec<Vec<VariableId>> =
        plain[0].noise_components.iter().map(|component| vec![component.variable()]).collect();
    assert_eq!(blocks.len(), 3);
    let grouped = score_anomalies_blocks_with(&model, &data, &query, &blocks, &ctx).unwrap();
    assert_eq!(plain, grouped);
}

#[test]
fn one_block_for_every_ancestor_carries_the_whole_score() {
    let data = anomaly_table();
    let model = fitted(collider_dag(), &data);
    let query = AnomalyAttributionQuery::new([variable(2)], 1000);
    let ctx = ExecutionContext::for_tests(7);
    let blocks = vec![vec![variable(0), variable(1), variable(2)]];
    let scores = score_anomalies_blocks_with(&model, &data, &query, &blocks, &ctx).unwrap();
    let target = &scores[0];
    assert_eq!(target.noise_components.as_ref(), &[ComponentId::from_variable(variable(0))]);
    for row in 0..target.rows.len() {
        let expected = target.scores[row] - target.baseline_scores[row];
        assert!(
            (target.noise_contributions[row] - expected).abs() < 1e-9,
            "row {row}: {} vs {expected}",
            target.noise_contributions[row]
        );
    }
}

#[test]
fn two_blocks_are_efficient_and_named_by_their_smallest_member() {
    let data = anomaly_table();
    let model = fitted(collider_dag(), &data);
    let query = AnomalyAttributionQuery::new([variable(2)], 1000);
    let ctx = ExecutionContext::for_tests(7);
    let blocks = vec![vec![variable(1), variable(0)], vec![variable(2)]];
    let scores = score_anomalies_blocks_with(&model, &data, &query, &blocks, &ctx).unwrap();
    let target = &scores[0];
    assert_eq!(
        target.noise_components.as_ref(),
        &[ComponentId::from_variable(variable(0)), ComponentId::from_variable(variable(2))]
    );
    for row in 0..target.rows.len() {
        let share: f64 = target.noise_contributions[row * 2..row * 2 + 2].iter().sum();
        let expected = target.scores[row] - target.baseline_scores[row];
        assert!((share - expected).abs() < 1e-9, "row {row}: {share} vs {expected}");
    }
    // The outlying row is the anomaly, and the block that holds `b` carries it.
    let last = target.rows.len() - 1;
    let mut by_row: Vec<usize> = (0..target.rows.len()).collect();
    by_row.sort_by(|x, y| target.scores[*y].total_cmp(&target.scores[*x]));
    assert_eq!(by_row[0], last);
    assert!(
        target.noise_contributions[last * 2].abs() > target.noise_contributions[last * 2 + 1].abs()
    );
}

#[test]
fn blocks_must_partition_the_ancestors() {
    let data = anomaly_table();
    let model = fitted(collider_dag(), &data);
    let query = AnomalyAttributionQuery::new([variable(2)], 1000);
    let ctx = ExecutionContext::for_tests(7);
    let missing = vec![vec![variable(0)], vec![variable(2)]];
    assert!(matches!(
        score_anomalies_blocks_with(&model, &data, &query, &missing, &ctx),
        Err(AttributionError::InvalidInput { .. })
    ));
    let overlapping = vec![vec![variable(0), variable(1)], vec![variable(1), variable(2)]];
    assert!(matches!(
        score_anomalies_blocks_with(&model, &data, &query, &overlapping, &ctx),
        Err(AttributionError::InvalidInput { .. })
    ));
}

/// Two populations of `rows` units each, in one table (baseline first), over the collider
/// `a -> t <- b`. The noise sequences are shared, so the populations differ only by what
/// `scale`, `shift` and `intercept` put in.
fn two_populations(
    rows: usize,
    scale: f64,
    shift: f64,
    intercept: f64,
    coefficient_b: f64,
) -> TabularData {
    let (s1, s2, s3) = sines(rows);
    let mut a = Vec::with_capacity(2 * rows);
    let mut b = Vec::with_capacity(2 * rows);
    let mut t = Vec::with_capacity(2 * rows);
    for population in 0..2 {
        let (factor, offset, constant) =
            if population == 0 { (1.0, 0.0, 0.0) } else { (scale, shift, intercept) };
        for i in 0..rows {
            let a_i = factor * s1[i] + offset;
            let b_i = factor * (-0.6 * s1[i] + 0.8 * s2[i]) + offset;
            a.push(a_i);
            b.push(b_i);
            t.push(a_i + coefficient_b * b_i + constant + 0.3 * s3[i]);
        }
    }
    table(&[("a", a.as_slice()), ("b", b.as_slice()), ("t", t.as_slice())])
}

fn change_query(rows: usize) -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        variable(2),
        PopulationSelector::TimeRange { start: 0, end: rows },
        PopulationSelector::TimeRange { start: rows, end: 2 * rows },
    )
    .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() })
}

fn population_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[test]
fn a_block_change_matches_the_known_shift_of_its_members_and_of_the_target() {
    // Comparison: a and b both shifted by +1, and the mechanism of t shifted by +0.5, with
    // t = a + 2b + noise. With the block {a, b} and the player {t} the game is additive:
    // the block carries (1 + 2) * 1 = 3 and t carries 0.5.
    let rows = 300;
    let data = two_populations(rows, 1.0, 1.0, 0.5, 2.0);
    let model = fitted(collider_dag(), &data);
    let options = DistributionChangeOptions {
        measure: DifferenceMeasure::MeanDiff,
        n_samples: 20_000,
        seed: 5,
    };
    let blocks = vec![vec![variable(0), variable(1)], vec![variable(2)]];
    let ctx = ExecutionContext::for_tests(11);
    let result = distribution_change_blocks(
        &model,
        &data,
        &change_query(rows),
        &options,
        &blocks,
        None,
        &ctx,
    )
    .unwrap();
    assert!((result.total_change - 3.5).abs() < 0.15, "total {}", result.total_change);
    let share = |component: u32| {
        result
            .contributions
            .iter()
            .find(|c| c.component == ComponentId::from_variable(variable(component)))
            .map(|c| c.contribution)
            .unwrap()
    };
    assert_eq!(result.contributions.len(), 2);
    assert!((share(0) - 3.0).abs() < 0.2, "block {}", share(0));
    assert!((share(2) - 0.5).abs() < 0.2, "target {}", share(2));
    let sum: f64 = result.contribution_sum();
    assert!(
        (sum - result.total_change).abs() < 1e-9,
        "efficiency {sum} vs {}",
        result.total_change
    );
}

#[test]
fn the_noise_of_a_block_is_resampled_jointly() {
    // Both populations: a and b share a sine (correlation -0.6), t = a + b + noise. The
    // comparison scales a and b by two. The variance of t is Var(a + b) + Var(noise): with the
    // dependence kept it is about 0.525 and 1.725, so the change is about 1.2; resampling a and
    // b independently would give about 1.125 and 4.125, a change near 3.
    let rows = 300;
    let data = two_populations(rows, 2.0, 0.0, 0.0, 1.0);
    let model = fitted(collider_dag(), &data);
    let t = data.float64_values(variable(2)).unwrap();
    let observed = population_variance(&t[rows..]) - population_variance(&t[..rows]);
    assert!((observed - 1.2).abs() < 0.2, "the data change by {observed}");
    let options = DistributionChangeOptions {
        measure: DifferenceMeasure::VarianceDiff,
        n_samples: 20_000,
        seed: 9,
    };
    let query = change_query(rows);
    let ctx = ExecutionContext::for_tests(3);
    let blocks = vec![vec![variable(0), variable(1)], vec![variable(2)]];
    let joint =
        distribution_change_blocks(&model, &data, &query, &options, &blocks, None, &ctx).unwrap();
    assert!(
        (joint.total_change - observed).abs() < 0.15,
        "joint resampling reproduces the observed change: {} vs {observed}",
        joint.total_change
    );
    let independent = distribution_change(&model, &data, &query, &options, &ctx).unwrap();
    assert!(
        independent.total_change > 2.0 * joint.total_change,
        "independent noise overstates the change: {} vs {}",
        independent.total_change,
        joint.total_change
    );
}

#[test]
fn a_change_attribution_needs_every_ancestor_in_a_block() {
    let rows = 60;
    let data = two_populations(rows, 1.0, 1.0, 0.0, 2.0);
    let model = fitted(collider_dag(), &data);
    let options = DistributionChangeOptions::default();
    let ctx = ExecutionContext::for_tests(1);
    let missing = vec![vec![variable(0)], vec![variable(2)]];
    assert!(matches!(
        distribution_change_blocks(
            &model,
            &data,
            &change_query(rows),
            &options,
            &missing,
            None,
            &ctx
        ),
        Err(AttributionError::InvalidInput { .. })
    ));
}
