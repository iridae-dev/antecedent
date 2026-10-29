//! Exact exhaustive targets for order, structure, and CI-screened graph MCMC.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};
use antecedent_discovery::DbnPosterior;
use antecedent_discovery::{
    AdjacencyMask, CiScreenedPosterior, DiscoveryConstraints, DiscoveryWorkspace,
    ExactDagPosterior, GraphPosterior, GraphPrior, OrderMcmc, StructureMcmc,
};
use antecedent_state::GraphScoreFamily;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

mod common;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/graph_mcmc_exact/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("graph MCMC fixture"))
        .expect("parse graph MCMC fixture")
}

fn case_data(case: &JsonValue) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let rows = case["data"].as_array().unwrap();
    let columns: Vec<Vec<f64>> = (0..names.len())
        .map(|column| {
            rows.iter().map(|row| row.as_array().unwrap()[column].as_f64().unwrap()).collect()
        })
        .collect();
    let storage = common::continuous_storage(&names, columns);
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (TabularData::new(storage), variables, names)
}

fn named_marginals(post: &GraphPosterior, names: &[String]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for from in 0..names.len() {
        for to in 0..names.len() {
            if from != to {
                out.insert(
                    format!("{}->{}", names[from], names[to]),
                    post.edge_marginals()[from * names.len() + to],
                );
            }
        }
    }
    out
}

fn case_data_permuted(
    case: &JsonValue,
    perm: &[usize],
) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let names: Vec<String> = perm.iter().map(|&i| names[i].clone()).collect();
    let rows = case["data"].as_array().unwrap();
    let columns: Vec<Vec<f64>> = perm
        .iter()
        .map(|&src_col| {
            rows.iter().map(|row| row.as_array().unwrap()[src_col].as_f64().unwrap()).collect()
        })
        .collect();
    let storage = common::continuous_storage(&names, columns);
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (TabularData::new(storage), variables, names)
}

fn expected(target: &JsonValue) -> BTreeMap<String, f64> {
    target["edge_marginals"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.as_f64().unwrap()))
        .collect()
}

fn schedule(fixture: &JsonValue) -> (u32, u32, u32, u32) {
    let value = &fixture["acceptance"]["schedule"];
    (
        value["chains"].as_u64().unwrap() as u32,
        value["warmup"].as_u64().unwrap() as u32,
        value["draws"].as_u64().unwrap() as u32,
        value["thin"].as_u64().unwrap() as u32,
    )
}

fn max_error(actual: &BTreeMap<String, f64>, target: &BTreeMap<String, f64>) -> f64 {
    actual.iter().map(|(key, value)| (value - target[key]).abs()).fold(0.0, f64::max)
}

#[test]
fn graph_mcmc_marginals_against_exact_exhaustive_targets() {
    // MM-011: Order MCMC subtracts log|#topo orders| so edge marginals match the
    // exhaustive uniform-DAG Gaussian-BIC target (Structure MCMC already did).
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    let seeds: Vec<u64> = fixture["acceptance"]["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = case_data(case);
        let target = expected(&case["targets"]["uniform_all_dags"]);
        for &seed in &seeds {
            let mut workspace = DiscoveryWorkspace::default();
            let structure = StructureMcmc::new()
                .with_schedule(chains, warmup, draws, thin)
                .with_diagnostics_gate(false);
            let structure_post = structure
                .run(
                    &data,
                    &variables,
                    &GraphPrior::uniform(),
                    GraphScoreFamily::GaussianBic,
                    &mut workspace,
                    &ExecutionContext::for_tests(seed),
                )
                .unwrap();
            let structure_error = max_error(&named_marginals(&structure_post, &names), &target);
            assert!(
                structure_error <= band,
                "{} seed={seed}: structure MCMC error {structure_error} > {band}",
                case["name"]
            );

            let mut workspace = DiscoveryWorkspace::default();
            let order = OrderMcmc::new()
                .with_schedule(chains, warmup, draws, thin)
                .with_diagnostics_gate(false);
            let order_post = order
                .run(
                    &data,
                    &variables,
                    &GraphPrior::uniform(),
                    GraphScoreFamily::GaussianBic,
                    &mut workspace,
                    &ExecutionContext::for_tests(seed),
                )
                .unwrap();
            let order_error = max_error(&named_marginals(&order_post, &names), &target);
            assert!(
                order_error <= band,
                "{} seed={seed}: order MCMC error {order_error} > {band}",
                case["name"]
            );
        }
    }
}

#[test]
fn order_mcmc_named_marginals_invariant_to_column_relabeling() {
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    let case = &fixture["cases"][0];
    let target = expected(&case["targets"]["uniform_all_dags"]);
    let perm = [2usize, 0, 1];
    let (data, variables, names) = case_data_permuted(case, &perm);
    let mut workspace = DiscoveryWorkspace::default();
    let order =
        OrderMcmc::new().with_schedule(chains, warmup, draws, thin).with_diagnostics_gate(false);
    let post = order
        .run(
            &data,
            &variables,
            &GraphPrior::uniform(),
            GraphScoreFamily::GaussianBic,
            &mut workspace,
            &ExecutionContext::for_tests(401),
        )
        .unwrap();
    let error = max_error(&named_marginals(&post, &names), &target);
    assert!(error <= band, "relabeled order MCMC error {error} > {band} (names={names:?})");
}

#[test]
fn ci_screened_structure_mcmc_against_exact_restricted_targets() {
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = case_data(case);
        let target = expected(&case["targets"]["uniform_screened_pairs"]);
        let mcmc = StructureMcmc::new()
            .with_schedule(chains, warmup, draws, thin)
            .with_diagnostics_gate(false);
        let mut algorithm = CiScreenedPosterior::new()
            .with_constraints(DiscoveryConstraints {
                alpha: 0.01,
                max_cond_size: 3,
                ..Default::default()
            })
            .with_mcmc(mcmc);
        algorithm.fdr = None;
        let mut workspace = DiscoveryWorkspace::default();
        let post = algorithm
            .run(
                &data,
                &variables,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut workspace,
                &ExecutionContext::for_tests(499),
            )
            .unwrap();
        let error = max_error(&named_marginals(&post, &names), &target);
        assert!(error <= band, "{} screened max error={error} > {band}", case["name"]);
    }
}

#[test]
fn exact_enumeration_matches_the_independent_stdlib_enumeration_to_rounding() {
    // The fixture's edge marginals are reproduced by
    // conformance/discovery/graph_mcmc_exact/reference_exact.py (standard library only,
    // Gaussian elimination, 2.5e-14 agreement), so a rounding-level comparison with the
    // product's exact posterior checks the local score, parent-set decomposition, acyclicity
    // filter, Bernoulli edge prior and normalisation, not just the sampler band.
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = case_data(case);
        for (key, prior) in [
            ("uniform_all_dags", GraphPrior::uniform()),
            ("bernoulli_0_2_all_dags", GraphPrior::bernoulli_edges(0.2).unwrap()),
        ] {
            let mut workspace = DiscoveryWorkspace::default();
            let post = ExactDagPosterior::new()
                .run(
                    &data,
                    &variables,
                    &prior,
                    GraphScoreFamily::GaussianBic,
                    &mut workspace,
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            let target = &case["targets"][key];
            assert_eq!(post.n_graphs() as u64, target["n_dags"].as_u64().unwrap());
            let total: f64 = post.weights().iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "{} {key}: weights sum {total}", case["name"]);
            let error = max_error(&named_marginals(&post, &names), &expected(target));
            assert!(error < 1e-9, "{} {key}: exact marginal error {error}", case["name"]);
        }
    }
}

/// Markov-equivalence signature of a DAG by the Verma-Pearl characterisation: the skeleton
/// plus the unshielded colliders. It is computed here from the directed edges alone, with no
/// use of the product's essential-graph construction.
type Signature = (Vec<(usize, usize)>, Vec<(usize, usize, usize)>);

fn equivalence_signature(post: &GraphPosterior, atom: usize, n: usize) -> Signature {
    let mask = &post.atoms()[atom];
    let adjacent = |a: usize, b: usize| mask.has_edge(n, a, b) || mask.has_edge(n, b, a);
    let skeleton = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .filter(|&(a, b)| adjacent(a, b))
        .collect();
    let mut colliders = Vec::new();
    for c in 0..n {
        for a in 0..n {
            for b in a + 1..n {
                if a != c
                    && b != c
                    && mask.has_edge(n, a, c)
                    && mask.has_edge(n, b, c)
                    && !adjacent(a, b)
                {
                    colliders.push((a, c, b));
                }
            }
        }
    }
    (skeleton, colliders)
}

#[test]
fn class_collapse_partitions_every_dag_by_skeleton_and_unshielded_colliders() {
    // Over all labelled DAGs on n nodes the number of Markov equivalence classes is 11 for
    // n = 3 and 185 for n = 4 (Gillispie and Perlman 2002), and two DAGs share a class exactly
    // when their skeletons and unshielded colliders agree (Verma and Pearl 1990). The class
    // posterior must be that partition with the pooled exact DAG mass.
    for (n, classes) in [(3usize, 11usize), (4, 185)] {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ n as u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for i in 0..n {
            let column: Vec<f64> = (0..120)
                .map(|row| {
                    let parent = if i == 0 { 0.0 } else { columns[i - 1][row] };
                    next() + 0.6 * parent
                })
                .collect();
            columns.push(column);
        }
        let data = TabularData::new(common::continuous_storage(&names, columns));
        let variables: Vec<VariableId> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
        let mut workspace = DiscoveryWorkspace::default();
        let dags = ExactDagPosterior::new()
            .run(
                &data,
                &variables,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut workspace,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        // Per class: pooled mass and the union of the members' directed edges. An edge is
        // compelled exactly when every member orients it the same way, so the union holds one
        // bit for a compelled edge and both bits for a reversible one, which is the encoding
        // of the essential graph the posterior uses.
        let mut pooled: BTreeMap<Signature, (f64, AdjacencyMask)> = BTreeMap::new();
        for (atom, weight) in dags.weights().iter().enumerate() {
            let mask = dags.atoms()[atom].clone();
            let entry = pooled
                .entry(equivalence_signature(&dags, atom, n))
                .or_insert((0.0, AdjacencyMask::EMPTY));
            entry.0 += weight;
            for i in (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).filter(|&(i, j)| i != j) {
                if mask.has_edge(n, i.0, i.1) {
                    entry.1 =
                        entry.1.clone().with_bit(antecedent_discovery::edge_bit(n, i.0, i.1), true);
                }
            }
        }
        assert_eq!(pooled.len(), classes, "n={n}: signature classes");
        let collapsed = dags.to_equivalence_classes().unwrap();
        assert_eq!(collapsed.n_graphs(), classes, "n={n}: collapsed classes");
        for (mass, essential) in pooled.values() {
            let at = collapsed
                .atoms()
                .iter()
                .position(|atom| atom == essential)
                .unwrap_or_else(|| panic!("n={n}: no collapsed atom equals the member union"));
            let got = collapsed.weights()[at];
            assert!((got - mass).abs() < 1e-12, "n={n}: pooled class mass {mass} vs {got}");
        }
        let actual: Vec<f64> = collapsed.weights().to_vec();
        assert!((actual.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

/// Naive least-squares residual sum by normal equations and Gaussian elimination, written
/// for this test and sharing no code with the product's cached-Gram Cholesky score.
fn naive_rss(design: &[Vec<f64>], y: &[f64]) -> f64 {
    let k = design.len();
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|r| {
            let mut row: Vec<f64> = (0..k)
                .map(|c| design[r].iter().zip(&design[c]).map(|(u, v)| u * v).sum())
                .collect();
            row.push(design[r].iter().zip(y).map(|(u, v)| u * v).sum());
            row
        })
        .collect();
    for c in 0..k {
        let pivot = (c..k).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, pivot);
        for r in c + 1..k {
            let f = a[r][c] / a[c][c];
            for j in c..=k {
                a[r][j] -= f * a[c][j];
            }
        }
    }
    let mut beta = vec![0.0; k];
    for r in (0..k).rev() {
        beta[r] = (a[r][k] - (r + 1..k).map(|j| a[r][j] * beta[j]).sum::<f64>()) / a[r][r];
    }
    (0..y.len())
        .map(|i| {
            let fit: f64 = (0..k).map(|j| beta[j] * design[j][i]).sum();
            (y[i] - fit).powi(2)
        })
        .sum()
}

fn series_unit(n_obs: usize, state: &mut u64) -> Vec<Vec<f64>> {
    let mut next = || {
        *state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let (mut x, mut y) = (vec![0.0; n_obs], vec![0.0; n_obs]);
    for t in 0..n_obs {
        let (px, py) = if t >= 2 { (x[t - 2], y[t - 1]) } else { (0.0, 0.0) };
        x[t] = 0.5 * px + next();
        y[t] = 0.8 * py + 0.9 * x[t] + 0.4 * if t >= 1 { x[t - 1] } else { 0.0 } + 0.5 * next();
    }
    vec![x, y]
}

fn time_series(columns: &[Vec<f64>]) -> TimeSeriesData {
    let names = ["x", "y"];
    let length = columns[0].len();
    TimeSeriesData::try_new(
        common::continuous_storage(&names, columns.to_vec()),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length },
    )
    .unwrap()
}

#[test]
fn dbn_exact_template_posterior_matches_brute_force_over_a_two_unit_panel() {
    // Two variables, maximum lag two: 3 contemporaneous DAGs times 2^8 lag masks. Every
    // template is scored by regressing each node on its contemporaneous and lagged parents
    // over the lag windows of each unit separately (no window crosses a unit boundary), summing
    // Gaussian log likelihoods at the fitted variance minus (k/2) ln n, and normalising.
    let (p, max_lag) = (2usize, 2usize);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let units = [series_unit(70, &mut state), series_unit(45, &mut state)];
    let rows: usize = units.iter().map(|u| u[0].len() - max_lag).sum();
    // Per node regressors: column (lag, variable) over all stacked windows.
    let value = |lag: usize, var: usize| -> Vec<f64> {
        units.iter().flat_map(|u| (max_lag..u[0].len()).map(move |t| u[var][t - lag])).collect()
    };
    let local = |node: usize, parents: &[(usize, usize)]| -> f64 {
        let mut design = vec![vec![1.0; rows]];
        design.extend(parents.iter().map(|&(lag, var)| value(lag, var)));
        let rss = naive_rss(&design, &value(0, node));
        let n = rows as f64;
        -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI).ln() + (rss / n).ln())
            - 0.5 * (parents.len() + 1) as f64 * n.ln()
    };
    let datasets: Vec<TimeSeriesData> = units.iter().map(|u| time_series(u)).collect();
    for (prior, edge_p) in
        [(GraphPrior::uniform(), None), (GraphPrior::bernoulli_edges(0.3).unwrap(), Some(0.3_f64))]
    {
        let contemporaneous: [&[(usize, usize)]; 3] = [&[], &[(0, 1)], &[(1, 0)]];
        let mut logs = Vec::new();
        let mut instances = Vec::new();
        for edges in contemporaneous {
            for lag_mask in 0u32..(1 << (max_lag * p * p)) {
                let mut total = 0.0;
                for node in 0..p {
                    let mut parents: Vec<(usize, usize)> =
                        edges.iter().filter(|e| e.1 == node).map(|e| (0, e.0)).collect();
                    for lag in 1..=max_lag {
                        for from in 0..p {
                            if lag_mask >> ((lag - 1) * p * p + from * p + node) & 1 == 1 {
                                parents.push((lag, from));
                            }
                        }
                    }
                    total += local(node, &parents);
                }
                if let Some(q) = edge_p {
                    let lag_bits = (max_lag * p * p) as i32;
                    let present = lag_mask.count_ones() as i32;
                    let contemporaneous_bits = (p * (p - 1)) as i32;
                    total += f64::from(present + edges.len() as i32) * q.ln()
                        + f64::from(lag_bits + contemporaneous_bits - present - edges.len() as i32)
                            * (1.0 - q).ln();
                }
                logs.push(total);
                instances.push((edges, lag_mask));
            }
        }
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logs.iter().map(|l| (l - top).exp()).collect();
        let z: f64 = weights.iter().sum();
        let mut lagged = vec![0.0; max_lag * p * p];
        let mut contemp = vec![0.0; p * p];
        for ((edges, lag_mask), w) in instances.iter().zip(&weights) {
            for (b, slot) in lagged.iter_mut().enumerate() {
                if lag_mask >> b & 1 == 1 {
                    *slot += w / z;
                }
            }
            for &(from, to) in *edges {
                contemp[from * p + to] += w / z;
            }
        }
        let variables: Vec<VariableId> = (0..p).map(|i| VariableId::from_raw(i as u32)).collect();
        let post = DbnPosterior::new(max_lag as u32)
            .run_panel(
                &datasets,
                &variables,
                &prior,
                GraphScoreFamily::GaussianBic,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(post.n_graphs(), instances.len());
        for (b, expect) in lagged.iter().enumerate() {
            let got = post.lagged_edge_marginals().unwrap()[b];
            assert!((got - expect).abs() < 1e-9, "lag bit {b}: {got} vs {expect}");
        }
        for (b, expect) in contemp.iter().enumerate() {
            let got = post.edge_marginals()[b];
            assert!((got - expect).abs() < 1e-9, "contemporaneous bit {b}: {got} vs {expect}");
        }
    }
}

#[test]
fn graph_mcmc_fixture_records_exact_target_and_the_retained_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["generator_retained"], true);
    assert_eq!(oracle["generator"], "conformance/discovery/graph_mcmc_exact/reference_exact.py");
    assert_eq!(oracle["environment_lock"], "conformance/discovery/graph_mcmc_exact/reference_exact.py.lock");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert!(root.join(oracle["generator"].as_str().unwrap()).is_file());
    assert!(root.join(oracle["environment_lock"].as_str().unwrap()).is_file());
    // The first numeric targets came from a harness that was not retained.
    let historical = &fixture["historical_oracle"];
    assert_eq!(historical["generator_retained"], false);
    assert_eq!(historical["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(
        historical["generation_location"],
        "temporary external harness; not retained"
    );
    assert_eq!(
        historical["packages"]["numpy"]["metadata_sha256"].as_str().unwrap().len(),
        64
    );
    assert_eq!(fixture["cases"][0]["targets"]["uniform_all_dags"]["n_dags"], 25);
}
