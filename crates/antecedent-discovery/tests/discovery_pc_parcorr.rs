//! Static PC `ParCorr` recovery on a small Gaussian SEM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{DiscoveryWorkspace, Pc};
use antecedent_graph::DenseNodeId;

mod common;

fn gaussian_chain(n: usize, seed: u64, middle_offset: f64) -> TabularData {
    // X0 → X1 → X2 linear Gaussian.
    let mut x0 = vec![0.0; n];
    let mut x1 = vec![0.0; n];
    let mut x2 = vec![0.0; n];
    let mut rng = antecedent_core::CausalRng::from_seed(seed);
    let mut next_gauss = || antecedent_kernels::standard_normal(&mut rng);
    for i in 0..n {
        x0[i] = next_gauss();
        x1[i] = 0.9 * x0[i] + 0.1 * next_gauss();
        x2[i] = 0.9 * x1[i] + 0.1 * next_gauss();
    }
    for value in &mut x1 {
        *value += middle_offset;
    }
    TabularData::new(common::continuous_storage(&["x0", "x1", "x2"], vec![x0, x1, x2]))
}

#[test]
fn parcorr_recovers_chain_skeleton() {
    let data = gaussian_chain(2000, 7, 0.0);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let constraints = antecedent_discovery::DiscoveryConstraints {
        alpha: 0.05,
        max_cond_size: 2,
        ..Default::default()
    };
    let pc = Pc::new().with_fdr(false).with_constraints(constraints);
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
    let g = &result.evidence.graph;
    assert!(g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)), "expected 0—1");
    assert!(g.has_edge(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)), "expected 1—2");
    assert!(
        !g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)),
        "0—2 should be removed given 1; edges={:?}",
        g.edges()
    );
}

#[test]
fn pc_graph_is_invariant_to_conditioning_column_offset() {
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let constraints = antecedent_discovery::DiscoveryConstraints {
        alpha: 0.05,
        max_cond_size: 2,
        ..Default::default()
    };
    let pc = Pc::new().with_fdr(false).with_constraints(constraints);
    let ctx = ExecutionContext::for_tests(7);
    let mut base_ws = DiscoveryWorkspace::default();
    let mut shifted_ws = DiscoveryWorkspace::default();
    let base = pc.run(&gaussian_chain(2000, 7, 0.0), &vars, &mut base_ws, &ctx).unwrap();
    let shifted = pc.run(&gaussian_chain(2000, 7, 10_000.0), &vars, &mut shifted_ws, &ctx).unwrap();
    assert_eq!(base.evidence.graph.edges(), shifted.evidence.graph.edges());
}
