//! Static PC `ParCorr` recovery on a small Gaussian SEM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_discovery::{DiscoveryWorkspace, Pc};
use causal_graph::DenseNodeId;

fn gaussian_chain(n: usize, seed: u64) -> TabularData {
    // X0 → X1 → X2 linear Gaussian.
    let mut b = CausalSchemaBuilder::new();
    for name in ["x0", "x1", "x2"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut x0 = vec![0.0; n];
    let mut x1 = vec![0.0; n];
    let mut x2 = vec![0.0; n];
    let mut state = seed;
    // Box–Muller from a simple LCG.
    let mut next_gauss = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = ((state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = ((state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    for i in 0..n {
        x0[i] = next_gauss();
        x1[i] = 0.9 * x0[i] + 0.1 * next_gauss();
        x2[i] = 0.9 * x1[i] + 0.1 * next_gauss();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(x0),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(x1),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(x2),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TabularData::new(storage)
}

#[test]
fn parcorr_recovers_chain_skeleton() {
    let data = gaussian_chain(2000, 7);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let constraints = causal_discovery::DiscoveryConstraints {
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
