//! LPCMCI end-to-end invariant: final evidence and sepsets must agree with the final PAG.
//!
//! `run_lpcmci_algorithm` threads `scored`/`sepsets` accumulators through repeated phases,
//! each preliminary iteration rebuilding a fresh complete PAG from scratch. Without the
//! reconciliation step in `lpcmci_phases.rs`, those accumulators can describe edges that no
//! longer exist in the final oriented PAG (or omit sepsets for edges that really were
//! separated). This test runs the real algorithm on synthetic data and asserts the invariant
//! holds end-to-end, not just at the unit level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Lpcmci, TemporalConstraints};
use antecedent_graph::{DenseNodeId, NodeRef};

/// Small deterministic LCG so the test doesn't depend on an external RNG crate (matches the
/// hand-rolled-generator convention already used by other tests in this crate, e.g.
/// `discovery_pc_parcorr.rs`).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.0
    }

    /// Standard normal draw via Box-Muller, using the top bits of two LCG draws.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        let u2 = ((self.next_u64() >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Synthetic 3-variable process: `x0` is AR(1); `x1` depends on `x0` contemporaneously;
/// `x2` depends on `x0` at lag 1. Enough structure to exercise both the ancestral (lagged)
/// and non-ancestral (contemporaneous) removal phases, and enough non-trivial adjacency
/// for repeated preliminary iterations to matter.
fn generate_series(seed: u64, n: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let mut rng = Lcg(seed);
    let mut x0 = vec![0.0_f64; n];
    let mut x1 = vec![0.0_f64; n];
    let mut x2 = vec![0.0_f64; n];
    for t in 0..n {
        let prev0 = if t > 0 { x0[t - 1] } else { 0.0 };
        x0[t] = 0.3 * prev0 + rng.next_gaussian();
        x1[t] = 0.6 * x0[t] + 0.2 * rng.next_gaussian();
        x2[t] = 0.5 * prev0 + rng.next_gaussian();
    }

    let mut builder = CausalSchemaBuilder::new();
    for name in ["x0", "x1", "x2"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = vec![
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
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let variables = vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    (data, variables)
}

/// Run LPCMCI with the given number of preliminary iterations and assert the
/// evidence/sepsets/PAG reconciliation invariant end-to-end.
fn assert_reconciled_with_final_pag(n_preliminary_iterations: u32) {
    let (data, variables) = generate_series(0xC0FF_EE01_u64, 250);
    let algorithm = Lpcmci::new()
        .with_fdr(false)
        .with_n_preliminary_iterations(n_preliminary_iterations)
        .with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                min_lag: Lag::CONTEMPORANEOUS,
                max_lag: Lag::from_raw(2),
            },
            alpha: 0.2,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        });
    let mut workspace = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(0xABCD);
    let result = algorithm.run(&data, &variables, &mut workspace, &ctx).unwrap();

    let graph = &result.evidence.graph;

    // Reverse map (variable, lag) -> dense node id, built from the *result* graph's own
    // node list — the same authority the reconciliation step filters against.
    let mut node_of: HashMap<(u32, u32), DenseNodeId> = HashMap::new();
    for i in 0..graph.node_count() {
        if let NodeRef::Lagged { variable, lag } = graph.nodes()[i] {
            node_of.insert((variable.raw(), lag.raw()), DenseNodeId::from_raw(i as u32));
        }
    }

    // Every evidence link must correspond to a live edge in the final PAG.
    for link in result.evidence.links.iter() {
        let a = node_of
            .get(&(link.link.source.raw(), link.link.source_lag.raw()))
            .copied()
            .unwrap_or_else(|| panic!("evidence link source not in final PAG: {:?}", link.link));
        let b = node_of
            .get(&(link.link.target.raw(), link.link.target_lag.raw()))
            .copied()
            .unwrap_or_else(|| panic!("evidence link target not in final PAG: {:?}", link.link));
        assert!(
            graph.has_edge(a, b),
            "evidence link {:?} has no corresponding edge in the final PAG",
            link.link
        );
    }

    // No sepset key may correspond to an edge that survived to the final PAG.
    for &(x, x_lag, y, y_lag) in result.sepsets.keys() {
        let (Some(&a), Some(&b)) =
            (node_of.get(&(x.raw(), x_lag.raw())), node_of.get(&(y.raw(), y_lag.raw())))
        else {
            continue;
        };
        assert!(
            !graph.has_edge(a, b),
            "sepset for ({x:?}, {x_lag:?}, {y:?}, {y_lag:?}) corresponds to a surviving edge"
        );
    }
}

#[test]
fn lpcmci_evidence_and_sepsets_agree_with_final_pag_default_preliminary_iterations() {
    // Default n_preliminary_iterations (1) — still exercises the ancestral + non-ancestral
    // phases sharing the accumulators after a single preliminary-phase fresh PAG rebuild.
    assert_reconciled_with_final_pag(1);
}

#[test]
fn lpcmci_evidence_and_sepsets_agree_with_final_pag_more_preliminary_iterations() {
    // More preliminary iterations means more fresh-PAG rebuilds accumulating into the same
    // `scored`/`sepsets`, i.e. more opportunity for the pre-fix divergence to occur.
    assert_reconciled_with_final_pag(4);
}
