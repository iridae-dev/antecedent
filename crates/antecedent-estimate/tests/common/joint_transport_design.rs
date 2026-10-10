//! Independent known-Gaussian source/target DGP and original checked structural proof.
#![allow(clippy::cast_precision_loss, reason = "bounded source and row indices")]
use super::candidate;
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, SourceData, SourceSharing, TargetData, VaryingBlock,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{TransportFormula, TransportIdentification, TransportIdentifier};
use std::sync::Arc;
pub const FEATURE: u32 = 1;
pub const DRAWS: usize = 4096;
pub fn proof() -> TransportIdentification {
    // Actual structural standardization checker, rather than a caller-created certificate.
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(FEATURE)]).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0., 1.])),
            ),
        }),
        "source",
        "target",
        [VariableId::from_raw(0)],
    );
    let identified = TransportIdentifier::new().identify(&diagram, &query).unwrap();
    assert!(matches!(
        &identified,
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { .. },
            ..
        }
    ));
    identified
}

pub fn source(
    n: usize,
    position: usize,
    seed: u64,
    varying: VaryingBlock,
    sharing: SourceSharing,
    quadratic: bool,
) -> SourceData {
    let mut rng = candidate::Generator::new(seed);
    let block = if sharing == SourceSharing::SharedVaryingBlock { 0 } else { position };
    let intercept = 0.4 + 0.7 * block as f64;
    let slope = if varying == VaryingBlock::Intercept { 0.5 } else { 0.5 + 0.3 * block as f64 };
    let variance: f64 = if position == 0 { 1. } else { 2.25 };
    let (mut x, mut t, mut y) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let z = 2. * rng.uniform() - 1.;
        let a = rng.binary(if position == 0 { 0.5 } else { 0.35 }) == 1;
        y.push(
            intercept
                + slope * z
                + (if a { 1. } else { 0. })
                    * (2. + 0.4 * z + if quadratic { 0.2 * z * z } else { 0. })
                + variance.sqrt() * rng.normal(),
        );
        x.push(z);
        t.push(a);
    }
    SourceData {
        id: format!("s{position}"),
        identity: DataIdentity {
            snapshot_digest: format!("{seed}-{position}"),
            datum_ids: (0..n).map(|i| format!("s{position}-{seed}-{i}")).collect(),
        },
        treatment: t,
        outcome: y,
        covariates: vec![x],
        noise_variance: variance,
    }
}
pub fn target() -> TargetData {
    TargetData {
        identity: DataIdentity {
            snapshot_digest: "fixed_target_x".into(),
            datum_ids: (0..4).map(|i| format!("target-{i}")).collect(),
        },
        rows: 4,
        covariates: vec![vec![-0.25, 0.25, 0.25, 0.55]],
    }
}
