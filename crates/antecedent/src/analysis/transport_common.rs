//! Helpers shared by the exact, statistical, learned-trial and grid transport modalities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, IdentityDomain, NodeRef, RegimeId, VariableId,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportDerivation, ClassicalTransportQuery, SidLimits};
use antecedent_io::{IoError, transport_proof::TransportProofWire};
use serde::Serialize;
use std::sync::Arc;

/// Wrap any conversion failure as the one transport IO error.
pub(super) fn err(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(error.to_string())
}

/// Carry an estimation failure across the prepared-study boundary, keeping a
/// registered refusal code as [`IoError::Refused`] rather than flattening it.
pub(super) fn estimate_err(error: antecedent_estimate::EstimationError) -> IoError {
    IoError::from(error)
}

/// Hex digest of a wire value under `domain`.
pub(super) fn digest(domain: IdentityDomain, value: &impl Serialize) -> Result<String, IoError> {
    Ok(antecedent_io::identity::digest_wire(domain, value)?.to_hex())
}

/// The graph fields every transport artifact carries.
pub(super) struct GraphFields<'a> {
    pub(super) nodes: &'a [u32],
    pub(super) directed: &'a [(u32, u32)],
    pub(super) bidirected: &'a [(u32, u32)],
    pub(super) selections: &'a [u32],
}

/// Rebuild the selection diagram an artifact declares and check its proof against it,
/// under the consumer's own limits. Nothing in the artifact is trusted before the check.
pub(super) fn rebuild_checked_proof(
    fields: &GraphFields<'_>,
    proof: &TransportProofWire,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, ClassicalTransportDerivation), IoError> {
    let mut graph = Admg::empty();
    for node in fields.nodes {
        graph.add_node(NodeRef::Static(VariableId::from_raw(*node))).map_err(err)?;
    }
    for &(a, b) in fields.directed {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).map_err(err)?;
    }
    for &(a, b) in fields.bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).map_err(err)?;
    }
    graph.validate().map_err(err)?;
    let diagram = SelectionDiagram::try_new(
        graph,
        fields.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
    )
    .map_err(err)?;
    let query = ClassicalTransportQuery {
        outcomes: proof.proof.outcomes.iter().copied().map(VariableId::from_raw).collect(),
        treatments: proof.proof.treatments.iter().copied().map(VariableId::from_raw).collect(),
        source: Arc::from(proof.proof.source.as_str()),
        target: Arc::from(proof.proof.target.as_str()),
    };
    let derivation = proof.check(
        &diagram,
        &query,
        SidLimits { steps: limits.operations, depth: limits.depth },
        ctx,
    )?;
    Ok((diagram, derivation))
}

/// Point each catalog binding at the snapshot its replacement providers carry.
///
/// One policy for every modality: a regime whose providers vanish, or that now spans more
/// than one snapshot, refuses. Keeping the old snapshot id for an unbound regime would
/// leave a binding that names data the handle no longer holds.
pub(super) fn rebind_snapshots<'a>(
    catalog: &EvidenceCatalog,
    snapshots_of: impl Fn(RegimeId) -> Vec<&'a str>,
) -> Result<EvidenceCatalog, IoError> {
    let mut rebound = catalog.clone();
    let mut bindings = rebound.bindings.to_vec();
    for binding in &mut bindings {
        let mut snapshots = snapshots_of(binding.regime);
        snapshots.sort_unstable();
        snapshots.dedup();
        match snapshots.as_slice() {
            [] => return Err(err("missing bound snapshot")),
            [snapshot] => binding.snapshot_identity = Arc::from(*snapshot),
            _ => return Err(err("regime has inconsistent snapshot identities")),
        }
    }
    rebound.bindings = bindings.into();
    Ok(rebound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceKind, EvidenceRegime,
        RegimeBinding, RegimeKind, SamplingDesign, TargetSampling, VariableCoordinate,
        VariableDomain,
    };

    fn catalog_bound_to(snapshot: &str) -> EvidenceCatalog {
        let coordinate = |n| VariableCoordinate {
            variable: VariableId::from_raw(n),
            domain: VariableDomain::Binary,
            unit: None,
        };
        EvidenceCatalog::try_new(
            [Environment::try_new("target", [coordinate(0), coordinate(1)], []).unwrap()],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [VariableId::from_raw(0), VariableId::from_raw(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from(snapshot),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            }],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap()
    }

    #[test]
    fn rebinding_follows_the_replacement_snapshot() {
        let catalog = catalog_bound_to("old");
        let rebound = rebind_snapshots(&catalog, |_| vec!["new", "new"]).unwrap();
        assert_eq!(rebound.bindings[0].snapshot_identity.as_ref(), "new");
        assert_eq!(catalog.bindings[0].snapshot_identity.as_ref(), "old");
    }

    #[test]
    fn rebinding_refuses_an_unbound_or_ambiguous_regime() {
        let catalog = catalog_bound_to("old");
        let missing = rebind_snapshots(&catalog, |_| Vec::new()).unwrap_err();
        assert!(missing.to_string().contains("missing bound snapshot"), "{missing}");
        let split = rebind_snapshots(&catalog, |_| vec!["a", "b"]).unwrap_err();
        assert!(split.to_string().contains("inconsistent snapshot identities"), "{split}");
    }
}
