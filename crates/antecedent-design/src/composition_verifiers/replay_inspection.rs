//! Replay inspection of retained receipts, queries, bridges and chains.

use super::{
    FACT_SOURCE_DIGEST, Inspected, LAW_MARGINAL, MAX_SNIFF_BYTES, MAX_VALUES, TRUST_UNVERIFIED,
    coordinate_ids, failure, publish, publish_coordinates, push_value, trust_label,
};
use crate::composition_bundle::{
    BundleStage, FACT_LAW, FACT_SNAPSHOT, FACT_TRUST, LAW_JOINT, LAW_MEAN_ONLY, NodeFailure,
};
use crate::decision_artifact::source_digest;
use antecedent_core::ExecutionContext;
use antecedent_io_core::bridge_artifact::{ScientificBridgeAny, ScientificBridgeArtifactError};
use antecedent_io_core::distribution_artifact::{DistributionArtifact, DrawAlignment};
use antecedent_io_core::observation_query_artifact::{
    ObservationQueryArtifactError, ObservationQueryArtifactWire,
};
use antecedent_io_core::propagation_chain_record::{
    PropagationChainError, PropagationChainRecordWire,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use std::collections::BTreeMap;

// Historical work and frozen scores are separate independently validated objects.
// Neither reports a joint outcome law or authenticates a native provider.
pub(super) fn inspect_recalculation_receipt(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    use antecedent_io_core::recalc_receipt_artifact::RecalcReceiptArtifact;
    let artifact = RecalcReceiptArtifact::from_bytes(bytes, None)
        .map_err(|e| failure(BundleStage::SwappedEvidence, e.to_string()))?;
    let meta = artifact.meta();
    let mut facts = BTreeMap::from([
        ("execution_status".into(), "historical_receipt".into()),
        (FACT_TRUST.into(), TRUST_UNVERIFIED.into()),
    ]);
    if let Some(stage) = meta.requested.iter().find(|s| s.stage == "data_snapshot") {
        facts.insert(FACT_SNAPSHOT.into(), stage.own.clone());
    }
    if let Some(stage) =
        meta.entries.iter().find(|s| s.stage == "score_artifact" && s.tag != "refused")
    {
        facts.insert("recalc_score_identity".into(), stage.identity.clone());
    }
    Ok(Inspected {
        identity: artifact.receipt_identity().to_owned(),
        facts,
        ..Inspected::default()
    })
}
pub(super) fn inspect_frozen_scores(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    use antecedent_io_core::frozen_scores_artifact::FrozenScoreTable;
    let artifact = FrozenScoreTable::from_bytes(bytes, None)
        .map_err(|e| failure(BundleStage::SwappedEvidence, e.to_string()))?;
    Ok(Inspected {
        identity: artifact.identity().to_owned(),
        facts: BTreeMap::from([
            (FACT_SNAPSHOT.into(), artifact.meta().snapshot_digest.clone()),
            ("recalc_score_identity".into(), artifact.meta().fit_identity.clone()),
            ("reusable_state".into(), "frozen_same_row_scores".into()),
            (FACT_TRUST.into(), TRUST_UNVERIFIED.into()),
        ]),
        ..Inspected::default()
    })
}

/// The facts of a draw table that a node publishes as its law: joint draws only
/// for aligned rows, the table's trust and snapshot, the digest a decision over it
/// retains and the coordinates it carries. The whole-set quantity digest is left to
/// the distribution itself: a node downstream of one (a bridge) changes a column.
fn publish_draws(facts: &mut BTreeMap<String, String>, artifact: &DistributionArtifact) {
    let metadata = artifact.metadata();
    let identity = &metadata.identity;
    publish(
        facts,
        FACT_LAW,
        match identity.alignment {
            DrawAlignment::Joint => LAW_JOINT,
            DrawAlignment::IndependentMarginals => LAW_MARGINAL,
        },
    );
    publish(facts, FACT_TRUST, trust_label(metadata.trust));
    publish(
        facts,
        FACT_SNAPSHOT,
        format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
    );
    publish(facts, FACT_SOURCE_DIGEST, source_digest(artifact));
    publish_coordinates(facts, &coordinate_ids(artifact.quantities()), false);
}

fn push_draw_means(values: &mut Vec<(String, f64)>, artifact: &DistributionArtifact) {
    push_value(values, "n_draws".to_owned(), artifact.n_draws() as f64);
    for coordinate in 0..artifact.shape()[1].min(MAX_VALUES) {
        if let Ok(mean) = artifact.mean(coordinate) {
            push_value(values, format!("mean.{coordinate}"), mean);
        }
    }
}

fn observation_failure(error: &ObservationQueryArtifactError) -> NodeFailure {
    let stage = match error {
        ObservationQueryArtifactError::UnsupportedVersion { .. } => {
            BundleStage::IncompatibleVersion
        }
        ObservationQueryArtifactError::LimitsExceeded(_) => BundleStage::Oversized,
        ObservationQueryArtifactError::Cancelled => BundleStage::CallbackUnavailable,
        _ => BundleStage::TamperedQuantity,
    };
    failure(stage, error.to_string())
}

/// An observation query is replayed: its identity is recomputed from the premises,
/// the query and the reading, and its result is computed again and required to
/// equal the stored one. A query that redraws parameters publishes its aligned
/// draws, so a decision over them rests on this node as on a distribution.
pub(super) fn inspect_observation_query(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    if bytes.len() > MAX_SNIFF_BYTES {
        return Err(failure(BundleStage::Oversized, "the observation query exceeds its bound"));
    }
    let wire = ObservationQueryArtifactWire::decode(bytes).map_err(|e| observation_failure(&e))?;
    let ctx = ExecutionContext::production(1, 1);
    let consumed = wire.consume(None, &ctx).map_err(|e| observation_failure(&e))?;
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_TRUST, TRUST_UNVERIFIED);
    publish(&mut facts, "observation_operation", consumed.operation.clone());
    publish(&mut facts, "inference_basis", consumed.inference_basis.clone());
    let mut values = Vec::new();
    if let Ok(draws) = wire.distribution() {
        publish_draws(&mut facts, &draws);
        push_draw_means(&mut values, &draws);
    }
    Ok(Inspected { identity: consumed.identity, facts, values, ..Inspected::default() })
}

fn bridge_failure(error: &ScientificBridgeArtifactError) -> NodeFailure {
    let stage = match error {
        ScientificBridgeArtifactError::UnsupportedVersion { .. } => {
            BundleStage::IncompatibleVersion
        }
        ScientificBridgeArtifactError::LimitsExceeded(_) => BundleStage::Oversized,
        _ => BundleStage::TamperedQuantity,
    };
    failure(stage, error.to_string())
}

/// A scientific bridge is replayed: the contract is supported again and the value
/// (or, for a bridge of draws, every mapped draw) is derived again and required to
/// equal the stored one bit for bit. A bridge of draws publishes the bridged
/// aligned table, so a decision over it rests on this node; a bridged mean
/// publishes `mean_only`.
pub(super) fn inspect_scientific_bridge(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    if bytes.len() > MAX_SNIFF_BYTES {
        return Err(failure(BundleStage::Oversized, "the scientific bridge exceeds its bound"));
    }
    let artifact = ScientificBridgeAny::from_bytes(bytes, None).map_err(|e| bridge_failure(&e))?;
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_TRUST, TRUST_UNVERIFIED);
    let mut values = Vec::new();
    let mut sources = Vec::new();
    match &artifact {
        ScientificBridgeAny::Point(point) => {
            let bridged = point.bridged();
            publish(&mut facts, FACT_LAW, LAW_MEAN_ONLY);
            publish(&mut facts, FACT_SNAPSHOT, bridged.claim.identities.data_snapshot.to_hex());
            let target = ScientificQuantityWire::from(&bridged.contract.target.quantity);
            publish_coordinates(&mut facts, &coordinate_ids(std::slice::from_ref(&target)), false);
            push_value(&mut values, "value".to_owned(), bridged.value);
            push_value(&mut values, "source_value".to_owned(), bridged.source_value);
        }
        ScientificBridgeAny::Draws(draws) => {
            let bridged = draws.bridged();
            publish_draws(&mut facts, bridged.distribution());
            sources.push(source_digest(bridged.source()));
            push_draw_means(&mut values, bridged.distribution());
            push_value(&mut values, "variance".to_owned(), bridged.claim().variance);
        }
        _ => {
            return Err(failure(
                BundleStage::SwappedEvidence,
                "an unknown scientific bridge shape",
            ));
        }
    }
    Ok(Inspected {
        identity: artifact.identity().to_owned(),
        facts,
        values,
        sources,
        ..Inspected::default()
    })
}

fn chain_failure(error: &PropagationChainError) -> NodeFailure {
    let stage = match error {
        PropagationChainError::UnsupportedVersion { .. } => BundleStage::IncompatibleVersion,
        PropagationChainError::LimitsExceeded(_) => BundleStage::Oversized,
        PropagationChainError::Cancelled => BundleStage::CallbackUnavailable,
        _ => BundleStage::TamperedQuantity,
    };
    failure(stage, error.to_string())
}

/// A propagation chain is replayed stage by stage from its stored draws. The
/// record keeps the summary of every stage and not the output column, so the node
/// carries the answer's value and spread and is not a source of draws.
pub(super) fn inspect_propagation_chain(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    if bytes.len() > MAX_SNIFF_BYTES {
        return Err(failure(BundleStage::Oversized, "the propagation chain exceeds its bound"));
    }
    let record = PropagationChainRecordWire::decode(bytes).map_err(|e| chain_failure(&e))?;
    let ctx = ExecutionContext::production(1, 1);
    let consumed = record.consume(None, &ctx).map_err(|e| chain_failure(&e))?;
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_TRUST, TRUST_UNVERIFIED);
    publish(&mut facts, "chain_label", consumed.label.clone());
    let mut values = Vec::new();
    if let Some(answer) = consumed.answer() {
        for name in ["mean", "variance"] {
            if let Some(value) = answer.value(name) {
                push_value(&mut values, format!("answer.{name}"), value);
            }
        }
    }
    push_value(&mut values, "stages".to_owned(), consumed.results.len() as f64);
    Ok(Inspected { identity: consumed.identity, facts, values, ..Inspected::default() })
}
