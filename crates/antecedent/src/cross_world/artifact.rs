//! The replayable cross-world artifact: export from a computed effect, and
//! consume by replay plus the independent closed-form recomputation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::cross_world_artifact::{
    CROSS_WORLD_ARTIFACT_FEATURE, CROSS_WORLD_ARTIFACT_VERSION, CROSS_WORLD_ESTIMAND,
    CrossWorldArtifactError, CrossWorldArtifactWire, CrossWorldQueryWire, cross_world_data_digest,
    cross_world_identity,
};

use crate::error::CausalError;
use crate::gcm::NestedOutcomeMechanism;

use super::closed_form::{closed_form_agrees, closed_form_point};
use super::{
    CROSS_WORLD_MAX_ROWS, CrossWorldEffect, CrossWorldOptions, evaluate_cross_world_effect,
};

impl CrossWorldEffect {
    /// Export the replayable artifact.
    ///
    /// # Errors
    ///
    /// [`CausalError::Resource`] when the table exceeds the format's bounds
    /// (the consumer would refuse it), or an encoding failure.
    pub fn export_artifact(&self) -> Result<Vec<u8>, CausalError> {
        self.artifact_wire().and_then(|wire| wire.export()).map_err(|error| match error {
            CrossWorldArtifactError::LimitsExceeded(what) => CausalError::Resource {
                message: format!(
                    "the cross-world artifact format does not carry this table ({what} exceed \
                     the format bound of {CROSS_WORLD_MAX_ROWS} rows and {} variables)",
                    antecedent_identify::cross_world::CROSS_WORLD_MAX_NODES
                ),
            },
            other => CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string())),
        })
    }

    fn artifact_wire(&self) -> Result<CrossWorldArtifactWire, CrossWorldArtifactError> {
        CrossWorldArtifactWire {
            version: CROSS_WORLD_ARTIFACT_VERSION,
            required_features: vec![CROSS_WORLD_ARTIFACT_FEATURE.into()],
            variable_names: self.names.clone(),
            graph_edges: self.witness.graph_edges.clone(),
            query: CrossWorldQueryWire::from_query(&self.query),
            mechanism: self.mechanism.wire_tag().into(),
            estimand: CROSS_WORLD_ESTIMAND.into(),
            columns: self.columns.clone(),
            witness: self.witness.clone(),
            point_bits: self.point.to_bits(),
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
    }
}

/// Replay a cross-world artifact from its premises and accept it only when the
/// digests, the witness and the point all replay.
///
/// This is deterministic replay by the same evaluator, plus, for both mechanism
/// families, a separate closed-form recomputation of the point
/// ([`CrossWorldEffect::independently_verified`]; skipped for a singular or
/// ill-conditioned design). It detects corruption and
/// re-sealed edits that change the derivation or the point; it does not detect a
/// producer that seals a wrong table or query on purpose.
///
/// # Errors
///
/// A [`CrossWorldArtifactError`] naming the failed check.
pub fn consume_cross_world_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    consume_inner(bytes, None, ctx)
}

/// [`consume_cross_world_artifact`] that also requires the artifact's factual
/// table to have the caller's data digest ([`cross_world_data_digest`]).
///
/// # Errors
///
/// As [`consume_cross_world_artifact`], and
/// [`CrossWorldArtifactError::DataIdentityMismatch`] when the artifact was
/// computed on other data.
pub fn consume_cross_world_artifact_for_data(
    bytes: &[u8],
    expected_data_digest: &str,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    consume_inner(bytes, Some(expected_data_digest), ctx)
}

fn consume_inner(
    bytes: &[u8],
    expected_data_digest: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    let wire = CrossWorldArtifactWire::decode(bytes)?;
    wire.check_bounds()?;
    let n = wire.variable_names.len();
    if cross_world_identity(&wire).map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?
        != wire.premises_digest
    {
        return Err(CrossWorldArtifactError::PremisesMismatch);
    }
    let table_digest = cross_world_data_digest(&wire.variable_names, &wire.columns)
        .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    if table_digest != wire.data_digest
        || expected_data_digest.is_some_and(|expected| expected != table_digest)
    {
        return Err(CrossWorldArtifactError::DataIdentityMismatch);
    }
    if wire.estimand != CROSS_WORLD_ESTIMAND {
        return Err(CrossWorldArtifactError::UnsupportedSemantics("estimand"));
    }
    let mechanism = NestedOutcomeMechanism::from_wire_tag(&wire.mechanism)
        .ok_or(CrossWorldArtifactError::UnsupportedSemantics("mechanism family"))?;
    let query = wire.query.to_query()?;
    let bound = u32::try_from(n).unwrap_or(u32::MAX);
    let mut dag = Dag::with_variables(bound);
    for &(a, b) in &wire.graph_edges {
        if a >= bound || b >= bound {
            return Err(CrossWorldArtifactError::Malformed(
                "edge names an unknown variable".into(),
            ));
        }
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    }
    // The witness is verified against its own stored graph, so the graph the
    // replay uses must be that one.
    let mut stored_edges = wire.graph_edges.clone();
    stored_edges.sort_unstable();
    stored_edges.dedup();
    if stored_edges != wire.witness.graph_edges {
        return Err(CrossWorldArtifactError::WitnessMismatch(
            "the witness was derived on a different graph than the artifact stores".into(),
        ));
    }
    wire.witness.verify(n, &query).map_err(CrossWorldArtifactError::WitnessMismatch)?;
    let data = TabularData::from_f64_columns(
        wire.variable_names.iter().map(String::as_str).zip(wire.columns.iter().map(Vec::as_slice)),
    )
    .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    let mut effect = evaluate_cross_world_effect(
        dag,
        &data,
        &query,
        CrossWorldOptions { mechanism, interval_requested: false },
        ctx,
    )
    .map_err(|e| CrossWorldArtifactError::Replay(e.to_string()))?;
    if effect.point.to_bits() != wire.point_bits || effect.witness != wire.witness {
        return Err(CrossWorldArtifactError::PointMismatch);
    }
    // Both mechanism families are recomputed from the stored table by a separate
    // implementation; `None` (a singular or refused design) leaves the effect
    // replay-verified only.
    if let Some(closed) = closed_form_point(
        n,
        &wire.graph_edges,
        &wire.columns,
        &query,
        mechanism,
        wire.witness.outcome,
    ) {
        if !closed_form_agrees(closed, effect.point) {
            return Err(CrossWorldArtifactError::IndependentCheckMismatch);
        }
        effect.independently_verified = true;
    }
    Ok(effect)
}
