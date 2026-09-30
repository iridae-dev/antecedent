//! Replayable artifact for one cross-world edge contrast.
//!
//! Format version 1. The artifact keeps everything the estimand depends on: the
//! variable names, the graph, the cross-world query (worlds, per-edge routes,
//! coupling, observations), the mechanism family, the factual table, the witness
//! that derived the estimand, and the point. A consumer trusts none of it: it
//! recomputes the premises digest and the data digest, re-checks the query on
//! the graph and requires the stored witness to equal the recomputed derivation,
//! refits the structural model from the stored table, replays the coupled
//! operation with the same evaluator and accepts only the identical point.
//!
//! Two digests, two typed errors. The *premises digest* binds the variable
//! names, graph, query, estimand, mechanism family and the data digest; the
//! *data digest* binds the names and the factual table's bits, so a caller can
//! compare it with a digest of their own data. A stale premises digest is
//! [`CrossWorldArtifactError::PremisesMismatch`], a table that no longer matches
//! its digest is [`CrossWorldArtifactError::DataIdentityMismatch`].
//!
//! What replay protects against: corruption and any edit that is not re-sealed,
//! and an edit that is re-sealed but changes the derivation or the point. What it
//! does not protect against: a producer that seals a wrong table or query on
//! purpose (the consumer replays whatever premises it is given), and an
//! evaluator bug (the same evaluator recomputes the point; for the
//! linear-Gaussian family the facade additionally cross-checks the point with a
//! separate closed-form OLS implementation).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::EdgeRoute;
use antecedent_core::{
    CrossWorldQuery, ExogenousCoupling, IdentityDomain, VariableId, WorldId, WorldObservation,
    WorldSpec,
};
use antecedent_identify::cross_world::{CROSS_WORLD_MAX_NODES, CrossWorldWitness};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const CROSS_WORLD_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const CROSS_WORLD_ARTIFACT_FEATURE: &str = "cross_world_edge_contrast_v1";
/// Most factual rows an artifact may carry, enforced when it is exported and when
/// it is consumed.
pub const CROSS_WORLD_MAX_ROWS: usize = 100_000;
/// The estimand this artifact names.
pub const CROSS_WORLD_ESTIMAND: &str = "mean over units of observed(plus world) - observed(minus world) under shared abduced exogenous terms";

/// Why a cross-world artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CrossWorldArtifactError {
    /// The bytes do not decode as this format.
    #[error("cross-world artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker or version is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The factual table does not match its data digest, or does not match the
    /// data digest the caller expected.
    #[error("data identity mismatch")]
    DataIdentityMismatch,
    /// The artifact could not be encoded.
    #[error("cross-world artifact does not encode: {0}")]
    Encode(String),
    /// The stored premises are not a well-formed query, graph or table.
    #[error("malformed premises: {0}")]
    Malformed(String),
    /// The query does not check on the stored graph, or the stored witness
    /// differs from the recomputed derivation.
    #[error("witness does not replay: {0}")]
    WitnessMismatch(String),
    /// The recomputed point differs from the stored point.
    #[error("point does not replay")]
    PointMismatch,
    /// A separate closed-form recomputation of the point (linear-Gaussian family)
    /// disagrees with the replayed point.
    #[error("independent closed-form check disagrees with the replayed point")]
    IndependentCheckMismatch,
    /// The replay itself refused or failed.
    #[error("replay failed: {0}")]
    Replay(String),
}

impl From<IoError> for CrossWorldArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

/// One world on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldWorldWire {
    /// `(variable, value bits)`, canonical.
    pub interventions: Vec<(u32, u64)>,
    /// `(parent, child, source world)`, canonical.
    pub routes: Vec<(u32, u32, u8)>,
}

/// A cross-world query on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldQueryWire {
    /// Coupling tag.
    pub coupling: String,
    /// Worlds in declaration order.
    pub worlds: Vec<CrossWorldWorldWire>,
    /// Added observation `(world, variable)`.
    pub plus: (u8, u32),
    /// Subtracted observation `(world, variable)`.
    pub minus: (u8, u32),
}

impl CrossWorldQueryWire {
    /// Encode a query.
    #[must_use]
    pub fn from_query(query: &CrossWorldQuery) -> Self {
        Self {
            coupling: query.coupling().tag().into(),
            worlds: query
                .worlds()
                .iter()
                .map(|w| CrossWorldWorldWire {
                    interventions: w.interventions().map(|(v, x)| (v.raw(), x.to_bits())).collect(),
                    routes: w
                        .routes()
                        .iter()
                        .map(|r| {
                            (
                                r.parent.raw(),
                                r.child.raw(),
                                u8::try_from(r.source.index()).unwrap_or(u8::MAX),
                            )
                        })
                        .collect(),
                })
                .collect(),
            plus: (
                u8::try_from(query.plus().world.index()).unwrap_or(u8::MAX),
                query.plus().variable.raw(),
            ),
            minus: (
                u8::try_from(query.minus().world.index()).unwrap_or(u8::MAX),
                query.minus().variable.raw(),
            ),
        }
    }

    /// Decode and validate a query.
    ///
    /// # Errors
    /// An unknown coupling or a query that does not validate.
    pub fn to_query(&self) -> Result<CrossWorldQuery, CrossWorldArtifactError> {
        let malformed =
            |e: antecedent_core::QueryError| CrossWorldArtifactError::Malformed(e.to_string());
        if self.coupling != ExogenousCoupling::SharedAbducedExogenous.tag() {
            return Err(CrossWorldArtifactError::UnsupportedSemantics("exogenous coupling"));
        }
        let worlds = self
            .worlds
            .iter()
            .map(|w| {
                WorldSpec::new(
                    w.interventions
                        .iter()
                        .map(|&(v, bits)| (VariableId::from_raw(v), f64::from_bits(bits))),
                    w.routes.iter().map(|&(p, c, s)| EdgeRoute {
                        parent: VariableId::from_raw(p),
                        child: VariableId::from_raw(c),
                        source: WorldId::new(s),
                    }),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(malformed)?;
        let observation = |(world, variable): (u8, u32)| WorldObservation {
            world: WorldId::new(world),
            variable: VariableId::from_raw(variable),
        };
        CrossWorldQuery::new(
            worlds,
            ExogenousCoupling::SharedAbducedExogenous,
            observation(self.plus),
            observation(self.minus),
        )
        .map_err(malformed)
    }
}

/// Versioned cross-world execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Variable names in variable-id order.
    pub variable_names: Vec<String>,
    /// Directed edges as variable-id pairs, sorted.
    pub graph_edges: Vec<(u32, u32)>,
    /// The cross-world query.
    pub query: CrossWorldQueryWire,
    /// Mechanism family: `linear_gaussian` or `non_separable_basis`.
    pub mechanism: String,
    /// The estimand named by [`CROSS_WORLD_ESTIMAND`].
    pub estimand: String,
    /// The factual table, one column per variable in id order.
    pub columns: Vec<Vec<f64>>,
    /// The derivation that licensed the estimand.
    pub witness: CrossWorldWitness,
    /// IEEE bits of the point.
    pub point_bits: u64,
    /// Digest of the names and the factual table ([`cross_world_data_digest`]).
    pub data_digest: String,
    /// Digest of the names, graph, query, estimand, mechanism family and data
    /// digest ([`cross_world_identity`]).
    pub premises_digest: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// Identity of the factual table: the variable names and every value's bits.
///
/// # Errors
/// Encoding failure.
pub fn cross_world_data_digest(names: &[String], columns: &[Vec<f64>]) -> Result<String, IoError> {
    let table: Vec<Vec<u64>> =
        columns.iter().map(|c| c.iter().map(|x| x.to_bits()).collect()).collect();
    Ok(crate::identity::digest_wire(
        IdentityDomain::DataSnapshot,
        &("cross_world_edge_contrast_v1.table", names, table),
    )?
    .to_hex())
}

/// Scientific identity of a cross-world execution: the variable names, graph,
/// canonical query (worlds, per-edge routes, coupling, observations), estimand,
/// mechanism family and the stored data digest. The point and witness are
/// outputs, not premises; the table enters through its digest.
///
/// # Errors
/// Encoding failure.
pub fn cross_world_identity(wire: &CrossWorldArtifactWire) -> Result<String, IoError> {
    let mut edges = wire.graph_edges.clone();
    edges.sort_unstable();
    Ok(crate::identity::digest_wire(
        IdentityDomain::Program,
        &(
            "cross_world_edge_contrast_v1",
            &wire.variable_names,
            edges,
            &wire.query,
            &wire.estimand,
            &wire.mechanism,
            &wire.data_digest,
        ),
    )?
    .to_hex())
}

impl CrossWorldArtifactWire {
    /// Refuse a table outside the format's bounds: one to
    /// [`CROSS_WORLD_MAX_NODES`] variables, one to [`CROSS_WORLD_MAX_ROWS`] rows,
    /// one column per variable. The producer and the consumer both call this, so
    /// an artifact this writer exports is one this reader accepts.
    ///
    /// # Errors
    /// [`CrossWorldArtifactError::LimitsExceeded`] or `Malformed`.
    pub fn check_bounds(&self) -> Result<(), CrossWorldArtifactError> {
        let n = self.variable_names.len();
        if n == 0 || n > CROSS_WORLD_MAX_NODES {
            return Err(CrossWorldArtifactError::LimitsExceeded("variables"));
        }
        let rows = self.columns.first().map_or(0, Vec::len);
        if rows > CROSS_WORLD_MAX_ROWS {
            return Err(CrossWorldArtifactError::LimitsExceeded("rows"));
        }
        if self.columns.len() != n || self.columns.iter().any(|c| c.len() != rows) || rows == 0 {
            return Err(CrossWorldArtifactError::Malformed("table shape".into()));
        }
        Ok(())
    }

    /// Seal an artifact: check the bounds, then fill the data digest and the
    /// premises digest.
    ///
    /// # Errors
    /// A bound exceeded, or the premises do not encode.
    pub fn sealed(mut self) -> Result<Self, CrossWorldArtifactError> {
        self.check_bounds()?;
        self.data_digest = cross_world_data_digest(&self.variable_names, &self.columns)
            .map_err(|e| CrossWorldArtifactError::Encode(e.to_string()))?;
        self.premises_digest = cross_world_identity(&self)
            .map_err(|e| CrossWorldArtifactError::Encode(e.to_string()))?;
        Ok(self)
    }

    /// Encode as CBOR, refusing a table outside the bounds.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, CrossWorldArtifactError> {
        self.check_bounds()?;
        crate::to_cbor(self).map_err(|e| CrossWorldArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature first.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, CrossWorldArtifactError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != CROSS_WORLD_ARTIFACT_VERSION {
            return Err(CrossWorldArtifactError::UnsupportedSemantics("artifact version"));
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [CROSS_WORLD_ARTIFACT_FEATURE] {
            return Err(CrossWorldArtifactError::UnsupportedSemantics("required features"));
        }
        Ok(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire() -> CrossWorldArtifactWire {
        let query = CrossWorldQueryWire {
            coupling: "shared_abduced_exogenous".into(),
            worlds: vec![
                CrossWorldWorldWire { interventions: vec![(0, 0.0f64.to_bits())], routes: vec![] },
                CrossWorldWorldWire {
                    interventions: vec![(0, 1.0f64.to_bits())],
                    routes: vec![(0, 1, 0)],
                },
            ],
            plus: (1, 1),
            minus: (0, 1),
        };
        let witness = antecedent_identify::cross_world::check_cross_world_edges(
            2,
            &[(0, 1)],
            &query.to_query().unwrap(),
        )
        .unwrap();
        CrossWorldArtifactWire {
            version: CROSS_WORLD_ARTIFACT_VERSION,
            required_features: vec![CROSS_WORLD_ARTIFACT_FEATURE.into()],
            variable_names: vec!["x".into(), "y".into()],
            graph_edges: vec![(0, 1)],
            query,
            mechanism: "linear_gaussian".into(),
            estimand: CROSS_WORLD_ESTIMAND.into(),
            columns: vec![vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 2.0]],
            witness,
            point_bits: 0,
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
        .unwrap()
    }

    /// Every wire struct rejects a field it does not know, so a later format's
    /// additions are refused by this reader rather than silently dropped.
    #[test]
    fn every_wire_struct_rejects_unknown_fields() {
        fn closed<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T, what: &str) {
            let mut json = serde_json::to_value(value).unwrap();
            assert!(serde_json::from_value::<T>(json.clone()).is_ok(), "{what} round trip");
            json.as_object_mut().unwrap().insert("unexpected".into(), serde_json::json!(1));
            assert!(serde_json::from_value::<T>(json).is_err(), "{what} accepted an unknown field");
        }
        let w = wire();
        closed(&w, "artifact");
        closed(&w.query, "query");
        closed(&w.query.worlds[0], "world");
        closed(&w.witness, "witness");
        closed(&w.witness.counterfactual_nodes[0], "counterfactual node");
    }

    #[test]
    fn the_producer_enforces_the_row_bound_the_consumer_enforces() {
        let mut w = wire();
        w.columns = vec![vec![0.5; CROSS_WORLD_MAX_ROWS + 1]; 2];
        assert_eq!(w.check_bounds(), Err(CrossWorldArtifactError::LimitsExceeded("rows")));
        assert_eq!(
            w.clone().sealed().unwrap_err(),
            CrossWorldArtifactError::LimitsExceeded("rows")
        );
        assert_eq!(w.export().unwrap_err(), CrossWorldArtifactError::LimitsExceeded("rows"));
        let mut w = wire();
        w.columns = vec![vec![0.5; CROSS_WORLD_MAX_ROWS]; 2];
        assert!(w.check_bounds().is_ok());
    }

    #[test]
    fn the_data_digest_binds_the_table_and_the_premises_digest_binds_the_data_digest() {
        let w = wire();
        let mut edited = w.clone();
        edited.columns[1][0] += 1.0;
        assert_ne!(
            cross_world_data_digest(&edited.variable_names, &edited.columns).unwrap(),
            w.data_digest
        );
        // The premises digest does not read the table, only its digest.
        assert_eq!(cross_world_identity(&edited).unwrap(), w.premises_digest);
        let mut renamed = w.clone();
        renamed.variable_names[0] = "z".into();
        assert_ne!(
            cross_world_data_digest(&renamed.variable_names, &renamed.columns).unwrap(),
            w.data_digest
        );
    }
}
