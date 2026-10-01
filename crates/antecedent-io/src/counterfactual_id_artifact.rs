//! Replayable artifact for one counterfactual identification on a bounded ADMG
//! (2.2B X8): format `counterfactual_id_admg_v1`.
//!
//! A separate format, not a version of the cross-world edge-contrast artifact:
//! its payload is a finite joint law and an identified functional, not an
//! abduced table. The artifact keeps the variable names and levels, the graph
//! (directed and bidirected edges), the counterfactual event query, the search
//! limits and memory cap the derivation ran under, the canonical derivation
//! with its search accounting and counterfactual graph, the joint law, and the
//! point. Every wire struct rejects unknown fields, so the 2.2A reader refuses
//! these bytes and this reader refuses the 2.2A bytes.
//!
//! Two digests, two typed errors. The *premises digest* binds the names,
//! levels, graph, query, estimand, contract, search limits and memory cap, and
//! the data digest; the *data digest* binds the law (axes, probability bits,
//! counts, origin, population and snapshot identity). The consumer
//! (`antecedent::counterfactual_id::consume_counterfactual_id_artifact`) checks
//! both, re-derives under the stored limits (refusing limits above its own
//! before any work), requires the identical derivation and search accounting,
//! re-evaluates the identical point, and recomputes it with a separate
//! direct-sum evaluator.
//!
//! What replay does not protect against: a producer that seals a wrong graph,
//! query or law on purpose (the consumer replays whatever premises it is
//! given), and an error in the identification theory shared by the producer
//! and the consumer (the same derivation code runs twice; the independent
//! evaluator re-checks the arithmetic of the functional, not its derivation).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, EdgeRoute, ExogenousCoupling, IdentityDomain,
    VariableId, WorldId, WorldSpec,
};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_MAX_LEVELS, COUNTERFACTUAL_ID_MAX_VARIABLES, CounterfactualGraphRecord,
    CounterfactualIdSearchRecord,
};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const COUNTERFACTUAL_ID_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const COUNTERFACTUAL_ID_ARTIFACT_FEATURE: &str = "counterfactual_id_admg_v1";
/// The estimand the artifact names.
pub const COUNTERFACTUAL_ID_ESTIMAND: &str = "P(Y_x = y | X = x') = P(Y_x = y, X = x') / P(X = x'), its distribution over the levels of Y and E[Y_x | X = x'] - E[Y | X = x'], worlds sharing latent exogenous terms";

/// Why a counterfactual-identification artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CounterfactualIdArtifactError {
    /// The bytes do not decode as this format.
    #[error("counterfactual identification artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker, version, estimand or contract is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection or limit exceeds the consumer's bound; refused before
    /// any work. Retry with larger limits; this is not a claim the artifact is
    /// invalid.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The law does not match its data digest, or not the caller's.
    #[error("data identity mismatch")]
    DataIdentityMismatch,
    /// The artifact could not be encoded.
    #[error("counterfactual identification artifact does not encode: {0}")]
    Encode(String),
    /// The stored premises are not a well-formed problem, query or law.
    #[error("malformed premises: {0}")]
    Malformed(String),
    /// The re-derivation refused, or differs from the stored derivation, search
    /// accounting or counterfactual graph.
    #[error("derivation does not replay: {0}")]
    DerivationMismatch(String),
    /// The re-evaluated point differs from the stored point.
    #[error("point does not replay")]
    PointMismatch,
    /// The separate direct-sum evaluation disagrees with the replayed point.
    #[error("independent direct-sum evaluation disagrees with the replayed point")]
    IndependentCheckMismatch,
}

impl CounterfactualIdArtifactError {
    /// The registered `(reason code, detail)` pair this error surfaces as: a
    /// limits refusal is a bound (`counterfactual_id.bounds_exceeded`); every
    /// other failure is an artifact that does not replay
    /// (`counterfactual_id.invalid_artifact`).
    #[must_use]
    pub const fn reason(&self) -> (&'static str, &'static str) {
        match self {
            Self::LimitsExceeded(_) => ("route_not_supported", "counterfactual_id.bounds_exceeded"),
            _ => ("invalid_argument", "counterfactual_id.invalid_artifact"),
        }
    }
}

impl From<IoError> for CounterfactualIdArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

/// One world on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdWorldWire {
    /// `(variable, value bits)`, canonical.
    pub interventions: Vec<(u32, u64)>,
    /// `(parent, child, source world)`, canonical.
    pub routes: Vec<(u32, u32, u8)>,
}

/// A counterfactual event query on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdQueryWire {
    /// Coupling tag.
    pub coupling: String,
    /// Worlds in declaration order.
    pub worlds: Vec<CounterfactualIdWorldWire>,
    /// Event atoms `(world, variable, level bits)`, canonical.
    pub event: Vec<(u8, u32, u64)>,
    /// Conditioning atoms `(world, variable, level bits)`, canonical.
    pub given: Vec<(u8, u32, u64)>,
}

impl CounterfactualIdQueryWire {
    /// Encode a query.
    #[must_use]
    pub fn from_query(query: &CounterfactualEventQuery) -> Self {
        let atom = |a: &CounterfactualEvent| {
            (u8::try_from(a.world().index()).unwrap_or(u8::MAX), a.variable().raw(), a.level_bits())
        };
        Self {
            coupling: query.coupling().tag().into(),
            worlds: query
                .worlds()
                .iter()
                .map(|w| CounterfactualIdWorldWire {
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
            event: query.event().iter().map(atom).collect(),
            given: query.given().iter().map(atom).collect(),
        }
    }

    /// Decode and validate a query.
    ///
    /// # Errors
    /// An unknown coupling or a query that does not validate.
    pub fn to_query(&self) -> Result<CounterfactualEventQuery, CounterfactualIdArtifactError> {
        let malformed = |e: antecedent_core::QueryError| {
            CounterfactualIdArtifactError::Malformed(e.to_string())
        };
        let coupling =
            [ExogenousCoupling::SharedLatentExogenous, ExogenousCoupling::SharedAbducedExogenous]
                .into_iter()
                .find(|c| c.tag() == self.coupling)
                .ok_or(CounterfactualIdArtifactError::UnsupportedSemantics("exogenous coupling"))?;
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
        let atoms = |list: &[(u8, u32, u64)]| {
            list.iter()
                .map(|&(w, v, bits)| {
                    CounterfactualEvent::new(
                        WorldId::new(w),
                        VariableId::from_raw(v),
                        f64::from_bits(bits),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(malformed)
        };
        let query = CounterfactualEventQuery::new(
            worlds,
            coupling,
            atoms(&self.event)?,
            atoms(&self.given)?,
        )
        .map_err(malformed)?;
        // A canonical query re-encodes to itself: an unsorted or duplicated
        // conjunction on the wire is malformed, not silently repaired.
        if Self::from_query(&query) != *self {
            return Err(CounterfactualIdArtifactError::Malformed(
                "the stored query is not canonical".into(),
            ));
        }
        Ok(query)
    }
}

/// One axis of the stored joint law.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdAxisWire {
    /// Variable.
    pub variable: u32,
    /// Level bits in the law's physical order.
    pub levels: Vec<u64>,
}

/// The joint law the point was computed on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdLawWire {
    /// Axes in physical order (last fastest).
    pub axes: Vec<CounterfactualIdAxisWire>,
    /// Probability bits, row-major.
    pub probabilities: Vec<u64>,
    /// Cell counts of an empirical law; `None` for a supplied exact law.
    pub counts: Option<Vec<u64>>,
    /// Law origin (`supplied_exact` or `empirical_plugin`).
    pub origin: String,
    /// Population identity.
    pub population: String,
    /// Snapshot identity.
    pub snapshot: String,
}

/// The stored point, as IEEE bits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdPointWire {
    /// `P(event | given)`.
    pub probability: u64,
    /// `P(given)`.
    pub conditioning_probability: u64,
    /// `(outcome level, numerator)` per level.
    pub numerators: Vec<(u64, u64)>,
    /// `E[Y_x | X = x']`.
    pub counterfactual_mean: Option<u64>,
    /// `E[Y | X = x']`.
    pub observed_mean: Option<u64>,
    /// The contrast.
    pub effect: Option<u64>,
}

/// Search limits and accounting on the wire.
pub type CounterfactualIdSearchWire = CounterfactualIdSearchRecord;

/// Versioned counterfactual identification with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Variable names in variable-id order.
    pub variable_names: Vec<String>,
    /// Level bits per variable, declared order.
    pub levels: Vec<Vec<u64>>,
    /// Directed edges, sorted.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges `(low, high)`, sorted.
    pub bidirected: Vec<(u32, u32)>,
    /// The counterfactual event query.
    pub query: CounterfactualIdQueryWire,
    /// The estimand ([`COUNTERFACTUAL_ID_ESTIMAND`]).
    pub estimand: String,
    /// The contract the derivation attests.
    pub contract: String,
    /// Search limits, memory cap and accounting of the derivation.
    pub search: CounterfactualIdSearchWire,
    /// Canonical text of the derivation.
    pub derivation: String,
    /// The counterfactual graph of the requested level.
    pub counterfactual_graph: Option<CounterfactualGraphRecord>,
    /// The joint law.
    pub law: CounterfactualIdLawWire,
    /// The point.
    pub point: CounterfactualIdPointWire,
    /// Digest of the law ([`counterfactual_id_data_digest`]).
    pub data_digest: String,
    /// Digest of the premises ([`counterfactual_id_identity`]).
    pub premises_digest: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// Identity of the joint law: axes, probability bits, counts, origin,
/// population and snapshot identity.
///
/// # Errors
/// Encoding failure.
#[doc(hidden)]
pub fn counterfactual_id_data_digest(law: &CounterfactualIdLawWire) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(
        IdentityDomain::DataSnapshot,
        &("counterfactual_id_admg_v1.law", law),
    )?
    .to_hex())
}

/// Scientific identity: names, levels, graph, query, estimand, contract, the
/// search limits and memory cap, and the data digest. The derivation, its
/// accounting and the point are outputs, not premises.
///
/// # Errors
/// Encoding failure.
#[doc(hidden)]
pub fn counterfactual_id_identity(wire: &CounterfactualIdArtifactWire) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(
        IdentityDomain::Program,
        &(
            "counterfactual_id_admg_v1",
            &wire.variable_names,
            &wire.levels,
            &wire.directed,
            &wire.bidirected,
            &wire.query,
            &wire.estimand,
            &wire.contract,
            (wire.search.operations_limit, wire.search.depth_limit, wire.search.memory_limit_bytes),
            &wire.data_digest,
        ),
    )?
    .to_hex())
}

impl CounterfactualIdArtifactWire {
    /// Refuse an artifact outside the format's bounds: one to
    /// [`COUNTERFACTUAL_ID_MAX_VARIABLES`] named variables, each with one to
    /// [`COUNTERFACTUAL_ID_MAX_LEVELS`] levels, a law with one axis per variable
    /// and one probability (and count) per cell. Producer and consumer both
    /// call this.
    ///
    /// # Errors
    /// [`CounterfactualIdArtifactError::LimitsExceeded`] or `Malformed`.
    pub fn check_bounds(&self) -> Result<(), CounterfactualIdArtifactError> {
        let n = self.variable_names.len();
        if n == 0 || n > COUNTERFACTUAL_ID_MAX_VARIABLES {
            return Err(CounterfactualIdArtifactError::LimitsExceeded("variables"));
        }
        if self.levels.iter().any(|l| l.len() > COUNTERFACTUAL_ID_MAX_LEVELS) {
            return Err(CounterfactualIdArtifactError::LimitsExceeded("levels"));
        }
        if self.levels.len() != n || self.law.axes.len() != n {
            return Err(CounterfactualIdArtifactError::Malformed("variable count".into()));
        }
        let cells: usize = self.law.axes.iter().map(|a| a.levels.len()).product();
        if self.law.axes.iter().any(|a| a.levels.len() > COUNTERFACTUAL_ID_MAX_LEVELS) {
            return Err(CounterfactualIdArtifactError::LimitsExceeded("law levels"));
        }
        if self.law.probabilities.len() != cells
            || self.law.counts.as_ref().is_some_and(|c| c.len() != cells)
        {
            return Err(CounterfactualIdArtifactError::Malformed("law shape".into()));
        }
        Ok(())
    }

    /// Seal: check the bounds, then fill the data and premises digests.
    ///
    /// # Errors
    /// A bound exceeded, or the premises do not encode.
    pub fn sealed(mut self) -> Result<Self, CounterfactualIdArtifactError> {
        self.check_bounds()?;
        self.data_digest = counterfactual_id_data_digest(&self.law)
            .map_err(|e| CounterfactualIdArtifactError::Encode(e.to_string()))?;
        self.premises_digest = counterfactual_id_identity(&self)
            .map_err(|e| CounterfactualIdArtifactError::Encode(e.to_string()))?;
        Ok(self)
    }

    /// Encode as CBOR, refusing an artifact outside the bounds.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, CounterfactualIdArtifactError> {
        self.check_bounds()?;
        crate::to_cbor(self).map_err(|e| CounterfactualIdArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature first.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, CounterfactualIdArtifactError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != COUNTERFACTUAL_ID_ARTIFACT_VERSION {
            return Err(CounterfactualIdArtifactError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [COUNTERFACTUAL_ID_ARTIFACT_FEATURE] {
            return Err(CounterfactualIdArtifactError::UnsupportedSemantics("required features"));
        }
        Ok(wire)
    }

    /// Recompute both digests and compare them with the stored ones (and the
    /// data digest with the caller's, when given).
    ///
    /// # Errors
    /// [`CounterfactualIdArtifactError::PremisesMismatch`] or `DataIdentityMismatch`.
    pub fn check_digests(
        &self,
        expected_data_digest: Option<&str>,
    ) -> Result<(), CounterfactualIdArtifactError> {
        let data = counterfactual_id_data_digest(&self.law)
            .map_err(|e| CounterfactualIdArtifactError::Malformed(e.to_string()))?;
        if data != self.data_digest || expected_data_digest.is_some_and(|e| e != data) {
            return Err(CounterfactualIdArtifactError::DataIdentityMismatch);
        }
        let premises = counterfactual_id_identity(self)
            .map_err(|e| CounterfactualIdArtifactError::Malformed(e.to_string()))?;
        if premises != self.premises_digest {
            return Err(CounterfactualIdArtifactError::PremisesMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CounterfactualIdArtifactWire {
        let query = CounterfactualEventQuery::effect_on_treated(
            VariableId::from_raw(0),
            1.0,
            0.0,
            VariableId::from_raw(1),
            1.0,
        )
        .unwrap();
        CounterfactualIdArtifactWire {
            version: COUNTERFACTUAL_ID_ARTIFACT_VERSION,
            required_features: vec![COUNTERFACTUAL_ID_ARTIFACT_FEATURE.into()],
            variable_names: vec!["x".into(), "y".into()],
            levels: vec![vec![0, 1.0f64.to_bits()]; 2],
            directed: vec![(0, 1)],
            bidirected: vec![],
            query: CounterfactualIdQueryWire::from_query(&query),
            estimand: COUNTERFACTUAL_ID_ESTIMAND.into(),
            contract: "effect_on_treated_admg_v1".into(),
            search: CounterfactualIdSearchRecord {
                operations_limit: 10,
                depth_limit: 5,
                memory_limit_bytes: 1024,
                operations_consumed: 3,
                depth_reached: 2,
            },
            derivation: "d".into(),
            counterfactual_graph: None,
            law: CounterfactualIdLawWire {
                axes: vec![
                    CounterfactualIdAxisWire { variable: 0, levels: vec![0, 1.0f64.to_bits()] },
                    CounterfactualIdAxisWire { variable: 1, levels: vec![0, 1.0f64.to_bits()] },
                ],
                probabilities: vec![0.25f64.to_bits(); 4],
                counts: None,
                origin: "supplied_exact".into(),
                population: "p".into(),
                snapshot: "s".into(),
            },
            point: CounterfactualIdPointWire {
                probability: 0,
                conditioning_probability: 0,
                numerators: vec![],
                counterfactual_mean: None,
                observed_mean: None,
                effect: None,
            },
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
        .unwrap()
    }

    fn with_extra_field(value: serde_json::Value, path: &[&str]) -> serde_json::Value {
        let mut value = value;
        let mut slot = &mut value;
        for key in path {
            slot = if let Ok(i) = key.parse::<usize>() { &mut slot[i] } else { &mut slot[*key] };
        }
        slot.as_object_mut().expect("an object").insert("unexpected".into(), 1.into());
        value
    }

    #[test]
    fn every_wire_struct_rejects_unknown_fields() {
        let wire = sample();
        let json = serde_json::to_value(&wire).unwrap();
        // The artifact itself, the query, a world, the law, an axis, the point,
        // the search record: each refuses an unknown field.
        for path in [
            &[][..],
            &["query"][..],
            &["query", "worlds", "0"][..],
            &["law"][..],
            &["law", "axes", "0"][..],
            &["point"][..],
            &["search"][..],
        ] {
            let tampered = with_extra_field(json.clone(), path);
            assert!(
                serde_json::from_value::<CounterfactualIdArtifactWire>(tampered).is_err(),
                "{path:?} accepted an unknown field"
            );
        }
        // The counterfactual graph record and its nodes too.
        let graph = serde_json::json!({
            "nodes": [{"variable": 0, "value": null, "subscript": [], "parents": [], "unexpected": 1}],
            "bidirected": [], "districts": [[0]]
        });
        assert!(serde_json::from_value::<CounterfactualGraphRecord>(graph).is_err());
        let graph = serde_json::json!({
            "nodes": [], "bidirected": [], "districts": [], "unexpected": 1
        });
        assert!(serde_json::from_value::<CounterfactualGraphRecord>(graph).is_err());
        // Round trip of the untampered value.
        let back: CounterfactualIdArtifactWire = serde_json::from_value(json).unwrap();
        assert_eq!(back.premises_digest, wire.premises_digest);
    }

    #[test]
    fn digests_bind_premises_and_data_separately() {
        let wire = sample();
        wire.check_digests(None).unwrap();
        wire.check_digests(Some(&wire.data_digest.clone())).unwrap();
        assert_eq!(
            wire.check_digests(Some("other")),
            Err(CounterfactualIdArtifactError::DataIdentityMismatch)
        );
        let mut law = wire.clone();
        law.law.probabilities[0] = 0.5f64.to_bits();
        assert_eq!(
            law.check_digests(None),
            Err(CounterfactualIdArtifactError::DataIdentityMismatch)
        );
        let mut premises = wire.clone();
        premises.search.depth_limit += 1;
        assert_eq!(
            premises.check_digests(None),
            Err(CounterfactualIdArtifactError::PremisesMismatch)
        );
        // Outputs are not premises: the derivation text is not digested.
        let mut output = wire;
        output.derivation.push('x');
        output.check_digests(None).unwrap();
        assert_eq!(
            CounterfactualIdArtifactError::LimitsExceeded("x").reason(),
            ("route_not_supported", "counterfactual_id.bounds_exceeded")
        );
        assert_eq!(
            CounterfactualIdArtifactError::PointMismatch.reason(),
            ("invalid_argument", "counterfactual_id.invalid_artifact")
        );
    }
}
