//! Portable, estimator-free wire of the general finite counterfactual
//! transport artifact (kind `counterfactual_transport_v1`).
//!
//! The artifact keeps every premise a consumer needs to decide the request
//! again: the model class (variables with their declared levels, directed and
//! bidirected edges), the counterfactual query (worlds with their interventions
//! and edge routes, event and conditioning conjunctions, target population,
//! sources with their selection targets and controllable variables), the
//! complete evidence catalog and, for an identified query, every exact law
//! table the point was computed from. It also keeps what was concluded: the
//! canonical derivation with its search receipt and the point with the identity
//! of every law it read, or, for a proven obstruction, the two-model witness.
//!
//! Two digests, two typed errors. The *premises digest* binds the model class,
//! query, variable names, search limits and memory cap, evaluation limits, and
//! the conclusion (derivation text and receipt, or witness). The *data digest*
//! binds the evidence catalog and every law table (axes, probability bits,
//! population, regime and snapshot identity). A fresh consumer holding only the
//! bytes and an expected identity (one or both digests, obtained independently
//! of the bytes) re-runs identification under the stored limits (refusing
//! limits above its own before any work), requires the identical derivation,
//! re-evaluates the point and compares it bit for bit, or, for an obstruction,
//! re-verifies the witness by exact enumeration.
//!
//! What replay does not protect against: a producer that seals a wrong model
//! class, query or law on purpose (the digests bind what was sealed, not what is
//! true), and an error in the identification theory shared by producer and
//! consumer (the same decision procedure runs twice; the independent finite-SCM
//! oracle lives in the test suites). The artifact is point-only: no interval, no
//! posterior and no sampling uncertainty exist in it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{IdentityDomain, VariableId};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_CONTRACT, COUNTERFACTUAL_TRANSPORT_MAX_ATOMS,
    COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES, COUNTERFACTUAL_TRANSPORT_MAX_LIMITS,
    COUNTERFACTUAL_TRANSPORT_MAX_REGIMES, COUNTERFACTUAL_TRANSPORT_MAX_SOURCES,
    COUNTERFACTUAL_TRANSPORT_MAX_WORLDS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES, CounterfactualTransportDerivation,
    CounterfactualTransportProblem, CounterfactualTransportQuery,
    CounterfactualTransportSearchRecord, CounterfactualTransportSource,
    CounterfactualTransportWitness,
};
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::counterfactual_id_artifact::CounterfactualIdQueryWire;
use crate::exact_law_wire::ExactLawWire;
use crate::transport_catalog_wire::EvidenceCatalogWire;

/// The artifact format this reader writes and accepts.
pub const COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE: &str =
    crate::artifact_kinds::COUNTERFACTUAL_TRANSPORT;
/// Conclusion tag: the query is identified and the artifact carries a point.
pub const COUNTERFACTUAL_TRANSPORT_IDENTIFIED: &str = "identified";
/// Conclusion tag: the query is proven not transportable and the artifact
/// carries the two-model witness.
pub const COUNTERFACTUAL_TRANSPORT_OBSTRUCTED: &str = "proven_non_transportable";

/// Detail of an artifact that is not a replayable counterfactual transport
/// artifact (malformed, resealed with a different conclusion, or a digest
/// mismatch).
const DETAIL_INVALID_ARTIFACT: &str = "counterfactual_transport.invalid_artifact";
/// Detail of a stored bound above the consumer's or the contract's.
const DETAIL_BOUNDS_EXCEEDED: &str = "counterfactual_transport.bounds_exceeded";
/// Detail of a stored derivation the replay does not reproduce.
const DETAIL_INVALID_DERIVATION: &str = "counterfactual_transport.invalid_derivation";
/// Detail of a stored witness that does not verify.
const DETAIL_INVALID_WITNESS: &str = "counterfactual_transport.invalid_witness";
/// Detail of a stored law that is not a valid exact law.
const DETAIL_INVALID_EVIDENCE: &str = "counterfactual_transport.invalid_evidence";

/// Why a counterfactual transport artifact was refused. Every variant names the
/// slot of the artifact it concerns ([`Self::slot`]) and carries the registered
/// refusal it surfaces as ([`Self::refusal`]).
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CounterfactualTransportArtifactError {
    /// The bytes do not decode as this format.
    #[error("counterfactual transport artifact does not decode: {0}")]
    Decode(String),
    /// The artifact could not be encoded.
    #[error("counterfactual transport artifact does not encode: {0}")]
    Encode(String),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A required feature or conclusion tag is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection or limit exceeds the consumer's (or the contract's)
    /// bound; refused before any work. Retry with larger limits: this is not a
    /// claim that the artifact is invalid.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// A stored slot is not a well-formed premise.
    #[error("malformed {slot}: {message}")]
    Malformed {
        /// The artifact slot.
        slot: &'static str,
        /// What is wrong.
        message: String,
    },
    /// The premises digest does not match the stored premises and conclusion.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The evidence does not match its data digest.
    #[error("data identity mismatch")]
    DataIdentityMismatch,
    /// The artifact is self-consistent but is not the one the caller expected.
    #[error("the artifact's {slot} identity is not the expected one")]
    ExpectedIdentityMismatch {
        /// `premises` or `data`.
        slot: &'static str,
    },
    /// The re-decision refused, or differs from the stored derivation, search
    /// receipt or conclusion class.
    #[error("derivation does not replay: {0}")]
    DerivationMismatch(String),
    /// The stored witness does not verify, or differs from what its
    /// verification establishes.
    #[error("obstruction does not verify: {0}")]
    ObstructionInvalid(String),
    /// A stored law is not a valid exact law.
    #[error("stored law is invalid: {0}")]
    LawInvalid(String),
    /// A re-evaluated field differs from the stored result.
    #[error("{field} does not replay")]
    ResultMismatch {
        /// The result field that differs.
        field: &'static str,
    },
    /// The replayed decision or evaluation refused (budget, cancellation,
    /// support failure, evidence the catalog does not bind).
    #[error("replay refused: {message}")]
    Replay {
        /// Registered reason code of the refusal.
        code: &'static str,
        /// `<counterfactual_transport.detail>: <message>` of the refusal.
        message: String,
    },
    /// The caller's variable names differ from the verified mapping.
    #[error("variable names differ from the verified mapping")]
    NamesMismatch,
}

impl From<IoError> for CounterfactualTransportArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

impl CounterfactualTransportArtifactError {
    /// The artifact slot this failure concerns: `artifact`, `version`,
    /// `required_features`, `status`, `limits`, `problem`, `query`, `catalog`,
    /// `laws`, `variable_names`, `derivation`, `obstruction`, `result`,
    /// `premises_digest`, `data_digest`, `expected_identity` or `replay`.
    #[must_use]
    pub const fn slot(&self) -> &'static str {
        match self {
            Self::Decode(_) | Self::Encode(_) => "artifact",
            Self::UnsupportedVersion { .. } => "version",
            Self::UnsupportedSemantics(slot) | Self::Malformed { slot, .. } => slot,
            Self::LimitsExceeded(_) => "limits",
            Self::PremisesMismatch => "premises_digest",
            Self::DataIdentityMismatch => "data_digest",
            Self::ExpectedIdentityMismatch { .. } => "expected_identity",
            Self::DerivationMismatch(_) => "derivation",
            Self::ObstructionInvalid(_) => "obstruction",
            Self::LawInvalid(_) => "laws",
            Self::ResultMismatch { .. } => "result",
            Self::Replay { .. } => "replay",
            Self::NamesMismatch => "variable_names",
        }
    }

    /// The registered reason code and the message (`<detail>: <slot>: <what>`)
    /// this failure surfaces as: a bound is `route_not_supported`, a derivation
    /// or witness that does not replay is `transport_not_certified`, a law that
    /// is not a valid exact law is `invalid_argument`, a replayed refusal keeps
    /// its own code, and every other artifact defect is `invalid_argument`.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, String) {
        use antecedent_core::reason_code;
        let slot = self.slot();
        match self {
            Self::LimitsExceeded(_) => (
                reason_code!("route_not_supported"),
                format!("{DETAIL_BOUNDS_EXCEEDED}: {slot}: {self}"),
            ),
            Self::DerivationMismatch(_) => (
                reason_code!("transport_not_certified"),
                format!("{DETAIL_INVALID_DERIVATION}: {slot}: {self}"),
            ),
            Self::ObstructionInvalid(_) => (
                reason_code!("transport_not_certified"),
                format!("{DETAIL_INVALID_WITNESS}: {slot}: {self}"),
            ),
            Self::LawInvalid(_) => (
                reason_code!("invalid_argument"),
                format!("{DETAIL_INVALID_EVIDENCE}: {slot}: {self}"),
            ),
            Self::Replay { code, message } => (*code, message.clone()),
            _ => (
                reason_code!("invalid_argument"),
                format!("{DETAIL_INVALID_ARTIFACT}: {slot}: {self}"),
            ),
        }
    }
}

fn malformed(
    slot: &'static str,
    message: impl Into<String>,
) -> CounterfactualTransportArtifactError {
    CounterfactualTransportArtifactError::Malformed { slot, message: message.into() }
}

// ---------------------------------------------------------------------------
// The model class.
// ---------------------------------------------------------------------------

/// The model class on the wire: variables ascending by id with their declared
/// levels as IEEE bits (declared order), directed edges `(parent, child)` and
/// bidirected edges `(low, high)`, each sorted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportProblemWire {
    /// `(variable, level bits)`, ascending by variable.
    pub variables: Vec<(u32, Vec<u64>)>,
    /// Directed edges, sorted.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges, sorted.
    pub bidirected: Vec<(u32, u32)>,
}

impl CounterfactualTransportProblemWire {
    /// Encode a model class.
    #[must_use]
    pub fn from_problem(problem: &CounterfactualTransportProblem) -> Self {
        let ids = problem.variables();
        Self {
            variables: ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    (id.raw(), problem.levels(index).iter().map(|x| (x + 0.0).to_bits()).collect())
                })
                .collect(),
            directed: problem
                .directed()
                .iter()
                .map(|&(a, b)| (ids[a].raw(), ids[b].raw()))
                .collect(),
            bidirected: problem
                .bidirected()
                .iter()
                .map(|&(a, b)| (ids[a].raw(), ids[b].raw()))
                .collect(),
        }
    }

    /// Decode and validate a model class. A canonical class re-encodes to
    /// itself: an unsorted or duplicated list is malformed, not repaired.
    ///
    /// # Errors
    /// `Malformed` for a class the contract refuses or a non-canonical encoding.
    pub fn to_problem(
        &self,
    ) -> Result<CounterfactualTransportProblem, CounterfactualTransportArtifactError> {
        let variables: Vec<(VariableId, Vec<f64>)> = self
            .variables
            .iter()
            .map(|(id, bits)| {
                (VariableId::from_raw(*id), bits.iter().map(|b| f64::from_bits(*b)).collect())
            })
            .collect();
        let edge = |&(a, b): &(u32, u32)| (VariableId::from_raw(a), VariableId::from_raw(b));
        let directed: Vec<(VariableId, VariableId)> = self.directed.iter().map(edge).collect();
        let bidirected: Vec<(VariableId, VariableId)> = self.bidirected.iter().map(edge).collect();
        let problem = CounterfactualTransportProblem::new(&variables, &directed, &bidirected)
            .map_err(|error| malformed("problem", error.to_string()))?;
        if Self::from_problem(&problem) != *self {
            return Err(malformed("problem", "the stored model class is not canonical"));
        }
        Ok(problem)
    }
}

// ---------------------------------------------------------------------------
// The query.
// ---------------------------------------------------------------------------

/// One source population of a stored query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportSourceWire {
    /// Population identity.
    pub population: String,
    /// Variables whose mechanism may differ between this source and the target
    /// (the selection diagram's targets), sorted.
    pub selection_targets: Vec<u32>,
    /// Variables this source can experiment on, sorted.
    pub controllable: Vec<u32>,
}

/// A counterfactual transport query on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportQueryWire {
    /// Worlds, edge routes, event and conditioning conjunctions.
    pub event: CounterfactualIdQueryWire,
    /// Target population.
    pub target: String,
    /// Sources, ordered by population.
    pub sources: Vec<CounterfactualTransportSourceWire>,
}

impl CounterfactualTransportQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &CounterfactualTransportQuery) -> Self {
        let query = query.canonical();
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            event: CounterfactualIdQueryWire::from_query(&query.event),
            target: query.target.to_string(),
            sources: query
                .sources
                .iter()
                .map(|s| CounterfactualTransportSourceWire {
                    population: s.population.to_string(),
                    selection_targets: raw(&s.selection_targets),
                    controllable: raw(&s.controllable),
                })
                .collect(),
        }
    }

    /// Decode and validate a query. A canonical query re-encodes to itself.
    ///
    /// # Errors
    /// `Malformed` for a query that does not validate or is not canonical.
    pub fn to_query(
        &self,
    ) -> Result<CounterfactualTransportQuery, CounterfactualTransportArtifactError> {
        let event = self.event.to_query().map_err(|error| malformed("query", error.to_string()))?;
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let query = CounterfactualTransportQuery {
            event,
            target: Arc::from(self.target.as_str()),
            sources: self
                .sources
                .iter()
                .map(|s| CounterfactualTransportSource {
                    population: Arc::from(s.population.as_str()),
                    selection_targets: ids(&s.selection_targets).into(),
                    controllable: ids(&s.controllable).into(),
                })
                .collect::<Vec<_>>()
                .into(),
        };
        if Self::from_query(&query) != *self {
            return Err(malformed("query", "the stored query is not canonical"));
        }
        Ok(query)
    }
}

// ---------------------------------------------------------------------------
// Conclusions.
// ---------------------------------------------------------------------------

/// Search limits a decision ran under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportSearchLimitsWire {
    /// Operation limit.
    pub operations: usize,
    /// Depth limit.
    pub depth: usize,
    /// Effective memory cap in bytes.
    pub memory_limit_bytes: u64,
}

/// A cited catalog regime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportCitedRegimeWire {
    /// Regime id.
    pub regime: u32,
    /// Population of the regime.
    pub population: String,
    /// Intervened variables.
    pub interventions: Vec<u32>,
}

/// A stored derivation: the canonical text is the replayed identity of the
/// whole derivation (items, functionals, reduction, rule trace and receipt).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportDerivationWire {
    /// The attested contract.
    pub contract: String,
    /// Canonical text of the derivation.
    pub canonical_text: String,
    /// Search limits and accounting of the decision.
    pub search: CounterfactualTransportSearchRecord,
    /// Whether the original conditioning mass is checked at evaluation.
    pub conditioning_mass_checked: bool,
    /// Catalog regimes the formula cites, ascending.
    pub cited: Vec<CounterfactualTransportCitedRegimeWire>,
}

impl CounterfactualTransportDerivationWire {
    /// Encode a derivation.
    #[must_use]
    pub fn from_derivation(derivation: &CounterfactualTransportDerivation) -> Self {
        Self {
            contract: derivation.contract.to_owned(),
            canonical_text: derivation.canonical_text(),
            search: derivation.search.clone(),
            conditioning_mass_checked: derivation.conditioning_mass_checked,
            cited: derivation
                .cited
                .iter()
                .map(|c| CounterfactualTransportCitedRegimeWire {
                    regime: c.regime.raw(),
                    population: c.population.to_string(),
                    interventions: c.interventions.iter().map(|v| v.raw()).collect(),
                })
                .collect(),
        }
    }
}

/// A stored proven obstruction: the portable two-model witness and what its
/// verification established.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportObstructionWire {
    /// The two-model witness.
    pub witness: CounterfactualTransportWitness,
    /// Regime-law cells compared by the verification.
    pub cells_compared: u64,
    /// Query value of the first model, exact `"n/d"`.
    pub first_value: String,
    /// Query value of the second model, exact `"n/d"`.
    pub second_value: String,
    /// Why no derivation exists (descriptive; not re-derived by a consumer).
    pub reason: String,
}

/// One law the evaluation read, identified by its population, regime,
/// intervention world, snapshot and content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportCitedLawWire {
    /// Population of the law.
    pub population: String,
    /// Catalog regime of the law.
    pub regime: u32,
    /// `(intervened variable, level bits)` of the law's intervention world.
    pub interventions: Vec<(u32, u64)>,
    /// Physical snapshot identity the provider supplied.
    pub snapshot_identity: String,
    /// Content identity (hex digest) of the table.
    pub content_identity: String,
}

/// The stored point, as IEEE bits, with the laws it read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportResultWire {
    /// `P*(event and conditioning)` of the (possibly reduced) query.
    pub numerator: u64,
    /// `P*(conditioning)` of the (possibly reduced) query.
    pub denominator: u64,
    /// The requested probability.
    pub probability: u64,
    /// The original conditioning event's mass, when derived.
    pub conditioning_mass: Option<u64>,
    /// Whether the original conditioning event was checked to have positive mass.
    pub conditioning_mass_checked: bool,
    /// Regime-law cells read.
    pub cells_read: u64,
    /// Every law that was read, in the evaluation's canonical order.
    pub cited_laws: Vec<CounterfactualTransportCitedLawWire>,
}

// ---------------------------------------------------------------------------
// The artifact.
// ---------------------------------------------------------------------------

/// The identity a consumer holds independently of the bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CounterfactualTransportExpectedIdentity {
    /// Expected premises digest, when held.
    pub premises_digest: Option<String>,
    /// Expected data digest, when held.
    pub data_digest: Option<String>,
}

/// Versioned counterfactual transport artifact with every premise needed for
/// replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// [`COUNTERFACTUAL_TRANSPORT_IDENTIFIED`] or
    /// [`COUNTERFACTUAL_TRANSPORT_OBSTRUCTED`].
    pub status: String,
    /// The model class.
    pub problem: CounterfactualTransportProblemWire,
    /// The query with its selection diagram and controllable sets.
    pub query: CounterfactualTransportQueryWire,
    /// Name of every variable in ascending id order, or empty when none were named.
    pub variable_names: Vec<String>,
    /// Search limits and memory cap the decision ran under.
    pub search: CounterfactualTransportSearchLimitsWire,
    /// The complete evidence catalog.
    pub catalog: EvidenceCatalogWire,
    /// The derivation, when identified.
    pub derivation: Option<CounterfactualTransportDerivationWire>,
    /// The obstruction, when proven non-transportable.
    pub obstruction: Option<CounterfactualTransportObstructionWire>,
    /// Exact law tables in canonical order (empty for an obstruction).
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under.
    pub max_support_rows: usize,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// The point, when identified.
    pub result: Option<CounterfactualTransportResultWire>,
    /// Digest of the premises and conclusion ([`Self::expected_premises_digest`]).
    pub premises_digest: String,
    /// Digest of the evidence ([`Self::expected_data_digest`]).
    pub data_digest: String,
}

/// Every CBOR-encoded element of `items`, sorted: an order-independent view.
fn sorted_cbor<T: Serialize>(items: &[T]) -> Result<Vec<Vec<u8>>, IoError> {
    let mut out = items.iter().map(crate::to_cbor).collect::<Result<Vec<_>, _>>()?;
    out.sort();
    Ok(out)
}

/// Digest of the evidence: the whole catalog (environments, regimes and
/// bindings each sorted, and the target sampling contract) and every law table,
/// sorted. A refresh with changed numbers changes it.
///
/// # Errors
/// Canonical encoding failure.
// doc(hidden): the producer and the consumer both seal and check with it.
#[doc(hidden)]
pub fn counterfactual_transport_data_digest(
    catalog: &EvidenceCatalogWire,
    laws: &[ExactLawWire],
) -> Result<String, IoError> {
    let catalog_part = (
        sorted_cbor(&catalog.environments)?,
        sorted_cbor(&catalog.regimes)?,
        sorted_cbor(&catalog.bindings)?,
        &catalog.target_sampling,
    );
    crate::digest_hex(
        IdentityDomain::TransportCertificate,
        &("counterfactual_transport_data_v1", catalog_part, sorted_cbor(laws)?),
    )
}

/// Everything the premises digest binds, in canonical form.
#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    status: &'a str,
    problem: &'a CounterfactualTransportProblemWire,
    query: &'a CounterfactualTransportQueryWire,
    variable_names: &'a [String],
    search: &'a CounterfactualTransportSearchLimitsWire,
    evaluation: (usize, usize, usize),
    derivation: &'a Option<CounterfactualTransportDerivationWire>,
    obstruction: &'a Option<CounterfactualTransportObstructionWire>,
}

/// Laws in the canonical order: by CBOR of `(population, regime, interventions)`.
fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions))
                .map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

impl CounterfactualTransportArtifactWire {
    /// The wire of an identified query: the derivation, the exact laws the point
    /// was computed on and the stored point, sealed with both digests.
    ///
    /// # Errors
    /// A bound exceeded, laws that do not encode, or names that do not cover the
    /// model class.
    pub fn identified(
        derivation: &CounterfactualTransportDerivation,
        data: &ExactTransportData,
        limits: ExactEvaluationLimits,
        variable_names: &[String],
        result: CounterfactualTransportResultWire,
    ) -> Result<Self, CounterfactualTransportArtifactError> {
        let laws = canonical_laws(data)?;
        Self {
            version: COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE.into()],
            status: COUNTERFACTUAL_TRANSPORT_IDENTIFIED.into(),
            problem: CounterfactualTransportProblemWire::from_problem(&derivation.problem),
            query: CounterfactualTransportQueryWire::from_query(&derivation.query),
            variable_names: variable_names.to_vec(),
            search: CounterfactualTransportSearchLimitsWire {
                operations: derivation.search.operations_limit,
                depth: derivation.search.depth_limit,
                memory_limit_bytes: derivation.search.memory_limit_bytes,
            },
            catalog: EvidenceCatalogWire::from_catalog(&derivation.catalog),
            derivation: Some(CounterfactualTransportDerivationWire::from_derivation(derivation)),
            obstruction: None,
            laws,
            max_support_rows: data.max_support_rows(),
            operation_limit: limits.operations,
            depth_limit: limits.depth,
            result: Some(result),
            premises_digest: String::new(),
            data_digest: String::new(),
        }
        .sealed()
    }

    /// The wire of a proven obstruction: the model class, query, catalog and
    /// two-model witness, sealed with both digests. No law table is carried: a
    /// witness proves the declared evidence family does not determine the query
    /// whatever the laws are.
    ///
    /// # Errors
    /// A bound exceeded or names that do not cover the model class.
    pub fn obstructed(
        problem: &CounterfactualTransportProblem,
        query: &CounterfactualTransportQuery,
        catalog: &antecedent_core::EvidenceCatalog,
        search: CounterfactualTransportSearchLimitsWire,
        variable_names: &[String],
        obstruction: CounterfactualTransportObstructionWire,
    ) -> Result<Self, CounterfactualTransportArtifactError> {
        let defaults = ExactEvaluationLimits::default();
        Self {
            version: COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE.into()],
            status: COUNTERFACTUAL_TRANSPORT_OBSTRUCTED.into(),
            problem: CounterfactualTransportProblemWire::from_problem(problem),
            query: CounterfactualTransportQueryWire::from_query(query),
            variable_names: variable_names.to_vec(),
            search,
            catalog: EvidenceCatalogWire::from_catalog(catalog),
            derivation: None,
            obstruction: Some(obstruction),
            laws: Vec::new(),
            max_support_rows: 0,
            operation_limit: defaults.operations,
            depth_limit: defaults.depth,
            result: None,
            premises_digest: String::new(),
            data_digest: String::new(),
        }
        .sealed()
    }

    fn premises_view(&self) -> PremisesView<'_> {
        PremisesView {
            tag: "counterfactual_transport_v1",
            status: &self.status,
            problem: &self.problem,
            query: &self.query,
            variable_names: &self.variable_names,
            search: &self.search,
            evaluation: (self.operation_limit, self.depth_limit, self.max_support_rows),
            derivation: &self.derivation,
            obstruction: &self.obstruction,
        }
    }

    /// The premises digest the stored premises should carry. Recomputing it
    /// grants nothing: a consumer still re-decides and re-evaluates.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        crate::digest_hex(IdentityDomain::TransportCertificate, &self.premises_view())
    }

    /// The data digest the stored catalog and laws should carry.
    ///
    /// # Errors
    /// The catalog or laws do not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        counterfactual_transport_data_digest(&self.catalog, &self.laws)
    }

    /// Check the shape the format guarantees and the contract's bounds: both
    /// producer and consumer call it.
    ///
    /// # Errors
    /// `UnsupportedSemantics` for another feature or conclusion tag,
    /// `LimitsExceeded` above a contract bound, `Malformed` otherwise.
    pub fn validate_shape(&self) -> Result<(), CounterfactualTransportArtifactError> {
        use CounterfactualTransportArtifactError as E;
        if self.required_features != [COUNTERFACTUAL_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(E::UnsupportedSemantics("required_features"));
        }
        let identified = match self.status.as_str() {
            COUNTERFACTUAL_TRANSPORT_IDENTIFIED => true,
            COUNTERFACTUAL_TRANSPORT_OBSTRUCTED => false,
            _ => return Err(E::UnsupportedSemantics("status")),
        };
        let n = self.problem.variables.len();
        if n == 0 || n > COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES {
            return Err(E::LimitsExceeded("variables"));
        }
        if self.query.sources.len() > COUNTERFACTUAL_TRANSPORT_MAX_SOURCES {
            return Err(E::LimitsExceeded("sources"));
        }
        if self.query.event.worlds.len() > COUNTERFACTUAL_TRANSPORT_MAX_WORLDS {
            return Err(E::LimitsExceeded("worlds"));
        }
        if self.query.event.event.len() + self.query.event.given.len()
            > COUNTERFACTUAL_TRANSPORT_MAX_ATOMS
        {
            return Err(E::LimitsExceeded("atoms"));
        }
        if self.catalog.regimes.len() > COUNTERFACTUAL_TRANSPORT_MAX_REGIMES {
            return Err(E::LimitsExceeded("regimes"));
        }
        let maxima = COUNTERFACTUAL_TRANSPORT_MAX_LIMITS;
        if self.search.operations > maxima.operations
            || self.search.depth > maxima.depth
            || self.search.memory_limit_bytes > COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES
        {
            return Err(E::LimitsExceeded("search limits"));
        }
        if !self.variable_names.is_empty()
            && (self.variable_names.len() != n
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len() != n)
        {
            return Err(malformed(
                "variable_names",
                "names must be empty or one distinct, non-blank name per variable",
            ));
        }
        if identified {
            let (Some(derivation), None, Some(result)) =
                (&self.derivation, &self.obstruction, &self.result)
            else {
                return Err(malformed(
                    "status",
                    "an identified artifact carries a derivation and a result and no obstruction",
                ));
            };
            if derivation.contract != COUNTERFACTUAL_TRANSPORT_CONTRACT {
                return Err(E::UnsupportedSemantics("contract"));
            }
            if (derivation.search.operations_limit, derivation.search.depth_limit)
                != (self.search.operations, self.search.depth)
                || derivation.search.memory_limit_bytes != self.search.memory_limit_bytes
            {
                return Err(malformed(
                    "derivation",
                    "the stored search limits differ from the derivation's search receipt",
                ));
            }
            if result.cited_laws.len() > self.laws.len() {
                return Err(malformed("result", "more cited laws than stored laws"));
            }
            if derivation.conditioning_mass_checked != result.conditioning_mass_checked {
                return Err(malformed(
                    "result",
                    "the stored conditioning-mass check differs from the derivation's",
                ));
            }
        } else {
            let (None, Some(_), None) = (&self.derivation, &self.obstruction, &self.result) else {
                return Err(malformed(
                    "status",
                    "an obstruction artifact carries an obstruction and no derivation or result",
                ));
            };
            if !self.laws.is_empty() {
                return Err(malformed("laws", "an obstruction artifact carries no law table"));
            }
        }
        Ok(())
    }

    /// Seal: check the shape and bounds, then fill both digests.
    ///
    /// # Errors
    /// A shape or bound failure, or premises that do not encode.
    pub fn sealed(mut self) -> Result<Self, CounterfactualTransportArtifactError> {
        self.validate_shape()?;
        self.data_digest = self.expected_data_digest()?;
        self.premises_digest = self.expected_premises_digest()?;
        Ok(self)
    }

    /// The identity a consumer can hold: both digests.
    #[must_use]
    pub fn identity(&self) -> CounterfactualTransportExpectedIdentity {
        CounterfactualTransportExpectedIdentity {
            premises_digest: Some(self.premises_digest.clone()),
            data_digest: Some(self.data_digest.clone()),
        }
    }

    /// Encode as CBOR, refusing an artifact outside the format's shape.
    ///
    /// # Errors
    /// A shape or bound failure, or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, CounterfactualTransportArtifactError> {
        self.validate_shape()?;
        crate::to_cbor(self)
            .map_err(|e| CounterfactualTransportArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version before the payload is interpreted,
    /// then any other feature or conclusion tag.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature, or a shape
    /// failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, CounterfactualTransportArtifactError> {
        let wire: Self = crate::decode_with_version(bytes, |version: u32| {
            if version == COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(CounterfactualTransportArtifactError::UnsupportedVersion { version })
            }
        })?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Recompute both digests and compare them with the stored ones, then with
    /// the identity the caller holds independently of the bytes.
    ///
    /// # Errors
    /// `PremisesMismatch`, `DataIdentityMismatch` or `ExpectedIdentityMismatch`.
    pub fn check_digests(
        &self,
        expected: &CounterfactualTransportExpectedIdentity,
    ) -> Result<(), CounterfactualTransportArtifactError> {
        use CounterfactualTransportArtifactError as E;
        let premises = self
            .expected_premises_digest()
            .map_err(|e| malformed("premises_digest", e.to_string()))?;
        if premises != self.premises_digest {
            return Err(E::PremisesMismatch);
        }
        let data =
            self.expected_data_digest().map_err(|e| malformed("data_digest", e.to_string()))?;
        if data != self.data_digest {
            return Err(E::DataIdentityMismatch);
        }
        if expected.premises_digest.as_deref().is_some_and(|e| e != premises) {
            return Err(E::ExpectedIdentityMismatch { slot: "premises" });
        }
        if expected.data_digest.as_deref().is_some_and(|e| e != data) {
            return Err(E::ExpectedIdentityMismatch { slot: "data" });
        }
        Ok(())
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// `NamesMismatch` unless `names` equals the stored mapping.
    pub fn check_variable_names(
        &self,
        names: &[String],
    ) -> Result<(), CounterfactualTransportArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(CounterfactualTransportArtifactError::NamesMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CounterfactualEvent, CounterfactualEventQuery, ExogenousCoupling, WorldId, WorldSpec,
    };

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn problem() -> CounterfactualTransportProblem {
        CounterfactualTransportProblem::new(
            &[(v(0), vec![0.0, 1.0]), (v(1), vec![0.0, 1.0])],
            &[(v(0), v(1))],
            &[],
        )
        .unwrap()
    }

    fn query() -> CounterfactualTransportQuery {
        let event = CounterfactualEventQuery::new(
            vec![WorldSpec::new([(v(0), 1.0)], []).unwrap()],
            ExogenousCoupling::SharedLatentExogenous,
            [CounterfactualEvent::new(WorldId::new(0), v(1), 1.0).unwrap()],
            [],
        )
        .unwrap();
        CounterfactualTransportQuery {
            event,
            target: Arc::from("target"),
            sources: vec![CounterfactualTransportSource {
                population: Arc::from("source"),
                selection_targets: vec![v(1), v(0)].into(),
                controllable: vec![v(0)].into(),
            }]
            .into(),
        }
    }

    #[test]
    fn problem_and_query_round_trip_canonically() {
        let wire = CounterfactualTransportProblemWire::from_problem(&problem());
        assert_eq!(wire.to_problem().unwrap(), problem());
        let query_wire = CounterfactualTransportQueryWire::from_query(&query());
        // Selection targets are canonical (sorted) on the wire.
        assert_eq!(query_wire.sources[0].selection_targets, vec![0, 1]);
        assert_eq!(query_wire.to_query().unwrap(), query().canonical());
        // A non-canonical encoding is malformed, not repaired.
        let mut unsorted = query_wire;
        unsorted.sources[0].selection_targets = vec![1, 0];
        let error = unsorted.to_query().unwrap_err();
        assert_eq!(error.slot(), "query");
        let mut reversed = wire;
        reversed.variables.reverse();
        assert_eq!(reversed.to_problem().unwrap_err().slot(), "problem");
    }

    #[test]
    fn refusal_pairs_follow_the_slot_taxonomy() {
        use CounterfactualTransportArtifactError as E;
        let (code, message) = E::LimitsExceeded("search operation limit").refusal();
        assert_eq!(code, "route_not_supported");
        assert!(message.starts_with("counterfactual_transport.bounds_exceeded: limits:"));
        let (code, message) = E::DerivationMismatch("x".into()).refusal();
        assert_eq!(code, "transport_not_certified");
        assert!(message.starts_with("counterfactual_transport.invalid_derivation: derivation:"));
        let (code, message) = E::ResultMismatch { field: "probability" }.refusal();
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with("counterfactual_transport.invalid_artifact: result:"));
        assert_eq!(E::PremisesMismatch.slot(), "premises_digest");
        assert_eq!(E::UnsupportedSemantics("status").slot(), "status");
    }
}
