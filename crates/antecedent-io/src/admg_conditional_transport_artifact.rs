//! Independent point-result artifacts for ADMG conditional transport (2.2B B1).
//!
//! Format version 1 (`checked_admg_conditional_point_v1`). A consumer trusts
//! nothing in the artifact. Before any work (before either digest is hashed) it
//! refuses stored limits above its own maxima and a graph or query above the
//! route's size bounds. It then checks the premises digest (graph, selections, query,
//! proof, expression arena, search and evaluation limits, requests, variable
//! names) and the separate data-identity digest (the whole catalog and every
//! law's snapshot), re-checks the proof with the independent checkers under the
//! PRODUCER's stored search limits (every rule-2 move, the maximality of the
//! moved set, the reduced joint's sID proof), re-decides the query under those
//! same limits and requires the identical record, re-binds the joint to the
//! stored catalog and compares the cited leaves, re-validates the laws and
//! recomputes every conditional point bit for bit.
//!
//! The consumer is independent of the artifact, not of the implementation: it
//! re-runs the same decision procedure and the same exact evaluator as the
//! producer. Only the proof check ([`ConditionalTransportDerivation::from_record_checked`]:
//! the augmented-graph rule-2 criterion and the sID checker) is code distinct
//! from the search that produced the proof. The reduction check in the
//! consumer path only polls cancellation; it is at most a few separation
//! tests over a graph already refused above the route's size bounds.
//!
//! Proven obstructions travel in a separate format
//! (`checked_admg_conditional_obstruction_v1`,
//! [`AdmgConditionalObstructionWire`]): the proof record and its exactly
//! re-verified two-model witness, with no catalog or law.
//!
//! What replay does NOT protect against: laws that are not the populations'
//! real laws (the data digest names snapshots, it does not authenticate them),
//! a causal graph or selection diagram that is wrong about the world (the
//! identification is conditional on it), a bug shared by producer and consumer
//! (in the search or in the evaluator: both sides compute the same wrong
//! value, so it replays identically), and a re-sealed artifact whose every
//! premise was consistently replaced: the digests are integrity checks, not
//! signatures, so the consumer re-derives everything and a consistent forgery is
//! simply a different, correctly checked analysis.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, IdentityDomain, SearchLimits, VariableId,
};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    BoundConditionalTransportFunctional, ClassicalTransportQuery,
    ConditionalNonTransportabilityProof, ConditionalNonTransportabilityRecord,
    ConditionalTransportDecision, ConditionalTransportDerivation, ConditionalTransportQuery,
    ConditionalTransportRecord, IdentificationError, admg_conditional_refusal,
    decide_admg_conditional_transport,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const ADMG_CONDITIONAL_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const ADMG_CONDITIONAL_ARTIFACT_FEATURE: &str = "checked_admg_conditional_point_v1";
/// The only uncertainty claim of this route: exact laws, a point, no interval.
pub const ADMG_CONDITIONAL_POINT_ONLY: &str = "point_only";

/// Why an ADMG conditional transport artifact was refused. Every kind carries
/// its registered `(reason code, admg_transport.* detail)` pair ([`Self::refusal`]).
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AdmgConditionalArtifactError {
    /// The feature marker, result shape or name mapping is not this format's.
    #[error("admg_transport.invalid_artifact: unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("admg_transport.invalid_artifact: unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// The artifact claims an interval or carries counted laws.
    #[error("admg_transport.interval_withheld: the route publishes exact points only")]
    IntervalWithheld,
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("admg_transport.consumer_limits: consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored proof does not check, or the re-decision fails.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The re-decided derivation, re-bound leaves or a recomputed point differ.
    #[error("admg_transport.replay_mismatch: {0} does not replay")]
    ReplayMismatch(&'static str),
    /// A stored law is not a valid law or law set.
    #[error("admg_transport.invalid_artifact: invalid law: {0}")]
    LawInvalid(String),
    /// The premises digest does not match the stored premises.
    #[error("admg_transport.premises_mismatch: premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("admg_transport.data_identity_mismatch: data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the verified name mapping.
    #[error("admg_transport.invalid_artifact: variable names do not match the verified mapping")]
    NamesMismatch,
}

impl AdmgConditionalArtifactError {
    /// The registered `(reason code, admg_transport.* detail)` pair of this failure.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        use antecedent_core::reason_code;
        match self {
            Self::UnsupportedSemantics(_)
            | Self::UnsupportedVersion { .. }
            | Self::LawInvalid(_)
            | Self::NamesMismatch => {
                (reason_code!("invalid_argument"), "admg_transport.invalid_artifact")
            }
            Self::IntervalWithheld => {
                (reason_code!("cell_not_licensed"), "admg_transport.interval_withheld")
            }
            Self::LimitsExceeded(_) => {
                (reason_code!("route_not_supported"), "admg_transport.consumer_limits")
            }
            Self::ProofMismatch(inner) => admg_conditional_refusal(inner).unwrap_or((
                reason_code!("transport_not_certified"),
                "admg_transport.invalid_derivation",
            )),
            Self::ReplayMismatch(_) => {
                (reason_code!("transport_not_certified"), "admg_transport.replay_mismatch")
            }
            Self::PremisesMismatch => {
                (reason_code!("transport_not_certified"), "admg_transport.premises_mismatch")
            }
            Self::DataIdentityMismatch => {
                (reason_code!("transport_not_certified"), "admg_transport.data_identity_mismatch")
            }
        }
    }
}

/// A conditional-route identification error as the io error a caller sees: the
/// record's `reason=<code>: <detail>: ...` refusal when the route owns it.
#[must_use]
#[doc(hidden)]
pub fn admg_conditional_identification_error(error: IdentificationError) -> IoError {
    match admg_conditional_refusal(&error) {
        Some((code, detail)) => {
            let text = error.to_string();
            let message = if text.starts_with(detail) { text } else { format!("{detail}: {text}") };
            IoError::Refused { code, message }
        }
        None => error.into(),
    }
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct AdmgConditionalConsumeLimits {
    /// Largest search limits the consumer replays under.
    pub search: SearchLimits,
    /// Exact-evaluation operation and depth limits of the point replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
    /// Most requests an artifact may carry.
    pub max_requests: usize,
}

impl Default for AdmgConditionalConsumeLimits {
    fn default() -> Self {
        Self {
            search: antecedent_identify::ADMG_CONDITIONAL_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 1024,
            max_law_cells: 1_000_000,
            max_requests: 64,
        }
    }
}

/// Portable conditional query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmgConditionalQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Conditioned coordinates.
    pub conditioned_on: Vec<u32>,
    /// Source population holding every experiment.
    pub source: String,
    /// Target population.
    pub target: String,
}

impl AdmgConditionalQueryWire {
    /// Encode a query.
    #[must_use]
    pub fn from_query(query: &ConditionalTransportQuery) -> Self {
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            outcomes: raw(&query.base.outcomes),
            treatments: raw(&query.base.treatments),
            conditioned_on: raw(&query.conditioned_on),
            source: query.base.source.to_string(),
            target: query.base.target.to_string(),
        }
    }
    /// Decode. Validation happens when the query is re-checked.
    #[must_use]
    pub fn to_query(&self) -> ConditionalTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Arc<[_]>>();
        ConditionalTransportQuery {
            base: ClassicalTransportQuery {
                outcomes: ids(&self.outcomes),
                treatments: ids(&self.treatments),
                source: Arc::from(self.source.as_str()),
                target: Arc::from(self.target.as_str()),
            },
            conditioned_on: ids(&self.conditioned_on),
        }
    }
}

/// Conditional point distribution in atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdmgConditionalPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl AdmgConditionalPointWire {
    /// Encode a distribution.
    #[must_use]
    pub fn from_distribution(distribution: &ExactDistribution) -> Self {
        Self {
            outcomes: distribution.outcomes.iter().map(|v| v.raw()).collect(),
            atoms: distribution
                .atoms
                .iter()
                .map(|atom| atom.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: distribution.probabilities.to_vec(),
        }
    }

    /// Bit-exact replay; non-finite probabilities fail closed.
    fn replays(&self, recomputed: &Self) -> bool {
        self.outcomes == recomputed.outcomes
            && self.atoms == recomputed.atoms
            && self.probabilities.len() == recomputed.probabilities.len()
            && self.probabilities.iter().zip(&recomputed.probabilities).all(|(stored, fresh)| {
                stored.is_finite() && fresh.is_finite() && stored.to_bits() == fresh.to_bits()
            })
    }
}

/// Versioned conditional point execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmgConditionalArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Causal graph.
    pub graph: AdmgWire,
    /// Selection targets of the diagram, sorted.
    pub selections: Vec<u32>,
    /// The conditional query.
    pub query: AdmgConditionalQueryWire,
    /// Search operation limit the producer decided under.
    pub search_operations: usize,
    /// Search depth limit the producer decided under.
    pub search_depth: usize,
    /// Checked proof: rule-2 moves, remaining set and the reduced joint's sID record.
    pub proof: ConditionalTransportRecord,
    /// Expression arena of the reduced joint's derivation.
    pub expression: ExprArenaWire,
    /// `(population, regime)` of every leaf of the bound joint, sorted.
    pub bindings: Vec<(String, Option<u32>)>,
    /// Evidence catalog.
    pub catalog: EvidenceCatalogWire,
    /// Exact laws in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under.
    pub max_support_rows: usize,
    /// Requests binding the treatments and conditioned variables.
    pub requests: Vec<Vec<(u32, ValueWire)>>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// Variable name of every graph coordinate, or empty.
    pub variable_names: Vec<String>,
    /// Conditional point of every request, in request order.
    pub results: Vec<AdmgConditionalPointWire>,
    /// The uncertainty claim; always [`ADMG_CONDITIONAL_POINT_ONLY`].
    pub uncertainty: String,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the data identity (refresh-replaceable snapshot ids).
    pub data_digest: String,
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedAdmgConditional {
    /// The re-checked, re-bound functional.
    pub functional: BoundConditionalTransportFunctional,
    /// The rebuilt selection diagram.
    pub diagram: SelectionDiagram,
    /// The re-validated laws.
    pub data: ExactTransportData,
    /// The requests.
    pub requests: Vec<Assignment>,
    /// The recomputed conditional point of every request.
    pub distributions: Vec<ExactDistribution>,
    /// The decoded artifact.
    pub wire: AdmgConditionalArtifactWire,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    selections: &'a [u32],
    query: &'a AdmgConditionalQueryWire,
    proof: &'a ConditionalTransportRecord,
    expression: &'a ExprArenaWire,
    search: (usize, usize),
    evaluation: (usize, usize, usize),
    requests: &'a [Vec<(u32, ValueWire)>],
    variable_names: &'a [String],
}

/// The whole evidence catalog (environments, regimes, bindings with their
/// snapshot and dataset ids), the sorted `(regime, snapshot, dataset)` of every
/// binding and the `(population, regime, snapshot)` of every law.
fn data_digest(catalog: &EvidenceCatalog, laws: &[ExactLawWire]) -> Result<String, IoError> {
    let mut bindings = catalog
        .bindings
        .iter()
        .map(|b| {
            (
                b.regime.raw(),
                b.snapshot_identity.to_string(),
                b.dataset_identity.as_deref().map(str::to_owned),
            )
        })
        .collect::<Vec<_>>();
    bindings.sort();
    let mut snapshots = laws
        .iter()
        .map(|law| (law.population.clone(), law.regime, law.snapshot.clone()))
        .collect::<Vec<_>>();
    snapshots.sort();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &(
            "admg_conditional_data_v1",
            EvidenceCatalogWire::from_catalog(catalog),
            bindings,
            snapshots,
        ),
    )?
    .to_hex())
}

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

fn bindings_of(functional: &BoundConditionalTransportFunctional) -> Vec<(String, Option<u32>)> {
    functional.cited_leaves().into_iter().map(|(p, r)| (p.to_string(), r)).collect()
}

fn request_wire(request: &Assignment) -> Vec<(u32, ValueWire)> {
    request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect()
}

fn request_of(wire: &[(u32, ValueWire)]) -> Assignment {
    Assignment::from_pairs(wire.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())))
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// The checked premises and results an artifact is built from.
pub struct AdmgConditionalArtifactInput<'a> {
    /// The selection diagram the functional was decided on.
    pub diagram: &'a SelectionDiagram,
    /// The checked, catalog-bound functional.
    pub functional: &'a BoundConditionalTransportFunctional,
    /// Search limits the functional was decided under.
    pub search: SearchLimits,
    /// Retained exact laws.
    pub data: &'a ExactTransportData,
    /// Requests, in order.
    pub requests: &'a [Assignment],
    /// Evaluation limits.
    pub limits: ExactEvaluationLimits,
    /// Variable name of every graph coordinate, or empty.
    pub variable_names: &'a [String],
    /// The conditional point of every request.
    pub results: &'a [ExactDistribution],
}

impl AdmgConditionalArtifactWire {
    /// Build an artifact from checked premises and points. Producer-side bounds
    /// are enforced here: search limits above the route's maxima and counted
    /// laws are refused.
    ///
    /// # Errors
    /// The premises do not encode, exceed the route's bounds, or the results are
    /// inconsistent.
    pub fn checked(input: &AdmgConditionalArtifactInput<'_>) -> Result<Self, IoError> {
        let maxima = antecedent_identify::ADMG_CONDITIONAL_DEFAULT_LIMITS;
        if input.search.operations > maxima.operations || input.search.depth > maxima.depth {
            return Err(AdmgConditionalArtifactError::LimitsExceeded("search limits").into());
        }
        if input.data.laws().iter().any(|law| law.empirical_counts().is_some()) {
            return Err(AdmgConditionalArtifactError::IntervalWithheld.into());
        }
        let derivation = input.functional.derivation();
        let laws = canonical_laws(input.data)?;
        let mut selections =
            input.diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>();
        selections.sort_unstable();
        let mut wire = Self {
            version: ADMG_CONDITIONAL_ARTIFACT_VERSION,
            required_features: vec![ADMG_CONDITIONAL_ARTIFACT_FEATURE.into()],
            graph: admg_to_wire(input.diagram.causal_graph())?,
            selections,
            query: AdmgConditionalQueryWire::from_query(derivation.query()),
            search_operations: input.search.operations,
            search_depth: input.search.depth,
            proof: derivation.to_record(),
            expression: expr_arena_to_wire(derivation.joint().arena())?,
            bindings: bindings_of(input.functional),
            catalog: EvidenceCatalogWire::from_catalog(input.functional.catalog()),
            data_digest: data_digest(input.functional.catalog(), &laws)?,
            laws,
            max_support_rows: input.data.max_support_rows(),
            requests: input.requests.iter().map(request_wire).collect(),
            operation_limit: input.limits.operations,
            depth_limit: input.limits.depth,
            variable_names: input.variable_names.to_vec(),
            results: input
                .results
                .iter()
                .map(AdmgConditionalPointWire::from_distribution)
                .collect(),
            uncertainty: ADMG_CONDITIONAL_POINT_ONLY.into(),
            premises_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn premises_view(&self) -> PremisesView<'_> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        PremisesView {
            tag: "admg_conditional_point_v1",
            graph,
            selections: &self.selections,
            query: &self.query,
            proof: &self.proof,
            expression: &self.expression,
            search: (self.search_operations, self.search_depth),
            evaluation: (self.operation_limit, self.depth_limit, self.max_support_rows),
            requests: &self.requests,
            variable_names: &self.variable_names,
        }
    }

    /// The digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-checks every premise.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &self.premises_view(),
        )?
        .to_hex())
    }

    /// The data-identity digest the stored snapshot ids should carry.
    ///
    /// # Errors
    /// The catalog does not decode or the digest does not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_digest(&self.catalog.to_catalog()?, &self.laws)
    }

    fn validate_shape(&self) -> Result<(), AdmgConditionalArtifactError> {
        let unsupported = AdmgConditionalArtifactError::UnsupportedSemantics;
        if self.required_features != [ADMG_CONDITIONAL_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        if self.uncertainty != ADMG_CONDITIONAL_POINT_ONLY
            || self.laws.iter().any(|law| law.empirical_counts.is_some())
        {
            return Err(AdmgConditionalArtifactError::IntervalWithheld);
        }
        if self.requests.is_empty() || self.results.len() != self.requests.len() {
            return Err(unsupported("one point result per request"));
        }
        if self.results.iter().any(|result| {
            result.atoms.len() != result.probabilities.len()
                || result.outcomes != self.query.outcomes
        }) {
            return Err(unsupported("point result shape"));
        }
        let nodes = self.graph.node_count;
        if !self.variable_names.is_empty()
            && (u32::try_from(self.variable_names.len()).ok() != Some(nodes)
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != self.variable_names.len())
        {
            return Err(unsupported("variable names"));
        }
        Ok(())
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// [`AdmgConditionalArtifactError::NamesMismatch`] unless `names` equals the
    /// stored mapping.
    pub fn check_variable_names(
        &self,
        names: &[String],
    ) -> Result<(), AdmgConditionalArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(AdmgConditionalArtifactError::NamesMismatch)
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`AdmgConditionalArtifactError::UnsupportedVersion`] for another
    /// version, or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != ADMG_CONDITIONAL_ARTIFACT_VERSION {
            return Err(AdmgConditionalArtifactError::UnsupportedVersion { version: peek.version }.into());
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &AdmgConditionalConsumeLimits,
    ) -> Result<(), AdmgConditionalArtifactError> {
        use antecedent_identify::{
            ADMG_CONDITIONAL_MAX_CONDITIONED, ADMG_CONDITIONAL_MAX_OBSERVED,
            ADMG_CONDITIONAL_MAX_TREATMENTS,
        };
        let exceeded = AdmgConditionalArtifactError::LimitsExceeded;
        // The route's size bounds, checked before any digest is hashed: an
        // oversized graph or query is never decoded further or re-checked.
        let nodes = usize::try_from(self.graph.node_count).unwrap_or(usize::MAX);
        let max_edges = nodes.saturating_mul(nodes.saturating_sub(1));
        if nodes > ADMG_CONDITIONAL_MAX_OBSERVED
            || self.graph.directed.len() > max_edges
            || self.graph.bidirected.len() > max_edges
            || self.selections.len() > nodes
        {
            return Err(exceeded("graph size"));
        }
        if self.query.outcomes.len() > nodes
            || self.query.treatments.len() > ADMG_CONDITIONAL_MAX_TREATMENTS
            || self.query.conditioned_on.len() > ADMG_CONDITIONAL_MAX_CONDITIONED
            || self.proof.conditioned_on.len() > ADMG_CONDITIONAL_MAX_CONDITIONED
            || self.proof.moves.len() + self.proof.remaining.len()
                > ADMG_CONDITIONAL_MAX_CONDITIONED
        {
            return Err(exceeded("query size"));
        }
        if self.search_operations > limits.search.operations {
            return Err(exceeded("search operation limit"));
        }
        if self.search_depth > limits.search.depth {
            return Err(exceeded("search depth limit"));
        }
        if self.operation_limit > limits.evaluation.operations {
            return Err(exceeded("operation limit"));
        }
        if self.depth_limit > limits.evaluation.depth {
            return Err(exceeded("depth limit"));
        }
        if self.max_support_rows > limits.max_support_rows {
            return Err(exceeded("support rows"));
        }
        if self.laws.len() > limits.max_laws {
            return Err(exceeded("law count"));
        }
        if self.requests.len() > limits.max_requests {
            return Err(exceeded("request count"));
        }
        if self.laws.iter().any(|law| law.probabilities.len() > limits.max_law_cells) {
            return Err(exceeded("law cells"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, replaying the
    /// proof and the decision under the producer's stored search limits, then
    /// recompute every point. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure (a typed [`AdmgConditionalArtifactError`]), an
    /// evaluation refusal, or a point that does not replay bit for bit.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: AdmgConditionalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedAdmgConditional, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(AdmgConditionalArtifactError::PremisesMismatch.into());
        }
        let catalog = wire.catalog.to_catalog()?;
        if data_digest(&catalog, &wire.laws)? != wire.data_digest {
            return Err(AdmgConditionalArtifactError::DataIdentityMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|_| AdmgConditionalArtifactError::UnsupportedSemantics("selection diagram"))?;
        let query = wire.query.to_query();
        let arena = expr_arena_from_wire(&wire.expression)?;
        let stored = SearchLimits { operations: wire.search_operations, depth: wire.search_depth };
        let derivation = ConditionalTransportDerivation::from_record_checked(
            wire.proof.clone(),
            arena,
            &diagram,
            &query,
            stored,
            ctx,
        )
        .map_err(AdmgConditionalArtifactError::ProofMismatch)?;
        // The producer's decision replays under its own stored limits.
        let decided = decide_admg_conditional_transport(&diagram, &query, &catalog, stored, ctx)
            .map_err(AdmgConditionalArtifactError::ProofMismatch)?;
        let ConditionalTransportDecision::Identified(decided) = decided else {
            return Err(AdmgConditionalArtifactError::ReplayMismatch("the decision").into());
        };
        if crate::to_cbor(&decided.derivation().to_record())? != crate::to_cbor(&wire.proof)? {
            return Err(AdmgConditionalArtifactError::ReplayMismatch("the derivation").into());
        }
        let functional = derivation
            .bind_catalog(&catalog)
            .map_err(AdmgConditionalArtifactError::ProofMismatch)?;
        if bindings_of(&functional) != wire.bindings {
            return Err(AdmgConditionalArtifactError::ReplayMismatch("the leaf bindings").into());
        }
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| AdmgConditionalArtifactError::LawInvalid(error.to_string()))?;
        let data = ExactTransportData::try_new(laws, wire.max_support_rows)
            .map_err(|error| AdmgConditionalArtifactError::LawInvalid(error.to_string()))?;
        let requests = wire.requests.iter().map(|r| request_of(r)).collect::<Vec<_>>();
        let evaluation =
            ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit };
        let mut distributions = Vec::with_capacity(requests.len());
        for (request, stored) in requests.iter().zip(&wire.results) {
            let distribution = antecedent_estimate::prepare_exact_admg_conditional_transport(
                &functional,
                data.clone(),
                request,
                evaluation,
                ctx,
            )
            .and_then(|plan| plan.evaluate(ctx))?;
            if !stored.replays(&AdmgConditionalPointWire::from_distribution(&distribution)) {
                return Err(AdmgConditionalArtifactError::ReplayMismatch("a point").into());
            }
            distributions.push(distribution);
        }
        Ok(ConsumedAdmgConditional { functional, diagram, data, requests, distributions, wire })
    }
}

/// The obstruction format this reader writes and accepts.
pub const ADMG_CONDITIONAL_OBSTRUCTION_VERSION: u32 = 1;
/// The feature marker of the accepted obstruction format.
pub const ADMG_CONDITIONAL_OBSTRUCTION_FEATURE: &str = "checked_admg_conditional_obstruction_v1";

/// Versioned proven obstruction: the diagram, the query and the proof record
/// (moves, remaining set, the reduced joint's s-hedge and the two-model
/// witness). A consumer re-verifies the witness by exact enumeration and
/// re-checks the moves and the s-hedge; it trusts nothing stored.
///
/// The witness is self-certifying, so no decision is replayed and no catalog
/// or law is stored: two models that agree on every source experimental law
/// and on the target observational law, and differ on the query, refute every
/// formula over any catalog of those laws. The consumer runs the same verifier
/// code as the producer (independent of the artifact, not of the
/// implementation); the test suites re-check witnesses with separate code.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdmgConditionalObstructionWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Causal graph.
    pub graph: AdmgWire,
    /// Selection targets of the diagram, sorted.
    pub selections: Vec<u32>,
    /// The conditional query.
    pub query: AdmgConditionalQueryWire,
    /// The proof record.
    pub proof: ConditionalNonTransportabilityRecord,
    /// Variable name of every graph coordinate, or empty.
    pub variable_names: Vec<String>,
    /// Digest of the canonical premises (graph, selections, query, proof, names).
    pub premises_digest: String,
}

/// A consumed obstruction: the re-verified proof and its diagram.
pub struct ConsumedAdmgConditionalObstruction {
    /// The re-verified proof.
    pub proof: ConditionalNonTransportabilityProof,
    /// The rebuilt selection diagram.
    pub diagram: SelectionDiagram,
    /// The decoded artifact.
    pub wire: AdmgConditionalObstructionWire,
}

#[derive(Serialize)]
struct ObstructionPremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    selections: &'a [u32],
    query: &'a AdmgConditionalQueryWire,
    proof: &'a ConditionalNonTransportabilityRecord,
    variable_names: &'a [String],
}

impl AdmgConditionalObstructionWire {
    /// Build an obstruction artifact from a proof decided on `diagram`; the
    /// proof is re-checked before it is written.
    ///
    /// # Errors
    /// The proof does not re-check on `diagram`, or the premises do not encode.
    pub fn checked(
        diagram: &SelectionDiagram,
        proof: &ConditionalNonTransportabilityProof,
        variable_names: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        proof
            .recheck(diagram, proof.query(), ctx)
            .map_err(AdmgConditionalArtifactError::ProofMismatch)?;
        let mut selections =
            diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>();
        selections.sort_unstable();
        let mut wire = Self {
            version: ADMG_CONDITIONAL_OBSTRUCTION_VERSION,
            required_features: vec![ADMG_CONDITIONAL_OBSTRUCTION_FEATURE.into()],
            graph: admg_to_wire(diagram.causal_graph())?,
            selections,
            query: AdmgConditionalQueryWire::from_query(proof.query()),
            proof: proof.to_record(),
            variable_names: variable_names.to_vec(),
            premises_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-verifies the witness.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &ObstructionPremisesView {
                tag: "admg_conditional_obstruction_v1",
                graph,
                selections: &self.selections,
                query: &self.query,
                proof: &self.proof,
                variable_names: &self.variable_names,
            },
        )?
        .to_hex())
    }

    fn validate_shape(&self) -> Result<(), AdmgConditionalArtifactError> {
        let unsupported = AdmgConditionalArtifactError::UnsupportedSemantics;
        if self.required_features != [ADMG_CONDITIONAL_OBSTRUCTION_FEATURE] {
            return Err(unsupported("required features"));
        }
        let nodes = self.graph.node_count;
        if !self.variable_names.is_empty()
            && (u32::try_from(self.variable_names.len()).ok() != Some(nodes)
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != self.variable_names.len())
        {
            return Err(unsupported("variable names"));
        }
        Ok(())
    }

    /// The route's size bounds, checked before any digest is hashed.
    fn check_bounds(&self) -> Result<(), AdmgConditionalArtifactError> {
        use antecedent_identify::{
            ADMG_CONDITIONAL_MAX_CONDITIONED, ADMG_CONDITIONAL_MAX_OBSERVED,
            ADMG_CONDITIONAL_MAX_TREATMENTS, CONDITIONAL_WITNESS_MAX_LATENT_LEVELS,
        };
        let exceeded = AdmgConditionalArtifactError::LimitsExceeded;
        let nodes = usize::try_from(self.graph.node_count).unwrap_or(usize::MAX);
        let max_edges = nodes.saturating_mul(nodes.saturating_sub(1));
        if nodes > ADMG_CONDITIONAL_MAX_OBSERVED
            || self.graph.directed.len() > max_edges
            || self.graph.bidirected.len() > max_edges
            || self.selections.len() > nodes
        {
            return Err(exceeded("graph size"));
        }
        if self.query.outcomes.len() > nodes
            || self.query.treatments.len() > ADMG_CONDITIONAL_MAX_TREATMENTS
            || self.query.conditioned_on.len() > ADMG_CONDITIONAL_MAX_CONDITIONED
            || self.proof.moves.len() + self.proof.remaining.len()
                > ADMG_CONDITIONAL_MAX_CONDITIONED
        {
            return Err(exceeded("query size"));
        }
        // Witness shape bounds; the verifier also refuses any model whose
        // enumeration exceeds its work bound before any arithmetic.
        let witness = &self.proof.witness;
        for model in [&witness.first, &witness.second] {
            if model.latents.len() > max_edges
                || model.source.len() > nodes
                || model.target.len() > nodes
                || model
                    .latents
                    .iter()
                    .any(|l| l.probabilities.len() > CONDITIONAL_WITNESS_MAX_LATENT_LEVELS)
            {
                return Err(exceeded("witness size"));
            }
        }
        Ok(())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`AdmgConditionalArtifactError::UnsupportedVersion`] for another
    /// version, or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != ADMG_CONDITIONAL_OBSTRUCTION_VERSION {
            return Err(AdmgConditionalArtifactError::UnsupportedVersion { version: peek.version }.into());
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// [`AdmgConditionalArtifactError::NamesMismatch`] unless `names` equals the
    /// stored mapping.
    pub fn check_variable_names(
        &self,
        names: &[String],
    ) -> Result<(), AdmgConditionalArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(AdmgConditionalArtifactError::NamesMismatch)
        }
    }

    /// Decode and re-verify: size bounds before any digest, the premises digest,
    /// then the proof from its record (the witness by exact enumeration, the
    /// moves and their maximality, the reduced joint's s-hedge).
    ///
    /// # Errors
    /// A typed [`AdmgConditionalArtifactError`]: consumer limits, premises
    /// mismatch, or a proof that does not verify (`invalid_derivation`).
    pub fn consume(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<ConsumedAdmgConditionalObstruction, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_bounds()?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(AdmgConditionalArtifactError::PremisesMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|_| AdmgConditionalArtifactError::UnsupportedSemantics("selection diagram"))?;
        let query = wire.query.to_query();
        let proof = ConditionalNonTransportabilityProof::from_record_checked(
            wire.proof.clone(),
            &diagram,
            &query,
            ctx,
        )
        .map_err(AdmgConditionalArtifactError::ProofMismatch)?;
        Ok(ConsumedAdmgConditionalObstruction { proof, diagram, wire })
    }
}
