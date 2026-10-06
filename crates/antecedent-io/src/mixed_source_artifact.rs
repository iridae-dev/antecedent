//! Independent point-result artifacts for mixed-source proof search (X9).
//!
//! Format version 1. A consumer trusts nothing in the artifact. It re-runs the
//! bounded mixed-source decision on the stored graph, query and catalog under its
//! own search limits and accepts only the identical derivation: the frozen
//! rule-set version, every step with its premises and source distributions, and
//! the expression arena. The independent proof checker replays every step inside
//! that decision. It re-binds every leaf to the regime that supplies it and compares
//! the source-named bindings, re-validates the laws, and recomputes every
//! request's point bit for bit. The verified identity is two digests: the premises
//! digest (graph, query, proof, expression, search and evaluation limits, requests
//! and the variable-name mapping) and the data-identity digest (catalog snapshot
//! and dataset ids and every law's snapshot). The route is point-only: no interval
//! is stored, and one that appears is refused. Limits an artifact records are
//! provenance; a stored limit larger than the consumer's refuses.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    mz_transport_artifact::MzSourceWire,
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, IdentityDomain, InterventionAssignment, SearchLimits,
    VariableId,
};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMixedSourceFunctional, IdentificationError, MixedSourceDerivation,
    MixedSourceDerivationRecord, MixedSourceQuery, ZTransportSourceSpec, bind_mixed_source_catalog,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const MIXED_SOURCE_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const MIXED_SOURCE_ARTIFACT_FEATURE: &str = "checked_mixed_source_point_v1";
/// The only uncertainty claim of this route: exact laws, a point, no interval.
pub const MIXED_SOURCE_POINT_ONLY: &str = "point_only";

/// Why a mixed-source artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MixedSourceArtifactError {
    /// The feature marker, claim or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-run decision did not reproduce the stored derivation.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// The stored source-named bindings differ from the re-derived ones.
    #[error("source bindings do not match the checked binding")]
    BindingMismatch,
    /// A stored law is not a valid law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// A recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct MixedSourceConsumeLimits {
    /// Limits of the re-run decision.
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

impl Default for MixedSourceConsumeLimits {
    fn default() -> Self {
        Self {
            search: antecedent_identify::MIXED_SOURCE_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
            max_requests: 64,
        }
    }
}

/// Portable canonical mixed-source query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MixedSourceQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Target population.
    pub target: String,
    /// Declared sources of the theorem-scoped routes, in canonical order.
    pub sources: Vec<MzSourceWire>,
}

impl MixedSourceQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &MixedSourceQuery) -> Self {
        let query = query.canonical();
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            outcomes: raw(&query.outcomes),
            treatments: raw(&query.treatments),
            target: query.target.to_string(),
            sources: query
                .sources
                .iter()
                .map(|s| MzSourceWire {
                    population: s.population.to_string(),
                    controllable: raw(&s.controllable),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect(),
                    selection_targets: raw(&s.selection_targets),
                })
                .collect(),
        }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(&self) -> MixedSourceQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        MixedSourceQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            target: Arc::from(self.target.as_str()),
            sources: self
                .sources
                .iter()
                .map(|s| ZTransportSourceSpec {
                    population: Arc::from(s.population.as_str()),
                    controllable: ids(&s.controllable).into(),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|(v, x)| InterventionAssignment {
                            variable: VariableId::from_raw(*v),
                            value: x.to_value(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                    selection_targets: ids(&s.selection_targets).into(),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

/// Point distribution in canonical atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MixedSourcePointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl MixedSourcePointWire {
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

/// Versioned mixed-source point execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedSourceArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Shared causal graph.
    pub graph: AdmgWire,
    /// Canonical query.
    pub query: MixedSourceQueryWire,
    /// Search operation limit the producer decided under; bound by the premises digest.
    pub search_operations: usize,
    /// Search depth limit the producer decided under; bound by the premises digest.
    pub search_depth: usize,
    /// Checked proof premises: rule set, steps with premises and source
    /// distributions, and the search receipt.
    pub proof: MixedSourceDerivationRecord,
    /// Expression arena of the checked derivation.
    pub expression: ExprArenaWire,
    /// `(regime, study, population)` of every cited distribution: the source-named leaves.
    pub bindings: Vec<(u32, String, String)>,
    /// Evidence catalog, including declared sampling and study identities.
    pub catalog: EvidenceCatalogWire,
    /// Laws of the cited regimes in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under.
    pub max_support_rows: usize,
    /// Target requests.
    pub requests: Vec<Vec<(u32, ValueWire)>>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// Variable name of every graph coordinate, or empty when the producer named
    /// none; bound by the premises digest.
    pub variable_names: Vec<String>,
    /// Point result of every request, in request order.
    pub results: Vec<MixedSourcePointWire>,
    /// The uncertainty claim; always [`MIXED_SOURCE_POINT_ONLY`].
    pub uncertainty: String,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the data identity (refresh-replaceable snapshot ids).
    pub data_digest: String,
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedMixedSource {
    /// The re-decided, re-bound functional.
    pub functional: BoundMixedSourceFunctional,
    /// The re-validated laws.
    pub data: ExactTransportData,
    /// The target requests.
    pub requests: Vec<Assignment>,
    /// The recomputed point of every request.
    pub distributions: Vec<ExactDistribution>,
    /// The decoded artifact.
    pub wire: MixedSourceArtifactWire,
}

/// Everything the premises digest binds, in canonical form.
#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    query: &'a MixedSourceQueryWire,
    proof: &'a MixedSourceDerivationRecord,
    expression: &'a ExprArenaWire,
    search: (usize, usize),
    evaluation: (usize, usize, usize),
    requests: &'a [Vec<(u32, ValueWire)>],
    variable_names: &'a [String],
}

impl MixedSourceArtifactWire {
    fn premises_view(&self) -> PremisesView<'_> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        PremisesView {
            tag: "mixed_source_point_v1",
            graph,
            query: &self.query,
            proof: &self.proof,
            expression: &self.expression,
            search: (self.search_operations, self.search_depth),
            evaluation: (self.operation_limit, self.depth_limit, self.max_support_rows),
            requests: &self.requests,
            variable_names: &self.variable_names,
        }
    }
}

/// `(regime, snapshot, dataset)` of every catalog binding and
/// `(population, regime, snapshot)` of every law, sorted.
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
        &("mixed_source_data_v1", bindings, snapshots),
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

fn bindings_of(functional: &BoundMixedSourceFunctional) -> Vec<(u32, String, String)> {
    functional
        .cited_sources()
        .into_iter()
        .map(|(id, study, population)| (id.raw(), study.to_string(), population.to_string()))
        .collect()
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
pub struct MixedSourceArtifactInput<'a> {
    /// Shared causal graph the functional was decided on.
    pub graph: &'a antecedent_graph::Admg,
    /// The checked, catalog-bound functional.
    pub functional: &'a BoundMixedSourceFunctional,
    /// Search limits the functional was decided under.
    pub search: SearchLimits,
    /// Retained laws.
    pub data: &'a ExactTransportData,
    /// Requests, in order.
    pub requests: &'a [Assignment],
    /// Evaluation limits.
    pub limits: ExactEvaluationLimits,
    /// Variable name of every graph coordinate, or empty.
    pub variable_names: &'a [String],
    /// The point of every request.
    pub results: &'a [ExactDistribution],
}

impl MixedSourceArtifactWire {
    /// Build an artifact from checked premises and points.
    ///
    /// # Errors
    /// The premises do not encode, or the results are inconsistent.
    pub fn checked(input: &MixedSourceArtifactInput<'_>) -> Result<Self, IoError> {
        let graph = admg_to_wire(input.graph)?;
        let derivation = input.functional.derivation();
        let laws = canonical_laws(input.data)?;
        let mut wire = Self {
            version: MIXED_SOURCE_ARTIFACT_VERSION,
            required_features: vec![MIXED_SOURCE_ARTIFACT_FEATURE.into()],
            graph,
            query: MixedSourceQueryWire::from_query(derivation.query()),
            search_operations: input.search.operations,
            search_depth: input.search.depth,
            proof: derivation.to_record(),
            expression: expr_arena_to_wire(derivation.arena())?,
            bindings: bindings_of(input.functional),
            catalog: EvidenceCatalogWire::from_catalog(input.functional.catalog()),
            data_digest: data_digest(input.functional.catalog(), &laws)?,
            laws,
            max_support_rows: input.data.max_support_rows(),
            requests: input.requests.iter().map(request_wire).collect(),
            operation_limit: input.limits.operations,
            depth_limit: input.limits.depth,
            variable_names: input.variable_names.to_vec(),
            results: input.results.iter().map(MixedSourcePointWire::from_distribution).collect(),
            uncertainty: MIXED_SOURCE_POINT_ONLY.into(),
            premises_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-decides and re-binds every premise.
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
    /// The catalog or laws do not decode or encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_digest(&self.catalog.to_catalog()?, &self.laws)
    }

    fn validate_shape(&self) -> Result<(), MixedSourceArtifactError> {
        let unsupported = MixedSourceArtifactError::UnsupportedSemantics;
        if self.required_features != [MIXED_SOURCE_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        if self.uncertainty != MIXED_SOURCE_POINT_ONLY {
            return Err(unsupported("this route publishes points only"));
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
    /// [`MixedSourceArtifactError::NamesMismatch`] unless `names` equals the stored
    /// mapping (an artifact without a mapping accepts none but the empty one).
    pub fn check_variable_names(&self, names: &[String]) -> Result<(), MixedSourceArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(MixedSourceArtifactError::NamesMismatch)
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
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != MIXED_SOURCE_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &MixedSourceConsumeLimits,
    ) -> Result<(), MixedSourceArtifactError> {
        let exceeded = MixedSourceArtifactError::LimitsExceeded;
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
        if self.laws.iter().any(|law| {
            law.probabilities.len() > limits.max_law_cells
                || law.empirical_counts.as_ref().is_some_and(|c| c.len() > limits.max_law_cells)
        }) {
            return Err(exceeded("law cells"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then recompute
    /// every point. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, an evaluation refusal, or a point that does not
    /// replay bit for bit.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: MixedSourceConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedMixedSource, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(MixedSourceArtifactError::PremisesMismatch.into());
        }
        let catalog = wire.catalog.to_catalog()?;
        if data_digest(&catalog, &wire.laws)? != wire.data_digest {
            return Err(MixedSourceArtifactError::DataIdentityMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let query = wire.query.to_query();
        let arena = expr_arena_from_wire(&wire.expression)?;
        let derivation = MixedSourceDerivation::from_record_checked(
            &graph,
            &query,
            &catalog,
            &wire.proof,
            &arena,
            limits.search,
            ctx,
        )
        .map_err(MixedSourceArtifactError::ProofMismatch)?;
        let functional = bind_mixed_source_catalog(&graph, &derivation, &catalog)
            .map_err(MixedSourceArtifactError::CatalogBinding)?;
        if bindings_of(&functional) != wire.bindings {
            return Err(MixedSourceArtifactError::BindingMismatch.into());
        }
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| MixedSourceArtifactError::LawInvalid(error.to_string()))?;
        let data = ExactTransportData::try_new(laws, limits.max_support_rows)
            .map_err(|error| MixedSourceArtifactError::LawInvalid(error.to_string()))?;
        let requests = wire.requests.iter().map(|r| request_of(r)).collect::<Vec<_>>();
        let mut distributions = Vec::with_capacity(requests.len());
        for (request, stored) in requests.iter().zip(&wire.results) {
            let distribution = antecedent_estimate::evaluate_exact_mixed_source(
                &functional,
                data.clone(),
                request.clone(),
                limits.evaluation,
                ctx,
            )
            .map_err(|error| antecedent_estimate::refuse_eval(&error))?;
            if !stored.replays(&MixedSourcePointWire::from_distribution(&distribution)) {
                return Err(MixedSourceArtifactError::PointMismatch.into());
            }
            distributions.push(distribution);
        }
        Ok(ConsumedMixedSource { functional, data, requests, distributions, wire })
    }

    /// [`Self::consume_with_limits`] under the default consumer limits.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    #[doc(hidden)]
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<ConsumedMixedSource, IoError> {
        Self::consume_with_limits(bytes, MixedSourceConsumeLimits::default(), ctx)
    }
}
