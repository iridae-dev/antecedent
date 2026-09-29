//! Independent point-result artifacts for multi-source limited-experiment transport.
//!
//! Format version 1. A consumer trusts nothing in the artifact. It re-runs the
//! bounded `TR^mz` decision on the stored graph, query and catalog under its own
//! search limits and accepts only the identical derivation, including the search
//! receipt of the stages that preceded it; re-binds every leaf to the regime of
//! its own population and compares the source-specific bindings; re-validates the
//! laws; recomputes the point bit for bit; and re-derives the interval licensing
//! decision from the laws and the catalog's declared sampling. It does not re-run
//! a seeded bootstrap: the stored interval bookkeeping must be consistent with
//! that decision and with itself. Limits an artifact records are provenance; a
//! stored limit larger than the consumer's refuses.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{
    ExecutionContext, IdentityDomain, InterventionAssignment, SearchLimits, VariableId,
};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMzTransportFunctional, IdentificationError, MzTransportDerivation,
    MzTransportDerivationRecord, MzTransportQuery, ZTransportSourceSpec, bind_mz_transport_catalog,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const MZ_TRANSPORT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const MZ_TRANSPORT_ARTIFACT_FEATURE: &str = "checked_mz_transport_point_v1";
/// Uncertainty status: exact laws, no sampling uncertainty.
pub const MZ_POINT_ONLY: &str = "point_only";
/// Uncertainty status: an interval was licensable but withheld for a stated reason.
pub const MZ_WITHHELD: &str = "withheld";
/// Uncertainty status: a nominal pointwise percentile-bootstrap interval.
pub const MZ_NOMINAL_INTERVAL: &str = "nominal_interval";

/// Why an mz-transport artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MzTransportArtifactError {
    /// The feature marker or result shape is not this format's.
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
    /// The stored source-specific bindings differ from the re-derived ones.
    #[error("source bindings do not match the checked binding")]
    BindingMismatch,
    /// A stored law is not a valid law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// The recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// The stored interval bookkeeping contradicts the licensing decision.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct MzTransportConsumeLimits {
    /// Limits of the re-run identification search.
    pub search: SearchLimits,
    /// Exact-evaluation operation and depth limits of the point replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
}

impl Default for MzTransportConsumeLimits {
    fn default() -> Self {
        Self {
            search: antecedent_identify::MZ_TRANSPORT_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
        }
    }
}

/// One source of a stored query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzSourceWire {
    /// Source population.
    pub population: String,
    /// Declared controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment levels.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Selection targets on this source's diagram.
    pub selection_targets: Vec<u32>,
}

/// Portable canonical mz query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Target population.
    pub target: String,
    /// Sources in canonical order.
    pub sources: Vec<MzSourceWire>,
}

impl MzTransportQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &MzTransportQuery) -> Self {
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
    pub fn to_query(&self) -> MzTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        MzTransportQuery {
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
pub struct MzTransportPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl MzTransportPointWire {
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

/// Interval bookkeeping of one executed request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzUncertaintyWire {
    /// [`MZ_POINT_ONLY`], [`MZ_WITHHELD`] or [`MZ_NOMINAL_INTERVAL`].
    pub status: String,
    /// `no_interval_reported`, the withheld reason, or
    /// `estimator_grid_not_measured` for a nominal interval.
    pub reason: String,
    /// Interval method of a nominal interval.
    pub method: Option<String>,
    /// Nominal level of a nominal interval.
    pub coverage_target: Option<f64>,
    /// Requested bootstrap replicates; zero when no bootstrap ran.
    pub replicates_requested: u32,
    /// Replicates where every request evaluated.
    pub replicates_ok: u32,
    /// Replicates that failed.
    pub replicates_failed: u32,
    /// `(outcome, lower, upper)` pointwise mean intervals.
    pub mean_intervals: Vec<(u32, f64, f64)>,
}

impl MzUncertaintyWire {
    /// Exact laws: no sampling uncertainty.
    #[must_use]
    pub fn point_only() -> Self {
        Self {
            status: MZ_POINT_ONLY.into(),
            reason: "no_interval_reported".into(),
            method: None,
            coverage_target: None,
            replicates_requested: 0,
            replicates_ok: 0,
            replicates_failed: 0,
            mean_intervals: Vec::new(),
        }
    }
}

/// Versioned mz point execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MzTransportArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Shared causal graph.
    pub graph: AdmgWire,
    /// Canonical query with each source's selection targets and controllables.
    pub query: MzTransportQueryWire,
    /// Search limits the producer decided under. Provenance only.
    pub search_operations: usize,
    /// Search depth limit the producer decided under. Provenance only.
    pub search_depth: usize,
    /// Checked proof premises, including the search receipt.
    pub proof: MzTransportDerivationRecord,
    /// Expression arena of the checked derivation.
    pub expression: ExprArenaWire,
    /// `(regime, population)` of every cited regime: the source-specific bindings.
    pub bindings: Vec<(u32, String)>,
    /// Evidence catalog, including declared sampling and study identities.
    pub catalog: EvidenceCatalogWire,
    /// Target and source laws in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under. Provenance only.
    pub max_support_rows: usize,
    /// Target request.
    pub request: Vec<(u32, ValueWire)>,
    /// Evaluation operation limit. Provenance only.
    pub operation_limit: usize,
    /// Evaluation depth limit. Provenance only.
    pub depth_limit: usize,
    /// Point result.
    pub result: MzTransportPointWire,
    /// Interval bookkeeping.
    pub uncertainty: MzUncertaintyWire,
    /// Digest of the canonical premises: graph, query, proof and expression.
    pub premises_digest: String,
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedMzTransport {
    /// The re-decided, re-bound functional.
    pub functional: BoundMzTransportFunctional,
    /// The re-validated laws.
    pub data: ExactTransportData,
    /// The target request.
    pub request: Assignment,
    /// The recomputed point.
    pub distribution: ExactDistribution,
    /// The decoded artifact.
    pub wire: MzTransportArtifactWire,
}

fn premises_digest(
    graph: &AdmgWire,
    query: &MzTransportQueryWire,
    proof: &MzTransportDerivationRecord,
    expression: &ExprArenaWire,
) -> Result<String, IoError> {
    let mut graph = graph.clone();
    graph.directed.sort_unstable();
    graph.bidirected.sort_unstable();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("mz_transport_point_v1", graph, query, proof, expression),
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

fn bindings_of(functional: &BoundMzTransportFunctional) -> Vec<(u32, String)> {
    functional.cited_populations().into_iter().map(|(id, p)| (id.raw(), p.to_string())).collect()
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl MzTransportArtifactWire {
    /// Build an artifact from checked premises, a point and its bookkeeping.
    ///
    /// # Errors
    /// The premises do not encode, or the bookkeeping is inconsistent.
    #[allow(clippy::too_many_arguments)] // Every premise of the artifact, explicitly.
    pub fn checked(
        graph: &antecedent_graph::Admg,
        functional: &BoundMzTransportFunctional,
        search: SearchLimits,
        data: &ExactTransportData,
        request: &Assignment,
        limits: ExactEvaluationLimits,
        result: &ExactDistribution,
        uncertainty: MzUncertaintyWire,
    ) -> Result<Self, IoError> {
        let graph = admg_to_wire(graph)?;
        let derivation = functional.derivation();
        let query = MzTransportQueryWire::from_query(derivation.query());
        let proof = derivation.to_record();
        let expression = expr_arena_to_wire(derivation.arena())?;
        let premises_digest = premises_digest(&graph, &query, &proof, &expression)?;
        let wire = Self {
            version: MZ_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![MZ_TRANSPORT_ARTIFACT_FEATURE.into()],
            graph,
            query,
            search_operations: search.operations,
            search_depth: search.depth,
            proof,
            expression,
            bindings: bindings_of(functional),
            catalog: EvidenceCatalogWire::from_catalog(functional.catalog()),
            laws: canonical_laws(data)?,
            max_support_rows: data.max_support_rows(),
            request: request
                .entries()
                .iter()
                .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                .collect(),
            operation_limit: limits.operations,
            depth_limit: limits.depth,
            result: MzTransportPointWire::from_distribution(result),
            uncertainty,
            premises_digest,
        };
        wire.validate_shape()?;
        check_uncertainty(functional, data, &wire.uncertainty, &wire.result)?;
        Ok(wire)
    }

    /// The digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-decides and re-binds every premise.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        premises_digest(&self.graph, &self.query, &self.proof, &self.expression)
    }

    fn validate_shape(&self) -> Result<(), MzTransportArtifactError> {
        if self.required_features != [MZ_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(MzTransportArtifactError::UnsupportedSemantics("required features"));
        }
        if self.result.atoms.len() != self.result.probabilities.len()
            || self.result.outcomes != self.query.outcomes
        {
            return Err(MzTransportArtifactError::UnsupportedSemantics("point result shape"));
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
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != MZ_TRANSPORT_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &MzTransportConsumeLimits,
    ) -> Result<(), MzTransportArtifactError> {
        let exceeded = MzTransportArtifactError::LimitsExceeded;
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
        if self.laws.iter().any(|law| {
            law.probabilities.len() > limits.max_law_cells
                || law.empirical_counts.as_ref().is_some_and(|c| c.len() > limits.max_law_cells)
        }) {
            return Err(exceeded("law cells"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then recompute
    /// the point. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, an evaluation refusal, a point that does not
    /// replay, or bookkeeping that contradicts the licensing decision.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: MzTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedMzTransport, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(MzTransportArtifactError::PremisesMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let query = wire.query.to_query();
        let catalog = wire.catalog.to_catalog()?;
        let arena = expr_arena_from_wire(&wire.expression)?;
        let derivation = MzTransportDerivation::from_record_checked(
            &graph,
            &query,
            &catalog,
            &wire.proof,
            &arena,
            limits.search,
            ctx,
        )
        .map_err(MzTransportArtifactError::ProofMismatch)?;
        let functional = bind_mz_transport_catalog(&graph, &derivation, &catalog)
            .map_err(MzTransportArtifactError::CatalogBinding)?;
        if bindings_of(&functional) != wire.bindings {
            return Err(MzTransportArtifactError::BindingMismatch.into());
        }
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| MzTransportArtifactError::LawInvalid(error.to_string()))?;
        let data = ExactTransportData::try_new(laws, limits.max_support_rows)
            .map_err(|error| MzTransportArtifactError::LawInvalid(error.to_string()))?;
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        let distribution = antecedent_estimate::evaluate_exact_mz_transport(
            &functional,
            data.clone(),
            request.clone(),
            limits.evaluation,
            ctx,
        )
        .map_err(|error| antecedent_estimate::refuse_eval(&error))?;
        if !wire.result.replays(&MzTransportPointWire::from_distribution(&distribution)) {
            return Err(MzTransportArtifactError::PointMismatch.into());
        }
        check_uncertainty(&functional, &data, &wire.uncertainty, &wire.result)?;
        Ok(ConsumedMzTransport { functional, data, request, distribution, wire })
    }

    /// [`Self::consume_with_limits`] under the default consumer limits.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<ConsumedMzTransport, IoError> {
        Self::consume_with_limits(bytes, MzTransportConsumeLimits::default(), ctx)
    }
}

/// Re-derive the interval licensing decision and check the stored bookkeeping
/// against it: exact laws are point-only; counted laws with undeclared or
/// unsupported dependence are withheld for exactly that reason; otherwise the
/// bootstrap replicate rule decides between a nominal interval and a withheld one.
fn check_uncertainty(
    functional: &BoundMzTransportFunctional,
    data: &ExactTransportData,
    stored: &MzUncertaintyWire,
    point: &MzTransportPointWire,
) -> Result<(), MzTransportArtifactError> {
    let bad = MzTransportArtifactError::UncertaintyMismatch;
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return if *stored == MzUncertaintyWire::point_only() {
            Ok(())
        } else {
            Err(bad("exact laws carry no interval"))
        };
    }
    if let Err(reason) = antecedent_estimate::mz_sampling_dependence(functional) {
        return if stored.status == MZ_WITHHELD
            && stored.reason == reason
            && stored.mean_intervals.is_empty()
        {
            Ok(())
        } else {
            Err(bad("interval must be withheld for the declared sampling"))
        };
    }
    if stored.replicates_ok + stored.replicates_failed != stored.replicates_requested {
        return Err(bad("replicate accounting"));
    }
    let withheld_for = |reason: &str| {
        stored.status == MZ_WITHHELD && stored.reason == reason && stored.mean_intervals.is_empty()
    };
    if let Err(reason) = antecedent_estimate::ReplicatePolicy::BOOTSTRAP.decide(
        stored.replicates_requested,
        stored.replicates_ok,
        stored.replicates_failed,
    ) {
        return if withheld_for(reason) {
            Ok(())
        } else {
            Err(bad("replicate rule withholds this interval"))
        };
    }
    // Enough replicates: a nominal interval, or one withheld because the
    // replicate columns were not numerically summarizable.
    if withheld_for(antecedent_estimate::INTERVAL_NUMERICAL_FAILURE) {
        return Ok(());
    }
    let outcomes = stored.mean_intervals.iter().map(|(v, _, _)| *v).collect::<Vec<_>>();
    if stored.status != MZ_NOMINAL_INTERVAL
        || stored.reason != antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED
        || stored.method.as_deref() != Some(antecedent_estimate::PERCENTILE_BOOTSTRAP)
        || !stored.coverage_target.is_some_and(|c| c.is_finite() && 0.0 < c && c < 1.0)
        || outcomes != point.outcomes
        || stored
            .mean_intervals
            .iter()
            .any(|(_, lo, hi)| !(lo.is_finite() && hi.is_finite() && lo <= hi))
    {
        return Err(bad("nominal interval bookkeeping"));
    }
    Ok(())
}
