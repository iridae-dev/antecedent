//! Independent point-result artifacts for exact binary observation recovery
//! (2.2B X10): graph-licensed recovery, not MAR/IPCW.
//!
//! Format version 1. A consumer trusts nothing in the artifact. Under its own
//! maxima (a stored limit above them refuses before any work) it re-decides the
//! recovery from the stored m-graph, roles, catalog and effect query under the
//! producer's stored limits, runs the independent formula checker on the stored
//! derivation record and arena, and accepts only an identical record and
//! expression (formula and downstream effect). It re-validates the stored
//! observed pattern law, recomputes the recovered law and every effect point
//! and compares them bit for bit. The verified identity is two digests: the
//! premises digest (m-graph, roles, effect query, derivation record, both
//! expression arenas, evaluation limits, requests and the variable-name
//! mapping) and the data-identity digest (the catalog's snapshot bindings and
//! the observed law's population, regime, snapshot, axes and every cell bit for
//! bit).
//!
//! What replay does not protect against: a producer that supplies a fabricated
//! observed pattern law consistently (the consumer checks the derivation and the
//! arithmetic, not that the table describes real data), and the m-graph's
//! untestable premises (for example that no `X_i -> R_i` edge exists): they are
//! declared and bound into the premises digest, never verified from data.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{EvidenceCatalog, ExecutionContext, IdentityDomain, RegimeId, VariableId};
use antecedent_estimate::{RecoveredLaw, evaluate_exact_recovery, evaluate_recovered_effect};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ObservationRecoveryQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery,
    RecoveryDerivation, RecoveryDerivationRecord, RecoveryDetail, RecoveryError, RecoveryLimits,
    verify_observation_recovery,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const RECOVERY_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const RECOVERY_ARTIFACT_FEATURE: &str = "checked_observation_recovery_point_v1";
/// The only uncertainty claim of this route.
pub const RECOVERY_POINT_ONLY: &str = "point_only";

/// Why a recovery artifact was refused. Every kind carries a registered reason
/// code and the X10 detail it maps to.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RecoveryArtifactError {
    /// The feature marker, claim or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-decided derivation does not reproduce the stored one.
    #[error("derivation does not verify: {0}")]
    ProofMismatch(RecoveryError),
    /// A stored law does not validate.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// The recomputed recovered law differs from the stored one.
    #[error("recovered law does not replay")]
    RecoveredMismatch,
    /// A recomputed effect point differs from the stored one.
    #[error("effect point does not replay")]
    EffectMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the verified mapping.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
    /// Recomputation refused with the route's own typed reason.
    #[error("{0}")]
    Refused(RecoveryError),
}

impl RecoveryArtifactError {
    /// `(reason code, detail)` of this refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        let detail = match self {
            Self::LimitsExceeded(_) => RecoveryDetail::BoundsExceeded,
            Self::ProofMismatch(error) | Self::Refused(error)
                if error.detail != RecoveryDetail::InvalidDerivation =>
            {
                error.detail
            }
            _ => RecoveryDetail::InvalidDerivation,
        };
        (detail.reason_code(), detail.detail())
    }
}

impl From<RecoveryArtifactError> for IoError {
    fn from(error: RecoveryArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: recovery artifact: {error}") }
    }
}

/// Convert a route refusal into an io refusal that keeps its reason code.
#[must_use]
#[doc(hidden)]
pub fn recovery_io_error(error: &RecoveryError) -> IoError {
    IoError::Refused { code: error.reason_code(), message: error.to_string() }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct RecoveryConsumeLimits {
    /// Maxima of the stored decision limits.
    pub decision: RecoveryLimits,
    /// Maxima of the stored evaluation limits.
    pub evaluation: ExactEvaluationLimits,
    /// Most effect requests.
    pub max_requests: usize,
}

impl Default for RecoveryConsumeLimits {
    fn default() -> Self {
        Self {
            decision: RECOVERY_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            max_requests: 64,
        }
    }
}

/// Portable recovery query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecoveryQueryWire {
    /// Population.
    pub population: String,
    /// Observed regime.
    pub observed_regime: u32,
    /// `(variable, response, proxy)`, canonical order.
    pub partially_observed: Vec<(u32, u32, u32)>,
    /// Fully observed, sorted.
    pub fully_observed: Vec<u32>,
}

impl RecoveryQueryWire {
    /// Encode in canonical form.
    #[must_use]
    pub fn from_query(query: &ObservationRecoveryQuery) -> Self {
        let query = query.canonical();
        Self {
            population: query.population.to_string(),
            observed_regime: query.observed_regime.raw(),
            partially_observed: query
                .partially_observed
                .iter()
                .map(|p| (p.variable.raw(), p.response.raw(), p.proxy.raw()))
                .collect(),
            fully_observed: query.fully_observed.iter().map(|v| v.raw()).collect(),
        }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(&self) -> ObservationRecoveryQuery {
        let v = VariableId::from_raw;
        ObservationRecoveryQuery {
            population: Arc::from(self.population.as_str()),
            observed_regime: RegimeId::from_raw(self.observed_regime),
            partially_observed: self
                .partially_observed
                .iter()
                .map(|(x, r, p)| PartiallyObserved {
                    variable: v(*x),
                    response: v(*r),
                    proxy: v(*p),
                })
                .collect::<Vec<_>>()
                .into(),
            fully_observed: self.fully_observed.iter().copied().map(v).collect::<Vec<_>>().into(),
        }
    }
}

/// Portable downstream effect query: the causal graph by variable ids.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecoveredEffectWire {
    /// Graph nodes (variable ids) in dense order.
    pub nodes: Vec<u32>,
    /// Directed edges by variable id, sorted.
    pub directed: Vec<(u32, u32)>,
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
}

impl RecoveredEffectWire {
    /// Encode.
    ///
    /// # Errors
    /// A non-static node or a bidirected edge (never a recovered handoff).
    pub fn from_query(effect: &RecoveredEffectQuery) -> Result<Self, IoError> {
        let graph = &effect.graph;
        let var = |d: DenseNodeId| match graph.nodes()[d.as_usize()] {
            NodeRef::Static(v) => Ok(v.raw()),
            _ => Err(IoError::Convert("the effect graph must have static nodes".into())),
        };
        if graph.has_bidirected() {
            return Err(IoError::Convert("the effect graph must be a DAG".into()));
        }
        let mut nodes = Vec::new();
        let mut directed = Vec::new();
        for i in 0..graph.node_count() {
            let d = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
            nodes.push(var(d)?);
            for child in graph.children(d) {
                directed.push((var(d)?, var(*child)?));
            }
        }
        directed.sort_unstable();
        Ok(Self {
            nodes,
            directed,
            outcomes: effect.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: effect.treatments.iter().map(|v| v.raw()).collect(),
        })
    }

    /// Decode.
    ///
    /// # Errors
    /// Unknown endpoints or a cycle.
    pub fn to_query(&self) -> Result<RecoveredEffectQuery, IoError> {
        let mut graph = Admg::empty();
        for n in &self.nodes {
            graph
                .add_node(NodeRef::Static(VariableId::from_raw(*n)))
                .map_err(crate::error::convert_err)?;
        }
        let dense = |v: u32| {
            self.nodes
                .iter()
                .position(|n| *n == v)
                .and_then(|i| u32::try_from(i).ok())
                .map(DenseNodeId::from_raw)
                .ok_or_else(|| IoError::Convert("effect edge endpoint is not a node".into()))
        };
        for (a, b) in &self.directed {
            graph.insert_directed(dense(*a)?, dense(*b)?).map_err(crate::error::convert_err)?;
        }
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        Ok(RecoveredEffectQuery {
            graph,
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
        })
    }
}

/// A dense law's axes and cells.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecoveryTableWire {
    /// `(variable, levels)` per axis; last axis fastest.
    pub axes: Vec<(u32, Vec<ValueWire>)>,
    /// Cell probabilities.
    pub probabilities: Vec<f64>,
}

impl RecoveryTableWire {
    fn of_law(law: &RecoveredLaw) -> Self {
        Self {
            axes: law
                .law()
                .axes()
                .iter()
                .map(|a| (a.variable.raw(), a.values.iter().map(ValueWire::from_value).collect()))
                .collect(),
            probabilities: law.law().probabilities().to_vec(),
        }
    }

    fn of_distribution(distribution: &ExactDistribution) -> Self {
        Self {
            axes: distribution
                .outcomes
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let column =
                        distribution.atoms.iter().map(|atom| ValueWire::from_value(&atom[i]));
                    (v.raw(), column.collect())
                })
                .collect(),
            probabilities: distribution.probabilities.to_vec(),
        }
    }

    /// Bit-exact replay; non-finite values fail closed.
    fn replays(&self, fresh: &Self) -> bool {
        self.axes == fresh.axes
            && self.probabilities.len() == fresh.probabilities.len()
            && self
                .probabilities
                .iter()
                .zip(&fresh.probabilities)
                .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
    }
}

/// Versioned observation-recovery result with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The m-graph (static nodes `0..n`).
    pub graph: AdmgWire,
    /// Canonical roles.
    pub query: RecoveryQueryWire,
    /// Downstream effect query, when one was identified.
    pub effect: Option<RecoveredEffectWire>,
    /// Checked derivation record, including the search receipt and stored limits.
    pub derivation: RecoveryDerivationRecord,
    /// Recovery formula arena.
    pub expression: ExprArenaWire,
    /// Downstream effect arena.
    pub effect_expression: Option<ExprArenaWire>,
    /// Evidence catalog naming the observed pattern law.
    pub catalog: EvidenceCatalogWire,
    /// The observed pattern law.
    pub observed_law: ExactLawWire,
    /// The recovered law.
    pub recovered: RecoveryTableWire,
    /// Canonical identity of the recovered law's catalog descriptor
    /// (`origin=recovered:...`).
    pub recovered_identity: String,
    /// Effect requests.
    pub requests: Vec<Vec<(u32, ValueWire)>>,
    /// Effect point per request.
    pub effects: Vec<RecoveryTableWire>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// Variable name of every m-graph node, or empty.
    pub variable_names: Vec<String>,
    /// Always [`RECOVERY_POINT_ONLY`].
    pub uncertainty: String,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the data identity.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    query: &'a RecoveryQueryWire,
    effect: &'a Option<RecoveredEffectWire>,
    derivation: &'a RecoveryDerivationRecord,
    expression: &'a ExprArenaWire,
    effect_expression: &'a Option<ExprArenaWire>,
    evaluation: (usize, usize),
    requests: &'a [Vec<(u32, ValueWire)>],
    variable_names: &'a [String],
}

impl RecoveryArtifactWire {
    fn premises_view(&self) -> PremisesView<'_> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        PremisesView {
            tag: "observation_recovery_point_v1",
            graph,
            query: &self.query,
            effect: &self.effect,
            derivation: &self.derivation,
            expression: &self.expression,
            effect_expression: &self.effect_expression,
            evaluation: (self.operation_limit, self.depth_limit),
            requests: &self.requests,
            variable_names: &self.variable_names,
        }
    }
}

fn data_digest(catalog: &EvidenceCatalog, law: &ExactLawWire) -> Result<String, IoError> {
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
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &(
            "observation_recovery_data_v1",
            bindings,
            (&law.population, law.regime, &law.snapshot),
            // The table's content, bit for bit: data identity is never left to replay.
            (&law.axes, law.probabilities.iter().map(|p| p.to_bits()).collect::<Vec<u64>>()),
        ),
    )?
    .to_hex())
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
pub struct RecoveryArtifactInput<'a> {
    /// The m-graph the derivation was decided on (static nodes `0..n`).
    pub graph: &'a Admg,
    /// The downstream effect query, when one was identified.
    pub effect: Option<&'a RecoveredEffectQuery>,
    /// The checked derivation.
    pub derivation: &'a RecoveryDerivation,
    /// The catalog naming the observed pattern law.
    pub catalog: &'a EvidenceCatalog,
    /// The observed pattern law.
    pub observed: &'a antecedent_expr::ExactDiscreteLaw,
    /// The recovered law.
    pub recovered: &'a RecoveredLaw,
    /// Effect requests.
    pub requests: &'a [Assignment],
    /// Effect points, one per request.
    pub effects: &'a [ExactDistribution],
    /// Evaluation limits.
    pub limits: ExactEvaluationLimits,
    /// Variable names of every m-graph node, or empty.
    pub variable_names: &'a [String],
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedRecovery {
    /// The re-decided derivation.
    pub derivation: RecoveryDerivation,
    /// The recomputed recovered law.
    pub recovered: RecoveredLaw,
    /// The recomputed effect points.
    pub effects: Vec<ExactDistribution>,
    /// The decoded artifact.
    pub wire: RecoveryArtifactWire,
}

impl RecoveryArtifactWire {
    /// Build an artifact from checked premises and results.
    ///
    /// # Errors
    /// Premises that do not encode, an m-graph whose nodes are not `0..n`, or
    /// inconsistent results.
    pub fn checked(input: &RecoveryArtifactInput<'_>) -> Result<Self, IoError> {
        for (i, node) in input.graph.nodes().iter().enumerate() {
            if *node
                != NodeRef::Static(VariableId::from_raw(
                    u32::try_from(i).map_err(|_| IoError::TooLarge)?,
                ))
            {
                return Err(IoError::Convert("the m-graph must have static nodes 0..n".into()));
            }
        }
        // Producer-side bounds: nothing is exported that a default consumer refuses.
        let defaults = RecoveryConsumeLimits::default();
        if input.requests.len() > defaults.max_requests
            || input.observed.probabilities().len()
                > antecedent_identify::RECOVERY_MAX_OBSERVED_CELLS
            || input.limits.operations > defaults.evaluation.operations
            || input.limits.depth > defaults.evaluation.depth
        {
            return Err(RecoveryArtifactError::LimitsExceeded("export bounds").into());
        }
        let derivation = input.derivation;
        let observed_law = ExactLawWire::from_law(input.observed);
        let mut wire = Self {
            version: RECOVERY_ARTIFACT_VERSION,
            required_features: vec![RECOVERY_ARTIFACT_FEATURE.into()],
            graph: admg_to_wire(input.graph)?,
            query: RecoveryQueryWire::from_query(derivation.query()),
            effect: input.effect.map(RecoveredEffectWire::from_query).transpose()?,
            derivation: derivation.record().clone(),
            expression: expr_arena_to_wire(derivation.arena())?,
            effect_expression: derivation
                .effect()
                .map(|e| expr_arena_to_wire(e.arena()))
                .transpose()?,
            catalog: EvidenceCatalogWire::from_catalog(input.catalog),
            data_digest: data_digest(input.catalog, &observed_law)?,
            observed_law,
            recovered: RecoveryTableWire::of_law(input.recovered),
            recovered_identity: input.recovered.descriptor().canonical_identity(),
            requests: input.requests.iter().map(request_wire).collect(),
            effects: input.effects.iter().map(RecoveryTableWire::of_distribution).collect(),
            operation_limit: input.limits.operations,
            depth_limit: input.limits.depth,
            variable_names: input.variable_names.to_vec(),
            uncertainty: RECOVERY_POINT_ONLY.into(),
            premises_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The digest the stored premises should carry. Recomputing it grants nothing:
    /// a consumer still re-decides and recomputes every premise.
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
    /// The catalog does not decode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_digest(&self.catalog.to_catalog()?, &self.observed_law)
    }

    fn validate_shape(&self) -> Result<(), RecoveryArtifactError> {
        let unsupported = RecoveryArtifactError::UnsupportedSemantics;
        if self.required_features != [RECOVERY_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        if self.uncertainty != RECOVERY_POINT_ONLY {
            return Err(unsupported("this route publishes points only"));
        }
        if self.effects.len() != self.requests.len()
            || (self.effect.is_none() && !self.requests.is_empty())
            || self.effect.is_some() != self.effect_expression.is_some()
        {
            return Err(unsupported(
                "one effect point per request, and requests only with an effect",
            ));
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
    /// [`RecoveryArtifactError::NamesMismatch`] unless `names` equals the mapping.
    pub fn check_variable_names(&self, names: &[String]) -> Result<(), RecoveryArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(RecoveryArtifactError::NamesMismatch)
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
        if peek.version != RECOVERY_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(&self, limits: &RecoveryConsumeLimits) -> Result<(), RecoveryArtifactError> {
        let exceeded = RecoveryArtifactError::LimitsExceeded;
        let receipt = &self.derivation.receipt;
        if receipt.operations_limit > limits.decision.search.operations {
            return Err(exceeded("search operation limit"));
        }
        if receipt.depth_limit > limits.decision.search.depth {
            return Err(exceeded("search depth limit"));
        }
        if receipt.memory_bytes > limits.decision.memory_bytes {
            return Err(exceeded("search memory cap"));
        }
        if self.operation_limit > limits.evaluation.operations {
            return Err(exceeded("operation limit"));
        }
        if self.depth_limit > limits.evaluation.depth {
            return Err(exceeded("depth limit"));
        }
        if self.requests.len() > limits.max_requests {
            return Err(exceeded("request count"));
        }
        if self.observed_law.probabilities.len() > antecedent_identify::RECOVERY_MAX_OBSERVED_CELLS
        {
            return Err(exceeded("observed-law cells"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then recompute
    /// the recovered law and every effect point. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, with the X10 reason code and detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: RecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedRecovery, IoError> {
        Self::consume_typed(bytes, limits, ctx).map_err(IoError::from)
    }

    /// [`Self::consume_with_limits`] with the typed refusal kind, for callers and
    /// tests that match it.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    #[doc(hidden)]
    pub fn consume_typed(
        bytes: &[u8],
        limits: RecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedRecovery, RecoveryArtifactError> {
        let unsupported = |_: IoError| RecoveryArtifactError::UnsupportedSemantics("decode");
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => RecoveryArtifactError::UnsupportedSemantics("shape"),
            other => unsupported(other),
        })?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest().map_err(unsupported)? != wire.premises_digest {
            return Err(RecoveryArtifactError::PremisesMismatch);
        }
        let catalog = wire.catalog.to_catalog().map_err(unsupported)?;
        if data_digest(&catalog, &wire.observed_law).map_err(unsupported)? != wire.data_digest {
            return Err(RecoveryArtifactError::DataIdentityMismatch);
        }
        let graph = admg_from_wire(&wire.graph).map_err(unsupported)?;
        let query = wire.query.to_query();
        let effect = wire
            .effect
            .as_ref()
            .map(RecoveredEffectWire::to_query)
            .transpose()
            .map_err(unsupported)?;
        let arena = expr_arena_from_wire(&wire.expression).map_err(unsupported)?;
        let effect_arena = wire
            .effect_expression
            .as_ref()
            .map(expr_arena_from_wire)
            .transpose()
            .map_err(unsupported)?;
        let derivation = verify_observation_recovery(
            &graph,
            &query,
            &catalog,
            effect.as_ref(),
            &wire.derivation,
            &arena,
            effect_arena.as_ref(),
            limits.decision,
            ctx,
        )
        .map_err(RecoveryArtifactError::ProofMismatch)?;
        let observed = wire
            .observed_law
            .to_law()
            .map_err(|e| RecoveryArtifactError::LawInvalid(e.to_string()))?;
        let recovered = evaluate_exact_recovery(&derivation, &observed, ctx)
            .map_err(RecoveryArtifactError::Refused)?;
        if !wire.recovered.replays(&RecoveryTableWire::of_law(&recovered))
            || recovered.descriptor().canonical_identity() != wire.recovered_identity
        {
            return Err(RecoveryArtifactError::RecoveredMismatch);
        }
        let evaluation =
            ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit };
        let mut effects = Vec::with_capacity(wire.requests.len());
        for (request, stored) in wire.requests.iter().zip(&wire.effects) {
            let point = evaluate_recovered_effect(
                &derivation,
                &recovered,
                request_of(request),
                evaluation,
                ctx,
            )
            .map_err(RecoveryArtifactError::Refused)?;
            if !stored.replays(&RecoveryTableWire::of_distribution(&point)) {
                return Err(RecoveryArtifactError::EffectMismatch);
            }
            effects.push(point);
        }
        Ok(ConsumedRecovery { derivation, recovered, effects, wire })
    }
}
