//! Independent point-result artifacts for single-source z transport.
//!
//! Format version 2. A consumer trusts nothing in the artifact: the proof is
//! re-verified, the catalog re-bound, the laws re-validated and re-counted, the
//! program re-checked and the point recomputed, all under the consumer's own
//! limits. The limits an artifact records are provenance; a stored limit larger
//! than the consumer's refuses. Version 1 artifacts carried an optional program
//! and a `Debug`-rendered graph signature and are refused as unsupported.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{
        ExprArenaWire, FunctionalProgramWire, expr_arena_from_wire, expr_arena_to_wire,
        functional_program_from_wire, functional_program_to_wire,
    },
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{
    ExecutionContext, IdentityDomain, InterventionAssignment, TransportOutcomeKind, VariableId,
};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData, FunctionalProgram,
    ProgramLimits,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    BoundZTransportFunctional, IdentificationError, SidLimits, ZTransportDerivation,
    ZTransportDerivationRecord, ZTransportQuery, bind_z_transport_catalog,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const Z_TRANSPORT_ARTIFACT_VERSION: u32 = 2;
/// The feature marker of the accepted format.
pub const Z_TRANSPORT_ARTIFACT_FEATURE: &str = "checked_z_transport_point_v2";
/// The one interval status this point format reports.
pub const Z_TRANSPORT_NO_INTERVAL: &str = "no_interval_reported";

/// Why a z-transport artifact was refused by a consumer check. Callers match the
/// kind; they never parse the message.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ZTransportArtifactError {
    /// The feature marker, interval status or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A limit the artifact recorded exceeds the consumer's, or a stored
    /// collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored derivation does not verify against the stored graph and query.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// A stored law is not a valid exact law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// The stored program does not match the checked proof and binding.
    #[error("program mismatch: {0}")]
    ProgramMismatch(&'static str),
    /// The recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
}

impl ZTransportArtifactError {
    /// The stable transport outcome kind this refusal maps to.
    #[must_use]
    pub const fn transport_outcome_kind(&self) -> TransportOutcomeKind {
        match self {
            Self::ProofMismatch(error) | Self::CatalogBinding(error) => {
                error.transport_outcome_kind()
            }
            Self::LimitsExceeded(_) => TransportOutcomeKind::BudgetCancel,
            Self::UnsupportedSemantics(_)
            | Self::LawInvalid(_)
            | Self::ProgramMismatch(_)
            | Self::PointMismatch
            | Self::PremisesMismatch => TransportOutcomeKind::InvalidInput,
        }
    }
}

/// Bounds a consumer imposes on an artifact it replays. Nothing the artifact
/// stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct ZTransportConsumeLimits {
    /// Exact-evaluation operation and depth limits of the replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
}

impl Default for ZTransportConsumeLimits {
    fn default() -> Self {
        Self {
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
        }
    }
}

impl ZTransportConsumeLimits {
    /// The identification limits the replay verifies the proof under.
    #[must_use]
    pub const fn identification(&self) -> SidLimits {
        SidLimits { steps: self.evaluation.operations, depth: self.evaluation.depth }
    }
}

/// Versioned z-transport point execution, including every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Causal graph.
    pub graph: AdmgWire,
    /// Selection targets.
    pub selections: Vec<u32>,
    /// The theorem query.
    pub query: ZTransportQueryWire,
    /// Target effect request.
    pub request: Vec<(u32, ValueWire)>,
    /// Operation limit the producer evaluated under. Provenance only.
    pub operation_limit: usize,
    /// Depth limit the producer evaluated under. Provenance only.
    pub depth_limit: usize,
    /// Checked proof premises.
    pub proof: ZTransportDerivationRecord,
    /// Expression nodes produced by the checked derivation.
    pub expression: ExprArenaWire,
    /// Owned, structurally checked provider-bound program.
    pub program: FunctionalProgramWire,
    /// Evidence catalog binding factor authority.
    pub catalog: EvidenceCatalogWire,
    /// Joint source and target laws with provider snapshots, in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under. Provenance only.
    pub max_support_rows: usize,
    /// Point result recomputed by independent consumers.
    pub result: ZTransportPointWire,
    /// This format never reports an interval.
    pub interval_status: String,
    /// Semantic digest of the canonical premises: sorted graph, selections,
    /// query, proof record and expression. Stable across re-encodings of the
    /// same premises, unlike a digest of the artifact bytes.
    pub premises_digest: String,
}

/// Portable graph/query coordinates and exact intervention values.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatment variables.
    pub treatments: Vec<u32>,
    /// Controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment assignment.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
}

impl ZTransportQueryWire {
    /// Encode a query.
    #[must_use]
    pub fn from_query(query: &ZTransportQuery) -> Self {
        Self {
            outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: query.treatments.iter().map(|v| v.raw()).collect(),
            controllable: query.controllable.iter().map(|v| v.raw()).collect(),
            experiment_assignment: query
                .experiment_assignment
                .iter()
                .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                .collect(),
            source: query.source.to_string(),
            target: query.target.to_string(),
        }
    }

    /// Decode a query. Validation happens against the diagram it is used with.
    #[must_use]
    pub fn to_query(&self) -> ZTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        ZTransportQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            controllable: ids(&self.controllable).into(),
            experiment_assignment: self
                .experiment_assignment
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect::<Vec<_>>()
                .into(),
            source: Arc::from(self.source.as_str()),
            target: Arc::from(self.target.as_str()),
        }
    }
}

/// Complete point distribution in canonical atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZTransportPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl ZTransportPointWire {
    fn from_distribution(distribution: &ExactDistribution) -> Self {
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

    /// Whether a recomputed point replays this stored point.
    ///
    /// Equality policy: the replay runs the same deterministic exact evaluator
    /// on the same laws, so probabilities are compared bit for bit; a stored or
    /// recomputed probability that is not finite fails closed, and NaN never
    /// equals itself.
    fn replays(&self, recomputed: &Self) -> bool {
        self.outcomes == recomputed.outcomes
            && self.atoms == recomputed.atoms
            && self.probabilities.len() == recomputed.probabilities.len()
            && self.probabilities.iter().zip(&recomputed.probabilities).all(|(stored, fresh)| {
                stored.is_finite() && fresh.is_finite() && stored.to_bits() == fresh.to_bits()
            })
    }
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedZTransport {
    /// The declared selection diagram.
    pub diagram: SelectionDiagram,
    /// The re-verified, re-bound functional.
    pub functional: BoundZTransportFunctional,
    /// The re-validated laws.
    pub data: ExactTransportData,
    /// The target request.
    pub request: Assignment,
    /// The limits the replay evaluated under (the consumer's).
    pub limits: ExactEvaluationLimits,
    /// The recomputed point.
    pub distribution: ExactDistribution,
    /// The decoded artifact.
    pub wire: ZTransportArtifactWire,
}

/// Semantic digest of the premises an artifact replays.
fn premises_digest(
    graph: &AdmgWire,
    selections: &[u32],
    query: &ZTransportQueryWire,
    proof: &ZTransportDerivationRecord,
    expression: &ExprArenaWire,
) -> Result<String, IoError> {
    let mut graph = graph.clone();
    graph.directed.sort_unstable();
    graph.bidirected.sort_unstable();
    let mut selections = selections.to_vec();
    selections.sort_unstable();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("z_transport_point_v2", graph, selections, query, proof, expression),
    )?
    .to_hex())
}

/// Sort laws into a canonical order so the same evidence supplied in another
/// order produces the same artifact.
fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            let key = crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions));
            key.map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl ZTransportArtifactWire {
    /// Validate and construct an artifact from checked premises and a point result.
    ///
    /// # Errors
    /// The premises do not encode, or the program does not match the functional.
    pub fn checked(
        diagram: &SelectionDiagram,
        functional: &BoundZTransportFunctional,
        data: &ExactTransportData,
        request: &Assignment,
        limits: ExactEvaluationLimits,
        result: &ExactDistribution,
        program: &FunctionalProgram,
    ) -> Result<Self, IoError> {
        let graph = admg_to_wire(diagram.causal_graph())?;
        let selections = diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>();
        let query = ZTransportQueryWire::from_query(functional.derivation().query());
        let proof = functional.derivation().to_record();
        let expression = expr_arena_to_wire(functional.derivation().arena())?;
        let premises_digest = premises_digest(&graph, &selections, &query, &proof, &expression)?;
        let wire = Self {
            version: Z_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![Z_TRANSPORT_ARTIFACT_FEATURE.into()],
            graph,
            selections,
            query,
            request: request
                .entries()
                .iter()
                .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                .collect(),
            operation_limit: limits.operations,
            depth_limit: limits.depth,
            proof,
            expression,
            program: functional_program_to_wire(program)?,
            catalog: EvidenceCatalogWire::from_catalog(functional.catalog()),
            laws: canonical_laws(data)?,
            max_support_rows: data.max_support_rows(),
            result: ZTransportPointWire::from_distribution(result),
            interval_status: Z_TRANSPORT_NO_INTERVAL.into(),
            premises_digest,
        };
        wire.validate_shape()?;
        validate_functional_program(program, functional)?;
        Ok(wire)
    }

    fn validate_shape(&self) -> Result<(), ZTransportArtifactError> {
        if self.required_features != [Z_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(ZTransportArtifactError::UnsupportedSemantics("required features"));
        }
        if self.interval_status != Z_TRANSPORT_NO_INTERVAL {
            return Err(ZTransportArtifactError::UnsupportedSemantics("interval status"));
        }
        if self.result.atoms.len() != self.result.probabilities.len()
            || self.result.outcomes != self.query.outcomes
        {
            return Err(ZTransportArtifactError::UnsupportedSemantics("point result shape"));
        }
        Ok(())
    }

    /// Encode the versioned artifact as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode an artifact, refusing any version other than the accepted one
    /// before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`] for another format version; a decoding
    /// failure otherwise.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != Z_TRANSPORT_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Rebuild checked proof and provider objects under the consumer's limits
    /// without executing the point formula.
    ///
    /// # Errors
    /// An unsupported version, a stored limit above the consumer's, a proof,
    /// binding, law or program that does not check, or a premises digest that
    /// does not match.
    pub fn reconstruct(
        bytes: &[u8],
        limits: ZTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            SelectionDiagram,
            BoundZTransportFunctional,
            ExactTransportData,
            Assignment,
            ExactEvaluationLimits,
            Self,
        ),
        IoError,
    > {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        let expected = premises_digest(
            &wire.graph,
            &wire.selections,
            &wire.query,
            &wire.proof,
            &wire.expression,
        )?;
        if expected != wire.premises_digest {
            return Err(ZTransportArtifactError::PremisesMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            Arc::<[VariableId]>::from(
                wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            ),
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
        let query = wire.query.to_query();
        let arena = expr_arena_from_wire(&wire.expression)?;
        let proof = ZTransportDerivation::from_record_checked(
            &diagram,
            &query,
            &wire.proof,
            arena,
            limits.identification(),
            ctx,
        )
        .map_err(ZTransportArtifactError::ProofMismatch)?;
        let catalog = wire.catalog.to_catalog()?;
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog)
            .map_err(ZTransportArtifactError::CatalogBinding)?;
        let program = functional_program_from_wire(&wire.program, ProgramLimits::default())?;
        if program.mapping().source != proof.root()
            || program.mapping().executable != functional.root()
            || program.arena().len() != functional.arena().len()
        {
            return Err(ZTransportArtifactError::ProgramMismatch(
                "program roots do not match the checked proof and binding",
            )
            .into());
        }
        validate_functional_program(&program, &functional)?;
        program
            .compile()
            .map_err(|error| IoError::from(antecedent_estimate::refuse_eval(&error)))?;
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ZTransportArtifactError::LawInvalid(error.to_string()))?;
        let data = ExactTransportData::try_new(laws, limits.max_support_rows)
            .map_err(|error| ZTransportArtifactError::LawInvalid(error.to_string()))?;
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        Ok((diagram, functional, data, request, limits.evaluation, wire))
    }

    fn check_limits(
        &self,
        limits: &ZTransportConsumeLimits,
    ) -> Result<(), ZTransportArtifactError> {
        if self.operation_limit > limits.evaluation.operations {
            return Err(ZTransportArtifactError::LimitsExceeded("operation limit"));
        }
        if self.depth_limit > limits.evaluation.depth {
            return Err(ZTransportArtifactError::LimitsExceeded("depth limit"));
        }
        if self.max_support_rows > limits.max_support_rows {
            return Err(ZTransportArtifactError::LimitsExceeded("support rows"));
        }
        if self.laws.len() > limits.max_laws {
            return Err(ZTransportArtifactError::LimitsExceeded("law count"));
        }
        if self.laws.iter().any(|law| {
            law.probabilities.len() > limits.max_law_cells
                || law.empirical_counts.as_ref().is_some_and(|c| c.len() > limits.max_law_cells)
        }) {
            return Err(ZTransportArtifactError::LimitsExceeded("law cells"));
        }
        Ok(())
    }

    /// Decode, recheck the proof and evidence bindings under the consumer's
    /// limits, then recompute the point result. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, an evaluation refusal, or a point that does
    /// not replay.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: ZTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedZTransport, IoError> {
        let (diagram, functional, data, request, evaluation, wire) =
            Self::reconstruct(bytes, limits, ctx)?;
        let plan = antecedent_estimate::prepare_exact_z_transport(
            &functional,
            data.clone(),
            request.clone(),
            evaluation,
            ctx,
        )
        .map_err(|error| antecedent_estimate::refuse_eval(&error))?;
        let distribution =
            plan.evaluate(ctx).map_err(|error| antecedent_estimate::refuse_eval(&error))?;
        if !wire.result.replays(&ZTransportPointWire::from_distribution(&distribution)) {
            return Err(ZTransportArtifactError::PointMismatch.into());
        }
        Ok(ConsumedZTransport {
            diagram,
            functional,
            data,
            request,
            limits: evaluation,
            distribution,
            wire,
        })
    }

    /// [`Self::consume_with_limits`] under the default consumer limits.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<(SelectionDiagram, ExactDistribution), IoError> {
        let consumed = Self::consume_with_limits(bytes, ZTransportConsumeLimits::default(), ctx)?;
        Ok((consumed.diagram, consumed.distribution))
    }
}

/// The schema name of a z-transport program variable: one name per graph
/// coordinate, shared by every environment that declares it.
#[must_use]
pub fn program_variable_name(variable: VariableId) -> String {
    format!("v{}", variable.raw())
}

fn validate_functional_program(
    program: &FunctionalProgram,
    functional: &BoundZTransportFunctional,
) -> Result<(), IoError> {
    if program.mapping().source != functional.derivation().root()
        || program.mapping().executable != functional.root()
    {
        return Err(ZTransportArtifactError::ProgramMismatch(
            "program roots do not match the checked formula",
        )
        .into());
    }
    if program.arena() != functional.arena() {
        return Err(ZTransportArtifactError::ProgramMismatch(
            "program arena does not match the bound formula",
        )
        .into());
    }
    let expected = functional
        .catalog()
        .environments
        .iter()
        .flat_map(|environment| environment.variables.iter())
        .map(|coordinate| (coordinate.variable.raw(), program_variable_name(coordinate.variable)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let supplied = program
        .schema()
        .variables()
        .map(|(id, variable)| (id.raw(), variable.name.to_string()))
        .collect::<std::collections::BTreeMap<_, _>>();
    if supplied != expected {
        return Err(ZTransportArtifactError::ProgramMismatch(
            "program schema does not match the evidence catalog",
        )
        .into());
    }
    Ok(())
}
