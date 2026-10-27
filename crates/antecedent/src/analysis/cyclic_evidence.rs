// SPDX-License-Identifier: MIT OR Apache-2.0
//! Identity-bound evidence and decision composition of native cyclic products.
//! Numerical roots, posteriors, bootstrap laws and selected algebraic sets keep
//! their distinct meanings. No adapter performs implicit root selection, drops
//! failed draw slots or upgrades a specified model to graphical identification.
use super::{
    PreparedStudy,
    cyclic::{CyclicFamilyPreparedState, ExecutedCyclicFamily},
};
use antecedent_contract::{
    EvidenceBinding, EvidenceCalibration, EvidenceEnvelope, EvidenceOrigins, EvidencePort,
    EvidenceSupport, QuantityRole, ScientificQuantity, UncertaintyAxes, WorldTime,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SupportStatus};
use antecedent_io::{
    cyclic_family_artifact::{
        CyclicFamilyAnswer as A, CyclicFamilyReport, CyclicFamilyRequest as R,
        consume_cyclic_family_with_request, consume_cyclic_family_with_request_and_contract,
    },
    cyclic_system_artifact::{
        CyclicArtifactError, MAX_CYCLIC_ARTIFACT_BYTES, cyclic_decode_admission,
        cyclic_serialized_size,
    },
    distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    },
    quantity_wire::{DistributionMeaningWire, ScientificQuantityWire},
};
use antecedent_model::{equation_system::Interval, nonlinear_equilibrium::EquilibriumOutcome};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Declared physical variable identities and units; never inferred from a graph index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceSchema {
    /// Schema/model-system identity shared by the variables.
    pub id: String,
    /// Target population, explicit even for a specified physical system.
    pub population: String,
    /// Stable variable names and units in endogenous coordinate order.
    pub variables: Vec<(String, String)>,
}
/// Typed native output coordinate; array positions are never accepted as scientific labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicEvidenceSelector {
    /// Prespecified structural action and noise row, or its abducted factual response.
    Action {
        /// World index or stable declared world identity.
        world: usize,
        /// Aligned explicit disturbance row; zero for an abduction posterior.
        noise_row: usize,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Read the abducted factual response instead of a supplied-noise row.
        counterfactual: bool,
    },
    /// Aligned posterior/bootstrap response column. Noise row applies only to parameter/ bootstrap products.
    Response {
        /// World index or stable declared world identity.
        world: usize,
        /// Aligned explicit disturbance row; zero for an abduction posterior.
        noise_row: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// State of a deterministic ODE path.
    Ode {
        /// Observation-time index; None denotes stationarity.
        time: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// A stochastic-process moment; absence of time denotes stationarity.
    OuMean {
        /// Observation-time index; None denotes stationarity.
        time: Option<usize>,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// Row-major covariance of the same declared stochastic law.
    OuCovariance {
        /// Observation-time index; None denotes stationarity.
        time: Option<usize>,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Second dense endogenous coordinate of the covariance.
        other: usize,
    },
    /// A stationary stochastic moment after a named equation-intervention world.
    OuRegime {
        /// Declared equation-replacement world index.
        world: usize,
        /// Endogenous coordinate.
        variable: usize,
        /// Second coordinate for a covariance; None selects the mean.
        other: Option<usize>,
    },
    /// Certified enclosure of one root; this selector supplies no equilibrium probabilities.
    Root {
        /// Certified root enclosure index, with no probability implied.
        root: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// Outer solution set, including all unresolved boxes without a selection law.
    SolutionSet {
        /// Endogenous coordinate of the bounded equation system.
        variable: usize,
    },
    /// Theorem-derived finite-law functional evaluation.
    Functional {
        /// Retained identified-functional assignment index.
        assignment: usize,
    },
    /// An actual likelihood of a fitted model, or one finite polynomial candidate.
    Likelihood {
        /// Finite likelihood candidate index; None denotes the fitted continuous model.
        candidate: Option<usize>,
    },
    /// A graphical derivation/PAG. Qualitative evidence supplies no numeric outcome law.
    Graph,
    /// Coupled finite-law polynomial outcomes with mass and numerical enclosures retained.
    PolynomialWorld {
        /// World index or stable declared world identity.
        world: String,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// The interventional law within one recovered draw: its equilibrium mean,
    /// with its variance retained as a diagnostic. Not parameter uncertainty.
    RecoveredDraw {
        /// World position in the request.
        world: usize,
        /// Draw position in the law.
        draw: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// One typed world of a specified model, by scientific world id.
    World {
        /// Scientific world id.
        world: u32,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Observation-time index of a trajectory; absent for an equilibrium.
        time: Option<usize>,
    },
    /// One declared quantity of a fitted continuous-time family, across the
    /// replicates, draws, or delta-method law of its declared uncertainty method.
    DynamicQuantity {
        /// Position in the request's quantity list.
        quantity: usize,
    },
}
/// One coordinate's native scientific meaning and original aligned payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceColumn {
    /// Full semantic scientific coordinate, derived from the selector and schema.
    pub quantity: ScientificQuantityWire,
    /// Existing contract evidence-envelope identity.
    pub envelope_identity: String,
    /// Existing typed evidence-port spelling.
    pub port: String,
    /// Independent uncertainty axes; an empty list explicitly means none supplied.
    pub uncertainty_axes: Vec<String>,
    /// Original aligned samples; None retains a failed/undefined slot.
    pub samples: Vec<Option<f64>>,
    /// Numerical enclosures, never relabeled as confidence/credible intervals.
    pub numerical_enclosures: Vec<Option<Interval>>,
    /// Probability bounds when supplied by the finite joint selection/parameter law.
    pub mass_bounds: Option<Vec<antecedent_estimate::polynomial_scm::MassBounds>>,
    /// Actual semantic time and world identifiers, not merely a plotting index.
    pub world: String,
    /// Explicit initial-value time or stationary marker.
    pub time: String,
    /// Numerical/selection failures and qualitative/likelihood details.
    pub diagnostics: serde_json::Value,
}
/// Checked, replayable projection of a native cyclic source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceReport {
    /// Complete source, target schema and projection identity.
    pub identity: String,
    /// Independently retained original scientific-source identity.
    pub source_identity: String,
    /// The specified-model/identification distinction remains explicit.
    pub scope: String,
    /// Every original requested coordinate and unresolved sample slot.
    pub columns: Vec<CyclicEvidenceColumn>,
    /// Version-three native recovery identities; absent in historical version two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_origins: Option<antecedent_io::cyclic_family_artifact::CyclicRecoveryOrigins>,
    /// Actual retained source diagnostics, including convergence and failed mass.
    #[serde(with = "shared_family_report")]
    pub source: Arc<CyclicFamilyReport>,
}

// Preserve the historical inline report encoding while sharing its native storage.
// This does not require serde's optional `rc` feature or change the public wire format.
mod shared_family_report {
    use super::CyclicFamilyReport;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    pub fn serialize<S: serde::Serializer>(
        report: &Arc<CyclicFamilyReport>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        report.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<CyclicFamilyReport>, D::Error> {
        CyclicFamilyReport::deserialize(deserializer).map(Arc::new)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u16,
    #[serde(with = "source_bytes")]
    source: Vec<u8>,
    source_identity: String,
    schema: CyclicEvidenceSchema,
    selectors: Vec<CyclicEvidenceSelector>,
    report: CyclicEvidenceReport,
}
mod source_bytes {
    pub fn serialize<S: serde::Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        struct Bytes;
        impl serde::de::Visitor<'_> for Bytes {
            type Value = Vec<u8>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("bounded original cyclic source bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, value: &[u8]) -> Result<Vec<u8>, E> {
                if value.len() > super::MAX_CYCLIC_ARTIFACT_BYTES {
                    return Err(E::custom("source byte limit"));
                }
                Ok(value.to_vec())
            }
            fn visit_byte_buf<E: serde::de::Error>(self, value: Vec<u8>) -> Result<Vec<u8>, E> {
                if value.len() > super::MAX_CYCLIC_ARTIFACT_BYTES {
                    return Err(E::custom("source byte limit"));
                }
                Ok(value)
            }
        }
        deserializer.deserialize_byte_buf(Bytes)
    }
}
/// Native evidence whose constructor can only read an executing prepared handle
/// or an independently replayed original artifact.
#[derive(Clone, Debug)]
pub struct CyclicEvidence {
    report: CyclicEvidenceReport,
    source: CyclicEvidenceSource,
    schema: CyclicEvidenceSchema,
    selectors: Vec<CyclicEvidenceSelector>,
    envelopes: Vec<EvidenceEnvelope>,
    version: u16,
    // Keep accepted historical bytes exactly; re-export never upgrades their standing.
    original_encoding: Option<(Arc<[u8]>, Arc<antecedent_core::execution::AdmissionLease>)>,
    _retention: Arc<Vec<antecedent_core::execution::AdmissionLease>>,
}

/// Independently verified family source, with no reconstructed prepared study.
#[derive(Clone, Debug)]
pub struct ConsumedCyclicFamily {
    request: Arc<R>,
    report: Arc<CyclicFamilyReport>,
    contract: Option<Arc<antecedent_io::contract_section::cyclic::CyclicContractWire>>,
    bytes: Arc<[u8]>,
    _retention: Arc<antecedent_core::execution::AdmissionLease>,
}

impl ConsumedCyclicFamily {
    /// Verify through the existing independent native consumer exactly once.
    ///
    /// # Errors
    /// Invalid source, identity mismatch, failed replay, cancellation or resource refusal.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let retention = Arc::new(cyclic_decode_admission(bytes, ctx)?);
        let (request, report, contract) =
            consume_cyclic_family_with_request_and_contract(bytes, expected_identity, ctx)?;
        Ok(Self {
            request: Arc::new(request),
            report: Arc::new(report),
            contract: contract.map(Arc::new),
            bytes: Arc::from(bytes),
            _retention: retention,
        })
    }

    /// Actual independently verified producer section; absent on historical v1.
    /// Reading it does not reconstruct a live study, compile schedules or replay.
    #[must_use]
    pub fn contract_metadata(
        &self,
    ) -> Option<&antecedent_io::contract_section::cyclic::CyclicContractWire> {
        self.contract.as_deref()
    }

    /// The exact independently replayed result, retaining its native admissions.
    #[must_use]
    pub fn report(&self) -> &CyclicFamilyReport {
        &self.report
    }

    /// Original independently checked declaration, without a reconstructed study.
    #[must_use]
    pub fn request(&self) -> &R {
        &self.request
    }

    /// Source identity already checked by the independent consumer.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.report.identity
    }

    /// Original verified encoding; re-export never reconstructs or executes a source.
    #[must_use]
    pub fn artifact(&self) -> &[u8] {
        &self.bytes
    }

    /// Prepare a new branch from an independently verified law without
    /// reconstructing a live source study or repeating source recovery.
    /// # Errors
    /// A missing law, invalid/unsupported worlds, cancellation or admission failure.
    pub fn prepare_recovered_worlds(
        &self,
        worlds: Vec<antecedent_io::cyclic_family_artifact::CyclicWorldWire>,
        outcome: u32,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<CyclicFamilyPreparedState>, CyclicArtifactError> {
        let request = R::retained_recovery_worlds(
            Arc::clone(&self.request),
            Arc::clone(&self.report),
            Arc::new(self.clone()),
            worlds,
            outcome,
        )?;
        crate::Study::prepare_cyclic_family(&request, ctx)
    }

    /// Project verified retained output without replaying, fitting or encoding again.
    ///
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        CyclicEvidence::build(
            &self.request,
            &self.report,
            CyclicEvidenceSource::Replayed(Arc::clone(&self.bytes)),
            schema,
            selectors,
            ctx,
        )
    }
}

/// Explicit provenance of the native system-to-family source bridge.
/// Both identities are canonical native request identities, not replacement labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CyclicSystemBridgeProvenance {
    /// Identity of the original executed/independently verified system.
    pub source_identity: String,
    /// Identity of the corresponding compiled family request and contract.
    pub family_identity: String,
    /// Domain-separated binding of those two source identities.
    pub identity: String,
}

fn system_bridge_provenance(
    source_identity: &str,
    family_identity: &str,
) -> CyclicSystemBridgeProvenance {
    let mut bytes = Vec::with_capacity(source_identity.len() + family_identity.len());
    bytes.extend_from_slice(source_identity.as_bytes());
    bytes.extend_from_slice(family_identity.as_bytes());
    CyclicSystemBridgeProvenance {
        source_identity: source_identity.into(),
        family_identity: family_identity.into(),
        identity: CanonicalWriter::digest_bytes_hex(
            "antecedent.cyclic_system_family_bridge.v1",
            &bytes,
        ),
    }
}

/// A compiled family view of the exact original system output, with no new execution.
/// It retains metadata and the shared report; it contains no reconstructed study.
#[derive(Clone, Debug)]
pub struct CyclicSystemFamilyBridge {
    program: Arc<antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram>,
    report: Arc<CyclicFamilyReport>,
    contract: Arc<antecedent_io::contract_section::cyclic::CyclicContractWire>,
    provenance: CyclicSystemBridgeProvenance,
}

impl CyclicSystemFamilyBridge {
    fn from_system(
        request: &antecedent_io::cyclic_system_artifact::CyclicExecutionRequest,
        report: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionReport>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let program =
            Arc::new(antecedent_io::cyclic_family_artifact::CompiledCyclicFamilyProgram::compile(
                &R::System { request: request.clone() },
                ctx,
            )?);
        let report = Arc::new(program.bind_system_execution(report, ctx)?);
        let A::System(system) = &report.answer else {
            return Err(invalid("system bridge answer"));
        };
        let provenance = system_bridge_provenance(&system.identity, &report.identity);
        let contract = Arc::new(program.contract()?);
        Ok(Self { program, report, contract, provenance })
    }

    /// Exact family report; its System answer shares the original report Arc.
    #[must_use]
    pub fn report(&self) -> &CyclicFamilyReport {
        &self.report
    }

    /// Native compiled contract of the corresponding family source.
    #[must_use]
    pub fn contract_metadata(
        &self,
    ) -> &antecedent_io::contract_section::cyclic::CyclicContractWire {
        &self.contract
    }

    /// The original source ID, corresponding family ID, and explicit bridge ID.
    #[must_use]
    pub fn provenance(&self) -> &CyclicSystemBridgeProvenance {
        &self.provenance
    }

    /// Encode this exact source bridge through the existing family codec.
    /// # Errors
    /// Cancellation or encoding/resource failure.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CyclicArtifactError> {
        self.program.encode_execution(&self.report, ctx)
    }

    /// Project compatible native outputs without executing or exporting the source.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        CyclicEvidence::build(
            self.program.request(),
            &self.report,
            CyclicEvidenceSource::SystemBridge(Arc::new(self.clone())),
            schema,
            selectors,
            ctx,
        )
    }
}

impl super::cyclic::ExecutedCyclicSystem {
    /// Retain compatible family metadata around this exact system execution.
    /// # Errors
    /// Invalid family metadata, cancellation or resource refusal.
    pub fn family_bridge(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<CyclicSystemFamilyBridge>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        let mut cache =
            self.family_bridge.lock().map_err(|_| invalid("system bridge cache lock poisoned"))?;
        if let Some(bridge) = cache.as_ref() {
            return Ok(Arc::clone(bridge));
        }
        let bridge = Arc::new(CyclicSystemFamilyBridge::from_system(
            self.prepared().request(),
            Arc::clone(self.shared_report()),
            ctx,
        )?);
        *cache = Some(Arc::clone(&bridge));
        Ok(bridge)
    }

    /// Compose the actual retained system output through its explicit native bridge.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        self.family_bridge(ctx)?.evidence(schema, selectors, ctx)
    }
}

/// Independently verified system source with no reconstructed PreparedStudy.
#[derive(Clone, Debug)]
pub struct ConsumedCyclicSystem {
    request: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionRequest>,
    report: Arc<antecedent_io::cyclic_system_artifact::CyclicExecutionReport>,
    bytes: Arc<[u8]>,
    _retention: Arc<antecedent_core::execution::AdmissionLease>,
    family_bridge: Arc<std::sync::Mutex<Option<Arc<CyclicSystemFamilyBridge>>>>,
}

impl ConsumedCyclicSystem {
    /// Invoke the independent system consumer exactly once and retain its source.
    /// # Errors
    /// Framing, identity, replay, cancellation or resource failure.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let retention = Arc::new(cyclic_decode_admission(bytes, ctx)?);
        let (request, report) =
            antecedent_io::cyclic_system_artifact::consume_cyclic_artifact_with_request(
                bytes,
                expected_identity,
                ctx,
            )?;
        Ok(Self {
            request: Arc::new(request),
            report: Arc::new(report),
            bytes: Arc::from(bytes),
            _retention: retention,
            family_bridge: Arc::default(),
        })
    }

    /// Exact independently replayed output, borrowing native immutable storage.
    #[must_use]
    pub fn report(&self) -> &antecedent_io::cyclic_system_artifact::CyclicExecutionReport {
        &self.report
    }

    /// Independently verified original declaration, without a live study.
    #[must_use]
    pub fn request(&self) -> &antecedent_io::cyclic_system_artifact::CyclicExecutionRequest {
        &self.request
    }

    /// Original verified source identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.report.identity
    }

    /// Original verified bytes; exporting them never reconstructs or executes a study.
    #[must_use]
    pub fn artifact(&self) -> &[u8] {
        &self.bytes
    }

    /// Compile corresponding family metadata around the verified report once.
    /// # Errors
    /// Invalid metadata, cancellation or resource failure.
    pub fn family_bridge(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<CyclicSystemFamilyBridge>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        let mut cache = self
            .family_bridge
            .lock()
            .map_err(|_| invalid("consumed system bridge cache lock poisoned"))?;
        if let Some(bridge) = cache.as_ref() {
            return Ok(Arc::clone(bridge));
        }
        let bridge = Arc::new(CyclicSystemFamilyBridge::from_system(
            &self.request,
            Arc::clone(&self.report),
            ctx,
        )?);
        *cache = Some(Arc::clone(&bridge));
        Ok(bridge)
    }

    /// Project verified output directly without another replay, fit or solve.
    /// # Errors
    /// Invalid schema/selectors, cancellation or resource failure.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        self.family_bridge(ctx)?.evidence(schema, selectors, ctx)
    }
}

/// Live sources retain the exact executed owner; consumed sources retain their
/// original bytes. Only explicit export encodes a live source.
#[derive(Clone, Debug)]
enum CyclicEvidenceSource {
    Live(Arc<ExecutedCyclicFamily>),
    SystemBridge(Arc<CyclicSystemFamilyBridge>),
    Replayed(Arc<[u8]>),
}

impl CyclicEvidenceSource {
    fn retained_encoding_len(&self) -> usize {
        match self {
            Self::Live(_) | Self::SystemBridge(_) => 0,
            Self::Replayed(bytes) => bytes.len(),
        }
    }

    fn encoding(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<std::borrow::Cow<'_, [u8]>, CyclicArtifactError> {
        match self {
            Self::Live(executed) => executed.export(ctx).map(std::borrow::Cow::Owned),
            Self::SystemBridge(bridge) => bridge.export(ctx).map(std::borrow::Cow::Owned),
            Self::Replayed(bytes) => Ok(std::borrow::Cow::Borrowed(bytes.as_ref())),
        }
    }
}

/// Bytes an artifact admission reserves: four copies of the source, eight of the serialised
/// projection and a fixed overhead, saturating so an oversized request is refused by the pool
/// instead of wrapping.
fn admission_reserve(source_len: usize, serialized_len: usize, overhead: u64) -> u64 {
    (source_len as u64)
        .saturating_mul(4)
        .saturating_add((serialized_len as u64).saturating_mul(8))
        .saturating_add(overhead)
}

fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
}
/// Hex digest of the JSON encoding of `value` under `tag`.
///
/// The encoding is versioned by the tag: the digest of a tuple of borrowed parts is the digest
/// of its `serde_json` array, so the parts, their order and their serialized form are part of
/// the identity, and nothing is cloned or cleared to build it.
fn digest<T: Serialize>(tag: &str, value: &T) -> Result<String, CyclicArtifactError> {
    Ok(CanonicalWriter::digest_bytes_hex(tag, &serde_json::to_vec(value).map_err(invalid)?))
}
fn state(outcome: &EquilibriumOutcome, variable: usize) -> Option<f64> {
    match outcome {
        EquilibriumOutcome::Solved(value) => value.values.get(variable).copied(),
        EquilibriumOutcome::Failed { .. } => None,
    }
}
fn licensed_state(outcome: &EquilibriumOutcome, variable: usize) -> Option<f64> {
    match outcome {
        EquilibriumOutcome::Solved(value) if value.globally_unique && value.existence_certified => {
            value.values.get(variable).copied()
        }
        _ => None,
    }
}

/// The scientific functional a projected coordinate reports. The spelling is the contract's
/// `functional_id`, so it is part of the quantity identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Functional {
    RecoveredIntervention,
    RecoveredInterventionalMean,
    InterventionalOutcome,
    InterventionalMean,
    SpecifiedNoiseResponse,
    FactualCounterfactual,
    TrajectoryState,
    NoSolution,
    DynamicCovariance,
    DynamicQuantity,
    ConditionalStructuralResponse,
    Outcome,
    PredictiveCovariance,
    Mean,
    LogLikelihood,
    ConditionalTrajectory,
    SolutionCoordinate,
    ModelSolutionSet,
    IdentifiedProbability,
    IdentifiedDensity,
    GraphicalEvidence,
    CounterfactualOutcomeSet,
}

impl Functional {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RecoveredIntervention => "recovered_intervention",
            Self::RecoveredInterventionalMean => "recovered_interventional_mean",
            Self::InterventionalOutcome => "interventional_outcome",
            Self::InterventionalMean => "interventional_mean",
            Self::SpecifiedNoiseResponse => "specified_noise_response",
            Self::FactualCounterfactual => "factual_counterfactual",
            Self::TrajectoryState => "trajectory_state",
            Self::NoSolution => "no_solution",
            Self::DynamicCovariance => "dynamic_covariance",
            Self::DynamicQuantity => "dynamic_quantity",
            Self::ConditionalStructuralResponse => "conditional_structural_response",
            Self::Outcome => "outcome",
            Self::PredictiveCovariance => "predictive_covariance",
            Self::Mean => "mean",
            Self::LogLikelihood => "log_likelihood",
            Self::ConditionalTrajectory => "conditional_trajectory",
            Self::SolutionCoordinate => "solution_coordinate",
            Self::ModelSolutionSet => "model_solution_set",
            Self::IdentifiedProbability => "identified_probability",
            Self::IdentifiedDensity => "identified_density",
            Self::GraphicalEvidence => "graphical_evidence",
            Self::CounterfactualOutcomeSet => "counterfactual_outcome_set",
        }
    }
}

/// An independent uncertainty axis of a projected column; the spelling is the column's
/// `uncertainty_axes` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Predictive,
    Sampling,
    Parameter,
}

impl Axis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Predictive => "predictive",
            Self::Sampling => "sampling",
            Self::Parameter => "parameter",
        }
    }
}

/// The world a projected coordinate belongs to; the rendering is the evidence binding's
/// `world_id` and enters the regime digest.
#[derive(Clone, Debug)]
enum World {
    /// A world identity the source declared.
    Declared(String),
    /// A fixed label of the family.
    Fixed(&'static str),
    /// A world by position or scientific id.
    Indexed(u64),
    /// One disturbance row of a world.
    Noise { world: usize, noise_row: usize },
    /// One recovered draw of a world.
    Draw { world: usize, draw: usize },
    /// An equation-replacement world.
    Equation(usize),
    /// A certified root enclosure.
    Root(usize),
    /// A declared quantity of a dynamic family.
    Quantity(usize),
    /// A retained identified-functional assignment.
    Assignment(usize),
}

impl std::fmt::Display for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declared(id) => f.write_str(id),
            Self::Fixed(label) => f.write_str(label),
            Self::Indexed(world) => write!(f, "world:{world}"),
            Self::Noise { world, noise_row } => write!(f, "world:{world}:noise:{noise_row}"),
            Self::Draw { world, draw } => write!(f, "world:{world}:draw:{draw}"),
            Self::Equation(world) => write!(f, "equation-world:{world}"),
            Self::Root(root) => write!(f, "root_enclosure:{root}"),
            Self::Quantity(quantity) => write!(f, "quantity:{quantity}"),
            Self::Assignment(assignment) => write!(f, "assignment:{assignment}"),
        }
    }
}

impl World {
    fn indexed(world: usize) -> Self {
        Self::Indexed(world as u64)
    }
}

/// The time of a projected coordinate; the rendering is the evidence binding's `time_id`.
#[derive(Clone, Copy, Debug)]
enum Time {
    Equilibrium,
    Stationary,
    Algebraic,
    Static,
    Trajectory,
    Dynamic,
    /// An observation-time index.
    Step(usize),
    /// An observation time.
    At(f64),
    /// The evaluation time of an ODE state, rendered as the evidence binding spells it.
    Instant(f64),
}

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Equilibrium => f.write_str("equilibrium"),
            Self::Stationary => f.write_str("stationary"),
            Self::Algebraic => f.write_str("algebraic"),
            Self::Static => f.write_str("static"),
            Self::Trajectory => f.write_str("trajectory"),
            Self::Dynamic => f.write_str("dynamic"),
            Self::Step(index) => write!(f, "time:{index}"),
            Self::At(time) => write!(f, "t:{time}"),
            Self::Instant(time) => write!(f, "time:{time}"),
        }
    }
}

/// Retained details of a projection, serialized once into the column's JSON diagnostics.
trait Detail {
    fn to_json(&self) -> Result<serde_json::Value, serde_json::Error>;
}

impl<T: Serialize + ?Sized> Detail for T {
    fn to_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

fn detail<'a, T: Serialize + 'a>(value: T) -> Option<Box<dyn Detail + 'a>> {
    Some(Box::new(value))
}

/// A recovered world together with the draw origin that produced it.
#[derive(Serialize)]
struct RecoveredWithOrigin<'a> {
    world: &'a antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: &'a antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire,
}

/// One recovered draw with the variance of its interventional law.
#[derive(Serialize)]
struct WithinDraw<'a> {
    variance: Option<f64>,
    draw: &'a antecedent_io::cyclic_family_artifact::RecoveredEvaluatedDraw,
}

/// The certificate, route and masses of one typed world.
struct WorldContext<'a>(&'a antecedent_io::cyclic_family_artifact::CyclicWorldAnswer);

impl Serialize for WorldContext<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let world = self.0;
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("certificate", &world.certificate)?;
        map.serialize_entry("solved_by", &world.solved_by)?;
        map.serialize_entry("failed_mass", &world.failed_mass)?;
        map.serialize_entry("withheld_mass", &world.withheld_mass)?;
        map.serialize_entry("actions", &world.actions)?;
        map.end()
    }
}

/// A Gaussian world: the variance of the projected coordinate and the world.
#[derive(Serialize)]
struct GaussianWorld<'a> {
    variance: Option<&'a f64>,
    world: WorldContext<'a>,
}

/// A world that returned no solution or was withheld.
#[derive(Serialize)]
struct UnsolvedWorld<'a> {
    outcome: &'a antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer,
    world: WorldContext<'a>,
}

/// Diagnostics of a bootstrap sampling law of a dynamic quantity.
#[derive(Serialize)]
struct SamplingLawDetail<'a, E: Serialize, S: Serialize, Q: Serialize, R: Serialize, M: Serialize> {
    estimate: Option<&'a E>,
    standard_error: Option<&'a S>,
    failed_mass: Option<&'a Q>,
    replicate_failed_mass: &'a R,
    method: &'a M,
}

/// Diagnostics of the observed information of a dynamic quantity.
#[derive(Serialize)]
struct InformationDetail<'a, S: Serialize, G: Serialize> {
    standard_error: Option<&'a S>,
    regularity: &'a G,
}

/// Diagnostics of the posterior of a dynamic quantity.
#[derive(Serialize)]
struct PosteriorDetail<'a, Q: Serialize, H: Serialize, B: Serialize, T: Serialize, P: Serialize> {
    failed_mass: Option<&'a Q>,
    max_rhat: &'a H,
    min_bulk_ess: &'a B,
    min_tail_ess: &'a T,
    diagnostics_pass: &'a P,
}

/// A certified root enclosure: no selection law is implied.
#[derive(Serialize)]
struct RootDetail {
    unresolved_regions: usize,
    selection_law: bool,
}

/// The outer solution set of a bounded system.
#[derive(Serialize)]
struct SolutionSetDetail {
    certified_roots: usize,
    unresolved_regions: usize,
    selection_law: bool,
    outer_enclosure: bool,
}

/// The coupled finite-law outcomes of a polynomial causal model.
#[derive(Serialize)]
struct PolynomialWorldDetail<'a, W: Serialize> {
    world_mass: &'a W,
    positive_evidence_certified: bool,
}

struct Projection<'a> {
    samples: Vec<Option<f64>>,
    bounds: Vec<Option<Interval>>,
    masses: Option<Vec<antecedent_estimate::polynomial_scm::MassBounds>>,
    port: EvidencePort,
    axes: Vec<Axis>,
    world: World,
    time: Time,
    functional: Functional,
    variable: usize,
    /// Second coordinate of a covariance quantity.
    other: Option<usize>,
    /// Source details retained beside the numbers; `None` serializes as JSON null.
    diagnostics: Option<Box<dyn Detail + 'a>>,
}
fn point<'a>(
    value: Option<f64>,
    variable: usize,
    world: World,
    time: Time,
    functional: Functional,
    port: EvidencePort,
) -> Projection<'a> {
    Projection {
        samples: vec![value],
        bounds: vec![],
        masses: None,
        port,
        axes: vec![],
        world,
        time,
        functional,
        variable,
        other: None,
        diagnostics: None,
    }
}

fn selector_mismatch() -> CyclicArtifactError {
    invalid("cyclic evidence selector differs from source family or coordinate bounds")
}

/// A retained native answer that can project the coordinates a selector names.
///
/// Each answer kind reads only its own payload: the dispatch is on the answer, and the
/// selectors a kind cannot serve are a selector mismatch, never a coerced reading.
trait EvidenceSource {
    /// Project the coordinate `selector` names out of this answer, as asked by `request`.
    ///
    /// # Errors
    /// A selector the answer kind or its coordinate bounds do not admit.
    fn project<'a>(
        &'a self,
        request: &'a R,
        selector: &CyclicEvidenceSelector,
    ) -> Result<Projection<'a>, CyclicArtifactError>;
}

/// Draw-aligned projection of one recovered world.
///
/// Supplied draws declare no meaning and stay on the interventional port. Draws
/// from a declared sampling law or posterior project the per-draw interventional
/// mean onto that port: parameter uncertainty is not relabelled as outcome
/// variation, and a sampling law is not read as a posterior.
fn recovered_response<'a>(
    found: &'a antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: &'a antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire,
    world: usize,
    variable: usize,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let mut samples = Vec::with_capacity(found.draws.len());
    let mut masses = Vec::with_capacity(found.draws.len());
    for draw in &found.draws {
        masses.push(antecedent_estimate::polynomial_scm::MassBounds {
            lower: draw.weight,
            upper: draw.weight,
        });
        samples.push(match &draw.outcome {
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
                mean,
                ..
            } => Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Inadmissible { .. }
            | antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::NotCertified { .. } => {
                None
            }
        });
    }
    let (port, axis, functional) = match origin.law_meaning() {
        None => {
            (EvidencePort::InterventionalLaw, Axis::Predictive, Functional::RecoveredIntervention)
        }
        Some("sampling_law") => {
            (EvidencePort::SamplingLaw, Axis::Sampling, Functional::RecoveredInterventionalMean)
        }
        Some(_) => (
            EvidencePort::ParameterPosterior,
            Axis::Parameter,
            Functional::RecoveredInterventionalMean,
        ),
    };
    let mut p = point(None, variable, World::indexed(world), Time::Equilibrium, functional, port);
    p.samples = samples;
    p.masses = Some(masses);
    p.axes = vec![axis];
    p.diagnostics = if origin.is_supplied() {
        detail(found)
    } else {
        detail(RecoveredWithOrigin { world: found, origin })
    };
    Ok(p)
}

/// The interventional law within one recovered draw.
fn recovered_within(
    found: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    world: usize,
    draw: usize,
    variable: usize,
) -> Result<Projection<'_>, CyclicArtifactError> {
    let item = found.draws.get(draw).ok_or_else(selector_mismatch)?;
    let (value, variance) = match &item.outcome {
        antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
            mean,
            covariance,
            ..
        } => {
            let n = mean.len();
            let value = *mean.get(variable).ok_or_else(selector_mismatch)?;
            (Some(value), covariance.get(variable * n + variable).copied())
        }
        _ => (None, None),
    };
    let mut p = point(
        value,
        variable,
        World::Draw { world, draw },
        Time::Equilibrium,
        Functional::InterventionalOutcome,
        EvidencePort::InterventionalLaw,
    );
    p.axes = vec![Axis::Predictive];
    p.masses = Some(vec![antecedent_estimate::polynomial_scm::MassBounds {
        lower: item.weight,
        upper: item.weight,
    }]);
    p.diagnostics = detail(WithinDraw { variance, draw: item });
    Ok(p)
}

/// Projection of one typed world of a specified model.
fn typed_world(
    found: &antecedent_io::cyclic_family_artifact::CyclicWorldAnswer,
    variable: usize,
    time: Option<usize>,
) -> Result<Projection<'_>, CyclicArtifactError> {
    use antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as O;
    if variable >= found.variables.len() {
        return Err(selector_mismatch());
    }
    let label = World::Indexed(u64::from(found.world));
    let mut p = match (&found.outcome, time) {
        (O::GaussianLaw { mean, covariance, .. }, None) => {
            let n = mean.len();
            let mut p = point(
                Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
                variable,
                label,
                Time::Equilibrium,
                Functional::InterventionalMean,
                EvidencePort::InterventionalLaw,
            );
            p.axes = vec![Axis::Predictive];
            p.diagnostics = detail(GaussianWorld {
                variance: covariance.get(variable * n + variable),
                world: WorldContext(found),
            });
            p
        }
        (O::EquilibriumDraws { outcomes }, None) => {
            let mut p = point(
                None,
                variable,
                label,
                Time::Equilibrium,
                Functional::SpecifiedNoiseResponse,
                EvidencePort::Scenario,
            );
            p.samples = outcomes.iter().map(|outcome| state(outcome, variable)).collect();
            p.diagnostics = detail(WorldContext(found));
            p
        }
        (O::Trajectory { trajectory }, Some(index)) => {
            let value = trajectory
                .states
                .get(index)
                .and_then(|row| row.get(variable))
                .copied()
                .ok_or_else(selector_mismatch)?;
            let mut p = point(
                Some(value),
                variable,
                label,
                Time::Step(index),
                Functional::TrajectoryState,
                EvidencePort::Scenario,
            );
            p.diagnostics = detail(WorldContext(found));
            p
        }
        (O::NoSolution { .. } | O::Withheld, _) => {
            let mut p = point(
                None,
                variable,
                label,
                Time::Equilibrium,
                Functional::NoSolution,
                EvidencePort::Scenario,
            );
            p.diagnostics =
                detail(UnsolvedWorld { outcome: &found.outcome, world: WorldContext(found) });
            p
        }
        _ => return Err(selector_mismatch()),
    };
    p.bounds.clear();
    Ok(p)
}

/// Endogenous coordinates of one dynamic quantity.
const fn quantity_coordinates(
    quantity: &antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity,
) -> (usize, Option<usize>) {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    match quantity {
        Q::OdeState { variable, .. }
        | Q::OuStationaryMean { variable, .. }
        | Q::OuMechanismMean { variable, .. } => (*variable, None),
        Q::OuStationaryCovariance { variable, other, .. }
        | Q::OuMechanismCovariance { variable, other, .. } => (*variable, Some(*other)),
    }
}

/// Projection of one declared quantity of a fitted continuous-time family.
fn dynamic_quantity<'a>(
    quantities: &[antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity],
    inference: &'a antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer,
    quantity: usize,
) -> Result<Projection<'a>, CyclicArtifactError> {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    use antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer as D;
    let spec = quantities.get(quantity).ok_or_else(selector_mismatch)?;
    let (variable, other) = quantity_coordinates(spec);
    let time = match spec {
        Q::OdeState { time, .. } => Time::Instant(*time),
        _ => Time::Stationary,
    };
    let label = World::Quantity(quantity);
    let functional =
        if other.is_some() { Functional::DynamicCovariance } else { Functional::DynamicQuantity };
    let mut p = match inference {
        D::SamplingLaw(law) => {
            let mut p = point(None, variable, label, time, functional, EvidencePort::SamplingLaw);
            p.samples = law
                .replicates
                .iter()
                .map(|replicate| replicate.quantities.get(quantity).copied().flatten())
                .collect();
            p.masses = Some(
                law.weights
                    .iter()
                    .map(|weight| antecedent_estimate::polynomial_scm::MassBounds {
                        lower: *weight,
                        upper: *weight,
                    })
                    .collect(),
            );
            p.axes = vec![Axis::Sampling];
            p.diagnostics = detail(SamplingLawDetail {
                estimate: law.quantity_estimates.get(quantity),
                standard_error: law.quantity_standard_errors.get(quantity),
                failed_mass: law.quantity_failed_mass.get(quantity),
                replicate_failed_mass: &law.failed_mass,
                method: &law.method,
            });
            p
        }
        D::Information(information) => {
            let mut p = point(
                information.quantity_estimates.get(quantity).copied().flatten(),
                variable,
                label,
                time,
                functional,
                EvidencePort::CovarianceInfluence,
            );
            p.axes = vec![Axis::Sampling];
            p.diagnostics = detail(InformationDetail {
                standard_error: information.quantity_standard_errors.get(quantity),
                regularity: &information.regularity,
            });
            p
        }
        D::Posterior(posterior) => {
            let mut p =
                point(None, variable, label, time, functional, EvidencePort::ParameterPosterior);
            p.samples = posterior
                .quantity_draws
                .iter()
                .map(|draw| draw.get(quantity).copied().flatten())
                .collect();
            p.axes = vec![Axis::Parameter];
            p.diagnostics = detail(PosteriorDetail {
                failed_mass: posterior.quantity_failed_mass.get(quantity),
                max_rhat: &posterior.max_rhat,
                min_bulk_ess: &posterior.min_bulk_ess,
                min_tail_ess: &posterior.min_tail_ess,
                diagnostics_pass: &posterior.diagnostics_pass,
            });
            p
        }
    };
    p.other = other;
    Ok(p)
}

/// The fitted log likelihood, the only coordinate a point fit offers.
fn log_likelihood<'a>(
    value: f64,
    time: Time,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Likelihood { candidate: None } = selector else {
        return Err(selector_mismatch());
    };
    Ok(point(
        Some(value),
        0,
        World::Fixed("training"),
        time,
        Functional::LogLikelihood,
        EvidencePort::Likelihood,
    ))
}

/// Named structural-action worlds: the supplied-noise rows or the abducted factual response.
fn actions<'a>(
    worlds: &'a [antecedent_io::cyclic_family_artifact::CyclicActionWorldReport],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Action { world, noise_row, variable, counterfactual } = selector
    else {
        return Err(selector_mismatch());
    };
    let w = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let value = if *counterfactual { w.counterfactual.as_ref() } else { w.rows.get(*noise_row) }
        .ok_or_else(selector_mismatch)?;
    let mut p = point(
        state(value, *variable),
        *variable,
        World::Declared(w.id.clone()),
        Time::Equilibrium,
        if *counterfactual {
            Functional::FactualCounterfactual
        } else {
            Functional::SpecifiedNoiseResponse
        },
        EvidencePort::Scenario,
    );
    // A residual-checked numeric state is not an outer enclosure of all solutions.
    p.bounds.clear();
    p.diagnostics = detail(value);
    Ok(p)
}

/// Parameter posterior responses, `(world, parameter_draw, noise_row)`.
fn parameter_posterior<'a>(
    worlds: &[Vec<Vec<EquilibriumOutcome>>],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let values = draws
        .iter()
        .map(|draw| {
            draw.get(*noise_row)
                .map(|value| licensed_state(value, *variable))
                .ok_or_else(selector_mismatch)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut p = point(
        None,
        *variable,
        World::Noise { world: *world, noise_row: *noise_row },
        Time::Equilibrium,
        Functional::ConditionalStructuralResponse,
        EvidencePort::ParameterPosterior,
    );
    p.samples = values;
    p.axes = vec![Axis::Parameter];
    Ok(p)
}

/// Counterfactual predictions sharing each factual disturbance across worlds.
/// `joint` marks the joint parameter/factual posterior, which adds the parameter axis.
fn counterfactual_responses<'a>(
    worlds: &[Vec<EquilibriumOutcome>],
    joint: bool,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    if *noise_row != 0 {
        return Err(selector_mismatch());
    }
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let mut p = point(
        None,
        *variable,
        World::indexed(*world),
        Time::Equilibrium,
        Functional::Outcome,
        EvidencePort::InterventionalLaw,
    );
    p.samples = draws.iter().map(|v| licensed_state(v, *variable)).collect();
    p.axes = vec![Axis::Predictive];
    if joint {
        p.axes.push(Axis::Parameter);
    }
    Ok(p)
}

/// Conditional frequentist bootstrap responses; a failed fit keeps its slot.
fn bootstrap<'a>(
    worlds: &[Vec<Option<Vec<EquilibriumOutcome>>>],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let mut p = point(
        None,
        *variable,
        World::Noise { world: *world, noise_row: *noise_row },
        Time::Equilibrium,
        Functional::ConditionalStructuralResponse,
        EvidencePort::SamplingLaw,
    );
    p.samples = draws
        .iter()
        .map(|draw| match draw {
            Some(rows) => rows
                .get(*noise_row)
                .map(|v| licensed_state(v, *variable))
                .ok_or_else(selector_mismatch),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;
    p.axes = vec![Axis::Sampling];
    Ok(p)
}

/// Stationary laws of the shared-mechanism family in the requested equation worlds.
fn ou_mechanisms<'a>(
    laws: &'a [antecedent_estimate::ou_mechanisms::OuMechanismOutcome],
    fit_log_likelihood: Option<f64>,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::OuRegime { world, variable, other } => {
            let outcome = laws.get(*world).ok_or_else(selector_mismatch)?;
            let functional =
                if other.is_some() { Functional::PredictiveCovariance } else { Functional::Mean };
            let port = if other.is_some() { EvidencePort::Likelihood } else { EvidencePort::Mean };
            let antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) = outcome
            else {
                let mut p = point(
                    None,
                    *variable,
                    World::Equation(*world),
                    Time::Stationary,
                    functional,
                    port,
                );
                p.diagnostics = detail(outcome);
                return Ok(p);
            };
            let n = law.mean.len();
            if *variable >= n || other.is_some_and(|j| j >= n) {
                return Err(selector_mismatch());
            }
            Ok(point(
                Some(other.map_or(law.mean[*variable], |j| law.covariance[*variable * n + j])),
                *variable,
                World::Equation(*world),
                Time::Stationary,
                functional,
                port,
            ))
        }
        CyclicEvidenceSelector::Likelihood { candidate: None } => {
            let value = fit_log_likelihood.ok_or_else(selector_mismatch)?;
            log_likelihood(value, Time::Stationary, selector)
        }
        _ => Err(selector_mismatch()),
    }
}

/// Deterministic ODE path.
fn ode_path<'a>(
    path: &antecedent_model::continuous_cyclic::OdeTrajectory,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Ode { time, variable } = selector else {
        return Err(selector_mismatch());
    };
    Ok(point(
        Some(*path.states.get(*time).and_then(|s| s.get(*variable)).ok_or_else(selector_mismatch)?),
        *variable,
        World::Fixed("initial_value_world"),
        Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
        Functional::ConditionalTrajectory,
        EvidencePort::Mean,
    ))
}

/// Stationary OU moments, read from the declared model and the stored covariance.
fn ou_stationary<'a>(
    request: &R,
    covariance: &[f64],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let R::OuStationary { model } = request else {
        return Err(selector_mismatch());
    };
    match selector {
        CyclicEvidenceSelector::OuMean { time: None, variable } => Ok(point(
            Some(*model.center.get(*variable).ok_or_else(selector_mismatch)?),
            *variable,
            World::Fixed("stationary"),
            Time::Stationary,
            Functional::Mean,
            EvidencePort::Mean,
        )),
        CyclicEvidenceSelector::OuCovariance { time: None, variable, other } => {
            let n = model.center.len();
            if *variable >= n || *other >= n {
                return Err(selector_mismatch());
            }
            Ok(point(
                Some(covariance[*variable * n + other]),
                *variable,
                World::Fixed("stationary"),
                Time::Stationary,
                Functional::PredictiveCovariance,
                EvidencePort::Likelihood,
            ))
        }
        _ => Err(selector_mismatch()),
    }
}

/// Finite-horizon OU mean and covariance.
fn ou_transient<'a>(
    path: &antecedent_model::continuous_cyclic::OuMoments,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::OuMean { time: Some(time), variable } => Ok(point(
            Some(
                *path
                    .means
                    .get(*time)
                    .and_then(|s| s.get(*variable))
                    .ok_or_else(selector_mismatch)?,
            ),
            *variable,
            World::Fixed("initial_law_world"),
            Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
            Functional::Mean,
            EvidencePort::Mean,
        )),
        CyclicEvidenceSelector::OuCovariance { time: Some(time), variable, other } => {
            let row = path.means.get(*time).ok_or_else(selector_mismatch)?;
            let n = row.len();
            if *variable >= n || *other >= n {
                return Err(selector_mismatch());
            }
            let covariance = path.covariances.get(*time).ok_or_else(selector_mismatch)?;
            Ok(point(
                Some(*covariance.get(*variable * n + other).ok_or_else(selector_mismatch)?),
                *variable,
                World::Fixed("initial_law_world"),
                Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
                Functional::PredictiveCovariance,
                EvidencePort::Likelihood,
            ))
        }
        _ => Err(selector_mismatch()),
    }
}

/// Certified roots and unresolved regions of a bounded polynomial system.
fn polynomial_solutions<'a>(
    solutions: &antecedent_model::equation_system::IsolatedSolutions,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::Root { root, variable } => {
            let enclosure = *solutions
                .roots
                .get(*root)
                .and_then(|s| s.get(*variable))
                .ok_or_else(selector_mismatch)?;
            let mut p = point(
                None,
                *variable,
                World::Root(*root),
                Time::Algebraic,
                Functional::SolutionCoordinate,
                EvidencePort::Scenario,
            );
            p.bounds = vec![Some(enclosure)];
            p.diagnostics = detail(RootDetail {
                unresolved_regions: solutions.unresolved.len(),
                selection_law: false,
            });
            Ok(p)
        }
        CyclicEvidenceSelector::SolutionSet { variable } => {
            let mut p = point(
                None,
                *variable,
                World::Fixed("all_model_solutions"),
                Time::Algebraic,
                Functional::ModelSolutionSet,
                EvidencePort::Scenario,
            );
            p.samples.clear();
            p.bounds = solutions
                .roots
                .iter()
                .chain(&solutions.unresolved)
                .map(|region| {
                    region.get(*variable).copied().map(Some).ok_or_else(selector_mismatch)
                })
                .collect::<Result<Vec<_>, _>>()?;
            p.diagnostics = detail(SolutionSetDetail {
                certified_roots: solutions.roots.len(),
                unresolved_regions: solutions.unresolved.len(),
                selection_law: false,
                outer_enclosure: true,
            });
            Ok(p)
        }
        _ => Err(selector_mismatch()),
    }
}

/// The single outcome a scalar functional projection reads.
fn single_outcome(
    source: &antecedent_io::cyclic_graph::CyclicGraphRequest,
    message: &'static str,
) -> Result<usize, CyclicArtifactError> {
    use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
    let outcomes = match source {
        G::Identify { outcomes, .. } | G::Adjust { outcomes, .. } => outcomes,
        G::Calculus { y, .. } => y,
        _ => return Err(selector_mismatch()),
    };
    if outcomes.len() != 1 {
        return Err(invalid(message));
    }
    Ok(outcomes[0])
}

/// One typed world of a specified model, by scientific world id.
fn world_of<'a>(
    worlds: &'a [antecedent_io::cyclic_family_artifact::CyclicWorldAnswer],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::World { world, variable, time } = selector else {
        return Err(selector_mismatch());
    };
    let found = worlds.iter().find(|item| item.world == *world).ok_or_else(selector_mismatch)?;
    typed_world(found, *variable, *time)
}

impl EvidenceSource for A {
    #[allow(
        clippy::too_many_lines,
        reason = "one dispatch per answer kind keeps each kind's selectors and uncertainty port explicit"
    )]
    fn project<'a>(
        &'a self,
        request: &'a R,
        selector: &CyclicEvidenceSelector,
    ) -> Result<Projection<'a>, CyclicArtifactError> {
        use CyclicEvidenceSelector as S;
        match self {
            A::EquilibriumActions(worlds) => actions(worlds, selector),
            A::ParameterPosterior { worlds, .. } => parameter_posterior(worlds, selector),
            A::JointCounterfactual { worlds, .. } => {
                counterfactual_responses(worlds, true, selector)
            }
            A::PartialCounterfactual { worlds, .. } => {
                counterfactual_responses(worlds, false, selector)
            }
            A::FrequentistBootstrap { worlds, .. } => bootstrap(worlds, selector),
            A::OuMechanisms { laws, fit } => {
                ou_mechanisms(laws, fit.as_ref().map(|fit| fit.log_likelihood), selector)
            }
            A::Ode(path) => ode_path(path, selector),
            A::OuStationary(covariance) => ou_stationary(request, covariance, selector),
            A::OuTransient(path) => ou_transient(path, selector),
            A::Polynomial(solutions) => polynomial_solutions(solutions, selector),
            A::Functional(report) => {
                let (R::Functional { source, .. }, S::Functional { assignment }) =
                    (request, selector)
                else {
                    return Err(selector_mismatch());
                };
                let outcome = single_outcome(
                    source,
                    "scalar evidence projection requires a single outcome; joint events retain the full functional report",
                )?;
                let value = report.values.get(*assignment).ok_or_else(selector_mismatch)?;
                let mut p = point(
                    value.value,
                    outcome,
                    World::Assignment(*assignment),
                    Time::Static,
                    Functional::IdentifiedProbability,
                    EvidencePort::Mean,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::GaussianFunctional(report) => {
                let (R::GaussianFunctional { source, .. }, S::Functional { assignment }) =
                    (request, selector)
                else {
                    return Err(selector_mismatch());
                };
                let outcome = single_outcome(
                    source,
                    "scalar Gaussian density evidence requires one outcome",
                )?;
                let value = report.values.get(*assignment).ok_or_else(selector_mismatch)?;
                let mut p = point(
                    value.value,
                    outcome,
                    World::Assignment(*assignment),
                    Time::Static,
                    Functional::IdentifiedDensity,
                    EvidencePort::Likelihood,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::AdamFit(fit) => {
                log_likelihood(fit.training_log_likelihood, Time::Equilibrium, selector)
            }
            A::GradientFit(fit) => log_likelihood(fit.log_likelihood, Time::Static, selector),
            A::OdeFit(fit) => log_likelihood(fit.log_likelihood, Time::Trajectory, selector),
            A::OuFit(fit) => log_likelihood(fit.log_likelihood, Time::Stationary, selector),
            A::StructureFit { fit, .. } => {
                log_likelihood(fit.fit.log_likelihood, Time::Equilibrium, selector)
            }
            A::PolynomialFit(fit) => {
                let S::Likelihood { candidate: Some(candidate) } = selector else {
                    return Err(selector_mismatch());
                };
                let c = fit.candidates.get(*candidate).ok_or_else(selector_mismatch)?;
                let mut p = point(
                    None,
                    0,
                    World::Declared(c.candidate.clone()),
                    Time::Algebraic,
                    Functional::LogLikelihood,
                    EvidencePort::Likelihood,
                );
                p.bounds =
                    vec![c.lower.zip(c.upper).map(|(lower, upper)| Interval { lower, upper })];
                p.diagnostics = detail(c);
                Ok(p)
            }
            A::RecoveredLinear(report) => match selector {
                S::Response { world, noise_row, variable } => {
                    let R::RecoveredLinear { uncertainty, .. } = request else {
                        return Err(selector_mismatch());
                    };
                    if *noise_row != 0 {
                        return Err(selector_mismatch());
                    }
                    let found = report.worlds.get(*world).ok_or_else(selector_mismatch)?;
                    recovered_response(found, uncertainty, *world, *variable)
                }
                S::RecoveredDraw { world, draw, variable } => recovered_within(
                    report.worlds.get(*world).ok_or_else(selector_mismatch)?,
                    *world,
                    *draw,
                    *variable,
                ),
                _ => Err(selector_mismatch()),
            },
            A::ExperimentalRecovery(report) => match selector {
                S::Response { world, noise_row, variable } => {
                    if *noise_row != 0 {
                        return Err(selector_mismatch());
                    }
                    let found = &report.worlds.get(*world).ok_or_else(selector_mismatch)?.answer;
                    recovered_response(found, &report.law.origin, *world, *variable)
                }
                S::RecoveredDraw { world, draw, variable } => recovered_within(
                    &report.worlds.get(*world).ok_or_else(selector_mismatch)?.answer,
                    *world,
                    *draw,
                    *variable,
                ),
                _ => Err(selector_mismatch()),
            },
            A::Worlds(worlds) => world_of(worlds, selector),
            A::DynamicInference { fit, inference } => match selector {
                S::DynamicQuantity { quantity } => {
                    let R::DynamicInference { quantities, .. } = request else {
                        return Err(selector_mismatch());
                    };
                    dynamic_quantity(quantities, inference, *quantity)
                }
                S::Likelihood { candidate: None } => {
                    log_likelihood(fit.log_likelihood(), Time::Dynamic, selector)
                }
                _ => Err(selector_mismatch()),
            },
            A::System(report) => {
                let S::Action { world, noise_row, variable, counterfactual } = selector else {
                    return Err(selector_mismatch());
                };
                let value = if *counterfactual {
                    report.counterfactuals.as_ref().and_then(|rows| rows.get(*world))
                } else {
                    report.worlds.get(*world).and_then(|rows| rows.get(*noise_row))
                }
                .ok_or_else(selector_mismatch)?;
                let mut p = point(
                    Some(*value.values.get(*variable).ok_or_else(selector_mismatch)?),
                    *variable,
                    World::indexed(*world),
                    Time::Equilibrium,
                    if *counterfactual {
                        Functional::FactualCounterfactual
                    } else {
                        Functional::SpecifiedNoiseResponse
                    },
                    EvidencePort::Scenario,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::Graph(graph) => {
                let S::Graph = selector else {
                    return Err(selector_mismatch());
                };
                let mut p = point(
                    None,
                    0,
                    World::Fixed("graphical"),
                    Time::Static,
                    Functional::GraphicalEvidence,
                    EvidencePort::Scenario,
                );
                p.diagnostics = detail(graph);
                Ok(p)
            }
            A::PolynomialCausal(report) => {
                let S::PolynomialWorld { world, variable } = selector else {
                    return Err(selector_mismatch());
                };
                let mut p = point(
                    None,
                    *variable,
                    World::Declared(world.clone()),
                    Time::Algebraic,
                    Functional::CounterfactualOutcomeSet,
                    EvidencePort::Scenario,
                );
                p.samples.clear();
                p.bounds.clear();
                let mut masses = Vec::new();
                for atom in &report.atoms {
                    let bound = match atom.states.get(world).ok_or_else(selector_mismatch)? {
                        antecedent_estimate::polynomial_scm::PolynomialSelectedState::Selected {
                            enclosure,
                        } => Some(*enclosure.get(*variable).ok_or_else(selector_mismatch)?),
                        _ => None,
                    };
                    p.samples.push(None);
                    p.bounds.push(bound);
                    if let Some(mass) = atom.posterior_mass {
                        masses.push(mass);
                    }
                }
                p.diagnostics = detail(PolynomialWorldDetail {
                    world_mass: &report.world_mass,
                    positive_evidence_certified: masses.len() == report.atoms.len()
                        && masses.iter().any(|m| m.lower > 0.0),
                });
                p.masses = Some(masses);
                p.axes = vec![Axis::Predictive, Axis::Parameter];
                Ok(p)
            }
        }
    }
}
fn source_order(request: &R) -> usize {
    let request = request.recovery_recipe();
    use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
    let graph_order = |graph: &G| match graph {
        G::Identify { graph, .. }
        | G::Backdoor { graph, .. }
        | G::Adjust { graph, .. }
        | G::Calculus { graph, .. } => graph.nodes as usize,
        G::Bipartite { variables, .. } | G::DomainInvariance { variables, .. } => *variables,
        G::Discover { columns, oracle, .. } => {
            oracle.as_ref().map_or(columns.len(), |g| g.nodes as usize)
        }
    };
    match request {
        R::RetainedExperimentalRecovery { inputs, .. } => source_order(inputs.recipe()),
        R::AdamFit { model, .. }
        | R::EquilibriumRoots { model, .. }
        | R::EquilibriumActions { model, .. }
        | R::GradientFit { model, .. }
        | R::FrequentistBootstrap { model, .. }
        | R::ParameterPosterior { model, .. }
        | R::JointCounterfactual { model, .. }
        | R::PartialCounterfactual { model, .. }
        | R::StructureFit { model, .. } => model.order(),
        R::Ode { model, .. } | R::OdeFit { model, .. } => model.equations.len(),
        R::OuMechanisms { model, .. } => model.equations.len(),
        R::OuStationary { model, .. } | R::OuTransient { model, .. } | R::OuFit { model, .. } => {
            model.center.len()
        }
        R::Polynomial { model, .. } => model.variables.len(),
        R::PolynomialCausal { program } => {
            program.candidates.first().map_or(0, |c| c.model.variables.len())
        }
        R::PolynomialFit { candidates, .. } => {
            candidates.first().map_or(0, |c| c.model.variables.len())
        }
        R::Graph { request } => graph_order(request),
        R::Functional { source, .. } | R::GaussianFunctional { source, .. } => graph_order(source),
        R::RecoveredLinear { nodes, .. } => nodes.len(),
        R::ExperimentalRecovery { regimes, dropped, .. } => {
            let dropped: std::collections::BTreeSet<u32> = dropped.iter().copied().collect();
            regimes
                .iter()
                .flat_map(|regime| regime.units.iter().copied())
                .filter(|unit| !dropped.contains(unit))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        }
        R::LinearWorlds { variables, .. }
        | R::EquilibriumWorlds { variables, .. }
        | R::OdeWorlds { variables, .. } => variables.len(),
        R::DynamicInference { problem, .. } => problem.order(),
        R::System { request } => request.model.order(),
    }
}

impl PreparedStudy<CyclicFamilyPreparedState> {
    /// Bind retained native outputs to typed evidence without fitting or sampling again.
    /// # Errors
    /// Invalid schema, incompatible selectors, cancellation or resource refusal.
    pub fn cyclic_evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        self.execute_retained(ctx)?.evidence(schema, selectors, ctx)
    }
}

impl ExecutedCyclicFamily {
    /// Project this exact execution without fitting, sampling, solving or encoding.
    ///
    /// # Errors
    /// Invalid schema, incompatible selectors, cancellation or resource refusal.
    pub fn evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        CyclicEvidence::build(
            self.prepared().state.program.request(),
            self.shared_report(),
            CyclicEvidenceSource::Live(Arc::new(self.clone())),
            schema,
            selectors,
            ctx,
        )
    }
}

impl CyclicEvidence {
    #[allow(
        clippy::too_many_lines,
        reason = "source-bound evidence construction validates coordinates and retains native envelopes atomically"
    )]
    fn build(
        request: &R,
        source: &Arc<CyclicFamilyReport>,
        artifact: CyclicEvidenceSource,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        let version = if matches!(source.answer, A::ExperimentalRecovery(_)) { 3 } else { 2 };
        Self::build_version(request, source, artifact, schema, selectors, version, ctx)
    }

    fn build_version(
        request: &R,
        source: &Arc<CyclicFamilyReport>,
        artifact: CyclicEvidenceSource,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        version: u16,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        if schema.id.trim().is_empty()
            || schema.id.len() > 256
            || schema.population.len() > 256
            || schema.population.trim().is_empty()
            || schema.variables.is_empty()
            || schema.variables.len() > 128
            || schema.variables.len() != source_order(request)
            || schema.variables.iter().enumerate().any(|(i, (id, units))| {
                id.trim().is_empty()
                    || id.len() > 256
                    || units.len() > 128
                    || units.trim().is_empty()
                    || schema.variables[..i].iter().any(|(old, _)| old == id)
            })
            || selectors.is_empty()
            || selectors.len() > 1024
        {
            return Err(invalid("cyclic evidence schema/selector limits"));
        }
        let draws = match &source.answer {
            A::ParameterPosterior { worlds, .. } => worlds.first().map_or(0, Vec::len),
            A::FrequentistBootstrap { worlds, .. } => worlds.first().map_or(0, Vec::len),
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. } => {
                worlds.first().map_or(0, Vec::len)
            }
            A::PolynomialCausal(report) => report.atoms.len(),
            A::RecoveredLinear(report) => {
                report.worlds.first().map_or(0, |world| world.draws.len())
            }
            A::ExperimentalRecovery(report) => report.law.draws.len(),
            A::Worlds(worlds) => worlds
                .iter()
                .map(|world| match &world.outcome {
                    antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer::EquilibriumDraws {
                        outcomes,
                    } => outcomes.len(),
                    _ => 1,
                })
                .max()
                .unwrap_or(1),
            A::DynamicInference { inference, .. } => match inference.as_ref() {
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::SamplingLaw(law) => {
                    law.replicates.len()
                }
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::Posterior(
                    posterior,
                ) => posterior.quantity_draws.len(),
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::Information(_) => 1,
            },
            A::Polynomial(report) => report.roots.len().saturating_add(report.unresolved.len()),
            _ => 1,
        };
        if selectors.len().saturating_mul(draws) > 1_000_000 {
            return Err(invalid("cyclic evidence projection cell limit"));
        }
        let reserve = admission_reserve(
            artifact.retained_encoding_len(),
            cyclic_serialized_size(&(source.as_ref(), &schema, &selectors), ctx)?,
            4096,
        );
        let mut retention =
            vec![ctx.acquire_admission(0, reserve).map_err(CyclicArtifactError::Admission)?];
        let recovery_origins = if version == 3 {
            antecedent_io::cyclic_family_artifact::recovery_origins(request, source)?
        } else { None };
        if version == 3 && recovery_origins.is_none() { return Err(invalid("version-three evidence needs retained recovery origins")); }
        let identity = if version == 3 {
            digest("antecedent.cyclic_evidence.v2", &(&source.identity, &schema, &selectors, &recovery_origins))?
        } else {
            digest("antecedent.cyclic_evidence.v1", &(&source.identity, &schema, &selectors))?
        };
        let mut columns = Vec::new();
        let mut envelopes = Vec::new();
        for selector in &selectors {
            let p = source.answer.project(request, selector)?;
            // The typed details are serialized once, here, into the column's JSON diagnostics.
            let diagnostics = match &p.diagnostics {
                Some(detail) => detail.to_json().map_err(invalid)?,
                None => serde_json::Value::Null,
            };
            let (world, time) = (p.world.to_string(), p.time.to_string());
            let projection_bytes = cyclic_serialized_size(
                &(&diagnostics, &p.samples, &p.bounds, &p.masses, &world, &time),
                ctx,
            )? as u64;
            retention.push(
                ctx.acquire_admission(0, projection_bytes.saturating_mul(8).saturating_add(4096))
                    .map_err(CyclicArtifactError::Admission)?,
            );
            let (name, units) = schema
                .variables
                .get(p.variable)
                .ok_or_else(|| invalid("evidence variable outside declared schema"))?;
            let regime =
                digest("antecedent.cyclic_evidence.regime.v1", &(&source.identity, &world, &time))?;
            let quantity = ScientificQuantity {
                variable_id: format!("{}:{name}", schema.id),
                variable_name: name.clone(),
                role: QuantityRole::Outcome,
                units: if p.functional == Functional::IdentifiedDensity {
                    format!("1/({units})")
                } else if matches!(
                    p.functional,
                    Functional::LogLikelihood | Functional::IdentifiedProbability
                ) {
                    "dimensionless".into()
                } else if let Some(other) = p.other {
                    let second = schema
                        .variables
                        .get(other)
                        .ok_or_else(|| invalid("covariance variable outside schema"))?;
                    format!("({units})*({})", second.1)
                } else if let CyclicEvidenceSelector::OuCovariance { other, .. }
                | CyclicEvidenceSelector::OuRegime { other: Some(other), .. } = selector
                {
                    let second = schema
                        .variables
                        .get(*other)
                        .ok_or_else(|| invalid("covariance variable outside schema"))?;
                    format!("({units})*({})", second.1)
                } else {
                    units.clone()
                },
                population_id: schema.population.clone(),
                regime_id: regime,
                horizon: 0,
                functional_id: p.functional.as_str().into(),
                conditioning: vec![],
                transform_id: "identity".into(),
            };
            let premises: Arc<[Arc<str>]> = vec![
                Arc::from("conditional_on_declared_source_semantics_and_assumptions"),
                Arc::from("numerical_diagnostics_are_not_identification_certificates"),
            ]
            .into();
            let envelope = EvidenceEnvelope::try_new(
                EvidenceBinding::try_new(
                    quantity.clone(),
                    WorldTime { world_id: world.as_str().into(), time_id: time.as_str().into() },
                    premises,
                    EvidenceOrigins {
                        source: Some(recovery_origins.as_ref().map_or(source.identity.as_str(), |origins| origins.evidence.as_str()).into()),
                        selection: if matches!(
                            request,
                            R::PolynomialCausal { .. } | R::PolynomialFit { .. }
                        ) {
                            Some(source.identity.clone().into())
                        } else {
                            None
                        },
                        training: if matches!(
                            request,
                            R::AdamFit { .. }
                                | R::GaussianFunctional {law:antecedent_io::cyclic_gaussian_functional::GaussianCyclicLaw::Fit{..},..}
                                | R::OuMechanisms { .. }
                                | R::GradientFit { .. }
                                | R::OdeFit { .. }
                                | R::OuFit { .. }
                                | R::PolynomialFit { .. }
                                | R::StructureFit { .. }
                                | R::ParameterPosterior { .. }
                                | R::JointCounterfactual { .. }
                                | R::FrequentistBootstrap { .. }
                                | R::ExperimentalRecovery { .. }
                                | R::RetainedExperimentalRecovery { .. }
                                | R::DynamicInference { .. }
                                | R::System {
                                    request: antecedent_io::cyclic_system_artifact::CyclicExecutionRequest {
                                        fit: Some(_),
                                        ..
                                    }
                                }
                        ) {
                            Some(recovery_origins.as_ref().map_or(source.identity.as_str(), |origins| origins.fit.as_str()).into())
                        } else {
                            None
                        },
                        prior: if p.axes.contains(&Axis::Parameter) {
                            recovery_origins.as_ref().map_or_else(|| Some(source.identity.clone().into()), |origins| origins.prior.as_deref().map(Into::into))
                        } else {
                            None
                        },
                    },
                    EvidenceSupport {
                        status: SupportStatus::MissingEvidence,
                        domain_id: "empirical_support_not_assessed".into(),
                    },
                    EvidenceCalibration::ScopeNotAssessed,
                )
                .map_err(invalid)?,
                UncertaintyAxes::absent(),
            )
            .map_err(invalid)?;
            columns.push(CyclicEvidenceColumn {
                quantity: ScientificQuantityWire::from(&quantity),
                envelope_identity: envelope.identity().to_hex(),
                port: p.port.as_str().into(),
                uncertainty_axes: p.axes.iter().map(|axis| axis.as_str().to_owned()).collect(),
                samples: p.samples,
                numerical_enclosures: p.bounds,
                mass_bounds: p.masses,
                world,
                time,
                diagnostics,
            });
            envelopes.push(envelope);
        }
        let scope = if matches!(source.answer, A::GaussianFunctional(_)) {
            "theorem_identified_conditional_on_declared_Gaussian_observational_density"
        } else if matches!(source.answer, A::Functional(_)) {
            "theorem_identified_conditional_on_declared_observational_law"
        } else if matches!(source.answer, A::Graph(_)) {
            "theorem_scoped_graphical_evidence"
        } else if matches!(source.answer, A::ExperimentalRecovery(_)) {
            "recovered_from_declared_clamp_experiments_conditional_on_the_recipe"
        } else {
            "specified_or_fitted_model_conditional"
        }
        .into();
        Ok(Self {
            report: CyclicEvidenceReport {
                identity,
                recovery_origins,
                source_identity: source.identity.clone(),
                scope,
                columns,
                source: Arc::clone(source),
            },
            source: artifact,
            schema,
            selectors,
            envelopes,
            version, original_encoding: None,
            _retention: Arc::new(retention),
        })
    }
    /// Checked projection, with failures and numerical enclosures in original slots.
    #[must_use]
    pub fn report(&self) -> &CyclicEvidenceReport {
        &self.report
    }
    /// Projection identity without serializing samples or the source report.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.report.identity
    }

    /// Explicit provenance when the source is the family bridge of a system.
    /// The original system ID is retained inside the verified family report.
    #[must_use]
    pub fn system_bridge(&self) -> Option<CyclicSystemBridgeProvenance> {
        match &self.report.source.answer {
            A::System(system) => {
                Some(system_bridge_provenance(&system.identity, &self.report.source_identity))
            }
            _ => None,
        }
    }

    /// Original source identity without materializing the source payload.
    #[must_use]
    pub fn source_identity(&self) -> &str {
        &self.report.source_identity
    }

    /// Scientific scope; this never upgrades specified models to identification.
    #[must_use]
    pub fn scope(&self) -> &str {
        &self.report.scope
    }

    /// Borrow projected coordinates and their original slot/port diagnostics.
    #[must_use]
    pub fn columns(&self) -> &[CyclicEvidenceColumn] {
        &self.report.columns
    }

    /// Read scientific coordinates without copying or serializing any samples.
    pub fn quantities(&self) -> impl ExactSizeIterator<Item = &ScientificQuantityWire> {
        self.report.columns.iter().map(|column| &column.quantity)
    }

    /// Existing contract envelopes, suitable for evidence binding/port checks.
    #[must_use]
    pub fn envelopes(&self) -> &[EvidenceEnvelope] {
        &self.envelopes
    }
    /// Export the original source plus projection; independent loading re-executes the source.
    /// # Errors
    /// Resource or container/size refusal.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        if let Some((original,_owner)) = &self.original_encoding {
            let _output = ctx.acquire_admission(0, original.len() as u64).map_err(CyclicArtifactError::Admission)?;
            return Ok(original.to_vec());
        }
        let source = self.source.encoding(ctx)?;
        let _admission = ctx
            .acquire_admission(
                0,
                admission_reserve(
                    source.len(),
                    cyclic_serialized_size(&(&self.report, &self.schema, &self.selectors), ctx)?,
                    65536,
                ),
            )
            .map_err(CyclicArtifactError::Admission)?;
        let bytes = antecedent_io::to_cbor(&Record {
            version: self.version,
            source: source.into_owned(),
            source_identity: self.report.source_identity.clone(),
            schema: self.schema.clone(),
            selectors: self.selectors.clone(),
            report: self.report.clone(),
        })
        .map_err(invalid)?;
        if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("cyclic evidence byte limit"));
        }
        let _lease = ctx
            .acquire_admission(0, (bytes.len() as u64).saturating_mul(4))
            .map_err(CyclicArtifactError::Admission)?;
        antecedent_io::encode_container(
            &antecedent_io::ArtifactContainer {
                kind: "cyclic_evidence",
                note: "typed_native_cyclic_projection",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: &self.report.identity,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io::OutSection {
                id: "cyclic_evidence_v1",
                content_type: antecedent_io::CONTENT_BYTES,
                bytes: &bytes,
            }],
        )
        .map_err(|e| invalid(e.into_io("cyclic evidence layout")))
    }
    /// Independently verify source, target, projection, uncertainty meaning and numerical output.
    /// # Errors
    /// Identity mismatch, malformed envelope, replay failure, cancellation or resources.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        if expected_identity.len() != 64
            || !expected_identity.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid("cyclic evidence identity"));
        }
        if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("cyclic evidence byte limit"));
        }
        let _lease = cyclic_decode_admission(bytes, ctx)?;
        let parts = antecedent_io::decode_container(
            "cyclic_evidence",
            MAX_CYCLIC_ARTIFACT_BYTES,
            &["cyclic_evidence_v1"],
            bytes,
        )
        .map_err(|e| invalid(e.into_io("cyclic evidence layout")))?;
        let record: Record = antecedent_io::from_cbor(
            parts.first().ok_or_else(|| invalid("missing evidence section"))?,
        )
        .map_err(invalid)?;
        if !matches!(record.version, 2 | 3)
            || record.report.identity != expected_identity
            || (if record.version == 3 {
                digest("antecedent.cyclic_evidence.v2", &(&record.source_identity, &record.schema, &record.selectors, &record.report.recovery_origins))?
            } else {
                if record.report.recovery_origins.is_some() { return Err(invalid("historical evidence has no version-three origin standing")); }
                digest("antecedent.cyclic_evidence.v1", &(&record.source_identity, &record.schema, &record.selectors))?
            }) != expected_identity
        {
            return Err(invalid("cyclic evidence identity mismatch"));
        }
        let (request, source) =
            consume_cyclic_family_with_request(&record.source, &record.source_identity, ctx)?;
        let mut evidence = Self::build_version(
            &request,
            &Arc::new(source),
            CyclicEvidenceSource::Replayed(record.source.into()),
            record.schema,
            record.selectors,
            record.version,
            ctx,
        )?;
        if evidence.report != record.report {
            return Err(invalid("cyclic evidence differs from original source replay"));
        }
        let owner = Arc::new(ctx.acquire_admission(0,bytes.len() as u64).map_err(CyclicArtifactError::Admission)?);
        evidence.original_encoding = Some((Arc::from(bytes),owner));
        Ok(evidence)
    }
    /// Existing decision engine's affine-mean route, conditional on the reported source.
    /// A posterior/ bootstrap column is not silently averaged into an outcome law.
    /// # Errors
    /// Wrong evidence port, unresolved slots, incompatible decision contract or numeric failure.
    pub fn decide_means(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        let mut coordinates = Vec::new();
        let mut means = Vec::new();
        for column in &self.report.columns {
            if column.port != "mean" || column.samples.len() != 1 || column.samples[0].is_none() {
                return Err(invalid(
                    "decision needs point/mean evidence; unresolved states and other uncertainty ports are retained, not coerced",
                ));
            }
            coordinates
                .push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
            means.push(column.samples[0].ok_or_else(|| invalid("missing mean"))?);
        }
        let source = antecedent_design::decision_eval::MeanSource {
            coordinates,
            means,
            estimate: antecedent_design::decision_eval::MeanEstimate::Frequentist,
            provider_id: "native:cyclic_model_conditional".into(),
            snapshot_id: self.report.source_identity.clone(),
            causal_contract_id: self.report.identity.clone(),
            rng_id: "deterministic_native_projection".into(),
        };
        let mut result =
            antecedent_design::decision_eval::evaluate_contract_on_means(contract, &source)
                .map_err(|e| invalid(format!("{e:?}")))?;
        result.assumptions.push(self.report.scope.clone());
        let result = antecedent_design::decision_artifact::mean_result_to_json(&result, &source)
            .map_err(invalid)?;
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":self.report.scope,"result":serde_json::from_str::<serde_json::Value>(&result).map_err(invalid)?}),
        )
    }
    /// Evaluate conservative solution/scenario utility enclosures with the existing
    /// robust set decision engine. No equilibrium probabilities or nonparametric
    /// identification claim are inferred. Unresolved boxes remain inside the set.
    /// # Errors
    /// Missing/undefined states, incompatible quantities, probability criteria,
    /// hard chance constraints or an invalid existing decision contract.
    pub fn decide_enclosures(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        use antecedent_design::{
            decision_adapters::{IdentifiedUtility, evaluate_identified_sets},
            decision_contract::{AdmissibilityRules, AdmissibleDecisionContract},
            decision_robustness::AtomSupport,
        };
        if !contract.constraints.is_empty() {
            return Err(invalid(
                "a solution enclosure does not supply a chance-constraint probability law",
            ));
        }
        let coordinates = self
            .report
            .columns
            .iter()
            .map(|c| ScientificQuantity::try_from(c.quantity.clone()).map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let mut utilities = Vec::new();
        for action in &contract.actions {
            let mut inputs = Vec::new();
            for quantity in &action.inputs {
                let i = coordinates
                    .iter()
                    .position(|c| c.require_same_coordinate(quantity).is_ok())
                    .ok_or_else(|| {
                        invalid("solution decision quantity does not match original evidence")
                    })?;
                let column = &self.report.columns[i];
                if column.port != "scenario" || column.numerical_enclosures.is_empty() {
                    return Err(invalid(
                        "solution decision requires a declared numerical solution/scenario enclosure",
                    ));
                }
                if column.mass_bounds.is_some()
                    && (column
                        .diagnostics
                        .get("positive_evidence_certified")
                        .and_then(serde_json::Value::as_bool)
                        != Some(true)
                        || column
                            .mass_bounds
                            .as_ref()
                            .is_some_and(|m| m.len() != column.numerical_enclosures.len()))
                {
                    return Err(invalid("posterior selection evidence is not certified positive"));
                }
                let mut lower = f64::INFINITY;
                let mut upper = f64::NEG_INFINITY;
                for (k, bound) in column.numerical_enclosures.iter().enumerate() {
                    if column
                        .mass_bounds
                        .as_ref()
                        .is_some_and(|m| m.get(k).is_some_and(|p| p.upper == 0.0))
                    {
                        continue;
                    }
                    let value=bound.ok_or_else(||invalid("undefined or unresolved selected-state mass has no decision utility enclosure"))?;
                    lower = lower.min(value.lower);
                    upper = upper.max(value.upper);
                }
                if !lower.is_finite() || !upper.is_finite() {
                    return Err(invalid("empty/undefined solution enclosure"));
                }
                inputs.push(Interval { lower, upper });
            }
            let enclosure = enclose_utility(&action.utility, &inputs)?;
            utilities.push(IdentifiedUtility {
                action_id: action.id.clone(),
                utility: antecedent_core::IdentifiedSet::try_new(enclosure.lower, enclosure.upper)
                    .map_err(invalid)?,
                hard_exclusions: vec![],
                support: AtomSupport::unassessed(),
            });
        }
        let result = evaluate_identified_sets(
            &AdmissibleDecisionContract {
                contract: contract.clone(),
                rules: AdmissibilityRules::default(),
            },
            &utilities,
        )
        .map_err(|e| invalid(format!("{e:?}")))?;
        let actions=result.actions.iter().map(|a|serde_json::json!({"id":a.id,"utility":a.utility,"eligible":a.eligible,"dominated_by":a.dominated_by,"possibly_optimal":a.possibly_optimal,"necessarily_optimal":a.necessarily_optimal,"max_regret":a.max_regret})).collect::<Vec<_>>();
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":"conditional_model_solution_outer_enclosure","actions":actions,"verdict":format!("{:?}",result.verdict),"lower_leader":result.lower_leader,"upper_leader":result.upper_leader,"conflicting_leaders":result.conflicting_leaders,"selection_probabilities_inferred":false}),
        )
    }
    /// Build the existing aligned joint-law artifact for a complete posterior column set.
    /// Bootstrap, root sets, finite mass bounds and failed slots cannot be coerced to a belief law.
    /// # Errors
    /// Different axes/alignments, missing mass, wrong port or invalid artifact coordinates.
    pub fn joint_law(&self) -> Result<DistributionArtifact, CyclicArtifactError> {
        let columns = self.report.columns.iter().collect::<Vec<_>>();
        let provenance = if let Some(origin) = &self.report.recovery_origins {
            recovery_distribution_provenance(origin)
        } else {
            DistributionProvenance {
                source_id: self.report.source_identity.clone(),
                provider_id: "native:cyclic_model_conditional".into(),
                rng_id: format!("retained:{}", self.report.source_identity),
                snapshot_id: self.report.source_identity.clone(),
                causal_contract_id: self.report.identity.clone(),
            }
        };
        cyclic_joint_columns(&columns,provenance,self.version == 3)
    }

    /// Compose explicit native evidence objects only under an established draw pairing.
    /// Source evidence, statistical fit and row-alignment identities are separate.
    /// # Errors
    /// Shared evidence declared independent, unknown/mismatched pairing, failed slots,
    /// incompatible quantity declarations or an uncertainty port without an outcome law.
    pub fn joint_law_with(&self, others: &[&Self], dependence: Option<&str>) -> Result<DistributionArtifact,CyclicArtifactError> {
        if others.len() > 256 { return Err(invalid("at most 256 additional evidence products")); }
        if others.is_empty() && dependence.is_none() { return self.joint_law(); }
        let sources = std::iter::once(self).chain(others.iter().copied()).collect::<Vec<_>>();
        if dependence == Some("independent") {
            for (i,first) in sources.iter().enumerate() {
                for second in &sources[i+1..] {
                    if let (Some(a),Some(b)) = (&first.report.recovery_origins,&second.report.recovery_origins) {
                        if a.evidence == b.evidence {
                            return Err(joint_refusal(antecedent_core::reason_code!("evidence_double_counted"),
                                "cyclic_evidence.shared_training_declared_independent: these products read the same experiment evidence; supply their verified aligned joint law"));
                        }
                    }
                }
            }
            return Err(joint_refusal(antecedent_core::reason_code!("evidence_overlap_unknown"),
                "cyclic_evidence.independence_not_established: different labels do not prove disjoint observations or independent laws; supply an explicit joint law"));
        }
        if dependence != Some("aligned") {
            return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.alignment_declaration_required: select aligned only for native products carrying the same ordered-law pairing proof"));
        }
        let origin = self.report.recovery_origins.as_ref().ok_or_else(|| joint_refusal(
            antecedent_core::reason_code!("joint_law_required"),
            "cyclic_evidence.legacy_alignment_unavailable: this historical envelope does not attest cross-product recovery origins; independently replay its source and explicitly project new evidence"))?;
        let mut columns = Vec::new();
        for source in sources {
            let found = source.report.recovery_origins.as_ref().ok_or_else(|| joint_refusal(
                antecedent_core::reason_code!("joint_law_required"), "cyclic_evidence.alignment_unavailable: independently project version-three evidence from its verified source"))?;
            if found.alignment != origin.alignment || found.fit != origin.fit || found.law != origin.law {
                return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.draw_alignment_mismatch: equal draw counts are insufficient; provide the actual shared ordered law or an explicit coupling"));
            }
            columns.extend(source.report.columns.iter());
        }
        cyclic_joint_columns(&columns,recovery_distribution_provenance(origin),true)
    }

}

fn joint_refusal(code: &'static str, message: &str) -> CyclicArtifactError {
    CyclicArtifactError::ScientificRefusal {
        code,detail:"cyclic_evidence.joint_law",message:message.into(),
        next_attempt:"declare an eligible shared joint law with compatible quantities, units, populations and uncertainty ports".into(),
    }
}
fn recovery_distribution_provenance(origin: &antecedent_io::cyclic_family_artifact::CyclicRecoveryOrigins) -> DistributionProvenance {
    DistributionProvenance {
        source_id: origin.evidence.clone(), provider_id: "native:cyclic_model_conditional".into(),
        rng_id: origin.alignment.clone(), snapshot_id: origin.law.clone(),
        // The fitting contract is shared; each target remains in its quantity/regime identity.
        causal_contract_id: origin.fit.clone(),
    }
}
fn cyclic_joint_columns(columns: &[&CyclicEvidenceColumn], provenance: DistributionProvenance,
    exact_recovery_weights: bool,
) -> Result<DistributionArtifact,CyclicArtifactError> {
    let first = columns.first().ok_or_else(||invalid("empty evidence"))?;
    if !matches!(first.port.as_str(),"parameter_posterior"|"interventional_law") {
        return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
            "cyclic_evidence.uncertainty_port: sampling laws and means cannot become posterior or outcome distributions"));
    }
    let count = first.samples.len();
    if count == 0 { return Err(invalid("empty law")); }
    use antecedent_io::distribution_artifact::{MAX_DISTRIBUTION_COORDINATES,MAX_DISTRIBUTION_DRAWS,MAX_DISTRIBUTION_ARTIFACT_BYTES};
    if columns.len() > MAX_DISTRIBUTION_COORDINATES || count > MAX_DISTRIBUTION_DRAWS
        || count.checked_mul(columns.len()).and_then(|cells|cells.checked_mul(std::mem::size_of::<f64>()))
            .is_none_or(|bytes|bytes > MAX_DISTRIBUTION_ARTIFACT_BYTES) {
        return Err(invalid("joint law output exceeds native distribution limits"));
    }
    let weights_of = |column: &CyclicEvidenceColumn| -> Result<Option<Vec<f64>>,CyclicArtifactError> {
        let Some(bounds) = &column.mass_bounds else { return Ok(None); };
        // Only an actual posterior of recovered functional draws is licensed
        // here. A Gaussian mean with variance in diagnostics is not an outcome draw.
        if !exact_recovery_weights || column.port != "parameter_posterior" || bounds.len() != count {
            return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.mass_or_mean_not_a_law: numerical sets, uncertain mass and projected Gaussian means require their scientific law adapter"));
        }
        let mut weights = Vec::with_capacity(count);
        for mass in bounds {
            if mass.lower != mass.upper || !mass.lower.is_finite() || mass.lower < 0.0 {
                return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.nonexact_mass: preserve mass bounds and provide a complete weighted law"));
            }
            weights.push(mass.lower);
        }
        if (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.incomplete_mass: failed or withheld mass cannot be discarded"));
        }
        Ok(Some(weights))
    };
    let weights = weights_of(first)?;
    let population = &first.quantity.population_id;
    let mut quantities = Vec::with_capacity(columns.len());
    for column in columns {
        if column.port != first.port || column.uncertainty_axes != first.uncertainty_axes
            || column.samples.len() != count || column.samples.iter().any(Option::is_none)
            || weights_of(column)? != weights {
            return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.incomplete_or_misaligned_law: retain every original draw/weight, failed slot and uncertainty meaning"));
        }
        if &column.quantity.population_id != population {
            return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                "cyclic_evidence.population_bridge_required: declare a scientific population bridge before composing these quantities"));
        }
        for previous in &quantities {
            let previous: &ScientificQuantity = previous;
            if previous.variable_name == column.quantity.variable_name && previous.units != column.quantity.units {
                return Err(joint_refusal(antecedent_core::reason_code!("joint_law_required"),
                    "cyclic_evidence.units_bridge_required: declare a unit conversion before composing the same variable under incompatible units"));
            }
        }
        quantities.push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
    }
    let mut rows = Vec::with_capacity(count.checked_mul(columns.len()).ok_or_else(||invalid("joint law shape overflow"))?);
    for row in 0..count {
        for column in columns { rows.push(column.samples[row].ok_or_else(||invalid("missing draw"))?); }
    }
    let meaning = if first.port == "parameter_posterior" {
        DistributionMeaningWire::CausalFunctionalPosterior
    } else { DistributionMeaningWire::InterventionalPredictive };
    let identity = DistributionIdentity::new(meaning,&quantities,DrawAlignment::Joint,provenance).map_err(invalid)?;
    DistributionArtifact::new(DistributionMetadata {
        version:1,identity,axes:["draw".into(),"quantity".into()],shape:[count,quantities.len()],
        weights,supported:None,calibration:DistributionCalibration::Unmeasured,
        trust:DistributionTrust::Unverified,legacy_posterior:None,legacy_bindings:None,
    },rows).map_err(invalid)
}

fn enclose_utility(
    expression: &antecedent_design::decision_contract::UtilityExpr,
    inputs: &[Interval],
) -> Result<Interval, CyclicArtifactError> {
    use antecedent_design::decision_contract::UtilityExpr as U;
    let output = match expression {
        U::Const(value) => Interval { lower: *value, upper: *value },
        U::Input(i) => *inputs.get(*i).ok_or_else(|| invalid("utility input index"))?,
        U::Neg(a) => enclose_utility(a, inputs)?.scale_outward(-1.0),
        U::Add(a, b) => enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?),
        U::Sub(a, b) => {
            enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?.scale_outward(-1.0))
        }
        U::Mul(a, b) => enclose_utility(a, inputs)?.multiply_outward(enclose_utility(b, inputs)?),
        U::Min(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.min(b.lower), upper: a.upper.min(b.upper) }
        }
        U::Max(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.max(b.lower), upper: a.upper.max(b.upper) }
        }
    };
    if !output.lower.is_finite() || !output.upper.is_finite() {
        return Err(invalid("nonfinite utility enclosure"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Study;
    #[test]
    fn system_evidence_bridge_preserves_original_source_and_loaded_ownership() {
        use antecedent_model::nonlinear_equilibrium::{
            Activation, Equation, EquilibriumModel, SolvePolicy,
        };
        let ctx = ExecutionContext::for_tests(8);
        let request = antecedent_io::cyclic_system_artifact::CyclicExecutionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "A".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.4],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                    Equation {
                        id: "B".into(),
                        intercept: 0.0,
                        weights: vec![0.5, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                ],
                noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            },
            fit: None,
            worlds: vec![vec![]],
            noise: vec![vec![1.0, 0.0]],
            initial: vec![0.0; 2],
            policy: SolvePolicy::default(),
            factual: None,
        };
        let prepared = Study::prepare_cyclic(request, &ctx).unwrap();
        let executed = prepared.execute_retained(&ctx).unwrap();
        let source_identity = executed.identity().to_owned();
        let original_bytes = executed.export("cyclic_equilibrium", &ctx).unwrap();
        let bridge = executed.family_bridge(&ctx).unwrap();
        assert!(Arc::ptr_eq(&bridge, &executed.family_bridge(&ctx).unwrap()));
        let A::System(shared) = &bridge.report().answer else {
            panic!("system bridge");
        };
        assert!(Arc::ptr_eq(shared, executed.shared_report()));
        assert_eq!(bridge.provenance().source_identity, source_identity);
        assert_eq!(
            bridge.provenance().family_identity,
            bridge.contract_metadata().request_identity
        );
        assert_ne!(bridge.provenance().family_identity, source_identity);
        let schema = CyclicEvidenceSchema {
            id: "physical-system".into(),
            population: "system".into(),
            variables: vec![("A".into(), "meter".into()), ("B".into(), "meter".into())],
        };
        let selectors = vec![CyclicEvidenceSelector::Action {
            world: 0,
            noise_row: 0,
            variable: 1,
            counterfactual: false,
        }];
        let evidence = executed.evidence(schema.clone(), selectors.clone(), &ctx).unwrap();
        assert_eq!(evidence.system_bridge().as_ref(), Some(bridge.provenance()));
        assert!((evidence.columns()[0].samples[0].unwrap() - 0.625).abs() < 1e-8);
        assert_eq!(executed.export("cyclic_equilibrium", &ctx).unwrap(), original_bytes);
        let evidence_bytes = evidence.export(&ctx).unwrap();
        let loaded =
            ConsumedCyclicSystem::consume(&original_bytes, &source_identity, &ctx).unwrap();
        let loaded_bridge = loaded.family_bridge(&ctx).unwrap();
        let A::System(loaded_shared) = &loaded_bridge.report().answer else {
            panic!("loaded system bridge");
        };
        assert!(Arc::ptr_eq(loaded_shared, &loaded.report));
        assert_eq!(loaded_bridge.provenance(), bridge.provenance());
        drop(prepared);
        drop(executed);
        drop(bridge);
        drop(evidence);
        let composed = loaded.evidence(schema, selectors, &ctx).unwrap();
        drop(loaded);
        drop(loaded_bridge);
        assert_eq!(composed.export(&ctx).unwrap(), evidence_bytes);
        let consumed = CyclicEvidence::consume(&evidence_bytes, composed.identity(), &ctx).unwrap();
        assert_eq!(consumed.report(), composed.report());
        assert_eq!(consumed.system_bridge(), composed.system_bridge());
    }

    #[test]
    fn world_and_time_labels_keep_their_identity_spellings() {
        assert_eq!(World::Noise { world: 2, noise_row: 3 }.to_string(), "world:2:noise:3");
        assert_eq!(World::Draw { world: 1, draw: 4 }.to_string(), "world:1:draw:4");
        assert_eq!(World::indexed(5).to_string(), "world:5");
        assert_eq!(World::Equation(0).to_string(), "equation-world:0");
        assert_eq!(World::Root(2).to_string(), "root_enclosure:2");
        assert_eq!(World::Quantity(1).to_string(), "quantity:1");
        assert_eq!(World::Assignment(7).to_string(), "assignment:7");
        assert_eq!(Time::Step(3).to_string(), "time:3");
        assert_eq!(Time::At(0.5).to_string(), "t:0.5");
        assert_eq!(Time::Equilibrium.to_string(), "equilibrium");
        assert_eq!(Functional::IdentifiedDensity.as_str(), "identified_density");
        assert_eq!(Axis::Parameter.as_str(), "parameter");
    }

    #[test]
    fn consumed_contract_metadata_retains_verified_v2_and_historical_v1_absence() {
        let ctx = ExecutionContext::for_tests(17);
        let request = R::OuStationary {
            model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck {
                drift: vec![-2.0],
                center: vec![3.0],
                diffusion: vec![4.0],
            },
        };
        // Export now writes the contract-bearing record, so the historical record is the
        // golden the previous-version encoder wrote (see `antecedent-io`'s previous-version
        // tests); its identity is that of the request the golden pins.
        let v1: &[u8] = include_bytes!("../../../antecedent-io/tests/fixtures/cyclic_family_v1.bin");
        let v1_identity = "4ee65847b277589a4dae5e547f4b6999fee7ae504965c5c4be71aa0e5037cde7";
        let old = ConsumedCyclicFamily::consume(v1, v1_identity, &ctx).unwrap();
        assert!(old.contract_metadata().is_none());
        assert_eq!(old.artifact(), v1);
        let (report, current) = request.export(&ctx).unwrap();
        let current = ConsumedCyclicFamily::consume(&current, &report.identity, &ctx).unwrap();
        assert!(current.contract_metadata().is_some());
        let prepared = Study::prepare_cyclic_family(&request, &ctx).unwrap();
        let executed = prepared.execute_retained(&ctx).unwrap();
        let bytes = executed.export(&ctx).unwrap();
        let loaded = ConsumedCyclicFamily::consume(&bytes, executed.identity(), &ctx).unwrap();
        assert_eq!(loaded.contract_metadata(), Some(&prepared.contract_metadata().section));
        let cloned = loaded.clone();
        assert!(std::ptr::eq(
            loaded.contract_metadata().unwrap(),
            cloned.contract_metadata().unwrap()
        ));
        let observed = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            assert_eq!(loaded.contract_metadata(), Some(&prepared.contract_metadata().section));
            assert_eq!(loaded.report(), executed.report());
            Ok::<_, CyclicArtifactError>(())
        });
        assert!(observed.result.is_ok());
        assert!(observed.report.is_empty());
        assert_eq!(loaded.artifact(), bytes.as_slice());
    }

    #[test]
    fn executed_evidence_shares_source_and_encodes_lazily_without_reexecution() {
        use antecedent_core::execution_attempt::{Operation, observe_with_context};

        let ctx = ExecutionContext::for_tests(8);
        let request = R::OuStationary {
            model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck {
                drift: vec![-2.0],
                center: vec![3.0],
                diffusion: vec![4.0],
            },
        };
        let prepared = Study::prepare_cyclic_family(&request, &ctx).unwrap();
        let executed = prepared.execute_retained(&ctx).unwrap();
        let source_bytes = executed.export(&ctx).unwrap();
        let schema = CyclicEvidenceSchema {
            id: "physical-system".into(),
            population: "system".into(),
            variables: vec![("position".into(), "meter".into())],
        };
        let selectors = vec![CyclicEvidenceSelector::OuMean { time: None, variable: 0 }];
        let count = super::super::cyclic::cyclic_family_execution_count();
        let projection = observe_with_context(&ctx, || {
            executed.evidence(schema.clone(), selectors.clone(), &ctx)
        });
        for operation in [
            Operation::AdjustedFit,
            Operation::PosteriorFit,
            Operation::PosteriorDraw,
            Operation::LeastSquaresSolve,
        ] {
            assert_eq!(projection.report.counts(operation).attempted, 0);
        }
        let evidence = projection.result.unwrap();
        assert!(matches!(evidence.source, CyclicEvidenceSource::Live(_)));
        assert!(Arc::ptr_eq(&evidence.report.source, executed.shared_report()));
        assert_eq!(evidence.identity(), evidence.report.identity);
        assert_eq!(evidence.source_identity(), executed.identity());
        assert_eq!(evidence.quantities().len(), 1);
        assert_eq!(evidence.columns()[0].samples, vec![Some(3.0)]);
        assert_eq!(super::super::cyclic::cyclic_family_execution_count(), count);

        // The former byte-backed construction must produce exactly the same record.
        let byte_backed = CyclicEvidence::build(
            &request,
            executed.shared_report(),
            CyclicEvidenceSource::Replayed(source_bytes.into()),
            schema,
            selectors,
            &ctx,
        )
        .unwrap();
        let original_bytes = byte_backed.export(&ctx).unwrap();
        let encoded = observe_with_context(&ctx, || evidence.export(&ctx));
        for operation in [
            Operation::AdjustedFit,
            Operation::PosteriorFit,
            Operation::PosteriorDraw,
            Operation::LeastSquaresSolve,
        ] {
            assert_eq!(encoded.report.counts(operation).attempted, 0);
        }
        let bytes = encoded.result.unwrap();
        assert_eq!(bytes, original_bytes);
        assert_eq!(super::super::cyclic::cyclic_family_execution_count(), count);
        drop(prepared);
        drop(executed);
        drop(byte_backed);
        // The live evidence itself pins the original report, prepared source and leases.
        assert_eq!(evidence.export(&ctx).unwrap(), bytes);
        assert_eq!(super::super::cyclic::cyclic_family_execution_count(), count);
        let consumed = CyclicEvidence::consume(&bytes, evidence.identity(), &ctx).unwrap();
        assert!(matches!(consumed.source, CyclicEvidenceSource::Replayed(_)));
        assert_eq!(consumed.report(), evidence.report());
        assert_eq!(consumed.export(&ctx).unwrap(), bytes);
    }

    #[test]
    fn retained_evidence_preserves_ports_identity_and_replay_without_calibration_gate() {
        let ctx = ExecutionContext::for_tests(8);
        let request = R::OuStationary {
            model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck {
                drift: vec![-2.0],
                center: vec![3.0],
                diffusion: vec![4.0],
            },
        };
        let prepared = Study::prepare_cyclic_family(&request, &ctx).unwrap();
        let schema = CyclicEvidenceSchema {
            id: "physical-system".into(),
            population: "system".into(),
            variables: vec![("position".into(), "meter".into())],
        };
        let selectors = vec![CyclicEvidenceSelector::OuMean { time: None, variable: 0 }];
        let evidence = prepared.cyclic_evidence(schema.clone(), selectors.clone(), &ctx).unwrap();
        assert_eq!(prepared.inspect().preparation_evaluations, 0);
        assert_eq!(evidence.report.columns[0].samples, vec![Some(3.0)]);
        assert_eq!(evidence.report.columns[0].port, "mean");
        assert!(matches!(
            evidence.envelopes[0].binding().calibration,
            EvidenceCalibration::ScopeNotAssessed
        ));
        assert!(evidence.envelopes[0].binding().origins.training.is_none());
        let bytes = evidence.export(&ctx).unwrap();
        let consumed = CyclicEvidence::consume(&bytes, &evidence.report.identity, &ctx).unwrap();
        assert_eq!(consumed.report, evidence.report);
        assert!(CyclicEvidence::consume(&bytes, &"0".repeat(64), &ctx).is_err());
        let mut different = schema;
        different.variables[0].1 = "centimeter".into();
        let refreshed = prepared.cyclic_evidence(different, selectors, &ctx).unwrap();
        assert_ne!(refreshed.report.identity, evidence.report.identity);
        assert!(evidence.joint_law().is_err());
        ctx.cancellation.cancel();
        assert!(
            prepared
                .cyclic_evidence(refreshed.schema.clone(), refreshed.selectors.clone(), &ctx)
                .is_err()
        );
    }
}

#[cfg(test)]
mod recovery_origin_tests {
    use super::*;
    use crate::Study;
    use antecedent_io::cyclic_family_artifact::{CyclicWorldWire,CyclicInterventionWire,RecoveryInferenceWire,RecoveryPriorWire,CoefficientPriorWire,DisturbancePriorWire};

    fn request(seed: u64) -> R {
        let mut request: R = serde_json::from_str(include_str!("../../../../parity/cyclic/recovered-asymptotic-normal/request.json")).unwrap();
        let R::ExperimentalRecovery { inference, .. } = &mut request else { panic!("experimental fixture") };
        *inference = RecoveryInferenceWire::Posterior { prior: RecoveryPriorWire {
            coefficients: CoefficientPriorWire::Flat,
            disturbance: DisturbancePriorWire::Known { variances: vec![1.0,1.0] },
        }, draws: 8, burn_in: 0, seed };
        request
    }
    fn schema(id: &str) -> CyclicEvidenceSchema {
        CyclicEvidenceSchema { id:id.into(),population:"original population".into(),variables:vec![("A".into(),"meter".into()),("B".into(),"meter".into())] }
    }
    fn selector() -> Vec<CyclicEvidenceSelector> {
        vec![CyclicEvidenceSelector::Response { world:0,noise_row:0,variable:1 }]
    }
    fn branch(source: &ExecutedCyclicFamily, level: f64, ctx: &ExecutionContext) -> ExecutedCyclicFamily {
        source.prepare_recovered_worlds(vec![CyclicWorldWire { world:47,
            interventions:vec![CyclicInterventionWire::Set { variable:0,value:level }],initial_states:vec![],
        }],1,ctx).unwrap().execute_retained(ctx).unwrap()
    }
    #[test]
    fn world_branches_share_actual_evidence_and_draw_alignment_but_keep_targets_distinct() {
        let ctx = ExecutionContext::for_tests(19);
        let source = Study::prepare_cyclic_family(&request(19),&ctx).unwrap().execute_retained(&ctx).unwrap();
        let left = branch(&source,3.0,&ctx).evidence(schema("A3"),selector(),&ctx).unwrap();
        let right = branch(&source,4.0,&ctx).evidence(schema("A4"),selector(),&ctx).unwrap();
        assert_ne!(left.source_identity(),right.source_identity());
        assert_ne!(left.identity(),right.identity());
        let a=left.report.recovery_origins.as_ref().unwrap();
        let b=right.report.recovery_origins.as_ref().unwrap();
        assert_eq!(a,b);
        assert_ne!(a.evidence,left.source_identity());
        assert_eq!(left.envelopes()[0].binding().origins.training.as_deref(),Some(a.fit.as_str()));
        let error=left.joint_law_with(&[&right],Some("independent")).unwrap_err().to_string();
        assert!(error.contains("reason=evidence_double_counted"));
        let joint=left.joint_law_with(&[&right],Some("aligned")).unwrap();
        assert_eq!(joint.shape(),[8,2]);
        for (row,values) in joint.draws().chunks_exact(2).enumerate() {
            assert_eq!(values[0],left.report.columns[0].samples[row].unwrap());
            assert_eq!(values[1],right.report.columns[0].samples[row].unwrap());
        }
        assert!(joint.metadata().weights.is_some());
        let encoded=right.export(&ctx).unwrap();
        let loaded=CyclicEvidence::consume(&encoded,right.identity(),&ctx).unwrap();
        assert_eq!(loaded.report.recovery_origins,right.report.recovery_origins);
        assert_eq!(loaded.export(&ctx).unwrap(),encoded);
        let mut tampered=right.clone();
        tampered.report.recovery_origins.as_mut().unwrap().fit="0".repeat(64);
        tampered.report.identity=digest("antecedent.cyclic_evidence.v2",&(&tampered.report.source_identity,&tampered.schema,&tampered.selectors,&tampered.report.recovery_origins)).unwrap();
        let resealed=tampered.export(&ctx).unwrap();
        assert!(CyclicEvidence::consume(&resealed,tampered.identity(),&ctx).is_err());
        assert_eq!(left.joint_law_with(&[&loaded],Some("aligned")).unwrap(),joint);
        let different=Study::prepare_cyclic_family(&request(20),&ctx).unwrap().execute_retained(&ctx).unwrap();
        let misaligned=branch(&different,4.0,&ctx).evidence(schema("other-seed"),selector(),&ctx).unwrap();
        assert_eq!(misaligned.report.recovery_origins.as_ref().unwrap().evidence,a.evidence);
        assert!(left.joint_law_with(&[&misaligned],Some("aligned")).unwrap_err().to_string().contains("draw_alignment_mismatch"));
        assert!(left.joint_law_with(&[&misaligned],Some("independent")).unwrap_err().to_string().contains("evidence_double_counted"));
    }
    #[test]
    fn historical_v2_envelope_bytes_identity_and_origin_standing_are_preserved() {
        let ctx=ExecutionContext::for_tests(19);
        let executed=Study::prepare_cyclic_family(&request(19),&ctx).unwrap().execute_retained(&ctx).unwrap();
        let legacy=CyclicEvidence::build_version(executed.request(),executed.shared_report(),
            CyclicEvidenceSource::Live(Arc::new(executed.clone())),schema("historical"),selector(),2,&ctx).unwrap();
        #[derive(Serialize)]
        struct HistoricalReport<'a> { identity:&'a str,source_identity:&'a str,scope:&'a str,columns:&'a [CyclicEvidenceColumn],source:&'a CyclicFamilyReport }
        let report=&legacy.report;
        let historical=HistoricalReport { identity:&report.identity,source_identity:&report.source_identity,
            scope:&report.scope,columns:&report.columns,source:&report.source };
        assert_eq!(antecedent_io::to_cbor(&historical).unwrap(),antecedent_io::to_cbor(report).unwrap());
        assert!(legacy.report.recovery_origins.is_none());
        assert_eq!(legacy.identity(),digest("antecedent.cyclic_evidence.v1",&(&report.source_identity,&legacy.schema,&legacy.selectors)).unwrap());
        assert_eq!(legacy.envelopes()[0].binding().origins.training.as_deref(),Some(legacy.source_identity()));
        let bytes=legacy.export(&ctx).unwrap();
        let consumed=CyclicEvidence::consume(&bytes,legacy.identity(),&ctx).unwrap();
        assert_eq!(consumed.report,legacy.report);
        assert_eq!(consumed.export(&ctx).unwrap(),bytes);
        assert!(consumed.joint_law_with(&[&legacy],Some("aligned")).unwrap_err().to_string().contains("legacy_alignment_unavailable"));
    }
}
