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
    EvidenceSupport, JointEventKind, JointEventQuantity, JointEventVariable, QuantityRole,
    ScientificQuantity, UncertaintyAxes, WorldTime,
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
    quantity_wire::ScientificQuantityWire,
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
    /// The predicted hidden state of a held-out intervention trajectory of a trajectory OU
    /// fit: its mean, or its process covariance. Sampling uncertainty of the fitted
    /// parameters is retained as a separate diagnostic.
    OuTrajectoryState {
        /// Prediction design index.
        design: usize,
        /// Prediction-time index.
        time: usize,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Second coordinate for a process covariance; None selects the mean.
        other: Option<usize>,
    },
    /// The stationary law of a sustained-intervention world of a trajectory OU fit.
    OuTrajectoryStationary {
        /// Stationary design index.
        world: usize,
        /// Dense endogenous coordinate.
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

mod composition;
mod projection;

#[cfg(test)]
use projection::event_point;
mod system_bridge;

pub use system_bridge::{ConsumedCyclicSystem, CyclicSystemBridgeProvenance, CyclicSystemFamilyBridge};
use system_bridge::system_bridge_provenance;

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
fn supported_state(outcome: &EquilibriumOutcome, variable: usize) -> Option<f64> {
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

/// The non-convergence of a fitted source, as the structured fact its columns carry.
///
/// A fit that stopped on its iteration or line-search bound still returns its best iterate.
/// These reports carry no claim, so host status does not apply; the column diagnostics are
/// where a consumer reads that the fitted values were not stationary points of the score.
fn fit_convergence(answer: &A) -> Option<serde_json::Value> {
    let (family, iterations, score_norm, line_search_exhausted) = match answer {
        A::OdeFit(fit) if !fit.converged => ("ode_fit", fit.iterations, Some(fit.score_norm), None),
        A::OuFit(fit) if !fit.converged => ("ou_fit", fit.sweeps, None, None),
        A::OuTrajectory(report) if !report.fit.fitting.converged => (
            "ou_trajectory",
            report.fit.fitting.iterations,
            Some(report.fit.fitting.score_norm),
            Some(report.fit.fitting.line_search_exhausted),
        ),
        A::SdeAnalysis(report) if report.fit.as_ref().is_some_and(|fit| !fit.fitting.converged) => {
            let fitting = report.fit.as_ref().map(|fit| &fit.fitting);
            (
                "sde_analysis",
                fitting.map_or(0, |f| f.iterations),
                fitting.map(|f| f.score_norm),
                fitting.map(|f| f.line_search_exhausted),
            )
        }
        A::OuMechanisms { fit: Some(fit), .. } if !fit.converged => {
            ("ou_mechanisms", fit.iterations, Some(fit.score_norm), Some(fit.line_search_exhausted))
        }
        _ => return None,
    };
    Some(serde_json::json!({
        "family": family,
        "converged": false,
        "iterations": iterations,
        "score_norm": score_norm,
        "line_search_exhausted": line_search_exhausted,
        "consequence": "the fitted values did not reach the declared score tolerance; they are the last iterate, not a stationary point",
    }))
}

/// Add `fit_convergence` to a column's diagnostics when its source fit did not converge.
fn attach_fit_convergence(diagnostics: serde_json::Value, answer: &A) -> serde_json::Value {
    let Some(convergence) = fit_convergence(answer) else {
        return diagnostics;
    };
    match diagnostics {
        serde_json::Value::Object(mut map) => {
            map.insert("fit_convergence".into(), convergence);
            serde_json::Value::Object(map)
        }
        serde_json::Value::Null => serde_json::json!({ "fit_convergence": convergence }),
        other => serde_json::json!({ "detail": other, "fit_convergence": convergence }),
    }
}

#[allow(clippy::unnecessary_wraps, reason = "projection diagnostics use an optional detail value")]
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
    /// The ordered outcomes of a joint event; `variable` is then its first outcome.
    joint: Option<Vec<usize>>,
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
        joint: None,
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

fn source_order(request: &R) -> usize {
    use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
    let request = request.recovery_recipe();
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
        R::OuMechanisms { model, .. } | R::OuTrajectoryFit { model, .. } => model.equations.len(),
        R::SdeAnalysis { model, .. } => model.variables.len(),
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

    #[allow(
        clippy::too_many_lines,
        reason = "artifact construction checks its complete versioned contract"
    )]
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
            || schema.variables.len() > antecedent_model::nonlinear_equilibrium::MAX_ORDER
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
        } else {
            None
        };
        if version == 3 && recovery_origins.is_none() {
            return Err(invalid("version-three evidence needs retained recovery origins"));
        }
        let identity = if version == 3 {
            digest(
                "antecedent.cyclic_evidence.v2",
                &(&source.identity, &schema, &selectors, &recovery_origins),
            )?
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
            let diagnostics = attach_fit_convergence(diagnostics, &source.answer);
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
            let joint_quantity = p
                .joint
                .as_ref()
                .map(|outcomes| {
                    let variables = outcomes
                        .iter()
                        .map(|index| {
                            schema
                                .variables
                                .get(*index)
                                .map(|(name, units)| JointEventVariable {
                                    name: name.clone(),
                                    units: units.clone(),
                                })
                                .ok_or_else(|| invalid("joint event variable outside schema"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let event = JointEventQuantity {
                        schema_id: schema.id.clone(),
                        variables,
                        kind: if p.functional == Functional::IdentifiedDensity {
                            JointEventKind::Density
                        } else {
                            JointEventKind::Probability
                        },
                        population_id: schema.population.clone(),
                        regime_id: regime.clone(),
                        horizon: 0,
                        conditioning: vec![],
                        transform_id: "identity".into(),
                    };
                    event
                        .validate_in_schema(&schema.variables)
                        .map_err(|e| invalid(format!("joint event: {e:?}")))?;
                    event.to_coordinate().map_err(|e| invalid(format!("joint event: {e:?}")))
                })
                .transpose()?;
            let quantity = if let Some(joint_quantity) = joint_quantity {
                joint_quantity
            } else {
                ScientificQuantity {
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
                    | CyclicEvidenceSelector::OuRegime { other: Some(other), .. } =
                        selector
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
                }
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
                            R::SdeAnalysis { trajectories, .. } if !trajectories.is_empty()
                        ) || matches!(
                            request,
                            R::AdamFit { .. }
                                | R::GaussianFunctional {law:antecedent_io::cyclic_gaussian_functional::GaussianCyclicLaw::Fit{..},..}
                                | R::OuMechanisms { .. }
                                | R::GradientFit { .. }
                                | R::OdeFit { .. }
                                | R::OuFit { .. }
                                | R::OuTrajectoryFit { .. }
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
                    EvidenceCalibration::NotMeasured,
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
            version,
            original_encoding: None,
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
    #[must_use]
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
        if let Some((original, _owner)) = &self.original_encoding {
            let _output = ctx
                .acquire_admission(0, original.len() as u64)
                .map_err(CyclicArtifactError::Admission)?;
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
                digest(
                    "antecedent.cyclic_evidence.v2",
                    &(
                        &record.source_identity,
                        &record.schema,
                        &record.selectors,
                        &record.report.recovery_origins,
                    ),
                )?
            } else {
                if record.report.recovery_origins.is_some() {
                    return Err(invalid(
                        "historical evidence has no version-three origin standing",
                    ));
                }
                digest(
                    "antecedent.cyclic_evidence.v1",
                    &(&record.source_identity, &record.schema, &record.selectors),
                )?
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
        let owner = Arc::new(
            ctx.acquire_admission(0, bytes.len() as u64).map_err(CyclicArtifactError::Admission)?,
        );
        evidence.original_encoding = Some((Arc::from(bytes), owner));
        Ok(evidence)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Study;
    #[test]
    fn a_non_converged_fit_marks_its_evidence_columns_and_a_converged_fit_leaves_them_alone() {
        use antecedent_estimate::cyclic_dynamics::{OdeFit, OuFit};
        use antecedent_model::continuous_cyclic::{OdeSystem, OrnsteinUhlenbeck};
        let ode = |converged| {
            A::OdeFit(OdeFit {
                model: OdeSystem { equations: vec![], decay: vec![] },
                log_likelihood: -1.0,
                iterations: 1,
                score_norm: 0.5,
                converged,
            })
        };
        let ou = |converged| {
            A::OuFit(OuFit {
                model: OrnsteinUhlenbeck { drift: vec![], center: vec![], diffusion: vec![] },
                log_likelihood: -1.0,
                sweeps: 1,
                converged,
            })
        };
        let marked = attach_fit_convergence(serde_json::Value::Null, &ode(false));
        assert_eq!(marked["fit_convergence"]["family"], "ode_fit");
        assert_eq!(marked["fit_convergence"]["converged"], false);
        assert_eq!(marked["fit_convergence"]["iterations"], 1);
        assert_eq!(marked["fit_convergence"]["score_norm"], 0.5);
        let merged = attach_fit_convergence(serde_json::json!({ "kept": 1 }), &ou(false));
        assert_eq!(merged["kept"], 1);
        assert_eq!(merged["fit_convergence"]["family"], "ou_fit");
        let wrapped = attach_fit_convergence(serde_json::json!([1]), &ou(false));
        assert_eq!(wrapped["detail"], serde_json::json!([1]));
        assert_eq!(
            attach_fit_convergence(serde_json::Value::Null, &ode(true)),
            serde_json::Value::Null
        );
        assert_eq!(
            attach_fit_convergence(serde_json::json!({ "kept": 1 }), &ou(true)),
            serde_json::json!({ "kept": 1 })
        );
    }
    #[test]
    fn an_event_over_several_outcomes_projects_a_joint_coordinate_and_one_outcome_stays_scalar() {
        let joint = event_point(
            Some(0.5),
            &[2, 1],
            World::Assignment(0),
            Functional::IdentifiedProbability,
            EvidencePort::Mean,
        );
        assert_eq!(joint.joint, Some(vec![2, 1]));
        assert_eq!(joint.variable, 2);
        let scalar = event_point(
            Some(0.5),
            &[2],
            World::Assignment(0),
            Functional::IdentifiedProbability,
            EvidencePort::Mean,
        );
        assert_eq!(scalar.joint, None);
        assert_eq!(scalar.variable, 2);
    }
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
        let v1: &[u8] =
            include_bytes!("../../../antecedent-io/tests/fixtures/cyclic_family_v1.bin");
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
            EvidenceCalibration::NotMeasured
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
    use antecedent_io::cyclic_family_artifact::{
        CoefficientPriorWire, CyclicInterventionWire, CyclicWorldWire, DisturbancePriorWire,
        RecoveryInferenceWire, RecoveryPriorWire,
    };

    fn request(seed: u64) -> R {
        let mut request: R = serde_json::from_str(include_str!(
            "../../../../parity/cyclic/recovered-asymptotic-normal/request.json"
        ))
        .unwrap();
        let R::ExperimentalRecovery { inference, .. } = &mut request else {
            panic!("experimental fixture")
        };
        *inference = RecoveryInferenceWire::Posterior {
            prior: RecoveryPriorWire {
                coefficients: CoefficientPriorWire::Flat,
                disturbance: DisturbancePriorWire::Known { variances: vec![1.0, 1.0] },
            },
            draws: 8,
            burn_in: 0,
            seed,
        };
        request
    }
    fn schema(id: &str) -> CyclicEvidenceSchema {
        CyclicEvidenceSchema {
            id: id.into(),
            population: "original population".into(),
            variables: vec![("A".into(), "meter".into()), ("B".into(), "meter".into())],
        }
    }
    fn selector() -> Vec<CyclicEvidenceSelector> {
        vec![CyclicEvidenceSelector::Response { world: 0, noise_row: 0, variable: 1 }]
    }
    fn branch(
        source: &ExecutedCyclicFamily,
        level: f64,
        ctx: &ExecutionContext,
    ) -> ExecutedCyclicFamily {
        source
            .prepare_recovered_worlds(
                vec![CyclicWorldWire {
                    world: 47,
                    interventions: vec![CyclicInterventionWire::Set { variable: 0, value: level }],
                    initial_states: vec![],
                }],
                1,
                ctx,
            )
            .unwrap()
            .execute_retained(ctx)
            .unwrap()
    }
    #[test]
    #[allow(clippy::float_cmp, reason = "the test checks exact shared draws")]
    fn world_branches_share_actual_evidence_and_draw_alignment_but_keep_targets_distinct() {
        let ctx = ExecutionContext::for_tests(19);
        let source = Study::prepare_cyclic_family(&request(19), &ctx)
            .unwrap()
            .execute_retained(&ctx)
            .unwrap();
        let left = branch(&source, 3.0, &ctx).evidence(schema("A3"), selector(), &ctx).unwrap();
        let right = branch(&source, 4.0, &ctx).evidence(schema("A4"), selector(), &ctx).unwrap();
        assert_ne!(left.source_identity(), right.source_identity());
        assert_ne!(left.identity(), right.identity());
        let a = left.report.recovery_origins.as_ref().unwrap();
        let b = right.report.recovery_origins.as_ref().unwrap();
        assert_eq!(a, b);
        assert_ne!(a.evidence, left.source_identity());
        assert_eq!(left.envelopes()[0].binding().origins.training.as_deref(), Some(a.fit.as_str()));
        let error = left.joint_law_with(&[&right], Some("independent")).unwrap_err().to_string();
        assert!(error.contains("reason=evidence_double_counted"));
        let joint = left.joint_law_with(&[&right], Some("aligned")).unwrap();
        assert_eq!(joint.shape(), [8, 2]);
        for (row, values) in joint.draws().chunks_exact(2).enumerate() {
            assert_eq!(values[0], left.report.columns[0].samples[row].unwrap());
            assert_eq!(values[1], right.report.columns[0].samples[row].unwrap());
        }
        assert!(joint.metadata().weights.is_some());
        let encoded = right.export(&ctx).unwrap();
        let loaded = CyclicEvidence::consume(&encoded, right.identity(), &ctx).unwrap();
        assert_eq!(loaded.report.recovery_origins, right.report.recovery_origins);
        assert_eq!(loaded.export(&ctx).unwrap(), encoded);
        let mut tampered = right.clone();
        tampered.report.recovery_origins.as_mut().unwrap().fit = "0".repeat(64);
        tampered.report.identity = digest(
            "antecedent.cyclic_evidence.v2",
            &(
                &tampered.report.source_identity,
                &tampered.schema,
                &tampered.selectors,
                &tampered.report.recovery_origins,
            ),
        )
        .unwrap();
        let resealed = tampered.export(&ctx).unwrap();
        assert!(CyclicEvidence::consume(&resealed, tampered.identity(), &ctx).is_err());
        assert_eq!(left.joint_law_with(&[&loaded], Some("aligned")).unwrap(), joint);
        let different = Study::prepare_cyclic_family(&request(20), &ctx)
            .unwrap()
            .execute_retained(&ctx)
            .unwrap();
        let misaligned =
            branch(&different, 4.0, &ctx).evidence(schema("other-seed"), selector(), &ctx).unwrap();
        assert_eq!(misaligned.report.recovery_origins.as_ref().unwrap().evidence, a.evidence);
        assert!(
            left.joint_law_with(&[&misaligned], Some("aligned"))
                .unwrap_err()
                .to_string()
                .contains("draw_alignment_mismatch")
        );
        assert!(
            left.joint_law_with(&[&misaligned], Some("independent"))
                .unwrap_err()
                .to_string()
                .contains("evidence_double_counted")
        );
    }
    #[test]
    #[allow(
        clippy::items_after_statements,
        reason = "the local legacy wire belongs to this fixture"
    )]
    fn historical_v2_envelope_bytes_identity_and_origin_standing_are_preserved() {
        let ctx = ExecutionContext::for_tests(19);
        let executed = Study::prepare_cyclic_family(&request(19), &ctx)
            .unwrap()
            .execute_retained(&ctx)
            .unwrap();
        let legacy = CyclicEvidence::build_version(
            executed.request(),
            executed.shared_report(),
            CyclicEvidenceSource::Live(Arc::new(executed.clone())),
            schema("historical"),
            selector(),
            2,
            &ctx,
        )
        .unwrap();
        #[derive(Serialize)]
        struct HistoricalReport<'a> {
            identity: &'a str,
            source_identity: &'a str,
            scope: &'a str,
            columns: &'a [CyclicEvidenceColumn],
            source: &'a CyclicFamilyReport,
        }
        let report = &legacy.report;
        let historical = HistoricalReport {
            identity: &report.identity,
            source_identity: &report.source_identity,
            scope: &report.scope,
            columns: &report.columns,
            source: &report.source,
        };
        assert_eq!(
            antecedent_io::to_cbor(&historical).unwrap(),
            antecedent_io::to_cbor(report).unwrap()
        );
        assert!(legacy.report.recovery_origins.is_none());
        assert_eq!(
            legacy.identity(),
            digest(
                "antecedent.cyclic_evidence.v1",
                &(&report.source_identity, &legacy.schema, &legacy.selectors)
            )
            .unwrap()
        );
        assert_eq!(
            legacy.envelopes()[0].binding().origins.training.as_deref(),
            Some(legacy.source_identity())
        );
        let bytes = legacy.export(&ctx).unwrap();
        let consumed = CyclicEvidence::consume(&bytes, legacy.identity(), &ctx).unwrap();
        assert_eq!(consumed.report, legacy.report);
        assert_eq!(consumed.export(&ctx).unwrap(), bytes);
        assert!(
            consumed
                .joint_law_with(&[&legacy], Some("aligned"))
                .unwrap_err()
                .to_string()
                .contains("legacy_alignment_unavailable")
        );
    }
}
