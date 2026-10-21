//! Independent replay of the existing compiled Gaussian joint-response engine.
//!
//! Artifacts preserve exact design, prior, numerical policy, world projections, target and
//! source proof. Replay is historical numerical evidence only. Current causal use requires
//! caller-supplied live compiled authority with the same complete request identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::scientific_target_wire::ScientificTargetWire;
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{
    CanonicalWriter, ExecutionContext, KernelPolicy, ScientificTarget, VariableId,
};
use antecedent_estimate::response::{
    JointLinearError, JointLinearEvaluation, JointLinearInference, PreparedJointLinear,
};
use antecedent_estimate::{
    BayesianBackendKind, BayesianGComputationAte, PreparedBayesianProblem, SerialDependence,
};
use antecedent_identify::{CompiledIdentificationProduct, IdentificationResult};
use antecedent_prob::{
    CoefficientCorrelation, ContrastCoding, GaussianCoefficientPrior, InvGammaPrior, PriorSet,
    PriorSpec,
};
use antecedent_stats::{CompiledDesign, DesignColumnMap, DesignColumnRole, StandardizationRecord};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Current joint-response artifact feature version.
pub const JOINT_RESPONSE_ARTIFACT_VERSION: u16 = 1;
/// Maximum decoded or encoded artifact bytes.
pub const MAX_JOINT_RESPONSE_ARTIFACT_BYTES: usize = 64_000_000;

/// Borrowed compiled source whose original engine and scientific declarations are encoded.
#[derive(Clone, Copy)]
pub struct JointResponseArtifactSource<'a> {
    /// Full retained scientific target.
    pub target: &'a ScientificTarget,
    /// Actual checked numerical design and inference options.
    pub numerical: &'a PreparedJointLinear,
    /// Mean design row of each world in declared output order.
    pub world_weights: &'a [Vec<f64>],
    /// Coefficients of the assembled contrast, when requested.
    pub contrast: Option<&'a [f64]>,
    /// Exact original source data snapshot identity.
    pub source_digest: [u8; 32],
    /// Actual source identification and derivation.
    pub source_identification: &'a IdentificationResult,
    /// Target-bound world and assembled support retained by the compiler.
    pub compiled_identification: &'a CompiledIdentificationProduct,
}

/// Version, identity, replay, resource or original numerical failure.
#[derive(Debug, thiserror::Error)]
pub enum JointResponseArtifactError {
    /// Container or scientific-target conversion failed.
    #[error(transparent)]
    Io(#[from] crate::IoError),
    /// Original shared numerical engine refused the stored request.
    #[error(transparent)]
    Numerical(#[from] JointLinearError),
    /// Unsupported or malformed stored semantics.
    #[error("joint response artifact: {0}")]
    Invalid(String),
    /// Stored request or result differs from independent replay or live authority.
    #[error("joint response artifact does not match its request, result or current authority")]
    Mismatch,
    /// Caller budget refused before encoding/decoding allocations.
    #[error("joint response artifact resource refusal: {0}")]
    Resource(String),
    /// No complete artifact or result is issued after cancellation.
    #[error("joint response artifact cancelled")]
    Cancelled,
}

/// Independently replayed historical numerical result; never current scientific permission.
#[derive(Debug)]
pub struct HistoricalJointResponse {
    target: ScientificTarget,
    identity: [u8; 32],
    source_digest: [u8; 32],
    evaluation: JointLinearEvaluation,
    retention: Arc<AdmissionLease>,
}
impl HistoricalJointResponse {
    /// Historical question reconstructed through the validated target constructors.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Complete compiled numerical/scientific request identity.
    #[must_use]
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
    /// Original source snapshot identity, not a statement that it remains current.
    #[must_use]
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    /// Numerical replay using the same shared response engine.
    #[must_use]
    pub fn evaluation(&self) -> &JointLinearEvaluation {
        &self.evaluation
    }
    /// Historical observations alone cannot authorize current decisions or causal claims.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
    /// Transfer numerical replay into a caller that separately validated current authority.
    #[must_use]
    pub fn into_evaluation(self) -> JointLinearEvaluation {
        self.evaluation
    }
    /// Admitted decode scratch retained with this historical owner, not process RSS.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u16,
    identity: [u8; 32],
    request: RequestWire,
    result: ResultWire,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    target: ScientificTargetWire,
    source_digest: [u8; 32],
    source_proof: crate::IdentificationResultWire,
    compiled_proof: crate::compiled_identification_wire::CompiledIdentificationWire,
    problem: ProblemWire,
    inference: InferenceWire,
    kernel: [bool; 3],
    weights: Vec<Vec<u64>>,
    contrast: Option<Vec<u64>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProblemWire {
    rows: usize,
    columns: usize,
    matrix: Vec<u64>,
    outcome: Vec<u64>,
    selection: Vec<usize>,
    roles: Vec<RoleWire>,
    method: String,
    adjustment: Vec<u32>,
    active: u64,
    control: u64,
    coefficient_names: Option<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum RoleWire {
    Intercept,
    Treatment,
    Covariate(u32),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
enum InferenceWire {
    Frequentist,
    Bayesian {
        backend: String,
        likelihood: String,
        draws: usize,
        seed: u64,
        prior_scale: u64,
        prior: Option<PriorWire>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriorWire {
    specs: Vec<PriorSpecWire>,
    contrast: Option<String>,
    categorical: Vec<u32>,
    restrictions: Vec<(String, String)>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PriorSpecWire {
    Gaussian { mean: Vec<u64>, variance: Vec<u64> },
    InvGamma { shape: u64, scale: u64 },
    KnownVariance { value: u64 },
    Correlation { dim: usize, values: Vec<u64> },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultWire {
    means: Vec<u64>,
    covariance: Vec<u64>,
    contrast: Option<[u64; 2]>,
    draws: Option<Vec<u64>>,
    contrast_draws: Option<Vec<u64>>,
    count: usize,
    failed: Vec<bool>,
}

/// Compute the complete request identity without fitting or creating a public builder.
///
/// # Errors
/// Unsupported design semantics, invalid scientific binding, or encoding failure.
pub fn joint_response_identity(
    source: JointResponseArtifactSource<'_>,
) -> Result<[u8; 32], JointResponseArtifactError> {
    request_identity(&request_wire(source)?)
}

/// Encode original inputs and already executed joint output in the existing checked container.
///
/// # Errors
/// Unsupported configuration, invalid binding, encoding limits, admission or cancellation.
pub fn encode_joint_response(
    source: JointResponseArtifactSource<'_>,
    evaluation: &JointLinearEvaluation,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, JointResponseArtifactError> {
    cancelled(ctx)?;
    let problem = source.numerical.problem();
    let numbers = problem
        .design
        .matrix
        .len()
        .checked_add(problem.design.outcome.len())
        .and_then(|n| n.checked_add(evaluation.posterior_draws.as_ref().map_or(0, |v| v.len())))
        .and_then(|n| n.checked_add(source.world_weights.iter().map(Vec::len).sum::<usize>()))
        .ok_or_else(|| invalid("encoding quote overflow"))?;
    let bytes = numbers
        .checked_mul(96)
        .and_then(|n| n.checked_add(1_000_000))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| invalid("encoding quote overflow"))?;
    let _encoding = admit(ctx, bytes)?;
    let request = request_wire(source)?;
    let record = Record {
        version: JOINT_RESPONSE_ARTIFACT_VERSION,
        identity: request_identity(&request)?,
        request,
        result: result_wire(evaluation),
    };
    let mut counter = ByteCounter(0);
    serde_json::to_writer(&mut counter, &record).map_err(invalid)?;
    if counter.0 > MAX_JOINT_RESPONSE_ARTIFACT_BYTES {
        return Err(invalid("payload exceeds byte ceiling"));
    }
    let _serialized = admit(
        ctx,
        (counter.0 as u64).checked_mul(3).ok_or_else(|| invalid("serialization quote overflow"))?,
    )?;
    let bytes = serde_json::to_vec(&record).map_err(invalid)?;
    if bytes.len() > MAX_JOINT_RESPONSE_ARTIFACT_BYTES {
        return Err(invalid("payload exceeds byte ceiling"));
    }
    cancelled(ctx)?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: "joint_response",
            note: "historical_joint_gaussian_execution",
            max_bytes: MAX_JOINT_RESPONSE_ARTIFACT_BYTES,
            artifact_id: "joint_response",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: "joint_response_v1",
            content_type: crate::CONTENT_BYTES,
            bytes: &bytes,
        }],
    )
    .map_err(|error| JointResponseArtifactError::Io(error.into_io("joint response encoding")))
}

/// Decode and independently replay through the same retained Gaussian response engine.
///
/// The result is historical. Neither a matching digest nor reproducible numbers grants
/// current scientific permission. Use the facade's live-program comparison for current use.
///
/// # Errors
/// Unknown versions/fields, limits, corrupt identity, mismatched results, numerical refusal,
/// cancellation or admission failure.
pub fn consume_joint_response(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<HistoricalJointResponse, JointResponseArtifactError> {
    cancelled(ctx)?;
    if bytes.len() > MAX_JOINT_RESPONSE_ARTIFACT_BYTES {
        return Err(invalid("input exceeds byte ceiling"));
    }
    // Read only the bounded manifest before admission. A compressed input can be much
    // smaller than its declared logical payload; input length cannot quote decompression.
    let reader = crate::ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let logical = reader.manifest().sections.iter().try_fold(0_u64, |total, section| {
        total.checked_add(section.uncompressed_size).ok_or_else(|| invalid("decode quote overflow"))
    })?;
    if logical > MAX_JOINT_RESPONSE_ARTIFACT_BYTES as u64 {
        return Err(invalid("decoded payload exceeds byte ceiling"));
    }
    let quote = logical
        .checked_mul(8)
        .and_then(|n| n.checked_add(bytes.len() as u64))
        .and_then(|n| n.checked_add(65536))
        .ok_or_else(|| invalid("decode quote overflow"))?;
    let retention = Arc::new(admit(ctx, quote)?);
    let sections = crate::decode_container(
        "joint_response",
        MAX_JOINT_RESPONSE_ARTIFACT_BYTES,
        &["joint_response_v1"],
        bytes,
    )
    .map_err(|error| JointResponseArtifactError::Io(error.into_io("joint response decoding")))?;
    let record: Record = serde_json::from_slice(
        sections.first().ok_or_else(|| invalid("missing joint response feature"))?,
    )
    .map_err(invalid)?;
    if record.version != JOINT_RESPONSE_ARTIFACT_VERSION {
        return Err(invalid("unsupported feature version"));
    }
    if record.identity != request_identity(&record.request)? {
        return Err(JointResponseArtifactError::Mismatch);
    }
    let target = record.request.target.to_domain()?;
    let _historical_proof = record.request.compiled_proof.to_domain(&target)?;
    let problem = problem_from_wire(&record.request.problem)?;
    let kernel = kernel_from_wire(record.request.kernel);
    let program =
        PreparedJointLinear::new(problem, inference_from_wire(&record.request.inference)?, kernel)?;
    let weights = record.request.weights.iter().map(|row| floats(row)).collect::<Vec<_>>();
    let contrast = record.request.contrast.as_deref().map(floats);
    let mut context = ctx.clone();
    context.kernel_policy = kernel;
    let evaluation = program.execute(&weights, contrast.as_deref(), &context)?;
    if result_wire(&evaluation) != record.result {
        return Err(JointResponseArtifactError::Mismatch);
    }
    cancelled(ctx)?;
    Ok(HistoricalJointResponse {
        target,
        identity: record.identity,
        source_digest: record.request.source_digest,
        evaluation,
        retention,
    })
}

fn request_wire(
    source: JointResponseArtifactSource<'_>,
) -> Result<RequestWire, JointResponseArtifactError> {
    source.compiled_identification.validate_target(source.target).map_err(invalid)?;
    let kernel = source.numerical.kernel_policy();
    Ok(RequestWire {
        target: ScientificTargetWire::from_domain(source.target)?,
        source_digest: source.source_digest,
        source_proof: crate::identification_to_wire(source.source_identification)?,
        compiled_proof:
            crate::compiled_identification_wire::CompiledIdentificationWire::from_domain(
                source.compiled_identification,
                source.target,
            )?,
        problem: problem_to_wire(source.numerical.problem())?,
        inference: inference_to_wire(source.numerical.inference()),
        kernel: [kernel.allow_portable_optimized, kernel.allow_arch_simd, kernel.force_scalar],
        weights: source.world_weights.iter().map(|row| bits(row)).collect(),
        contrast: source.contrast.map(bits),
    })
}
fn request_identity(request: &RequestWire) -> Result<[u8; 32], JointResponseArtifactError> {
    let bytes = serde_json::to_vec(request).map_err(invalid)?;
    Ok(CanonicalWriter::digest_bytes("antecedent.joint_response.compiled_request.v1", &bytes))
}
fn problem_to_wire(
    problem: &PreparedBayesianProblem,
) -> Result<ProblemWire, JointResponseArtifactError> {
    if !problem.design.contrasts.is_empty()
        || !problem.design.smooths.is_empty()
        || !problem.design.standardization.entries.is_empty()
        || problem.design.columns.iter().any(|column| {
            column.contrast_idx.is_some()
                || column.smooth_idx.is_some()
                || column.standardization_idx.is_some()
        })
        || problem.unit_ids.is_some()
        || !matches!(problem.serial_dependence, SerialDependence::Iid)
        || problem.overlap != antecedent_estimate::OverlapPolicy::ExplicitOverride
    {
        return Err(invalid(
            "unsupported non-raw or dependent linear design; no metadata is discarded",
        ));
    }
    Ok(ProblemWire {
        rows: problem.design.nrows,
        columns: problem.design.ncols,
        matrix: bits(&problem.design.matrix),
        outcome: bits(&problem.design.outcome),
        selection: problem.design.row_selection.to_vec(),
        roles: problem
            .design
            .columns
            .iter()
            .map(|column| match column.role {
                DesignColumnRole::Intercept => RoleWire::Intercept,
                DesignColumnRole::Treatment => RoleWire::Treatment,
                DesignColumnRole::Covariate(id) => RoleWire::Covariate(id.raw()),
            })
            .collect(),
        method: problem.method.to_string(),
        adjustment: problem.adjustment_set.iter().map(|id| id.raw()).collect(),
        active: problem.active.to_bits(),
        control: problem.control.to_bits(),
        coefficient_names: problem
            .coef_names
            .as_ref()
            .map(|names| names.iter().map(ToString::to_string).collect()),
    })
}
fn problem_from_wire(
    wire: &ProblemWire,
) -> Result<PreparedBayesianProblem, JointResponseArtifactError> {
    if wire.rows == 0
        || wire.columns == 0
        || wire.columns > 4096
        || wire.rows.checked_mul(wire.columns) != Some(wire.matrix.len())
        || wire.outcome.len() != wire.rows
        || wire.selection.len() != wire.rows
        || wire.roles.len() != wire.columns
        || wire.coefficient_names.as_ref().is_some_and(|names| names.len() != wire.columns)
        || wire.selection.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid("invalid retained design shape or row selection"));
    }
    let matrix = floats(&wire.matrix);
    let outcome = floats(&wire.outcome);
    if matrix.iter().chain(&outcome).any(|v| !v.is_finite()) {
        return Err(invalid("retained design is not finite"));
    }
    Ok(PreparedBayesianProblem {
        design: CompiledDesign {
            nrows: wire.rows,
            ncols: wire.columns,
            matrix: matrix.into(),
            outcome: outcome.into(),
            row_selection: wire.selection.clone().into(),
            columns: DesignColumnMap::from_roles(wire.roles.iter().map(|role| match role {
                RoleWire::Intercept => DesignColumnRole::Intercept,
                RoleWire::Treatment => DesignColumnRole::Treatment,
                RoleWire::Covariate(id) => DesignColumnRole::Covariate(VariableId::from_raw(*id)),
            })),
            contrasts: vec![],
            smooths: vec![],
            standardization: StandardizationRecord::default(),
        },
        method: Arc::from(wire.method.as_str()),
        adjustment_set: wire
            .adjustment
            .iter()
            .copied()
            .map(VariableId::from_raw)
            .collect::<Vec<_>>()
            .into(),
        active: f64::from_bits(wire.active),
        control: f64::from_bits(wire.control),
        overlap: antecedent_estimate::OverlapPolicy::ExplicitOverride,
        coef_names: wire.coefficient_names.as_ref().map(|names| {
            names.iter().map(|name| Arc::<str>::from(name.as_str())).collect::<Vec<_>>().into()
        }),
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    })
}
fn inference_to_wire(inference: &JointLinearInference) -> InferenceWire {
    match inference {
        JointLinearInference::Frequentist => InferenceWire::Frequentist,
        JointLinearInference::Bayesian(config) => InferenceWire::Bayesian {
            backend: match config.backend {
                BayesianBackendKind::ConjugateGaussian => "conjugate",
                BayesianBackendKind::Laplace => "laplace",
                BayesianBackendKind::Hmc => "hmc",
            }
            .into(),
            likelihood: match config.likelihood {
                antecedent_prob::BayesLikelihood::GaussianIdentity => "gaussian",
                antecedent_prob::BayesLikelihood::BernoulliLogit => "bernoulli",
                antecedent_prob::BayesLikelihood::BernoulliProbit => "probit",
                antecedent_prob::BayesLikelihood::PoissonLog => "poisson",
            }
            .into(),
            draws: config.n_draws,
            seed: config.seed,
            prior_scale: config.prior_scale.to_bits(),
            prior: config.prior.as_ref().map(prior_to_wire),
        },
    }
}
fn inference_from_wire(
    wire: &InferenceWire,
) -> Result<JointLinearInference, JointResponseArtifactError> {
    Ok(match wire {
        InferenceWire::Frequentist => JointLinearInference::Frequentist,
        InferenceWire::Bayesian { backend, likelihood, draws, seed, prior_scale, prior } => {
            if !(2..=100_000).contains(draws) {
                return Err(invalid("posterior draw bound"));
            }
            JointLinearInference::Bayesian(BayesianGComputationAte {
                backend: match backend.as_str() {
                    "conjugate" => BayesianBackendKind::ConjugateGaussian,
                    "laplace" => BayesianBackendKind::Laplace,
                    "hmc" => BayesianBackendKind::Hmc,
                    _ => return Err(invalid("unknown posterior backend")),
                },
                likelihood: match likelihood.as_str() {
                    "gaussian" => antecedent_prob::BayesLikelihood::GaussianIdentity,
                    "bernoulli" => antecedent_prob::BayesLikelihood::BernoulliLogit,
                    "probit" => antecedent_prob::BayesLikelihood::BernoulliProbit,
                    "poisson" => antecedent_prob::BayesLikelihood::PoissonLog,
                    _ => return Err(invalid("unknown likelihood")),
                },
                n_draws: *draws,
                seed: *seed,
                prior_scale: f64::from_bits(*prior_scale),
                prior: prior.as_ref().map(prior_from_wire).transpose()?,
                overlap: antecedent_estimate::OverlapPolicy::ExplicitOverride,
            })
        }
    })
}
fn prior_to_wire(prior: &PriorSet) -> PriorWire {
    PriorWire {
        specs: prior
            .specs
            .iter()
            .map(|spec| match spec {
                PriorSpec::GaussianCoefficients(p) => {
                    PriorSpecWire::Gaussian { mean: bits(&p.mean), variance: bits(&p.variance) }
                }
                PriorSpec::ResidualInvGamma(p) => {
                    PriorSpecWire::InvGamma { shape: p.shape.to_bits(), scale: p.scale.to_bits() }
                }
                PriorSpec::KnownResidualVariance(value) => {
                    PriorSpecWire::KnownVariance { value: value.to_bits() }
                }
                PriorSpec::CoefficientCorrelation(p) => {
                    PriorSpecWire::Correlation { dim: p.dim(), values: bits(p.matrix()) }
                }
            })
            .collect(),
        contrast: prior.contrast.map(|coding| {
            match coding {
                ContrastCoding::Treatment => "treatment",
                ContrastCoding::Sum => "sum",
            }
            .into()
        }),
        categorical: prior.categorical.iter().map(|id| id.raw()).collect(),
        restrictions: prior
            .restrictions
            .iter()
            .map(|restriction| (restriction.id.to_string(), restriction.description.to_string()))
            .collect(),
    }
}
fn prior_from_wire(wire: &PriorWire) -> Result<PriorSet, JointResponseArtifactError> {
    let specs = wire
        .specs
        .iter()
        .map(|spec| {
            Ok(match spec {
                PriorSpecWire::Gaussian { mean, variance } => {
                    PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                        mean: floats(mean).into(),
                        variance: floats(variance).into(),
                    })
                }
                PriorSpecWire::InvGamma { shape, scale } => {
                    PriorSpec::ResidualInvGamma(InvGammaPrior {
                        shape: f64::from_bits(*shape),
                        scale: f64::from_bits(*scale),
                    })
                }
                PriorSpecWire::KnownVariance { value } => {
                    PriorSpec::KnownResidualVariance(f64::from_bits(*value))
                }
                PriorSpecWire::Correlation { dim, values } => PriorSpec::CoefficientCorrelation(
                    CoefficientCorrelation::new(*dim, floats(values))
                        .map_err(|error| invalid(error.to_string()))?,
                ),
            })
        })
        .collect::<Result<Vec<_>, JointResponseArtifactError>>()?;
    Ok(PriorSet {
        specs,
        contrast: wire
            .contrast
            .as_deref()
            .map(|value| match value {
                "treatment" => Ok(ContrastCoding::Treatment),
                "sum" => Ok(ContrastCoding::Sum),
                _ => Err(invalid("unknown prior contrast coding")),
            })
            .transpose()?,
        categorical: wire.categorical.iter().copied().map(VariableId::from_raw).collect(),
        restrictions: wire
            .restrictions
            .iter()
            .map(|(id, description)| antecedent_core::PriorAssumption {
                id: Arc::from(id.as_str()),
                description: Arc::from(description.as_str()),
            })
            .collect(),
    })
}
fn result_wire(result: &JointLinearEvaluation) -> ResultWire {
    ResultWire {
        means: bits(&result.means),
        covariance: bits(&result.covariance.values),
        contrast: result.contrast.map(|c| [c.value.to_bits(), c.se.to_bits()]),
        draws: result.posterior_draws.as_deref().map(bits),
        contrast_draws: result.contrast_draws.as_deref().map(bits),
        count: result.draw_count,
        failed: result.failed_draws.to_vec(),
    }
}
fn kernel_from_wire(wire: [bool; 3]) -> KernelPolicy {
    KernelPolicy {
        allow_portable_optimized: wire[0],
        allow_arch_simd: wire[1],
        force_scalar: wire[2],
    }
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn floats(values: &[u64]) -> Vec<f64> {
    values.iter().copied().map(f64::from_bits).collect()
}
fn invalid(message: impl std::fmt::Display) -> JointResponseArtifactError {
    JointResponseArtifactError::Invalid(message.to_string())
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), JointResponseArtifactError> {
    if ctx.cancellation.is_cancelled() {
        Err(JointResponseArtifactError::Cancelled)
    } else {
        Ok(())
    }
}
fn admit(ctx: &ExecutionContext, bytes: u64) -> Result<AdmissionLease, JointResponseArtifactError> {
    let limit =
        [ctx.memory.soft_limit_bytes, ctx.memory.hard_limit_bytes].into_iter().flatten().min();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(JointResponseArtifactError::Resource("configured memory limit".into()));
    }
    ctx.acquire_admission(0, bytes)
        .map_err(|error| JointResponseArtifactError::Resource(error.to_string()))
}

struct ByteCounter(usize);
impl std::io::Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("serialized byte count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_response_prior_wire_preserves_all_prior_fields_and_ieee_values() {
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                    mean: vec![0.25, -0.0].into(),
                    variance: vec![2.0, 3.0].into(),
                }),
                PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 2.5, scale: 0.75 }),
                PriorSpec::KnownResidualVariance(0.3),
                PriorSpec::CoefficientCorrelation(
                    CoefficientCorrelation::new(2, vec![1.0, 0.2, 0.2, 1.0]).unwrap(),
                ),
            ],
            contrast: Some(ContrastCoding::Sum),
            categorical: vec![VariableId::from_raw(9)],
            restrictions: vec![antecedent_core::PriorAssumption {
                id: "restriction.id".into(),
                description: "scientific premise".into(),
            }],
        };
        let wire = prior_to_wire(&prior);
        let encoded = serde_json::to_vec(&wire).unwrap();
        let decoded: PriorWire = serde_json::from_slice(&encoded).unwrap();
        let reconstructed = prior_from_wire(&decoded).unwrap();
        assert_eq!(serde_json::to_vec(&prior_to_wire(&reconstructed)).unwrap(), encoded);
        assert!(
            matches!(&reconstructed.specs[0], PriorSpec::GaussianCoefficients(p) if p.mean[1].to_bits()==(-0.0_f64).to_bits())
        );
    }
}
