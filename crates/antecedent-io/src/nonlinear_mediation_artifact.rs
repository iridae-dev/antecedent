//! Nonlinear continuous-mediator mediation artifact (`nonlinear_mediation_v1`).
//!
//! A bounded, checksummed, sectioned container holding one natural-effects estimate:
//!
//! * `nonlinear_mediation.meta` (CBOR): version, feature marker, the estimand
//!   (`natural_effects`), the premises with their declared or checked status, the model
//!   specifications, the configuration, the NDE / NIE / total effects, the integration error of
//!   the declared `n` versus `2n` Gauss-Hermite nodes, the overlap diagnostics, the bootstrap
//!   record (seed and replicate ids, interval status closed), the identity digests, the
//!   **unmeasured** calibration coordinate and the standing caveats;
//! * `nonlinear_mediation.data` (little-endian `f64`): the compact analysis dataset
//!   (treatment, mediator, outcome, then each covariate), or nothing when the record binds its
//!   dataset by digest.
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! declarations and the embedded dataset, re-runs the core estimator (including the seeded
//! bootstrap) and refuses unless the whole stored record is bit-identical to the recomputed one.
//! It recomputes the identity digests from the same inputs and compares them with the stored ones
//! and, when the caller retained an identity of its own, with that: a changed premise, model
//! specification, dataset or result is refused even when the artifact was resealed consistently.
//!
//! What this does **not** say: bootstrap standard errors carry no coverage claim
//! (`calibration = "unmeasured"`), no public interval is produced
//! (`interval_status = "closed_calibration_unmeasured"`), and the cross-world independence and
//! sequential ignorability that identify the natural effects are declared, not tested. Unknown
//! major versions refuse.
//!
//! # Budgets
//!
//! The estimator has no row ceiling; three budgets bound what is spent on it, each refusing with
//! a typed detail instead of truncating anything:
//!
//! * **Computation:** the [`ExecutionContext`] given to [`NonlinearMediationArtifact::seal_with`]
//!   or [`NonlinearMediationArtifact::from_bytes_with`] bounds the planned working set (its hard
//!   memory limit) and stops a cancelled run (`nonlinear_mediation.memory_budget_exceeded`).
//! * **Export:** [`NonlinearMediationArtifact::to_bytes_within`] embeds the dataset only when the
//!   whole artifact fits the export budget, which is at most
//!   [`MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES`]
//!   (`nonlinear_mediation.export_budget_exceeded`); the estimate stays available in memory.
//! * **Replay:** a re-estimation needs the rows.
//!   [`NonlinearMediationArtifact::to_bytes_digest_bound`] exports the record without them,
//!   bound to the dataset by its digest (`identity.data_id`);
//!   [`NonlinearMediationArtifact::from_bytes_with`] then needs the caller to supply the dataset
//!   and refuses it unless the digest matches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::nonlinear_mediation::{
    NONLINEAR_MEDIATION_CALIBRATION, NONLINEAR_MEDIATION_INTERVAL_STATUS, NonlinearMediationConfig,
    NonlinearMediationEstimand, NonlinearMediationEstimate, NonlinearMediationInput,
    NonlinearMediationPremises, estimate_nonlinear_mediation,
};
use serde::{Deserialize, Serialize};

use crate::artifact_budget::{
    BudgetDetails, BudgetRefusal, admit_computation, admit_export, planned_bytes, unbounded_context,
};
use crate::convert::to_cbor;
use crate::width::lossless_u64;

/// The artifact major version this reader writes and accepts.
pub const NONLINEAR_MEDIATION_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const NONLINEAR_MEDIATION_ARTIFACT_FEATURE: &str = "nonlinear_mediation_v1";
/// Most analysis rows an artifact embedded under the earlier fixed ceiling.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: the estimate has no row ceiling; the export and computation budgets bound the artifact"
)]
pub const NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS: usize = 20_000;
/// Most bootstrap replicates a producer may request and a consumer will replay.
pub const NONLINEAR_MEDIATION_ARTIFACT_MAX_BOOTSTRAP: u32 = 2_000;
/// Most bytes an artifact may occupy: the largest export budget, enforced on consumption.
pub const MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Inference claim of the artifact: point estimates and diagnostics, no interval.
pub const NONLINEAR_MEDIATION_ARTIFACT_CLAIM: &str = "point_with_diagnostics";

const ARTIFACT_KIND: &str = crate::artifact_kinds::NONLINEAR_MEDIATION;
const META_SECTION: &str = "nonlinear_mediation_meta";
const DATA_SECTION: &str = "nonlinear_mediation_data";
const BUDGET: BudgetDetails = BudgetDetails {
    cancelled: (
        antecedent_core::reason_code!("cancelled_no_claim"),
        "nonlinear_mediation.cancelled",
    ),
    memory: "nonlinear_mediation.memory_budget_exceeded",
    export: "nonlinear_mediation.export_budget_exceeded",
};
/// `NonlinearMediationMeta::data_binding` of a record that holds no dataset, only its digest.
const DATA_BOUND_BY_DIGEST: &str = "digest";
const ESTIMAND_NATURAL: &str = "natural_effects";
const ESTIMAND_INTERVENTIONAL: &str = "interventional_effects";
const STATUS_DECLARED: &str = "declared";
const STATUS_CHECKED: &str = "checked";

/// Standing caveats attached to every artifact.
pub const NONLINEAR_MEDIATION_ARTIFACT_CAVEATS: [&str; 2] = [
    "bootstrap standard errors are calibration unmeasured: no coverage claim is made and no \
     public interval is produced; the point estimates and diagnostics are the claim",
    "natural direct and indirect effects need cross-world independence and sequential \
     ignorability, which are declared and cannot be tested from the data",
];

crate::sectioned_artifact_error!(
    /// Why a nonlinear-mediation artifact was refused or could not be produced.
    NonlinearMediationArtifactError,
    "nonlinear mediation",
    "nonlinear_mediation",
    "record",
    true
);

fn budget_refusal((code, message): BudgetRefusal) -> NonlinearMediationArtifactError {
    NonlinearMediationArtifactError::Refused { code, message }
}

fn invalid_request(message: &str) -> NonlinearMediationArtifactError {
    NonlinearMediationArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("nonlinear_mediation.invalid_request: {message}"),
    }
}

/// Declared graph premises on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Wire struct declares four independent graph premises; each flag is its own declaration.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent switch of the record; they do not form a state machine"
)]
pub struct MediationPremisesWire {
    /// An unmeasured treatment-outcome common cause is declared.
    pub unmeasured_treatment_outcome_confounding: bool,
    /// An unmeasured treatment-mediator common cause is declared.
    pub unmeasured_treatment_mediator_confounding: bool,
    /// An unmeasured mediator-outcome common cause is declared.
    pub unmeasured_mediator_outcome_confounding: bool,
    /// Declared mediator-outcome confounders that are descendants of the treatment.
    pub treatment_induced_mediator_outcome_confounders: Vec<String>,
    /// The cross-world independence is declared.
    pub cross_world_independence: bool,
}

/// Estimator configuration on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationConfigWire {
    /// `natural_effects` or `interventional_effects` (the latter is refused).
    pub estimand: String,
    /// Outcome polynomial degree in the standardized mediator.
    pub outcome_degree: u32,
    /// Declared Gauss-Hermite node count `n`.
    pub quadrature_nodes: u32,
    /// Largest accepted integration error, as a fraction of the outcome standard deviation.
    pub integration_tolerance: f64,
    /// Fewest rows required in each treatment arm.
    pub min_arm_count: u32,
    /// Largest accepted cross-world mediator mass outside the treated arm's range.
    pub max_support_violation: f64,
    /// Bootstrap replicates for the standard errors.
    pub bootstrap_replicates: u32,
    /// Bootstrap seed.
    pub seed: u64,
}

/// The compact analysis dataset on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationDataWire {
    /// Binary treatment coded 0 / 1.
    pub treatment: Vec<f64>,
    /// Continuous mediator.
    pub mediator: Vec<f64>,
    /// Outcome.
    pub outcome: Vec<f64>,
    /// Covariate names, in column order.
    pub covariate_names: Vec<String>,
    /// Covariate columns.
    pub covariates: Vec<Vec<f64>>,
}

/// A complete mediation request: what the producer declares and the core estimator consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationRequestWire {
    /// Declared premises.
    pub premises: MediationPremisesWire,
    /// Configuration.
    pub config: MediationConfigWire,
    /// The analysis dataset.
    pub data: MediationDataWire,
}

/// One premise with its status: `declared` (asserted by the caller) or `checked` (verified here).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationPremiseRecord {
    /// Premise name.
    pub name: String,
    /// `declared` or `checked`.
    pub status: String,
    /// Whether the premise holds (every premise holds in a sealed artifact; a violated one
    /// refuses before sealing).
    pub holds: bool,
}

/// Model specifications.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationModelSpecs {
    /// Mediator model.
    pub mediator: String,
    /// Outcome model.
    pub outcome: String,
    /// Integration rule.
    pub integration: String,
    /// Outcome polynomial degree.
    pub outcome_degree: u32,
    /// Declared quadrature nodes `n`.
    pub quadrature_nodes: u32,
    /// Covariate names.
    pub covariates: Vec<String>,
}

/// Overlap diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationOverlapWire {
    /// Treated rows.
    pub treated_count: u64,
    /// Control rows.
    pub control_count: u64,
    /// Smallest mediator value in the treated arm.
    pub treated_mediator_min: f64,
    /// Largest mediator value in the treated arm.
    pub treated_mediator_max: f64,
    /// Average cross-world mediator mass outside the treated arm's observed range.
    pub mediator_support_violation: f64,
}

/// Mediator-model diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationMediatorWire {
    /// Intercept, treatment, then covariate coefficients.
    pub coefficients: Vec<f64>,
    /// Residual variance.
    pub residual_variance: f64,
    /// Residual skewness (reported, not gated).
    pub residual_skewness: f64,
    /// Residual excess kurtosis (reported, not gated).
    pub residual_excess_kurtosis: f64,
}

/// Bootstrap record: seed, replicate ids and the closed interval status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationBootstrapWire {
    /// Seed.
    pub seed: u64,
    /// Replicates requested.
    pub replicates_requested: u32,
    /// Replicates that fit and integrated.
    pub replicates_succeeded: u32,
    /// Ids of every requested replicate (`0 .. requested`).
    pub replicate_ids: Vec<u64>,
    /// Ids of replicates that failed.
    pub failed_replicate_ids: Vec<u64>,
    /// Replicate-id scheme.
    pub id_scheme: String,
    /// Interval status: `closed_calibration_unmeasured`.
    pub interval_status: String,
}

/// The stored estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationResultWire {
    /// Natural direct effect.
    pub natural_direct: f64,
    /// Natural indirect effect.
    pub natural_indirect: f64,
    /// Total effect.
    pub total: f64,
    /// Bootstrap standard error of the natural direct effect.
    pub natural_direct_se: Option<f64>,
    /// Bootstrap standard error of the natural indirect effect.
    pub natural_indirect_se: Option<f64>,
    /// Bootstrap standard error of the total effect.
    pub total_se: Option<f64>,
    /// `|total_rowwise - (NDE + NIE)|`.
    pub identity_residual: f64,
    /// `n` versus `2n` node disagreement as a fraction of the outcome standard deviation.
    pub integration_error: f64,
    /// Declared node count `n`.
    pub coarse_nodes: u32,
    /// Node count `2n` the point estimates use.
    pub fine_nodes: u32,
    /// Outcome polynomial degree.
    pub outcome_degree: u32,
    /// Rows used.
    pub n_rows: u64,
    /// Overlap diagnostics.
    pub overlap: MediationOverlapWire,
    /// Mediator-model diagnostics.
    pub mediator: MediationMediatorWire,
    /// Bootstrap record.
    pub bootstrap: MediationBootstrapWire,
    /// Assumption identifiers the estimate relies on.
    pub assumptions: Vec<String>,
}

/// Identity digests. Each is BLAKE3 over a canonical encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationIdentity {
    /// Premises and their status records.
    pub premises_id: String,
    /// Configuration and model specifications.
    pub config_id: String,
    /// The embedded dataset.
    pub data_id: String,
    /// The stored estimate.
    pub result_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// The CBOR metadata section; also the self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// `natural_effects`.
    pub estimand: String,
    /// Declared premises.
    pub premises: MediationPremisesWire,
    /// Premises with their declared or checked status.
    pub premise_records: Vec<MediationPremiseRecord>,
    /// Model specifications.
    pub model_specs: MediationModelSpecs,
    /// Configuration.
    pub config: MediationConfigWire,
    /// Rows in the dataset.
    pub n_rows: u64,
    /// `digest` when the record holds no dataset and the caller supplies it on consumption;
    /// absent when the dataset is embedded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_binding: Option<String>,
    /// The stored estimate.
    pub result: MediationResultWire,
    /// Identity digests.
    pub identity: NonlinearMediationIdentity,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `closed_calibration_unmeasured`.
    pub interval_status: String,
    /// Always `point_with_diagnostics`.
    pub inference_claim: String,
    /// Standing caveats.
    pub caveats: Vec<String>,
}

fn count_u32(n: usize) -> Result<u32, NonlinearMediationArtifactError> {
    u32::try_from(n).map_err(|_| NonlinearMediationArtifactError::LimitsExceeded("count"))
}

fn index(n: u32) -> usize {
    n as usize
}

fn estimand_of(tag: &str) -> Result<NonlinearMediationEstimand, NonlinearMediationArtifactError> {
    match tag {
        ESTIMAND_NATURAL => Ok(NonlinearMediationEstimand::NaturalEffects),
        ESTIMAND_INTERVENTIONAL => Ok(NonlinearMediationEstimand::InterventionalEffects),
        _ => Err(invalid_request("unknown estimand tag")),
    }
}

fn premises_of(wire: &MediationPremisesWire) -> NonlinearMediationPremises {
    NonlinearMediationPremises {
        unmeasured_treatment_outcome_confounding: wire.unmeasured_treatment_outcome_confounding,
        unmeasured_treatment_mediator_confounding: wire.unmeasured_treatment_mediator_confounding,
        unmeasured_mediator_outcome_confounding: wire.unmeasured_mediator_outcome_confounding,
        treatment_induced_mediator_outcome_confounders: wire
            .treatment_induced_mediator_outcome_confounders
            .clone(),
        cross_world_independence: wire.cross_world_independence,
    }
}

fn config_of(
    wire: &MediationConfigWire,
) -> Result<NonlinearMediationConfig, NonlinearMediationArtifactError> {
    Ok(NonlinearMediationConfig {
        estimand: estimand_of(&wire.estimand)?,
        outcome_degree: index(wire.outcome_degree),
        quadrature_nodes: index(wire.quadrature_nodes),
        integration_tolerance: wire.integration_tolerance,
        min_arm_count: index(wire.min_arm_count),
        max_support_violation: wire.max_support_violation,
        bootstrap_replicates: wire.bootstrap_replicates,
        seed: wire.seed,
    })
}

fn check_limits(
    request: &NonlinearMediationRequestWire,
) -> Result<(), NonlinearMediationArtifactError> {
    // The covariate count is bounded by the bytes of the data section (a column is `n` reals),
    // by the byte budget of the artifact and by the replay's memory budget, not by a count.
    if request.config.bootstrap_replicates > NONLINEAR_MEDIATION_ARTIFACT_MAX_BOOTSTRAP {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("bootstrap replicates"));
    }
    Ok(())
}

fn check_shape(
    request: &NonlinearMediationRequestWire,
) -> Result<(), NonlinearMediationArtifactError> {
    let data = &request.data;
    let n = data.treatment.len();
    if data.mediator.len() != n
        || data.outcome.len() != n
        || data.covariates.iter().any(|c| c.len() != n)
    {
        return Err(invalid_request("columns differ in length"));
    }
    if data.covariate_names.len() != data.covariates.len() {
        return Err(invalid_request("every covariate needs exactly one name"));
    }
    Ok(())
}

/// Eight-byte words the estimate holds per row: the treatment, mediator, outcome and covariate
/// columns, the mediator and outcome designs (the latter grows with the polynomial degree and
/// its treatment interactions), the bootstrap resample and the per-row bookkeeping. A
/// conservative planning estimate.
fn words_per_row(covariates: usize, outcome_degree: u32) -> Option<usize> {
    let columns = covariates.checked_add(3)?.checked_mul(4)?;
    let polynomial = index(outcome_degree).checked_add(1)?.checked_mul(4)?;
    columns.checked_add(polynomial)?.checked_add(8)
}

fn estimate_of(
    request: &NonlinearMediationRequestWire,
    ctx: &ExecutionContext,
) -> Result<NonlinearMediationEstimate, NonlinearMediationArtifactError> {
    check_limits(request)?;
    check_shape(request)?;
    let words = words_per_row(request.data.covariates.len(), request.config.outcome_degree);
    let planned = words.and_then(|words| planned_bytes(request.data.treatment.len(), words, 0));
    admit_computation(BUDGET, planned, ctx).map_err(budget_refusal)?;
    let covariates: Vec<&[f64]> = request.data.covariates.iter().map(Vec::as_slice).collect();
    let input = NonlinearMediationInput {
        treatment: &request.data.treatment,
        mediator: &request.data.mediator,
        outcome: &request.data.outcome,
        covariates: &covariates,
    };
    Ok(estimate_nonlinear_mediation(
        &input,
        &premises_of(&request.premises),
        &config_of(&request.config)?,
    )?)
}

fn data_bytes(data: &MediationDataWire) -> Vec<u8> {
    let mut bytes = Vec::new();
    let columns =
        [&data.treatment, &data.mediator, &data.outcome].into_iter().chain(data.covariates.iter());
    for column in columns {
        for value in column {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn request_of(
    meta: &NonlinearMediationMeta,
    numbers: &[u8],
) -> Result<NonlinearMediationRequestWire, NonlinearMediationArtifactError> {
    let n = usize::try_from(meta.n_rows).unwrap_or(usize::MAX);
    let p = meta.model_specs.covariates.len();
    let columns = crate::width::columns_from_le(numbers, n, 3 + p)
        .ok_or_else(|| NonlinearMediationArtifactError::Malformed("data section length".into()))?;
    Ok(NonlinearMediationRequestWire {
        premises: meta.premises.clone(),
        config: meta.config.clone(),
        data: data_from_columns(columns, meta.model_specs.covariates.clone()),
    })
}

fn data_from_columns(
    mut columns: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
) -> MediationDataWire {
    let covariates = columns.split_off(3);
    let outcome = columns.pop().unwrap_or_default();
    let mediator = columns.pop().unwrap_or_default();
    let treatment = columns.pop().unwrap_or_default();
    MediationDataWire { treatment, mediator, outcome, covariate_names, covariates }
}

fn premise_records(premises: &MediationPremisesWire) -> Vec<MediationPremiseRecord> {
    let record = |name: &str, status: &str, holds: bool| MediationPremiseRecord {
        name: name.to_owned(),
        status: status.to_owned(),
        holds,
    };
    vec![
        record(
            "no_unmeasured_treatment_outcome_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_treatment_outcome_confounding,
        ),
        record(
            "no_unmeasured_treatment_mediator_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_treatment_mediator_confounding,
        ),
        record(
            "no_unmeasured_mediator_outcome_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_mediator_outcome_confounding,
        ),
        record(
            "no_treatment_induced_mediator_outcome_confounder",
            STATUS_DECLARED,
            premises.treatment_induced_mediator_outcome_confounders.is_empty(),
        ),
        record("cross_world_independence", STATUS_DECLARED, premises.cross_world_independence),
        record("linear_gaussian_mediator", STATUS_DECLARED, true),
        record("polynomial_outcome_in_mediator", STATUS_DECLARED, true),
        record("treatment_binary", STATUS_CHECKED, true),
        record("arm_overlap", STATUS_CHECKED, true),
        record("mediator_support_within_bound", STATUS_CHECKED, true),
        record("integration_error_within_tolerance", STATUS_CHECKED, true),
        record("total_equals_direct_plus_indirect", STATUS_CHECKED, true),
    ]
}

fn model_specs(request: &NonlinearMediationRequestWire) -> MediationModelSpecs {
    MediationModelSpecs {
        mediator: "linear_gaussian_in_treatment_and_covariates_homoscedastic_normal_residual"
            .to_owned(),
        outcome: "polynomial_in_standardized_mediator_with_treatment_interaction_linear_in_treatment_and_covariates"
            .to_owned(),
        integration: "gauss_hermite_n_versus_2n".to_owned(),
        outcome_degree: request.config.outcome_degree,
        quadrature_nodes: request.config.quadrature_nodes,
        covariates: request.data.covariate_names.clone(),
    }
}

fn result_wire(
    estimate: &NonlinearMediationEstimate,
) -> Result<MediationResultWire, NonlinearMediationArtifactError> {
    let boot = &estimate.bootstrap;
    Ok(MediationResultWire {
        natural_direct: estimate.natural_direct,
        natural_indirect: estimate.natural_indirect,
        total: estimate.total,
        natural_direct_se: estimate.natural_direct_se,
        natural_indirect_se: estimate.natural_indirect_se,
        total_se: estimate.total_se,
        identity_residual: estimate.identity_residual,
        integration_error: estimate.integration_error,
        coarse_nodes: count_u32(estimate.quadrature_nodes)?,
        fine_nodes: count_u32(
            estimate
                .quadrature_nodes
                .checked_mul(2)
                .ok_or(NonlinearMediationArtifactError::LimitsExceeded("count"))?,
        )?,
        outcome_degree: count_u32(estimate.outcome_degree)?,
        n_rows: lossless_u64(estimate.n_rows),
        overlap: MediationOverlapWire {
            treated_count: lossless_u64(estimate.overlap.treated_count),
            control_count: lossless_u64(estimate.overlap.control_count),
            treated_mediator_min: estimate.overlap.treated_mediator_range.0,
            treated_mediator_max: estimate.overlap.treated_mediator_range.1,
            mediator_support_violation: estimate.overlap.mediator_support_violation,
        },
        mediator: MediationMediatorWire {
            coefficients: estimate.mediator.coefficients.clone(),
            residual_variance: estimate.mediator.residual_variance,
            residual_skewness: estimate.mediator.residual_skewness,
            residual_excess_kurtosis: estimate.mediator.residual_excess_kurtosis,
        },
        bootstrap: MediationBootstrapWire {
            seed: boot.seed,
            replicates_requested: boot.replicates_requested,
            replicates_succeeded: boot.replicates_succeeded,
            replicate_ids: (0..u64::from(boot.replicates_requested)).collect(),
            failed_replicate_ids: boot.failed_replicate_ids.clone(),
            id_scheme: boot.id_scheme.to_owned(),
            interval_status: estimate.interval_status.to_owned(),
        },
        assumptions: estimate.assumptions.iter().map(|a| (*a).to_owned()).collect(),
    })
}

fn digest<T: Serialize>(tag: &str, value: &T) -> Result<String, NonlinearMediationArtifactError> {
    let bytes = to_cbor(value)?;
    Ok(CanonicalWriter::digest_bytes_hex(tag, &bytes))
}

fn data_digest(data: &MediationDataWire) -> Result<String, NonlinearMediationArtifactError> {
    let names = to_cbor(&data.covariate_names)?;
    let numbers = data_bytes(data);
    let hasher = CanonicalWriter::new("nonlinear_mediation_v1.data")
        .bytes(&names)
        .word(lossless_u64(data.treatment.len()))
        .bytes(&numbers)
        .finish();
    Ok(antecedent_core::to_hex(&hasher))
}

fn build_meta(
    request: &NonlinearMediationRequestWire,
    estimate: &NonlinearMediationEstimate,
) -> Result<NonlinearMediationMeta, NonlinearMediationArtifactError> {
    let records = premise_records(&request.premises);
    let specs = model_specs(request);
    let result = result_wire(estimate)?;
    let premises_id = digest("nonlinear_mediation_v1.premises", &(&request.premises, &records))?;
    let config_id = digest("nonlinear_mediation_v1.config", &(&request.config, &specs))?;
    let data_id = data_digest(&request.data)?;
    let result_id = digest("nonlinear_mediation_v1.result", &result)?;
    let whole = digest(
        "nonlinear_mediation_v1.identity",
        &(&premises_id, &config_id, &data_id, &result_id),
    )?;
    Ok(NonlinearMediationMeta {
        version: NONLINEAR_MEDIATION_ARTIFACT_VERSION,
        feature: NONLINEAR_MEDIATION_ARTIFACT_FEATURE.to_owned(),
        estimand: ESTIMAND_NATURAL.to_owned(),
        premises: request.premises.clone(),
        premise_records: records,
        model_specs: specs,
        config: request.config.clone(),
        n_rows: lossless_u64(request.data.treatment.len()),
        data_binding: None,
        result,
        identity: NonlinearMediationIdentity {
            premises_id,
            config_id,
            data_id,
            result_id,
            digest: whole,
        },
        calibration: NONLINEAR_MEDIATION_CALIBRATION.to_owned(),
        interval_status: NONLINEAR_MEDIATION_INTERVAL_STATUS.to_owned(),
        inference_claim: NONLINEAR_MEDIATION_ARTIFACT_CLAIM.to_owned(),
        caveats: NONLINEAR_MEDIATION_ARTIFACT_CAVEATS.iter().map(|c| (*c).to_owned()).collect(),
    })
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &NonlinearMediationIdentity,
    other: &NonlinearMediationIdentity,
) -> Option<&'static str> {
    if stored.premises_id != other.premises_id {
        Some("premises")
    } else if stored.config_id != other.config_id {
        Some("config")
    } else if stored.data_id != other.data_id {
        Some("data")
    } else if stored.result_id != other.result_id {
        Some("result")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

/// Encode a metadata section and a data section as a checksummed container.
///
/// Hidden: the producer path is [`NonlinearMediationArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &NonlinearMediationMeta,
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
    encode_parts_impl(meta, data, artifact_id, MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES)
}

fn encode_parts_impl(
    meta: &NonlinearMediationMeta,
    data: &[u8],
    artifact_id: &str,
    export_budget: usize,
) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
    // A consumer reads at most the byte cap of uncompressed sections, whatever the container
    // compresses to; the caller's budget bounds the container itself.
    let bytes = crate::artifact_codec::encode_sectioned_admitting(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "nonlinear_mediation_point_with_diagnostics_unmeasured",
            max_bytes: MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[crate::OutSection {
            id: DATA_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: data,
        }],
        |uncompressed| {
            admit_export(BUDGET, uncompressed, MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES).map_err(budget_refusal)
        },
    )?;
    admit_export(BUDGET, bytes.len(), export_budget.min(MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES))
        .map_err(budget_refusal)?;
    Ok(bytes)
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(NonlinearMediationMeta, Vec<u8>), NonlinearMediationArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(NonlinearMediationMeta, Vec<u8>), NonlinearMediationArtifactError> {
    let (meta, parts) = crate::artifact_codec::decode_sectioned::<NonlinearMediationArtifactError, NonlinearMediationMeta>(
        ARTIFACT_KIND,
        MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES,
        META_SECTION,
        &[DATA_SECTION],
        NONLINEAR_MEDIATION_ARTIFACT_VERSION,
        bytes,
    )?;
    let Ok([payload]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(NonlinearMediationArtifactError::Malformed("unsupported container layout".into()));
    };
    Ok((meta, payload))
}

/// A produced or consumed mediation artifact: the declared request, the recomputed estimate and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationArtifact {
    meta: NonlinearMediationMeta,
    request: NonlinearMediationRequestWire,
    estimate: NonlinearMediationEstimate,
}

impl NonlinearMediationArtifact {
    /// Run the core estimator on `request` and seal the answer with its identity, with no
    /// memory limit; see [`Self::seal_with`].
    ///
    /// # Errors
    /// The core estimator's refusals (`effect_not_identified` for declared confounding,
    /// `cross_world_not_identified`, `route_not_supported` for the interventional estimand, ...)
    /// and the format's bounds.
    pub fn seal(
        request: &NonlinearMediationRequestWire,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        Self::seal_with(request, &unbounded_context())
    }

    /// Run the core estimator on `request` under the computation budget of `ctx` and seal the
    /// answer.
    ///
    /// The row count is not limited here: the planned working set must fit the context's hard
    /// memory limit and the context must not be cancelled. How much of the estimate can be
    /// exported is decided later by [`Self::to_bytes_within`].
    ///
    /// # Errors
    /// The refusals of [`Self::seal`], `nonlinear_mediation.memory_budget_exceeded` and
    /// `nonlinear_mediation.cancelled`.
    pub fn seal_with(
        request: &NonlinearMediationRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        let estimate = estimate_of(request, ctx)?;
        let meta = build_meta(request, &estimate)?;
        Ok(Self { meta, request: request.clone(), estimate })
    }

    /// The metadata section, which is also the self-describing report.
    #[must_use]
    pub fn meta(&self) -> &NonlinearMediationMeta {
        &self.meta
    }

    /// The request the estimate was computed on.
    #[must_use]
    pub fn request(&self) -> &NonlinearMediationRequestWire {
        &self.request
    }

    /// The core estimate.
    #[must_use]
    pub fn estimate(&self) -> &NonlinearMediationEstimate {
        &self.estimate
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &NonlinearMediationIdentity {
        &self.meta.identity
    }

    /// Serialize through the checksummed sectioned container, embedding the dataset.
    ///
    /// # Errors
    /// An empty id, an encoding failure or `nonlinear_mediation.export_budget_exceeded` when the
    /// artifact would exceed [`MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES`].
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
        self.to_bytes_within(artifact_id, MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES)
    }

    /// Serialize, embedding the dataset, when the whole artifact fits `export_budget` bytes (at
    /// most [`MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES`]).
    ///
    /// # Errors
    /// An empty id, an encoding failure or `nonlinear_mediation.export_budget_exceeded`, which
    /// names the bytes needed and the budget. The estimate is untouched by the refusal.
    pub fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
        encode_parts_impl(&self.meta, &data_bytes(&self.request.data), artifact_id, export_budget)
    }

    /// Serialize the record without its dataset, bound to it by the digest
    /// `identity.data_id`. A consumer must supply the dataset to
    /// [`Self::from_bytes_with`], which refuses any other.
    ///
    /// # Errors
    /// An empty id or an encoding failure.
    pub fn to_bytes_digest_bound(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
        let mut meta = self.meta.clone();
        meta.data_binding = Some(DATA_BOUND_BY_DIGEST.to_owned());
        encode_parts_impl(&meta, &[], artifact_id, MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES)
    }

    /// Consume an artifact that embeds its dataset by re-estimation, with no memory limit; see
    /// [`Self::from_bytes_with`].
    ///
    /// `expected` is an identity the consumer retained independently; when given, every
    /// identity field must match it, so a resealed change of premises, configuration, dataset
    /// or result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`NonlinearMediationArtifactError::IdentityMismatch`]), a stored record that does not
    /// replay, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&NonlinearMediationIdentity>,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        Self::from_bytes_with(bytes, expected, None, &unbounded_context())
    }

    /// Consume an artifact by re-estimation within the replay budget of `ctx`.
    ///
    /// An artifact that embeds its dataset takes `data = None`. A digest-bound artifact
    /// ([`Self::to_bytes_digest_bound`]) takes the dataset to re-estimate on, and refuses one
    /// whose digest is not the stored `identity.data_id`
    /// (`IdentityMismatch { field: "data" }`) before any work is done on it.
    ///
    /// # Errors
    /// The refusals of [`Self::from_bytes`], `nonlinear_mediation.data_required` for a
    /// digest-bound artifact given no dataset, and `nonlinear_mediation.memory_budget_exceeded`
    /// or `nonlinear_mediation.cancelled` from the context.
    pub fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&NonlinearMediationIdentity>,
        data: Option<&MediationDataWire>,
        ctx: &ExecutionContext,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        let (meta, payload) = decode_parts_impl(bytes)?;
        let unsupported = NonlinearMediationArtifactError::UnsupportedSemantics;
        if meta.feature != NONLINEAR_MEDIATION_ARTIFACT_FEATURE {
            return Err(unsupported("feature marker"));
        }
        if meta.estimand != ESTIMAND_NATURAL {
            return Err(unsupported("estimand"));
        }
        if meta.inference_claim != NONLINEAR_MEDIATION_ARTIFACT_CLAIM {
            return Err(unsupported("inference claim"));
        }
        if meta.calibration != NONLINEAR_MEDIATION_CALIBRATION {
            return Err(unsupported("calibration"));
        }
        if meta.interval_status != NONLINEAR_MEDIATION_INTERVAL_STATUS {
            return Err(unsupported("interval status"));
        }
        if meta.caveats != NONLINEAR_MEDIATION_ARTIFACT_CAVEATS {
            return Err(unsupported("caveats"));
        }
        let request = match (meta.data_binding.as_deref(), data) {
            (None, None) => request_of(&meta, &payload)?,
            (None, Some(_)) => {
                return Err(unsupported("a dataset was supplied for an artifact that embeds it"));
            }
            (Some(DATA_BOUND_BY_DIGEST), Some(supplied)) => {
                bound_request(&meta, &payload, supplied)?
            }
            (Some(DATA_BOUND_BY_DIGEST), None) => {
                return Err(NonlinearMediationArtifactError::Refused {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    message: format!(
                        "nonlinear_mediation.data_required: the artifact is bound to its dataset \
                         by digest {}; supply the dataset to replay it",
                        meta.identity.data_id
                    ),
                });
            }
            (Some(_), _) => return Err(unsupported("data binding")),
        };
        let estimate = estimate_of(&request, ctx)?;
        let rebuilt = build_meta(&request, &estimate)?;
        if let Some(field) = identity_diff(&meta.identity, &rebuilt.identity) {
            return Err(if field == "result" {
                NonlinearMediationArtifactError::ResultMismatch("estimate")
            } else {
                NonlinearMediationArtifactError::IdentityMismatch { field }
            });
        }
        // The binding describes the container, not the estimate: compare the record as stored.
        let stored_form =
            NonlinearMediationMeta { data_binding: meta.data_binding.clone(), ..rebuilt.clone() };
        if to_cbor(&meta)? != to_cbor(&stored_form)? {
            return Err(NonlinearMediationArtifactError::ResultMismatch("record"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &rebuilt.identity) {
                return Err(NonlinearMediationArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta: rebuilt, request, estimate })
    }
}

/// The request of a digest-bound record: the stored declarations on the supplied dataset, once
/// the dataset's digest is the stored one.
fn bound_request(
    meta: &NonlinearMediationMeta,
    payload: &[u8],
    supplied: &MediationDataWire,
) -> Result<NonlinearMediationRequestWire, NonlinearMediationArtifactError> {
    if !payload.is_empty() {
        return Err(NonlinearMediationArtifactError::Malformed(
            "a digest-bound artifact embeds no data".into(),
        ));
    }
    if data_digest(supplied)? != meta.identity.data_id {
        return Err(NonlinearMediationArtifactError::IdentityMismatch { field: "data" });
    }
    Ok(NonlinearMediationRequestWire {
        premises: meta.premises.clone(),
        config: meta.config.clone(),
        data: supplied.clone(),
    })
}

#[cfg(test)]
mod data_section_tests {
    use super::data_bytes;
    use crate::nonlinear_mediation_artifact::MediationDataWire;

    #[test]
    fn data_section_round_trips_signed_zero_and_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let data = MediationDataWire {
            treatment: vec![-0.0, 1.0],
            mediator: vec![nan, -1.5],
            outcome: vec![0.0, f64::MIN_POSITIVE],
            covariate_names: vec!["x".into()],
            covariates: vec![vec![-0.0, nan]],
        };
        let bytes = data_bytes(&data);
        let columns = crate::width::columns_from_le(&bytes, 2, 4).unwrap();
        let again = super::data_from_columns(columns, data.covariate_names.clone());
        assert_eq!(data_bytes(&again), bytes);
        let bits = |sample: &[f64]| sample.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&again.treatment), bits(&data.treatment));
        assert_eq!(bits(&again.mediator), bits(&data.mediator));
        assert_eq!(bits(&again.outcome), bits(&data.outcome));
        assert_eq!(bits(&again.covariates[0]), bits(&data.covariates[0]));
    }
}
