//! Joint mechanism deviations of the registered surrogate z formula on the
//! prepared z stage, and their portable replay artifact (2.2B B3, X3).
//!
//! The prepared stage evaluates the exact joint assumption range on its
//! retained laws without re-identifying. The artifact is format version 3 of
//! the z sensitivity family: a separate wire from the v2 one-factor artifact
//! (which is unchanged, and refuses version 3; this reader refuses version 2).
//! It embeds the checked baseline point artifact, the perturbation
//! definition, the exact range, axis tipping points, frontier brackets, the
//! optimization receipt and the declared (withheld) sampling method, with a
//! premises digest and a separate data digest.
//!
//! A consumer refuses stored limits above its own maxima before any work,
//! checks both digests, refuses an interval or a relabelled claim, replays the
//! baseline under its own limits and recomputes every stored number under the
//! producer's stored search limits and effective memory cap, compared bit for
//! bit. What replay does NOT protect against: a producer that consistently
//! forges the baseline laws themselves (the embedded point artifact's own
//! consumer only checks that the laws reproduce its stored point); a
//! different but self-consistent choice of declared perturbation; and a
//! re-sealed change of the declared memory cap that stays at or above the
//! stored effective cap (and within the consumer's maxima). The numbers depend
//! only on the effective cap, and an effective cap below the declared one is
//! what any producer with a context hard limit writes, so replay can only
//! enforce `effective <= declared`; the declared cap is bound by the premises
//! digest, not by replay. The replay proves the numbers follow from the stored
//! premises, not that the premises are the scientifically right ones.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SearchStop, reason_code};
use antecedent_io::z_transport_artifact::{ZTransportArtifactWire, ZTransportConsumeLimits};
use antecedent_io::{IoError, from_cbor, to_cbor};
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityError, JointSensitivityLimits, TippingStatus,
    z_transport_joint_mechanism_sensitivity,
};
use serde::{Deserialize, Serialize};

use super::PreparedZTransport;
use super::joint_sensitivity_uncertainty::JointSamplingWire;

/// The joint sensitivity artifact format this reader writes and accepts.
pub const JOINT_SENSITIVITY_ARTIFACT_VERSION: u32 = 3;

fn joint_error(error: &JointSensitivityError) -> IoError {
    IoError::Refused { code: error.reason_code(), message: error.to_string() }
}

impl PreparedZTransport {
    /// Evaluate the exact joint assumption range of the retained checked
    /// surrogate formula on the retained laws; nothing is re-identified.
    ///
    /// # Errors
    ///
    /// The reason-coded refusal of the joint sensitivity contract.
    pub fn joint_mechanism_sensitivity(
        &self,
        spec: &JointDeviationSpec,
        ctx: &ExecutionContext,
    ) -> Result<JointMechanismSensitivityResult, IoError> {
        z_transport_joint_mechanism_sensitivity(
            self.diagram(),
            self.functional(),
            self.data(),
            spec,
            ctx,
        )
        .map_err(|error| joint_error(&error))
    }
}

/// One declared factor on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointFactorWire {
    /// Factor wire name.
    pub factor: String,
    /// Declared bound.
    pub max_fraction: f64,
}

/// The declared perturbation and analysis settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointPerturbationWire {
    /// Factors in canonical order.
    pub factors: Vec<JointFactorWire>,
    /// Decision threshold.
    pub decision_threshold: Option<f64>,
    /// Bracketing tolerance.
    pub tolerance: f64,
    /// Frontier grid size.
    pub frontier_points: usize,
    /// Declared search limits: operations, depth, memory cap.
    pub operations: usize,
    /// Declared depth limit.
    pub depth: usize,
    /// Declared memory cap.
    pub memory_bytes: u64,
}

/// Where the baseline came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointProvenanceWire {
    /// Digest of the embedded baseline artifact bytes.
    pub baseline_artifact_digest: String,
    /// The baseline artifact's own premises digest.
    pub baseline_premises_digest: String,
    /// Provider snapshots of the baseline laws.
    pub provider_snapshots: Vec<String>,
    /// Source regime of the formula.
    pub source_regime: u32,
    /// Query binding of the checked derivation.
    pub query_binding: String,
}

/// A tipping bracket on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointTippingWire {
    /// Held coordinate (parent fraction for frontier lines, `None` on an axis).
    pub parent_fraction: Option<f64>,
    /// Factor searched.
    pub factor: String,
    /// 2.1 analytic value on an axis.
    pub analytic: Option<f64>,
    /// Search status.
    pub status: String,
    /// `[lower, upper]` when resolved.
    pub bracket: Option<[f64; 2]>,
    /// Bisection iterations.
    pub iterations: usize,
}

/// The replayed numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointOutcomeWire {
    /// Baseline response.
    pub baseline: f64,
    /// Exact `[minimum, maximum]` over the box.
    pub range: [f64; 2],
    /// Kernel vertex per stratum at the minimum.
    pub minimizing_outcome_by_stratum: Vec<usize>,
    /// Kernel vertex per stratum at the maximum.
    pub maximizing_outcome_by_stratum: Vec<usize>,
    /// Parent vertex at the minimum.
    pub minimizing_parent_level: Option<usize>,
    /// Parent vertex at the maximum.
    pub maximizing_parent_level: Option<usize>,
    /// Axis tipping points.
    pub axis: Vec<JointTippingWire>,
    /// Frontier lines.
    pub frontier: Vec<JointTippingWire>,
    /// `joint_sensitivity.budget` when lines are unresolved.
    pub unresolved_detail: Option<String>,
}

/// The optimization receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointReceiptWire {
    /// Optimization method.
    pub method: String,
    /// `[kernel, parent]` fractions.
    pub fraction_box: [f64; 2],
    /// Effective memory cap the producer ran under.
    pub memory_limit_bytes: u64,
    /// Operations charged.
    pub operations_consumed: usize,
    /// Deepest bisection level.
    pub depth_reached: usize,
    /// Peak live-state estimate.
    pub live_state_bytes: u64,
    /// `search.<stop>` when the budget stopped bracketing.
    pub stop: Option<String>,
    /// Stages evaluated.
    pub explored: Vec<String>,
    /// Stages left unevaluated.
    pub unevaluated: Vec<String>,
}

/// Everything the replay recomputes and compares.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSensitivityBodyWire {
    /// Declared perturbation.
    pub perturbation: JointPerturbationWire,
    /// Baseline provenance.
    pub provenance: JointProvenanceWire,
    /// Replayed numbers.
    pub outcome: JointOutcomeWire,
    /// Optimization receipt.
    pub receipt: JointReceiptWire,
    /// Interpretation statement.
    pub interpretation: String,
    /// `assumption_range`.
    pub inference_claim: String,
    /// Declared sampling method.
    pub sampling: JointSamplingWire,
}

/// Versioned replay artifact of a joint mechanism sensitivity analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSensitivityArtifactWire {
    /// Format version ([`JOINT_SENSITIVITY_ARTIFACT_VERSION`]).
    pub version: u32,
    /// Complete checked baseline point artifact.
    pub baseline_artifact: Vec<u8>,
    /// Semantic body.
    pub body: JointSensitivityBodyWire,
    /// Digest of the scientific premises.
    pub premises_digest: String,
    /// Digest of the data identity.
    pub data_digest: String,
}

/// Consumer-side maxima; nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct JointSensitivityConsumeLimits {
    /// Limits of the baseline point replay.
    pub baseline: ZTransportConsumeLimits,
    /// Largest stored operation limit the consumer replays.
    pub max_operations: usize,
    /// Largest stored depth limit the consumer replays.
    pub max_depth: usize,
    /// Largest stored memory cap the consumer replays.
    pub max_memory_bytes: u64,
}

impl Default for JointSensitivityConsumeLimits {
    fn default() -> Self {
        let declared = JointSensitivityLimits::default();
        Self {
            baseline: ZTransportConsumeLimits::default(),
            max_operations: declared.operations,
            max_depth: declared.depth,
            max_memory_bytes: declared.memory_bytes,
        }
    }
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

fn hex(domain: &str, bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    antecedent_io::identity::payload_digest(domain, bytes).iter().fold(
        String::new(),
        |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        },
    )
}

fn stop_name(stop: SearchStop) -> String {
    stop.code().to_owned()
}

fn tipping_wire(
    parent_fraction: Option<f64>,
    factor: JointFactor,
    analytic: Option<f64>,
    status: TippingStatus,
    bracket: Option<antecedent_validate::TippingBracket>,
) -> JointTippingWire {
    JointTippingWire {
        parent_fraction,
        factor: factor.name().to_owned(),
        analytic,
        status: status.name().to_owned(),
        bracket: bracket.map(|b| [b.lower, b.upper]),
        iterations: bracket.map_or(0, |b| b.iterations),
    }
}

impl JointSensitivityArtifactWire {
    /// Replay the baseline artifact under the default consumer limits and
    /// build a checked joint sensitivity artifact.
    ///
    /// # Errors
    ///
    /// As [`Self::checked_with_limits`].
    pub fn checked(
        baseline_artifact: Vec<u8>,
        spec: &JointDeviationSpec,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Self::checked_with_limits(baseline_artifact, spec, ZTransportConsumeLimits::default(), ctx)
    }

    /// Build a checked artifact: the baseline is replayed through its own
    /// consumer under `limits`, the joint analysis runs on the replayed laws,
    /// and both digests are sealed. A frontier stopped by cancellation is not
    /// replayable and is refused.
    ///
    /// # Errors
    ///
    /// A baseline that does not replay, a joint-sensitivity refusal, or a
    /// cancelled analysis (`transport_budget_cancel` / `joint_sensitivity.budget`).
    pub fn checked_with_limits(
        baseline_artifact: Vec<u8>,
        spec: &JointDeviationSpec,
        limits: ZTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let consumed =
            ZTransportArtifactWire::consume_with_limits(&baseline_artifact, limits, ctx)?;
        let result = z_transport_joint_mechanism_sensitivity(
            &consumed.diagram,
            &consumed.functional,
            &consumed.data,
            spec,
            ctx,
        )
        .map_err(|error| joint_error(&error))?;
        if result.receipt.stop == Some(SearchStop::Cancelled) {
            return Err(IoError::Refused {
                code: reason_code!("transport_budget_cancel"),
                message: "joint_sensitivity.budget: a cancelled analysis is not replayable and is not exported".into(),
            });
        }
        let body = JointSensitivityBodyWire {
            perturbation: JointPerturbationWire {
                factors: result
                    .factors
                    .iter()
                    .map(|b| JointFactorWire {
                        factor: b.factor.name().to_owned(),
                        max_fraction: b.max_fraction,
                    })
                    .collect(),
                decision_threshold: spec.decision_threshold,
                tolerance: spec.tolerance,
                frontier_points: spec.frontier_points,
                operations: spec.limits.operations,
                depth: spec.limits.depth,
                memory_bytes: spec.limits.memory_bytes,
            },
            provenance: JointProvenanceWire {
                baseline_artifact_digest: hex("joint_sensitivity_baseline", &baseline_artifact),
                baseline_premises_digest: consumed.wire.premises_digest.clone(),
                provider_snapshots: consumed
                    .data
                    .laws()
                    .iter()
                    .map(|law| law.snapshot_identity().to_owned())
                    .collect(),
                source_regime: result.source_regime.raw(),
                query_binding: result.query_binding.clone(),
            },
            outcome: JointOutcomeWire {
                baseline: result.baseline,
                range: [result.range.minimum, result.range.maximum],
                minimizing_outcome_by_stratum: result.range.minimizing_outcome_by_stratum.clone(),
                maximizing_outcome_by_stratum: result.range.maximizing_outcome_by_stratum.clone(),
                minimizing_parent_level: result.range.minimizing_parent_level,
                maximizing_parent_level: result.range.maximizing_parent_level,
                axis: result
                    .axis_tipping
                    .iter()
                    .map(|a| tipping_wire(None, a.factor, a.analytic, a.status, a.bracket))
                    .collect(),
                frontier: result
                    .frontier
                    .iter()
                    .map(|p| {
                        tipping_wire(
                            Some(p.parent_fraction),
                            JointFactor::OutcomeKernel,
                            None,
                            p.status,
                            p.bracket,
                        )
                    })
                    .collect(),
                unresolved_detail: result.unresolved_detail.map(str::to_owned),
            },
            receipt: JointReceiptWire {
                method: result.receipt.method.to_owned(),
                fraction_box: result.receipt.fraction_box,
                memory_limit_bytes: result.receipt.memory_limit_bytes,
                operations_consumed: result.receipt.operations_consumed,
                depth_reached: result.receipt.depth_reached,
                live_state_bytes: result.receipt.live_state_bytes,
                stop: result.receipt.stop.map(stop_name),
                explored: result.receipt.explored.clone(),
                unevaluated: result.receipt.unevaluated.clone(),
            },
            interpretation: result.interpretation.to_owned(),
            inference_claim: result.inference_claim.to_owned(),
            sampling: JointSamplingWire {
                method: result.uncertainty.method.to_owned(),
                coverage_target: result.uncertainty.coverage_target.to_owned(),
                status: "withheld".into(),
                reason_code: result.uncertainty.reason_code.to_owned(),
                interval: None,
            },
        };
        let mut wire = Self {
            version: JOINT_SENSITIVITY_ARTIFACT_VERSION,
            baseline_artifact,
            body,
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = wire.computed_premises_digest()?;
        wire.data_digest = wire.computed_data_digest()?;
        Ok(wire)
    }

    /// Digest of the scientific premises: the baseline premises digest, the
    /// perturbation definition with its limits, the method, interpretation,
    /// inference claim and declared sampling method.
    ///
    /// # Errors
    ///
    /// Serialization failure.
    pub fn computed_premises_digest(&self) -> Result<String, IoError> {
        let premises = (
            &self.body.provenance.baseline_premises_digest,
            &self.body.perturbation,
            &self.body.receipt.method,
            &self.body.interpretation,
            &self.body.inference_claim,
            (
                &self.body.sampling.method,
                &self.body.sampling.coverage_target,
                &self.body.sampling.status,
                &self.body.sampling.reason_code,
            ),
        );
        Ok(hex("joint_sensitivity_premises", &to_cbor(&premises)?))
    }

    /// Digest of the data identity: the baseline bytes, provider snapshots
    /// and source regime.
    ///
    /// # Errors
    ///
    /// Serialization failure.
    pub fn computed_data_digest(&self) -> Result<String, IoError> {
        let data = (
            hex("joint_sensitivity_baseline", &self.baseline_artifact),
            &self.body.provenance.baseline_artifact_digest,
            &self.body.provenance.provider_snapshots,
            self.body.provenance.source_regime,
        );
        Ok(hex("joint_sensitivity_data", &to_cbor(&data)?))
    }

    /// Export as a portable CBOR item.
    ///
    /// # Errors
    ///
    /// Serialization failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        to_cbor(self)
    }

    /// [`Self::consume_with_limits`] under the default consumer limits.
    ///
    /// # Errors
    ///
    /// As [`Self::consume_with_limits`].
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<Self, IoError> {
        Self::consume_with_limits(bytes, JointSensitivityConsumeLimits::default(), ctx)
    }

    /// Independently verify an exported artifact and return the recomputed one.
    ///
    /// Order: version, decoding (unknown fields refused), stored limits against
    /// the consumer's maxima, the producer's 512 MiB memory ceiling and the
    /// consumer's hard memory limit (before any work), premises
    /// and data digests, the interval and inference-claim labels, then the
    /// baseline replay and a bit-for-bit recomputation of the body under the
    /// stored limits and effective memory cap.
    ///
    /// # Errors
    ///
    /// `UnsupportedVersion`, a decoding error, or a reason-coded
    /// `joint_sensitivity.*` refusal naming the failed check.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: JointSensitivityConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let peek: VersionPeek = from_cbor(bytes)?;
        if peek.version != JOINT_SENSITIVITY_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = from_cbor(bytes)?;
        let declared = &wire.body.perturbation;
        let effective = wire.body.receipt.memory_limit_bytes;
        let affordable = ctx.memory.hard_limit_bytes.is_none_or(|hard| hard >= effective);
        if declared.operations > limits.max_operations
            || declared.depth > limits.max_depth
            || declared.memory_bytes > limits.max_memory_bytes
            || declared.memory_bytes
                > antecedent_validate::joint_mechanism_sensitivity::JOINT_SENSITIVITY_MAX_MEMORY_BYTES
            || effective > declared.memory_bytes
            || !affordable
        {
            return Err(IoError::Refused {
                code: reason_code!("route_not_supported"),
                message: format!(
                    "joint_sensitivity.consumer_limits: stored operations {}, depth {}, memory cap {} (effective {}) exceed this consumer's maxima",
                    declared.operations, declared.depth, declared.memory_bytes, effective
                ),
            });
        }
        if wire.computed_premises_digest()? != wire.premises_digest {
            return Err(IoError::Refused {
                code: reason_code!("transport_not_certified"),
                message: "joint_sensitivity.premises_mismatch: the premises digest does not match the stored premises".into(),
            });
        }
        if wire.computed_data_digest()? != wire.data_digest {
            return Err(IoError::Refused {
                code: reason_code!("transport_not_certified"),
                message: "joint_sensitivity.data_identity_mismatch: the data digest does not match the baseline bytes, snapshots or regime".into(),
            });
        }
        if wire.body.sampling.interval.is_some() {
            return Err(IoError::Refused {
                code: reason_code!("cell_not_licensed"),
                message: "joint_sensitivity.interval_withheld: an artifact carrying a sampling interval is refused; the interval route is not licensed".into(),
            });
        }
        if wire.body.inference_claim
            != antecedent_validate::joint_mechanism_sensitivity::ASSUMPTION_RANGE_CLAIM
        {
            return Err(IoError::Refused {
                code: reason_code!("estimator_inference_mismatch"),
                message: format!(
                    "joint_sensitivity.union_labelled_ci: the assumption range is labelled {:?}; a union over the sensitivity set is never a confidence interval",
                    wire.body.inference_claim
                ),
            });
        }
        let spec = wire.replay_spec()?;
        let recomputed =
            Self::checked_with_limits(wire.baseline_artifact.clone(), &spec, limits.baseline, ctx)?;
        let mut expected = recomputed.body;
        // The declared cap is a premise (premises digest) that replay cannot
        // re-derive: the replay ran under the stored effective cap, which the
        // receipt comparison checks, and only `effective <= declared` is
        // enforced above. A re-sealed declared cap at or above the effective
        // one is therefore accepted (see the module docs).
        expected.perturbation.memory_bytes = wire.body.perturbation.memory_bytes;
        if to_cbor(&expected)? != to_cbor(&wire.body)? {
            return Err(IoError::Refused {
                code: reason_code!("transport_not_certified"),
                message: "joint_sensitivity.replay_mismatch: the recomputed range, frontier, receipt or sampling descriptor differs from the stored one".into(),
            });
        }
        Ok(wire)
    }

    /// The declared spec the replay runs under: stored factors, threshold,
    /// tolerance, grid and limits, with the stored effective memory cap.
    fn replay_spec(&self) -> Result<JointDeviationSpec, IoError> {
        let declared = &self.body.perturbation;
        let factors = declared
            .factors
            .iter()
            .map(|f| {
                JointFactor::from_name(&f.factor)
                    .map(|factor| JointFactorBound { factor, max_fraction: f.max_fraction })
                    .ok_or_else(|| IoError::Refused {
                        code: reason_code!("invalid_argument"),
                        message: format!("joint_sensitivity.unknown_factor: {:?}", f.factor),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(JointDeviationSpec {
            factors,
            total_budget: None,
            decision_threshold: declared.decision_threshold,
            tolerance: declared.tolerance,
            frontier_points: declared.frontier_points,
            limits: JointSensitivityLimits {
                operations: declared.operations,
                depth: declared.depth,
                memory_bytes: self.body.receipt.memory_limit_bytes,
            },
        })
    }
}
