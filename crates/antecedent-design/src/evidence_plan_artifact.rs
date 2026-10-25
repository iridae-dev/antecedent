//! Durable evidence plan (`evidence_plan_v1`) and its independent consumer.
//!
//! The artifact carries the plan's whole derivation, not a summary: one sealed
//! `repair_search_receipt_v1` container per theorem route (premises, candidates,
//! obligations, every subset's classification and the search receipt) and, when the
//! plan is valued, the sealed `design_ranking_v1` container its values came from,
//! together with the ordered options, the routes no checker decides and the plan's
//! identity under a digest.
//!
//! The consumer trusts none of the stored classifications or values. It first
//! checks the container, the version and the digest, then consumes every embedded
//! repair report under its own limits (rebuilding each family, re-running its
//! theorem checker on every stored hypothetical delta and replaying the whole
//! search), consumes the embedded ranking by independent recomputation, rebuilds the
//! plan from those replayed parts and accepts only an identical option list, outcome
//! and identity. A resealed edit of an option, a gate, a cost or a value is refused
//! with `evidence_plan_artifact.replay_mismatch` naming the section.
//!
//! A search stopped by a budget stays a budget: the replayed reports keep their
//! unevaluated subsets, and nothing here states that no study closes the gap.
//! Replay does not protect against a producer that states other premises or another
//! candidate universe and re-seals honestly, and it does not check that arriving
//! studies will deliver their declared evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, blake3_hex, reason_code};
use antecedent_io_core::convert::{from_cbor, to_cbor};
use serde::{Deserialize, Serialize};

use crate::design_ranking_artifact::{
    ConsumeExpectation, DesignRankingArtifactError, DesignRankingArtifactWire, consume,
};
use crate::evidence_plan::{
    EvidencePlan, EvidencePlanError, PlanInputs, PlanOption, PlanValue, UnverifiedRoute,
};
use crate::repair::RepairReport;
use crate::repair_artifact::{RepairArtifactError, RepairConsumeLimits, RepairReportArtifact};

/// Wire schema version.
pub const EVIDENCE_PLAN_ARTIFACT_VERSION: u32 = 1;
/// Artifact kind.
pub const EVIDENCE_PLAN_ARTIFACT_KIND: &str = "evidence_plan_v1";
/// Largest container, in bytes, a consumer reads.
pub const MAX_EVIDENCE_PLAN_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most theorem routes one plan embeds.
pub const MAX_PLAN_ROUTES: usize = 16;
/// Most options a stored plan may list.
pub const MAX_PLAN_OPTIONS: usize = 16_384;

const BODY_SECTION: &str = "plan_body";

/// A refused plan artifact: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct EvidencePlanArtifactError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `evidence_plan_artifact.*` (or the embedded artifact's own) detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl EvidencePlanArtifactError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "evidence_plan_artifact.invalid_artifact",
            message: message.into(),
        }
    }

    fn unsupported_version(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "evidence_plan_artifact.unsupported_version",
            message: message.into(),
        }
    }

    fn bounds(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("route_not_supported"),
            detail: "evidence_plan_artifact.bounds_exceeded",
            message: message.into(),
        }
    }

    fn mismatch(section: &str, message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "evidence_plan_artifact.replay_mismatch",
            message: format!("{section}: {}", message.into()),
        }
    }
}

impl From<RepairArtifactError> for EvidencePlanArtifactError {
    fn from(error: RepairArtifactError) -> Self {
        Self { code: error.code, detail: error.detail, message: error.message }
    }
}

impl From<EvidencePlanError> for EvidencePlanArtifactError {
    fn from(error: EvidencePlanError) -> Self {
        Self { code: error.code, detail: error.detail, message: error.message }
    }
}

impl From<DesignRankingArtifactError> for EvidencePlanArtifactError {
    fn from(error: DesignRankingArtifactError) -> Self {
        match error.refusal() {
            Some(refusal) => Self {
                code: refusal.code,
                detail: "evidence_plan_artifact.value_replay",
                message: format!("{}: {}", refusal.detail, error),
            },
            None => Self::invalid(error.to_string()),
        }
    }
}

/// A route no theorem checker decides, on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnverifiedRouteWire {
    /// The route.
    pub route: String,
    /// The obligations that route owes.
    pub obligations: Vec<String>,
    /// Why nothing verifies it.
    pub reason: String,
}

/// A stored option's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionValueWire {
    /// Expected value of sample information.
    pub evsi: f64,
    /// Expected value of perfect information.
    pub evpi: f64,
    /// Standard error of the EVSI.
    pub mc_stderr: f64,
    /// Integration method name.
    pub integration: String,
    /// Net value, only with a cost map.
    pub net_value: Option<f64>,
}

/// A stored option.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanOptionWire {
    /// Position in the plan's order.
    pub rank: u64,
    /// Route name.
    pub route: String,
    /// Semantic ids of the studies.
    pub members: Vec<String>,
    /// Gate name.
    pub gate: String,
    /// The checker's reasons.
    pub reasons: Vec<String>,
    /// Obligation ids the evidence passes the necessary screen for.
    pub resolves: Vec<String>,
    /// Every other obligation id of the route.
    pub unresolved: Vec<String>,
    /// Total declared cost units.
    pub cost_units: u64,
    /// Total declared sample budget.
    pub sample_budget: u64,
    /// The value, for a valued study.
    pub value: Option<OptionValueWire>,
    /// Whether the value could not separate it from the next option.
    pub rank_uncertain: bool,
}

fn option_wire(option: &PlanOption) -> PlanOptionWire {
    PlanOptionWire {
        rank: option.rank as u64,
        route: option.route.to_owned(),
        members: option.members.iter().map(ToString::to_string).collect(),
        gate: option.gate.as_str().to_owned(),
        reasons: option.reasons.clone(),
        resolves: option.resolves.iter().map(ToString::to_string).collect(),
        unresolved: option.unresolved.iter().map(ToString::to_string).collect(),
        cost_units: option.cost.units,
        sample_budget: option.cost.sample_budget,
        value: option.value.as_ref().map(|v| OptionValueWire {
            evsi: v.evsi,
            evpi: v.evpi,
            mc_stderr: v.mc_stderr,
            integration: v.integration.to_owned(),
            net_value: v.net_value,
        }),
        rank_uncertain: option.rank_uncertain,
    }
}

fn unverified_wire(route: &UnverifiedRoute) -> UnverifiedRouteWire {
    UnverifiedRouteWire {
        route: route.route.to_string(),
        obligations: route.obligations.iter().map(ToString::to_string).collect(),
        reason: route.reason.to_string(),
    }
}

/// A durable, independently replayable evidence plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePlanArtifact {
    /// Schema version.
    pub version: u32,
    /// `evidence_plan_v1`.
    pub kind: String,
    /// One sealed `repair_search_receipt_v1` container per route, in route order.
    pub repairs: Vec<Vec<u8>>,
    /// The sealed `design_ranking_v1` container the values came from, if valued.
    pub ranking: Option<Vec<u8>>,
    /// Digest of that ranking artifact, as the plan recorded it.
    pub value_identity: Option<String>,
    /// Routes no theorem checker decides.
    pub unverified: Vec<UnverifiedRouteWire>,
    /// The ordered options.
    pub options: Vec<PlanOptionWire>,
    /// `closing`, `none_certified` or `budget_exhausted`.
    pub outcome: String,
    /// The plan's identity.
    pub plan_identity: String,
    /// Digest of everything above.
    pub digest: String,
}

impl EvidencePlanArtifact {
    fn compute_digest(&self) -> Result<String, EvidencePlanArtifactError> {
        let mut body = self.clone();
        body.digest = String::new();
        let bytes =
            to_cbor(&body).map_err(|e| EvidencePlanArtifactError::invalid(e.to_string()))?;
        Ok(blake3_hex(&bytes))
    }

    /// Recompute the digest after an edit; a producer's last step.
    ///
    /// # Errors
    /// An encoding failure.
    pub fn sealed(mut self) -> Result<Self, EvidencePlanArtifactError> {
        self.digest = self.compute_digest()?;
        Ok(self)
    }

    /// Seal a finished plan with the artifacts it was computed from.
    ///
    /// `repairs` are the per-route repair artifacts in the plan's route order and
    /// `ranking` the value evaluation's artifact, present exactly when the plan is
    /// valued.
    ///
    /// # Errors
    /// A route or value that does not match the plan, or an encoding failure.
    pub fn build(
        repairs: &[RepairReportArtifact],
        ranking: Option<&DesignRankingArtifactWire>,
        plan: &EvidencePlan,
    ) -> Result<Self, EvidencePlanArtifactError> {
        if repairs.is_empty()
            || repairs.len() != plan.routes.len()
            || repairs.len() > MAX_PLAN_ROUTES
        {
            return Err(EvidencePlanArtifactError::invalid(
                "one repair artifact per route of the plan is required",
            ));
        }
        for (artifact, route) in repairs.iter().zip(&plan.routes) {
            if artifact.premises.family != route.family
                || artifact.premises.contract != route.contract
            {
                return Err(EvidencePlanArtifactError::invalid(
                    "a repair artifact does not belong to its route of the plan",
                ));
            }
        }
        if ranking.is_some() != plan.value_identity.is_some()
            || ranking.is_some_and(|wire| Some(&wire.digest) != plan.value_identity.as_ref())
        {
            return Err(EvidencePlanArtifactError::invalid(
                "the value artifact must be exactly the evaluation the plan was valued by",
            ));
        }
        let mut stored = Vec::with_capacity(repairs.len());
        for artifact in repairs {
            stored.push(artifact.to_bytes("evidence-plan-route")?);
        }
        let ranking_bytes = ranking
            .map(|wire| {
                wire.to_bytes("evidence-plan-value")
                    .map_err(|e| EvidencePlanArtifactError::invalid(e.to_string()))
            })
            .transpose()?;
        let mut artifact = Self {
            version: EVIDENCE_PLAN_ARTIFACT_VERSION,
            kind: EVIDENCE_PLAN_ARTIFACT_KIND.to_owned(),
            repairs: stored,
            ranking: ranking_bytes,
            value_identity: plan.value_identity.clone(),
            unverified: plan.unverified.iter().map(unverified_wire).collect(),
            options: plan.options.iter().map(option_wire).collect(),
            outcome: plan.outcome.as_str().to_owned(),
            plan_identity: plan.identity(),
            digest: String::new(),
        };
        artifact.digest = artifact.compute_digest()?;
        Ok(artifact)
    }

    /// Serialize through the checksummed, sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an oversized payload or an encoding failure.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, EvidencePlanArtifactError> {
        if artifact_id.trim().is_empty() {
            return Err(EvidencePlanArtifactError::invalid("missing artifact id"));
        }
        let body = to_cbor(self).map_err(|e| EvidencePlanArtifactError::invalid(e.to_string()))?;
        antecedent_io_core::encode_container(
            &antecedent_io_core::ArtifactContainer {
                kind: EVIDENCE_PLAN_ARTIFACT_KIND,
                note: "evidence_plan",
                max_bytes: MAX_EVIDENCE_PLAN_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io_core::OutSection {
                id: BODY_SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &body,
            }],
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::FileTooLarge => {
                EvidencePlanArtifactError::bounds("the evidence plan artifact is too large")
            }
            antecedent_io_core::CodecError::MissingArtifactId => {
                EvidencePlanArtifactError::invalid("missing artifact id")
            }
            other => EvidencePlanArtifactError::invalid(
                other.into_io("evidence plan artifact").to_string(),
            ),
        })
    }

    /// Decode a container under the bounded reader. The decoded artifact is not
    /// trusted until [`Self::consume`] replays it.
    ///
    /// # Errors
    /// `evidence_plan_artifact.bounds_exceeded` for an oversized container,
    /// `evidence_plan_artifact.unsupported_version` for another version or kind and
    /// `evidence_plan_artifact.invalid_artifact` for a corrupt container.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EvidencePlanArtifactError> {
        let parts = antecedent_io_core::decode_container(
            EVIDENCE_PLAN_ARTIFACT_KIND,
            MAX_EVIDENCE_PLAN_ARTIFACT_BYTES,
            &[BODY_SECTION],
            bytes,
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::FileTooLarge => {
                EvidencePlanArtifactError::bounds("the evidence plan artifact is too large")
            }
            antecedent_io_core::CodecError::SectionsTooLarge => {
                EvidencePlanArtifactError::bounds("the evidence plan body section is too large")
            }
            antecedent_io_core::CodecError::Layout => {
                EvidencePlanArtifactError::unsupported_version(
                    "unsupported evidence plan artifact kind or layout",
                )
            }
            other => EvidencePlanArtifactError::invalid(
                other.into_io("evidence plan artifact").to_string(),
            ),
        })?;
        let Ok([body]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(EvidencePlanArtifactError::unsupported_version(
                "unsupported evidence plan artifact kind or layout",
            ));
        };
        let artifact: Self =
            from_cbor(&body).map_err(|e| EvidencePlanArtifactError::invalid(e.to_string()))?;
        if artifact.version != EVIDENCE_PLAN_ARTIFACT_VERSION
            || artifact.kind != EVIDENCE_PLAN_ARTIFACT_KIND
        {
            return Err(EvidencePlanArtifactError::unsupported_version(
                "unsupported evidence plan artifact version",
            ));
        }
        if artifact.repairs.is_empty()
            || artifact.repairs.len() > MAX_PLAN_ROUTES
            || artifact.options.len() > MAX_PLAN_OPTIONS
        {
            return Err(EvidencePlanArtifactError::bounds(
                "a stored collection exceeds the evidence plan artifact's declared bounds",
            ));
        }
        Ok(artifact)
    }

    /// Replay the plan: consume every embedded repair report under `limits`,
    /// consume the embedded ranking, rebuild the plan from the replayed parts and
    /// require the stored options, outcome and identity.
    ///
    /// # Errors
    /// A digest that does not match, an embedded artifact that does not replay (its
    /// own refusal is returned), or a rebuilt plan that differs from the stored one
    /// (`evidence_plan_artifact.replay_mismatch`, naming the section).
    pub fn consume(
        &self,
        limits: RepairConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<EvidencePlan, EvidencePlanArtifactError> {
        if self.version != EVIDENCE_PLAN_ARTIFACT_VERSION
            || self.kind != EVIDENCE_PLAN_ARTIFACT_KIND
        {
            return Err(EvidencePlanArtifactError::unsupported_version(
                "unsupported evidence plan artifact version",
            ));
        }
        if self.compute_digest()? != self.digest {
            return Err(EvidencePlanArtifactError::invalid(
                "an evidence plan artifact digest does not match",
            ));
        }
        let mut reports: Vec<RepairReport> = Vec::with_capacity(self.repairs.len());
        let mut universe: Option<Vec<Arc<str>>> = None;
        let mut candidates = Vec::new();
        for bytes in &self.repairs {
            let artifact = RepairReportArtifact::from_bytes(bytes)?;
            let report = artifact.consume(limits, ctx)?;
            let declared = artifact.candidates()?;
            let ids: Vec<Arc<str>> = declared
                .iter()
                .map(crate::study_candidate::DurableStudyCandidate::semantic_id)
                .collect();
            match &universe {
                None => {
                    universe = Some(ids);
                    candidates = declared;
                }
                Some(first) if *first == ids => {}
                Some(_) => {
                    return Err(EvidencePlanArtifactError::mismatch(
                        "candidates",
                        "the routes were searched over different candidate universes",
                    ));
                }
            }
            reports.push(report);
        }
        let value = match (&self.ranking, &self.value_identity) {
            (Some(bytes), Some(identity)) => {
                let expectation = ConsumeExpectation {
                    artifact_identity: Some(identity.clone()),
                    ..ConsumeExpectation::default()
                };
                let consumed = consume(bytes, &expectation)?;
                Some(PlanValue::from_ranking(&consumed.ranking, &consumed.identity)?)
            }
            (None, None) => None,
            _ => {
                return Err(EvidencePlanArtifactError::invalid(
                    "a value artifact and its identity are stored together",
                ));
            }
        };
        let unverified: Vec<UnverifiedRoute> = self
            .unverified
            .iter()
            .map(|route| UnverifiedRoute {
                route: Arc::from(route.route.as_str()),
                obligations: route.obligations.iter().map(|o| Arc::from(o.as_str())).collect(),
                reason: Arc::from(route.reason.as_str()),
            })
            .collect();
        let plan = EvidencePlan::build(
            PlanInputs { reports: &reports, candidates: &candidates, value: value.as_ref() },
            unverified,
        )?;
        let rebuilt: Vec<PlanOptionWire> = plan.options.iter().map(option_wire).collect();
        if rebuilt != self.options {
            let at = rebuilt.iter().zip(&self.options).position(|(a, b)| a != b);
            return Err(EvidencePlanArtifactError::mismatch(
                "options",
                match at {
                    Some(index) => {
                        format!("option {index} replays differently from the stored one")
                    }
                    None => format!(
                        "the replay lists {} options, the artifact {}",
                        rebuilt.len(),
                        self.options.len()
                    ),
                },
            ));
        }
        if plan.outcome.as_str() != self.outcome {
            return Err(EvidencePlanArtifactError::mismatch(
                "outcome",
                format!("stored `{}`, replayed `{}`", self.outcome, plan.outcome.as_str()),
            ));
        }
        if plan.identity() != self.plan_identity {
            return Err(EvidencePlanArtifactError::mismatch(
                "identity",
                "the replayed plan has another identity than the stored one",
            ));
        }
        Ok(plan)
    }
}
