//! Internal checked joint transport recalculation inspection.
//! No calibrated public posterior route is issued by this candidate adapter.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::recalc_bayesian::{BayesianLaw, PosteriorSummarySpec};
use super::recalc_receipt::{DecisionValue, RecalcReceipt, UtilitySpec};
use antecedent_core::{TransportQuery, recalc::RecalcPlan};
use antecedent_estimate::joint_bayesian_transport::{
    JointTransportFit, JointTransportModel, JointTransportOptions, SourceData, TargetData,
};
use antecedent_graph::SelectionDiagram;

/// Raw structural and numerical premises, never a caller-authored transport certificate.
#[derive(Clone, Debug)]
pub struct JointBayesianRequest {
    /// Original selection diagram used by the existing structural identifier.
    pub diagram: SelectionDiagram,
    /// Original target/source causal response query and checked evidence catalog.
    pub query: TransportQuery,
    /// Declared joint source-effect sharing and Gaussian priors.
    pub model: JointTransportModel,
    /// Actual source units and identity of the likelihood-consumed observations.
    pub sources: Vec<SourceData>,
    /// Actual target covariates and their identity; target outcomes are never accepted.
    pub target: TargetData,
    /// Actual original engine draw count and stream seed.
    pub options: JointTransportOptions,
    /// Inspection projection over native emitted target-effect draws.
    pub summary: PosteriorSummarySpec,
    /// Native expected effect net-benefit rule, with calibration still unmeasured.
    pub utility: UtilitySpec,
}
/// Inspection-only candidate outcome. It is not a public licensed interval result.
#[derive(Debug)]
pub struct JointBayesianRecalcOutcome {
    /// Actual shared invalidation plan.
    pub plan: RecalcPlan,
    /// Actual checked identification, posterior solves and aligned draw work.
    pub receipt: RecalcReceipt,
    /// Original Gaussian posterior inspection, never a calibrated public interval.
    pub law: BayesianLaw,
    /// Candidate expected effect decision projection.
    pub decision: DecisionValue,
}
struct JointBayesianLive {
    identification: std::sync::Arc<antecedent_identify::TransportIdentification>,
    fit: std::sync::Arc<JointTransportFit>,
    artifact:
        std::sync::Arc<antecedent_io::joint_bayesian_transport_artifact::JointBayesianArtifactWire>,
    law: BayesianLaw,
    decision: DecisionValue,
}
/// Internal retention boundary of a genuine raw-query checked transport fit.
pub struct JointBayesianSession {
    previous: antecedent_core::recalc::StageIdentities,
    live: Option<JointBayesianLive>,
    boundary: antecedent_core::recalc::Boundary,
}
fn literal(label: &str, text: &str) -> antecedent_core::recalc::StageIdentity {
    antecedent_core::recalc::StageIdentity::of(label, &[text.as_bytes()])
}
impl JointBayesianRequest {
    fn validate(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), super::recalc_receipt::RecalcRunError> {
        use super::recalc_receipt::RecalcRunError;
        self.summary.validate()?;
        self.utility.validate()?;
        if ctx.cancellation.is_cancelled() {
            return Err(RecalcRunError::Request("recalc.cancelled"));
        }
        if self.diagram.causal_graph().has_bidirected() {
            return Err(RecalcRunError::Request("recalc.joint_bayesian_graph_unsupported"));
        }
        self.metadata_guard()?;
        if self.sources.is_empty()
            || self.sources.len() > 16
            || self.model.features.len() > 32
            || !(2..=100_000).contains(&self.options.draws)
            || self.target.rows == 0
            || self.target.rows > 100_000
            || self.target.covariates.len() != self.model.features.len()
            || self.target.covariates.iter().any(|column| column.len() != self.target.rows)
            || self.sources.iter().any(|source| {
                source.treatment.is_empty()
                    || source.treatment.len() > 100_000
                    || source.outcome.len() != source.treatment.len()
                    || source.covariates.len() != self.model.features.len()
                    || source.covariates.iter().any(|column| column.len() != source.treatment.len())
            })
        {
            return Err(RecalcRunError::Request("recalc.joint_bayesian_invalid_data"));
        }
        let raw = self
            .sources
            .iter()
            .fold(0usize, |values, source| {
                values.saturating_add(
                    source.treatment.len().saturating_mul(2 + self.model.features.len()),
                )
            })
            .saturating_add(self.target.rows.saturating_mul(self.model.features.len()));
        let p = antecedent_estimate::joint_bayesian_transport::JOINT_TRANSPORT_MAX_PARAMETERS;
        let values = raw
            .saturating_mul(p)
            .saturating_add(self.options.draws.saturating_mul(p + 17))
            .saturating_add(p.saturating_mul(p).saturating_mul(32));
        if raw > 1_000_000 || values > 16_000_000 {
            return Err(RecalcRunError::Request("recalc.bayesian_workspace_limit"));
        }
        Ok(())
    }
    fn metadata_guard(&self) -> Result<(), super::recalc_receipt::RecalcRunError> {
        use antecedent_estimate::joint_bayesian_transport::{
            JOINT_TRANSPORT_MAX_PARAMETERS, PriorProvenance,
        };
        let identity_bytes =
            |identity: &antecedent_estimate::joint_bayesian_transport::DataIdentity| {
                identity
                    .datum_ids
                    .iter()
                    .fold(identity.snapshot_digest.len(), |size, id| size.saturating_add(id.len()))
            };
        let mut bytes = identity_bytes(&self.target.identity);
        for source in &self.sources {
            bytes = bytes
                .saturating_add(source.id.len())
                .saturating_add(identity_bytes(&source.identity));
        }
        for prior in [&self.model.priors.invariant, &self.model.priors.varying] {
            if prior.mean.len() > JOINT_TRANSPORT_MAX_PARAMETERS
                || prior.covariance.len() > JOINT_TRANSPORT_MAX_PARAMETERS.pow(2)
            {
                return Err(super::recalc_receipt::RecalcRunError::Request(
                    "recalc.bayesian_workspace_limit",
                ));
            }
            if let PriorProvenance::Bank { bank_id, consumed } = &prior.provenance {
                if consumed.len() > 16 {
                    return Err(super::recalc_receipt::RecalcRunError::Request(
                        "recalc.bayesian_workspace_limit",
                    ));
                }
                bytes = bytes.saturating_add(bank_id.len());
                for identity in consumed {
                    bytes = bytes.saturating_add(identity_bytes(identity));
                }
            }
        }
        if bytes > 1_000_000 {
            return Err(super::recalc_receipt::RecalcRunError::Request(
                "recalc.bayesian_workspace_limit",
            ));
        }
        Ok(())
    }
    /// Dependency identities include every source unit, prior and consumed-evidence declaration.
    #[must_use]
    pub fn identities(&self) -> antecedent_core::recalc::StageIdentities {
        use antecedent_core::recalc::{Stage, StageIdentities};
        let mut ids = StageIdentities::new();
        ids.set(Stage::Graph, literal("joint.graph.v1", &format!("{:?}", self.diagram)));
        ids.set(Stage::Query, literal("joint.query.v1", &format!("{:?}", self.query)));
        ids.set(Stage::Regime, literal("joint.regime.v1", "randomized.binary.known_noise"));
        ids.set(
            Stage::Evidence,
            literal("joint.evidence.v1", &format!("{:?}", self.query.catalog)),
        );
        ids.set(
            Stage::SourcePopulation,
            literal(
                "joint.source.v1",
                &format!("{:?}", self.sources.iter().map(|source| &source.id).collect::<Vec<_>>()),
            ),
        );
        ids.set(Stage::TargetPopulation, literal("joint.target.v1", "declared.target.covariates"));
        ids.set(
            Stage::DataSnapshot,
            literal("joint.units.v1", &format!("{:?}:{:?}", self.sources, self.target)),
        );
        ids.set(Stage::RowDesign, literal("joint.rows.v1", "identity_bound_source_units"));
        ids.set(Stage::TreatmentGrid, literal("joint.actions.v1", "binary.0_1"));
        ids.set(
            Stage::LearnerFoldsRng,
            literal("joint.inference.v1", &format!("{:?}:{:?}", self.model, self.options)),
        );
        ids.set(
            Stage::Identification,
            literal("joint.checked.v1", "original.transport.identifier"),
        );
        ids.set(Stage::ScoreArtifact, literal("joint.posterior.v1", "native.aligned.candidate"));
        ids.set(Stage::Law, literal("joint.projection.v1", &format!("{:?}", self.summary)));
        ids.set(Stage::Utility, super::recalc_cell::utility_identity(self.utility));
        ids.set(Stage::Decision, literal("joint.decision.v1", "candidate.expected_net_benefit"));
        ids
    }
}
impl Default for JointBayesianSession {
    fn default() -> Self {
        Self::new()
    }
}
impl JointBayesianSession {
    /// Empty internal candidate session, with no public interval authority.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: antecedent_core::recalc::StageIdentities::new(),
            live: None,
            boundary: antecedent_core::recalc::Boundary::InProcess,
        }
    }
    /// Actual native retained candidate fit for inspection only.
    #[must_use]
    pub fn fit(&self) -> Option<&JointTransportFit> {
        self.live.as_ref().map(|live| live.fit.as_ref())
    }
    /// Last successful dependency identities.
    #[must_use]
    pub const fn identities(&self) -> &antecedent_core::recalc::StageIdentities {
        &self.previous
    }
    /// Existing sealed candidate artifact; no repeated fitting or sampling at export.
    /// # Errors
    /// No retained candidate or refused original artifact encoding.
    pub fn export_result(&self) -> Result<Vec<u8>, super::recalc_receipt::RecalcRunError> {
        let live = self.live.as_ref().ok_or(super::recalc_receipt::RecalcRunError::NoLiveState(
            antecedent_core::recalc::Stage::ScoreArtifact,
        ))?;
        live.artifact.export().map_err(|error| crate::CausalError::Serialization(error).into())
    }
}
/// Execute a raw-query checked candidate model or inspect its retained target-effect draws.
/// This internal route publishes no calibrated posterior or confidence interval.
/// # Errors
/// Original identification/likelihood/prior ownership refusals, bounds, or receipt errors.
pub fn execute_joint_bayesian_with_receipt(
    session: &mut JointBayesianSession,
    request: &JointBayesianRequest,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointBayesianRecalcOutcome, super::recalc_receipt::RecalcRunError> {
    use super::recalc_receipt::{Counter, LawValue, RecalcRunError, ReceiptRecorder, decide};
    use antecedent_core::recalc::{
        Boundary, RecalcCapabilities, RequestSupport, RetargetSupport, Stage, StageStatus,
    };
    use antecedent_expr::execution_counts::{StaticWork, count_static_work, note_static_work};
    use antecedent_identify::execution_counts::count_checked_identifications;
    use antecedent_prob::fit_counts::count_bayesian_work;
    use std::sync::Arc;
    request.validate(ctx)?;
    let ids = request.identities();
    let plan = RecalcPlan::plan(
        &session.previous,
        &ids,
        &RecalcCapabilities {
            retarget: RetargetSupport::NotDeclared,
            request: RequestSupport::OnGrid,
            boundary: session.boundary,
        },
    );
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    let changed = |stage| matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }));
    let mut recorder = ReceiptRecorder::new();
    let old = session.live.as_ref();
    let (identification, fit, artifact) = if changed(Stage::ScoreArtifact) {
        let raw = request
            .sources
            .iter()
            .fold(0usize, |values, source| {
                values.saturating_add(
                    source.treatment.len().saturating_mul(2 + request.model.features.len()),
                )
            })
            .saturating_add(request.target.rows.saturating_mul(request.model.features.len()));
        let p = antecedent_estimate::joint_bayesian_transport::JOINT_TRANSPORT_MAX_PARAMETERS;
        let bytes = u64::try_from(
            raw.saturating_mul(p)
                .saturating_add(request.options.draws.saturating_mul(p + 17))
                .saturating_add(p.saturating_mul(p).saturating_mul(32)),
        )
        .unwrap_or(u64::MAX)
        .saturating_mul(16);
        if ctx
            .memory
            .hard_limit_bytes
            .into_iter()
            .chain(ctx.memory.soft_limit_bytes)
            .any(|limit| bytes > limit)
        {
            return Err(RecalcRunError::Request("recalc.memory_budget_exceeded"));
        }
        let (identification, checks) = count_checked_identifications(|| {
            if changed(Stage::Identification) {
                antecedent_identify::TransportIdentifier
                    .identify(&request.diagram, &request.query)
                    .map(Arc::new)
                    .map_err(crate::CausalError::from)
                    .map_err(RecalcRunError::from)
            } else {
                old.map(|live| Arc::clone(&live.identification))
                    .ok_or(RecalcRunError::NoLiveState(Stage::Identification))
            }
        });
        recorder.record(Counter::Identification, checks);
        let identification = identification?;
        let (produced, work) = count_bayesian_work(|| {
            antecedent_io::joint_bayesian_transport_artifact::JointBayesianArtifactWire::build(
                &identification,
                &request.model,
                &request.sources,
                &request.target,
                request.options,
                ctx,
            )
        });
        recorder.record(Counter::ModelFit, work.model_fits);
        recorder.record(Counter::PosteriorDraw, work.posterior_draws);
        let (artifact, fit) = produced.map_err(crate::CausalError::Serialization)?;
        (identification, Arc::new(fit), Arc::new(artifact))
    } else {
        let live = old.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        (Arc::clone(&live.identification), Arc::clone(&live.fit), Arc::clone(&live.artifact))
    };
    let law = if changed(Stage::Law) {
        let (law, work) = count_static_work(|| {
            let width = fit.draws.names.len();
            let column = fit
                .draws
                .names
                .iter()
                .position(|name| name == "effect.target")
                .ok_or(RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?;
            let values =
                fit.draws.values.chunks_exact(width).map(|row| row[column]).collect::<Vec<_>>();
            let law = super::recalc_bayesian::summarize_issued_values(&values, request.summary)?;
            note_static_work(StaticWork::LawSummary);
            Ok::<_, RecalcRunError>(law)
        });
        recorder.record(Counter::LawSummary, work.law_summaries);
        law?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?.law
    };
    let decision = if changed(Stage::Decision) {
        decide(LawValue { ate: law.mean, std_error: f64::NAN }, request.utility, &mut recorder)?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan)?;
    session.live = Some(JointBayesianLive { identification, fit, artifact, law, decision });
    session.previous = ids;
    session.boundary = Boundary::InProcess;
    Ok(JointBayesianRecalcOutcome { plan, receipt, law, decision })
}
