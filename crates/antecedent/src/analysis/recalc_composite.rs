//! One shared dependency plan for an actual retained native response and attested callbacks.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::native_claims::{NativeDecisionSource, NativeResponseClaim, NativeResponseContext};
use super::recalc_external::{
    ExternalCallbackError, ExternalCallbackRequest, ExternalCallbackSession, ExternalMeanProvider,
};
use super::recalc_receipt::{Counter, RecalcReceipt, RecalcRunError, ReceiptRecorder};
use super::recalc_static::{
    StaticResponseRequest, StaticResponseSession, execute_static_response_with_receipt,
};
use antecedent_core::recalc::{
    Boundary, Branch, RecalcCapabilities, RecalcPlan, RequestSupport, RetargetSupport, Stage,
    StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{ExecutionContext, ExternalTrustState, ProgramBinding};
use antecedent_design::composition_boundary::{
    DecisionInput, NativeExecutionRecord, SupportPolicy, SupportedDecision, TrustEvidence,
    TrustRequirement, evaluate_with_support,
};
use antecedent_design::decision_contract::{DecisionContract, SourceRepresentation};
use antecedent_design::decision_eval::MeanSource;
use antecedent_io::distribution_artifact::DistributionCalibration;
use std::collections::BTreeMap;
use std::sync::Arc;
/// Complete causal native request, bounded callback branches, and original terminal contract.
#[derive(Clone, Debug)]
pub struct CompositeRequest {
    /// Executing original native response request.
    pub native: StaticResponseRequest,
    /// Full native scientific coordinates bound to the actual producer's graph/program identities.
    pub native_program: ProgramBinding,
    /// Ordered callback branches; their full scientific coordinates must match the native program.
    pub externals: Vec<ExternalCallbackRequest>,
    /// Original affine expectation decision contract.
    pub decision_contract: DecisionContract,
    /// Original per-action support policy.
    pub support_policy: SupportPolicy,
    /// Explicit ordered original input preference; every actual input appears exactly once.
    pub input_order: Vec<String>,
}
/// Typed native, external, planner or original terminal refusal.
#[derive(Debug)]
pub enum CompositeError {
    /// Bounded preflight refusal.
    Request(&'static str),
    /// Original native refusal.
    Native(RecalcRunError),
    /// Original callback refusal, including actual failed attempt evidence.
    External {
        /// Actual callback branch that refused or failed.
        branch: Branch,
        /// Original typed callback error and counted attempt report.
        cause: ExternalCallbackError,
    },
    /// Original scientific binding or decision refusal.
    Binding(Box<antecedent_core::ExternalRefusal>),
    /// Original shared dependency refusal.
    Refused(Box<RecalcPlan>),
    /// Actual receipt construction failure.
    Receipt(String),
}
impl CompositeError {
    /// Actual registered code for a bounded combined-request refusal.
    #[must_use]
    pub fn request_reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Request("composite_recalc.request_unsupported") => {
                Some(antecedent_core::reason_code!("route_not_supported"))
            }
            Self::Request("composite_recalc.native_state_mismatch") => {
                Some(antecedent_core::reason_code!("invalid_argument"))
            }
            Self::Request(detail) => Some(ExternalCallbackError::Request(detail).refusal_code()),
            _ => None,
        }
    }
}
/// Terminal original decision and the one actual shared workflow receipt.
pub struct CompositeOutcome {
    /// Original shared dependency plan.
    pub plan: RecalcPlan,
    /// All actual native, callback and terminal work in this execution.
    pub receipt: RecalcReceipt,
    /// Original supported decision retaining selected input/trust provenance.
    pub decision: Arc<SupportedDecision>,
    /// Exact old identities used to seal this receipt independently.
    pub previous: StageIdentities,
    /// Exact requested identities used to seal this receipt independently.
    pub requested: StageIdentities,
    /// Actual native/provider boundary capabilities.
    pub capabilities: RecalcCapabilities,
}
pub(crate) struct CompositePublication {
    pub(crate) receipt: RecalcReceipt,
    pub(crate) previous: StageIdentities,
    pub(crate) requested: StageIdentities,
    pub(crate) capabilities: RecalcCapabilities,
    pub(crate) input_order: Vec<String>,
    pub(crate) branches: Vec<Branch>,
}
/// Transactional scientific state with separately retained external side-effect attempts.
pub struct CompositeSession {
    native: StaticResponseSession,
    previous: StageIdentities,
    callbacks: BTreeMap<Branch, ExternalCallbackSession>,
    decision: Option<Arc<SupportedDecision>>,
    publication: Option<CompositePublication>,
}
impl CompositeSession {
    /// Begin from an actually executed native response; flags cannot supply the producing state.
    /// # Errors
    /// The supplied native request differs from held checked native state.
    pub fn from_native(
        native: StaticResponseSession,
        request: &StaticResponseRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CompositeError> {
        let native_plan = native.plan(request, ctx);
        if native.result().is_none()
            || native_plan
                .entries()
                .iter()
                .any(|entry| !matches!(entry.status, StageStatus::Reused { .. }))
        {
            return Err(CompositeError::Request("composite_recalc.native_state_mismatch"));
        }
        Ok(Self {
            native,
            previous: request.identities(),
            callbacks: BTreeMap::new(),
            decision: None,
            publication: None,
        })
    }
    /// Actual retained native result and producing receipt.
    #[must_use]
    pub fn native(&self) -> &StaticResponseSession {
        &self.native
    }
    /// Last successfully published original terminal decision, without executing any branch.
    #[must_use]
    pub fn decision(&self) -> Option<&SupportedDecision> {
        self.decision.as_deref()
    }
    /// Actual successfully bound callback output, without invoking its provider.
    #[must_use]
    pub fn callback(&self, branch: Branch) -> Option<&ExternalCallbackSession> {
        self.callbacks.get(&branch)
    }
    pub(crate) fn publication(&self) -> Option<&CompositePublication> {
        self.publication.as_ref()
    }
    fn inputs(
        &self,
        request: &CompositeRequest,
    ) -> Result<(StageIdentities, RecalcCapabilities), CompositeError> {
        if !decision_bounded(&request.decision_contract) {
            return Err(CompositeError::Request("composite_recalc.request_unsupported"));
        }
        request.native_program.validate().map_err(|e| CompositeError::Binding(Box::new(e)))?;
        if request.externals.len() != 2
            || request.externals[0].branch == request.externals[1].branch
            || request.decision_contract.actions.len() > 64
            || request.decision_contract.constraints.len() > 128
        {
            return Err(CompositeError::Request("composite_recalc.request_unsupported"));
        }
        let expected_inputs: std::collections::BTreeSet<_> = std::iter::once("native".to_owned())
            .chain(request.externals.iter().map(|e| format!("external.{}", e.branch.index())))
            .collect();
        if request.input_order.len() != 3
            || request.input_order.iter().any(|id| id.len() > 32)
            || request.input_order.iter().cloned().collect::<std::collections::BTreeSet<_>>()
                != expected_inputs
        {
            return Err(CompositeError::Request("composite_recalc.request_unsupported"));
        }
        let identity = request
            .decision_contract
            .identity()
            .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?;
        let requirement = request
            .decision_contract
            .source_requirement()
            .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?
            .ok_or(CompositeError::Request("external_recalc.quantity_unsupported"))?;
        if requirement.check(&[SourceRepresentation::Mean]).is_err() {
            return Err(CompositeError::Request("external_recalc.quantity_unsupported"));
        }
        let native_ctx = self
            .native
            .producing_context()
            .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?;
        let native_plan = self.native.plan(&request.native, native_ctx);
        if matches!(native_plan.status(Stage::Identification), Some(StageStatus::Reused { .. })) {
            checked_native(&self.native, &request.native_program)?;
        }
        let mut ids = request.native.identities();
        for external in &request.externals {
            external
                .validate()
                .map_err(|cause| CompositeError::External { branch: external.branch, cause })?;
            if external.program != request.native_program {
                return Err(CompositeError::Request("external_recalc.contract_mismatch"));
            }
            let empty = ExternalCallbackSession::new();
            let session = self.callbacks.get(&external.branch).unwrap_or(&empty);
            let (_, own, _) = session.plan_inputs(external, true);
            for stage in
                [Stage::ExternalStudy(external.branch), Stage::ProviderRequest(external.branch)]
            {
                ids.set(stage, own.own(stage));
            }
        }
        let utility = ids.own(Stage::Utility);
        ids.set(
            Stage::Utility,
            StageIdentity::of(
                "composite.utility.v1",
                &[
                    utility.as_bytes(),
                    identity.as_bytes(),
                    format!("{:?}", request.support_policy).as_bytes(),
                    format!("{:?}", request.input_order).as_bytes(),
                ],
            ),
        );
        ids.set(Stage::Decision, StageIdentity::of("composite.original_support_terminal.v1", &[]));
        Ok((
            ids,
            RecalcCapabilities {
                boundary: Boundary::InProcess,
                request: RequestSupport::OnGrid,
                retarget: RetargetSupport::NotDeclared,
            },
        ))
    }
    /// Plan the one shared workflow without invoking a provider or terminal engine.
    /// # Errors
    /// Original bounded scientific request refusal.
    pub fn plan(&self, request: &CompositeRequest) -> Result<RecalcPlan, CompositeError> {
        let (ids, cap) = self.inputs(request)?;
        Ok(RecalcPlan::plan(&self.previous, &ids, &cap))
    }
    /// Execute native refreshes, only changed callback branches, and the original terminal engine.
    /// # Errors
    /// Original native/callback/binding/decision refusal; scientific publication is transactional.
    pub fn execute(
        &mut self,
        request: &CompositeRequest,
        providers: &mut BTreeMap<Branch, &mut dyn ExternalMeanProvider>,
        ctx: &ExecutionContext,
    ) -> Result<CompositeOutcome, CompositeError> {
        let (ids, cap) = self.inputs(request)?;
        let previous = self.previous.clone();
        let plan = RecalcPlan::plan(&previous, &ids, &cap);
        if !plan.is_executable() {
            return Err(CompositeError::Refused(Box::new(plan)));
        }
        let mut native = self.native.clone();
        let mut callbacks = self.callbacks.clone();
        let attempt = (|| -> Result<(RecalcReceipt, Arc<SupportedDecision>), CompositeError> {
            let mut recorder = ReceiptRecorder::new();
            let native_out =
                execute_static_response_with_receipt(&mut native, &request.native, ctx)
                    .map_err(CompositeError::Native)?;
            add_receipt(&mut recorder, &native_out.receipt);
            // Native program identities are checked after any real refresh, before foreign work.
            let native_claim = checked_native(&native, &request.native_program)?;
            for external in &request.externals {
                let session = callbacks.entry(external.branch).or_default();
                if matches!(
                    plan.status(Stage::ProviderRequest(external.branch)),
                    Some(StageStatus::Recomputed { .. })
                ) {
                    session.force_recompute(external.branch);
                }
                let provider = providers
                    .get_mut(&external.branch)
                    .map(|provider| &mut **provider as &mut dyn ExternalMeanProvider);
                match session.execute(external, provider, ctx) {
                    Ok(out) => add_receipt(&mut recorder, &out.receipt),
                    Err(error) => {
                        return Err(CompositeError::External {
                            branch: external.branch,
                            cause: error,
                        });
                    }
                }
            }
            let decision =
                if matches!(plan.status(Stage::Decision), Some(StageStatus::Recomputed { .. })) {
                    let inputs = decision_inputs(&native, &native_claim, &callbacks, request)?;
                    let decision = evaluate_with_support(
                        &request.decision_contract,
                        &inputs,
                        &request.support_policy,
                    )
                    .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?;
                    recorder.record(Counter::Decision, 1);
                    Arc::new(decision)
                } else {
                    Arc::clone(
                        self.decision
                            .as_ref()
                            .ok_or(CompositeError::Request("external_recalc.output_unavailable"))?,
                    )
                };
            let receipt =
                recorder.finish(&plan).map_err(|e| CompositeError::Receipt(e.to_string()))?;
            Ok((receipt, decision))
        })();
        let (receipt, decision) = match attempt {
            Ok(result) => result,
            Err(error) => {
                for (branch, attempted) in &callbacks {
                    self.callbacks.entry(*branch).or_default().absorb_attempts(attempted);
                }
                return Err(error);
            }
        };
        self.native = native;
        self.callbacks = callbacks;
        self.previous = ids.clone();
        self.decision = Some(Arc::clone(&decision));
        self.publication = Some(CompositePublication {
            receipt: receipt.clone(),
            previous: previous.clone(),
            requested: ids.clone(),
            capabilities: cap,
            input_order: request.input_order.clone(),
            branches: request.externals.iter().map(|e| e.branch).collect(),
        });
        Ok(CompositeOutcome {
            plan,
            receipt,
            decision,
            previous,
            requested: ids,
            capabilities: cap,
        })
    }
}
fn checked_native(
    session: &StaticResponseSession,
    program: &ProgramBinding,
) -> Result<NativeResponseClaim, CompositeError> {
    let result = session
        .result()
        .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?;
    let contract = result
        .executed_contract
        .as_ref()
        .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?;
    if program.graph_id != format!("graph:{}", contract.identities.identification)
        || program.contract_id
            != antecedent_io::external_binding_wire::native_contract_identity(
                &program.graph_id,
                result
                    .response
                    .as_ref()
                    .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?
                    .identification_status
                    .as_str(),
                None,
            )
            .map_err(|e| CompositeError::Receipt(format!("{e:?}")))?
    {
        return Err(CompositeError::Request("composite_recalc.native_state_mismatch"));
    }
    let context = NativeResponseContext {
        program: program.clone(),
        snapshot_id: contract.identities.data_snapshot.to_hex(),
        rng_id: format!(
            "native:seed:{}",
            session
                .producing_context()
                .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?
                .rng
                .master_seed()
        ),
        calibration: DistributionCalibration::PointOnly,
    };
    NativeResponseClaim::from_response(
        result
            .response
            .as_ref()
            .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?,
        session
            .schema()
            .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?,
        &context,
    )
    .map_err(|e| CompositeError::Binding(Box::new(e)))
}
fn decision_inputs(
    native: &StaticResponseSession,
    claim: &NativeResponseClaim,
    callbacks: &BTreeMap<Branch, ExternalCallbackSession>,
    request: &CompositeRequest,
) -> Result<Vec<DecisionInput>, CompositeError> {
    let requirement = request
        .decision_contract
        .source_requirement()
        .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?
        .ok_or(CompositeError::Request("external_recalc.quantity_unsupported"))?;
    let native_input =
        claim.decision_source(&requirement).map_err(|e| CompositeError::Binding(Box::new(e)))?;
    let NativeDecisionSource::Mean(source) = native_input.source else {
        return Err(CompositeError::Request("external_recalc.quantity_unsupported"));
    };
    let proof = TrustEvidence::NativeExecution(NativeExecutionRecord {
        execution_id: native
            .producing_receipt()
            .ok_or(CompositeError::Request("composite_recalc.native_state_mismatch"))?
            .identity()
            .to_hex(),
        provider_id: source.provider_id.clone(),
        snapshot_id: source.snapshot_id.clone(),
    });
    let mut inputs = vec![
        DecisionInput::from_mean_source(
            "native",
            source,
            &native_input.point_status,
            &proof,
            TrustRequirement::Unrestricted,
        )
        .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?,
    ];
    for external in &request.externals {
        let issued = callbacks[&external.branch]
            .issued_claim()
            .ok_or(CompositeError::Request("external_recalc.output_unavailable"))?;
        let source = MeanSource {
            coordinates: issued.quantities().to_vec(),
            means: issued
                .values()
                .ok_or(CompositeError::Request("external_recalc.invalid_output"))?
                .to_vec(),
            provider_id: external.descriptor.identity.provider_id.clone(),
            snapshot_id: external.descriptor.identity.snapshot_id.clone(),
            causal_contract_id: external.program.contract_id.clone(),
            rng_id: format!("callback.seed:{}", external.seed),
        };
        let evidence = TrustEvidence::External(ExternalTrustState::ExternallyAttested {
            attestor: external.descriptor.identity.provider_id.clone(),
        });
        inputs.push(
            DecisionInput::from_mean_source(
                &format!("external.{}", external.branch.index()),
                source,
                issued
                    .point_status()
                    .ok_or(CompositeError::Request("external_recalc.invalid_output"))?,
                &evidence,
                TrustRequirement::Unrestricted,
            )
            .map_err(|e| CompositeError::Binding(Box::new(e.to_refusal())))?,
        );
    }
    inputs.sort_by_key(|input| request.input_order.iter().position(|id| id == input.id()));
    Ok(inputs)
}
fn add_receipt(recorder: &mut ReceiptRecorder, receipt: &RecalcReceipt) {
    for entry in receipt.entries() {
        let counter = if let Stage::ProviderRequest(branch) = entry.stage {
            Some(Counter::ExternalInvocation(branch))
        } else {
            None
        };
        let c = entry.counts;
        for (counter, count) in [
            (Counter::Identification, c.identifications),
            (Counter::FoldFit, c.fold_fits),
            (Counter::ModelFit, c.model_fits),
            (Counter::ScoreComputation, c.score_computations),
            (Counter::Reweight, c.reweights),
            (Counter::Decision, c.decisions),
            (Counter::FactorBuild, c.factor_builds),
            (Counter::ProgramCompilation, c.program_compilations),
            (Counter::ProviderBinding, c.provider_bindings),
            (Counter::FactorEvaluation, c.factor_evaluations),
            (Counter::Integration, c.integrations),
            (Counter::ProviderCall, c.provider_calls),
            (Counter::LawSummary, c.law_summaries),
            (Counter::PosteriorDraw, c.posterior_draws),
        ] {
            if count > 0 {
                recorder.record(counter, count);
            }
        }
        if let Some(counter) = counter {
            recorder.record(counter, c.external_invocations);
        }
    }
}

fn decision_bounded(contract: &DecisionContract) -> bool {
    use antecedent_design::decision_contract::UtilityExpr;
    if contract.actions.len() > 64
        || !contract.constraints.is_empty()
        || contract.utility_units.len() > 256
        || contract.target_population.len() > 256
    {
        return false;
    }
    for action in &contract.actions {
        if action.id.len() > 256
            || action.inputs.len() > 64
            || action.inputs.iter().any(|q| !super::recalc_external::quantity_bounded(q))
        {
            return false;
        }
        let mut stack = vec![(&action.utility, 0_usize)];
        let mut nodes = 0;
        while let Some((expr, depth)) = stack.pop() {
            nodes += 1;
            if depth >= 64 || nodes > 2048 {
                return false;
            }
            match expr {
                UtilityExpr::Const(value) => {
                    if !value.is_finite() {
                        return false;
                    }
                }
                UtilityExpr::Input(_) => {}
                UtilityExpr::Neg(inner) => stack.push((inner, depth + 1)),
                UtilityExpr::Add(a, b)
                | UtilityExpr::Sub(a, b)
                | UtilityExpr::Mul(a, b)
                | UtilityExpr::Min(a, b)
                | UtilityExpr::Max(a, b) => {
                    stack.push((a, depth + 1));
                    stack.push((b, depth + 1));
                }
            }
        }
    }
    true
}
