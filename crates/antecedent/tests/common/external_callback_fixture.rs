//! Independent foreign mean callbacks, actual invocation receipts and portable replay.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_external::*;
use antecedent_core::recalc::Branch;
use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalProgramClaim,
    ExternalResponse, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProgramBinding,
    ProviderObjectIdentity, SupportStatus,
};

pub(crate) fn request(branch: u8, policy: CallbackPolicy) -> ExternalCallbackRequest {
    let program = ProgramBinding {
        graph_id: "checked-graph".into(),
        contract_id: "checked-contract".into(),
        treatment_id: "t".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![-1.0, 0.0, 1.0],
        dose_units: "mg".into(),
        outcome_units: "kg".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    };
    ExternalCallbackRequest {
        branch: Branch::new(branch).unwrap(),
        claim: ExternalProgramClaim::declared_by(&program),
        contract: CheckedCausalContract {
            graph_id: program.graph_id.clone(),
            identification: IdentificationStatus::NonparametricallyIdentified,
            estimand: program.expected_quantities(),
            accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
            required_evidence_ids: vec!["foreign-data".into()],
            required_assumption_ids: vec!["foreign-model".into()],
            equivalences: vec![],
        },
        descriptor: ProviderDescriptor {
            identity: ProviderObjectIdentity {
                provider_id: "mean-provider".into(),
                object_id: "curve".into(),
                version_id: "v1".into(),
                snapshot_id: "source1".into(),
                request_id: program.identity(),
            },
            environment_id: "runtime-v1".into(),
            policy,
            idempotency_supported: false,
        },
        program,
        columns: vec![("z".into(), vec![-1.0, 1.0])],
        model_parameters: vec![("intercept".into(), 3.0), ("gain".into(), 0.5)],
        seed: 41,
        idempotency_key: None,
    }
}
pub(crate) struct Provider {
    pub(crate) descriptor: ProviderDescriptor,
    pub(crate) calls: u64,
    pub(crate) fail: bool,
    pub(crate) offset: f64,
    pub(crate) wrong_quantity: bool,
    pub(crate) cancel: bool,
    pub(crate) support: Vec<SupportStatus>,
}
impl Provider {
    pub(crate) fn new(request: &ExternalCallbackRequest) -> Self {
        Self {
            descriptor: request.descriptor.clone(),
            calls: 0,
            fail: false,
            offset: 0.0,
            wrong_quantity: false,
            cancel: false,
            support: vec![SupportStatus::Supported; 3],
        }
    }
}
impl ExternalMeanProvider for Provider {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }
    fn invoke(
        &mut self,
        request: &ExternalCallbackRequest,
        ctx: &CallbackContext,
    ) -> Result<ExternalResponse, CallbackFailure> {
        self.calls += 1;
        assert_eq!(ctx.rng.master_seed(), request.seed);
        if self.fail {
            return Err(CallbackFailure {
                message: "foreign failure after entering callback".into(),
            });
        }
        if self.cancel {
            ctx.cancellation.cancel();
        }
        let z_mean = request.columns[0].1.iter().sum::<f64>() / 2.0;
        let jitter = if self.descriptor.policy == CallbackPolicy::Seeded {
            f64::from(u32::try_from(request.seed % 11).unwrap()) * 0.01
        } else {
            0.0
        };
        let values = request
            .program
            .dose_grid
            .iter()
            .map(|dose| {
                request.model_parameters[0].1
                    + request.model_parameters[1].1 * dose
                    + 2.0 * z_mean
                    + jitter
                    + self.offset
            })
            .collect();
        let mut quantities = request.program.expected_quantities();
        if self.wrong_quantity {
            quantities[1].units = "wrong-unit".into();
        }
        Ok(ExternalResponse {
            header: ExternalResultHeader {
                object: ExternalScientificObject::Law(LawProviderContract {
                    identity: self.descriptor.identity.clone(),
                    quantities: quantities.clone(),
                    meaning: DistributionMeaning::InterventionalPredictive,
                    capabilities: vec![ExternalCapability::Mean, ExternalCapability::Intervention],
                }),
                graph_id: request.program.graph_id.clone(),
                quantities,
                evidence_ids: vec!["foreign-data".into()],
                assumption_ids: vec!["foreign-model".into()],
                trust: ExternalTrustState::ExternallyAttested { attestor: "foreign-team".into() },
            },
            values,
            uncertainty: ExternalUncertaintyMeaning::None,
            point_support: Some(self.support.clone()),
        })
    }
}
