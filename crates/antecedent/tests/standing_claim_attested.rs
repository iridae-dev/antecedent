//! Caller-attested validator evidence on a sealed claim: it is carried as attested, never
//! re-verifiable, and it changes no other standing facet of the claim.

#[path = "fixtures_common/mod.rs"]
mod common;

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_contract::ClaimEnvelope;
use antecedent_core::ExecutionContext;
use antecedent_validate::{
    CustomEffectValidator, RefutationProblem, RefutationReport, ValidationError,
};
use common::fixtures::confounded_scm;

const NAME: &str = "caller.attested";

struct FixedValidator {
    pass: bool,
}

impl CustomEffectValidator for FixedValidator {
    fn name(&self) -> &'static str {
        NAME
    }

    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        Ok(RefutationReport::new(
            NAME,
            problem.original.ate,
            problem.original.ate,
            if self.pass { 1.0 } else { 0.0 },
            true,
            self.pass,
            (!self.pass).then(|| Arc::from("forced fail")),
            0,
        ))
    }
}

fn sealed(validators: Option<bool>) -> ClaimEnvelope {
    let ctx = ExecutionContext::for_tests(43);
    let (data, dag, query) = confounded_scm(400, 43);
    let mut study = Study::tabular(data.clone()).graph(dag).query(query).refute(RefuteSuite::Cheap);
    if let Some(pass) = validators {
        let validator: Arc<dyn CustomEffectValidator> = Arc::new(FixedValidator { pass });
        study = study.custom_validators(vec![validator]);
    }
    let prepared = study.build().unwrap().prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    result.claim(&contract, &ctx).unwrap()
}

#[test]
fn site_attested_validator_evidence_is_never_reverifiable_and_raises_nothing() {
    let plain = sealed(None);
    assert!(plain.attested.is_empty());
    for pass in [true, false] {
        let claim = sealed(Some(pass));
        assert_eq!(claim.attested.len(), 1);
        let evidence = &claim.attested[0];
        assert_eq!(&*evidence.name, NAME);
        assert_eq!(&*evidence.kind, "custom_validator");
        assert_eq!(evidence.passed, pass);
        assert!(!evidence.reverifiable, "attested evidence is never re-verifiable");
        assert_eq!(evidence.config_digest, None);
        assert_eq!(evidence.payload_digest, None);
        // The attestation changes no other standing facet of the claim.
        assert_eq!(claim.kind, plain.kind);
        assert_eq!(claim.domains, plain.domains);
        assert_eq!(claim.calibration.status, plain.calibration.status);
        assert_eq!(claim.reasoning.identification, plain.reasoning.identification);
        // It is part of what the claim id binds.
        assert_ne!(claim.claim_id, plain.claim_id);
    }
}
