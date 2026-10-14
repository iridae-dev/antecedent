//! Compatibility class of the observation-recovery derivation record.
//!
//! Class: requires regeneration. The m-graph signature and the derivation and scientific
//! identities are built from canonical length-delimited digests instead of formatted
//! text, so a derivation record written earlier does not equal the record a consumer
//! re-decides. The consumer never trusts a stored record: it re-decides the derivation
//! and refuses a record that differs, which is why the refusal is a derivation mismatch
//! rather than a version gate. The rule version itself is also spelled by behavior
//! (`binary_observation_recovery.v1`), and the earlier spelling is not accepted. Coverage
//! records keyed by the earlier identity are preserved, no longer match, and stay
//! unmeasured until the whole method is measured again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use antecedent_core::ExecutionContext;
use antecedent_identify::recovery::m_graph_signature;
use antecedent_identify::{
    RECOVERY_DEFAULT_LIMITS, RECOVERY_RULE_VERSION, RecoveryDecision, RecoveryDetail,
    RecoveryLimits, StoredRecovery, decide_observation_recovery, verify_observation_recovery,
};
use recovery_scm::MModel;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// O -> X0, X0 -> X1, R0 <- O, R1 <- X0.
fn two_partial() -> MModel {
    MModel::new(2, 1, &[(2, 0), (0, 1)], &[vec![2], vec![0]], 11).unwrap()
}

#[test]
fn a_derivation_record_in_the_earlier_text_format_is_refused_and_regenerated() {
    let model = two_partial();
    let query = model.query();
    let catalog = model.catalog();
    let decision = decide_observation_recovery(
        &model.graph,
        &query,
        &catalog,
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap();
    let RecoveryDecision::Recovered(derivation) = decision else {
        panic!("the compliant m-graph is recovered");
    };

    // The current identities are canonical digests under the new domains.
    let record = derivation.record().clone();
    assert_eq!(record.rule_version, RECOVERY_RULE_VERSION);
    assert!(record.graph_signature.starts_with("mgraph.v2:"), "{}", record.graph_signature);
    assert_eq!(record.graph_signature, m_graph_signature(&model.graph));
    assert!(derivation.identity().starts_with(RECOVERY_RULE_VERSION));
    assert!(
        derivation.scientific_identity().starts_with("recovery_scientific.v2:"),
        "{}",
        derivation.scientific_identity()
    );

    let verify = |record: &antecedent_identify::RecoveryDerivationRecord| {
        verify_observation_recovery(
            &model.graph,
            &query,
            &catalog,
            None,
            &StoredRecovery { record, arena: derivation.arena(), effect_arena: None },
            RECOVERY_DEFAULT_LIMITS,
            &ctx(),
        )
    };
    // The record as a current producer wrote it verifies.
    assert_eq!(verify(&record).expect("current record").identity(), derivation.identity());

    // The earlier signature was the formatted node and edge lists; a record carrying it
    // is not the record the consumer re-decides.
    let mut earlier = record;
    earlier.graph_signature =
        "mgraph.v1;nodes=[0, 1, 2, 3, 4];directed=[(0, 1), (2, 0)]".to_owned();
    assert_eq!(verify(&earlier).unwrap_err().detail, RecoveryDetail::InvalidDerivation);

    // The earlier rule-version spelling is not accepted either.
    let mut spelled = derivation.record().clone();
    spelled.rule_version = "x10.recovery.v1".to_owned();
    assert_eq!(verify(&spelled).unwrap_err().detail, RecoveryDetail::InvalidDerivation);
}
