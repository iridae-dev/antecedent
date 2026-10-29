//! A retained summary reloads as a fresh reconstruction, and a withdrawn
//! decision does not keep its permission in an independent consumer.
//!
//! The series and prior are the autoregressive fixture already used by the
//! period-refresh contract. This test does not pin those coefficients again:
//! the refresh is compared with a rebuild from the retained rows and with the
//! summary-only artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent::analysis::{
    FlagshipDependence, FlagshipWorkflow, consume_composition_verification_artifact, load_decision,
    replay_decision,
};
use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshSpec, SummaryOnlyFit};
use antecedent_contract::composition_verification::{PriorAuthority, Revalidation};
use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_state::{ConjugateOlsPrior, OlsRefresh, OlsSnapshot, merge_within_tolerance};

const S: [f64; 18] =
    [0.3, 0.9, 0.4, 1.1, 0.8, 1.5, 1.0, 1.6, 1.2, 1.9, 1.1, 1.7, 1.5, 2.0, 1.4, 2.2, 1.6, 2.1];

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn window(start: i64, end: i64, snapshot: &str) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "two_slice_graph".into(),
        horizon: 2,
        lag_alignment: vec![("v".into(), 1)],
        intervention_history: vec!["a1=1".into(), "a2=0".into()],
        selection_targets: set(&["s_t1"]),
        regimes: set(&["observational"]),
        period: ObservationPeriod { start, end },
        unit_ids: set(&["u1", "u2"]),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(
        vec![0.0; 3],
        vec![0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3],
        3.0,
        2.0,
    )
    .unwrap()
}

fn close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| merge_within_tolerance(*a, *b))
}

fn assert_same_summary(cached: &OlsSnapshot, fresh: &OlsSnapshot) {
    assert_eq!(cached.source, fresh.source);
    assert_eq!(cached.source_digest, fresh.source_digest);
    assert_eq!(cached.ledger, fresh.ledger);
    assert_eq!(cached.n, fresh.n);
    assert!(close(&cached.xtx, &fresh.xtx));
    assert!(close(&cached.xty, &fresh.xty));
    assert!(merge_within_tolerance(cached.yty, fresh.yty));
    assert!(close(cached.beta.as_deref().unwrap(), fresh.beta.as_deref().unwrap()));
    assert!(merge_within_tolerance(
        cached.residual_variance.unwrap(),
        fresh.residual_variance.unwrap()
    ));
}

fn on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn the workflow")
        .join()
        .expect("the workflow panicked");
}

#[test]
fn refreshed_summary_reloads_and_withdrawal_drops_permission() {
    let spec = PeriodRefreshSpec {
        series_id: "v",
        cadence: ObservationCadence::new(1).unwrap(),
        initial_law: "law_1",
        standing: MechanismStanding::DeclaredInvariant,
        lags: 2,
    };
    let mut refresh =
        PeriodRefresh::start(&spec, &window(0, 6, "snap_1"), &prior(), &S[0..6]).unwrap();
    refresh
        .refresh(&window(6, 12, "snap_2"), ObservationCadence::new(1).unwrap(), "law_1", &S[6..12])
        .unwrap();
    refresh.check_against_fresh().unwrap();
    let fit = refresh.fit();
    let width = fit.rows()[0].x.len();
    let rebuilt = OlsRefresh::from_rows(fit.source(), width, fit.rows().to_vec()).unwrap();
    assert_same_summary(&rebuilt.snapshot(), &fit.snapshot());
    let snapshot = fit.snapshot();
    let bytes = refresh.export_summary_only().unwrap();
    drop(refresh);
    let consumed = SummaryOnlyFit::consume(&bytes).unwrap();
    assert_eq!(consumed.n(), snapshot.n);
    assert!(
        consumed
            .beta()
            .unwrap()
            .iter()
            .zip(snapshot.beta.as_deref().unwrap())
            .all(|(left, right)| merge_within_tolerance(*left, *right))
    );

    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(0xF1A6);
        let mut workflow =
            FlagshipWorkflow::execute(&ctx, FlagshipDependence::Independent).unwrap();
        workflow.period_change((0, 49), &ctx).unwrap();
        let decision = workflow.decision().unwrap();
        let mean = decision.joint.value("mean").unwrap();
        let variance = decision.joint.value("variance").unwrap();
        let artifact = workflow.export_decision().unwrap();
        let loaded =
            load_decision(&artifact.bytes, Some(&artifact.identity), workflow.authority(), &ctx)
                .unwrap();
        assert_eq!(loaded.mean.to_bits(), mean.to_bits());
        assert_eq!(loaded.variance.to_bits(), variance.to_bits());
        let historical = replay_decision(&artifact.bytes, Some(&artifact.identity), &ctx).unwrap();
        let verification = workflow.withdraw_cluster().unwrap();
        let withdrawn_load =
            load_decision(&artifact.bytes, Some(&artifact.identity), workflow.authority(), &ctx)
                .err()
                .expect("a withdrawn cluster refuses to load the decision");
        let withdrawn_load = withdrawn_load.to_string();
        assert!(
            withdrawn_load.starts_with("evidence_ledger.result_not_valid: result ")
                && withdrawn_load
                    .ends_with(" is recompute: its evidence changed after it was registered"),
            "{withdrawn_load}"
        );
        assert_eq!(
            replay_decision(&artifact.bytes, Some(&artifact.identity), &ctx).unwrap(),
            historical
        );
        assert_eq!(verification.classify("decision").unwrap().class, Revalidation::Withdrawn);
        let identity = verification.identity().to_owned();
        let bytes = verification.export("cluster-withdrawal").unwrap();
        drop(verification);
        drop(workflow);
        let consumed = consume_composition_verification_artifact(&bytes, Some(&identity)).unwrap();
        let decision = consumed.revalidation().claim("decision").unwrap();
        assert_eq!(decision.class, Revalidation::Withdrawn);
        assert_eq!(decision.authority, PriorAuthority::NotRetained);
        assert_eq!(
            consumed
                .require_valid("decision")
                .err()
                .expect("a withdrawn claim is not valid")
                .to_string(),
            "reason=claim_license_withdrawn: composition_verification.input_retired: refused at \
             stage `revalidate`; claim `decision`; origin cluster_trial; license lost: retired"
        );
        assert_eq!(
            consumed.revalidation().claim("temporal_drift").unwrap().class,
            Revalidation::StillValid
        );
    });
}
