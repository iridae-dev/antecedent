//! Checked design-effect recalculation against independent structural and covariance truth.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_design::{
    DesignModel, DesignRequest, DesignRunError, DesignSession, consume_design_with_data,
    execute_design_with_receipt,
};
use antecedent::analysis::recalc_receipt::{RecalcOutcome, RecalcRunError, UtilitySpec};
use antecedent_core::ExecutionContext;
use antecedent_core::recalc::{ResumeContext, Stage};
use antecedent_stats::fit_counts::count_least_squares_solves;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(119)
}
fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 2., cost: 0.5 }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} vs {b}");
}
fn request(
    model: DesignModel,
    columns: Vec<(String, Vec<f64>)>,
    edges: Vec<(u32, u32)>,
) -> DesignRequest {
    DesignRequest { columns, edges, treatment: 0, outcome: 1, model, utility: utility() }
}
fn iv(strength: f64) -> DesignRequest {
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..20 {
        for instrument in [0., 1.] {
            for confounder in [-1., 1.] {
                for error in [-1., 1.] {
                    let treatment = strength * instrument + confounder;
                    z.push(instrument);
                    t.push(treatment);
                    y.push(2. * treatment + 0.3 * confounder + 0.4 * error);
                }
            }
        }
    }
    request(
        DesignModel::Iv2Sls { instrument: 2 },
        vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        vec![(2, 0), (0, 1)],
    )
}
fn frontdoor() -> DesignRequest {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..10 {
        for treatment in [0., 1.] {
            for disturbance in [-1., 1.] {
                for error in [-1., 1.] {
                    let mediator = 0.5 * treatment + disturbance;
                    t.push(treatment);
                    m.push(mediator);
                    y.push(3. * mediator + 0.4 * error);
                }
            }
        }
    }
    request(
        DesignModel::Frontdoor { mediator: 2 },
        vec![("t".into(), t), ("y".into(), y), ("m".into(), m)],
        vec![(0, 2), (2, 1)],
    )
}
fn rd() -> DesignRequest {
    let (mut t, mut r, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..20 {
        for running in [-0.5, -0.25, 0.25, 0.5] {
            for error in [-1., 1.] {
                let treatment = f64::from(running >= 0.);
                t.push(treatment);
                r.push(running);
                y.push(
                    1. + 2.5 * treatment + 0.3 * running + 0.8 * treatment * running + 0.4 * error,
                );
            }
        }
    }
    request(
        DesignModel::Rd { running_variable: 2, cutoff: 0., bandwidth: 1. },
        vec![("t".into(), t), ("y".into(), y), ("r".into(), r)],
        vec![(2, 0), (0, 1), (2, 1)],
    )
}
fn run(s: &mut DesignSession, r: &DesignRequest) -> RecalcOutcome {
    let (outcome, solves) =
        count_least_squares_solves(|| execute_design_with_receipt(s, r, &ctx()));
    let outcome = outcome.unwrap();
    assert_eq!(
        solves,
        outcome.receipt.totals().model_fits,
        "source solver observer agrees with actual receipt"
    );
    outcome
}
fn same(a: &RecalcOutcome, b: &RecalcOutcome) {
    close(a.law.ate, b.law.ate);
    if a.law.std_error.is_nan() {
        assert!(b.law.std_error.is_nan());
    } else {
        close(a.law.std_error, b.law.std_error);
    }
    close(a.decision.net_benefit, b.decision.net_benefit);
}

#[test]
fn design_first_fits_match_iv_rd_frontdoor_structural_and_covariance_truth() {
    for (r, truth, solves) in
        [(iv(1.), 2., None), (rd(), 2.5, Some(1)), (frontdoor(), 1.5, Some(2))]
    {
        let mut session = DesignSession::new();
        let first = run(&mut session, &r);
        close(first.law.ate, truth);
        assert_eq!(first.receipt.totals().identifications, 1);
        if let Some(solves) = solves {
            assert_eq!(first.receipt.totals().model_fits, solves);
        } else {
            assert!(first.receipt.totals().model_fits >= 2);
            assert!(first.law.std_error.is_nan());
            let d = session
                .result()
                .unwrap()
                .estimate
                .as_effect()
                .unwrap()
                .first_stage_diagnostics
                .as_ref()
                .unwrap();
            assert!(d.anderson_rubin.is_some_and(|(lo, hi, _)| lo.is_finite() && hi.is_finite()));
        }
        if matches!(r.model, DesignModel::Rd { .. }) {
            close(first.law.std_error.powi(2), 0.04 * 160. / 156.);
        }
        if matches!(r.model, DesignModel::Frontdoor { .. }) {
            close(first.law.std_error.powi(2), 0.4505);
        }
        same(&first, &run(&mut DesignSession::new(), &r));
    }
}

#[test]
fn design_unchanged_and_utility_reuse_actual_immutable_checked_results() {
    for mut r in [iv(1.), rd(), frontdoor()] {
        let mut session = DesignSession::new();
        run(&mut session, &r);
        let result = std::ptr::from_ref(session.result().unwrap());
        assert_eq!(run(&mut session, &r).receipt.totals().total(), 0);
        assert_eq!(std::ptr::from_ref(session.result().unwrap()), result);
        r.utility.cost = 10.;
        let updated = run(&mut session, &r);
        assert_eq!(updated.receipt.totals().model_fits, 0);
        assert_eq!(updated.receipt.totals().decisions, 1);
        assert_eq!(updated.plan.status(Stage::ScoreArtifact).unwrap().tag(), "reused");
        assert_eq!(std::ptr::from_ref(session.result().unwrap()), result);
        same(&updated, &run(&mut DesignSession::new(), &r));
    }
}

#[test]
fn design_new_outcomes_refit_but_reuse_checked_causal_identification() {
    for mut r in [iv(1.), rd(), frontdoor()] {
        let mut session = DesignSession::new();
        let first = run(&mut session, &r);
        let t = r.columns[0].1.clone();
        for (y, t) in r.columns[1].1.iter_mut().zip(t) {
            *y += t;
        }
        // Adding a direct treatment effect violates the front-door structural restriction;
        // use a scaling change instead for that declared linear mediation-only model.
        if matches!(r.model, DesignModel::Frontdoor { .. }) {
            r = frontdoor();
            for y in &mut r.columns[1].1 {
                *y *= 2.;
            }
        }
        let updated = run(&mut session, &r);
        assert!(updated.receipt.totals().model_fits > 0);
        assert_eq!(updated.receipt.totals().identifications, 0);
        assert_eq!(updated.plan.status(Stage::Identification).unwrap().tag(), "reused");
        close(
            updated.law.ate,
            if matches!(r.model, DesignModel::Frontdoor { .. }) {
                first.law.ate * 2.
            } else {
                first.law.ate + 1.
            },
        );
        same(&updated, &run(&mut DesignSession::new(), &r));
    }
}

fn assert_source_backed_replay(
    bytes: &[u8],
    r: &DesignRequest,
    first: &RecalcOutcome,
    artifact_digest: [u8; 32],
) {
    let (replayed, solves) =
        count_least_squares_solves(|| consume_design_with_data(bytes, Some(r), &ctx()).unwrap());
    assert!(solves > 0);
    assert_eq!(replayed.outcome.receipt.totals().model_fits, solves);
    assert_eq!(replayed.artifact_digest, artifact_digest);
    same(first, &replayed.outcome);
    assert!(replayed.session.is_live());
    let (missing, solves) =
        count_least_squares_solves(|| consume_design_with_data(bytes, None, &ctx()));
    assert!(matches!(
        missing,
        Err(DesignRunError::Recalc(RecalcRunError::Request("recalc.design_data_unavailable")))
    ));
    assert_eq!(solves, 0);
    for changed in [
        {
            let mut changed = r.clone();
            changed.columns[1].1[0] += 0.125;
            changed
        },
        {
            let mut changed = r.clone();
            changed.columns[0].0 = "changed_schema".into();
            changed
        },
    ] {
        assert!(matches!(
            consume_design_with_data(bytes, Some(&changed), &ctx()),
            Err(DesignRunError::Recalc(RecalcRunError::Request("recalc.design_artifact_mismatch")))
        ));
    }
    assert!(matches!(
        consume_design_with_data(bytes, Some(r), &ExecutionContext::for_tests(120)),
        Err(DesignRunError::Recalc(RecalcRunError::Request("recalc.design_artifact_mismatch")))
    ));
    if let DesignModel::Rd { .. } = r.model {
        let mut changed = r.clone();
        if let DesignModel::Rd { bandwidth, .. } = &mut changed.model {
            *bandwidth = 0.6;
        }
        // All rows remain in this window and the numerical answer is equal,
        // but the original checked-program binding must still reject it.
        assert!(matches!(
            consume_design_with_data(bytes, Some(&changed), &ctx()),
            Err(DesignRunError::Recalc(RecalcRunError::Request("recalc.design_artifact_mismatch")))
        ));
    }
}

#[test]
fn design_actual_artifact_consumers_and_fresh_raw_data_refits_are_honest() {
    for r in [iv(1.), rd(), frontdoor()] {
        let mut session = DesignSession::new();
        let first = run(&mut session, &r);
        let bytes = session.export_result().unwrap();
        let (consumed, solves) =
            count_least_squares_solves(|| antecedent_io::consume_analysis_result(&bytes).unwrap());
        assert_eq!(solves, 0);
        if matches!(r.model, DesignModel::Rd { .. }) {
            // The existing RD wire lacks the checked running-variable operation.
            // Its consumer names this dependency rather than inventing replay support.
            assert!(!consumed.acceptance.accepts_as_verified_program());
            assert!(
                consumed
                    .acceptance
                    .unresolved
                    .iter()
                    .any(|reason| reason.as_ref() == "dependencies.checked_rd_operation")
            );
        } else {
            assert!(consumed.acceptance.accepts_as_verified_program());
        }
        close(consumed.body.estimate.unwrap(), first.law.ate);
        assert!(antecedent_io::consume_analysis_result(&bytes[..bytes.len() / 2]).is_err());
        assert_source_backed_replay(&bytes, &r, &first, consumed.artifact_digest);
        let mut absent = DesignSession::resume(
            session.identities().clone(),
            ResumeContext {
                portable_fit: true,
                portable_scores: true,
                scores_snapshot_bound: true,
                supplied_provider: true,
                supplied_data: false,
            },
        );
        let (failed, solves) =
            count_least_squares_solves(|| execute_design_with_receipt(&mut absent, &r, &ctx()));
        assert!(matches!(failed, Err(DesignRunError::Recalc(RecalcRunError::Refused(_)))));
        assert_eq!(solves, 0);
        assert!(!absent.is_live());
        let mut raw = DesignSession::resume(
            session.identities().clone(),
            ResumeContext { supplied_data: true, ..ResumeContext::default() },
        );
        let rebuilt = run(&mut raw, &r);
        assert!(rebuilt.receipt.totals().model_fits > 0);
        same(&first, &rebuilt);
    }
}

#[test]
fn design_weak_iv_exposes_actual_ar_diagnostics_without_an_invented_decision() {
    let mut session = DesignSession::new();
    run(&mut session, &iv(1.));
    let old = std::ptr::from_ref(session.result().unwrap());
    let (failed, solves) = count_least_squares_solves(|| {
        execute_design_with_receipt(&mut session, &iv(0.005), &ctx())
    });
    let Err(DesignRunError::WeakInstrument { point, model_fits, diagnostics }) = failed else {
        panic!("expected actual AR decision-bound refusal");
    };
    close(point, 2.);
    assert!(solves > 0);
    assert_eq!(model_fits, solves);
    assert!(
        !diagnostics.anderson_rubin.is_some_and(|(lo, hi, _)| lo.is_finite() && hi.is_finite())
    );
    assert_eq!(std::ptr::from_ref(session.result().unwrap()), old);
    assert_eq!(run(&mut session, &iv(1.)).receipt.totals().total(), 0);
}

#[test]
fn design_cutoff_mediator_and_resource_refusals_preserve_previous_state() {
    for r in [iv(1.), rd(), frontdoor()] {
        let mut session = DesignSession::new();
        run(&mut session, &r);
        let old = std::ptr::from_ref(session.result().unwrap());
        let cancelled = ctx();
        cancelled.cancellation.cancel();
        let ((), solves) = count_least_squares_solves(|| {
            assert!(execute_design_with_receipt(&mut session, &r, &cancelled).is_err())
        });
        assert_eq!(solves, 0);
        let mut changed = r.clone();
        changed.columns[1].1[0] += 0.1;
        let mut budget = ctx();
        budget.memory.hard_limit_bytes = Some(1);
        let ((), solves) = count_least_squares_solves(|| {
            assert!(execute_design_with_receipt(&mut session, &changed, &budget).is_err())
        });
        assert_eq!(solves, 0);
        changed = r.clone();
        match &mut changed.model {
            DesignModel::Iv2Sls { instrument } => *instrument = 1,
            DesignModel::Rd { cutoff, .. } => *cutoff = 2.,
            DesignModel::Frontdoor { mediator } => *mediator = 1,
        };
        assert!(execute_design_with_receipt(&mut session, &changed, &ctx()).is_err());
        assert_eq!(std::ptr::from_ref(session.result().unwrap()), old);
        assert_eq!(run(&mut session, &r).receipt.totals().total(), 0);
    }
}

#[test]
fn design_role_exclusion_and_window_changes_reidentify_actual_checked_engines() {
    for mut r in [iv(1.0), rd(), frontdoor()] {
        if !matches!(r.model, DesignModel::Frontdoor { .. }) {
            r.columns.push(("alternate".into(), r.columns[2].1.clone()));
        }
        let mut session = DesignSession::new();
        let baseline = run(&mut session, &r);
        match &mut r.model {
            DesignModel::Iv2Sls { instrument } => {
                *instrument = 3;
                r.edges = vec![(3, 0), (0, 1)];
            }
            DesignModel::Rd { running_variable, .. } => {
                *running_variable = 3;
                r.edges = vec![(3, 0), (0, 1), (3, 1)];
            }
            DesignModel::Frontdoor { mediator } => {
                *mediator = 0;
                r.columns.swap(0, 2);
                r.treatment = 2;
                r.edges = vec![(2, 0), (0, 1)];
            }
        }
        let changed = run(&mut session, &r);
        assert_eq!(changed.receipt.totals().identifications, 1);
        assert!(changed.receipt.totals().model_fits > 0);
        same(&baseline, &changed);
        same(&changed, &run(&mut DesignSession::new(), &r));
        if let DesignModel::Rd { bandwidth, .. } = &mut r.model {
            *bandwidth = 0.6;
            let window = run(&mut session, &r);
            assert_eq!(window.receipt.totals().identifications, 1);
            assert_eq!(window.receipt.totals().model_fits, 1);
            same(&changed, &window);
            if let DesignModel::Rd { running_variable, cutoff, .. } = &mut r.model {
                *cutoff = 0.1;
                for running in &mut r.columns[usize::try_from(*running_variable).unwrap()].1 {
                    *running += 0.1;
                }
            }
            let shifted = run(&mut session, &r);
            assert_eq!(shifted.receipt.totals().identifications, 1);
            assert_eq!(shifted.receipt.totals().model_fits, 1);
            same(&window, &shifted);
        } else {
            r.edges.push((
                if matches!(r.model, DesignModel::Iv2Sls { .. }) { 3 } else { r.treatment },
                1,
            ));
            let (failed, solves) = count_least_squares_solves(|| {
                execute_design_with_receipt(&mut session, &r, &ctx())
            });
            assert!(
                failed.is_err(),
                "instrument exclusion/front-door complete mediation must be checked"
            );
            assert_eq!(solves, 0, "causal graph refusal precedes numerical fits");
        }
    }
}

#[test]
fn design_receipt_artifact_preserves_real_model_solves_and_summary_without_reweights() {
    use antecedent_core::recalc::{RecalcCapabilities, RetargetSupport, StageIdentities};
    use antecedent_io::recalc_receipt_artifact::{CountsWire, RecalcReceiptArtifact};
    use std::collections::BTreeMap;
    for r in [iv(1.0), rd(), frontdoor()] {
        let mut session = DesignSession::new();
        let first = run(&mut session, &r);
        assert_eq!(first.receipt.totals().law_summaries, 1);
        assert_eq!(first.receipt.totals().reweights, 0);
        let counts: BTreeMap<_, _> = first
            .receipt
            .entries()
            .iter()
            .map(|e| {
                let c = e.counts;
                (
                    e.stage,
                    CountsWire {
                        identifications: c.identifications,
                        model_fits: c.model_fits,
                        law_summaries: c.law_summaries,
                        decisions: c.decisions,
                        ..CountsWire::default()
                    },
                )
            })
            .collect();
        let artifact = RecalcReceiptArtifact::seal(
            &StageIdentities::new(),
            session.identities(),
            &RecalcCapabilities::in_process(RetargetSupport::NotDeclared),
            &counts,
        )
        .unwrap();
        assert_eq!(artifact.receipt_identity(), first.receipt.identity().to_hex());
        let bytes = artifact.to_bytes("design-actual-solves-summary").unwrap();
        let restored =
            RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity())).unwrap();
        assert_eq!(restored.receipt_identity(), artifact.receipt_identity());
        assert!(RecalcReceiptArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    }
}
