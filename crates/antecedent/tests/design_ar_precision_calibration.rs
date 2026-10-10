//! Original checked IV recalculation: finite-F AR diagnostics and full run accounting.
//! Independent observed-confounder SCM and finite-F pins; unions are never replaced
//! with a finite scalar confidence interval. Public standing remains unchanged.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, reason = "bounded diagnostic sample dimensions")]
#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent::analysis::recalc_design::{
    DesignModel, DesignRequest, DesignRunError, DesignSession, consume_design_with_data,
    execute_design_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::ExecutionContext;
use antecedent_stats::fit_counts::count_least_squares_solves;
use antecedent_stats::{
    FaerBackend, FirstStageDiagnostics, LeastSquaresWorkspace, anderson_rubin_confidence_set,
    anderson_rubin_kf_critical,
};
use calibration::{grid_n, grid_seed, map_replicates, n_sim};

fn request(n: usize, seed: u64, strength: f64) -> DesignRequest {
    let mut generator = candidate::Generator::new(seed);
    let (mut z, mut u, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let instrument = generator.binary(0.5) as f64;
        let confounder = generator.normal();
        let treatment = strength * instrument + confounder + 0.5 * generator.normal();
        let outcome = 2. * treatment + 0.8 * confounder + 0.5 * generator.normal();
        z.push(instrument);
        u.push(confounder);
        t.push(treatment);
        y.push(outcome);
    }
    DesignRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z), ("u".into(), u)],
        edges: vec![(2, 0), (3, 0), (3, 1), (0, 1)],
        treatment: 0,
        outcome: 1,
        model: DesignModel::Iv2Sls { instrument: 2 },
        utility: UtilitySpec { benefit_per_unit: 1., cost: 0. },
    }
}
fn centered(value: &[f64]) -> Vec<f64> {
    let mean = value.iter().sum::<f64>() / value.len() as f64;
    value.iter().map(|a| a - mean).collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
struct Oracle {
    quadratic: [f64; 3],
    first_stage_f: f64,
    partial_r2: f64,
}
fn oracle(r: &DesignRequest, critical: f64) -> Oracle {
    let t = centered(&r.columns[0].1);
    let y = centered(&r.columns[1].1);
    let z = centered(&r.columns[2].1);
    let zz = dot(&z, &z);
    let zt = dot(&z, &t);
    let zy = dot(&z, &y);
    let tt = dot(&t, &t);
    let yy = dot(&y, &y);
    let yt = dot(&y, &t);
    let t_projection = zt * zt / zz;
    let y_projection = zy * zy / zz;
    let cross = zy * zt / zz;
    let df = (t.len() - 2) as f64;
    Oracle {
        quadratic: [
            df * t_projection - critical * (tt - t_projection),
            -2. * df * cross + 2. * critical * (yt - cross),
            df * y_projection - critical * (yy - y_projection),
        ],
        first_stage_f: df * t_projection / (tt - t_projection),
        partial_r2: t_projection / tt,
    }
}
fn inspect(r: &DesignRequest, diagnostics: &FirstStageDiagnostics, critical: f64) -> bool {
    let n = r.columns[0].1.len();
    assert_eq!(diagnostics.df1, 1);
    assert_eq!(diagnostics.df2, n - 2);
    let independent = oracle(r, critical);
    assert!((diagnostics.f_statistic - independent.first_stage_f).abs() < 1e-7);
    assert!((diagnostics.partial_r2 - independent.partial_r2).abs() < 1e-9);
    let z = &r.columns[2].1;
    let exogenous = vec![1.; n];
    let original = anderson_rubin_confidence_set(
        &r.columns[1].1,
        &r.columns[0].1,
        z,
        n,
        1,
        &exogenous,
        1,
        0.95,
        &FaerBackend,
        &mut LeastSquaresWorkspace::default(),
    )
    .unwrap();
    assert_eq!(diagnostics.anderson_rubin, original.0);
    assert_eq!(diagnostics.uncertainty_withheld, original.1);
    // Independent exact raw-data quadratic, not original AR statistic/inversion.
    let [a, b, c] = independent.quadratic;
    let accepts = 4. * a + 2. * b + c <= 0.;
    if let Some((lo, hi, level)) = diagnostics.anderson_rubin {
        assert!((level - 0.95).abs() < 1e-12);
        assert_eq!(lo <= 2. && 2. <= hi, accepts);
    } else if diagnostics.uncertainty_withheld == Some("anderson_rubin_set_is_union") {
        assert!(a < 0. && b * b - 4. * a * c > 0., "union must retain disjoint-ray withholding");
    } else {
        assert_eq!(diagnostics.uncertainty_withheld, Some("anderson_rubin_set_empty"));
        assert!(!accepts);
    }
    accepts
}
struct Observation {
    accepts: bool,
    finite: bool,
    union: bool,
    model_fits: u64,
}
fn execute_one(r: &DesignRequest, seed: u64, critical: f64, check_replay: bool) -> Observation {
    let ctx = ExecutionContext::for_tests(seed);
    let mut session = DesignSession::new();
    let (executed, solves) =
        count_least_squares_solves(|| execute_design_with_receipt(&mut session, r, &ctx));
    let (diagnostics, finite) = match executed {
        Ok(outcome) => {
            assert_eq!(solves, outcome.receipt.totals().model_fits);
            assert!(outcome.law.std_error.is_nan(), "no fabricated IV Wald SE");
            assert_eq!(outcome.receipt.totals().identifications, 1);
            let diagnostic = session
                .result()
                .unwrap()
                .estimate
                .as_effect()
                .unwrap()
                .first_stage_diagnostics
                .clone()
                .unwrap();
            assert!(
                diagnostic
                    .anderson_rubin
                    .is_some_and(|(lo, hi, _)| lo.is_finite() && hi.is_finite())
            );
            if check_replay {
                let bytes = session.export_result().unwrap();
                let (replayed, replay_solves) =
                    count_least_squares_solves(|| consume_design_with_data(&bytes, Some(r), &ctx));
                let replayed = replayed.unwrap();
                assert_eq!(replayed.outcome.receipt.totals().model_fits, replay_solves);
                assert!(replay_solves > 0);
                assert_eq!(
                    replayed
                        .session
                        .result()
                        .unwrap()
                        .estimate
                        .as_effect()
                        .unwrap()
                        .first_stage_diagnostics
                        .as_ref()
                        .unwrap(),
                    &diagnostic
                );
                assert!(consume_design_with_data(&bytes, None, &ctx).is_err());
            }
            let (reused, reuse_solves) =
                count_least_squares_solves(|| execute_design_with_receipt(&mut session, r, &ctx));
            assert_eq!(reuse_solves, 0);
            assert_eq!(reused.unwrap().receipt.totals().total(), 0);
            (diagnostic, true)
        }
        Err(DesignRunError::WeakInstrument { point, model_fits, diagnostics }) => {
            assert!(point.is_finite());
            assert_eq!(model_fits, solves);
            assert!(!session.is_live(), "refused fit cannot install reusable state");
            assert!(
                !diagnostics
                    .anderson_rubin
                    .is_some_and(|(lo, hi, _)| lo.is_finite() && hi.is_finite())
            );
            (*diagnostics, false)
        }
        Err(error) => panic!("failed run counts as diagnostic failure: {error}"),
    };
    assert!(solves >= 2, "actual IV and AR auxiliary solves must execute");
    let accepts = inspect(r, &diagnostics, critical);
    Observation {
        accepts,
        finite,
        union: diagnostics.uncertainty_withheld == Some("anderson_rubin_set_is_union"),
        model_fits: solves,
    }
}

#[test]
#[ignore = "calibration: final measurement only"]
fn checked_design_iv_ar_full_method_error_rate_and_weak_boundaries() {
    let pins: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/recalculation/design_ar/expected.json"
    ))
    .unwrap();
    assert!((pins["effect"].as_f64().unwrap() - 2.).abs() < f64::EPSILON);
    let n = grid_n(300);
    let row = pins["finite_f_oracles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["n"].as_u64() == Some(n as u64))
        .unwrap();
    let critical = row["f_critical"].as_f64().unwrap();
    assert_eq!(row["df"].as_u64(), Some((n - 2) as u64));
    assert!(
        (anderson_rubin_kf_critical(0.95, 1, n - 2) - critical).abs() < 1e-8,
        "independent Student-density quadrature must match original finite-F critical"
    );
    for strength in pins["strengths"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap()) {
        let runs = map_replicates(n_sim(), |rep| {
            let seed = grid_seed(0xa2_1000 + rep);
            execute_one(&request(n, seed, strength), seed, critical, rep == 0)
        });
        let count = runs.len() as f64;
        let accepted = runs.iter().filter(|r| r.accepts).count() as f64 / count;
        let finite = runs.iter().filter(|r| r.finite).count() as f64 / count;
        let unions = runs.iter().filter(|r| r.union).count() as f64 / count;
        assert!(
            (accepted - 0.95).abs() <= 3. * (0.95 * 0.05 / count).sqrt(),
            "all-run AR truth acceptance={accepted}; no conditioning on finite decision"
        );
        if strength < 0.03 {
            assert!(finite < 0.5, "weak DGP must exercise refused unbounded/union decisions");
        }
        if strength > 0.7 {
            assert!(finite > 0.9, "strong DGP must execute finite decisions");
        }
        println!(
            "diagnostic-precision checked_design_iv_ar_full_method_error_rate_and_weak_boundaries n={n} strength={strength} repetitions={count} known_effect=2 ar_truth_acceptance={accepted} finite_decision_fraction={finite} withheld_union_fraction={unions} actual_model_solves={} no_failed_runs_discarded; no_new_calibration_license",
            runs.iter().map(|r| r.model_fits).sum::<u64>()
        );
    }
}

#[test]
fn checked_design_iv_roles_and_source_diagnostics_match_independent_quadratic() {
    // Fixed engineering fixture: exact source/diagnostic parity, no empirical rate.
    let pins: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/recalculation/design_ar/expected.json"
    ))
    .unwrap();
    assert!((pins["effect"].as_f64().unwrap() - 2.).abs() < f64::EPSILON);
    let row = &pins["finite_f_oracles"][0];
    assert_eq!(row["n"].as_u64(), Some(150));
    assert_eq!(row["df"].as_u64(), Some(148));
    assert!((row["level"].as_f64().unwrap() - 0.95).abs() < f64::EPSILON);
    let critical = row["f_critical"].as_f64().unwrap();
    assert!((anderson_rubin_kf_critical(0.95, 1, 148) - critical).abs() < 1e-8);
    let r = request(150, 119, pins["strengths"][2].as_f64().unwrap());
    let observation = execute_one(&r, 119, critical, true);
    assert!(observation.finite);
    assert!(observation.model_fits > 0);
}
