//! Engineering contracts for prepared fitted-model reuse; numerical truth is separately registered.
use antecedent::{PreparedStudy, RefuteSuite, Study};
use antecedent_core::execution_attempt::{Operation, PhysicalEvent, observe_execution};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AnalyticSeKind, GlmAdjustmentAte, LinearAdjustmentAte};
use antecedent_graph::{Dag, DenseNodeId};

fn fixture(logit: bool) -> (TabularData, Dag, Vec<f64>) {
    let (mut t, mut y, mut z, mut weights) = (vec![], vec![], vec![], vec![]);
    for (a, b, successes) in [(0.0, 0.0, 15), (1.0, 0.0, 30), (0.0, 1.0, 24), (1.0, 1.0, 40)] {
        for row in 0..60 {
            t.push(a);
            z.push(b);
            weights.push(if b > 0.5 { 3.0 } else { 1.0 });
            y.push(if logit {
                f64::from(row < successes)
            } else {
                1.0 + 2.0 * a + 0.3 * b + if row % 2 == 0 { 0.4 } else { -0.4 }
            });
        }
    }
    let data = TabularData::from_f64_columns(vec![
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (data, graph, weights)
}
fn prepare_model(
    logit: bool,
    se: AnalyticSeKind,
    ctx: &ExecutionContext,
) -> (PreparedStudy, Vec<f64>) {
    let (data, graph, weights) = fixture(logit);
    let builder = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .refute(RefuteSuite::None);
    let study = if logit {
        builder
            .estimator(GlmAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(se))
            .build()
            .unwrap()
    } else {
        builder
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(se))
            .build()
            .unwrap()
    };
    (study.prepare(ctx).unwrap(), weights)
}
#[test]
fn prepared_models_retarget_without_refits_and_keep_their_own_uncertainty() {
    for (logit, se) in [
        (false, AnalyticSeKind::Hc3),
        (true, AnalyticSeKind::Homoskedastic),
        (true, AnalyticSeKind::Hc1),
    ] {
        let ctx = ExecutionContext::for_tests(31);
        let (prepared, weights) = prepare_model(logit, se, &ctx);
        assert!(
            prepared
                .preview_transform(antecedent_contract::TransformIntent::Retarget)
                .unwrap()
                .refused
        );
        assert!(prepared.score_table().is_none());
        assert!(prepared.score_reuse_identity().unwrap().is_none());
        let reuse = prepared.adjusted_fit_reuse_identity().unwrap().expect("typed fit key");
        assert!(
            prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).is_err(),
            "no fabricated fit before estimate"
        );
        let original = observe_execution(|| prepared.estimate_retained(&ctx));
        assert!(original.result.is_ok(), "{:?}", original.result);
        assert!(
            original.report.counts(Operation::GlmFit).completed > 0
                || original.report.counts(Operation::LeastSquaresSolve).completed > 0
                || original.report.physical(PhysicalEvent::ModelFit).unwrap_or(0) > 0,
            "the initial fit counter must be observable"
        );
        let preview =
            prepared.preview_transform(antecedent_contract::TransformIntent::Retarget).unwrap();
        assert!(!preview.refused);
        let result = observe_execution(|| {
            prepared.apply_retarget(&preview, &weights, &[VariableId::from_raw(2)], &ctx)
        });
        assert_eq!(result.report.physical(PhysicalEvent::ModelFit), Some(0));
        assert_eq!(result.report.counts(Operation::GlmFit).attempted, 0);
        assert_eq!(result.report.counts(Operation::LeastSquaresSolve).attempted, 0);
        let result = result.result.unwrap();
        assert_eq!(result.estimate().as_effect().unwrap().uncertainty.analytic.kind, Some(se));
        assert!(result.estimate().as_effect().unwrap().scores.table.is_none());
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "retarget.adjusted_standardization")
        );
        assert!(prepared.score_reuse_identity().unwrap().is_none());
        assert_eq!(prepared.adjusted_fit_reuse_identity().unwrap(), Some(reuse));
    }
}
#[test]
fn adjusted_retarget_refuses_invalid_parents_weights_cancellation_and_budget() {
    let ctx = ExecutionContext::for_tests(31);
    let (prepared, weights) = prepare_model(true, AnalyticSeKind::Homoskedastic, &ctx);
    prepared.estimate_retained(&ctx).unwrap();
    assert!(prepared.retarget(&weights, &[], &ctx).is_err());
    assert!(prepared.retarget(&weights, &[VariableId::from_raw(0)], &ctx).is_err());
    assert!(prepared.retarget(&weights[..3], &[VariableId::from_raw(2)], &ctx).is_err());
    let zero = vec![0.0; weights.len()];
    assert!(prepared.retarget(&zero, &[], &ctx).is_err());
    let tiny = ctx.clone().with_memory_limit(Some(1));
    assert!(prepared.retarget(&weights, &[VariableId::from_raw(2)], &tiny).is_err());
    let cancelled = ExecutionContext::for_tests(31);
    cancelled.cancellation.cancel();
    assert!(prepared.retarget(&weights, &[VariableId::from_raw(2)], &cancelled).is_err());
    assert!(
        prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).is_ok(),
        "failed calls preserve the last successful fitted state"
    );
}
#[test]
fn fitted_retarget_artifacts_keep_distinct_reuse_payloads_and_reexecute_on_the_live_snapshot() {
    let ctx = ExecutionContext::for_tests(31);
    let (prepared, weights) = prepare_model(true, AnalyticSeKind::Homoskedastic, &ctx);
    prepared.estimate_retained(&ctx).unwrap();
    let result = prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).unwrap();
    let causal = prepared.contract_for_result(&result).unwrap();
    let bases = result.calibration_bases(&causal).unwrap();
    assert!(
        bases.iter().all(|basis| basis.key.functional == "row_weights.mean"),
        "fixed-target intervals cannot borrow all-observed coverage"
    );
    let bytes = prepared.encode_contracted_result(&result, "adjusted-retarget", &ctx).unwrap();
    let read = antecedent_io::replay_analysis_result(&bytes).unwrap();
    assert!(
        !read.acceptance.accepts_as_replayed_program(),
        "hash-only fit metadata is not executable replay"
    );
    let contract = read.contract.unwrap();
    assert!(contract.score_reuse.is_none());
    let fit = contract.adjusted_fit_reuse.as_ref().expect("distinct model payload");
    assert_eq!(fit.model, "glm_response_mean_v1");
    assert_eq!(fit.adjustment_set, vec![2]);
    assert_eq!(
        Some(antecedent_io::adjusted_fit_reuse_digest(fit).unwrap().as_bytes()),
        contract.identities.score_reuse.as_ref()
    );
    let replay =
        prepared.reexecute_retarget(contract.target_weights.as_ref().unwrap(), &ctx).unwrap();
    assert_eq!(
        replay.estimate().as_effect().unwrap().ate.to_bits(),
        result.estimate().as_effect().unwrap().ate.to_bits()
    );
    let (fresh, _) = prepare_model(true, AnalyticSeKind::Homoskedastic, &ctx);
    assert!(
        fresh.reexecute_retarget(contract.target_weights.as_ref().unwrap(), &ctx).is_err(),
        "metadata alone cannot manufacture a retained fit"
    );
}

#[test]
fn adjusted_fit_metadata_rejects_tampering_and_failed_estimates_publish_nothing() {
    let ctx = ExecutionContext::for_tests(31);
    let (prepared, weights) = prepare_model(true, AnalyticSeKind::Homoskedastic, &ctx);
    let cancelled = ExecutionContext::for_tests(31);
    cancelled.cancellation.cancel();
    assert!(prepared.estimate_retained(&cancelled).is_err());
    assert!(prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).is_err());
    prepared.estimate_retained(&ctx).unwrap();
    let result = prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "adjusted-retarget", &ctx).unwrap();
    let base = antecedent_io::replay_analysis_result(&bytes).unwrap().contract.unwrap();
    for field in 0..5 {
        let mut changed = base.clone();
        let fit = changed.adjusted_fit_reuse.as_mut().unwrap();
        match field {
            0 => fit.format = 2,
            1 => fit.model = "aipw".into(),
            2 => fit.covariance = "unknown".into(),
            3 => fit.design.len += 1,
            _ => fit.row_index.digest[0] ^= 1,
        }
        assert!(antecedent_io::validate_contract_section(&changed).is_err());
    }
    let (other, _) = prepare_model(false, AnalyticSeKind::Hc3, &ctx);
    other.estimate_retained(&ctx).unwrap();
    assert!(other.reexecute_retarget(base.target_weights.as_ref().unwrap(), &ctx).is_err());
}

/// Independent four-cell logit algebra. Intentionally owed; excluded from engineering runs.
#[test]
fn adjusted_retarget_matches_independent_logit_standardization_and_fisher_covariance() {
    let ctx = ExecutionContext::for_tests(31);
    let (prepared, weights) = prepare_model(true, AnalyticSeKind::Homoskedastic, &ctx);
    prepared.estimate_retained(&ctx).unwrap();
    let result = prepared.retarget(&weights, &[VariableId::from_raw(2)], &ctx).unwrap();
    // Four exact binomial cells satisfy logit(p)=-log(3)+log(3)*t+log(2)*z.
    let cells = [(0.0, 0.0, 0.25), (1.0, 0.0, 0.5), (0.0, 1.0, 0.4), (1.0, 1.0, 2.0 / 3.0)];
    let mut info = [[0.0; 3]; 3];
    for (t, z, p) in cells {
        let x = [1.0, t, z];
        for i in 0..3 {
            for j in 0..3 {
                info[i][j] += 60.0 * p * (1.0 - p) * x[i] * x[j];
            }
        }
    }
    // Adjugate inverse independently specified, rather than production linear algebra.
    let a = info[0][0];
    let b = info[0][1];
    let c = info[0][2];
    let d = info[1][1];
    let e = info[1][2];
    let f = info[2][2];
    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    let inv = [
        [d * f - e * e, c * e - b * f, b * e - c * d],
        [c * e - b * f, a * f - c * c, b * c - a * e],
        [b * e - c * d, b * c - a * e, a * d - b * b],
    ];
    let mut gradient = [0.0; 3];
    for (z, q, p0, p1) in [(0.0, 0.25, 0.25, 0.5), (1.0, 0.75, 0.4, 2.0 / 3.0)] {
        gradient[0] += q * (p1 * (1.0 - p1) - p0 * (1.0 - p0));
        gradient[1] += q * p1 * (1.0 - p1);
        gradient[2] += q * z * (p1 * (1.0 - p1) - p0 * (1.0 - p0));
    }
    let mut variance: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            variance += gradient[i] * inv[i][j] * gradient[j] / det;
        }
    }
    let effect = result.estimate().as_effect().unwrap();
    assert!((effect.ate - 21.0 / 80.0).abs() < 1e-8);
    assert!((effect.uncertainty.analytic.se - variance.sqrt()).abs() < 1e-8);
}
