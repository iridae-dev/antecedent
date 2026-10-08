//! Selective adjusted-regression recalculation against independent finite algebra.
//!
//! These tests assess values, covariance and actual work, not statistical coverage.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_adjusted::{
    AdjustedContrast, AdjustedModel, AdjustedRequest, AdjustedSession,
    execute_adjusted_with_receipt,
};
use antecedent::analysis::recalc_receipt::{RecalcOutcome, TargetWeights, UtilitySpec};
use antecedent_core::recalc::{Stage, StageStatus};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::adjustment_resume::count_adjusted_model_fits;
use antecedent_estimate::categorical_treatment::{CategoricalTreatmentSpec, LevelScale};
use antecedent_estimate::vector_treatment::{
    NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput, VectorTreatmentOptions,
    fit_vector_treatment,
};
use antecedent_stats::{GlmFamily, GlmOptions};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(83)
}
fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 2.0, cost: 0.5 }
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-8, "actual={actual}, expected={expected}");
}
fn numeric(active: &[f64], control: &[f64]) -> AdjustedContrast {
    AdjustedContrast::Numeric { active: active.to_vec(), control: control.to_vec() }
}
fn status(outcome: &RecalcOutcome, stage: Stage, expected: &str) {
    assert_eq!(outcome.plan.status(stage).unwrap().tag(), expected, "{stage:?}");
    assert_eq!(outcome.receipt.entry(stage).unwrap().status.tag(), expected, "receipt {stage:?}");
}
fn run(session: &mut AdjustedSession, request: &AdjustedRequest) -> RecalcOutcome {
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(session, request, &ctx()));
    let outcome = result.unwrap();
    assert_eq!(
        fits,
        outcome.receipt.totals().model_fits,
        "external kernel observer agrees with receipt"
    );
    outcome
}
fn full(request: &AdjustedRequest) -> RecalcOutcome {
    run(&mut AdjustedSession::new(), request)
}
fn same_law(selective: &RecalcOutcome, independent: &RecalcOutcome) {
    close(selective.law.ate, independent.law.ate);
    close(selective.law.std_error, independent.law.std_error);
    close(selective.decision.net_benefit, independent.decision.net_benefit);
}

/// Balanced structural model y=1+2t+.3z+.4e; every (t,z,e) combination occurs.
/// The disturbance is exactly orthogonal to all design columns.
fn linear() -> AdjustedRequest {
    let mut t = Vec::new();
    let mut z = Vec::new();
    let mut y = Vec::new();
    for treatment in [0.0, 1.0] {
        for confounder in [-1.0, 1.0] {
            for error in [-1.0, 1.0] {
                t.push(treatment);
                z.push(confounder);
                y.push(1.0 + 2.0 * treatment + 0.3 * confounder + 0.4 * error);
            }
        }
    }
    AdjustedRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![2],
        model: AdjustedModel::Linear { covariance: VectorCovariance::ModelBased },
        contrast: numeric(&[1.0], &[0.0]),
        target: None,
        utility: utility(),
    }
}

/// Exact finite logit table: p00=1/4,p10=1/2,p01=2/5,p11=2/3.
/// Sixty observations per cell make the fitted score exactly zero at
/// b=(-ln3,ln3,ln2). The g-computation contrast is (1/4+4/15)/2.
fn glm() -> AdjustedRequest {
    let mut t = Vec::new();
    let mut z = Vec::new();
    let mut y = Vec::new();
    for (treatment, confounder, successes) in
        [(0.0, 0.0, 15), (1.0, 0.0, 30), (0.0, 1.0, 24), (1.0, 1.0, 40)]
    {
        for row in 0..60 {
            t.push(treatment);
            z.push(confounder);
            y.push(f64::from(row < successes));
        }
    }
    AdjustedRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![2],
        model: AdjustedModel::Glm {
            family: GlmFamily::BinomialLogit,
            options: GlmOptions::default(),
        },
        contrast: numeric(&[1.0], &[0.0]),
        target: None,
        utility: utility(),
    }
}

/// Independent Fisher information from four binomial strata, ordered 1,Z,T.
/// Invert the symmetric 3x3 matrix by its cofactor formula, rather than using
/// an estimator or its retained covariance to derive the expected delta SE.
fn glm_covariance() -> [f64; 9] {
    let v00 = 3.0 / 16.0;
    let v10 = 1.0 / 4.0;
    let v01 = 6.0 / 25.0;
    let v11 = 2.0 / 9.0;
    let a = 60.0 * (v00 + v10 + v01 + v11);
    let b = 60.0 * (v01 + v11);
    let c = 60.0 * (v10 + v11);
    let d = b;
    let e = 60.0 * v11;
    let f = c;
    let determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    [
        (d * f - e * e) / determinant,
        (c * e - b * f) / determinant,
        (b * e - c * d) / determinant,
        (c * e - b * f) / determinant,
        (a * f - c * c) / determinant,
        (b * c - a * e) / determinant,
        (b * e - c * d) / determinant,
        (b * c - a * e) / determinant,
        (a * d - b * b) / determinant,
    ]
}

fn glm_delta_variance(z1_mass: f64) -> f64 {
    let g0 = (1.0 - z1_mass) * (1.0 / 4.0 - 3.0 / 16.0) + z1_mass * (2.0 / 9.0 - 6.0 / 25.0);
    let gradient =
        [g0, z1_mass * (2.0 / 9.0 - 6.0 / 25.0), (1.0 - z1_mass) / 4.0 + z1_mass * 2.0 / 9.0];
    let covariance = glm_covariance();
    (0..3)
        .map(|i| (0..3).map(|j| gradient[i] * covariance[3 * i + j] * gradient[j]).sum::<f64>())
        .sum()
}

/// Joint treatments share a confounder and correlated residual treatment columns.
/// After adjustment: T1=U,T2=.5U+V. Its 2x2 inverse Gram block is
/// [[1.25,-.5],[-.5,1]]/16 and sigma²=16*.16/(16-4).
fn vector() -> AdjustedRequest {
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut z = Vec::new();
    let mut y = Vec::new();
    for u in [-1.0, 1.0] {
        for v in [-1.0, 1.0] {
            for confounder in [-1.0, 1.0] {
                for error in [-1.0, 1.0] {
                    let a = u + confounder;
                    let b = 0.5 * u + v + 0.5 * confounder;
                    t1.push(a);
                    t2.push(b);
                    z.push(confounder);
                    y.push(1.0 + 2.0 * a - b + 0.3 * confounder + 0.4 * error);
                }
            }
        }
    }
    AdjustedRequest {
        columns: vec![("t1".into(), t1), ("t2".into(), t2), ("y".into(), y), ("z".into(), z)],
        edges: vec![(3, 0), (3, 1), (3, 2), (0, 2), (1, 2)],
        treatments: vec![0, 1],
        outcome: 2,
        adjustment: vec![3],
        model: AdjustedModel::Linear { covariance: VectorCovariance::ModelBased },
        contrast: numeric(&[1.0, -1.0], &[0.0, 0.0]),
        target: None,
        utility: utility(),
    }
}

/// Three equally sized groups, means 2,5,9 with within-group SS=2 each.
/// sigma²=6/(9-3)=1; any pairwise difference has variance2/3.
fn categorical() -> AdjustedRequest {
    let mut levels = Vec::new();
    let mut codes = Vec::new();
    let mut y = Vec::new();
    for (code, label, mean) in [(0.0, "a", 2.0), (1.0, "b", 5.0), (2.0, "c", 9.0)] {
        for error in [-1.0, 0.0, 1.0] {
            levels.push(label.into());
            codes.push(code);
            y.push(mean + error);
        }
    }
    AdjustedRequest {
        columns: vec![("t".into(), codes), ("y".into(), y)],
        edges: vec![(0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![],
        model: AdjustedModel::Categorical {
            levels,
            spec: CategoricalTreatmentSpec {
                declared_levels: vec!["a".into(), "b".into(), "c".into()],
                scale: LevelScale::Unordered,
                reference: "a".into(),
                min_level_rows: 2,
                pairwise: vec![],
                monotonicity: None,
                covariance: VectorCovariance::ModelBased,
            },
        },
        contrast: AdjustedContrast::Categorical { from: "b".into(), to: "c".into() },
        target: None,
        utility: utility(),
    }
}

#[test]
fn adjusted_linear_first_rerun_utility_and_contrast_match_independent_algebra() {
    let mut request = linear();
    let mut session = AdjustedSession::new();
    let first = run(&mut session, &request);
    close(first.law.ate, 2.0);
    close(first.law.std_error.powi(2), 0.128);
    status(&first, Stage::Identification, "recomputed");
    status(&first, Stage::ScoreArtifact, "recomputed");
    assert_eq!(first.receipt.totals().model_fits, 1);
    assert_eq!(first.receipt.totals().identifications, 1);
    let (predictions, fits) = count_adjusted_model_fits(|| {
        session.predict(&[vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]])
    });
    let predictions = predictions.unwrap();
    close(predictions[0], 1.0);
    close(predictions[1], 3.0);
    close(predictions[2], 1.3);
    assert_eq!(fits, 0, "compatible prediction does not refit");
    close(session.covariance().unwrap()[8], 0.128);
    let unchanged = run(&mut session, &request);
    same_law(&unchanged, &first);
    assert_eq!(unchanged.receipt.totals().total(), 0);
    request.utility.cost = 10.0;
    let utility_changed = run(&mut session, &request);
    same_law(&utility_changed, &full(&request));
    status(&utility_changed, Stage::ScoreArtifact, "reused");
    status(&utility_changed, Stage::Law, "reused");
    assert_eq!(utility_changed.receipt.totals().decisions, 1);
    assert!(!utility_changed.decision.treat);
    request.contrast = numeric(&[0.0], &[1.0]);
    let reverse = run(&mut session, &request);
    close(reverse.law.ate, -2.0);
    same_law(&reverse, &full(&request));
    status(&reverse, Stage::Identification, "reused");
    status(&reverse, Stage::ScoreArtifact, "reused");
    status(&reverse, Stage::Law, "recomputed");
}

#[test]
fn adjusted_glm_retargets_probability_contrasts_without_reinterpreting_log_odds() {
    let mut request = glm();
    let mut session = AdjustedSession::new();
    let first = run(&mut session, &request);
    close(first.law.ate, 31.0 / 120.0);
    close(first.law.std_error.powi(2), glm_delta_variance(0.5));
    for (actual, expected) in session.covariance().unwrap().iter().zip(glm_covariance()) {
        close(*actual, expected);
    }
    assert!((first.law.ate - 3.0_f64.ln()).abs() > 0.5);
    request.target = Some(TargetWeights {
        weights: request.columns[2].1.iter().map(|z| if *z > 0.5 { 3.0 } else { 1.0 }).collect(),
        depends_on: vec![VariableId::from_raw(2)],
    });
    let targeted = run(&mut session, &request);
    close(targeted.law.ate, 21.0 / 80.0);
    close(targeted.law.std_error.powi(2), glm_delta_variance(0.75));
    same_law(&targeted, &full(&request));
    status(&targeted, Stage::ScoreArtifact, "reused");
    status(&targeted, Stage::Law, "recomputed");
    request.contrast = numeric(&[0.0], &[1.0]);
    let reverse = run(&mut session, &request);
    close(reverse.law.ate, -21.0 / 80.0);
    same_law(&reverse, &full(&request));
    status(&reverse, Stage::ScoreArtifact, "reused");
}

#[test]
fn adjusted_vector_reused_contrasts_preserve_joint_covariance() {
    let mut request = vector();
    let mut session = AdjustedSession::new();
    let difference = run(&mut session, &request);
    close(difference.law.ate, 3.0);
    let covariance_scale = 0.16 / 12.0;
    close(difference.law.std_error.powi(2), 3.25 * covariance_scale);
    assert!((difference.law.std_error.powi(2) - 2.25 * covariance_scale).abs() > 0.01);
    request.contrast = numeric(&[1.0, 1.0], &[0.0, 0.0]);
    let sum = run(&mut session, &request);
    close(sum.law.ate, 1.0);
    close(sum.law.std_error.powi(2), 1.25 * covariance_scale);
    same_law(&sum, &full(&request));
    status(&sum, Stage::ScoreArtifact, "reused");
    // The identity-link contrast must match the actual shared coefficient fit even
    // when outcome levels are so large that subtracting predictions loses the effect.
    let mut offset = vector();
    for y in &mut offset.columns[2].1 {
        *y += 1e12;
    }
    offset.contrast = numeric(&[0.3, 0.0], &[0.0, 0.0]);
    let direct = fit_vector_treatment(
        &VectorTreatmentInput {
            outcome: offset.columns[2].1.clone(),
            row_snapshot: "offset".into(),
            adjustment: vec![NamedColumn { name: "z".into(), values: offset.columns[3].1.clone() }],
            treatments: offset
                .treatments
                .iter()
                .map(|&id| TreatmentColumn {
                    name: offset.columns[id as usize].0.clone(),
                    values: offset.columns[id as usize].1.clone(),
                    adjustment_set: vec!["z".into()],
                    row_snapshot: "offset".into(),
                })
                .collect(),
        },
        &VectorTreatmentOptions::default(),
    )
    .unwrap();
    let result = run(&mut session, &offset);
    close(result.law.ate, 0.3 * direct.coefficients[0].estimate);
    close(result.law.std_error, 0.3 * direct.coefficients[0].standard_error);
    let mut weighted = offset.clone();
    weighted.target = Some(TargetWeights {
        weights: (0..offset.columns[2].1.len()).map(|i| if i % 2 == 0 { 1. } else { 3. }).collect(),
        depends_on: vec![],
    });
    let weighted = run(&mut session, &weighted);
    same_law(&result, &weighted);
    status(&weighted, Stage::ScoreArtifact, "reused");
}

#[test]
fn adjusted_categorical_reference_and_pairwise_contrasts_preserve_covariance() {
    let mut request = categorical();
    let mut session = AdjustedSession::new();
    let pair = run(&mut session, &request);
    close(pair.law.ate, 4.0);
    close(pair.law.std_error.powi(2), 2.0 / 3.0);
    request.contrast = AdjustedContrast::Categorical { from: "a".into(), to: "c".into() };
    let contrast = run(&mut session, &request);
    close(contrast.law.ate, 7.0);
    close(contrast.law.std_error.powi(2), 2.0 / 3.0);
    same_law(&contrast, &full(&request));
    status(&contrast, Stage::ScoreArtifact, "reused");
    if let AdjustedModel::Categorical { spec, .. } = &mut request.model {
        spec.reference = "b".into();
    }
    let changed_reference = run(&mut session, &request);
    same_law(&changed_reference, &full(&request));
    close(changed_reference.law.ate, 7.0);
    status(&changed_reference, Stage::ScoreArtifact, "recomputed");
    assert!(matches!(
        changed_reference.plan.status(Stage::ScoreArtifact),
        Some(StageStatus::Recomputed { .. })
    ));
}

#[test]
fn adjusted_scalar_paths_match_existing_study_engines_without_bootstrap() {
    use antecedent::{RefuteSuite, Study};
    use antecedent_core::AverageEffectQuery;
    use antecedent_data::TabularData;
    use antecedent_estimate::{GlmAdjustmentAte, LinearAdjustmentAte};
    use antecedent_graph::{Dag, DenseNodeId};
    for (request, use_glm) in [(linear(), false), (glm(), true)] {
        let columns: Vec<(&str, &[f64])> = request
            .columns
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
            .collect();
        let data = TabularData::from_f64_columns(columns).unwrap();
        let mut graph = Dag::with_variables(3);
        for &(from, to) in &request.edges {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let builder = Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .refute(RefuteSuite::None);
        let study = if use_glm {
            builder.estimator(GlmAdjustmentAte::new().with_bootstrap_replicates(0)).build().unwrap()
        } else {
            builder
                .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(0))
                .build()
                .unwrap()
        };
        let original = study.run(&ctx()).unwrap();
        let original = original.estimate.as_effect().unwrap();
        let retained = full(&request);
        close(retained.law.ate, original.ate);
        close(retained.law.std_error, original.se_analytic);
        assert!(original.se_bootstrap.is_none());
    }
}

#[test]
fn adjusted_changes_to_data_and_covariance_refit_without_reidentifying() {
    let mut request = linear();
    let mut session = AdjustedSession::new();
    run(&mut session, &request);
    let treatment = request.columns[0].1.clone();
    for (outcome, treatment) in request.columns[1].1.iter_mut().zip(treatment) {
        *outcome += treatment;
    }
    let changed = run(&mut session, &request);
    close(changed.law.ate, 3.0);
    assert_eq!(changed.receipt.totals().model_fits, 1);
    status(&changed, Stage::Identification, "reused");
    same_law(&changed, &full(&request));
    request.model = AdjustedModel::Linear { covariance: VectorCovariance::Hc3 };
    let robust = run(&mut session, &request);
    close(robust.law.std_error.powi(2), 0.2048);
    assert_eq!(robust.receipt.totals().model_fits, 1);
    status(&robust, Stage::Identification, "reused");
    same_law(&robust, &full(&request));
}

#[test]
fn adjusted_causal_and_support_refusals_preserve_live_fit() {
    let mut session = AdjustedSession::new();
    let request = linear();
    run(&mut session, &request);
    let mut invalid = request.clone();
    invalid.contrast = numeric(&[0.5], &[0.0]);
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(&mut session, &invalid, &ctx()));
    assert!(matches!(
        result.unwrap_err(),
        antecedent::analysis::recalc_receipt::RecalcRunError::Refused(_)
    ));
    assert_eq!(fits, 0);
    invalid = request.clone();
    invalid.adjustment.clear();
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(&mut session, &invalid, &ctx()));
    assert!(result.unwrap_err().to_string().contains("recalc.adjustment_not_identified"));
    assert_eq!(fits, 0);
    assert_eq!(run(&mut session, &request).receipt.totals().model_fits, 0);
    // A joint coefficient fit cannot stand in for a shared causal adjustment proof.
    let mut vector = vector();
    vector.adjustment.clear();
    let (result, fits) = count_adjusted_model_fits(|| {
        execute_adjusted_with_receipt(&mut AdjustedSession::new(), &vector, &ctx())
    });
    assert!(result.unwrap_err().to_string().contains("recalc.adjustment_not_identified"));
    assert_eq!(fits, 0);
    // Dummy regressions must pass that same proof screen.
    let mut cat = categorical();
    cat.columns.push(("z".into(), vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]));
    cat.edges.extend([(2, 0), (2, 1)]);
    let (result, fits) = count_adjusted_model_fits(|| {
        execute_adjusted_with_receipt(&mut AdjustedSession::new(), &cat, &ctx())
    });
    assert!(result.unwrap_err().to_string().contains("recalc.adjustment_not_identified"));
    assert_eq!(fits, 0);
}

#[test]
fn adjusted_receipt_artifact_roundtrip_and_fresh_boundary_never_fake_a_fit() {
    use antecedent_core::recalc::{
        RecalcCapabilities, ResumeContext, RetargetSupport, StageIdentities,
    };
    use antecedent_io::recalc_receipt_artifact::{CountsWire, RecalcReceiptArtifact};
    use std::collections::BTreeMap;
    let request = linear();
    let mut session = AdjustedSession::new();
    let first = run(&mut session, &request);
    let counts: BTreeMap<_, _> = first
        .receipt
        .entries()
        .iter()
        .map(|entry| {
            let c = entry.counts;
            (
                entry.stage,
                CountsWire {
                    identifications: c.identifications,
                    fold_fits: c.fold_fits,
                    model_fits: c.model_fits,
                    score_computations: c.score_computations,
                    reweights: c.reweights,
                    decisions: c.decisions,
                },
            )
        })
        .collect();
    let artifact = RecalcReceiptArtifact::seal(
        &StageIdentities::new(),
        session.identities(),
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
        &counts,
    )
    .unwrap();
    assert_eq!(artifact.receipt_identity(), first.receipt.identity().to_hex());
    let bytes = artifact.to_bytes("adjusted-first-fit").unwrap();
    let restored =
        RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity())).unwrap();
    assert_eq!(restored.receipt_identity(), artifact.receipt_identity());
    assert!(RecalcReceiptArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    // Portable flags are declarations, not coefficient or data artifacts. A fresh session
    // must refuse unchanged reuse when portable-fit flags are optimistically set without raw data.
    let mut fresh = AdjustedSession::resume(
        session.identities().clone(),
        ResumeContext {
            portable_fit: true,
            portable_scores: true,
            scores_snapshot_bound: true,
            supplied_data: false,
            supplied_provider: true,
        },
    );
    assert!(!fresh.is_live());
    assert!(matches!(
        fresh.plan(&request, &ctx()).first_refusal(),
        Some((_, antecedent_core::recalc::RefusalReason::Unavailable { .. }))
    ));
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(&mut fresh, &request, &ctx()));
    assert!(result.is_err());
    assert_eq!(fits, 0);
    // Explicitly supplying the raw request to a new session does a complete fresh fit.
    let mut raw_resume = AdjustedSession::resume(
        session.identities().clone(),
        ResumeContext { supplied_data: true, ..ResumeContext::default() },
    );
    let rerun = run(&mut raw_resume, &request);
    assert_eq!(rerun.receipt.totals().model_fits, 1);
    same_law(&first, &rerun);
}

#[test]
fn adjusted_sparse_categories_and_invalid_predictions_do_no_model_work() {
    let mut request = categorical();
    let mut session = AdjustedSession::new();
    run(&mut session, &request);
    if let AdjustedModel::Categorical { spec, .. } = &mut request.model {
        spec.min_level_rows = 4;
    }
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(&mut session, &request, &ctx()));
    assert!(result.is_err());
    assert_eq!(fits, 0, "sparse levels refuse before a numerical fit");
    assert_eq!(run(&mut session, &categorical()).receipt.totals().model_fits, 0);
    let (result, fits) = count_adjusted_model_fits(|| session.predict(&[vec![1.0, 1.0]]));
    assert!(result.is_err(), "two active dummy levels are not a categorical regime");
    assert_eq!(fits, 0);
    let mut numeric_session = AdjustedSession::new();
    run(&mut numeric_session, &linear());
    for row in [vec![0.0], vec![0.0, f64::NAN], vec![0.0, 0.5], vec![0.0, 2.0]] {
        let (result, fits) = count_adjusted_model_fits(|| numeric_session.predict(&[row]));
        assert!(result.is_err(), "schema, finite and binary treatment support checked");
        assert_eq!(fits, 0);
    }
}

#[test]
fn adjusted_reuse_borrows_covariance_without_refitting_or_copying() {
    let mut request = vector();
    let mut session = AdjustedSession::new();
    run(&mut session, &request);
    let covariance = session.covariance().unwrap().as_ptr();
    let unchanged = run(&mut session, &request);
    assert_eq!(unchanged.receipt.totals().model_fits, 0);
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
    request.utility.cost += 1.0;
    assert_eq!(run(&mut session, &request).receipt.totals().model_fits, 0);
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
    request.contrast = numeric(&[1.0, 1.0], &[0.0, 0.0]);
    assert_eq!(run(&mut session, &request).receipt.totals().model_fits, 0);
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
}

#[test]
fn adjusted_cancel_memory_and_numerical_refit_failures_are_transactional() {
    let original = linear();
    let mut session = AdjustedSession::new();
    run(&mut session, &original);
    let covariance = session.covariance().unwrap().as_ptr();
    let identities = session.identities().clone();
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let (result, fits) = count_adjusted_model_fits(|| {
        execute_adjusted_with_receipt(&mut session, &original, &cancelled)
    });
    assert!(result.unwrap_err().to_string().contains("recalc.cancelled"));
    assert_eq!(fits, 0);
    let mut changed = original.clone();
    changed.columns[1].1[0] += 0.1;
    let mut constrained = ctx();
    constrained.memory.hard_limit_bytes = Some(1);
    let (result, fits) = count_adjusted_model_fits(|| {
        execute_adjusted_with_receipt(&mut session, &changed, &constrained)
    });
    assert!(result.unwrap_err().to_string().contains("recalc.memory_budget_exceeded"));
    assert_eq!(fits, 0);
    // Collinear design has no unique retained covariance. The failed numerical refit
    // must leave the previous checked model and its allocation available for reuse.
    changed = original.clone();
    changed.columns[2].1 = changed.columns[0].1.clone();
    let (result, fits) =
        count_adjusted_model_fits(|| execute_adjusted_with_receipt(&mut session, &changed, &ctx()));
    assert!(result.is_err());
    assert_eq!(fits, 0, "only successful numerical fits are counted");
    assert_eq!(session.identities(), &identities);
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
    assert_eq!(run(&mut session, &original).receipt.totals().model_fits, 0);
}
