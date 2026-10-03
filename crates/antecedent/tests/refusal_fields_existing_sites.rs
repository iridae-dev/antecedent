//! Structured refusal fields at the existing refusal sites that live above the estimators
//! (2.2 preflight cell, completion): the real cross-fitted AIPW refusals under rank 174 of 175
//! and joint separation (both flow through the stats-layer conversion), the retargeted overlap
//! refusal, the batch retarget members, and the builder's estimator-setting conflict.
//!
//! Every message is pinned byte for byte as it was before the fields existed, and every figure
//! a refusal reports is checked against a value computed independently here. An entry the
//! failing step never computed must be absent, not zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "fixtures convert small row counts to f64, index with ranks, and compare exact copies"
)]

use antecedent::{
    BatchRetargetRequest, BatchStudy, CausalError, EstimatorId, RefuteSuite, RetargetClaim,
    RetargetContrast, Study,
};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{AipwAte, EstimationError, LinearAdjustmentAte, RefusalFields};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

fn stream(seed: u64, index: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, index)
}

fn frame(columns: &[(String, Vec<f64>)]) -> TabularData {
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn id(data: &TabularData, name: &str) -> VariableId {
    data.schema().id_of(name).unwrap()
}

/// Columns `t`, `y`, `x0..x{p-1}` with every `x` a parent of `t` and `y`, and `t -> y`.
fn wide(
    covariates: Vec<Vec<f64>>,
    t: Vec<f64>,
    y: Vec<f64>,
) -> (TabularData, Dag, AverageEffectQuery) {
    let p = covariates.len();
    let mut columns = vec![("t".to_string(), t), ("y".to_string(), y)];
    columns.extend(covariates.into_iter().enumerate().map(|(i, c)| (format!("x{i}"), c)));
    let data = frame(&columns);
    let mut graph = Dag::with_variables(u32::try_from(2 + p).unwrap());
    let node = |raw: usize| DenseNodeId::from_raw(u32::try_from(raw).unwrap());
    for i in 0..p {
        graph.insert_directed(node(2 + i), node(0)).unwrap();
        graph.insert_directed(node(2 + i), node(1)).unwrap();
    }
    graph.insert_directed(node(0), node(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(id(&data, "t"), id(&data, "y"));
    (data, graph, query)
}

fn prepare_aipw(
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    seed: u64,
) -> Result<antecedent::PreparedStudy, CausalError> {
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(seed))
}

fn absent(fields: &RefusalFields, what: &str) {
    assert!(fields.arm_ess.is_empty(), "{what}: {:?}", fields.arm_ess);
    assert!(fields.propensity_min.is_none() && fields.propensity_max.is_none(), "{what}");
    assert!(fields.propensity_quantiles.is_empty(), "{what}");
    assert!(fields.cluster_count.is_none() && fields.cluster_minimum.is_none(), "{what}");
}

/// The real cross-fitted AIPW refusal for one exact duplicate among 175 design columns: the
/// logistic IRLS' QR reports rank 174 of 175 and the refusal now carries both numbers.
#[test]
fn the_aipw_refusal_at_rank_174_of_175_carries_rank_and_column_count() {
    let n = 2000;
    let mut rng = stream(1, 1);
    let mut covariates: Vec<Vec<f64>> =
        (0..173).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
    covariates.push(covariates[5].clone());
    let t: Vec<f64> = (0..n).map(|_| f64::from(rng.next_f64() < 0.5)).collect();
    let y: Vec<f64> = (0..n).map(|i| t[i] + covariates[0][i] + standard_normal(&mut rng)).collect();
    let (data, graph, query) = wide(covariates, t, y);
    let error = prepare_aipw(data, graph, query, 1).expect_err("a duplicate column refuses");

    // Same class and message as before the fields: the plain stats-layer text.
    // The facade names the cell and wraps the refusal; peeled, it is the stats-layer error.
    assert!(
        matches!(error.peeled(), CausalError::Estimate(EstimationError::Stats(_))),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "rank deficient: rank=174 ncols=175");
    assert_eq!(error.reason_code(), None);

    let fields = error.refusal_fields().expect("the AIPW refusal carries structured fields");
    assert_eq!(fields.stage.as_deref(), Some("design_rank"));
    assert_eq!(fields.reason.as_deref(), Some("rank_deficient"));
    assert_eq!((fields.numerical_rank, fields.design_columns), (Some(174), Some(175)));
    assert!(fields.remedy.is_some());
    // The facade names the treatment and, from the fit-free preflight of the same design, the
    // dependent column (of the duplicated pair x5 / x173, whichever the scan order keeps last).
    assert_eq!(fields.subject.as_deref(), Some("effect(t -> y)"));
    assert_eq!(fields.implicated_columns.len(), 1, "{:?}", fields.implicated_columns);
    assert!(["x5", "x173"].contains(&fields.implicated_columns[0].as_str()));
    // No propensity was ever fitted.
    absent(&fields, "rank-deficient AIPW");
    assert!(fields.glm_iterations.is_none() && fields.boundary_margin.is_none());
}

/// 175 columns on 240 rows: random arm labels are linearly separable, so the strict
/// propensity fit refuses. Whichever defect the strict fit reports first, the refusal's text
/// is the stats text and its fields are that defect's own record.
#[test]
fn the_aipw_refusal_under_joint_separation_carries_the_fit_facts() {
    let n = 240;
    let p = 174;
    let mut rng = stream(2, 1);
    let covariates: Vec<Vec<f64>> =
        (0..p).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
    let t: Vec<f64> = (0..n).map(|_| f64::from(rng.next_f64() < 0.5)).collect();
    let y: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let (data, graph, query) = wide(covariates, t, y);
    let error = prepare_aipw(data, graph, query, 2).expect_err("a separable fit refuses");
    assert!(
        matches!(error.peeled(), CausalError::Estimate(EstimationError::Stats(_))),
        "{error:?}"
    );
    assert_eq!(error.reason_code(), None);

    let fields = error.refusal_fields().expect("the AIPW refusal carries structured fields");
    assert!(fields.remedy.is_some());
    assert_eq!(fields.subject.as_deref(), Some("effect(t -> y)"));
    absent(&fields, "joint separation");
    match fields.stage.as_deref() {
        Some("glm_fit") => {
            let reason = fields.reason.as_deref().unwrap();
            let text = match reason {
                "non_converged" => {
                    "backend error: GLM IRLS did not converge; refuse propensity/outcome scores"
                }
                "separated" => {
                    "backend error: GLM indicates (quasi-)complete separation; refuse propensity/outcome scores"
                }
                "boundary_saturated" => {
                    "backend error: GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not necessarily separation); refuse propensity/outcome scores"
                }
                other => panic!("unexpected GLM reason {other}"),
            };
            assert_eq!(error.to_string(), text);
            assert!(fields.glm_iterations.is_some_and(|k| k >= 1));
            let margin = fields.boundary_margin.expect("a binomial fit measures the margin").0;
            assert!((0.0..=0.5).contains(&margin), "{margin}");
            assert!(fields.boundary_count.is_some());
        }
        Some("design_rank") => {
            // The IRLS design lost numerical rank before any score existed: counts only.
            let (rank, columns) = (fields.numerical_rank.unwrap(), fields.design_columns.unwrap());
            assert_eq!(columns, 175);
            assert!(rank < columns, "{rank} of {columns}");
            assert_eq!(error.to_string(), format!("rank deficient: rank={rank} ncols={columns}"));
            assert!(fields.glm_iterations.is_none() && fields.boundary_margin.is_none());
        }
        other => panic!("unexpected stage {other:?}"),
    }
}

/// `z`-confounded data in which the treated arm only exists where `z < 0`.
fn confounded_retarget_study() -> (antecedent::PreparedStudy, Vec<f64>) {
    let mut rng = stream(14, 1);
    let n = 800usize;
    let (mut z, mut t, mut y, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t[i] = f64::from(zi < 0.0 && rng.next_f64() < 0.7);
        y[i] = t[i] + 0.2 * standard_normal(&mut rng);
        w[i] = f64::from(zi > 1.2);
    }
    let data = frame(&[("t".into(), t), ("y".into(), y), ("z".into(), z)]);
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (prepare_aipw(data, graph, query, 14).unwrap(), w)
}

/// A retarget onto rows the treated arm never reaches refuses as a support failure with the
/// same text, and the refusal now reports the per-arm effective sizes and the propensity range
/// it measured under the target.
#[test]
fn a_retarget_onto_unsupported_rows_reports_arm_sizes_and_the_propensity_range() {
    let (prepared, w) = confounded_retarget_study();
    let ctx = ExecutionContext::for_tests(14);
    let error = prepared.retarget(&w, &[VariableId::from_raw(2)], &ctx).unwrap_err();

    assert!(matches!(error.peeled(), CausalError::Support { .. }), "{error:?}");
    assert_eq!(
        error.to_string(),
        "refused: retarget refused: weighted overlap failed under the declared target weights"
    );
    assert_eq!(error.reason_code(), None);
    assert!(error.blocker_id().is_some(), "the classification is that of the wrapped refusal");

    let fields = error.refusal_fields().expect("the support refusal carries structured fields");
    assert_eq!(fields.stage.as_deref(), Some("retarget"));
    assert_eq!(fields.reason.as_deref(), Some("arm_effective_sample_size_below_minimum"));
    assert!(fields.remedy.is_some());
    assert_eq!(fields.subject.as_deref(), Some("effect(t -> y)"));

    // Independent oracle: unit weights are indicators, so the Kish size of an arm is the count
    // of its rows with z > 1.2 (the weight is a 0/1 indicator on the held-out rows).
    let control = fields.arm_ess.iter().find(|(label, _)| label == "control").unwrap().1.0;
    let active = fields.arm_ess.iter().find(|(label, _)| label == "active").unwrap().1.0;
    assert_eq!(fields.arm_ess.len(), 2);
    assert!(active < 10.0, "the treated arm has no row in the target: {active}");
    let target_rows = w.iter().filter(|&&v| v > 0.5).count() as f64;
    assert!(control >= 10.0 && control <= target_rows, "{control} of {target_rows}");

    // The raw propensity range among rows the target weights is a subrange of (0, 1).
    let (lo, hi) = (fields.propensity_min.unwrap().0, fields.propensity_max.unwrap().0);
    assert!(0.0 < lo && lo <= hi && hi < 1.0, "{lo} {hi}");
    // Quantiles of the finite raw propensities of rows the target weights, recomputed here
    // from the frozen table: nearest rank, ceil(p * m) - 1 over the m sorted scores.
    let table = prepared.score_table().unwrap();
    let mut scores: Vec<f64> = table
        .propensities
        .chunks(table.n_rows)
        .flat_map(|column| column.iter().zip(&w))
        .filter(|(p, weight)| **weight > 0.0 && p.is_finite())
        .map(|(p, _)| *p)
        .collect();
    scores.sort_by(f64::total_cmp);
    let levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
    assert_eq!(fields.propensity_quantiles.len(), levels.len());
    for ((p, value), level) in fields.propensity_quantiles.iter().zip(levels) {
        let rank = ((level * scores.len() as f64).ceil() as usize).clamp(1, scores.len());
        assert_eq!((p.0, value.0), (level, scores[rank - 1]));
    }
    assert_eq!(scores[0], lo);
    assert_eq!(scores[scores.len() - 1], hi);
    assert!(fields.cluster_count.is_none() && fields.numerical_rank.is_none());
    assert!(fields.glm_iterations.is_none() && fields.boundary_margin.is_none());
}

/// A refusal raised before any score table is read carries stage, reason and remedy only:
/// the arm sizes and propensity range are absent, not zero.
#[test]
fn a_dependence_refusal_before_any_support_is_computed_has_no_numbers() {
    let (prepared, w) = confounded_retarget_study();
    let ctx = ExecutionContext::for_tests(14);
    // Weights that depend on the treatment are refused before any weighted support exists.
    let error = prepared.retarget(&w, &[VariableId::from_raw(0)], &ctx).unwrap_err();
    assert_eq!(
        error.to_string(),
        "retarget depends_on must not include the treatment, an intervened coordinate, or a descendant"
    );
    assert!(matches!(
        error.peeled(),
        CausalError::Estimate(EstimationError::UnsupportedWithFields { .. })
    ));
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("retarget"));
    assert_eq!(fields.subject.as_deref(), Some("effect(t -> y)"));
    assert_eq!(fields.reason.as_deref(), Some("weights_depend_on_treatment_or_descendant"));
    assert!(fields.remedy.is_some());
    absent(&fields, "dependence refusal");
}

struct Batch {
    prepared: antecedent::PreparedBatch,
    z: Vec<f64>,
}

fn retarget_batch() -> Batch {
    let n = 500usize;
    let mut rng = stream(65, 0xE3);
    let (mut t1, mut y1, mut y2, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p1 = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        t1[i] = f64::from(rng.next_f64() < p1);
        y1[i] = 2.0 * t1[i] + zi + 0.3 * standard_normal(&mut rng);
        y2[i] = -t1[i] - 0.5 * zi + 0.3 * standard_normal(&mut rng);
    }
    let data =
        frame(&[("t1".into(), t1), ("y1".into(), y1), ("y2".into(), y2), ("z".into(), z.clone())]);
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let queries = [
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2)),
    ];
    let prepared = BatchStudy::new(data, graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&queries, &ExecutionContext::for_tests(65))
        .unwrap();
    Batch { prepared, z }
}

fn claim(name: &str, query_index: usize, weights: Vec<f64>) -> RetargetClaim {
    RetargetClaim {
        name: name.into(),
        query_index,
        weights,
        depends_on: vec![VariableId::from_raw(3)],
    }
}

/// Failed batch members name the claim or contrast as the refusal's subject; the overlap
/// failure carries the arm sizes and propensity range, a malformed declaration carries none.
#[test]
fn failed_batch_members_carry_the_claim_as_subject_and_only_what_they_measured() {
    let batch = retarget_batch();
    let scores = batch.prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let plus: Vec<f64> = rows.iter().map(|&r| (0.4 * batch.z[r as usize]).exp()).collect();
    let mut spike = vec![0.0; rows.len()];
    spike[..3].fill(1.0);
    let request = BatchRetargetRequest {
        claims: vec![
            claim("good", 0, plus.clone()),
            claim("spiked", 1, spike),
            claim("short", 1, plus[..10].to_vec()),
        ],
        contrasts: vec![RetargetContrast {
            name: "good_minus_spiked".into(),
            coefficients: vec![("good".into(), 1.0), ("spiked".into(), -1.0)],
        }],
        expected_snapshot: None,
    };
    let report =
        batch.prepared.retarget(&scores, &request, &ExecutionContext::for_tests(1)).unwrap();
    assert!(report.claims[0].outcome.is_ok());

    let spiked = report.claims[1].outcome.as_ref().unwrap_err();
    assert!(spiked.support_refused);
    assert_eq!(spiked.detail.as_deref(), Some("batch_retarget.weighted_overlap_failed"));
    assert_eq!(spiked.reason_code.as_deref(), Some("cell_not_licensed"));
    let fields = spiked.fields.as_deref().expect("a failed member carries fields");
    assert_eq!(fields.stage.as_deref(), Some("batch_retarget"));
    assert_eq!(fields.subject.as_deref(), Some("spiked"));
    assert_eq!(fields.arm_ess.len(), 2);
    assert!(fields.arm_ess.iter().all(|(_, ess)| ess.0 < 10.0), "{:?}", fields.arm_ess);
    // Only three rows carry weight, so at most three of the two arms' rows are counted.
    assert!(fields.arm_ess.iter().map(|(_, ess)| ess.0).sum::<f64>() <= 3.0 + 1e-9);
    assert!(fields.propensity_min.is_some() && fields.propensity_max.is_some());
    assert!(fields.remedy.is_some());

    let short = report.claims[2].outcome.as_ref().unwrap_err();
    assert_eq!(short.detail.as_deref(), Some("batch_retarget.incompatible_target"));
    assert_eq!(short.reason_code.as_deref(), Some("invalid_argument"));
    let fields = short.fields.as_deref().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("batch_retarget"));
    assert_eq!(fields.subject.as_deref(), Some("short"));
    assert_eq!(fields.reason.as_deref(), Some("batch_retarget.incompatible_target"));
    assert!(fields.remedy.is_some());
    absent(fields, "malformed weights");

    let contrast = report.contrasts[0].outcome.as_ref().unwrap_err();
    let fields = contrast.fields.as_deref().unwrap();
    assert_eq!(fields.subject.as_deref(), Some("good_minus_spiked"));
    assert_eq!(fields.reason.as_deref(), Some("batch_retarget.contrast_member_failed"));
    absent(fields, "failed contrast member");
}

/// Family-level declaration refusals name the offending claim or contrast.
#[test]
fn a_malformed_family_declaration_names_its_subject() {
    let batch = retarget_batch();
    let scores = batch.prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let plus: Vec<f64> = rows.iter().map(|&r| (0.4 * batch.z[r as usize]).exp()).collect();
    let ctx = ExecutionContext::for_tests(1);
    let refuse = |request: BatchRetargetRequest| {
        batch.prepared.retarget(&scores, &request, &ctx).unwrap_err()
    };

    let duplicate = refuse(BatchRetargetRequest {
        claims: vec![claim("a", 0, plus.clone()), claim("a", 0, plus.clone())],
        contrasts: vec![],
        expected_snapshot: None,
    });
    assert_eq!(duplicate.detail, "batch_retarget.duplicate_name");
    let fields = duplicate.refusal_fields();
    assert_eq!(fields.stage.as_deref(), Some("batch_retarget"));
    assert_eq!(fields.subject.as_deref(), Some("a"));
    assert_eq!(fields.reason.as_deref(), Some("batch_retarget.duplicate_name"));
    assert!(fields.remedy.is_some());

    let ghost = refuse(BatchRetargetRequest {
        claims: vec![claim("a", 0, plus)],
        contrasts: vec![RetargetContrast {
            name: "c".into(),
            coefficients: vec![("ghost".into(), 1.0)],
        }],
        expected_snapshot: None,
    });
    assert_eq!(ghost.detail, "batch_retarget.unknown_claim");
    assert_eq!(ghost.refusal_fields().subject.as_deref(), Some("c"));

    // A refusal about the whole family names no subject.
    let empty = refuse(BatchRetargetRequest::default());
    assert_eq!(empty.detail, "batch_retarget.empty_family");
    assert!(empty.refusal_fields().subject.is_none());
}

/// A configured estimator and an explicit bootstrap count both set the replicate count: the
/// study refuses to build with the text it always had, now naming the setting as subject.
#[test]
fn the_estimator_setting_conflict_names_the_setting() {
    let n = 60usize;
    let mut rng = stream(7, 1);
    let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let t: Vec<f64> =
        z.iter().map(|v| f64::from(*v + 0.3 * standard_normal(&mut rng) > 0.0)).collect();
    let y: Vec<f64> = (0..n).map(|i| t[i] + z[i]).collect();
    let data = frame(&[("t".into(), t), ("y".into(), y), ("z".into(), z)]);
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let error = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query.clone())
        .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(500))
        .bootstrap_replicates(100)
        .build()
        .expect_err("both set the replicate count");
    assert_eq!(
        error.to_string(),
        "conflicting configuration for bootstrap_replicates: set on both the builder and the \
         configured estimator; set it in one place (prefer the estimator)"
    );
    assert!(matches!(error.peeled(), CausalError::Conflict { what: "bootstrap_replicates", .. }));
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("study_build"));
    assert_eq!(fields.subject.as_deref(), Some("bootstrap_replicates"));
    assert!(fields.reason.is_some() && fields.remedy.is_some());
    absent(&fields, "builder conflict");

    // The batch entry point builds a study per query, so it refuses the same way.
    let batch_error = BatchStudy::new(data, graph)
        .estimator_spec(AipwAte::new())
        .bootstrap_replicates(100)
        .prepare(&[query], &ExecutionContext::for_tests(7))
        .expect_err("a configured batch estimator refuses an explicit bootstrap count");
    assert!(matches!(batch_error.peeled(), CausalError::Conflict { .. }), "{batch_error:?}");
    assert_eq!(
        batch_error.refusal_fields().unwrap().subject.as_deref(),
        Some("bootstrap_replicates")
    );
}

/// A regression estimator's own design fit (default linear adjustment, one-shot `run` and a
/// prepared `estimate`) refuses a duplicated adjustment column with the stats text, and the
/// facade names the effect and the dependent column.
#[test]
fn a_linear_adjustment_rank_refusal_names_the_effect_and_the_dependent_column() {
    let n = 300usize;
    let mut rng = stream(21, 1);
    let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let t: Vec<f64> = z.iter().map(|v| v + 0.5 * standard_normal(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + z[i] + standard_normal(&mut rng)).collect();
    let data =
        frame(&[("t".into(), t), ("y".into(), y), ("z".into(), z.clone()), ("zcopy".into(), z)]);
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);

    let named = |error: CausalError| {
        assert!(error.to_string().contains("rank deficient"), "{error}");
        assert!(
            matches!(error.peeled(), CausalError::Estimate(EstimationError::Stats(_))),
            "{error:?}"
        );
        assert_eq!(error.reason_code(), None);
        let fields = error.refusal_fields().expect("the refusal carries fields");
        assert_eq!(fields.stage.as_deref(), Some("design_rank"));
        assert_eq!(fields.subject.as_deref(), Some("effect(t -> y)"));
        assert!(fields.numerical_rank < fields.design_columns);
        assert_eq!(fields.implicated_columns.len(), 1, "{:?}", fields.implicated_columns);
        assert!(["z", "zcopy"].contains(&fields.implicated_columns[0].as_str()));
    };
    named(study.run(&ctx).expect_err("a duplicated column refuses a one-shot run"));
    let prepared = study.prepare(&ctx).unwrap();
    named(prepared.estimate(&data, &ctx).expect_err("and a prepared estimate"));
}
