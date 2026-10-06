//! Structured refusal fields at the existing refusal sites (2.2 preflight cell, completion).
//!
//! Each site refuses with the same message (pinned byte for byte below, as at the commit the
//! fields were added) and now also carries `RefusalFields`: stage, reason, remedy and the
//! numbers its failing step produced. A figure the step did not compute stays absent, never a
//! zero invented from a failed fit. Every numeric expectation is written out by hand here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::{
    ClusterDml, EstimationError, ExactF64, RefusalFields, RetargetRefusal, WeightedSupport,
    weighted_support_fields,
};
use antecedent_stats::{
    FaerBackend, FitDiagnostics, GlmFit, GlmOptions, PropensityWorkspace, SandwichKind, StatsError,
    coefficient_covariance, fit_propensity,
};

const NOT_CONVERGED: &str =
    "backend error: GLM IRLS did not converge; refuse propensity/outcome scores";
const SEPARATED: &str =
    "backend error: GLM indicates (quasi-)complete separation; refuse propensity/outcome scores";
const SATURATED: &str = "backend error: GLM fitted probabilities lie within 1e-8 of 0 or 1 \
                         (extreme scores, not necessarily separation); refuse \
                         propensity/outcome scores";

fn glm_fit(
    iterations: u32,
    converged: bool,
    separated: bool,
    saturated: bool,
    measured: Option<(f64, u64)>,
) -> GlmFit {
    GlmFit {
        coefficients: vec![0.0],
        iterations,
        converged,
        separated,
        boundary_saturated: saturated,
        boundary_margin: measured.map(|(margin, _)| margin),
        boundary_count: measured.map(|(_, count)| count),
        penalized: false,
        deviance: 0.0,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(1, None, "glm-irls", 0),
    }
}

fn refusal(fit: &GlmFit) -> EstimationError {
    EstimationError::from(fit.require_ok().unwrap_err())
}

/// Fields that only a fitted nuisance, a cluster count or a rank could fill stay empty.
fn assert_no_overlap_or_cluster_facts(fields: &RefusalFields) {
    assert!(fields.arm_ess.is_empty() && fields.propensity_min.is_none());
    assert!(fields.propensity_max.is_none() && fields.propensity_quantiles.is_empty());
    assert!(fields.cluster_count.is_none() && fields.cluster_minimum.is_none());
    assert!(fields.numerical_rank.is_none() && fields.design_columns.is_none());
    assert!(fields.implicated_columns.is_empty() && fields.subject.is_none());
}

#[test]
fn a_glm_that_did_not_converge_keeps_its_text_and_reports_the_iterations() {
    let error = refusal(&glm_fit(50, false, false, false, Some((0.25, 0))));
    assert_eq!(error.to_string(), NOT_CONVERGED);
    assert!(matches!(error, EstimationError::Stats(_)), "the variant is unchanged: {error:?}");
    let fields = error.refusal_fields().expect("a refused GLM carries fields");
    assert_eq!(fields.stage.as_deref(), Some("glm_fit"));
    assert_eq!(fields.reason.as_deref(), Some("non_converged"));
    assert!(fields.remedy.is_some());
    assert_eq!(fields.glm_iterations, Some(50));
    assert_eq!(fields.boundary_margin, Some(ExactF64(0.25)));
    assert_eq!(fields.boundary_count, Some(0));
    assert_no_overlap_or_cluster_facts(&fields);
}

#[test]
fn a_separated_glm_keeps_its_text_and_reports_the_margin_and_count() {
    let error = refusal(&glm_fit(7, true, true, true, Some((3.0e-12, 17))));
    assert_eq!(error.to_string(), SEPARATED);
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("glm_fit"));
    assert_eq!(fields.reason.as_deref(), Some("separated"));
    assert_eq!(fields.glm_iterations, Some(7));
    assert_eq!(fields.boundary_margin, Some(ExactF64(3.0e-12)));
    assert_eq!(fields.boundary_count, Some(17));
    assert_no_overlap_or_cluster_facts(&fields);
}

#[test]
fn a_boundary_saturated_glm_keeps_its_text_and_reports_the_margin_and_count() {
    let error = refusal(&glm_fit(9, true, false, true, Some((1.0e-10, 4))));
    assert_eq!(error.to_string(), SATURATED);
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.reason.as_deref(), Some("boundary_saturated"));
    assert_eq!(fields.glm_iterations, Some(9));
    assert_eq!(fields.boundary_margin, Some(ExactF64(1.0e-10)));
    assert_eq!(fields.boundary_count, Some(4));
}

/// A family that does not measure the probability margin (the fit has none) leaves it absent
/// rather than reporting zero.
#[test]
fn an_unmeasured_margin_is_absent_not_zero() {
    let error = refusal(&glm_fit(50, false, false, false, None));
    assert_eq!(error.to_string(), NOT_CONVERGED);
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.glm_iterations, Some(50));
    assert!(fields.boundary_margin.is_none() && fields.boundary_count.is_none());
}

/// An accepted fit refuses nothing, and a stats error that is not a refusal record has no
/// fields.
#[test]
fn only_refusal_records_derive_fields() {
    assert!(glm_fit(3, true, false, false, Some((0.4, 0))).require_ok().is_ok());
    let backend = EstimationError::stats_msg("backend text");
    assert!(backend.refusal_fields().is_none());
    assert_eq!(backend.to_string(), "backend error: backend text");
    assert!(EstimationError::data_msg("x").refusal_fields().is_none());
}

/// The real strict propensity fit on completely separated data refuses through the same
/// choke point; the facts are the fit's own, whichever defect the IRLS reports first.
#[test]
fn a_real_separated_propensity_fit_refuses_with_the_glm_record() {
    let n = 40usize;
    let z: Vec<f64> = (0..n).map(|i| i as f64 - 19.5).collect();
    let t: Vec<f64> = z.iter().map(|&v| f64::from(v > 0.0)).collect();
    let mut design = vec![1.0; n];
    design.extend_from_slice(&z);
    let error = fit_propensity(
        &design,
        n,
        2,
        &t,
        &FaerBackend,
        &mut PropensityWorkspace::default(),
        &GlmOptions::default(),
    )
    .unwrap_err();
    let converted = EstimationError::from(error.clone());
    assert_eq!(converted.to_string(), error.to_string());
    let fields = converted.refusal_fields().expect("the fit's refusal carries fields");
    match error {
        StatsError::GlmRefused { iterations, .. } => {
            let reason = fields.reason.as_deref().unwrap();
            assert!(["non_converged", "separated", "boundary_saturated"].contains(&reason));
            assert_eq!(fields.stage.as_deref(), Some("glm_fit"));
            assert_eq!(fields.glm_iterations, Some(u64::from(iterations)));
            assert!(fields.boundary_margin.is_some() && fields.boundary_count.is_some());
        }
        StatsError::RankDeficient { rank, ncols } => {
            assert_eq!(fields.stage.as_deref(), Some("design_rank"));
            assert_eq!(fields.numerical_rank, Some(rank as u64));
            assert_eq!(fields.design_columns, Some(ncols as u64));
        }
        other => panic!("unexpected refusal {other:?}"),
    }
}

/// The backend's rank refusal names only counts; those counts are the fields, and the
/// implicated columns, which the backend does not know, stay empty.
#[test]
fn a_rank_deficient_design_reports_rank_and_columns_but_names_no_column() {
    let error = EstimationError::from(StatsError::RankDeficient { rank: 174, ncols: 175 });
    assert_eq!(error.to_string(), "rank deficient: rank=174 ncols=175");
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("design_rank"));
    assert_eq!(fields.reason.as_deref(), Some("rank_deficient"));
    assert_eq!((fields.numerical_rank, fields.design_columns), (Some(174), Some(175)));
    assert!(fields.implicated_columns.is_empty(), "the backend cannot name a column");
    assert!(fields.remedy.is_some());
    assert!(fields.arm_ess.is_empty() && fields.propensity_min.is_none());
}

/// A GLM cluster sandwich over one cluster refuses with the text it always had, now with the
/// count and the minimum.
#[test]
fn a_glm_cluster_sandwich_with_one_cluster_reports_the_counts() {
    let x = vec![1.0; 6];
    let residuals = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    let groups = [7u32; 6];
    let error =
        coefficient_covariance(&x, 6, 1, &residuals, SandwichKind::Cluster { groups: &groups })
            .unwrap_err();
    let error = EstimationError::from(error);
    assert_eq!(
        error.to_string(),
        "shape error: cluster-robust variance requires at least 2 clusters"
    );
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("cluster_variance"));
    assert_eq!(fields.reason.as_deref(), Some("too_few_clusters"));
    assert_eq!((fields.cluster_count, fields.cluster_minimum), (Some(1), Some(2)));
    assert!(fields.remedy.is_some());
    assert!(fields.arm_ess.is_empty() && fields.propensity_min.is_none());
}

/// Multiway: the failing subset's own count is reported.
#[test]
fn a_multiway_sandwich_reports_the_failing_dimension_count() {
    let x = vec![1.0; 6];
    let residuals = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    let one = [4u32; 6];
    let three = [0u32, 1, 2, 0, 1, 2];
    let dims: [&[u32]; 2] = [&one, &three];
    let error =
        coefficient_covariance(&x, 6, 1, &residuals, SandwichKind::Multiway { dimensions: &dims })
            .unwrap_err();
    let error = EstimationError::from(error);
    assert_eq!(
        error.to_string(),
        "shape error: cluster-robust variance requires at least 2 clusters"
    );
    let fields = error.refusal_fields().unwrap();
    assert_eq!((fields.cluster_count, fields.cluster_minimum), (Some(1), Some(2)));
}

/// The cluster-DML shortfall is a coded refusal; its message is unchanged and the fields hold
/// the found and the declared minimum counts.
#[test]
fn cluster_dml_too_few_clusters_reports_found_and_minimum() {
    let ids: Vec<u32> = (0..20u32).map(|i| i % 5).collect();
    let error =
        ClusterDml::new(10).unwrap().declare_units(Some(ids.as_slice()), None, 20).unwrap_err();
    assert_eq!(
        error.to_string(),
        "reason=too_few_clusters: cluster_dml.too_few_clusters: 5 clusters are below the \
         declared minimum 10; a sandwich over cluster sums needs many independent clusters and \
         no interval or standard error is formed"
    );
    let fields = error.refusal_fields().unwrap();
    assert_eq!(fields.stage.as_deref(), Some("cluster_dml"));
    assert_eq!(fields.reason.as_deref(), Some("too_few_clusters"));
    assert_eq!((fields.cluster_count, fields.cluster_minimum), (Some(5), Some(10)));
    assert!(fields.remedy.is_some() && fields.arm_ess.is_empty());
}

#[test]
fn retarget_dependence_refusals_keep_their_text_and_carry_stage_reason_remedy() {
    let cases = [
        (
            RetargetRefusal::IllegalDependence,
            "retarget depends_on must not include the treatment, an intervened coordinate, or a descendant",
            "weights_depend_on_treatment_or_descendant",
        ),
        (
            RetargetRefusal::OutsideAdjustmentSet,
            "retarget weights must be a function of the certified adjustment set",
            "weights_outside_adjustment_set",
        ),
        (
            RetargetRefusal::DescendantClosureUnavailable,
            "retarget depends_on descendant closure requires a directed graph (DAG or ADMG); refusing rather than skipping the descendant check",
            "descendant_closure_unavailable",
        ),
        (
            RetargetRefusal::UndeclaredNonconstantWeights,
            "nonempty depends_on is required for nonconstant target weights",
            "undeclared_nonconstant_weights",
        ),
    ];
    for (refusal, text, reason) in cases {
        assert_eq!(refusal.as_str(), text);
        let fields = refusal.fields();
        assert_eq!(fields.stage.as_deref(), Some("retarget"));
        assert_eq!(fields.reason.as_deref(), Some(reason));
        assert!(fields.remedy.is_some());
        // No score exists at this stage: nothing numeric is reported, nothing is zero.
        assert_no_overlap_or_cluster_facts(&fields);
    }
}

const WEIGHTED_OVERLAP: &str =
    "retarget refused: weighted overlap failed under the declared target weights";

#[test]
fn a_failed_overlap_gate_reports_the_arm_sizes_and_propensity_range_it_measured() {
    assert_eq!(RetargetRefusal::WeightedOverlap.as_str(), WEIGHTED_OVERLAP);
    // Treated arm effectively 3 rows: below the 10-row minimum.
    let support = WeightedSupport {
        n_eff: 43.0,
        n_eff_by_arm: vec![40.0, 3.0],
        propensity_range: Some((0.2, 0.9)),
        overlap_ok: false,
    };
    let fields = weighted_support_fields(&[0, 1], &support);
    assert_eq!(fields.stage.as_deref(), Some("retarget"));
    assert_eq!(fields.reason.as_deref(), Some("arm_effective_sample_size_below_minimum"));
    assert_eq!(
        fields.arm_ess,
        vec![("control".to_string(), ExactF64(40.0)), ("active".to_string(), ExactF64(3.0))]
    );
    assert_eq!(fields.propensity_min, Some(ExactF64(0.2)));
    assert_eq!(fields.propensity_max, Some(ExactF64(0.9)));
    assert!(fields.propensity_quantiles.is_empty(), "the support holds no quantiles");
    assert!(fields.remedy.is_some());

    // Range touching 0 or 1 with healthy arms.
    let support = WeightedSupport {
        n_eff: 100.0,
        n_eff_by_arm: vec![50.0, 50.0],
        propensity_range: Some((1e-9, 0.9)),
        overlap_ok: false,
    };
    let fields = weighted_support_fields(&[0, 1], &support);
    assert_eq!(fields.reason.as_deref(), Some("propensity_range_touches_zero_or_one"));

    // Healthy arms and range: the extreme-share condition is what failed.
    let support = WeightedSupport {
        n_eff: 100.0,
        n_eff_by_arm: vec![50.0, 50.0],
        propensity_range: Some((0.05, 0.9)),
        overlap_ok: false,
    };
    let fields = weighted_support_fields(&[0, 1], &support);
    assert_eq!(fields.reason.as_deref(), Some("extreme_propensity_share_above_limit"));

    // Joint cells are labelled by their arm, not as control/active.
    let support = WeightedSupport {
        n_eff: 100.0,
        n_eff_by_arm: vec![30.0, 30.0, 2.0, 30.0],
        propensity_range: None,
        overlap_ok: false,
    };
    let fields = weighted_support_fields(&[0, 1, 2, 3], &support);
    let labels: Vec<&str> = fields.arm_ess.iter().map(|(label, _)| label.as_str()).collect();
    assert_eq!(labels, ["arm 0", "arm 1", "arm 2", "arm 3"]);
    assert!(fields.propensity_min.is_none() && fields.propensity_max.is_none());
}

/// A refusal raised before any arm support or fitted score exists has those entries absent,
/// not zero.
#[test]
fn an_overlap_refusal_without_a_computed_support_has_arm_and_propensity_facts_absent() {
    let support = WeightedSupport {
        n_eff: 12.0,
        n_eff_by_arm: Vec::new(),
        propensity_range: None,
        overlap_ok: false,
    };
    let fields = weighted_support_fields(&[0, 1], &support);
    assert_eq!(fields.reason.as_deref(), Some("arm_support_not_computed"));
    assert!(fields.arm_ess.is_empty(), "{:?}", fields.arm_ess);
    assert!(fields.propensity_min.is_none() && fields.propensity_max.is_none());
    assert!(fields.propensity_quantiles.is_empty());
}

/// The additive variants render exactly as the variants they extend.
#[test]
fn field_carrying_variants_render_as_their_plain_counterparts() {
    let plain = EstimationError::unsupported("same text");
    let rich = EstimationError::unsupported_with_fields("same text", RefusalFields::default());
    assert_eq!(plain.to_string(), rich.to_string());
    assert!(plain.refusal_fields().is_none() && rich.refusal_fields().is_some());
    assert_eq!(rich.remedy(), None);
    assert!(rich.refusal_fields().unwrap().remedy.is_none());
}
