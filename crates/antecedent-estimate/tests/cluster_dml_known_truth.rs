//! Repeated-sampling known-truth test of the clustered-DML sandwich standard error.
//!
//! A TEST with fixed seeds, not a coverage record. Clustered data carry a cluster-level random
//! effect on the outcome, on a covariate and on the treatment effect; the population average
//! effect is 2. For 300 replications the cluster-DML AIPW fit's receipt standard error
//! (`ClusterDml::receipt`) gives a 95% Wald band, and the test asserts that its coverage is
//! within three binomial standard errors of 0.95 under the declared few-cluster `t_{G-1}`
//! reference, that the same fit's iid standard error (the plug-in score covariance that
//! ignores clusters) undercovers by more than that tolerance on the same data, and that the
//! published `se_analytic` is the receipt value. The normal-reference coverage of the same
//! sandwich is printed so a finite-cluster shortfall is visible rather than absorbed into a
//! looser tolerance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "test fixtures index small literals and convert replicate counts for fractions"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AipwAte, AipwWorkspace, ClusterDml};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;
use antecedent_stats::student_t_ppf;

const REPS: u64 = 300;
const GROUPS: usize = 40;
const SIZE: usize = 10;
const TRUTH: f64 = 2.0;
const NORMAL_975: f64 = 1.959_963_984_540_054;

struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z0: Vec<f64>,
    z1: Vec<f64>,
    cluster: Vec<u32>,
}

/// `Y = (2 + 1.5 u_g) T + z0 + 0.5 z1 + s_g + 0.5 e`, `z0` shifted by a cluster effect and
/// treatment assigned from `z0`, `z1`: rows of a cluster share `u_g`, `s_g` and the shift.
fn draw(seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE4C);
    let n = GROUPS * SIZE;
    let mut raw = Raw {
        t: vec![0.0; n],
        y: vec![0.0; n],
        z0: vec![0.0; n],
        z1: vec![0.0; n],
        cluster: vec![0; n],
    };
    for g in 0..GROUPS {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        let effect = TRUTH + 1.5 * standard_normal(&mut rng);
        for k in 0..SIZE {
            let i = g * SIZE + k;
            let z0 = shift + standard_normal(&mut rng);
            let z1 = standard_normal(&mut rng);
            let eta = 0.6 * z0 - 0.4 * z1;
            let t = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
            raw.t[i] = t;
            raw.z0[i] = z0;
            raw.z1[i] = z1;
            raw.y[i] = effect * t + z0 + 0.5 * z1 + shock + 0.5 * standard_normal(&mut rng);
            raw.cluster[i] = u32::try_from(g).unwrap();
        }
    }
    raw
}

fn build(raw: &Raw) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = raw.t.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z0", RoleHint::Context),
        ("z1", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = [&raw.t, &raw.y, &raw.z0, &raw.z1]
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

#[test]
fn the_cluster_sandwich_receipt_covers_the_known_effect_and_the_iid_se_does_not() {
    let tolerance = 3.0 * (0.95_f64 * 0.05 / REPS as f64).sqrt();
    // The few-cluster reference the guide declares: t with G - 1 degrees of freedom.
    let t_multiplier = student_t_ppf(0.975, (GROUPS - 1) as f64);
    let (mut cover_t, mut cover_normal, mut cover_iid) = (0_u64, 0_u64, 0_u64);
    let (mut se_cluster, mut se_iid) = (0.0, 0.0);
    for rep in 0..REPS {
        let raw = draw(9_000 + rep);
        let (data, estimand, query) = build(&raw);
        let spec = ClusterDml::new(20).unwrap();
        let est = AipwAte {
            bootstrap_replicates: 0,
            cluster_ids: Some(raw.cluster.clone()),
            cluster_dml: Some(spec),
            ..AipwAte::new()
        };
        let problem = est.prepare(&data, &estimand, &query).unwrap();
        let fit = est
            .fit(
                &problem,
                &mut AipwWorkspace::default(),
                &ExecutionContext::for_tests(rep),
                AssumptionSet::new(),
            )
            .unwrap();
        let table = fit.score_table.as_ref().expect("the cluster-DML route keeps its table");
        let receipt = spec.receipt(table, &raw.cluster, None).unwrap();
        assert_eq!(receipt.n_clusters, GROUPS);
        assert_eq!(receipt.reference_df, GROUPS - 1);
        // The published SE is the receipt value.
        assert!((fit.se_analytic - receipt.cluster_sandwich_se).abs() < 1e-12);
        // The published reference is the declared t_(G-1).
        assert_eq!(fit.se_reference_df, Some((GROUPS - 1) as f64));
        // The iid SE of the same scores: the contrast of the table's arm columns ignoring clusters.
        let summary = table.summarize(None).unwrap();
        let iid = table.linear_contrast(&summary, &[-1.0, 1.0]).unwrap();
        let miss = (fit.ate - TRUTH).abs();
        cover_t += u64::from(miss <= t_multiplier * receipt.cluster_sandwich_se);
        cover_normal += u64::from(miss <= NORMAL_975 * receipt.cluster_sandwich_se);
        cover_iid += u64::from(miss <= NORMAL_975 * iid.se);
        se_cluster += receipt.cluster_sandwich_se;
        se_iid += iid.se;
    }
    let reps = REPS as f64;
    let (coverage_t, coverage_normal, coverage_iid) =
        (cover_t as f64 / reps, cover_normal as f64 / reps, cover_iid as f64 / reps);
    eprintln!(
        "cluster_dml_known_truth reps={REPS} groups={GROUPS} size={SIZE} \
         coverage_t={coverage_t} coverage_normal={coverage_normal} coverage_iid={coverage_iid} \
         mean_se_cluster={} mean_se_iid={} t_multiplier={t_multiplier}",
        se_cluster / reps,
        se_iid / reps
    );
    assert!(
        (coverage_t - 0.95).abs() <= tolerance,
        "t_(G-1) sandwich coverage {coverage_t} is not within {tolerance} of 0.95 \
         (normal reference: {coverage_normal})"
    );
    assert!(
        coverage_iid < 0.95 - tolerance,
        "the iid SE should undercover on clustered data, got {coverage_iid}"
    );
    assert!(se_cluster > se_iid, "the cluster SE must exceed the iid SE under a cluster effect");
}
