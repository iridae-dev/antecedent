//! Cluster-aware cross-fitted AIPW (2.2 E4): whole clusters own their folds, the variance is a
//! sandwich over cluster sums (checked against sums written here), no interval is published,
//! and few clusters, a closed independence unit, an interval request and an out-of-scope
//! population are refused with registered reason codes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals and build dense designs element by element"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, ClusterDml, EffectEstimate, EstimationError, IndependenceUnit,
    OverlapPolicy, PropensityNuisance, RidgeTuning, ScoreColumn, ScoreTable, cluster_fold_plan,
    provenance_withholds_interval,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

const FOLDS: usize = 5;

/// Raw columns of one clustered data set; `cluster[i]` is the label of row `i`.
#[derive(Clone)]
struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
    cluster: Vec<u32>,
}

impl Raw {
    fn permuted(&self, order: &[usize]) -> Self {
        let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
        Self {
            t: pick(&self.t),
            y: pick(&self.y),
            z: self.z.iter().map(|c| pick(c)).collect(),
            cluster: order.iter().map(|&i| self.cluster[i]).collect(),
        }
    }
}

/// `groups` clusters of `size` rows. Each cluster has a shared covariate shift, a shared
/// outcome shock and its own treatment effect `2 + u_g` (`u_g ~ N(0, 1)`), so rows of a cluster
/// are dependent. `Y = (2 + u_g) T + z0 + shock + noise`: the true average effect is 2.
fn draw(groups: usize, size: usize, seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE4);
    let n = groups * size;
    let mut raw =
        Raw { t: vec![0.0; n], y: vec![0.0; n], z: vec![vec![0.0; n]; 2], cluster: vec![0; n] };
    for g in 0..groups {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        let effect = 2.0 + standard_normal(&mut rng);
        for k in 0..size {
            let i = g * size + k;
            let z0 = shift + standard_normal(&mut rng);
            let z1 = standard_normal(&mut rng);
            let eta = 0.6 * z0 - 0.4 * z1;
            let t = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
            raw.t[i] = t;
            raw.z[0][i] = z0;
            raw.z[1][i] = z1;
            raw.y[i] = effect * t + z0 + 0.5 * z1 + shock + 0.5 * standard_normal(&mut rng);
            raw.cluster[i] = u32::try_from(g).unwrap();
        }
    }
    raw
}

fn build(raw: &Raw) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = raw.t.len();
    let p = raw.z.len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..p + 2 {
        let (name, hint) = match index {
            0 => ("t".to_string(), RoleHint::TreatmentCandidate),
            1 => ("y".to_string(), RoleHint::OutcomeCandidate),
            j => (format!("z{}", j - 2), RoleHint::Context),
        };
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
    let mut values = vec![raw.t.clone(), raw.y.clone()];
    values.extend(raw.z.iter().cloned());
    let columns = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..p).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

fn declared(raw: &Raw, min_clusters: usize) -> AipwAte {
    AipwAte {
        bootstrap_replicates: 0,
        cluster_ids: Some(raw.cluster.clone()),
        cluster_dml: Some(ClusterDml::new(min_clusters).unwrap()),
        ..AipwAte::new()
    }
}

fn plain() -> AipwAte {
    AipwAte { bootstrap_replicates: 0, ..AipwAte::new() }
}

fn run(est: &AipwAte, raw: &Raw, seed: u64) -> Result<EffectEstimate, EstimationError> {
    let (data, estimand, query) = build(raw);
    let problem = est.prepare(&data, &estimand, &query)?;
    est.fit(
        &problem,
        &mut AipwWorkspace::default(),
        &ExecutionContext::for_tests(seed),
        AssumptionSet::new(),
    )
}

fn table(estimate: &EffectEstimate) -> &ScoreTable {
    estimate.score_table.as_ref().expect("the cluster-DML route keeps its score table")
}

fn code_of(error: &EstimationError, code: &str) {
    assert!(error.to_string().contains(code), "expected {code}, got {error}");
}

/// Hand-computed cluster-sandwich standard error of the contrast scores of `table`:
/// `sqrt(G / (G - 1) * sum_g S_g^2) / n` with `S_g` the cluster sum of `psi_i - mean(psi)`.
fn oracle_cluster_se(table: &ScoreTable, clusters: &[u32]) -> f64 {
    let n = table.n_rows;
    let psi: Vec<f64> = (0..n).map(|i| table.scores[n + i] - table.scores[i]).collect();
    let mean = psi.iter().sum::<f64>() / n as f64;
    let mut sums: BTreeMap<u32, f64> = BTreeMap::new();
    for i in 0..n {
        *sums.entry(clusters[i]).or_insert(0.0) += psi[i] - mean;
    }
    let g = sums.len() as f64;
    let sum_sq: f64 = sums.values().map(|s| s * s).sum();
    (g / (g - 1.0) * sum_sq).sqrt() / n as f64
}

// ---- positive: truth, ownership, replay -----------------------------------------------------

/// Known truth: the cross-fitted point recovers the effect 2, whole clusters share a fold (no
/// cluster crosses folds), the folds are the documented whole-cluster plan, and the same seed
/// replays bit for bit.
#[test]
fn whole_clusters_own_their_folds_and_the_point_recovers_the_truth() {
    let raw = draw(60, 20, 11);
    let est = declared(&raw, 20);
    let a = run(&est, &raw, 5).unwrap();
    assert!((a.ate - 2.0).abs() < 0.6, "ate={}", a.ate);

    let t = table(&a);
    let mut fold_of: BTreeMap<u32, u32> = BTreeMap::new();
    for (&cluster, &fold) in raw.cluster.iter().zip(t.fold_ids.iter()) {
        assert_eq!(
            *fold_of.entry(cluster).or_insert(fold),
            fold,
            "cluster {cluster} crosses folds"
        );
    }
    assert_eq!(fold_of.len(), 60);
    assert_eq!(t.n_folds as usize, FOLDS);
    // Every fold owns clusters, dealt in a balanced way (60 clusters over 5 folds).
    let mut per_fold = [0usize; FOLDS];
    for &fold in fold_of.values() {
        per_fold[fold as usize] += 1;
    }
    assert!(per_fold.iter().all(|&c| c == 12), "{per_fold:?}");
    // The table's folds are the public whole-cluster plan at the recorded seed.
    let plan = cluster_fold_plan(&raw.cluster, FOLDS, a.crossfit_seed.unwrap()).unwrap();
    assert_eq!(t.fold_ids.to_vec(), plan);

    let again = run(&est, &raw, 5).unwrap();
    assert_eq!(a.ate.to_bits(), again.ate.to_bits());
    assert_eq!(t.fold_ids, table(&again).fold_ids);
    assert_eq!(
        t.scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        table(&again).scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// Row order does not move a cluster's fold (exact, at the plan) and permuting the rows of the
/// data together with their labels leaves the estimate unchanged to rounding.
#[test]
fn the_fold_plan_and_the_estimate_are_row_order_invariant() {
    let raw = draw(40, 10, 3);
    let seed = 99;
    let plan = cluster_fold_plan(&raw.cluster, FOLDS, seed).unwrap();
    let order: Vec<usize> = (0..raw.t.len()).map(|i| (i * 7 + 3) % raw.t.len()).collect();
    assert_eq!(
        order.iter().copied().collect::<std::collections::BTreeSet<_>>().len(),
        raw.t.len(),
        "the permutation must be a bijection"
    );
    let permuted = raw.permuted(&order);
    let permuted_plan = cluster_fold_plan(&permuted.cluster, FOLDS, seed).unwrap();
    let folds_by_cluster = |clusters: &[u32], plan: &[u32]| -> BTreeMap<u32, u32> {
        clusters.iter().copied().zip(plan.iter().copied()).collect()
    };
    assert_eq!(
        folds_by_cluster(&raw.cluster, &plan),
        folds_by_cluster(&permuted.cluster, &permuted_plan)
    );
    // Cluster rows' multiplicity does not matter either: the plan depends on distinct labels.
    let doubled: Vec<u32> = raw.cluster.iter().chain(raw.cluster.iter()).copied().collect();
    let doubled_plan = cluster_fold_plan(&doubled, FOLDS, seed).unwrap();
    assert_eq!(folds_by_cluster(&raw.cluster, &plan), folds_by_cluster(&doubled, &doubled_plan));

    let est = declared(&raw, 20);
    let a = run(&est, &raw, seed).unwrap();
    let b = run(&declared(&permuted, 20), &permuted, seed).unwrap();
    assert!((a.ate - b.ate).abs() < 1e-7, "{} vs {}", a.ate, b.ate);
}

/// The published result is a point and a score table only; the table names whole-cluster
/// folds; the wire form round-trips and keeps the marker.
#[test]
fn a_cluster_dml_fit_publishes_a_point_and_scores_but_no_interval() {
    let raw = draw(30, 10, 7);
    let est = declared(&raw, 10);
    let fit = run(&est, &raw, 2).unwrap();
    assert!(fit.ate.is_finite());
    assert!(fit.se_analytic.is_nan());
    assert!(fit.se_kind.is_none());
    assert!(fit.joint_covariance.is_none());
    assert!(fit.score_inference.is_none());
    assert!(fit.influence.is_none());
    assert_eq!(fit.crossfit_folds, Some(FOLDS));
    assert!(fit.crossfit_seed.is_some());

    let t = table(&fit);
    // The point is the score-contrast mean exactly.
    let n = t.n_rows;
    let mean: f64 = (0..n).map(|i| t.scores[n + i] - t.scores[i]).sum::<f64>() / n as f64;
    assert!((fit.ate - mean).abs() < 1e-12);
}

/// The score table names whole-cluster folds, the cluster count and a label fingerprint in its
/// provenance (which keeps its iid summaries unpublished), round-trips its wire form, and
/// differs from an unclustered table over the same rows.
#[test]
fn a_cluster_dml_score_table_round_trips_with_its_unit_identity() {
    let raw = draw(30, 10, 7);
    let fit = run(&declared(&raw, 10), &raw, 2).unwrap();
    let t = table(&fit);
    assert!(t.nuisance_provenance.contains(";cluster_dml=unit=cluster;clusters=30;digest="));
    assert!(provenance_withholds_interval(&t.nuisance_provenance));
    let wire = t.to_wire();
    assert_eq!(&ScoreTable::from_wire(wire.clone()).unwrap(), t);
    assert!(wire.nuisance_provenance.contains(";cluster_dml="));
    // The unclustered table carries no marker and draws different folds.
    let iid = run(&plain(), &raw, 2).unwrap();
    assert!(!provenance_withholds_interval(&table(&iid).nuisance_provenance));
    assert_ne!(table(&iid).fold_ids, t.fold_ids);
    // Other labels over the same rows give another fingerprint.
    let relabeled = Raw { cluster: raw.cluster.iter().map(|c| c + 1_000).collect(), ..raw.clone() };
    let other = run(&declared(&relabeled, 10), &relabeled, 2).unwrap();
    assert_ne!(table(&other).nuisance_provenance, t.nuisance_provenance);
}

// ---- variance: independent oracle and closed form -------------------------------------------

/// The receipt's cluster-sandwich standard error equals cluster sums written in this test from
/// the table's own scores, and exceeds the iid standard error when clusters share a shock.
#[test]
fn the_cluster_sandwich_equals_hand_computed_cluster_sums() {
    let raw = draw(60, 20, 21);
    let est = declared(&raw, 20);
    let fit = run(&est, &raw, 4).unwrap();
    let t = table(&fit);
    let receipt = est.cluster_dml.unwrap().receipt(t, &raw.cluster).unwrap();
    let oracle = oracle_cluster_se(t, &raw.cluster);
    assert!(
        (receipt.cluster_sandwich_se - oracle).abs() <= 1e-12 * oracle,
        "{} vs {oracle}",
        receipt.cluster_sandwich_se
    );
    assert_eq!(receipt.independence_unit, IndependenceUnit::Cluster);
    assert_eq!((receipt.n_clusters, receipt.n_rows, receipt.folds), (60, 1_200, FOLDS));
    assert_eq!(receipt.min_clusters, 20);
    assert_eq!(receipt.unit_digest.len(), 16);
    // The digest is a function of the label set, not of row order.
    let reversed: Vec<u32> = raw.cluster.iter().rev().copied().collect();
    let flipped = ScoreTable {
        fold_ids: t.fold_ids.iter().rev().copied().collect::<Vec<_>>().into(),
        scores: {
            let n = t.n_rows;
            (0..2)
                .flat_map(|c| (0..n).rev().map(move |i| (c, i)))
                .map(|(c, i)| t.scores[c * n + i])
                .collect::<Vec<_>>()
                .into()
        },
        ..t.clone()
    };
    let flipped_receipt = est.cluster_dml.unwrap().receipt(&flipped, &reversed).unwrap();
    assert_eq!(flipped_receipt.unit_digest, receipt.unit_digest);
    assert!((flipped_receipt.cluster_sandwich_se - receipt.cluster_sandwich_se).abs() < 1e-12);

    // Shared cluster shocks make the cluster-sandwich standard error larger than the iid one.
    let n = t.n_rows as f64;
    let psi: Vec<f64> = (0..t.n_rows).map(|i| t.scores[t.n_rows + i] - t.scores[i]).collect();
    let mean = psi.iter().sum::<f64>() / n;
    let iid_se = (psi.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    assert!(
        receipt.cluster_sandwich_se > 1.3 * iid_se,
        "{} vs {iid_se}",
        receipt.cluster_sandwich_se
    );
}

/// A table whose per-row contrast scores are exactly `+-1` by cluster has a closed-form
/// cluster variance: 12 clusters of 5 rows, `psi = c_g = (-1)^g`, mean 0, `S_g = +-5`,
/// `sum S_g^2 = 300`, so `Var = (12 / 11) * 300 / 60^2 = 1 / 11`.
#[test]
fn the_cluster_variance_has_its_closed_form_on_deterministic_scores() {
    let (groups, size) = (12usize, 5usize);
    let n = groups * size;
    let clusters: Vec<u32> = (0..n).map(|i| u32::try_from(i / size).unwrap()).collect();
    let psi: Vec<f64> = (0..n).map(|i| if (i / size) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut scores = vec![0.0; n];
    scores.extend(psi.iter().copied());
    let table = ScoreTable {
        observed_arm: vec![0; n].into(),
        propensities: vec![0.5; 2 * n].into(),
        observed_outcome: vec![0.0; n].into(),
        n_rows: n,
        row_index: (0..u32::try_from(n).unwrap()).collect::<Vec<_>>().into(),
        fold_ids: clusters.iter().map(|c| c % 5).collect::<Vec<_>>().into(),
        n_folds: 5,
        scores: scores.into(),
        columns: Arc::from([
            ScoreColumn { arm: 0, threshold: None },
            ScoreColumn { arm: 1, threshold: None },
        ]),
        adjustment_set: Arc::from([]),
        nuisance_provenance: Arc::from("closed-form fixture"),
        propensity_clip: None,
        treatment: VariableId::from_raw(0),
        intervened: Arc::from([]),
    };
    let spec = ClusterDml::new(10).unwrap();
    let receipt = spec.receipt(&table, &clusters).unwrap();
    let expected = (1.0_f64 / 11.0).sqrt();
    assert!(
        (receipt.cluster_sandwich_se - expected).abs() < 1e-14,
        "{} vs {expected}",
        receipt.cluster_sandwich_se
    );
    // With every row its own cluster the same scores give the HC1-type iid value instead.
    let singletons: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
    let mut one_per_row = table.clone();
    one_per_row.fold_ids = singletons.iter().map(|c| c % 5).collect::<Vec<_>>().into();
    let iid = spec.receipt(&one_per_row, &singletons).unwrap();
    let iid_expected = ((n as f64 / (n as f64 - 1.0)) * n as f64).sqrt() / n as f64;
    assert!((iid.cluster_sandwich_se - iid_expected).abs() < 1e-14);
    assert!(receipt.cluster_sandwich_se > 2.0 * iid.cluster_sandwich_se);
}

// ---- negative: dependence, few clusters, closed options -------------------------------------

/// An IID cross-fit over the same data splits clusters across folds: the rows of a cluster are
/// not independent of the nuisances that score them, and the receipt refuses such a table.
#[test]
fn iid_row_folds_split_clusters_and_are_refused_as_cluster_evidence() {
    let raw = draw(40, 10, 5);
    let iid = run(&plain(), &raw, 3).unwrap();
    let t = table(&iid);
    let mut seen: BTreeMap<u32, std::collections::BTreeSet<u32>> = BTreeMap::new();
    for (&cluster, &fold) in raw.cluster.iter().zip(t.fold_ids.iter()) {
        seen.entry(cluster).or_default().insert(fold);
    }
    assert!(seen.values().any(|folds| folds.len() > 1), "an iid plan should split a cluster");
    let error = ClusterDml::new(20).unwrap().receipt(t, &raw.cluster).unwrap_err();
    assert!(error.to_string().contains("split across cross-fit folds"), "{error}");
}

#[test]
fn few_clusters_are_refused_before_any_fit() {
    let raw = draw(9, 40, 8);
    let error = run(&declared(&raw, 10), &raw, 1).unwrap_err();
    code_of(&error, "too_few_clusters");
    assert!(error.to_string().contains("9 clusters"), "{error}");
    // Above the floor but below a stricter declaration.
    let raw = draw(24, 20, 8);
    let error = run(&declared(&raw, 30), &raw, 1).unwrap_err();
    code_of(&error, "too_few_clusters");
    assert!(run(&declared(&raw, 24), &raw, 1).is_ok());
    // The receipt applies the same minimum.
    let fit = run(&declared(&raw, 24), &raw, 1).unwrap();
    let error = ClusterDml::new(30).unwrap().receipt(table(&fit), &raw.cluster).unwrap_err();
    code_of(&error, "too_few_clusters");
    // A declaration below the floor is invalid.
    code_of(&ClusterDml::new(9).unwrap_err(), "invalid_argument");
}

#[test]
fn a_dyadic_unit_is_declarable_and_closed() {
    let raw = draw(30, 10, 9);
    let est = AipwAte { cluster_dml: Some(ClusterDml::dyadic(20)), ..declared(&raw, 20) };
    assert_eq!(est.cluster_dml.unwrap().independence_unit(), IndependenceUnit::Dyad);
    code_of(&run(&est, &raw, 1).unwrap_err(), "dyadic_dependence_not_licensed");
    let fit = run(&declared(&raw, 20), &raw, 1).unwrap();
    code_of(
        &ClusterDml::dyadic(20).receipt(table(&fit), &raw.cluster).unwrap_err(),
        "dyadic_dependence_not_licensed",
    );
    assert_ne!(
        ClusterDml::dyadic(20).canonical_key(),
        ClusterDml::new(20).unwrap().canonical_key()
    );
    assert_ne!(
        ClusterDml::new(20).unwrap().canonical_key(),
        ClusterDml::new(25).unwrap().canonical_key()
    );
}

#[test]
fn an_interval_request_is_refused_with_its_reason_code() {
    let raw = draw(30, 10, 2);
    let (data, estimand, query) = build(&raw);
    for requested in [
        AipwAte { bootstrap_replicates: 50, ..declared(&raw, 20) },
        AipwAte { se_kind: antecedent_estimate::AnalyticSeKind::Cluster, ..declared(&raw, 20) },
    ] {
        let error = requested.prepare(&data, &estimand, &query).unwrap_err();
        code_of(&error, "cluster_interval_not_licensed");
    }
    let est = declared(&raw, 20);
    let problem = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut workspace = AipwWorkspace::default();
    let point = est.fit(&problem, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
    let boot = AipwAte { bootstrap_replicates: 5, ..est };
    let error = boot.attach_bootstrap(&problem, &mut workspace, &ctx, point).unwrap_err();
    code_of(&error, "cluster_interval_not_licensed");
}

#[test]
fn scope_labels_and_combinations_outside_the_license_are_refused() {
    let raw = draw(30, 10, 4);
    let (data, estimand, query) = build(&raw);
    let trimmed = AipwAte {
        overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) },
        ..declared(&raw, 20)
    };
    code_of(&trimmed.prepare(&data, &estimand, &query).unwrap_err(), "route_not_supported");

    let penalized = AipwAte {
        propensity: PropensityNuisance::ridge_logistic(RidgeTuning::new(&[1.0], 3).unwrap()),
        ..declared(&raw, 20)
    };
    code_of(&penalized.prepare(&data, &estimand, &query).unwrap_err(), "route_not_supported");

    let unlabeled = AipwAte { cluster_ids: None, ..declared(&raw, 20) };
    code_of(&unlabeled.prepare(&data, &estimand, &query).unwrap_err(), "required_option_missing");

    let mut short = raw.cluster.clone();
    short.pop();
    let misaligned = AipwAte { cluster_ids: Some(short), ..declared(&raw, 20) };
    let error = misaligned.prepare(&data, &estimand, &query).unwrap_err();
    assert!(error.to_string().contains("cluster_ids length"), "{error}");

    // A problem prepared for one declaration is not fit under another.
    let problem = declared(&raw, 20).prepare(&data, &estimand, &query).unwrap();
    let error = plain()
        .fit(
            &problem,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("cluster-DML declaration"), "{error}");
}
