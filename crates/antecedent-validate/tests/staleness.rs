//! Staleness and drift diagnostics: synthetic shifts of known size, the false-alarm rate over
//! null seeds, every recommendation branch, artifact-only consumption, determinism across
//! threads, and refusal by a tiny admission pool.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "synthetic fixtures convert small counts and indices directly"
)]

use std::sync::Arc;

use antecedent_core::execution::AdmissionPool;
use antecedent_core::recalc::{Branch, Stage};
use antecedent_core::{
    CategoryDomainId, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, OwnedColumn,
    OwnedColumnarStorage, TableView, TabularData, UnknownCategoryPolicy, ValidityBitmap,
};
use antecedent_kernels::{sigmoid, standard_normal};
use antecedent_validate::staleness::{
    FindingKind, Recommendation, StalenessBaseline, StalenessError, StalenessPlan, StalenessPolicy,
    StalenessReport, assess_exported_baseline,
};

/// A synthetic design: `z1` on the grid {-2..2}, `z2` standard normal, a logistic treatment
/// and a linear outcome.
#[derive(Clone)]
struct Scenario {
    n: usize,
    seed: u64,
    z1_weights: [f64; 5],
    t_slope: f64,
    effect: f64,
    missing_z2: f64,
    treatment_levels: u32,
    periods: Option<(i64, usize, f64)>,
}

impl Scenario {
    fn null(n: usize, seed: u64) -> Self {
        Self {
            n,
            seed,
            z1_weights: [0.2; 5],
            t_slope: 0.3,
            effect: 2.0,
            missing_z2: 0.0,
            treatment_levels: 2,
            periods: None,
        }
    }

    fn columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        let mut rng = CausalRng::from_seed(self.seed);
        let (mut z1, mut z2, mut t, mut y, mut period) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let per_period = self.periods.map_or(self.n, |(_, count, _)| self.n.div_ceil(count));
        for i in 0..self.n {
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut level = 4usize;
            for (k, w) in self.z1_weights.iter().enumerate() {
                acc += w;
                if u < acc {
                    level = k;
                    break;
                }
            }
            let a = level as f64 - 2.0;
            let block = i / per_period;
            let drift = match self.periods {
                Some((_, count, shift)) if block >= count / 2 => shift,
                _ => 0.0,
            };
            let b = standard_normal(&mut rng) + drift;
            let e = sigmoid(0.4 * a + self.t_slope * b);
            let mut treatment = if rng.next_f64() < e { 1.0 } else { 0.0 };
            if self.treatment_levels > 2 && rng.next_f64() < 0.1 {
                treatment = 2.0;
            }
            let outcome = 1.0 + self.effect * treatment + a + 0.5 * b + standard_normal(&mut rng);
            let missing = rng.next_f64() < self.missing_z2;
            z1.push(a);
            z2.push(if missing { f64::NAN } else { b });
            t.push(treatment);
            y.push(outcome);
            if let Some((start, _, _)) = self.periods {
                period.push((start + block as i64) as f64);
            }
        }
        let mut columns = vec![("z1", z1), ("z2", z2), ("t", t), ("y", y)];
        if self.periods.is_some() {
            columns.push(("period", period));
        }
        columns
    }

    fn data(&self) -> TabularData {
        let columns = self.columns();
        TabularData::from_f64_columns(columns.iter().map(|(n, v)| (*n, v.as_slice()))).unwrap()
    }
}

fn plan(bayesian: bool) -> StalenessPlan {
    StalenessPlan {
        plan_identity: "plan:t->y|z1,z2".into(),
        treatment: Some("t".into()),
        outcome: Some("y".into()),
        adjustment: vec!["z1".into(), "z2".into()],
        bayesian,
    }
}

fn policy() -> StalenessPolicy {
    StalenessPolicy { subsample: 60, permutations: 99, ..StalenessPolicy::default() }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn assess(
    baseline: &Scenario,
    new: &Scenario,
    bayesian: bool,
    policy: &StalenessPolicy,
) -> StalenessReport {
    let base = StalenessBaseline::build(&baseline.data(), &plan(bayesian), policy, &ctx()).unwrap();
    base.assess(&new.data(), policy, &ctx()).unwrap()
}

fn exceeded(report: &StalenessReport, kind: FindingKind, subject: &str) -> bool {
    report.findings().any(|f| f.kind == kind && f.subject.starts_with(subject))
}

#[test]
fn identical_data_is_reused_with_no_changed_key() {
    let scenario = Scenario::null(600, 11);
    let report = assess(&scenario, &scenario, false, &policy());
    assert_eq!(
        report.recommendation,
        Recommendation::Reuse,
        "{:#?}",
        report.findings().collect::<Vec<_>>()
    );
    assert!(report.changed_keys.is_empty());
    assert!(report.invalidated.is_empty());
    assert_eq!(report.old_snapshot, report.new_snapshot);
    assert!(report.evaluations.len() > 10, "every diagnostic is evaluated and recorded");
}

#[test]
fn false_alarm_rate_over_null_seeds_stays_at_the_declared_level() {
    let policy = StalenessPolicy { alpha: 0.05, ..policy() };
    let seeds = 100u64;
    let mut alarms = 0u64;
    let mut test_based = 0u64;
    for seed in 0..seeds {
        let report = assess(
            &Scenario::null(1000, 1000 + seed),
            &Scenario::null(1000, 5000 + seed),
            false,
            &policy,
        );
        if report.recommendation != Recommendation::Reuse {
            alarms += 1;
        }
        if report.findings().any(|f| f.adjusted_p_value.is_some()) {
            test_based += 1;
        }
    }
    // alpha plus three binomial standard errors.
    let bound = (seeds as f64 * 0.05 + 3.0 * (seeds as f64 * 0.05 * 0.95).sqrt()).floor() as u64;
    assert!(test_based <= bound, "test-based false alarms {test_based} of {seeds} exceed {bound}");
    assert!(alarms <= bound, "false alarms {alarms} of {seeds} exceed {bound}");
}

#[test]
fn a_covariate_shift_of_known_size_is_detected_and_asks_for_a_refresh() {
    let base = Scenario::null(1000, 21);
    let shifted = Scenario { z1_weights: [0.05, 0.1, 0.15, 0.3, 0.4], ..Scenario::null(1000, 22) };
    let report = assess(&base, &shifted, false, &policy());
    assert!(exceeded(&report, FindingKind::CovariateShift, "z1:mean"), "{report:#?}");
    let z1 = report.findings().find(|f| f.subject == "z1:mean").unwrap();
    assert!(z1.statistic > 0.5 && z1.statistic < 0.9, "standardized shift {}", z1.statistic);
    assert_eq!(z1.threshold_name, "smd_threshold");
    assert!(z1.adjusted_p_value.unwrap() <= 0.01);
    assert!(exceeded(&report, FindingKind::JointShift, "joint"));
    assert!(!exceeded(&report, FindingKind::CovariateShift, "z2:mean"));
    assert_eq!(
        report.recommendation,
        Recommendation::Refresh,
        "{:#?}",
        report.findings().collect::<Vec<_>>()
    );
    assert_eq!(report.changed_keys, vec![Stage::DataSnapshot]);
    assert_eq!(
        report.invalidated,
        vec![Stage::DataSnapshot, Stage::ScoreArtifact, Stage::Law, Stage::Decision]
    );
    assert!(!report.invalidated.contains(&Stage::Identification));
}

#[test]
fn lost_positivity_on_the_adjustment_set_asks_to_reidentify() {
    let base = Scenario::null(1000, 31);
    let separated = Scenario { t_slope: 8.0, ..Scenario::null(1000, 32) };
    let report = assess(&base, &separated, false, &policy());
    assert!(exceeded(&report, FindingKind::PositivityChange, "propensity"), "{report:#?}");
    let positivity = report.findings().find(|f| f.kind == FindingKind::PositivityChange).unwrap();
    assert!(positivity.statistic > 0.1);
    assert!(positivity.auxiliary.iter().any(|(name, _)| name == "max_weight"));
    assert_eq!(report.recommendation, Recommendation::Reidentify);
    assert!(report.changed_keys.contains(&Stage::Evidence));
    assert!(report.invalidated.contains(&Stage::Identification));
    assert!(report.next_attempt.contains("identify again"));
}

#[test]
fn a_new_treatment_level_is_a_level_change_that_asks_to_reidentify() {
    let base = Scenario::null(800, 41);
    let three = Scenario { treatment_levels: 3, ..Scenario::null(800, 42) };
    let report = assess(&base, &three, false, &policy());
    assert!(exceeded(&report, FindingKind::LevelChange, "arm:2"), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Reidentify);
    assert!(report.changed_keys.contains(&Stage::TreatmentGrid));
}

#[test]
fn schema_change_and_too_few_rows_are_refused_with_a_next_attempt() {
    let base = Scenario::null(400, 51);
    let mut columns = Scenario::null(400, 52).columns();
    columns.push(("w", vec![0.0; 400]));
    let wider =
        TabularData::from_f64_columns(columns.iter().map(|(n, v)| (*n, v.as_slice()))).unwrap();
    let baseline = StalenessBaseline::build(&base.data(), &plan(false), &policy(), &ctx()).unwrap();
    let report = baseline.assess(&wider, &policy(), &ctx()).unwrap();
    assert_eq!(report.recommendation, Recommendation::Refuse);
    assert!(exceeded(&report, FindingKind::SchemaChange, "schema"));
    assert!(report.next_attempt.contains("project the new data"));

    let tiny = Scenario::null(12, 53).data();
    let report = baseline.assess(&tiny, &policy(), &ctx()).unwrap();
    assert_eq!(report.recommendation, Recommendation::Refuse);
    assert!(exceeded(&report, FindingKind::InsufficientData, "complete_rows"));
    assert!(!report.next_attempt.is_empty());
}

#[test]
fn a_missingness_change_changes_the_row_design() {
    let base = Scenario::null(1000, 61);
    let holes = Scenario { missing_z2: 0.2, ..Scenario::null(1000, 62) };
    let report = assess(&base, &holes, false, &policy());
    assert!(exceeded(&report, FindingKind::MissingnessChange, "z2"), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Refresh);
    assert!(report.changed_keys.contains(&Stage::RowDesign));
    assert!(report.invalidated.contains(&Stage::RowDesign));
}

#[test]
fn drift_across_new_periods_is_found_by_the_cusum_and_asks_to_reidentify() {
    let policy = StalenessPolicy { period_column: Some("period".into()), ..policy() };
    let base = Scenario { periods: Some((0, 6, 0.0)), ..Scenario::null(600, 71) };
    let drifting = Scenario { periods: Some((6, 6, 1.0)), ..Scenario::null(600, 72) };
    let report = assess(&base, &drifting, false, &policy);
    assert!(exceeded(&report, FindingKind::PeriodDrift, "z2"), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Reidentify);
    assert!(report.changed_keys.contains(&Stage::SourcePopulation));
    assert!(
        !report.evaluations.iter().any(|f| f.subject.starts_with("period")),
        "the period column is not tested for shift"
    );
}

#[test]
fn a_prior_likelihood_conflict_names_the_prior_key() {
    let base = Scenario::null(800, 81);
    let changed = Scenario { effect: 5.0, ..Scenario::null(800, 82) };
    let report = assess(&base, &changed, true, &policy());
    assert!(
        exceeded(&report, FindingKind::PriorLikelihoodConflict, "outcome_model"),
        "{report:#?}"
    );
    let prior = Stage::Prior(Branch::new(0).unwrap());
    assert!(report.changed_keys.contains(&prior));
    assert!(report.invalidated.contains(&prior));
    assert!(!report.invalidated.contains(&Stage::Identification));
    assert!(report.next_attempt.contains("restate the prior"));

    let same = assess(&base, &Scenario::null(800, 83), true, &policy());
    assert!(!exceeded(&same, FindingKind::PriorLikelihoodConflict, "outcome_model"));
}

#[test]
fn values_beyond_the_recorded_range_are_support_loss() {
    let base = Scenario::null(800, 91);
    let mut columns = Scenario::null(800, 92).columns();
    for value in columns[1].1.iter_mut().take(80) {
        *value += 10.0;
    }
    let data =
        TabularData::from_f64_columns(columns.iter().map(|(n, v)| (*n, v.as_slice()))).unwrap();
    let baseline = StalenessBaseline::build(&base.data(), &plan(false), &policy(), &ctx()).unwrap();
    let report = baseline.assess(&data, &policy(), &ctx()).unwrap();
    let support =
        report.findings().find(|f| f.kind == FindingKind::SupportLoss && f.subject == "z2");
    assert!(support.is_some(), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Reidentify);
}

fn with_region(data: &TabularData, labels: &[&str], rows: &[usize]) -> TabularData {
    let mut builder = CausalSchemaBuilder::new();
    let mut columns = Vec::new();
    for (index, variable) in data.storage().schema().variables().iter().enumerate() {
        builder
            .add_variable(
                Arc::clone(&variable.name),
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let OwnedColumn::Float64(column) = &data.storage().columns()[index] else { unreachable!() };
        columns.push(OwnedColumn::Float64(column.clone()));
    }
    builder
        .add_variable(
            "region",
            ValueType::Categorical,
            SmallRoleSet::empty(),
            None,
            Some(CategoryDomainId::from_raw(0)),
            MeasurementSpec::default(),
        )
        .unwrap();
    let domain = Arc::new(
        CategoryDomain::try_new(
            CategoryDomainId::from_raw(0),
            labels.iter().map(|l| CategoryLevel { label: Arc::from(*l) }).collect::<Vec<_>>(),
            false,
            None,
            UnknownCategoryPolicy::Fail,
        )
        .unwrap(),
    );
    let codes: Vec<CategoryCode> =
        rows.iter().map(|r| CategoryCode::from_raw(u32::try_from(*r).unwrap())).collect();
    let id = VariableId::from_raw(u32::try_from(columns.len()).unwrap());
    columns.push(OwnedColumn::Categorical(
        CategoricalColumn::try_new(id, codes, ValidityBitmap::all_valid(rows.len()), domain)
            .unwrap(),
    ));
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap(),
    )
}

#[test]
fn categorical_levels_that_appear_or_disappear_are_level_changes() {
    let n = 600;
    let old = with_region(
        &Scenario::null(n, 101).data(),
        &["a", "b", "c"],
        &(0..n).map(|i| i % 3).collect::<Vec<_>>(),
    );
    // The new domain lists "d" in place of "c": "c" disappears and "d" appears.
    let new = with_region(
        &Scenario::null(n, 102).data(),
        &["a", "b", "d"],
        &(0..n).map(|i| i % 3).collect::<Vec<_>>(),
    );
    let baseline = StalenessBaseline::build(&old, &plan(false), &policy(), &ctx()).unwrap();
    let report = baseline.assess(&new, &policy(), &ctx()).unwrap();
    assert!(exceeded(&report, FindingKind::LevelChange, "region:d"), "{report:#?}");
    assert!(exceeded(&report, FindingKind::LevelChange, "region:c"));
    assert_eq!(report.recommendation, Recommendation::Refresh);
}

#[test]
fn an_exported_baseline_is_consumed_without_the_producer() {
    let base = Scenario::null(500, 111);
    let new =
        Scenario { z1_weights: [0.05, 0.1, 0.15, 0.3, 0.4], ..Scenario::null(500, 112) }.data();
    let baseline = StalenessBaseline::build(&base.data(), &plan(false), &policy(), &ctx()).unwrap();
    let bytes = baseline.export().unwrap();
    drop(baseline);
    let direct =
        StalenessBaseline::consume(&bytes).unwrap().assess(&new, &policy(), &ctx()).unwrap();
    let consumed = assess_exported_baseline(&bytes, &new, &policy(), &ctx()).unwrap();
    assert_eq!(consumed.identity, direct.identity);
    assert_eq!(consumed.recommendation, direct.recommendation);

    let mut tampered = bytes.clone();
    let last = tampered.len() - 10;
    tampered[last] ^= 1;
    assert!(matches!(
        assess_exported_baseline(&tampered, &new, &policy(), &ctx()),
        Err(StalenessError::Artifact(_))
    ));
}

#[test]
fn the_report_identity_binds_snapshots_policy_and_findings() {
    let base = Scenario::null(500, 121);
    let new = Scenario::null(500, 122);
    let first = assess(&base, &new, false, &policy());
    let again = assess(&base, &new, false, &policy());
    assert_eq!(first.identity, again.identity);
    let other_policy = StalenessPolicy { alpha: 0.02, ..policy() };
    assert_ne!(first.identity, assess(&base, &new, false, &other_policy).identity);
    assert_ne!(first.identity, assess(&base, &Scenario::null(500, 123), false, &policy()).identity);
}

#[test]
fn reports_are_identical_across_thread_counts_and_concurrent_callers() {
    let base = Scenario::null(800, 131).data();
    let new =
        Scenario { z1_weights: [0.1, 0.15, 0.2, 0.25, 0.3], ..Scenario::null(800, 132) }.data();
    let baseline = StalenessBaseline::build(&base, &plan(true), &policy(), &ctx()).unwrap();
    let reference = baseline.assess(&new, &policy(), &ExecutionContext::for_tests(1)).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = [1u32, 2, 4, 8]
            .into_iter()
            .map(|threads| {
                let (baseline, new) = (&baseline, &new);
                scope.spawn(move || {
                    baseline
                        .assess(new, &policy(), &ExecutionContext::production(99, threads))
                        .unwrap()
                })
            })
            .collect();
        for handle in handles {
            let report = handle.join().unwrap();
            assert_eq!(report.identity, reference.identity);
            assert_eq!(report.recommendation, reference.recommendation);
            assert_eq!(report.evaluations.len(), reference.evaluations.len());
        }
    });
}

#[test]
fn a_tiny_admission_pool_refuses_before_allocating() {
    let data = Scenario::null(400, 141).data();
    let tiny = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(1, 64));
    assert!(matches!(
        StalenessBaseline::build(&data, &plan(false), &policy(), &tiny),
        Err(StalenessError::Admission { .. })
    ));
    let baseline = StalenessBaseline::build(&data, &plan(false), &policy(), &ctx()).unwrap();
    assert!(matches!(
        baseline.assess(&data, &policy(), &tiny),
        Err(StalenessError::Admission { .. })
    ));
}

#[test]
fn masked_numeric_and_categorical_profiles_exclude_screen_rows_and_keep_source_identity() {
    use antecedent_validate::staleness::snapshot_identity;

    let raw = Scenario::null(400, 220);
    let labels = (0..400).map(|row| usize::from(row >= 200)).collect::<Vec<_>>();
    let data = with_region(&raw.data(), &["screen", "analysis"], &labels);
    let mask = ValidityBitmap::from_bools(&(0..400).map(|row| row >= 200).collect::<Vec<_>>());
    let masked = data.with_analysis_mask(mask).unwrap();
    let baseline = StalenessBaseline::build(&masked, &plan(false), &policy(), &ctx()).unwrap();
    assert_eq!(baseline.wire().rows, 200);
    assert_eq!(baseline.wire().complete_rows, 200);
    assert_eq!(baseline.snapshot(), snapshot_identity(&masked));
    let region = baseline.wire().columns.iter().find(|column| column.name == "region").unwrap();
    let levels = region.levels.as_ref().unwrap();
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].label, "analysis");
    assert_eq!(levels[0].count, 200);

    let values = raw.columns();
    let compact =
        TabularData::from_f64_columns(values.iter().map(|(name, column)| (*name, &column[200..])))
            .unwrap();
    let compact = with_region(&compact, &["screen", "analysis"], &vec![1; 200]);
    let expected = StalenessBaseline::build(&compact, &plan(false), &policy(), &ctx()).unwrap();
    assert_eq!(baseline.wire().columns, expected.wire().columns);
    assert_eq!(baseline.wire().joint, expected.wire().joint);
    assert_eq!(baseline.wire().overlap, expected.wire().overlap);
    assert_eq!(baseline.wire().checks, expected.wire().checks);
    assert_ne!(baseline.snapshot(), expected.snapshot());
    let report = baseline.assess(&masked, &policy(), &ctx()).unwrap();
    assert_eq!(report.new_rows, 200);
    assert_eq!(report.old_snapshot, report.new_snapshot);
    assert_eq!(report.recommendation, Recommendation::Reuse);
}

#[test]
fn nonunit_active_observation_weights_refuse_unweighted_profiles() {
    let data = Scenario::null(400, 221).data();
    let weighted = data.with_weights(vec![2.0; 400].into()).unwrap();
    assert!(matches!(
        StalenessBaseline::build(&weighted, &plan(false), &policy(), &ctx()),
        Err(StalenessError::Unsupported(_))
    ));
    let ordinary = StalenessBaseline::build(&data, &plan(false), &policy(), &ctx()).unwrap();
    assert!(matches!(
        ordinary.assess(&weighted, &policy(), &ctx()),
        Err(StalenessError::Unsupported(_))
    ));
    let unit = data.with_weights(vec![1.0; 400].into()).unwrap();
    let unit_baseline = StalenessBaseline::build(&unit, &plan(false), &policy(), &ctx()).unwrap();
    assert_eq!(ordinary.wire().columns, unit_baseline.wire().columns);
    assert_eq!(ordinary.wire().overlap, unit_baseline.wire().overlap);
}
