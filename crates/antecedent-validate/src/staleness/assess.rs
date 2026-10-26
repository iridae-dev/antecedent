//! The comparison of new data with a baseline.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::ExecutionContext;
use antecedent_core::recalc::{Branch, Stage};
use antecedent_data::TabularData;
use antecedent_io_core::staleness_artifact::{
    JointProfileWire, LevelCountWire, OutcomeModelWire, StalenessBaselineWire, StalenessColumnWire,
};
use antecedent_kernels::chol_inverse;
use antecedent_stats::{chol_log_det, chol_solve, cholesky_spd};

use super::StalenessPlan;
use super::policy::StalenessPolicy;
use super::profile::{
    AT_MOST, LEVEL_DESIGN, LEVEL_IDENTIFICATION, ProfileInputs, ROLE_ADJUSTMENT, ROLE_TREATMENT,
    ReadTable, passes, period_series, plan_columns, profile,
};
use super::report::{
    FindingKind, Recommendation, StalenessFinding, StalenessReport, classes_and_products,
    invalidated_stages,
};
use super::stats::{
    NEW_SUBSAMPLE_OP, adjust, chi_square_homogeneity, energy_permutation, exceedance_p, f_sf,
    kolmogorov_sf, population_stability, smd, two_proportion, variance, welch,
};
use super::{StalenessError, snapshot_identity};

/// Evaluations under construction: a test-based one holds whether its effect size reached
/// its threshold until the multiplicity adjustment decides it.
#[derive(Default)]
struct Evaluator {
    evaluations: Vec<StalenessFinding>,
    effect_reached: Vec<Option<bool>>,
    not_evaluated: Vec<(String, String)>,
}

impl Evaluator {
    fn threshold(&mut self, finding: StalenessFinding) {
        self.evaluations.push(finding);
        self.effect_reached.push(None);
    }

    fn test(&mut self, mut finding: StalenessFinding, p_value: f64, effect_reached: bool) {
        finding.p_value = Some(p_value);
        finding.exceeded = false;
        self.evaluations.push(finding);
        self.effect_reached.push(Some(effect_reached));
    }

    fn skip(&mut self, what: impl Into<String>, why: impl Into<String>) {
        self.not_evaluated.push((what.into(), why.into()));
    }

    /// Adjust every test-based p-value as one family and decide each one.
    fn decide(&mut self, policy: &StalenessPolicy) {
        let tested: Vec<usize> =
            (0..self.evaluations.len()).filter(|&i| self.effect_reached[i].is_some()).collect();
        let raw: Vec<f64> =
            tested.iter().map(|&i| self.evaluations[i].p_value.unwrap_or(1.0)).collect();
        let adjusted = adjust(&raw, policy.multiplicity);
        for (&i, adjusted) in tested.iter().zip(adjusted) {
            let finding = &mut self.evaluations[i];
            finding.adjusted_p_value = Some(adjusted);
            finding.exceeded = self.effect_reached[i] == Some(true)
                && adjusted.is_finite()
                && adjusted <= policy.alpha;
        }
    }
}

fn keys_for_role(role: &str) -> (Recommendation, Vec<Stage>) {
    if role == ROLE_TREATMENT || role == ROLE_ADJUSTMENT {
        (Recommendation::Reidentify, vec![Stage::Evidence])
    } else {
        (Recommendation::Refresh, vec![Stage::DataSnapshot])
    }
}

/// Compare `data` with `baseline` under `policy`.
pub(crate) fn assess(
    baseline: &StalenessBaselineWire,
    data: &TabularData,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<StalenessReport, StalenessError> {
    super::check_cancelled(ctx)?;
    let table = ReadTable::new(data)?;
    let plan = StalenessPlan {
        plan_identity: baseline.plan_identity.clone(),
        treatment: baseline.treatment.clone(),
        outcome: baseline.outcome.clone(),
        adjustment: baseline.adjustment.clone(),
        bayesian: baseline.bayesian,
    };
    let period = policy.period_column.clone().or_else(|| baseline.period_column.clone());
    let mut profile_policy = policy.clone();
    profile_policy.period_column.clone_from(&period);
    let new_snapshot = snapshot_identity(data);
    let inputs = ProfileInputs {
        plan: &plan,
        policy: &profile_policy,
        reference: Some(baseline),
        subsample_op: NEW_SUBSAMPLE_OP,
    };
    let new = profile(&table, new_snapshot.clone(), &inputs);
    super::check_cancelled(ctx)?;
    let (ev, schema_ok) = diagnostics(baseline, &new, &table, &plan, policy, ctx)?;

    let recommendation = ev
        .evaluations
        .iter()
        .filter(|f| f.exceeded)
        .map(|f| f.forces)
        .max()
        .unwrap_or(Recommendation::Reuse);
    let mut keys: BTreeSet<Stage> =
        ev.evaluations.iter().filter(|f| f.exceeded).flat_map(|f| f.keys.iter().copied()).collect();
    if recommendation != Recommendation::Reuse {
        keys.insert(Stage::DataSnapshot);
        if new.complete_rows != baseline.complete_rows {
            keys.insert(Stage::RowDesign);
        }
    }
    let invalidated = invalidated_stages(&keys, &new_snapshot, baseline.bayesian);
    let changed_keys: Vec<Stage> = keys.into_iter().collect();
    let (changed_classes, invalidated_products) = classes_and_products(&changed_keys);
    let next_attempt = next_attempt(recommendation, &ev.evaluations, schema_ok, policy);
    let mut report = StalenessReport {
        recommendation,
        next_attempt,
        evaluations: ev.evaluations,
        not_evaluated: ev.not_evaluated,
        changed_keys,
        invalidated,
        changed_classes,
        invalidated_products,
        plan_identity: baseline.plan_identity.clone(),
        old_snapshot: baseline.snapshot.clone(),
        new_snapshot,
        baseline_identity: baseline.seal.clone(),
        policy_identity: policy.identity(),
        new_rows: new.rows,
        identity: String::new(),
    };
    report.identity = report.compute_identity();
    super::check_cancelled(ctx)?;
    Ok(report)
}

/// Run every diagnostic of `new` against `baseline` and decide the test-based ones.
/// Returns the evaluations and whether every plan column survived the schema change.
fn diagnostics(
    baseline: &StalenessBaselineWire,
    new: &StalenessBaselineWire,
    table: &ReadTable<'_>,
    plan: &StalenessPlan,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<(Evaluator, bool), StalenessError> {
    let checkpoint = || {
        if ctx.cancellation.is_cancelled() { Err(StalenessError::Cancelled) } else { Ok(()) }
    };
    checkpoint()?;
    let period = new.period_column.as_deref();
    let mut ev = Evaluator::default();
    let schema_ok = schema(baseline, new, plan, &mut ev);
    target(baseline, new, plan, policy, &mut ev);
    for old in &baseline.columns {
        if Some(old.name.as_str()) == period {
            continue;
        }
        let Some(current) = new.columns.iter().find(|c| {
            c.name == old.name && c.storage == old.storage && c.value_type == old.value_type
        }) else {
            continue;
        };
        column(old, current, table, (baseline.rows, new.rows), policy, &mut ev);
    }
    patterns(baseline, new, policy, &mut ev);
    checkpoint()?;
    joint(baseline.joint.as_ref(), new.joint.as_ref(), policy, &mut ev);
    checkpoint()?;
    overlap(baseline, new, policy, &mut ev);
    checks(baseline, new, &mut ev);
    if baseline.bayesian {
        match (&baseline.outcome_model, &new.outcome_model) {
            (Some(old), Some(current)) if old.design == current.design => {
                prior_conflict(old, current, policy, &mut ev);
            }
            _ => ev.skip(
                FindingKind::PriorLikelihoodConflict.label(),
                "the outcome regression of the plan cannot be formed on both snapshots",
            ),
        }
    }
    if let Some(period) = period {
        period_drift(table, period, policy, &mut ev);
    }
    ev.decide(policy);
    Ok((ev, schema_ok))
}

/// Schema change. Returns whether every plan column survived with its type.
fn schema(
    baseline: &StalenessBaselineWire,
    new: &StalenessBaselineWire,
    plan: &StalenessPlan,
    ev: &mut Evaluator,
) -> bool {
    let signature = |c: &StalenessColumnWire| (c.value_type.clone(), c.storage.clone());
    let old: BTreeMap<&str, _> =
        baseline.columns.iter().map(|c| (c.name.as_str(), signature(c))).collect();
    let current: BTreeMap<&str, _> =
        new.columns.iter().map(|c| (c.name.as_str(), signature(c))).collect();
    let added: Vec<&str> = current.keys().filter(|n| !old.contains_key(*n)).copied().collect();
    let removed: Vec<&str> = old.keys().filter(|n| !current.contains_key(*n)).copied().collect();
    let retyped: Vec<&str> = old
        .iter()
        .filter(|(n, s)| current.get(*n).is_some_and(|c| c != *s))
        .map(|(n, _)| *n)
        .collect();
    let reordered = common_order(&baseline.columns, &current) != common_order(&new.columns, &old);
    let changes = added.len() + removed.len() + retyped.len() + usize::from(reordered);
    let plan_cols = plan_columns(plan);
    let plan_ok =
        plan_cols.iter().all(|c| !removed.contains(&c.as_str()) && !retyped.contains(&c.as_str()));
    let detail =
        format!("added {added:?}, removed {removed:?}, retyped {retyped:?}, reordered {reordered}");
    ev.threshold(
        StalenessFinding::threshold(
            FindingKind::SchemaChange,
            "schema",
            ("changed_columns", changes as f64),
            ("schema_changes_allowed", 0.0),
            changes > 0,
            Recommendation::Refuse,
            vec![Stage::DataSnapshot, Stage::Evidence],
        )
        .with("added", added.len() as f64)
        .with("removed", removed.len() as f64)
        .with("retyped", retyped.len() as f64)
        .detail(detail),
    );
    if !plan_ok {
        ev.threshold(
            StalenessFinding::threshold(
                FindingKind::TargetIncompatible,
                "plan_columns",
                (
                    "missing_role_columns",
                    plan_cols
                        .iter()
                        .filter(|c| removed.contains(&c.as_str()) || retyped.contains(&c.as_str()))
                        .count() as f64,
                ),
                ("missing_role_columns_allowed", 0.0),
                true,
                Recommendation::Refuse,
                vec![Stage::DataSnapshot, Stage::Query],
            )
            .detail("a treatment, outcome or adjustment column is absent or retyped"),
        );
    }
    plan_ok
}

/// Names of `columns` that `other` also has, in `columns` order.
fn common_order<V>(columns: &[StalenessColumnWire], other: &BTreeMap<&str, V>) -> Vec<String> {
    columns.iter().filter(|c| other.contains_key(c.name.as_str())).map(|c| c.name.clone()).collect()
}

/// Sample-size floors and the treatment arms.
fn target(
    baseline: &StalenessBaselineWire,
    new: &StalenessBaselineWire,
    plan: &StalenessPlan,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    if !plan_columns(plan).is_empty() {
        ev.threshold(StalenessFinding::threshold(
            FindingKind::InsufficientData,
            "complete_rows",
            ("complete_rows", new.complete_rows as f64),
            ("min_rows", policy.min_rows as f64),
            new.complete_rows < policy.min_rows,
            Recommendation::Refuse,
            vec![Stage::DataSnapshot, Stage::RowDesign],
        ));
    }
    if baseline.arms.is_empty() {
        return;
    }
    let count = |arms: &[LevelCountWire], label: &str| {
        arms.iter().find(|a| a.label == label).map_or(0, |a| a.count)
    };
    for arm in baseline.arms.iter().filter(|a| a.count > 0) {
        let rows = count(&new.arms, &arm.label);
        if rows == 0 {
            ev.threshold(
                StalenessFinding::threshold(
                    FindingKind::TargetIncompatible,
                    format!("arm:{}", arm.label),
                    ("arm_rows", 0.0),
                    ("min_rows_per_arm", policy.min_rows_per_arm as f64),
                    true,
                    Recommendation::Refuse,
                    vec![Stage::DataSnapshot, Stage::Query],
                )
                .with("recorded_rows", arm.count as f64)
                .detail("a recorded treatment arm is absent: the contrast cannot be formed"),
            );
        } else {
            ev.threshold(
                StalenessFinding::threshold(
                    FindingKind::InsufficientData,
                    format!("arm:{}", arm.label),
                    ("arm_rows", rows as f64),
                    ("min_rows_per_arm", policy.min_rows_per_arm as f64),
                    rows < policy.min_rows_per_arm,
                    Recommendation::Refuse,
                    vec![Stage::DataSnapshot, Stage::RowDesign],
                )
                .with("recorded_rows", arm.count as f64),
            );
        }
    }
    for arm in &new.arms {
        if count(&baseline.arms, &arm.label) == 0 && arm.count >= policy.min_level_rows {
            ev.threshold(
                StalenessFinding::threshold(
                    FindingKind::LevelChange,
                    format!("arm:{}", arm.label),
                    ("new_level_rows", arm.count as f64),
                    ("min_level_rows", policy.min_level_rows as f64),
                    true,
                    Recommendation::Reidentify,
                    vec![Stage::Evidence, Stage::TreatmentGrid],
                )
                .detail("a treatment level the plan never saw appears"),
            );
        }
    }
}

/// Shift, support, levels and missingness of one column present in both snapshots.
fn column(
    old: &StalenessColumnWire,
    current: &StalenessColumnWire,
    table: &ReadTable<'_>,
    rows: (u64, u64),
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    let (support_forces, support_keys) = keys_for_role(&old.role);
    if let (Some(o), Some(c)) = (&old.numeric, &current.numeric) {
        if old.n_valid >= 2 && current.n_valid >= 2 {
            let (v0, v1) = (variance(old.n_valid, o.m2), variance(current.n_valid, c.m2));
            let effect = smd(o.mean, v0, c.mean, v1);
            if let Some((t, p)) = welch(old.n_valid, o.mean, v0, current.n_valid, c.mean, v1) {
                ev.test(
                    StalenessFinding::threshold(
                        FindingKind::CovariateShift,
                        format!("{}:mean", old.name),
                        ("standardized_mean_difference", effect),
                        ("smd_threshold", policy.smd_threshold),
                        false,
                        Recommendation::Refresh,
                        vec![Stage::DataSnapshot],
                    )
                    .with("welch_t", t)
                    .with("recorded_mean", o.mean)
                    .with("new_mean", c.mean),
                    p,
                    effect.abs() >= policy.smd_threshold,
                );
            }
            if let Some((statistic, df, p)) = chi_square_homogeneity(&o.bin_counts, &c.bin_counts) {
                let (psi, max_ratio) = population_stability(&o.bin_counts, &c.bin_counts);
                ev.test(
                    StalenessFinding::threshold(
                        FindingKind::CovariateShift,
                        format!("{}:distribution", old.name),
                        ("population_stability_index", psi),
                        ("psi_threshold", policy.psi_threshold),
                        false,
                        Recommendation::Refresh,
                        vec![Stage::DataSnapshot],
                    )
                    .with("chi_square", statistic)
                    .with("df", df as f64)
                    .with("max_density_ratio", max_ratio),
                    p,
                    psi >= policy.psi_threshold,
                );
            }
            support(old, o.min, o.max, table, policy, (support_forces, support_keys), ev);
        }
    }
    if let (Some(o), Some(c)) = (&old.levels, &current.levels) {
        levels(old, o, c, policy, (support_forces, &keys_for_role(&old.role).1), ev);
    }
    missingness(old, current, rows.0, rows.1, policy, ev);
}

fn support(
    old: &StalenessColumnWire,
    min: f64,
    max: f64,
    table: &ReadTable<'_>,
    policy: &StalenessPolicy,
    forces: (Recommendation, Vec<Stage>),
    ev: &mut Evaluator,
) {
    let Some(read) = table.get(&old.name).filter(|r| r.is_numeric()) else {
        return;
    };
    let (mut above, mut below, mut n1) = (0u64, 0u64, 0u64);
    for i in 0..table.rows {
        let v = read.value(i);
        if !v.is_finite() {
            continue;
        }
        n1 += 1;
        if v > max {
            above += 1;
        } else if v < min {
            below += 1;
        }
    }
    if n1 == 0 || !(min.is_finite() && max.is_finite()) {
        return;
    }
    let p = (2.0 * exceedance_p(old.n_valid, n1, above).min(exceedance_p(old.n_valid, n1, below)))
        .min(1.0);
    let fraction = (above + below) as f64 / n1 as f64;
    ev.test(
        StalenessFinding::threshold(
            FindingKind::SupportLoss,
            old.name.clone(),
            ("fraction_outside_recorded_range", fraction),
            ("support_tolerance", policy.support_tolerance),
            false,
            forces.0,
            forces.1,
        )
        .with("above_recorded_max", above as f64)
        .with("below_recorded_min", below as f64)
        .with("recorded_min", min)
        .with("recorded_max", max),
        p,
        fraction > policy.support_tolerance,
    );
}

fn levels(
    old: &StalenessColumnWire,
    recorded: &[LevelCountWire],
    current: &[LevelCountWire],
    policy: &StalenessPolicy,
    forces: (Recommendation, &[Stage]),
    ev: &mut Evaluator,
) {
    let labels: BTreeSet<&str> = recorded.iter().chain(current).map(|l| l.label.as_str()).collect();
    let count = |levels: &[LevelCountWire], label: &str| {
        levels.iter().find(|l| l.label == label).map_or(0, |l| l.count)
    };
    let o: Vec<u64> = labels.iter().map(|l| count(recorded, l)).collect();
    let c: Vec<u64> = labels.iter().map(|l| count(current, l)).collect();
    if let Some((statistic, df, p)) = chi_square_homogeneity(&o, &c) {
        let (psi, max_ratio) = population_stability(&o, &c);
        ev.test(
            StalenessFinding::threshold(
                FindingKind::CovariateShift,
                format!("{}:levels", old.name),
                ("population_stability_index", psi),
                ("psi_threshold", policy.psi_threshold),
                false,
                Recommendation::Refresh,
                vec![Stage::DataSnapshot],
            )
            .with("chi_square", statistic)
            .with("df", df as f64)
            .with("max_density_ratio", max_ratio),
            p,
            psi >= policy.psi_threshold,
        );
    }
    let (n0, n1): (u64, u64) = (o.iter().sum(), c.iter().sum());
    for (label, (&before, &after)) in labels.iter().zip(o.iter().zip(&c)) {
        if before == 0 && after >= policy.min_level_rows {
            ev.threshold(
                StalenessFinding::threshold(
                    FindingKind::LevelChange,
                    format!("{}:{label}", old.name),
                    ("new_level_rows", after as f64),
                    ("min_level_rows", policy.min_level_rows as f64),
                    true,
                    forces.0,
                    forces.1.to_vec(),
                )
                .detail("a level the baseline never saw appears"),
            );
        } else if before > 0 && after == 0 && n0 > 0 && n1 > 0 {
            let share = before as f64 / n0 as f64;
            let p = (1.0 - share).powf(n1 as f64);
            ev.test(
                StalenessFinding::threshold(
                    FindingKind::LevelChange,
                    format!("{}:{label}", old.name),
                    ("recorded_level_share", share),
                    ("alpha", policy.alpha),
                    false,
                    forces.0,
                    forces.1.to_vec(),
                )
                .detail("a recorded level is absent from the new data"),
                p,
                true,
            );
        }
    }
}

fn missingness(
    old: &StalenessColumnWire,
    current: &StalenessColumnWire,
    rows0: u64,
    rows1: u64,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    if old.n_missing == 0 && current.n_missing == 0 {
        return;
    }
    let Some((z, p)) = two_proportion(old.n_missing, rows0, current.n_missing, rows1) else {
        return;
    };
    let (r0, r1) = (old.n_missing as f64 / rows0 as f64, current.n_missing as f64 / rows1 as f64);
    ev.test(
        StalenessFinding::threshold(
            FindingKind::MissingnessChange,
            old.name.clone(),
            ("missing_rate_change", r1 - r0),
            ("missingness_threshold", policy.missingness_threshold),
            false,
            Recommendation::Refresh,
            vec![Stage::RowDesign],
        )
        .with("z", z)
        .with("recorded_missing_rate", r0)
        .with("new_missing_rate", r1),
        p,
        (r1 - r0).abs() >= policy.missingness_threshold,
    );
}

fn patterns(
    baseline: &StalenessBaselineWire,
    new: &StalenessBaselineWire,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    let (Some(old), Some(current)) = (&baseline.missing, &new.missing) else {
        return;
    };
    if old.columns != current.columns {
        ev.skip("missingness_patterns", "the plan's columns differ between the snapshots");
        return;
    }
    let (old_counts, new_counts) = (old.word_patterns(), current.word_patterns());
    let keys: BTreeSet<&Vec<u64>> = old_counts.iter().chain(&new_counts).map(|(p, _)| p).collect();
    if keys.iter().all(|p| p.is_empty()) {
        return;
    }
    let count = |patterns: &[(Vec<u64>, u64)], key: &Vec<u64>| {
        patterns.iter().find(|(p, _)| p == key).map_or(0, |(_, c)| *c)
    };
    let o: Vec<u64> = keys.iter().map(|k| count(&old_counts, k)).collect();
    let c: Vec<u64> = keys.iter().map(|k| count(&new_counts, k)).collect();
    let Some((statistic, df, p)) = chi_square_homogeneity(&o, &c) else {
        return;
    };
    let (n0, n1) = (o.iter().sum::<u64>() as f64, c.iter().sum::<u64>() as f64);
    let tv =
        0.5 * o.iter().zip(&c).map(|(a, b)| (*a as f64 / n0 - *b as f64 / n1).abs()).sum::<f64>();
    ev.test(
        StalenessFinding::threshold(
            FindingKind::MissingnessChange,
            "patterns",
            ("total_variation", tv),
            ("missingness_threshold", policy.missingness_threshold),
            false,
            Recommendation::Refresh,
            vec![Stage::RowDesign],
        )
        .with("chi_square", statistic)
        .with("df", df as f64),
        p,
        tv >= policy.missingness_threshold,
    );
}

/// Hotelling (Mahalanobis) and energy tests of the joint covariate distribution.
fn joint(
    old: Option<&JointProfileWire>,
    current: Option<&JointProfileWire>,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    let (Some(old), Some(current)) = (old, current) else {
        return;
    };
    let dim = old.columns.len();
    if current.columns != old.columns || dim == 0 {
        return;
    }
    let (n0, n1) = (old.n as f64, current.n as f64);
    let total = n0 + n1;
    if old.n > dim as u64 + 1 && current.n > dim as u64 + 1 {
        let pooled: Vec<f64> =
            old.m2.iter().zip(&current.m2).map(|(a, b)| (a + b) / (total - 2.0)).collect();
        let difference: Vec<f64> = current.mean.iter().zip(&old.mean).map(|(a, b)| a - b).collect();
        let solved =
            cholesky_spd(&pooled, dim).and_then(|chol| chol_solve(&chol, dim, &difference));
        match solved {
            Some(x) => {
                let d2: f64 = difference.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>().max(0.0);
                let t2 = n0 * n1 / total * d2;
                let p_dim = dim as f64;
                let f = (total - p_dim - 1.0) / (p_dim * (total - 2.0)) * t2;
                let p = f_sf(f, p_dim, total - p_dim - 1.0);
                ev.test(
                    StalenessFinding::threshold(
                        FindingKind::JointShift,
                        "joint:mean",
                        ("mahalanobis_distance", d2.sqrt()),
                        ("mahalanobis_threshold", policy.mahalanobis_threshold),
                        false,
                        Recommendation::Refresh,
                        vec![Stage::DataSnapshot],
                    )
                    .with("hotelling_t2", t2)
                    .with("f", f),
                    p,
                    d2.sqrt() >= policy.mahalanobis_threshold,
                );
            }
            None => ev.skip("joint:mean", "the pooled covariance of the covariates is singular"),
        }
    }
    let scale: Vec<f64> = (0..dim)
        .map(|j| {
            let sd = variance(old.n, old.m2[j * dim + j]).sqrt();
            if sd > 0.0 && sd.is_finite() { sd } else { 1.0 }
        })
        .collect();
    let standardize = |rows: &[f64]| -> Vec<f64> {
        rows.chunks(dim)
            .flat_map(|row| row.iter().zip(&old.mean).zip(&scale).map(|((v, m), s)| (v - m) / s))
            .collect()
    };
    let mut distances = Vec::new();
    match energy_permutation(
        &standardize(&old.subsample),
        &standardize(&current.subsample),
        dim,
        policy.permutations,
        policy.seed,
        &mut distances,
    ) {
        Some((statistic, p)) => ev.test(
            StalenessFinding::threshold(
                FindingKind::JointShift,
                "joint:distribution",
                ("energy_statistic", statistic),
                ("alpha", policy.alpha),
                false,
                Recommendation::Refresh,
                vec![Stage::DataSnapshot],
            )
            .with("recorded_subsample_rows", old.subsample_rows as f64)
            .with("new_subsample_rows", current.subsample_rows as f64),
            p,
            true,
        ),
        None => ev.skip("joint:distribution", "fewer than two subsample rows on a side"),
    }
}

/// Positivity and overlap of the treatment on the stored adjustment set.
fn overlap(
    baseline: &StalenessBaselineWire,
    new: &StalenessBaselineWire,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    let Some(old) = &baseline.overlap else {
        return;
    };
    let Some(current) = &new.overlap else {
        ev.skip(
            FindingKind::PositivityChange.label(),
            "the propensity of the treatment on the stored adjustment set cannot be fitted on \
             the new data",
        );
        return;
    };
    let positivity_lost = current.extreme_fraction > policy.max_extreme_fraction
        && current.extreme_fraction > old.extreme_fraction;
    ev.threshold(
        StalenessFinding::threshold(
            FindingKind::PositivityChange,
            "propensity",
            ("extreme_propensity_fraction", current.extreme_fraction),
            ("max_extreme_fraction", policy.max_extreme_fraction),
            positivity_lost,
            Recommendation::Reidentify,
            vec![Stage::Evidence],
        )
        .with("recorded_extreme_fraction", old.extreme_fraction)
        .with("min_propensity", current.min_propensity)
        .with("max_propensity", current.max_propensity)
        .with("max_weight", current.max_weight)
        .with("recorded_max_weight", old.max_weight),
    );
    let ratio = if old.ess_ratio > 0.0 { current.ess_ratio / old.ess_ratio } else { 1.0 };
    ev.threshold(
        StalenessFinding::threshold(
            FindingKind::OverlapChange,
            "propensity",
            ("ess_ratio_change", ratio),
            ("min_ess_ratio_change", policy.min_ess_ratio_change),
            ratio < policy.min_ess_ratio_change,
            Recommendation::Reidentify,
            vec![Stage::Evidence],
        )
        .with("recorded_ess_ratio", old.ess_ratio)
        .with("new_ess_ratio", current.ess_ratio)
        .with("max_weight", current.max_weight),
    );
}

/// Re-run each check the plan recorded as passed and compare.
fn checks(baseline: &StalenessBaselineWire, new: &StalenessBaselineWire, ev: &mut Evaluator) {
    for recorded in baseline.checks.iter().filter(|c| c.passed) {
        let Some(current) = new.checks.iter().find(|c| c.name == recorded.name) else {
            ev.skip(recorded.name.clone(), "the check cannot be formed on the new data");
            continue;
        };
        let regressed = !passes(current.statistic, recorded.threshold, &recorded.direction);
        let (forces, keys) = match recorded.level.as_str() {
            LEVEL_IDENTIFICATION => (Recommendation::Reidentify, vec![Stage::Evidence]),
            LEVEL_DESIGN => (Recommendation::Refresh, vec![Stage::RowDesign]),
            _ => (Recommendation::Refuse, vec![Stage::Query]),
        };
        ev.threshold(
            StalenessFinding::threshold(
                FindingKind::AssumptionRegression,
                recorded.name.clone(),
                ("check_statistic", current.statistic),
                (
                    if recorded.direction == AT_MOST {
                        "recorded_maximum"
                    } else {
                        "recorded_minimum"
                    },
                    recorded.threshold,
                ),
                regressed,
                forces,
                keys,
            )
            .with("recorded_statistic", recorded.statistic)
            .detail(format!("{} check recorded as passed", recorded.level)),
        );
    }
}

/// Box's prior predictive check of the new outcome-regression coefficients under the stored
/// posterior (conjugate normal-inverse-gamma under the policy's reference prior).
fn prior_conflict(
    old: &OutcomeModelWire,
    current: &OutcomeModelWire,
    policy: &StalenessPolicy,
    ev: &mut Evaluator,
) {
    let p = old.design.len() + 1;
    let what = FindingKind::PriorLikelihoodConflict.label();
    if current.n <= p as u64 + 1 || old.n == 0 {
        ev.skip(what, "the new data has too few complete rows for the outcome regression");
        return;
    }
    // Stored posterior: Λ = Λ₀ + XᵀX, η = Xᵀy (prior mean zero), a = a₀ + n/2,
    // b = b₀ + (yᵀy − ηᵀΛ⁻¹η)/2.
    let mut precision = old.xtx.clone();
    for j in 0..p {
        precision[j * p + j] += 1.0 / policy.prior_variance;
    }
    let Some(chol) = cholesky_spd(&precision, p) else {
        ev.skip(what, "the stored posterior precision is singular");
        return;
    };
    let Some(mean) = chol_solve(&chol, p, &old.xty) else {
        ev.skip(what, "the stored posterior precision is singular");
        return;
    };
    let quadratic: f64 = old.xty.iter().zip(&mean).map(|(a, b)| a * b).sum();
    let shape = policy.prior_shape + 0.5 * old.n as f64;
    let rate = (policy.prior_rate + 0.5 * (old.yty - quadratic)).max(f64::MIN_POSITIVE);
    let posterior_scale = chol_inverse(&chol, p);
    // New data: least-squares coefficients and their unscaled covariance.
    let Some(new_chol) = cholesky_spd(&current.xtx, p) else {
        ev.skip(what, "the new design of the outcome regression is singular");
        return;
    };
    let Some(beta) = chol_solve(&new_chol, p, &current.xty) else {
        ev.skip(what, "the new design of the outcome regression is singular");
        return;
    };
    let new_scale = chol_inverse(&new_chol, p);
    let combined: Vec<f64> = posterior_scale.iter().zip(&new_scale).map(|(a, b)| a + b).collect();
    let difference: Vec<f64> = beta.iter().zip(&mean).map(|(a, b)| a - b).collect();
    let Some(solved) = cholesky_spd(&combined, p).and_then(|c| chol_solve(&c, p, &difference))
    else {
        ev.skip(what, "the predictive covariance is singular");
        return;
    };
    let distance: f64 = difference.iter().zip(&solved).map(|(a, b)| a * b).sum::<f64>().max(0.0);
    let noise = rate / shape;
    let f = distance / (noise * p as f64);
    let p_value = f_sf(f, p as f64, 2.0 * shape);
    // Gaussian divergence of the new-data likelihood from the stored posterior.
    let fitted: f64 = beta.iter().zip(&current.xty).map(|(a, b)| a * b).sum();
    let residual = ((current.yty - fitted) / (current.n as f64 - p as f64)).max(f64::MIN_POSITIVE);
    let trace: f64 = (0..p)
        .map(|i| (0..p).map(|k| precision[i * p + k] * new_scale[k * p + i]).sum::<f64>())
        .sum();
    let mahalanobis: f64 = {
        let lambda_d: Vec<f64> =
            (0..p).map(|i| (0..p).map(|k| precision[i * p + k] * difference[k]).sum()).collect();
        difference.iter().zip(&lambda_d).map(|(a, b)| a * b).sum()
    };
    let log_det_old = p as f64 * noise.ln() - chol_log_det(&chol, p);
    let log_det_new = p as f64 * residual.ln() - chol_log_det(&new_chol, p);
    let divergence = 0.5
        * (residual / noise * trace + mahalanobis / noise - p as f64 + log_det_old - log_det_new);
    ev.test(
        StalenessFinding::threshold(
            FindingKind::PriorLikelihoodConflict,
            "outcome_model",
            ("box_predictive_f", f),
            ("alpha", policy.alpha),
            false,
            Recommendation::Refresh,
            Branch::new(0).map(Stage::Prior).into_iter().chain([Stage::DataSnapshot]).collect(),
        )
        .with("kl_divergence", divergence)
        .with("posterior_noise_variance", noise)
        .with("new_noise_variance", residual),
        p_value,
        true,
    );
}

/// CUSUM of per-period means of the new data (Brownian-bridge supremum).
fn period_drift(table: &ReadTable<'_>, period: &str, policy: &StalenessPolicy, ev: &mut Evaluator) {
    let what = FindingKind::PeriodDrift.label();
    let Some(series) = period_series(table, period) else {
        ev.skip(what, format!("the new data has no readable period column {period}"));
        return;
    };
    for column in series {
        let periods: Vec<&(String, u64, f64, f64)> =
            column.periods.iter().filter(|p| p.1 >= policy.min_rows_per_period).collect();
        if periods.len() < 2 {
            ev.skip(format!("{what}:{}", column.column), "fewer than two periods with enough rows");
            continue;
        }
        let n: f64 = periods.iter().map(|p| p.1 as f64).sum();
        let k = periods.len() as f64;
        let grand = periods.iter().map(|p| p.1 as f64 * p.2).sum::<f64>() / n;
        let within = periods.iter().map(|p| p.3).sum::<f64>() / (n - k);
        let sd = within.sqrt();
        if !(sd.is_finite() && sd > 0.0) {
            continue;
        }
        let mut cumulative = 0.0;
        let mut statistic = 0.0_f64;
        let mut at = 0usize;
        for (index, p) in periods.iter().enumerate().take(periods.len() - 1) {
            cumulative += p.1 as f64 * (p.2 - grand);
            let s = (cumulative / (sd * n.sqrt())).abs();
            if s > statistic {
                statistic = s;
                at = index;
            }
        }
        let largest = periods.iter().map(|p| ((p.2 - grand) / sd).abs()).fold(0.0_f64, f64::max);
        let after = periods.get(at + 1).map_or("", |p| p.0.as_str());
        ev.test(
            StalenessFinding::threshold(
                FindingKind::PeriodDrift,
                column.column.clone(),
                ("cusum_statistic", statistic),
                ("alpha", policy.alpha),
                false,
                Recommendation::Reidentify,
                vec![Stage::SourcePopulation, Stage::Evidence],
            )
            .with("periods", k)
            .with("largest_standardized_period_deviation", largest)
            .detail(format!("largest cumulative deviation before period {after}")),
            kolmogorov_sf(statistic),
            largest >= policy.smd_threshold,
        );
    }
}

fn next_attempt(
    recommendation: Recommendation,
    evaluations: &[StalenessFinding],
    plan_columns_intact: bool,
    policy: &StalenessPolicy,
) -> String {
    let exceeded = |kind: FindingKind| evaluations.iter().any(|f| f.exceeded && f.kind == kind);
    let subjects = |forces: Recommendation| -> String {
        let mut names: Vec<&str> = evaluations
            .iter()
            .filter(|f| f.exceeded && f.forces == forces)
            .map(|f| f.subject.as_str())
            .collect();
        names.dedup();
        names.join(", ")
    };
    match recommendation {
        Recommendation::Reuse => {
            "reuse the stored claim: no diagnostic exceeds its threshold, so no refit is needed"
                .into()
        }
        Recommendation::Refresh => {
            let mut text = format!(
                "refresh the prepared study on the new data ({}): identification and the plan are \
                 kept; the score or fit artifact, the law and the decision are recomputed",
                subjects(Recommendation::Refresh)
            );
            if exceeded(FindingKind::PriorLikelihoodConflict) {
                text.push_str(
                    "; restate the prior first, because the stored posterior conflicts with the \
                     likelihood of the new data",
                );
            }
            text
        }
        Recommendation::Reidentify => {
            let mut text = format!(
                "identify again on the new data (re-prepare the study): {} no longer support the \
                 stored identification or adjustment set",
                subjects(Recommendation::Reidentify)
            );
            if exceeded(FindingKind::PeriodDrift) {
                text.push_str(
                    "; the periods drift, so declare the mechanism invariant across periods or \
                     model period-level parameters before identifying again",
                );
            }
            text
        }
        Recommendation::Refuse => {
            if exceeded(FindingKind::SchemaChange) && plan_columns_intact {
                "project the new data onto the prepared schema (drop the added columns, restore \
                 the removed ones and the prepared order) and assess again, or re-prepare the plan \
                 on the new schema"
                    .into()
            } else if exceeded(FindingKind::SchemaChange)
                || exceeded(FindingKind::TargetIncompatible)
            {
                "re-specify the target on the columns and treatment levels the new data carries and \
                 re-prepare the plan"
                    .into()
            } else {
                format!(
                    "collect at least {} complete rows and {} rows per treatment arm, then assess \
                     again: uncertainty is not assessable on fewer",
                    policy.min_rows, policy.min_rows_per_arm
                )
            }
        }
    }
}
