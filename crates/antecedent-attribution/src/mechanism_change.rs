//! Mechanism-change *detection* — separate from attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, MechanismChangeQuery, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_model::{
    CompiledCausalModel, MechanismRegistry, ParentBatch, SelectionPolicy, infer_noise_column_rng,
};
use antecedent_stats::{
    FdrAdjustment, adjust_pvalues, change_point_two_sample, classifier_two_sample,
    kernel_two_sample, mean_diff_two_sample, residual_likelihood_ratio,
};

use crate::error::AttributionError;
use crate::population::{resolve_rows, subset_table};
use crate::result::MechanismChangeDetection;

/// Detection method selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MechanismChangeMethod {
    /// Compare residual distributions via Gaussian KL / LR proxy.
    LikelihoodRatio,
    /// Mean difference on structural residuals (mechanism noise).
    MeanDiff,
    /// Classifier / two-sample proxy on residuals.
    ClassifierTwoSample,
    /// Kernel two-sample (MMD² + RBF) on residuals.
    KernelTwoSample,
    /// Known-split change-point test on concatenated baseline→comparison residuals.
    ChangePoint,
}

/// Detect which mechanisms differ between baseline and comparison populations.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// Applies a Benjamini–Hochberg false-discovery-rate correction across
/// `query.targets` by default (see
/// [`detect_mechanism_changes_with_correction`] to override or disable this).
/// `query.targets` is bounded by `query.max_targets`, i.e. the API is designed
/// to scan multiple candidate mechanisms in a single call, so judging each
/// target against the nominal `significance_level` independently would let the
/// family-wise false-positive rate grow with the number of targets (≈40% at
/// ten truly-unchanged targets and α=0.05). BH is chosen as the default
/// because it is the same default `antecedent_discovery::PcmciPlus` uses for
/// an analogous multi-hypothesis family (`PcmciPlus::new` sets
/// `fdr: Some(FdrAdjustment::bh())`), and it controls FDR rather than the
/// stricter (and more conservative) family-wise error rate, which is the
/// usual choice when the family is a moderate, human-reviewed set of
/// candidate mechanisms rather than a single make-or-break decision.
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    detect_mechanism_changes_with_correction(
        graph_model,
        data,
        query,
        method,
        ctx,
        Some(FdrAdjustment::bh()),
    )
}

/// Detect which mechanisms differ between baseline and comparison populations,
/// with explicit control over the multiple-testing correction applied across
/// `query.targets`.
///
/// `correction = None` reproduces the historical, uncorrected behavior where
/// each target's raw p-value is compared against `query.significance_level`
/// independently. `Some(adjustment)` adjusts the family of per-target p-values
/// with `adjustment.method` (see `antecedent_stats::fdr`) before thresholding.
///
/// [`MechanismChangeDetection::p_value`] always carries the *raw* per-target
/// p-value, and [`MechanismChangeDetection::adjusted_p_value`] carries the
/// corrected one (`None` when `correction` is `None`). `changed` is decided
/// from the adjusted value when present, so the flag and the number behind it
/// always agree while the unadjusted per-test result stays visible.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes_with_correction(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
    correction: Option<FdrAdjustment>,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    query.validate()?;
    if query.targets.len() > query.max_targets {
        return Err(AttributionError::SizeLimit {
            kind: "targets",
            requested: query.targets.len(),
            max: query.max_targets,
        });
    }
    let alpha = query.significance_level.to_f64();
    let baseline = subset_table(data, &resolve_rows(data, &query.baseline)?)?;
    let comparison = subset_table(data, &resolve_rows(data, &query.comparison)?)?;

    let (base_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &baseline,
        SelectionPolicy::BestScore,
    )?;
    let base_model = graph_model.clone().with_mechanisms(base_mechs);

    let mut raw = Vec::with_capacity(query.targets.len());
    for &target in query.targets.iter() {
        let mut baseline_rng = ctx.rng.stream(residual_stream_id(target, 0));
        let mut comparison_rng = ctx.rng.stream(residual_stream_id(target, 1));
        let (stat, p_value, method_name) = match method {
            MechanismChangeMethod::LikelihoodRatio => {
                let rb = residuals(&base_model, &baseline, target, &mut baseline_rng)?;
                let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
                let (s, p) = residual_likelihood_ratio(&rb, &rc)?;
                (s, p, "likelihood_ratio")
            }
            MechanismChangeMethod::MeanDiff => {
                let rb = residuals(&base_model, &baseline, target, &mut baseline_rng)?;
                let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
                let (s, p) = mean_diff_two_sample(&rb, &rc)?;
                (s, p, "mean_diff")
            }
            MechanismChangeMethod::ClassifierTwoSample => {
                let rb = residuals(&base_model, &baseline, target, &mut baseline_rng)?;
                let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
                let (s, p) = classifier_two_sample(&rb, &rc)?;
                (s, p, "classifier_two_sample")
            }
            MechanismChangeMethod::KernelTwoSample => {
                let rb = residuals(&base_model, &baseline, target, &mut baseline_rng)?;
                let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
                let seed = 0x_4E12_A001u64
                    .wrapping_add(target.as_usize() as u64)
                    .wrapping_mul(0x9E37_79B9);
                let (s, p) = kernel_two_sample(&rb, &rc, seed)?;
                (s, p, "kernel_two_sample")
            }
            MechanismChangeMethod::ChangePoint => {
                let rb = residuals(&base_model, &baseline, target, &mut baseline_rng)?;
                let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
                let (s, p) = change_point_two_sample(&rb, &rc)?;
                (s, p, "change_point")
            }
        };
        raw.push((target, stat, p_value, method_name));
    }

    // Report the raw p-value and the adjusted one separately rather than overwriting the
    // former. `changed` is decided from whichever is authoritative, but a caller reading
    // `p_value` must get the unadjusted per-target quantity its documentation promises —
    // silently returning a corrected number under that name is exactly the kind of
    // mislabelled statistic this correction exists to guard against.
    let adjusted: Option<Vec<f64>> = correction.map(|adjustment| {
        let p_values: Vec<f64> = raw.iter().map(|&(_, _, p, _)| p).collect();
        adjust_pvalues(&p_values, adjustment.method)
    });

    let mut out = Vec::with_capacity(raw.len());
    for (i, (target, stat, raw_p, method_name)) in raw.into_iter().enumerate() {
        let adjusted_p_value = adjusted.as_ref().map(|a| a[i]);
        let decisive = adjusted_p_value.unwrap_or(raw_p);
        out.push(MechanismChangeDetection {
            variable: target,
            changed: decisive < alpha,
            statistic: stat,
            p_value: raw_p,
            adjusted_p_value,
            method: Arc::from(method_name),
        });
    }
    Ok(out)
}

/// Distinct RNG stream per (target, population) so seeding is deterministic
/// under [`ExecutionContext::for_tests`] / any fixed `master_seed` but doesn't
/// alias across targets or across the baseline/comparison populations.
fn residual_stream_id(target: VariableId, population: u64) -> u64 {
    0x_5EED_C0DEu64
        .wrapping_add(target.as_usize() as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(population)
}

fn residuals(
    model: &CompiledCausalModel,
    data: &TabularData,
    target: VariableId,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, AttributionError> {
    let dense =
        model.dense_of(target).ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather =
        model.gather_for(dense).ok_or(AttributionError::MissingArtifact("missing gather"))?;
    let n = data.row_count();
    let y = data.float64_values(target)?;
    let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
    for (pi, &p) in gather.parents.iter().enumerate() {
        let pv = model.output_layout.variables[p.as_usize()];
        let col = data.float64_values(pv)?;
        parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
    }
    let parents = ParentBatch {
        n_rows: n,
        n_parents: gather.n_parents(),
        values: &parent_mat[..gather.n_parents().saturating_mul(n)],
    };
    let mut noise = vec![0.0; n];
    infer_noise_column_rng(model.mechanisms.get(dense), &y, parents, &mut noise, rng)?;
    Ok(noise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    fn detects_y_shift_not_necessarily_x() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
    }

    fn two_period_data() -> (CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    #[test]
    fn kernel_two_sample_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::KernelTwoSample,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "kernel_two_sample");
    }

    #[test]
    fn change_point_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::ChangePoint,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "change_point");
    }

    #[test]
    fn mean_diff_same_mechanism_different_marginal_mean_no_false_positive() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            // Same y = 1 + 2x + noise; x ranges differ → different marginal y mean, same mechanism.
            let x = if i < 40 { (i % 40) as f64 * 0.1 } else { 5.0 + (i % 40) as f64 * 0.1 };
            xv.push(x);
            yv.push(1.0 + 2.0 * x + 0.01 * ((i % 7) as f64 - 3.0));
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(!y.changed, "residual mean diff should not flag mechanism change: {y:?}");
    }

    /// Null split of a homogeneous SCM: residual `MeanDiff` Type I smoke.
    /// Not a full calibration gate — only checks flag rate stays near α.
    #[test]
    fn mean_diff_null_split_type_i_smoke() {
        let n = 200usize;
        let mid = n / 2;
        let mut flags = 0usize;
        let trials = 20usize;
        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x4C01u64.wrapping_add(trial as u64))
                .rng
                .stream(0x7E51);
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let mut xv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);
            for _ in 0..n {
                let u1 = rng.next_f64().clamp(1e-12, 1.0);
                let u2 = rng.next_f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let x = z;
                let u1 = rng.next_f64().clamp(1e-12, 1.0);
                let u2 = rng.next_f64();
                let e = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                xv.push(x);
                yv.push(1.0 + 2.0 * x + 0.25 * e);
            }
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
                ),
            ];
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let model = CompiledCausalModel::compile(g).unwrap();
            let q = MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                0.05,
                10,
            );
            let dets = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
            if y.changed {
                flags += 1;
            }
        }
        // α=0.05 → expect ~1 flag in 20; allow a few extras for smoke (not a calibration gate).
        assert!(
            flags <= 4,
            "null-split Type I smoke: MeanDiff flagged {flags}/{trials} (want ≤4 at α=0.05)"
        );
    }

    /// Family-wise Type I inflation across several truly-unchanged targets, and control via the
    /// default Benjamini–Hochberg correction (Defect A).
    ///
    /// `mean_diff_null_split_type_i_smoke` above only checks the *per-target* flag rate with a
    /// single target, so it cannot see the family-wise effect: with `m` independent null targets
    /// each judged against α=0.05 independently, `P(>=1 false flag) = 1 - 0.95^m`, which is
    /// ≈40% at m=10. This test builds `m` independent, unrelated mechanisms (no true baseline →
    /// comparison change in any of them) and compares the family-wise flag rate (>=1 of the `m`
    /// targets flagged) with `correction: None` against the corrected default
    /// `detect_mechanism_changes`.
    #[test]
    fn mean_diff_null_split_family_wise_type_i_rate() {
        let m = 10usize;
        let n = 120usize;
        let mid = n / 2;
        let trials = 150usize;
        let alpha = 0.05;

        let mut uncorrected_family_flags = 0usize;
        let mut corrected_family_flags = 0usize;

        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x9A11u64.wrapping_add(trial as u64))
                .rng
                .stream(0x51EE);
            let mut b = CausalSchemaBuilder::new();
            for k in 0..m {
                b.add_variable(
                    format!("x{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
                b.add_variable(
                    format!("y{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
            }
            let schema = b.build().unwrap();
            let mut cols = Vec::with_capacity(2 * m);
            for k in 0..m {
                let mut xv = Vec::with_capacity(n);
                let mut yv = Vec::with_capacity(n);
                for _ in 0..n {
                    let u1 = rng.next_f64().clamp(1e-12, 1.0);
                    let u2 = rng.next_f64();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let x = z;
                    let u1 = rng.next_f64().clamp(1e-12, 1.0);
                    let u2 = rng.next_f64();
                    let e = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    xv.push(x);
                    yv.push(1.0 + 2.0 * x + 0.25 * e);
                }
                let validity = ValidityBitmap::all_valid(n);
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k) as u32),
                        Arc::from(xv),
                        validity.clone(),
                    )
                    .unwrap(),
                ));
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k + 1) as u32),
                        Arc::from(yv),
                        validity,
                    )
                    .unwrap(),
                ));
            }
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables((2 * m) as u32);
            for k in 0..m {
                g.insert_directed(
                    DenseNodeId::from_raw((2 * k) as u32),
                    DenseNodeId::from_raw((2 * k + 1) as u32),
                )
                .unwrap();
            }
            let model = CompiledCausalModel::compile(g).unwrap();
            let targets: Vec<VariableId> =
                (0..m).map(|k| VariableId::from_raw((2 * k + 1) as u32)).collect();
            let q = MechanismChangeQuery::new(
                targets,
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                alpha,
                m,
            );

            let uncorrected = detect_mechanism_changes_with_correction(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
                None,
            )
            .unwrap();
            if uncorrected.iter().any(|d| d.changed) {
                uncorrected_family_flags += 1;
            }

            let corrected = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            if corrected.iter().any(|d| d.changed) {
                corrected_family_flags += 1;
            }
        }

        let uncorrected_rate = uncorrected_family_flags as f64 / trials as f64;
        let corrected_rate = corrected_family_flags as f64 / trials as f64;
        // Expected uncorrected family-wise rate ≈ 1 - 0.95^10 ≈ 0.40; allow slop for MC noise.
        assert!(
            uncorrected_rate > 0.20,
            "expected inflated uncorrected family-wise rate (~0.40 expected), got \
             {uncorrected_rate} ({uncorrected_family_flags}/{trials})"
        );
        assert!(
            corrected_rate < uncorrected_rate - 0.10,
            "BH-corrected family-wise rate ({corrected_rate}, {corrected_family_flags}/{trials}) \
             should be well below the uncorrected rate ({uncorrected_rate})"
        );
        assert!(
            corrected_rate <= 0.15,
            "BH-corrected family-wise rate should stay near α=0.05 under the global null, got \
             {corrected_rate} ({corrected_family_flags}/{trials})"
        );
    }

    /// Defect B: `residuals()` must thread a properly seeded RNG into posterior noise recovery
    /// (`infer_noise_column_rng`) rather than resetting to a fixed `CausalRng::from_seed(0)` on
    /// every call (the old `infer_noise_column` shim).
    ///
    /// A `Discrete`-family target's noise-inference mode is always `Posterior` (see
    /// `infer_noise_column_rng` in `antecedent-model/src/mechanism.rs`), so its residual draws
    /// are genuinely a function of the RNG stream, not just of the data. This test forces
    /// `MechanismRegistry::standard()` to fit `y` as `Discrete` (by giving it few distinct
    /// values, since family eligibility is driven by data cardinality via `is_low_cardinality`,
    /// not by the declared `ValueType`) and then runs `detect_mechanism_changes` twice under two
    /// different `ExecutionContext` seeds. Post-fix, the seed is threaded through to the
    /// posterior draws, so the resulting `MeanDiff` statistic differs between the two seeds.
    /// Pre-fix, both calls silently reset to seed 0 regardless of `ExecutionContext`, so the
    /// statistic is bit-identical across seeds — the RNG parameter is a no-op for any
    /// `Posterior`-mode family.
    ///
    /// Per the task brief: this does **not** attempt to show a flipped `changed` verdict (the
    /// within-population draws stay i.i.d. conditioned on whichever seed is used, and the
    /// two-sample tests treat samples as unordered sets, so a verdict flip could not be
    /// constructed). It demonstrates the narrower, honest claim that the execution context's
    /// seed is actually being honored after the fix and was being silently ignored before it.
    #[test]
    fn discrete_target_residuals_depend_on_execution_context_seed() {
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // x has 4 distinct categories, y has 2 — both well under the is_low_cardinality(_, 8)
        // threshold that routes the target to MechanismRegistry::standard().discrete =
        // [Discrete, Constant], and the x >= 2 split gives Discrete a real edge over Constant.
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 4) as f64;
            xv.push(x);
            yv.push(if x >= 2.0 { 1.0 } else { 0.0 });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: n },
            PopulationSelector::TimeRange { start: 0, end: n },
            0.05,
            10,
        );

        let d1 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let d2 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(2),
        )
        .unwrap();
        let s1 = d1.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        let s2 = d2.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        // Separated by a real margin, not just `!=`: posterior-sampled noise under two
        // different seeds should move the statistic materially, and an exact-inequality
        // check would also pass on a one-ULP difference.
        assert!(
            (s1 - s2).abs() > 1e-9,
            "Discrete-family residual statistic should depend on the ExecutionContext seed \
             (s1={s1}, s2={s2}); identical values mean the RNG is being ignored"
        );
    }
}
