//! Mechanism-change *detection* (DESIGN.md §17.3) — separate from attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ExecutionContext, MechanismChangeQuery, VariableId};
use causal_data::{TableView, TabularData};
use causal_model::{
    CompiledCausalModel, MechanismRegistry, ParentBatch, SelectionPolicy, infer_noise_column,
};
use causal_stats::{
    change_point_two_sample, classifier_two_sample, kernel_two_sample, mean_diff_two_sample,
    residual_likelihood_ratio,
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
/// drive the target (DESIGN.md §17.3).
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    _ctx: &ExecutionContext,
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

    let mut out = Vec::with_capacity(query.targets.len());
    for &target in query.targets.iter() {
        let (stat, p_value, method_name) = match method {
            MechanismChangeMethod::LikelihoodRatio => {
                let rb = residuals(&base_model, &baseline, target)?;
                let rc = residuals(&base_model, &comparison, target)?;
                let (s, p) = residual_likelihood_ratio(&rb, &rc)?;
                (s, p, "likelihood_ratio")
            }
            MechanismChangeMethod::MeanDiff => {
                let rb = residuals(&base_model, &baseline, target)?;
                let rc = residuals(&base_model, &comparison, target)?;
                let (s, p) = mean_diff_two_sample(&rb, &rc)?;
                (s, p, "mean_diff")
            }
            MechanismChangeMethod::ClassifierTwoSample => {
                let rb = residuals(&base_model, &baseline, target)?;
                let rc = residuals(&base_model, &comparison, target)?;
                let (s, p) = classifier_two_sample(&rb, &rc)?;
                (s, p, "classifier_two_sample")
            }
            MechanismChangeMethod::KernelTwoSample => {
                let rb = residuals(&base_model, &baseline, target)?;
                let rc = residuals(&base_model, &comparison, target)?;
                let seed = 0x_4E12_A001u64
                    .wrapping_add(target.as_usize() as u64)
                    .wrapping_mul(0x9E37_79B9);
                let (s, p) = kernel_two_sample(&rb, &rc, seed)?;
                (s, p, "kernel_two_sample")
            }
            MechanismChangeMethod::ChangePoint => {
                let rb = residuals(&base_model, &baseline, target)?;
                let rc = residuals(&base_model, &comparison, target)?;
                let (s, p) = change_point_two_sample(&rb, &rc)?;
                (s, p, "change_point")
            }
        };
        out.push(MechanismChangeDetection {
            variable: target,
            changed: p_value < alpha,
            statistic: stat,
            p_value,
            method: Arc::from(method_name),
        });
    }
    Ok(out)
}

fn residuals(
    model: &CompiledCausalModel,
    data: &TabularData,
    target: VariableId,
) -> Result<Vec<f64>, AttributionError> {
    let dense = model
        .dense_of(target)
        .ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather = model
        .gather_for(dense)
        .ok_or(AttributionError::MissingArtifact("missing gather"))?;
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
    infer_noise_column(model.mechanisms.get(dense), &y, parents, &mut noise)?;
    Ok(noise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};

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

    /// Null split of a homogeneous SCM: residual MeanDiff Type I smoke.
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
}
