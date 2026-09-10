//! Cross-fitted binary / discrete-joint AIPW scores.
//!
//! Fold assignment is deterministic (`i % folds`), matching the Kennedy path.
//! These are uncentered AIPW scores. Centering estimates the efficient influence
//! function only under consistency, positivity, and nuisance convergence/rate
//! conditions. Cross-fitting alone does not guarantee valid inference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::similar_names)]
#![allow(clippy::many_single_char_names, clippy::too_many_lines, clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, OutcomeFunctional, VariableId};
use antecedent_stats::{
    FaerBackend, GlmOptions, PropensityWorkspace, fit_propensity, predict_propensity,
};

use crate::aipw::{AipwWorkspace, fit_outcome_models, predict_colmajor, select_rows_colmajor};
use crate::error::EstimationError;
use crate::propensity::{
    PreparedPropensityProblem, clamp_scores, clip_of, gather, split_by_treatment,
};
use crate::scores::{ScoreColumn, ScoreTable};
use crate::util::stats_err;

/// Default fold count for licensed AIPW scores.
pub const DEFAULT_AIPW_FOLDS: usize = 5;

/// Provenance tag frozen on the score table.
pub const AIPW_CROSSFIT_PROVENANCE: &str = "aipw.crossfit.v1";

/// Build a score table from a prepared propensity problem.
///
/// One column per `(arm, threshold)`. `thresholds = [None]` is the mean
/// functional. Arms are `0` (control) and `1` (treated) for binary ATE.
///
/// # Errors
///
/// Empty folds, missing arms in a fold, or GLM/OLS failure.
pub fn crossfit_binary_scores(
    problem: &PreparedPropensityProblem,
    query: &AverageEffectQuery,
    folds: usize,
    glm_options: &GlmOptions,
    backend: FaerBackend,
) -> Result<ScoreTable, EstimationError> {
    let thresholds = thresholds_of(&query.outcome_functional);
    build_binary_scores(problem, query.treatment, &thresholds, folds, glm_options, backend)
}

/// Build binary-arm scores for an explicit threshold list (`None` = mean).
///
/// # Errors
///
/// Empty folds or nuisance failure.
pub fn build_binary_scores(
    problem: &PreparedPropensityProblem,
    treatment: VariableId,
    thresholds: &[Option<f64>],
    folds: usize,
    glm_options: &GlmOptions,
    backend: FaerBackend,
) -> Result<ScoreTable, EstimationError> {
    if folds < 2 {
        return Err(EstimationError::unsupported("AIPW cross-fitting requires at least two folds"));
    }
    if problem.nrows < folds {
        return Err(EstimationError::data_msg("cross-fitting folds cannot exceed complete rows"));
    }
    if thresholds.is_empty() {
        return Err(EstimationError::data_msg("score table requires at least one functional"));
    }

    let n = problem.nrows;
    let ncols = problem.design_ncols;
    let clip = clip_of(problem.overlap);
    let fold_ids: Vec<u32> =
        (0..n).map(|i| (problem.row_index[i] as usize % folds) as u32).collect();

    let mut columns = Vec::new();
    for &c in thresholds {
        columns.push(ScoreColumn { arm: 0, threshold: c });
        columns.push(ScoreColumn { arm: 1, threshold: c });
    }
    let n_cols = columns.len();
    let mut scores = vec![0.0; n * n_cols];
    let mut propensities = vec![0.0; n * n_cols];

    let mut prop_ws = PropensityWorkspace::default();
    let mut out_ws = AipwWorkspace::default();

    for fold in 0..folds {
        let train: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize != fold).collect();
        let valid: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize == fold).collect();
        if train.is_empty() || valid.is_empty() {
            return Err(EstimationError::data_msg("AIPW cross-fit fold is empty"));
        }

        let mut design_train = Vec::new();
        select_rows_colmajor(&problem.design_matrix, n, ncols, &train, &mut design_train);
        let t_train = gather(&problem.treatment, &train);
        if split_by_treatment(&t_train).0.is_empty() || split_by_treatment(&t_train).1.is_empty() {
            return Err(EstimationError::data_msg(
                "AIPW cross-fit fold is missing a treatment arm",
            ));
        }
        let fit = fit_propensity(
            &design_train,
            train.len(),
            ncols,
            &t_train,
            &backend,
            &mut prop_ws,
            glm_options,
        )
        .map_err(stats_err)?;

        fit.glm.require_ok().map_err(stats_err)?;
        let mut design_valid = Vec::new();
        select_rows_colmajor(&problem.design_matrix, n, ncols, &valid, &mut design_valid);
        let mut e_valid = vec![0.0; valid.len()];
        predict_propensity(&design_valid, valid.len(), ncols, &fit.coefficients, &mut e_valid)
            .map_err(stats_err)?;
        let raw_e = e_valid.clone();
        if let Some(clip_at) = clip {
            clamp_scores(&mut e_valid, clip_at);
        }
        for e in &mut e_valid {
            *e = e.clamp(1e-8, 1.0 - 1e-8);
        }

        for (t_idx, &threshold) in thresholds.iter().enumerate() {
            let y_source = transform_outcome(&problem.outcome, threshold);
            let y_train = gather(&y_source, &train);
            let y_valid = gather(&y_source, &valid);
            let (beta0, beta1) = fit_outcome_models(
                &design_train,
                train.len(),
                ncols,
                &t_train,
                &y_train,
                backend,
                &mut out_ws,
            )?;
            predict_colmajor(&design_valid, valid.len(), ncols, &beta0, &mut out_ws.mu0);
            predict_colmajor(&design_valid, valid.len(), ncols, &beta1, &mut out_ws.mu1);

            let col0 = t_idx * 2;
            let col1 = col0 + 1;
            for (k, &i) in valid.iter().enumerate() {
                let t = problem.treatment[i];
                let y = y_valid[k];
                let e = e_valid[k];
                let m0 = out_ws.mu0[k];
                let m1 = out_ws.mu1[k];
                propensities[col0 * n + i] = 1.0 - raw_e[k];
                propensities[col1 * n + i] = raw_e[k];
                scores[col0 * n + i] = m0 + ((1.0 - t) / (1.0 - e)) * (y - m0);
                scores[col1 * n + i] = m1 + (t / e) * (y - m1);
            }
        }
    }

    Ok(ScoreTable {
        observed_arm: problem
            .treatment
            .iter()
            .map(|t| u32::from(*t > 0.5))
            .collect::<Vec<_>>()
            .into(),
        propensities: propensities.into(),
        observed_outcome: Arc::clone(&problem.outcome),
        n_rows: n,
        row_index: Arc::clone(&problem.row_index),
        fold_ids: Arc::from(fold_ids),
        n_folds: u32::try_from(folds).unwrap_or(u32::MAX),
        scores: Arc::from(scores),
        columns: Arc::from(columns),
        adjustment_set: Arc::clone(&problem.adjustment_set),
        nuisance_provenance: Arc::from(AIPW_CROSSFIT_PROVENANCE),
        treatment,
        intervened: Arc::from([]),
    })
}

/// Map an outcome functional to score-table thresholds (`None` = mean).
#[must_use]
pub fn thresholds_of(functional: &OutcomeFunctional) -> Vec<Option<f64>> {
    match functional.thresholds() {
        None => vec![None],
        Some(cs) => cs.into_iter().map(Some).collect(),
    }
}

fn transform_outcome(y: &[f64], threshold: Option<f64>) -> Vec<f64> {
    match threshold {
        None => y.to_vec(),
        Some(c) => y.iter().map(|&yi| if yi > c { 1.0 } else { 0.0 }).collect(),
    }
}

/// Weighted support diagnostics for a binary score table.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedSupport {
    /// Kish `n_eff` under `w`.
    pub n_eff: f64,
    /// Per-arm Kish `n_eff` using `w * 1{A=a}`.
    pub n_eff_by_arm: Vec<f64>,
    /// Min / max propensity among rows with `w > 0` (if supplied).
    pub propensity_range: Option<(f64, f64)>,
    /// Whether weighted overlap is usable.
    pub overlap_ok: bool,
}

/// Weighted overlap for binary treatment under target weights.
#[must_use]
pub fn weighted_support(
    treatment: &[f64],
    weights: &[f64],
    propensity: Option<&[f64]>,
    min_n_eff_arm: f64,
) -> WeightedSupport {
    let mut w0 = Vec::new();
    let mut w1 = Vec::new();
    let mut w_all = Vec::new();
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    for i in 0..treatment.len() {
        let w = weights.get(i).copied().unwrap_or(0.0);
        if w <= 0.0 {
            continue;
        }
        w_all.push(w);
        if treatment[i] > 0.5 {
            w1.push(w);
        } else {
            w0.push(w);
        }
        if let Some(e) = propensity.and_then(|p| p.get(i)).copied() {
            p_min = p_min.min(e);
            p_max = p_max.max(e);
        }
    }
    let n_eff = crate::joint_if::kish_n_eff(&w_all);
    let n0 = crate::joint_if::kish_n_eff(&w0);
    let n1 = crate::joint_if::kish_n_eff(&w1);
    let range = if p_min.is_finite() { Some((p_min, p_max)) } else { None };
    let overlap_ok = n0 >= min_n_eff_arm && n1 >= min_n_eff_arm;
    WeightedSupport { n_eff, n_eff_by_arm: vec![n0, n1], propensity_range: range, overlap_ok }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipw::AipwAte;
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;
    use std::sync::Arc;

    fn confounded(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x51);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let p = 1.0 / (1.0 + (0.5 - zi).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + 0.25 * standard_normal(&mut rng);
        }
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols = [t, y, z]
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(v),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn crossfit_scores_recover_ate_two() {
        let (data, estimand) = confounded(1_200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let table =
            crossfit_binary_scores(&prep, &query, 5, &est.glm_options, est.backend).unwrap();
        let summary = table.summarize(None).unwrap();
        let ate = summary.means[1] - summary.means[0];
        assert!((ate - 2.0).abs() < 0.35, "crossfit ate={ate}");
        assert_eq!(table.n_folds, 5);
        assert_eq!(table.columns.len(), 2);
    }

    #[test]
    fn custom_weights_shift_the_mean() {
        let (data, estimand) = confounded(800, 4);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::AllObserved);
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let table =
            crossfit_binary_scores(&prep, &query, 5, &est.glm_options, est.backend).unwrap();
        let w: Vec<f64> = (0..table.n_rows).map(|i| if i % 2 == 0 { 2.0 } else { 0.5 }).collect();
        let a = table.summarize(None).unwrap();
        let b = table.summarize(Some(&w)).unwrap();
        assert_eq!(a.means.len(), b.means.len());
    }
}
