//! Cell-saturated AIPW for discrete joint `Set` interventions.
//!
//! A multinomial-logit propensity over the `2^k` cells and a per-cell outcome
//! model, on the common adjustment set 1.4 already certifies. Interaction
//! contrasts are identified; they are not structurally zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::similar_names)]
#![allow(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use std::sync::Arc;

use antecedent_core::{OutcomeFunctional, VariableId};
use antecedent_data::TabularData;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, GlmOptions};

use crate::aipw::{predict_colmajor, select_rows_colmajor};
use crate::crossfit_aipw::{DEFAULT_AIPW_FOLDS, thresholds_of};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::propensity::clip_of;
use crate::scores::{LinearContrast, ScoreColumn, ScoreSummary, ScoreTable};
use crate::util::stats_err;

/// Maximum jointly intervened binary coordinates (8 cells at k=3).
pub const MAX_JOINT_BINARY: usize = 3;

/// Named refuse: coarsened `continuous_cell` is not a point CDE.
pub const POINT_CDE_UNLICENSED: &str = "point controlled direct effect is unlicensed; continuous_cell coarsens D onto a grid and is not do(D=d0)";

/// Cell-saturated joint AIPW estimator.
#[derive(Clone, Debug)]
pub struct CellSaturatedAipw {
    /// Dense backend.
    pub backend: FaerBackend,
    /// Cross-fit folds.
    pub folds: usize,
    /// GLM options retained for estimator configuration.
    pub glm_options: GlmOptions,
    /// Overlap policy (clip applied to each cell propensity before normalize).
    pub overlap: OverlapPolicy,
}

impl Default for CellSaturatedAipw {
    fn default() -> Self {
        Self::new()
    }
}

impl CellSaturatedAipw {
    /// Defaults: 5 folds, clip 0.01.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            folds: DEFAULT_AIPW_FOLDS,
            glm_options: GlmOptions::default(),
            overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        }
    }

    /// Fit cell scores for binary treatments `treatments` on `adjustment`.
    ///
    /// A continuous coordinate is refused: coarsening D is not `do(D=d0)`.
    ///
    /// # Errors
    ///
    /// Too many treatments, empty cells needed for a requested contrast,
    /// a fold missing a cell that appears in the data, or a continuous cell
    /// (point CDE is unlicensed).
    pub fn fit_scores(
        &self,
        data: &TabularData,
        treatments: &[VariableId],
        outcome: VariableId,
        adjustment: &[VariableId],
        functional: &OutcomeFunctional,
        continuous: Option<ContinuousCellSpec<'_>>,
    ) -> Result<ScoreTable, EstimationError> {
        if treatments.is_empty() || treatments.len() > MAX_JOINT_BINARY {
            return Err(EstimationError::unsupported(
                "cell-saturated AIPW supports 1..=3 binary treatments",
            ));
        }
        self.fit_scores_with_assignment(
            data, treatments, outcome, adjustment, functional, continuous, None, None,
        )
    }

    /// Fit cell scores using a caller-assigned fold vector and/or `[1 | Z…]` design.
    ///
    /// `fold_ids` and `design` are complete-case aligned with the prepared rows.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::fit_scores`], plus assignment shape mismatch.
    pub fn fit_scores_with_assignment(
        &self,
        data: &TabularData,
        treatments: &[VariableId],
        outcome: VariableId,
        adjustment: &[VariableId],
        functional: &OutcomeFunctional,
        continuous: Option<ContinuousCellSpec<'_>>,
        fold_ids: Option<&[u32]>,
        design: Option<&[f64]>,
    ) -> Result<ScoreTable, EstimationError> {
        if treatments.is_empty() || treatments.len() > MAX_JOINT_BINARY {
            return Err(EstimationError::unsupported(
                "cell-saturated AIPW supports 1..=3 binary treatments",
            ));
        }
        if continuous.is_some() {
            return Err(EstimationError::unsupported(POINT_CDE_UNLICENSED));
        }
        let mut prepared = prepare_cells(data, treatments, outcome, adjustment, continuous)?;
        if let Some(ids) = fold_ids {
            if ids.len() != prepared.n {
                return Err(EstimationError::data_msg(
                    "shared fold assignment length must match complete-case rows",
                ));
            }
            prepared.fold_assignment = Some(ids.to_vec());
        }
        if let Some(shared) = design {
            if shared.len() != prepared.design.len() {
                return Err(EstimationError::data_msg(
                    "shared covariate design length must match the prepared [1 | Z] matrix",
                ));
            }
            prepared.design = shared.to_vec();
            prepared.shared_design = true;
        }
        let thresholds = if functional.quantile_level().is_some() {
            crate::quantile::empirical_threshold_grid(&prepared.outcome, 19)?
                .into_iter()
                .map(Some)
                .collect()
        } else {
            thresholds_of(functional)
        };
        crossfit_cell_scores(&prepared, &thresholds, self)
    }
}

/// Optional discretized continuous coordinate alongside the binary cells.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousCellSpec<'a> {
    /// Continuous variable.
    pub variable: VariableId,
    /// Declared grid; each observed value is snapped to the nearest point.
    pub grid: &'a [f64],
}

struct PreparedCells {
    n: usize,
    ncols: usize,
    design: Vec<f64>,
    cell: Vec<u32>,
    n_cells: usize,
    outcome: Vec<f64>,
    row_index: Vec<u32>,
    adjustment_set: Arc<[VariableId]>,
    treatments: Arc<[VariableId]>,
    fold_assignment: Option<Vec<u32>>,
    shared_design: bool,
}

fn prepare_cells(
    data: &TabularData,
    treatments: &[VariableId],
    outcome: VariableId,
    adjustment: &[VariableId],
    continuous: Option<ContinuousCellSpec<'_>>,
) -> Result<PreparedCells, EstimationError> {
    let mut ids = Vec::with_capacity(2 + treatments.len() + adjustment.len());
    ids.extend_from_slice(treatments);
    ids.push(outcome);
    ids.extend_from_slice(adjustment);
    if continuous.is_some() {
        return Err(EstimationError::unsupported(POINT_CDE_UNLICENSED));
    }
    if let Some(c) = continuous {
        ids.push(c.variable);
    }
    let mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let t_cols: Vec<Vec<f64>> = treatments
        .iter()
        .map(|&id| data.float64_masked(id, &mask).map_err(EstimationError::from))
        .collect::<Result<_, _>>()?;
    let y = data.float64_masked(outcome, &mask).map_err(EstimationError::from)?;
    let n = y.len();
    if n == 0 {
        return Err(EstimationError::data_msg("no complete-case rows for cell AIPW"));
    }
    let k = treatments.len();
    let mut n_cells = 1usize << k;
    let mut extra: Option<Vec<usize>> = None;
    if let Some(spec) = continuous {
        if spec.grid.is_empty()
            || spec.grid.iter().any(|v| !v.is_finite())
            || spec.grid.windows(2).any(|p| p[0] >= p[1])
            || treatments.contains(&spec.variable)
            || adjustment.contains(&spec.variable)
            || spec.variable == outcome
        {
            return Err(EstimationError::data_msg(
                "continuous cell grid must be finite, increasing, non-empty and use a distinct coordinate",
            ));
        }
        let d = data.float64_masked(spec.variable, &mask).map_err(EstimationError::from)?;
        let snapped: Vec<usize> = d.iter().map(|&v| nearest_index(spec.grid, v)).collect();
        n_cells *= spec.grid.len();
        extra = Some(snapped);
    }
    let mut cell = vec![0u32; n];
    for i in 0..n {
        let mut code = 0u32;
        for (j, col) in t_cols.iter().enumerate() {
            let v = col[i];
            if v.abs() > 1e-12 && (v - 1.0).abs() > 1e-12 {
                return Err(EstimationError::data_msg(
                    "cell-saturated AIPW requires binary 0/1 treatments",
                ));
            }
            if v > 0.5 {
                code |= 1 << j;
            }
        }
        if let Some(snapped) = extra.as_ref() {
            code += u32::try_from(snapped[i] << k).unwrap_or(0);
        }
        cell[i] = code;
    }
    {
        for c in 0..n_cells {
            if !cell.iter().any(|&code| code as usize == c) {
                return Err(EstimationError::unsupported(
                    "unsupported cell: empty joint treatment combination",
                ));
            }
        }
    }
    let ncols = 1 + adjustment.len();
    let mut design = vec![0.0; n * ncols];
    for r in 0..n {
        design[r] = 1.0;
    }
    for (j, &z) in adjustment.iter().enumerate() {
        let col = data.float64_masked(z, &mask).map_err(EstimationError::from)?;
        let base = (1 + j) * n;
        design[base..base + n].copy_from_slice(&col[..n]);
    }
    let mut row_index = Vec::with_capacity(n);
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            row_index.push(u32::try_from(i).unwrap_or(u32::MAX));
        }
    }
    Ok(PreparedCells {
        n,
        ncols,
        design,
        cell,
        n_cells,
        outcome: y,
        row_index,
        adjustment_set: Arc::from(adjustment.to_vec()),
        treatments: treatments
            .iter()
            .copied()
            .chain(continuous.map(|s| s.variable))
            .collect::<Vec<_>>()
            .into(),
        fold_assignment: None,
        shared_design: false,
    })
}

fn nearest_index(grid: &[f64], value: f64) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &g) in grid.iter().enumerate() {
        let d = (g - value).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn crossfit_cell_scores(
    prepared: &PreparedCells,
    thresholds: &[Option<f64>],
    est: &CellSaturatedAipw,
) -> Result<ScoreTable, EstimationError> {
    let n = prepared.n;
    let folds = est.folds;
    if folds < 2 || n < folds {
        return Err(EstimationError::data_msg("cell AIPW cross-fitting requires folds in 2..=n"));
    }
    let n_cells = prepared.n_cells;
    let mut columns = Vec::new();
    for &thr in thresholds {
        for arm in 0..n_cells {
            columns
                .push(ScoreColumn { arm: u32::try_from(arm).unwrap_or(u32::MAX), threshold: thr });
        }
    }
    let mut scores = vec![0.0; n * columns.len()];
    let mut propensities = vec![0.0; n * columns.len()];
    let fold_ids: Vec<u32> = match prepared.fold_assignment.as_deref() {
        Some(ids) if ids.len() == n => ids.to_vec(),
        Some(_) => {
            return Err(EstimationError::data_msg(
                "shared fold assignment length must match complete-case rows",
            ));
        }
        None => (0..n).map(|i| (prepared.row_index[i] as usize % folds) as u32).collect(),
    };

    if fold_ids.iter().any(|&id| id as usize >= folds) {
        return Err(EstimationError::data_msg("shared fold ids must lie in 0..folds"));
    }
    let clip = clip_of(est.overlap);
    let mut out_ws = crate::aipw::AipwWorkspace::default();

    for fold in 0..folds {
        let train: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize != fold).collect();
        let valid: Vec<usize> = (0..n).filter(|&i| fold_ids[i] as usize == fold).collect();
        let mut design_train = Vec::new();
        select_rows_colmajor(&prepared.design, n, prepared.ncols, &train, &mut design_train);
        let mut design_valid = Vec::new();
        select_rows_colmajor(&prepared.design, n, prepared.ncols, &valid, &mut design_valid);

        let train_cells: Vec<_> = train.iter().map(|&i| prepared.cell[i] as usize).collect();
        for c in 0..n_cells {
            if !train_cells.contains(&c) {
                return Err(EstimationError::unsupported(
                    "unsupported cell: cross-fit training fold is missing a joint treatment combination",
                ));
            }
        }
        let e_valid = multinomial_propensity(
            &design_train,
            train.len(),
            &train_cells,
            &design_valid,
            valid.len(),
            prepared.ncols,
            n_cells,
            est.backend,
        )?;

        for (t_idx, &threshold) in thresholds.iter().enumerate() {
            for c in 0..n_cells {
                for (k, &i) in valid.iter().enumerate() {
                    propensities[(t_idx * n_cells + c) * n + i] = e_valid[c * valid.len() + k];
                }
            }
            let y_all = match threshold {
                None => prepared.outcome.clone(),
                Some(c) => prepared.outcome.iter().map(|&yi| f64::from(yi > c)).collect(),
            };
            for c in 0..n_cells {
                let cell_train: Vec<usize> =
                    train.iter().copied().filter(|&i| prepared.cell[i] as usize == c).collect();
                let mu = if cell_train.len() < prepared.ncols {
                    let y_c: Vec<f64> = cell_train.iter().map(|&i| y_all[i]).collect();
                    y_c.iter().sum::<f64>() / y_c.len().max(1) as f64
                } else {
                    let mut design_c = Vec::new();
                    select_rows_colmajor(
                        &prepared.design,
                        n,
                        prepared.ncols,
                        &cell_train,
                        &mut design_c,
                    );
                    let y_c: Vec<f64> = cell_train.iter().map(|&i| y_all[i]).collect();
                    let fit = est
                        .backend
                        .least_squares(
                            &design_c,
                            cell_train.len(),
                            prepared.ncols,
                            &y_c,
                            &mut out_ws.outcome,
                        )
                        .map_err(stats_err)?;
                    let mut pred = Vec::new();
                    predict_colmajor(
                        &design_valid,
                        valid.len(),
                        prepared.ncols,
                        &fit.coefficients,
                        &mut pred,
                    );
                    let col = t_idx * n_cells + c;
                    for (k, &i) in valid.iter().enumerate() {
                        let a = f64::from(prepared.cell[i] as usize == c);
                        let e = e_valid[c * valid.len() + k].max(clip.unwrap_or(1e-8));
                        scores[col * n + i] = pred[k] + (a / e) * (y_all[i] - pred[k]);
                    }
                    continue;
                };
                let col = t_idx * n_cells + c;
                for (k, &i) in valid.iter().enumerate() {
                    let a = f64::from(prepared.cell[i] as usize == c);
                    let e = e_valid[c * valid.len() + k].max(clip.unwrap_or(1e-8));
                    scores[col * n + i] = mu + (a / e) * (y_all[i] - mu);
                }
            }
        }
    }

    Ok(ScoreTable {
        observed_arm: prepared.cell.clone().into(),
        propensities: propensities.into(),
        observed_outcome: prepared.outcome.clone().into(),
        n_rows: n,
        row_index: Arc::from(prepared.row_index.clone()),
        fold_ids: Arc::from(fold_ids),
        n_folds: u32::try_from(folds).unwrap_or(u32::MAX),
        scores: Arc::from(scores),
        columns: Arc::from(columns),
        adjustment_set: Arc::clone(&prepared.adjustment_set),
        nuisance_provenance: Arc::from(
            if prepared.fold_assignment.is_some() || prepared.shared_design {
                "cell.aipw.crossfit.multinomial_logit.ols.v1;batch.shared_design"
            } else {
                "cell.aipw.crossfit.multinomial_logit.ols.v1"
            },
        ),
        treatment: prepared.treatments[0],
        intervened: Arc::clone(&prepared.treatments),
    })
}

// Reference-cell multinomial likelihood with Newton steps and backtracking.
// A tiny diagonal numerical stabilization is used only in the Newton solve;
// acceptance and convergence use the unpenalized likelihood/score.
fn multinomial_propensity(
    train: &[f64],
    n: usize,
    cells: &[usize],
    valid: &[f64],
    nv: usize,
    p: usize,
    k: usize,
    backend: FaerBackend,
) -> Result<Vec<f64>, EstimationError> {
    let d = (k - 1) * p;
    let mut beta = vec![0.0; d];
    let probabilities = |x: &[f64], rows: usize, b: &[f64]| {
        let mut out = vec![0.0; rows * k];
        for r in 0..rows {
            let mut max_eta: f64 = 0.0;
            for c in 0..k - 1 {
                let eta = (0..p).map(|j| x[j * rows + r] * b[c * p + j]).sum::<f64>();
                out[c * rows + r] = eta;
                max_eta = max_eta.max(eta);
            }
            let mut mass = (-max_eta).exp();
            out[(k - 1) * rows + r] = mass;
            for c in 0..k - 1 {
                out[c * rows + r] = (out[c * rows + r] - max_eta).exp();
                mass += out[c * rows + r];
            }
            for c in 0..k {
                out[c * rows + r] /= mass;
            }
        }
        out
    };
    let loss = |prob: &[f64]| -> f64 {
        cells
            .iter()
            .enumerate()
            .map(|(r, &c)| -prob[c * n + r].max(f64::MIN_POSITIVE).ln())
            .sum::<f64>()
            / n as f64
    };
    let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
    for _ in 0..100 {
        let prob = probabilities(train, n, &beta);
        let mut gradient = vec![0.0; d];
        let mut hessian = vec![0.0; d * d];
        for r in 0..n {
            for c in 0..k - 1 {
                for j in 0..p {
                    let a = c * p + j;
                    gradient[a] +=
                        train[j * n + r] * (prob[c * n + r] - f64::from(cells[r] == c)) / n as f64;
                    for e in 0..k - 1 {
                        for l in 0..p {
                            let b = e * p + l;
                            hessian[b * d + a] += train[j * n + r]
                                * train[l * n + r]
                                * prob[c * n + r]
                                * (f64::from(c == e) - prob[e * n + r])
                                / n as f64;
                        }
                    }
                }
            }
        }
        if gradient.iter().all(|v| v.abs() < 1e-8) {
            return Ok(probabilities(valid, nv, &beta));
        }
        for j in 0..d {
            hessian[j * d + j] += 1e-10;
        }
        let step = backend
            .least_squares(&hessian, d, d, &gradient, &mut ws)
            .map_err(stats_err)?
            .coefficients;
        let old = loss(&prob);
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let trial: Vec<_> = beta.iter().zip(&step).map(|(b, s)| b - scale * s).collect();
            if loss(&probabilities(train, n, &trial)) < old {
                beta = trial;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return Err(EstimationError::data_msg("multinomial propensity failed to converge"));
        }
    }
    Err(EstimationError::data_msg("multinomial propensity exceeded iteration limit"))
}

/// 2×2 interaction contrast `(μ_11 − μ_10) − (μ_01 − μ_00)` on the first
/// four cells of a mean-functional table.
///
/// # Errors
///
/// Fewer than four mean columns.
pub fn interaction_contrast(table: &ScoreTable) -> Result<LinearContrast, EstimationError> {
    contrast_named(table, "interaction")
}

/// Requested-cell minus all-zero control, with score-difference influence.
///
/// # Errors
///
/// Missing cells, an exceedance grid, or a contrast that cannot be formed.
pub fn cell_minus_control_contrast(
    table: &ScoreTable,
    requested_arm: u32,
) -> Result<(LinearContrast, Vec<f64>), EstimationError> {
    family_cell_contrast(table, requested_arm, "cell_minus_control")
}

/// Named cell contrast and its per-row score-difference influence.
///
/// `cell_minus_control` is requested cell minus arm 0. `interaction` is
/// `+μ00 −μ10 −μ01 +μ11` on a 2×2 table.
///
/// # Errors
///
/// Unknown name, a grid, or an unsupported cell.
pub fn family_cell_contrast(
    table: &ScoreTable,
    requested_arm: u32,
    spec: &str,
) -> Result<(LinearContrast, Vec<f64>), EstimationError> {
    let summary = table.summarize(None)?;
    let coeffs = match spec {
        "cell_minus_control" => cell_minus_control_coefficients(table, requested_arm)?,
        "interaction" => {
            let coeffs = interaction_coefficients(table)?;
            let first = table.columns.first().and_then(|c| c.threshold);
            let n_mean = table.columns.iter().filter(|c| c.threshold == first).count();
            if n_mean != 4 {
                return Err(EstimationError::unsupported(
                    "family interaction contrast requires a 2×2 cell table",
                ));
            }
            coeffs
        }
        _ => {
            return Err(EstimationError::unsupported("unknown cell contrast"));
        }
    };
    let contrast = table.linear_contrast(&summary, &coeffs)?;
    let scores = table.combine_scores(&coeffs)?;
    Ok((contrast, scores))
}

/// Coefficients for requested cell minus the all-zero control on the first
/// threshold slice.
///
/// # Errors
///
/// Missing cells or an exceedance grid.
pub fn cell_minus_control_coefficients(
    table: &ScoreTable,
    requested_arm: u32,
) -> Result<Vec<f64>, EstimationError> {
    if distinct_threshold_count(table) > 1 {
        return Err(EstimationError::unsupported(
            "cell_minus_control on an exceedance grid is not licensed; request a single threshold",
        ));
    }
    let first = table.columns.first().and_then(|c| c.threshold);
    let req = table
        .columns
        .iter()
        .position(|c| c.arm == requested_arm && c.threshold == first)
        .ok_or(EstimationError::unsupported("unsupported requested cell"))?;
    let ctl = table
        .columns
        .iter()
        .position(|c| c.arm == 0 && c.threshold == first)
        .ok_or(EstimationError::unsupported("control cell is missing"))?;
    let mut coeffs = vec![0.0; table.n_columns()];
    if req != ctl {
        coeffs[ctl] = -1.0;
        coeffs[req] = 1.0;
    }
    Ok(coeffs)
}

/// Coefficients `+μ00 −μ10 −μ01 +μ11` on the first four mean cells.
///
/// # Errors
///
/// An exceedance grid or fewer than four mean columns.
pub fn interaction_coefficients(table: &ScoreTable) -> Result<Vec<f64>, EstimationError> {
    if distinct_threshold_count(table) > 1 {
        return Err(EstimationError::unsupported(
            "interaction contrast on an exceedance grid is not licensed; request a single Exceedance threshold or a raw coefficient vector",
        ));
    }
    let first_threshold = table.columns.first().and_then(|c| c.threshold);
    let mean_cols: Vec<usize> = table
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.threshold == first_threshold)
        .map(|(i, _)| i)
        .collect();
    if mean_cols.len() < 4 {
        return Err(EstimationError::unsupported("interaction contrast requires a 2×2 cell table"));
    }
    let mut coeffs = vec![0.0; table.n_columns()];
    coeffs[mean_cols[0]] = 1.0;
    coeffs[mean_cols[1]] = -1.0;
    coeffs[mean_cols[2]] = -1.0;
    coeffs[mean_cols[3]] = 1.0;
    Ok(coeffs)
}

fn distinct_threshold_count(table: &ScoreTable) -> usize {
    let mut thresholds: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    thresholds.len()
}

/// Named or raw linear cell contrast.
///
/// `interaction` is the 2×2 interaction on cells `00,01,10,11`.
/// A raw contrast is `c0,c1,...` coefficients matching the mean columns.
///
/// # Errors
///
/// Unknown name or unsupported cell.
pub fn contrast_named(table: &ScoreTable, spec: &str) -> Result<LinearContrast, EstimationError> {
    let summary = table.summarize(None)?;
    contrast_from_summary(table, &summary, spec)
}

/// Evaluate a named contrast against a supplied (possibly weighted) summary.
pub fn contrast_from_summary(
    table: &ScoreTable,
    summary: &ScoreSummary,
    spec: &str,
) -> Result<LinearContrast, EstimationError> {
    let coeffs = if spec == "interaction" {
        interaction_coefficients(table)?
    } else if spec == "cell_minus_control" {
        return Err(EstimationError::unsupported(
            "cell_minus_control requires a requested arm; use family_cell_contrast",
        ));
    } else if let Some(raw) = parse_coefficients(spec) {
        if raw.len() != table.n_columns() {
            return Err(EstimationError::data_msg("contrast length must match score columns"));
        }
        raw
    } else {
        return Err(EstimationError::unsupported("unknown cell contrast"));
    };
    table.linear_contrast(summary, &coeffs)
}

/// Joint IF summary plus a named contrast.
///
/// # Errors
///
/// Contrast failure.
pub fn summarize_with_contrast(
    table: &ScoreTable,
    spec: &str,
    weights: Option<&[f64]>,
) -> Result<(ScoreSummary, LinearContrast), EstimationError> {
    let summary = table.summarize(weights)?;
    let contrast = contrast_from_summary(table, &summary, spec)?;
    Ok((summary, contrast))
}

fn parse_coefficients(spec: &str) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        out.push(part.trim().parse().ok()?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_kernels::standard_normal;

    fn interaction_dgp(n: usize) -> TabularData {
        let mut rng = ExecutionContext::for_tests(9).rng.stream(0xC11);
        let mut a = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            z[i] = zi;
            a[i] = f64::from(rng.next_f64() < 0.5);
            d[i] = f64::from(rng.next_f64() < 0.5);
            y[i] = 1.5 * a[i] * d[i] + 0.2 * zi + 0.25 * standard_normal(&mut rng);
        }
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("a", RoleHint::TreatmentCandidate),
            ("d", RoleHint::TreatmentCandidate),
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
        let cols = [a, d, y, z]
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
        TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
    }

    #[test]
    fn weighted_contrast_uses_target_summary_and_exceedance_cells() {
        let data = interaction_dgp(1000);
        let table = CellSaturatedAipw::new()
            .fit_scores(
                &data,
                &[VariableId::from_raw(0), VariableId::from_raw(1)],
                VariableId::from_raw(2),
                &[VariableId::from_raw(3)],
                &OutcomeFunctional::exceedance(0.5),
                None,
            )
            .unwrap();
        let weights: Vec<_> =
            (0..table.n_rows).map(|i| if i % 3 == 0 { 4.0 } else { 1.0 }).collect();
        let (summary, contrast) =
            summarize_with_contrast(&table, "interaction", Some(&weights)).unwrap();
        let expected = table.linear_contrast(&summary, &[1.0, -1.0, -1.0, 1.0]).unwrap();
        assert_eq!(contrast, expected);
    }

    #[test]
    fn missing_training_cell_refuses_instead_of_fabricating_scores() {
        let prepared = PreparedCells {
            n: 10,
            ncols: 1,
            design: vec![1.0; 10],
            cell: vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            n_cells: 2,
            outcome: vec![1.0; 10],
            row_index: (0..10).collect(),
            adjustment_set: Arc::from([]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            fold_assignment: None,
            shared_design: false,
        };
        let err = crossfit_cell_scores(&prepared, &[None], &CellSaturatedAipw::new()).unwrap_err();
        assert!(err.to_string().contains("training fold"));
    }

    #[test]
    fn cell_minus_control_is_requested_minus_arm_zero() {
        let data = interaction_dgp(2_000);
        let table = CellSaturatedAipw::new()
            .fit_scores(
                &data,
                &[VariableId::from_raw(0), VariableId::from_raw(1)],
                VariableId::from_raw(2),
                &[VariableId::from_raw(3)],
                &OutcomeFunctional::Mean,
                None,
            )
            .unwrap();
        let summary = table.summarize(None).unwrap();
        let arm11 = table.columns.iter().position(|c| c.arm == 3).unwrap();
        let (contrast, scores) = cell_minus_control_contrast(&table, 3).unwrap();
        assert!((contrast.value - (summary.means[arm11] - summary.means[0])).abs() < 1e-12);
        assert_eq!(scores.len(), table.n_rows);
        let expected = table.combine_scores(&cell_minus_control_coefficients(&table, 3).unwrap());
        assert_eq!(scores, expected.unwrap());
    }

    #[test]
    fn pure_interaction_recovers_truth() {
        let data = interaction_dgp(2_000);
        let est = CellSaturatedAipw::new();
        let table = est
            .fit_scores(
                &data,
                &[VariableId::from_raw(0), VariableId::from_raw(1)],
                VariableId::from_raw(2),
                &[VariableId::from_raw(3)],
                &OutcomeFunctional::Mean,
                None,
            )
            .unwrap();
        let contrast = interaction_contrast(&table).unwrap();
        assert!(
            (contrast.value - 1.5).abs() < 0.35,
            "interaction={} se={}",
            contrast.value,
            contrast.se
        );
        assert!(contrast.se > 0.0);
    }

    #[test]
    fn continuous_cell_refuses_point_cde() {
        let data = interaction_dgp(80);
        let err = CellSaturatedAipw::new()
            .fit_scores(
                &data,
                &[VariableId::from_raw(0)],
                VariableId::from_raw(2),
                &[VariableId::from_raw(3)],
                &OutcomeFunctional::Mean,
                Some(ContinuousCellSpec { variable: VariableId::from_raw(1), grid: &[0.0, 1.0] }),
            )
            .unwrap_err();
        assert!(err.to_string().contains("do(D=d0)"), "{err}");
    }

    #[test]
    fn unsupported_cell_refuses() {
        let data = interaction_dgp(200);
        let est = CellSaturatedAipw::new();
        let table = est
            .fit_scores(
                &data,
                &[VariableId::from_raw(0), VariableId::from_raw(1)],
                VariableId::from_raw(2),
                &[VariableId::from_raw(3)],
                &OutcomeFunctional::Mean,
                None,
            )
            .unwrap();
        let err = contrast_named(&table, "not-a-contrast").unwrap_err();
        assert!(err.to_string().contains("unknown cell contrast"));
    }
}
