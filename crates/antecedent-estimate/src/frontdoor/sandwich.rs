//! Stacked sandwich covariance and delta-method standard errors.

use super::{STAGE1_NCOLS, STAGE1_TREATMENT_COL, STAGE2_FIRST_MEDIATOR_COL};
use crate::error::EstimationError;
use crate::se::{AnalyticSeKind, require_clusters};

/// Whether the stacked meat keeps empirical cross-stage blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CrossStagePolicy {
    /// Full stacked HC meat (includes Cov across stages/mediators).
    Empirical,
    /// Zero cross-stage / cross-mediator blocks (independent-path Sobel delta).
    IndependentPaths,
}

/// Shared stage-1 / stage-2 views for stacked front-door sandwich SE.
#[derive(Clone, Copy, Debug)]
pub(super) struct StackedStageViews<'a> {
    pub(super) x1: &'a [f64],
    pub(super) stage1_resid: &'a [Vec<f64>],
    pub(super) stage1_coefs: &'a [Vec<f64>],
    pub(super) x2: &'a [f64],
    pub(super) stage2_ncols: usize,
    pub(super) stage2_resid: &'a [f64],
    pub(super) stage2_coefs: &'a [f64],
    pub(super) treatment_delta: f64,
}

pub(super) fn stacked_path_sum_se(
    stages: &StackedStageViews<'_>,
    se_kind: AnalyticSeKind,
    cluster_ids: Option<&[u32]>,
    cross_stage: CrossStagePolicy,
) -> Result<f64, EstimationError> {
    let (cov, grad) = stacked_theta_cov_and_grad(stages, se_kind, cluster_ids, cross_stage)?;
    let n_params = grad.len();
    let mut tmp = vec![0.0; n_params];
    for row in 0..n_params {
        let mut sum = 0.0;
        for col in 0..n_params {
            sum += cov[row * n_params + col] * grad[col];
        }
        tmp[row] = sum;
    }
    let mut var = 0.0;
    for (row, &g) in grad.iter().enumerate() {
        var += g * tmp[row];
    }
    if !var.is_finite() {
        return Ok(f64::NAN);
    }
    Ok(var.max(0.0).sqrt())
}

pub(super) fn stacked_theta_cov_and_grad(
    stages: &StackedStageViews<'_>,
    se_kind: AnalyticSeKind,
    cluster_ids: Option<&[u32]>,
    cross_stage: CrossStagePolicy,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let n_mediators = stages.stage1_resid.len();
    if n_mediators == 0 || stages.stage1_coefs.len() != n_mediators {
        return Err(EstimationError::unsupported("stacked front-door SE needs ≥1 mediator"));
    }
    let n_rows = stages.stage2_resid.len();
    if stages.x1.len() < n_rows * STAGE1_NCOLS || stages.x2.len() < n_rows * stages.stage2_ncols {
        return Err(EstimationError::data_msg("stacked front-door design buffer too short"));
    }
    for resid in stages.stage1_resid {
        if resid.len() != n_rows {
            return Err(EstimationError::data_msg("stage-1 residual length mismatch"));
        }
    }
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;

    // Block-diagonal bread A = diag(X1'X1, …, X1'X1, X2'X2). The stages are independent, so
    // each distinct block is inverted once, from the QR of its own design (the Gram matrix is
    // never formed), and A⁻¹ is never assembled densely.
    let inv1 = crate::util::xtx_inverse(stages.x1, n_rows, STAGE1_NCOLS)
        .ok_or_else(|| EstimationError::stats_msg("singular stacked front-door bread"))?;
    let inv2 = crate::util::xtx_inverse(stages.x2, n_rows, stages.stage2_ncols)
        .ok_or_else(|| EstimationError::stats_msg("singular stacked front-door bread"))?;
    let stage2_off = n_mediators * STAGE1_NCOLS;
    // (offset, width, inverse of the block) for every diagonal block of A.
    let mut blocks: Vec<(usize, usize, &[f64])> = (0..n_mediators)
        .map(|mediator_idx| (mediator_idx * STAGE1_NCOLS, STAGE1_NCOLS, inv1.as_slice()))
        .collect();
    blocks.push((stage2_off, stages.stage2_ncols, inv2.as_slice()));

    let meat = match se_kind {
        AnalyticSeKind::Homoskedastic => {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("estimator_inference_mismatch"),
                "FrontDoorTwoStage stacked SE does not implement classical Homoskedastic meat; use Hc0",
            ));
        }
        AnalyticSeKind::Hc0 | AnalyticSeKind::Hc1 => {
            let mut meat = stacked_hc_meat(stages, cross_stage);
            if matches!(se_kind, AnalyticSeKind::Hc1) {
                scale_meat_by_equation_df(&mut meat, stages, |n, p| n / (n - p))?;
            }
            meat
        }
        AnalyticSeKind::Cluster => {
            let groups = require_clusters(cluster_ids, n_rows)?;
            stacked_cluster_meat(stages, groups, cross_stage)?
        }
        AnalyticSeKind::Hc2
        | AnalyticSeKind::Hc3
        | AnalyticSeKind::Multiway
        | AnalyticSeKind::NeweyWest { .. }
        | AnalyticSeKind::PanelClusterHac { .. } => {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("estimator_inference_mismatch"),
                "FrontDoorTwoStage stacked SE supports Hc0/Hc1/Cluster only",
            ));
        }
    };

    // Σ = A⁻¹ B A⁻¹, block by block: Σ_ij = inv_i · B_ij · inv_j.
    let mut cov = vec![0.0; n_params * n_params];
    for &(row_off, row_width, row_inv) in &blocks {
        for &(col_off, col_width, col_inv) in &blocks {
            // tmp = inv_i · B_ij  (row_width × col_width)
            let mut tmp = vec![0.0; row_width * col_width];
            for r in 0..row_width {
                for mid in 0..row_width {
                    let weight = row_inv[r * row_width + mid];
                    for c in 0..col_width {
                        tmp[r * col_width + c] +=
                            weight * meat[(row_off + mid) * n_params + col_off + c];
                    }
                }
            }
            // Σ_ij = tmp · inv_j
            for r in 0..row_width {
                for c in 0..col_width {
                    let mut sum = 0.0;
                    for mid in 0..col_width {
                        sum += tmp[r * col_width + mid] * col_inv[mid * col_width + c];
                    }
                    cov[(row_off + r) * n_params + col_off + c] = sum;
                }
            }
        }
    }

    // ∇g for g = Δ Σⱼ β_{T→Mⱼ} β_{Mⱼ→Y}
    let mut grad = vec![0.0; n_params];
    for mediator_idx in 0..n_mediators {
        let beta_t_to_m = stages.stage1_coefs[mediator_idx][STAGE1_TREATMENT_COL];
        let beta_m_to_y = stages.stage2_coefs[STAGE2_FIRST_MEDIATOR_COL + mediator_idx];
        grad[mediator_idx * STAGE1_NCOLS + STAGE1_TREATMENT_COL] =
            stages.treatment_delta * beta_m_to_y;
        grad[stage2_off + STAGE2_FIRST_MEDIATOR_COL + mediator_idx] =
            stages.treatment_delta * beta_t_to_m;
    }
    Ok((cov, grad))
}

fn fill_row_score(score: &mut [f64], stages: &StackedStageViews<'_>, row: usize, n_rows: usize) {
    let n_mediators = stages.stage1_resid.len();
    score.fill(0.0);
    for (mediator_idx, resid) in stages.stage1_resid.iter().enumerate() {
        let e1 = resid[row];
        let off = mediator_idx * STAGE1_NCOLS;
        for col in 0..STAGE1_NCOLS {
            score[off + col] = e1 * stages.x1[col * n_rows + row];
        }
    }
    let e2 = stages.stage2_resid[row];
    let stage2_off = n_mediators * STAGE1_NCOLS;
    for col in 0..stages.stage2_ncols {
        score[stage2_off + col] = e2 * stages.x2[col * n_rows + row];
    }
}

fn zero_cross_blocks(meat: &mut [f64], n_params: usize, n_mediators: usize, stage2_ncols: usize) {
    // Zero every off-diagonal stage block (independent-path Sobel).
    let blocks: Vec<(usize, usize)> = {
        let mut block_specs = Vec::with_capacity(n_mediators + 1);
        for mediator_idx in 0..n_mediators {
            block_specs.push((mediator_idx * STAGE1_NCOLS, STAGE1_NCOLS));
        }
        block_specs.push((n_mediators * STAGE1_NCOLS, stage2_ncols));
        block_specs
    };
    for (bi, &(o1, n1)) in blocks.iter().enumerate() {
        for (bj, &(o2, n2)) in blocks.iter().enumerate() {
            if bi == bj {
                continue;
            }
            for row in 0..n1 {
                for col in 0..n2 {
                    meat[(o1 + row) * n_params + (o2 + col)] = 0.0;
                }
            }
        }
    }
}

/// Apply a small-sample factor `factor(n, p)` equation by equation.
///
/// Each stacked regression has its own residual degrees of freedom (`p = 2` for a stage-1
/// `M ~ 1 + T` fit, `p = 2 + k` for stage 2), so its scores are scaled by the square root of
/// its own factor: meat entry `(i, j)` is multiplied by `sqrt(f_i · f_j)`. This keeps "HC1" and
/// the cluster correction equal to their single-equation definitions on the diagonal blocks
/// instead of charging every coefficient for the parameters of the other regressions.
fn scale_meat_by_equation_df(
    meat: &mut [f64],
    stages: &StackedStageViews<'_>,
    factor: impl Fn(f64, f64) -> f64,
) -> Result<(), EstimationError> {
    let n_mediators = stages.stage1_resid.len();
    let n_rows = stages.stage2_resid.len();
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;
    if n_rows <= STAGE1_NCOLS.max(stages.stage2_ncols) {
        return Err(EstimationError::stats_msg("non-positive residual df for stacked SE"));
    }
    let root: Vec<f64> = (0..n_params)
        .map(|param| {
            let p =
                if param < n_mediators * STAGE1_NCOLS { STAGE1_NCOLS } else { stages.stage2_ncols };
            factor(n_rows as f64, p as f64).sqrt()
        })
        .collect();
    for i in 0..n_params {
        for j in 0..n_params {
            meat[i * n_params + j] *= root[i] * root[j];
        }
    }
    Ok(())
}

fn stacked_hc_meat(stages: &StackedStageViews<'_>, cross_stage: CrossStagePolicy) -> Vec<f64> {
    let n_mediators = stages.stage1_resid.len();
    let n_rows = stages.stage2_resid.len();
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;
    let mut meat = vec![0.0; n_params * n_params];
    let mut score = vec![0.0; n_params];
    for row in 0..n_rows {
        fill_row_score(&mut score, stages, row, n_rows);
        for i in 0..n_params {
            for j in 0..n_params {
                meat[i * n_params + j] += score[i] * score[j];
            }
        }
    }
    if cross_stage == CrossStagePolicy::IndependentPaths {
        zero_cross_blocks(&mut meat, n_params, n_mediators, stages.stage2_ncols);
    }
    meat
}

fn stacked_cluster_meat(
    stages: &StackedStageViews<'_>,
    groups: &[u32],
    cross_stage: CrossStagePolicy,
) -> Result<Vec<f64>, EstimationError> {
    let n_mediators = stages.stage1_resid.len();
    let n_rows = stages.stage2_resid.len();
    let n_params = n_mediators * STAGE1_NCOLS + stages.stage2_ncols;
    let mut totals: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    let mut score = vec![0.0; n_params];
    for (row, &group_id) in groups.iter().enumerate().take(n_rows) {
        fill_row_score(&mut score, stages, row, n_rows);
        let entry = totals.entry(group_id).or_insert_with(|| vec![0.0; n_params]);
        for (col, &s) in score.iter().enumerate() {
            entry[col] += s;
        }
    }
    let n_clusters = totals.len();
    if n_clusters < 2 {
        return Err(crate::se::few_clusters(n_clusters));
    }
    let mut meat = vec![0.0; n_params * n_params];
    for score_g in totals.values() {
        for i in 0..n_params {
            for j in 0..n_params {
                meat[i * n_params + j] += score_g[i] * score_g[j];
            }
        }
    }
    if cross_stage == CrossStagePolicy::IndependentPaths {
        zero_cross_blocks(&mut meat, n_params, n_mediators, stages.stage2_ncols);
    }
    let g = n_clusters as f64 / (n_clusters as f64 - 1.0);
    scale_meat_by_equation_df(&mut meat, stages, |n, p| g * (n - 1.0) / (n - p))?;
    Ok(meat)
}
