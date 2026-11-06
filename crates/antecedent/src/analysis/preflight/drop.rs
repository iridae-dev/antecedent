//! Rank-drop admission, priority and independent span verification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::rank::{dot, l2, normalized_column};
use super::{
    CausalError, ColumnPriority, EstimationError, ExecutionContext, Frame, PreflightInput,
    RankDropPlan, RankDropPolicy, RefusalFields, STAGE_RANK_DROP, SpanCheck, TableView, VariableId,
    reason_code, scan_rank, tolerance,
};

pub(super) fn rank_drop_refusal(
    subject: &str,
    reason: &str,
    implicated: Vec<String>,
    remedy: &str,
) -> CausalError {
    let fields = RefusalFields {
        stage: Some(STAGE_RANK_DROP.to_string()),
        subject: Some(subject.to_string()),
        reason: Some(reason.to_string()),
        implicated_columns: implicated,
        remedy: Some(remedy.to_string()),
        ..RefusalFields::default()
    };
    EstimationError::refused_detail_with_fields(
        reason_code!("rank_drop_not_supported"),
        subject,
        format!("{reason}; nothing was dropped"),
        fields,
    )
    .into()
}

/// Resolves the declared column priority to adjustment-column positions, refusing a priority that does not cover the set.
pub(super) fn column_priority_order(
    input: &PreflightInput<'_>,
    policy: &RankDropPolicy,
    frame: &Frame<'_>,
) -> Result<Vec<usize>, CausalError> {
    let p = input.adjustment.len();
    Ok(match &policy.priority {
        ColumnPriority::AdjustmentOrder => (0..p).collect(),
        ColumnPriority::Declared(ids) => {
            let mut used = vec![false; p];
            let mut order = Vec::with_capacity(p);
            for id in ids {
                let slot = (0..p).find(|&k| !used[k] && input.adjustment[k] == *id);
                match slot {
                    Some(k) => {
                        used[k] = true;
                        order.push(k);
                    }
                    None => {
                        return Err(rank_drop_refusal(
                            &frame.subject,
                            "the declared column priority names a column that is not in the \
                             adjustment set (or names it twice)",
                            Vec::new(),
                            "declare every adjustment column exactly once, highest priority first",
                        ));
                    }
                }
            }
            if order.len() != p {
                let missing: Vec<String> = (0..p)
                    .filter(|&k| !used[k])
                    .map(|k| frame.adjustment_names()[k].clone())
                    .collect();
                return Err(rank_drop_refusal(
                    &frame.subject,
                    "the declared column priority does not cover the adjustment set",
                    missing,
                    "declare every adjustment column exactly once, highest priority first",
                ));
            }
            order
        }
    })
}

/// A dropped column's residual after projecting onto the retained span must stay within
/// this multiple of the scan tolerance (rounding slack for the second projection).
const SPAN_SLACK: f64 = 10.0;

/// The adjustment columns a plan keeps, in the input's order.
pub(in crate::analysis) fn kept_adjustment_ids(
    input: &PreflightInput<'_>,
    plan: &RankDropPlan,
) -> Result<Vec<VariableId>, CausalError> {
    let schema = input.data.schema();
    let mut kept = Vec::with_capacity(plan.kept_adjustment.len());
    for &id in &input.adjustment {
        let meta = schema.get(id)?;
        if plan.kept_adjustment.iter().any(|name| name.as_str() == &*meta.name) {
            kept.push(id);
        }
    }
    Ok(kept)
}

/// Residual norm over own norm of `values` after projecting out an orthonormal `basis`
/// (twice, for orthogonality). A zero column lies in every span.
fn residual_over_norm(basis: &[Vec<f64>], values: &[f64]) -> f64 {
    let squared = dot(values, values);
    if squared.is_normal() {
        let norm = squared.sqrt();
        let mut w = values.to_vec();
        for _ in 0..2 {
            for q in basis {
                let c = dot(q, &w);
                for (wi, qi) in w.iter_mut().zip(q) {
                    *wi -= c * qi;
                }
            }
        }
        return l2(&w) / norm;
    }
    let Some((mut w, _)) = normalized_column(values) else {
        return 0.0;
    };
    for _ in 0..2 {
        for q in basis {
            let c = dot(q, &w);
            for (wi, qi) in w.iter_mut().zip(q) {
                *wi -= c * qi;
            }
        }
    }
    l2(&w)
}

/// Verify, independently of the plan's scan, that dropping `plan.dropped` left the column
/// space unchanged: the retained design has full numerical rank equal to the original's, and
/// every dropped column lies in the retained span within the scan tolerance.
///
/// # Errors
///
/// `rank_drop_not_supported` when the retained design is still rank deficient, its rank
/// differs from the original, or a dropped column is not in the retained span; the
/// preflight errors (unknown column, size limit, cancellation).
pub(in crate::analysis) fn verify_span_preserved(
    input: &PreflightInput<'_>,
    plan: &RankDropPlan,
    kept: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<SpanCheck, CausalError> {
    let frame = Frame::new(input, ctx)?;
    let columns = frame.adjustment_columns();
    let ones = vec![1.0; frame.rows_complete];
    let keep_mask: Vec<bool> = input.adjustment.iter().map(|id| kept.contains(id)).collect();
    let mut design: Vec<&[f64]> = vec![&ones];
    design.extend(
        columns.iter().zip(&keep_mask).filter(|(_, keep)| **keep).map(|(c, _)| c.as_slice()),
    );
    let tol = tolerance(frame.rows_complete, input.adjustment.len() + 1);
    let scan = scan_rank(&design, tol, ctx)?;
    let retained_rank = scan.basis.len();
    if retained_rank != design.len() || retained_rank != plan.numerical_rank {
        return Err(rank_drop_refusal(
            &frame.subject,
            "rank_drop_estimate.span_not_preserved: the retained design does not span the \
             original column space (its numerical rank differs from the original's or it is \
             still rank deficient)",
            Vec::new(),
            "the drop would change the fitted projection; keep the original adjustment set",
        ));
    }
    let mut worst = 0.0_f64;
    let mut worst_column = String::new();
    for ((column, keep), name) in columns.iter().zip(&keep_mask).zip(frame.adjustment_names()) {
        if *keep {
            continue;
        }
        let ratio = residual_over_norm(&scan.basis, column);
        if ratio > worst {
            worst = ratio;
            worst_column.clone_from(name);
        }
    }
    if worst > SPAN_SLACK * tol {
        return Err(rank_drop_refusal(
            &frame.subject,
            "rank_drop_estimate.span_not_preserved: a dropped column is not an exact linear \
             combination of the retained columns",
            vec![worst_column],
            "the drop would change the fitted projection; keep the original adjustment set",
        ));
    }
    Ok(SpanCheck {
        original_rank: plan.numerical_rank,
        retained_rank,
        max_dropped_residual_ratio: worst,
        tolerance: SPAN_SLACK * tol,
    })
}
