//! Candidate selection and estimation-row ownership.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CandidateProcedure, CandidateScreen, CandidateSelection, CausalError, StudyResult, TableView,
    TabularData, ValidityBitmap,
};

/// Membership bitmap (little-endian bit order) of `rows` over an `n`-row table, or `None` when a
/// row is outside the table or repeats.
pub(super) fn screen_row_bitmap(rows: &[antecedent_data::GlobalId], n: usize) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for &r in rows {
        let i = usize::try_from(r.get()).ok().filter(|&i| i < n)?;
        let bit = 1u8 << (i % 8);
        if bytes[i / 8] & bit != 0 {
            return None;
        }
        bytes[i / 8] |= bit;
    }
    Some(bytes)
}

pub(super) fn subset_estimate_rows(
    data: &TabularData,
    screen: Option<&CandidateScreen>,
) -> Result<TabularData, CausalError> {
    let Some(screen) = screen else {
        return Ok(data.clone());
    };
    if screen.screen_id.is_empty()
        || screen.estimate_rows.is_empty()
        || screen.screen_rows.is_empty()
    {
        return Err(CausalError::Compile {
            message:
                "recorded candidate screens require an id and nonempty screen/estimate row sets"
                    .into(),
        });
    }
    let n = data.row_count();
    let invalid = || CausalError::Compile {
        message: "candidate-selection row indexes must be unique and within the table".into(),
    };
    // One pass per row set checks bounds and duplicates while it builds the membership bitmap.
    screen_row_bitmap(&screen.screen_rows, n).ok_or_else(invalid)?;
    let bytes = screen_row_bitmap(&screen.estimate_rows, n).ok_or_else(invalid)?;
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(CausalError::from)?;
    data.with_analysis_mask(mask).map_err(CausalError::from)
}

/// Resolve the max-t critical value for a joint covariance, attaching a
/// diagnostic to every result when the band cannot be formed.
///
/// `max_t_critical` refuses on a non-PSD correlation matrix, a non-finite or
/// non-positive SE, or an asymmetric covariance. Unlike `.ok()`, this makes
/// that refusal visible: "not claimed" (no family declared) and "computation
/// failed" must read differently on the result.
pub(super) fn resolve_max_t_critical(
    results: &mut [StudyResult],
    cov: &antecedent_estimate::JointCovariance,
    what: &str,
) -> Option<f64> {
    match antecedent_estimate::max_t_critical(cov, 0.95, 4096, 1) {
        Ok(c) => Some(c),
        Err(err) => {
            for result in results.iter_mut() {
                result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.joint_if.max_t_unavailable",
                    antecedent_core::DiagnosticKind::Scientific,
                    antecedent_core::DiagnosticSeverity::Warning,
                    format!("max-t band unavailable for the {what}: {err}"),
                ));
            }
            None
        }
    }
}

/// Index of the smallest finite value (the first on ties); `None` when no value is finite, so a
/// NaN q-value can never be chosen or make the choice depend on the order of the slice.
pub(super) fn argmin_finite(values: &[f64]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

fn rows_disjoint(a: &[antecedent_data::GlobalId], b: &[antecedent_data::GlobalId]) -> bool {
    let mut left = a.to_vec();
    let mut right = b.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left.dedup();
    right.dedup();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => return false,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    true
}

#[allow(
    clippy::too_many_lines,
    reason = "ranks the family by the screen's procedure (BH, BY, max-t or none) and writes one selection record onto every result; the procedures share the same result walk"
)]
pub(super) fn attach_candidate_selection(
    results: &mut [StudyResult],
    screen: Option<&CandidateScreen>,
    bh: &[f64],
    by: &[f64],
    rank_stats: Option<&[(f64, f64)]>,
) {
    let family_size = results.len();
    let (selection, recorded) = match screen {
        None => (
            CandidateSelection {
                screen_id: std::sync::Arc::from("unrecorded"),
                procedure: CandidateProcedure::Unrecorded,
                winner_index: None,
                family_size,
                screen_rows: std::sync::Arc::from([]),
                estimate_rows: std::sync::Arc::from([]),
                disjoint: false,
            },
            false,
        ),
        Some(screen) => {
            let disjoint = rows_disjoint(&screen.screen_rows, &screen.estimate_rows);
            let winner_index = match screen.procedure {
                CandidateProcedure::BenjaminiHochberg => argmin_finite(bh),
                CandidateProcedure::BenjaminiYekutieli => argmin_finite(by),
                CandidateProcedure::Unrecorded => None,
                CandidateProcedure::MaxT => rank_stats.map_or_else(
                    || {
                        results
                            .iter()
                            .enumerate()
                            .filter(|(_, result)| {
                                result.estimate.as_effect().is_some_and(|effect| {
                                    effect.ate.is_finite()
                                        && effect.se_analytic().is_finite()
                                        && effect.se_analytic() >= 0.0
                                })
                            })
                            .max_by(|a, b| {
                                let score = |result: &StudyResult| {
                                    result.estimate.as_effect().map(|effect| {
                                        let scale = effect.se_analytic().abs().max(1e-12);
                                        effect.ate.abs() / scale
                                    })
                                };
                                score(a.1)
                                    .partial_cmp(&score(b.1))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    },
                    |stats| {
                        stats
                            .iter()
                            .enumerate()
                            .filter(|(_, (value, se))| {
                                value.is_finite() && se.is_finite() && *se >= 0.0
                            })
                            .max_by(|a, b| {
                                let sa = a.1.1.abs().max(1e-12);
                                let sb = b.1.1.abs().max(1e-12);
                                (a.1.0.abs() / sa)
                                    .partial_cmp(&(b.1.0.abs() / sb))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    },
                ),
            };
            (
                CandidateSelection {
                    screen_id: std::sync::Arc::clone(&screen.screen_id),
                    procedure: screen.procedure,
                    winner_index,
                    family_size,
                    screen_rows: std::sync::Arc::clone(&screen.screen_rows),
                    estimate_rows: std::sync::Arc::clone(&screen.estimate_rows),
                    disjoint,
                },
                true,
            )
        }
    };
    for result in results.iter_mut() {
        result.extras.candidate_selection = Some(selection.clone());
        if let Some(effect) = result.estimate.as_effect_mut() {
            effect.family.candidate_selection =
                Some(antecedent_estimate::CandidateSelectionRecord {
                    screen_id: std::sync::Arc::clone(&selection.screen_id),
                    procedure: std::sync::Arc::from(selection.procedure.as_str()),
                    winner_index: selection.winner_index,
                    family_size: selection.family_size,
                    screen_rows: std::sync::Arc::clone(&selection.screen_rows),
                    estimate_rows: std::sync::Arc::clone(&selection.estimate_rows),
                    disjoint: selection.disjoint,
                });
        }
        if recorded {
            let mut d = antecedent_core::Diagnostic::new(
                "batch.candidate_selection",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                format!(
                    "screen={} procedure={} winner={:?} disjoint={}",
                    selection.screen_id,
                    selection.procedure.as_str(),
                    selection.winner_index,
                    selection.disjoint
                ),
            );
            d.fields = std::sync::Arc::from([
                (std::sync::Arc::from("screen_id"), std::sync::Arc::clone(&selection.screen_id)),
                (
                    std::sync::Arc::from("procedure"),
                    std::sync::Arc::from(selection.procedure.as_str()),
                ),
                (
                    std::sync::Arc::from("winner_index"),
                    std::sync::Arc::from(
                        selection
                            .winner_index
                            .map_or_else(|| "unavailable".into(), |i| i.to_string()),
                    ),
                ),
                (
                    std::sync::Arc::from("disjoint"),
                    std::sync::Arc::from(if selection.disjoint { "true" } else { "false" }),
                ),
            ]);
            result.view.diagnostics.push(d);
            if !selection.disjoint {
                result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.candidate_selection.overlap",
                    antecedent_core::DiagnosticKind::Support,
                    antecedent_core::DiagnosticSeverity::Warning,
                    "screen and estimate row sets are not disjoint",
                ));
            }
        } else {
            result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.unrecorded",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "no candidate-selection split has been recorded; simultaneous inference covers only the fixed supplied family and excludes screening uncertainty",
            ));
        }
        if recorded {
            result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.ranked_on_reported_family",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "winner is ranked on the reported estimate-sample family; screen_rows are recorded provenance and were not used to select",
            ));
        }
    }
}
