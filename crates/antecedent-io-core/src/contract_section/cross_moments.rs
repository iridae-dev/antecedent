//! Point-estimate replay of checked instrumental-variable and linear front-door fits from
//! published cross-product moments.
//!
//! The contract carries `V'V` over the complete-case columns the fit read
//! ([`crate::CrossProductMomentsWire`]). The consumer re-solves the fit from those moments by
//! the normal equations, not by the estimator's code path, and compares the point estimate.
//! Uncertainty is not replayed here: IV withholds its interval (Anderson-Rubin on first-stage
//! diagnostics) and the front-door sandwich needs row-level scores, so both keep their existing
//! treatment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AnalysisResultContractWire;
use super::replay::{apply_moment_inverse, close_replay_value, invert_linear_moments};
use crate::contract_reason::ContractReason;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use crate::{CROSS_PRODUCT_INTERCEPT, CrossProductMomentsWire};

/// Symmetry bound of the stored Gram matrix relative to its largest entry (at least one).
const SYMMETRY_RELATIVE: f64 = 1e-10;
/// The intercept entry of the Gram matrix is the complete-case row count.
const ROW_COUNT_ENTRY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::ReplayedOrOne, ReplayNonFinite::Refuse);

/// The published snapshot's row count and moments.
pub(in crate::contract_section) struct Snapshot<'a> {
    pub(in crate::contract_section) row_count: u64,
    pub(in crate::contract_section) moments: Option<&'a CrossProductMomentsWire>,
}

impl<'a> Snapshot<'a> {
    pub(in crate::contract_section) fn of(
        contract: &'a AnalysisResultContractWire,
    ) -> Option<Self> {
        let snapshot = contract.data_snapshot.as_ref()?;
        Some(Self {
            row_count: snapshot.row_count,
            moments: snapshot.cross_product_moments.as_ref(),
        })
    }
}

/// The stored moments, once their shape, identity, finiteness and symmetry agree with the
/// columns and row count the lowering declares.
fn validated<'a>(
    snapshot: &Snapshot<'a>,
    expected_columns: &[u32],
    rows: u64,
) -> Result<&'a CrossProductMomentsWire, ContractReason> {
    let moments = snapshot.moments.ok_or(ContractReason::DataSnapshotCrossProductMoments)?;
    let k = expected_columns.len();
    if moments.format != 1
        || moments.columns != expected_columns
        || moments.complete_case_rows != rows
        || rows == 0
        || rows > snapshot.row_count
        || rows <= k as u64
        || moments.gram.len() != k * k
        || moments.gram.iter().any(|value| !value.is_finite())
    {
        return Err(ContractReason::DataSnapshotCrossProductMoments);
    }
    let scale = moments.gram.iter().map(|value| value.abs()).fold(0.0_f64, f64::max).max(1.0);
    let symmetry = ReplayTolerance::new(
        SYMMETRY_RELATIVE * scale,
        ReplayScale::Absolute,
        ReplayNonFinite::Refuse,
    );
    let symmetric = (0..k)
        .all(|a| (0..k).all(|b| symmetry.close(moments.gram[a * k + b], moments.gram[b * k + a])));
    if !symmetric || !ROW_COUNT_ENTRY.close(moments.gram[0], rows as f64) {
        return Err(ContractReason::DataSnapshotCrossProductMoments);
    }
    Ok(moments)
}

/// Row-major block `gram[rows][cols]` of a `k x k` Gram matrix.
fn block(gram: &[f64], k: usize, rows: &[usize], cols: &[usize]) -> Vec<f64> {
    rows.iter().flat_map(|&r| cols.iter().map(move |&c| gram[r * k + c])).collect()
}

/// `left (l x m) * right (m x n)`, row-major.
fn multiply(left: &[f64], right: &[f64], l: usize, m: usize, n: usize) -> Vec<f64> {
    (0..l)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| (0..m).map(|t| left[i * m + t] * right[t * n + j]).sum())
        .collect()
}

/// OLS coefficients of column `y` on `design` from the moments; `None` when singular.
fn ols(gram: &[f64], k: usize, design: &[usize], y: usize) -> Option<Vec<f64>> {
    let p = design.len();
    let inverse = invert_linear_moments(&block(gram, k, design, design), p)?;
    apply_moment_inverse(&inverse, &block(gram, k, design, &[y]), p)
}

fn delta(active_bits: u64, control_bits: u64) -> f64 {
    f64::from_bits(active_bits) - f64::from_bits(control_bits)
}

/// Replay the point estimate of the checked IV lowering.
///
/// `[1, adjustment..., instrument, treatment, outcome]`: the just-identified two-stage
/// least squares estimate `(Xe'P Xe)^-1 Xe'P y` with `Xe = [1, adjustment, treatment]`,
/// `W = [1, adjustment, instrument]` and `P = W (W'W)^-1 W'`. With one binary instrument and
/// no adjustment this is the Wald ratio.
pub(in crate::contract_section) fn verify_iv_moments(
    snapshot: &Snapshot<'_>,
    estimate: Option<f64>,
    lowering: &crate::CheckedIvLoweringWire,
) -> Result<(), ContractReason> {
    let mut columns = vec![CROSS_PRODUCT_INTERCEPT];
    columns.extend(&lowering.adjustment);
    columns.extend([lowering.instrument, lowering.treatment, lowering.outcome]);
    let moments = validated(snapshot, &columns, lowering.complete_case_rows)?;
    let k = columns.len();
    let exogenous = lowering.adjustment.len() + 1;
    let (instrument, treatment, outcome) = (exogenous, exogenous + 1, exogenous + 2);
    let exogenous_columns: Vec<usize> = (0..exogenous).collect();
    let excluded: Vec<usize> = exogenous_columns.iter().copied().chain([instrument]).collect();
    let structural: Vec<usize> = exogenous_columns.iter().copied().chain([treatment]).collect();
    let (q, p) = (excluded.len(), structural.len());
    let rank = ContractReason::DataSnapshotCrossProductMomentsRank;
    let w_inverse =
        invert_linear_moments(&block(&moments.gram, k, &excluded, &excluded), q).ok_or(rank)?;
    let a = block(&moments.gram, k, &structural, &excluded);
    let c = block(&moments.gram, k, &excluded, &[outcome]);
    let a_winv = multiply(&a, &w_inverse, p, q, q);
    let normal = multiply(&a_winv, &transpose(&a, p, q), p, q, p);
    let rhs = multiply(&a_winv, &c, p, q, 1);
    let normal_inverse = invert_linear_moments(&normal, p).ok_or(rank)?;
    let theta = apply_moment_inverse(&normal_inverse, &rhs, p).ok_or(rank)?;
    let expected = theta[p - 1] * delta(lowering.active_bits, lowering.control_bits);
    let actual = estimate.ok_or(ContractReason::BodyEstimate)?;
    if close_replay_value(actual, expected) { Ok(()) } else { Err(ContractReason::BodyEstimate) }
}

fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..cols).flat_map(|j| (0..rows).map(move |i| matrix[i * cols + j])).collect()
}

/// Replay the point estimate of the checked linear front-door lowering.
///
/// `[1, treatment, mediators..., outcome]`: stage one regresses each mediator on `[1, T]`,
/// stage two regresses the outcome on `[1, T, mediators...]`, and the estimate is
/// `sum_j beta_{T->Mj} * beta_{Mj->Y}` times the contrast.
pub(in crate::contract_section) fn verify_frontdoor_moments(
    snapshot: &Snapshot<'_>,
    estimate: Option<f64>,
    lowering: &crate::CheckedFrontDoorLoweringWire,
) -> Result<(), ContractReason> {
    let mut columns = vec![CROSS_PRODUCT_INTERCEPT, lowering.treatment];
    columns.extend(&lowering.mediators);
    columns.push(lowering.outcome);
    let moments = validated(snapshot, &columns, lowering.complete_case_rows)?;
    let k = columns.len();
    let mediators = lowering.mediators.len();
    let outcome = k - 1;
    let rank = ContractReason::DataSnapshotCrossProductMomentsRank;
    let stage2_design: Vec<usize> = (0..2 + mediators).collect();
    let stage2 = ols(&moments.gram, k, &stage2_design, outcome).ok_or(rank)?;
    let mut path_sum = 0.0;
    for mediator in 0..mediators {
        let stage1 = ols(&moments.gram, k, &[0, 1], 2 + mediator).ok_or(rank)?;
        path_sum += stage1[1] * stage2[2 + mediator];
    }
    let expected = path_sum * delta(lowering.active_bits, lowering.control_bits);
    let actual = estimate.ok_or(ContractReason::BodyEstimate)?;
    if close_replay_value(actual, expected) { Ok(()) } else { Err(ContractReason::BodyEstimate) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CheckedFrontDoorLoweringWire, CheckedIvLoweringWire};

    const N: usize = 40;

    fn noise(i: usize, salt: f64) -> f64 {
        ((i as f64 + 1.0) * salt).sin()
    }

    fn moments(columns: &[(u32, Vec<f64>)]) -> CrossProductMomentsWire {
        let k = columns.len();
        let mut gram = vec![0.0; k * k];
        for a in 0..k {
            for b in 0..k {
                gram[a * k + b] = columns[a].1.iter().zip(&columns[b].1).map(|(x, y)| x * y).sum();
            }
        }
        CrossProductMomentsWire {
            format: 1,
            complete_case_rows: N as u64,
            columns: columns.iter().map(|(id, _)| *id).collect(),
            gram,
        }
    }

    fn snapshot(moments: Option<&CrossProductMomentsWire>) -> Snapshot<'_> {
        Snapshot { row_count: N as u64, moments }
    }

    fn ones() -> (u32, Vec<f64>) {
        (CROSS_PRODUCT_INTERCEPT, vec![1.0; N])
    }

    /// Instrument 2 -> treatment 0 -> outcome 1 with a noise-free structural coefficient of 2.
    fn iv_columns() -> Vec<(u32, Vec<f64>)> {
        let z: Vec<f64> = (0..N).map(|i| (i % 2) as f64).collect();
        let t: Vec<f64> = (0..N).map(|i| 0.5 * z[i] + noise(i, 1.7)).collect();
        let y: Vec<f64> = t.iter().map(|t| 1.0 + 2.0 * t).collect();
        vec![ones(), (2, z), (0, t), (1, y)]
    }

    fn iv_lowering() -> CheckedIvLoweringWire {
        CheckedIvLoweringWire {
            format: 1,
            functional: 0,
            treatment: 0,
            outcome: 1,
            instrument: 2,
            adjustment: Vec::new(),
            active_bits: 1.0_f64.to_bits(),
            control_bits: 0.0_f64.to_bits(),
            instrument_active_bits: 1.0_f64.to_bits(),
            instrument_control_bits: 0.0_f64.to_bits(),
            procedure: "two_sls".into(),
            se_kind: "homoskedastic".into(),
            weak_instrument_uncertainty: "withheld".into(),
            complete_case_rows: N as u64,
        }
    }

    #[test]
    fn iv_estimate_is_replayed_from_the_moments() {
        let wire = moments(&iv_columns());
        let snap = snapshot(Some(&wire));
        assert_eq!(verify_iv_moments(&snap, Some(2.0), &iv_lowering()), Ok(()));
        assert_eq!(
            verify_iv_moments(&snap, Some(2.1), &iv_lowering()),
            Err(ContractReason::BodyEstimate)
        );
        assert_eq!(
            verify_iv_moments(&snapshot(None), Some(2.0), &iv_lowering()),
            Err(ContractReason::DataSnapshotCrossProductMoments)
        );
    }

    #[test]
    fn tampered_or_degenerate_iv_moments_are_refused() {
        let mut wire = moments(&iv_columns());
        wire.gram[4 + 2] += 1.0; // breaks symmetry
        assert_eq!(
            verify_iv_moments(&snapshot(Some(&wire)), Some(2.0), &iv_lowering()),
            Err(ContractReason::DataSnapshotCrossProductMoments)
        );
        let mut wrong_columns = moments(&iv_columns());
        wrong_columns.columns.swap(2, 3);
        assert_eq!(
            verify_iv_moments(&snapshot(Some(&wrong_columns)), Some(2.0), &iv_lowering()),
            Err(ContractReason::DataSnapshotCrossProductMoments)
        );
        let mut columns = iv_columns();
        columns[1].1 = vec![1.0; N]; // a constant instrument has no first stage
        assert_eq!(
            verify_iv_moments(&snapshot(Some(&moments(&columns))), Some(2.0), &iv_lowering()),
            Err(ContractReason::DataSnapshotCrossProductMomentsRank)
        );
    }

    fn frontdoor_columns() -> (Vec<(u32, Vec<f64>)>, f64) {
        let t: Vec<f64> = (0..N).map(|i| noise(i, 0.9)).collect();
        let m: Vec<f64> = (0..N).map(|i| 0.7 * t[i] + noise(i, 2.3)).collect();
        let y: Vec<f64> = (0..N).map(|i| 3.0 * m[i] + 0.2 * t[i]).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / N as f64;
        let (mt, mm) = (mean(&t), mean(&m));
        let cov: f64 = t.iter().zip(&m).map(|(a, b)| (a - mt) * (b - mm)).sum();
        let var: f64 = t.iter().map(|a| (a - mt) * (a - mt)).sum();
        (vec![ones(), (0, t), (3, m), (1, y)], cov / var * 3.0)
    }

    fn frontdoor_lowering() -> CheckedFrontDoorLoweringWire {
        CheckedFrontDoorLoweringWire {
            format: 1,
            functional: 0,
            executable: 0,
            treatment: 0,
            outcome: 1,
            mediators: vec![3],
            active_bits: 1.0_f64.to_bits(),
            control_bits: 0.0_f64.to_bits(),
            procedure: "linear_two_stage".into(),
            complete_case_rows: N as u64,
            overlap: String::new(),
            uncertainty: String::new(),
            arena: crate::ExprArenaWire {
                derivations: Vec::new(),
                var_sets: Vec::new(),
                interventions: Vec::new(),
                lists: Vec::new(),
                nodes: Vec::new(),
            },
        }
    }

    #[test]
    fn frontdoor_path_sum_is_replayed_from_the_moments() {
        let (columns, expected) = frontdoor_columns();
        let wire = moments(&columns);
        let snap = snapshot(Some(&wire));
        assert_eq!(verify_frontdoor_moments(&snap, Some(expected), &frontdoor_lowering()), Ok(()));
        assert_eq!(
            verify_frontdoor_moments(&snap, Some(expected + 0.5), &frontdoor_lowering()),
            Err(ContractReason::BodyEstimate)
        );
        let mut short = wire.clone();
        short.complete_case_rows -= 1;
        assert_eq!(
            verify_frontdoor_moments(
                &snapshot(Some(&short)),
                Some(expected),
                &frontdoor_lowering()
            ),
            Err(ContractReason::DataSnapshotCrossProductMoments)
        );
    }
}
