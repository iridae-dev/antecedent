//! Validate and replay sufficient statistics for linear fits.

use crate::AnalysisResultWire;
use crate::contract_reason::ContractReason;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};

use super::AnalysisResultContractWire;

pub(in crate::contract_section) fn valid_linear_fit_kind(fit_kind: &str) -> bool {
    match fit_kind {
        "ols" => true,
        _ => fit_kind.split_once(':').is_some_and(|(kind, bits)| {
            let Ok(bits) = u64::from_str_radix(bits, 16) else { return false };
            let value = f64::from_bits(bits);
            value.is_finite() && value > 0.0 && matches!(kind, "ridge" | "lasso" | "huber")
        }),
    }
}

pub(in crate::contract_section) fn verify_linear_fit_moments(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> Result<(), ContractReason> {
    if lowering.fit_kind != "ols"
        || lowering.se_kind != "homoskedastic"
        || lowering.bootstrap_replicates != 0
        || lowering.interval_method != "analytic_se"
        || !matches!(&lowering.population, crate::TargetPopulationWire::AllObserved)
    {
        return Err(ContractReason::DependenciesLinearFitSufficientStatistics);
    }
    let snapshot = contract
        .data_snapshot
        .as_ref()
        .ok_or(ContractReason::DependenciesLinearFitSufficientStatistics)?;
    let moments = snapshot
        .linear_fit_moments
        .as_ref()
        .ok_or(ContractReason::DependenciesLinearFitSufficientStatistics)?;
    let p = lowering.design_columns.len();
    if moments.format != 1
        || usize::try_from(moments.columns).ok() != Some(p)
        || moments.complete_case_rows != lowering.complete_case_rows
        || moments.complete_case_rows > snapshot.row_count
        || p < 2
        || moments.complete_case_rows <= p as u64
        || moments.gram.len() != p * p
        || moments.outcome_cross.len() != p
        || !moments.outcome_square.is_finite()
        || moments.gram.iter().chain(&moments.outcome_cross).any(|v| !v.is_finite())
    {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let scale = moments.gram.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let symmetry = ReplayTolerance::new(
        LINEAR_GRAM_SYMMETRY_RELATIVE * scale.max(1.0),
        ReplayScale::Absolute,
        ReplayNonFinite::Refuse,
    );
    if scale == 0.0
        || !scale.is_finite()
        || (0..p).any(|row| {
            (0..p).any(|column| {
                !symmetry.close(moments.gram[row * p + column], moments.gram[column * p + row])
            })
        })
    {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let inverse = invert_linear_moments(&moments.gram, p)
        .ok_or(ContractReason::DataSnapshotLinearFitMomentsRank)?;
    let beta = apply_moment_inverse(&inverse, &moments.outcome_cross, p)
        .ok_or(ContractReason::DataSnapshotLinearFitMomentsRank)?;
    let residual_square = moments.outcome_square
        - beta.iter().zip(&moments.outcome_cross).map(|(b, xy)| b * xy).sum::<f64>();
    let tolerance = 1e-9 * moments.outcome_square.abs().max(1.0);
    if !residual_square.is_finite() || residual_square < -tolerance {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let dof = (moments.complete_case_rows - p as u64) as f64;
    let treatment_column = lowering
        .design_columns
        .iter()
        .position(|column| column == "treatment")
        .ok_or(ContractReason::ProgramCheckedLinearAdjustmentBinding)?;
    if lowering.design_columns.iter().filter(|column| *column == "treatment").count() != 1
        || lowering.design_columns.first().map(String::as_str) != Some("intercept")
    {
        return Err(ContractReason::ProgramCheckedLinearAdjustmentBinding);
    }
    // The residual sum of squares is a difference of two sums of products of the data, so
    // an exact fit leaves only roundoff of the order of eps * outcome_square. That is a zero
    // residual, not a standard error of its square root.
    let residual_square = if residual_square.abs() <= 1e-12 * moments.outcome_square.abs() {
        0.0
    } else {
        residual_square
    };
    let variance =
        (residual_square.max(0.0) / dof) * inverse[treatment_column * p + treatment_column];
    if !variance.is_finite() || variance < 0.0 {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let delta = f64::from_bits(lowering.active_bits) - f64::from_bits(lowering.control_bits);
    let expected = beta[treatment_column] * delta;
    let expected_se = delta.abs() * variance.sqrt();
    let actual = body.estimate.ok_or(ContractReason::BodyEstimate)?;
    if !close_replay_value(actual, expected) {
        return Err(ContractReason::BodyEstimate);
    }
    let actual_se = body.standard_error.ok_or(ContractReason::BodyStandardError)?;
    if !close_replay_value(actual_se, expected_se) {
        return Err(ContractReason::BodyStandardError);
    }
    Ok(())
}

/// Inverse of a row-major `p x p` moment matrix, or `None` when it is singular, non-finite or
/// overflows. The replay deliberately stays off the estimator's fit path and shares only the
/// scale-free equilibrated Gauss-Jordan inverse of `antecedent-stats`, so a unit change in a
/// design column moves neither the refusal nor the verdict.
pub(in crate::contract_section) fn invert_linear_moments(
    gram: &[f64],
    p: usize,
) -> Option<Vec<f64>> {
    let inverse = antecedent_stats::invert_square(gram, p)?;
    inverse.iter().all(|v| v.is_finite()).then_some(inverse)
}

/// `inverse * cross` for a row-major `p x p` inverse; `None` when the product is not finite.
pub(in crate::contract_section) fn apply_moment_inverse(
    inverse: &[f64],
    cross: &[f64],
    p: usize,
) -> Option<Vec<f64>> {
    let beta: Vec<f64> = (0..p)
        .map(|row| (0..p).map(|column| inverse[row * p + column] * cross[column]).sum())
        .collect();
    beta.iter().all(|v| v.is_finite()).then_some(beta)
}

/// The contract replay's policy for a recomputed estimate or standard error: a relative
/// bound of `1e-8` against the recomputed value (absolute below one), non-finite refused.
/// The replay rebuilds the number from published moments by a different route than the
/// estimator took, so the two agree to the conditioning of the solve, not to an ulp.
const CONTRACT_REPLAY_VALUE: ReplayTolerance =
    ReplayTolerance::new(1e-8, ReplayScale::ReplayedOrOne, ReplayNonFinite::Refuse);

/// Replay policy of a stored three-by-three Gram entry: the intercept entry against the complete
/// case count, and each off-diagonal pair against its mirror, absolute to `1e-9`; non-finite
/// is refused.
pub(super) const GRAM_ENTRY_REPLAY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::Absolute, ReplayNonFinite::Refuse);

/// Symmetry bound of a stored linear-fit Gram matrix, relative to its largest entry (at least
/// one): the matrix is a sum of outer products and is symmetric up to the rounding of the
/// accumulation.
const LINEAR_GRAM_SYMMETRY_RELATIVE: f64 = 1e-10;

pub(in crate::contract_section) fn close_replay_value(actual: f64, expected: f64) -> bool {
    CONTRACT_REPLAY_VALUE.close(actual, expected)
}
