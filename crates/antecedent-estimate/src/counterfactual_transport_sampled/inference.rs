//! Delta-method standard error of the plug-in functional.
//!
//! The estimator is `f(p)` with `p` the stacked cell probabilities of the
//! distinct datasets and `f` the stored derivation evaluated on the finite laws
//! (the whole identification and transport functional, including its ratios and
//! the averaging of unread coordinates). Each dataset's cell counts are
//! multinomial; datasets are independent; the covariance of a dataset with
//! fixed strata is block diagonal. With `g_c = d f / d w_c` taken at the
//! normalized table (`f` evaluated at `w / sum(w)`, so the gradient is exactly
//! the gradient of the homogeneous extension), the variance of a dataset of
//! `N` units is
//!
//! `(1 / N) * sum_s [ sum_{c in s} p_c g_c^2 - (sum_{c in s} p_c g_c)^2 / w_s ]`
//!
//! over its strata `s` (one stratum, `w_s = 1`, for independent units), and the
//! variance of the estimator is the sum over datasets. The gradient is a central
//! difference with a stated relative step in each positive cell; cells with
//! probability zero contribute nothing to the plug-in variance and are not
//! differenced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_identify::CounterfactualTransportDerivation;

use super::counts::{LawFlavor, SampledRegimeCounts};
use super::overlap::DatasetGroups;
use super::{
    SAMPLED_TRANSPORT_GRADIENT_RELATIVE_STEP, SampledTransportLimits, SampledWork,
    evaluate_detailed, table_work,
};
use crate::error::EstimationError;

/// Why the standard error is not reported.
pub(crate) const SE_WITHHELD_PERTURBED_UNDEFINED: &str = "perturbed_evaluation_undefined";
/// The variance is not a finite non-negative number.
pub(crate) const SE_WITHHELD_NONFINITE: &str = "variance_not_finite";

/// What the delta method produced.
pub(crate) enum DeltaOutcome {
    /// A standard error and the evaluations spent on gradients.
    Computed { standard_error: f64, evaluations: u64 },
    /// No standard error, with a stable reason.
    Withheld { reason: &'static str },
}

/// Delta-method standard error at the estimated tables `probabilities`.
///
/// # Errors
///
/// Budget exhaustion or cancellation of an evaluation (a perturbed evaluation
/// that is merely undefined withholds the standard error instead).
pub(crate) fn delta_standard_error(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
    probabilities: &[Vec<f64>],
    groups: &DatasetGroups,
    limits: &SampledTransportLimits,
    ctx: &ExecutionContext,
    work: &mut SampledWork<'_>,
) -> Result<DeltaOutcome, EstimationError> {
    // A perturbed table is no longer the count shares, so its law is model-based.
    let mut flavors = Vec::with_capacity(tables.len());
    while flavors.len() < tables.len() {
        work.charge(0)?;
        flavors.resize(flavors.len().saturating_add(512).min(tables.len()), LawFlavor::Model);
    }
    let mut variance = 0.0;
    let mut evaluations = 0u64;
    // Reuse the table stack across every perturbation; only the current
    // physical dataset group needs to be changed and restored.
    for table in tables {
        let units = table_work(table, work)?;
        work.charge(units)?;
    }
    let mut trial = Vec::with_capacity(probabilities.len());
    for values in probabilities {
        let mut copied = Vec::with_capacity(values.len());
        for chunk in values.chunks(512) {
            work.charge(0)?;
            copied.extend_from_slice(chunk);
        }
        trial.push(copied);
    }
    for group in &groups.groups {
        let representative = group[0];
        let table = &tables[representative];
        let p = &probabilities[representative];
        let (total, _, _) = table.checked_report_counts(work)?;
        let total = total as f64;
        let units = table_work(table, work)?.saturating_mul(table.axes().len().saturating_add(1));
        work.charge(units)?;
        let (stratum_of, strata) = table.strata(work)?;
        let mut weight = vec![0.0f64; strata];
        let mut first = vec![0.0f64; strata];
        let mut second = vec![0.0f64; strata];
        for (cell, &pc) in p.iter().enumerate() {
            if cell % 512 == 0 {
                work.charge(0)?;
            }
            weight[stratum_of[cell]] += pc;
            if pc.is_nan() || pc <= 0.0 {
                continue;
            }
            let h = SAMPLED_TRANSPORT_GRADIENT_RELATIVE_STEP * pc;
            let mut value = [0.0f64; 2];
            for (slot, sign) in [1.0f64, -1.0].iter().enumerate() {
                work.charge(p.len().saturating_mul(group.len().saturating_add(4)))?;
                let mut perturbed = Vec::with_capacity(p.len());
                for chunk in p.chunks(512) {
                    work.charge(0)?;
                    perturbed.extend_from_slice(chunk);
                }
                perturbed[cell] += sign * h;
                let mut norm = 0.0;
                for chunk in perturbed.chunks(512) {
                    work.charge(0)?;
                    for q in chunk {
                        norm += q;
                    }
                }
                for chunk in perturbed.chunks_mut(512) {
                    work.charge(0)?;
                    for q in chunk {
                        *q /= norm;
                    }
                }
                for &member in group {
                    trial[member].clear();
                    for chunk in perturbed.chunks(512) {
                        work.charge(0)?;
                        trial[member].extend_from_slice(chunk);
                    }
                }
                evaluations += 1;
                match evaluate_detailed(derivation, tables, &trial, &flavors, limits, ctx, work) {
                    Ok((v, _)) => value[slot] = v.probability,
                    Err(EstimationError::Refused { code, .. })
                        if code != antecedent_core::reason_code!("transport_budget_cancel") =>
                    {
                        return Ok(DeltaOutcome::Withheld {
                            reason: SE_WITHHELD_PERTURBED_UNDEFINED,
                        });
                    }
                    Err(other) => return Err(other),
                }
            }
            for &member in group {
                trial[member].clear();
                for chunk in probabilities[member].chunks(512) {
                    work.charge(0)?;
                    trial[member].extend_from_slice(chunk);
                }
            }
            let g = (value[0] - value[1]) / (2.0 * h);
            first[stratum_of[cell]] += pc * g;
            second[stratum_of[cell]] += pc * g * g;
        }
        let mut dataset = 0.0;
        for s in 0..strata {
            if s % 512 == 0 {
                work.charge(0)?;
            }
            if weight[s] > 0.0 {
                dataset += second[s] - first[s] * first[s] / weight[s];
            }
        }
        variance += dataset / total;
    }
    if !variance.is_finite() || variance < -1e-12 {
        return Ok(DeltaOutcome::Withheld { reason: SE_WITHHELD_NONFINITE });
    }
    Ok(DeltaOutcome::Computed { standard_error: variance.max(0.0).sqrt(), evaluations })
}
