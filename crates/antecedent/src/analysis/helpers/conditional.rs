//! Conditional-effect result and quantile helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, ConditionalEffectQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, ExecutionContext, IdentifiedEstimand, Intervention, OutcomeFunctional,
    TableView, TabularData, VariableId, apply_outcome_functional, arm_n_eff,
    inference_from_influence_columns, tail_event_support,
};

/// Bound and monotonize threshold-major two-arm CDF values. Covariance
/// remains covariance of the raw fitted contrasts, not these projected CDFs.
pub(crate) fn project_conditional_cdf(cdf: &mut [f64]) -> Result<bool, CausalError> {
    if cdf.len() % 2 != 0 || cdf.iter().any(|x| !x.is_finite()) {
        return Err(CausalError::Compile { message: "invalid conditional CDF grid".into() });
    }
    let mut changed = false;
    for arm in 0..2 {
        let raw: Vec<_> = cdf.iter().skip(arm).step_by(2).copied().collect();
        let projected = antecedent_estimate::monotone_increasing(&raw);
        for (slot, value) in cdf.iter_mut().skip(arm).step_by(2).zip(projected) {
            let bounded = value.clamp(0.0, 1.0);
            changed |= slot.total_cmp(&bounded).is_ne();
            *slot = bounded;
        }
    }
    Ok(changed)
}

/// Choose one outcome grid from complete rows for every contributing conditional atom.
pub(crate) fn conditional_thresholds(
    data: &TabularData,
    query: &ConditionalEffectQuery,
    adjustment: impl Iterator<Item = VariableId>,
) -> Result<Option<Vec<f64>>, CausalError> {
    if query.inner.outcome_functional.quantile_level().is_none() {
        return Ok(query.inner.outcome_functional.thresholds());
    }
    if !matches!((&query.inner.control, &query.inner.active),
        (Intervention::Set { value: c, .. }, Intervention::Set { value: a, .. })
        if c.as_f64() == Some(0.0) && a.as_f64() == Some(1.0))
    {
        return Err(CausalError::Unsupported {
            message: "conditional quantiles require binary 0/1 arms; a linear mean model is not a distribution model",
        });
    }
    let mut ids: Vec<_> = adjustment.chain(query.inner.effect_modifiers.iter().copied()).collect();
    ids.extend([query.inner.treatment, query.inner.outcome]);
    let mask = data.complete_case_mask(&ids)?;
    let y = data.float64_masked(query.inner.outcome, &mask)?;
    Ok(Some(antecedent_estimate::empirical_threshold_grid(&y, 19)?))
}

/// Whether binary 0/1 `ConditionalEffect` exceedance/quantile uses cross-fitted AIPW scores.
#[must_use]
pub(crate) fn conditional_uses_crossfit_aipw(query: &ConditionalEffectQuery) -> bool {
    let functional = &query.inner.outcome_functional;
    if functional.thresholds().is_none() && functional.quantile_level().is_none() {
        return false;
    }
    matches!(
        (&query.inner.control, &query.inner.active),
        (Intervention::Set { value: c, .. }, Intervention::Set { value: a, .. })
            if c.as_f64() == Some(0.0) && a.as_f64() == Some(1.0)
    )
}

/// Name the estimator that actually produced a `ConditionalEffect` exceedance or
/// quantile on binary 0/1 arms.
///
/// Those cells record `aipw` (not `conditional.linear.adjustment`): the scalar,
/// arm CDFs, and influence columns come from cross-fitted AIPW scores with the
/// modifier added to the adjustment set. Non-binary levels and mean
/// `ConditionalEffect` keep the linear plugin.
pub(crate) fn conditional_score_estimator_diagnostic(
    query: &ConditionalEffectQuery,
) -> Option<Diagnostic> {
    conditional_uses_crossfit_aipw(query).then(|| {
        Diagnostic::new(
            "estimate.conditional.crossfit_aipw_scores",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "binary-arm conditional exceedance/quantile: the scalar, arm CDFs, and influence columns are cross-fitted AIPW scores (logistic propensity and per-arm outcome regressions on the modifier plus the adjustment set, standardized over the observed modifier distribution), not the conditional.linear.adjustment T×W interaction model named in the plan; the analytic SE is the AIPW influence-function SE",
        )
    })
}

/// Publish the actual conditional grid, whose coordinates have no score-table payload.
pub(crate) fn conditional_quantile_grid_diagnostic(
    data: &TabularData,
    query: &ConditionalEffectQuery,
    adjustment: impl Iterator<Item = VariableId>,
) -> Result<Option<Diagnostic>, CausalError> {
    if query.inner.outcome_functional.quantile_level().is_none() {
        return Ok(None);
    }
    let thresholds = conditional_thresholds(data, query, adjustment)?.unwrap_or_default();
    Ok(Some(Diagnostic::new(
        "estimate.functional.quantile_grid",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "frozen empirical thresholds: {thresholds:?}; CDF coordinates are threshold-major, control then active"
        ),
    )))
}

/// Invert the two raw CDF arms, after any frozen-weight atom mixture.
pub(crate) fn attach_conditional_quantile(
    out: &mut EffectEstimate,
    thresholds: &[f64],
    raw_cdf: &[f64],
    columns: &[Vec<f64>],
    supported: &[bool],
    tau: f64,
) -> Result<(), CausalError> {
    let mut arms = Vec::new();
    for arm in 0..2 {
        let means: Vec<_> = raw_cdf.iter().skip(arm).step_by(2).copied().collect();
        let support: Vec<_> = supported.iter().skip(arm).step_by(2).copied().collect();
        let influences: Vec<Vec<f64>> = columns
            .iter()
            .skip(arm)
            .step_by(2)
            .map(|c| {
                let mean = c.iter().sum::<f64>() / c.len() as f64;
                c.iter().map(|v| v - mean).collect()
            })
            .collect();
        arms.push(antecedent_estimate::quantile::invert_supported_cdf(
            thresholds,
            &means,
            &influences,
            &support,
            tau,
        )?);
    }
    let influence: Vec<_> =
        arms[1].influence.iter().zip(&arms[0].influence).map(|(a, b)| a - b).collect();
    out.ate = arms[1].value - arms[0].value;
    out.uncertainty.analytic.se =
        antecedent_estimate::joint_influence_covariance(&[&influence], None)?.se(0);
    out.scores.influence = Some(influence.into());
    out.uncertainty.bootstrap.se = None;
    out.family.simultaneous_interval = None;
    Ok(())
}

/// Evaluate each exceedance threshold on `ConditionalEffect` and attach `F_a(c)`.
///
/// Joint covariance and simultaneous bands are the 2K raw per-arm CDF
/// coordinates. Sparse tails refuse a finite band rather than publishing an
/// empty-cell or first-threshold SE.
pub(crate) fn attach_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &ConditionalEffectQuery,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let Some(thresholds) =
        conditional_thresholds(data, query, estimand.adjustment_set.iter().copied())?
    else {
        return Ok(estimate);
    };
    let y_orig = data.float64_values(query.inner.outcome).map_err(CausalError::from)?;
    let mut est = antecedent_estimate::ConditionalLinearAdjustment::new()
        .with_fold_seed(ctx.rng.master_seed());
    if query.inner.effect_modifiers.len() > 1 {
        est = est.with_joint_modifiers();
    }
    let mut raw_cdf = Vec::with_capacity(thresholds.len() * 2);
    let mut columns = Vec::with_capacity(thresholds.len() * 2);
    let mut event_n_eff = Vec::with_capacity(thresholds.len() * 2);
    let mut threshold_supported = Vec::with_capacity(thresholds.len() * 2);
    let mut first = None;
    let mut n_eff_by_arm = [0.0, 0.0];
    for &threshold in &thresholds {
        let data_c = apply_outcome_functional(
            data,
            query.inner.outcome,
            &OutcomeFunctional::exceedance(threshold),
        )?;
        let mut transformed_query = query.clone();
        transformed_query.inner.outcome_functional = OutcomeFunctional::Mean;
        let (point, scores) =
            est.estimate_with_arm_scores(&data_c, estimand, &transformed_query)?;
        raw_cdf.extend([1.0 - scores.means[0], 1.0 - scores.means[1]]);
        if scores.influence[0].len() != scores.influence[1].len()
            || scores.influence[0].len() != scores.row_index.len()
        {
            return Err(CausalError::Unsupported {
                message: "ConditionalEffect grid refused: missing per-arm influence for joint covariance",
            });
        }
        for arm in 0..2 {
            columns.push(scores.influence[arm].iter().map(|v| -v).collect::<Vec<_>>());
            let (events, supported) =
                tail_event_support(&scores.treatment, &scores.row_index, &y_orig, arm, threshold);
            event_n_eff.push(events);
            threshold_supported.push(supported);
        }
        n_eff_by_arm = arm_n_eff(&scores.treatment);
        if first.is_none() {
            first = Some(point);
        }
    }
    let mut out = first.unwrap_or(estimate);
    if columns.len() != thresholds.len() * 2 {
        return Err(CausalError::Unsupported {
            message: "ConditionalEffect grid refused: missing influence columns for joint covariance",
        });
    }
    let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    out.joint.covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    let n_eff = n_eff_by_arm[0] + n_eff_by_arm[1];
    out.scores.inference = Some(inference_from_influence_columns(
        &raw_cdf,
        &refs,
        &event_n_eff,
        &threshold_supported,
        antecedent_estimate::WeightedSupport {
            n_eff,
            n_eff_by_arm: n_eff_by_arm.to_vec(),
            propensity_range: None,
            overlap_ok: n_eff_by_arm
                .iter()
                .all(|n| *n >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS),
        },
    )?);
    if thresholds.len() == 1 && threshold_supported.iter().any(|&supported| !supported) {
        out.uncertainty.analytic.se = f64::NAN;
        out.uncertainty.bootstrap.se = None;
        out.scores.influence = None;
        out.family.simultaneous_interval = None;
    }
    let mut cdf = raw_cdf.clone();
    let rearranged = project_conditional_cdf(&mut cdf)?;
    out = out.with_monotone_rearranged(rearranged);
    out.joint.exceedance_cdf = Some(Arc::from(cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.uncertainty.analytic.se = f64::NAN;
        out.uncertainty.bootstrap.se = None;
        out.scores.influence = None;
        out.family.simultaneous_interval = None;
    }
    if let Some(tau) = query.inner.outcome_functional.quantile_level() {
        attach_conditional_quantile(
            &mut out,
            &thresholds,
            &raw_cdf,
            &columns,
            &threshold_supported,
            tau,
        )?;
    }
    Ok(out)
}
