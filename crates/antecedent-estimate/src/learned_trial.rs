//! Cross-fitted trial-to-nonparticipant transport with explicit sampling design.
#![allow(clippy::cast_possible_truncation)]
use crate::{EstimationError, trial_to_target_effect};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_identify::{TransportFormula, TransportIdentification};
use antecedent_learn::{
    DesignView, LearnerSpec, PredictionTask, TargetView, cross_fit_selected, resolve_for,
};
use serde::{Deserialize, Serialize};

/// Sampling units used by the joint outer bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialSampling {
    /// One IID cohort containing trial participants and nonparticipants.
    NestedCohort,
    /// Independent IID trial and representative target samples, with fixed sample sizes.
    IndependentSamples,
}

/// Explicit complete baseline design and known randomization probabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialAipwInput {
    /// Covariate coordinates, exactly the certified standardizers.
    pub features: Vec<u32>,
    /// Raw baseline covariates in column-major order, without an intercept.
    pub covariates: Vec<Vec<f64>>,
    /// Source outcomes; target entries are ignored.
    pub outcome: Vec<f64>,
    /// Randomized source treatment; target entries are ignored.
    pub treatment: Vec<bool>,
    /// Source membership.
    pub source: Vec<bool>,
    /// Known probability of treatment one in the source trial.
    pub randomization: Vec<f64>,
    /// Declared sampling design. Unknown and clustered designs are not accepted.
    pub sampling: TrialSampling,
}

/// Learner and inference settings for one certified target contrast.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialAipwOptions {
    /// Source outcome nuisance.
    pub outcome: LearnerSpec,
    /// Source membership nuisance in the observed sampling design.
    pub membership: LearnerSpec,
    /// Shared cross-fit fold count.
    pub folds: usize,
    /// Joint outer refits. Zero retains the point without an interval.
    pub bootstrap: u32,
    /// Nominal pointwise coverage; no calibration claim.
    pub coverage_level: f64,
}
impl Default for TrialAipwOptions {
    fn default() -> Self {
        Self {
            outcome: LearnerSpec::Ridge(crate::RidgeSpec::default()),
            membership: LearnerSpec::Logistic(crate::LogisticSpec::default()),
            folds: 5,
            bootstrap: 199,
            coverage_level: 0.95,
        }
    }
}

/// Held-out losses for the three fitted nuisance roles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialNuisanceDiagnostics {
    /// Binary source membership log loss across both samples.
    pub membership_logloss: f64,
    /// Source outcome RMSE, in treatment-zero/treatment-one order.
    pub outcome_rmse: [f64; 2],
}

/// Recompute held-out role diagnostics from retained score inputs.
/// # Errors
/// Incompatible shape, nonfinite prediction, or empty observed role.
pub fn trial_nuisance_diagnostics(
    input: &TrialAipwInput,
    membership: &[f64],
    mu0: &[f64],
    mu1: &[f64],
) -> Result<TrialNuisanceDiagnostics, EstimationError> {
    let n = input.source.len();
    if [membership, mu0, mu1].iter().any(|v| v.len() != n || v.iter().any(|x| !x.is_finite())) {
        return Err(EstimationError::data_msg("invalid retained nuisance predictions"));
    }
    let source: Vec<_> = input.source.iter().map(|s| f64::from(*s)).collect();
    let membership_logloss =
        antecedent_learn::diagnose(PredictionTask::BinaryProbability, &source, membership)
            .logloss
            .ok_or_else(|| EstimationError::data_msg("empty membership role"))?;
    let mut outcome_rmse = [0.; 2];
    for arm in 0..2 {
        let predictions = if arm == 0 { mu0 } else { mu1 };
        let rows: Vec<_> = (0..n)
            .filter(|i| input.source[*i] && usize::from(input.treatment[*i]) == arm)
            .collect();
        let observed: Vec<_> = rows.iter().map(|i| input.outcome[*i]).collect();
        let predicted: Vec<_> = rows.iter().map(|i| predictions[*i]).collect();
        outcome_rmse[arm] =
            antecedent_learn::diagnose(PredictionTask::Regression, &observed, &predicted)
                .rmse
                .ok_or_else(|| EstimationError::data_msg("empty source outcome role"))?;
    }
    Ok(TrialNuisanceDiagnostics { membership_logloss, outcome_rmse })
}

/// Recomputable score inputs and complete outer-bootstrap bookkeeping.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialAipwEstimate {
    /// Target nonparticipant ATE.
    pub estimate: f64,
    /// Nominal percentile interval, withheld when any replicate fails.
    pub interval: Option<(f64, f64)>,
    /// Why no interval was produced.
    pub uncertainty_reason: Option<String>,
    /// Successful replicate estimates, in original replicate order.
    pub replicates: Vec<(u32, f64)>,
    /// Failed replicate count; failures are never discarded to publish an interval.
    pub failures: u32,
    /// Out-of-fold source membership probabilities.
    pub membership: Vec<f64>,
    /// Out-of-fold source outcome mean under treatment zero.
    pub mu0: Vec<f64>,
    /// Out-of-fold source outcome mean under treatment one.
    pub mu1: Vec<f64>,
    /// Fitted nuisance implementations, in role/fold order.
    pub provenance: Vec<antecedent_learn::LearnerProvenance>,
    /// Held-out losses, separate from overlap and causal inference.
    pub diagnostics: TrialNuisanceDiagnostics,
    /// Separate source-selection and randomization overlap diagnostics.
    pub overlap: crate::TransportOverlapReport,
}

/// Validate certificate/provider compatibility before fitting any nuisance.
/// # Errors
/// Wrong formula, covariates, sampling input, or inference settings.
pub fn validate_trial_aipw(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &TrialAipwOptions,
) -> Result<(), EstimationError> {
    let over: &[VariableId] = match id {
        TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. } => &[],
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            ..
        } => over,
        _ => {
            return Err(EstimationError::data_msg(
                "learner trial AIPW requires a direct or standardization certificate",
            ));
        }
    };
    let mut expected: Vec<_> = over.iter().map(|v| v.raw()).collect();
    expected.sort_unstable();
    let mut actual = input.features.clone();
    actual.sort_unstable();
    let n = input.source.len();
    if expected != actual
        || input.covariates.len() != actual.len()
        || n == 0
        || n > u32::MAX as usize
        || input.outcome.len() != n
        || input.treatment.len() != n
        || input.randomization.len() != n
        || input.covariates.iter().any(|col| col.len() != n || col.iter().any(|v| !v.is_finite()))
        || options.folds < 2
        || options.folds > usize::from(u16::MAX) + 1
        || !options.coverage_level.is_finite()
        || !(0.0..1.0).contains(&options.coverage_level)
        || options.coverage_level == 0.0
    {
        return Err(EstimationError::data_msg("invalid certified trial AIPW input or options"));
    }
    for i in 0..n {
        if input.source[i]
            && (!input.outcome[i].is_finite()
                || !input.randomization[i].is_finite()
                || input.randomization[i] <= 0.0
                || input.randomization[i] >= 1.0)
        {
            return Err(EstimationError::data_msg(
                "source outcomes and known randomization probabilities are invalid",
            ));
        }
    }
    let strata = strata(input);
    if strata.iter().any(|rows| rows.len() < options.folds) {
        return Err(EstimationError::data_msg(
            "each source arm and target sample needs at least one row per fold",
        ));
    }
    options.outcome.validate().map_err(crate::learn_nuisance::learn_err)?;
    options.membership.validate().map_err(crate::learn_nuisance::learn_err)?;
    Ok(())
}

fn strata(input: &TrialAipwInput) -> [Vec<usize>; 3] {
    let mut rows = [vec![], vec![], vec![]];
    for (i, source) in input.source.iter().enumerate() {
        rows[if *source { usize::from(input.treatment[i]) } else { 2 }].push(i);
    }
    rows
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        Err(EstimationError::Refused {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            message: "trial transport cancelled".into(),
        })
    } else {
        Ok(())
    }
}

/// Fit the certified score with shared OOF roles and joint outer refits.
/// # Errors
/// Invalid inputs, unsupported certificate, cancellation, or numerical fit failure.
pub fn estimate_trial_aipw(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &TrialAipwOptions,
    ctx: &ExecutionContext,
) -> Result<TrialAipwEstimate, EstimationError> {
    validate_trial_aipw(id, input, options)?;
    cancelled(ctx)?;
    let n = input.source.len();
    let bytes = input
        .features
        .len()
        .checked_add(32)
        .and_then(|p| n.checked_mul(p))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(options.bootstrap as usize * 16))
        .ok_or_else(|| EstimationError::data_msg("trial workspace overflow"))?;
    if ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes as u64 > limit) {
        return Err(EstimationError::data_msg("trial workspace budget"));
    }
    let mut folds = vec![0u16; n];
    for rows in strata(input) {
        for (i, row) in rows.into_iter().enumerate() {
            folds[row] = (i % options.folds) as u16;
        }
    }
    let mut result = fit_point(id, input, options, &folds, ctx)?;
    use antecedent_data::{ResamplingPlan, fill_resample_indexes};
    let groups = match input.sampling {
        TrialSampling::NestedCohort => vec![(0..n).collect::<Vec<_>>()],
        TrialSampling::IndependentSamples => vec![
            (0..n).filter(|i| input.source[*i]).collect(),
            (0..n).filter(|i| !input.source[*i]).collect(),
        ],
    };
    // Replicates are independent: each draws from a stream keyed by (group, replicate)
    // and refits every nuisance on its own resample, so they run under the context's
    // thread budget (inner fits stay serial) and come back in replicate order.
    let outcomes = ctx.map_indexed(
        options.bootstrap as usize,
        |index, inner| -> Result<_, EstimationError> {
            cancelled(inner)?;
            let replicate = u32::try_from(index).unwrap_or(u32::MAX);
            let mut rows = Vec::with_capacity(n);
            for (group, members) in groups.iter().enumerate() {
                let mut rng = inner.rng.stream_for(
                    StreamDomain::Estimate,
                    0x5452_0000_0000_0000 | ((group as u64) << 32) | u64::from(replicate),
                );
                let mut selected = Vec::new();
                fill_resample_indexes(
                    ResamplingPlan::IidBootstrap,
                    members.len(),
                    &mut rng,
                    &mut selected,
                )
                .map_err(|e| EstimationError::data_msg(e.to_string()))?;
                rows.extend(selected.into_iter().map(|i| members[i as usize]));
            }
            let draw = TrialAipwInput {
                features: input.features.clone(),
                covariates: input
                    .covariates
                    .iter()
                    .map(|col| rows.iter().map(|i| col[*i]).collect())
                    .collect(),
                outcome: rows.iter().map(|i| input.outcome[*i]).collect(),
                treatment: rows.iter().map(|i| input.treatment[*i]).collect(),
                source: rows.iter().map(|i| input.source[*i]).collect(),
                randomization: rows.iter().map(|i| input.randomization[*i]).collect(),
                sampling: input.sampling,
            };
            let draw_folds: Vec<_> = rows.iter().map(|i| folds[*i]).collect();
            if let Ok(estimate) = fit_point(id, &draw, options, &draw_folds, inner) { Ok((replicate, Some(estimate.estimate))) } else {
                cancelled(inner)?;
                Ok((replicate, None))
            }
        },
    )?;
    for (replicate, estimate) in outcomes {
        match estimate {
            Some(value) => result.replicates.push((replicate, value)),
            None => result.failures += 1,
        }
    }
    if options.bootstrap >= 2 && result.failures == 0 {
        let values: Vec<_> = result.replicates.iter().map(|(_, value)| *value).collect();
        result.interval = Some(crate::statistical_transport::percentile_interval(
            &values,
            options.coverage_level,
        ));
        result.uncertainty_reason = None;
    } else {
        result.uncertainty_reason = Some(
            if result.failures > 0 {
                "bootstrap_replicate_failure"
            } else if options.bootstrap == 0 {
                "bootstrap_not_requested"
            } else {
                "insufficient_bootstrap_replicates"
            }
            .into(),
        );
    }
    Ok(result)
}

fn fit_point(
    id: &TransportIdentification,
    input: &TrialAipwInput,
    options: &TrialAipwOptions,
    folds: &[u16],
    ctx: &ExecutionContext,
) -> Result<TrialAipwEstimate, EstimationError> {
    cancelled(ctx)?;
    let n = input.source.len();
    let mut design = vec![1.0; n];
    for col in &input.covariates {
        design.extend_from_slice(col);
    }
    let x = DesignView::from_column_major(&design, n, input.features.len() + 1)
        .map_err(crate::learn_nuisance::learn_err)?;
    let source: Vec<f64> = input.source.iter().map(|s| f64::from(*s)).collect();
    let membership_factory = resolve_for(options.membership, PredictionTask::BinaryProbability)
        .map_err(crate::learn_nuisance::learn_err)?;
    let outcome_factory = resolve_for(options.outcome, PredictionTask::Regression)
        .map_err(crate::learn_nuisance::learn_err)?;
    let membership = cross_fit_selected(
        membership_factory.as_ref(),
        x,
        TargetView::new(&source),
        folds,
        &vec![true; n],
        ctx,
    )
    .map_err(crate::learn_nuisance::learn_err)?;
    let mut outcomes = Vec::new();
    for arm in [false, true] {
        let eligible: Vec<_> =
            (0..n).map(|i| input.source[i] && input.treatment[i] == arm).collect();
        outcomes.push(
            cross_fit_selected(
                outcome_factory.as_ref(),
                x,
                TargetView::new(&input.outcome),
                folds,
                &eligible,
                ctx,
            )
            .map_err(crate::learn_nuisance::learn_err)?,
        );
    }
    let effect = trial_to_target_effect(
        id,
        &input.outcome,
        &input.treatment,
        &input.source,
        &membership.predictions,
        &input.randomization,
        Some((&outcomes[0].predictions, &outcomes[1].predictions)),
    )?;
    let mut provenance = membership.model_provenance;
    for outcome in &outcomes {
        provenance.extend(outcome.model_provenance.clone());
    }
    let mu1 = outcomes
        .pop()
        .ok_or_else(|| EstimationError::data_msg("missing treated outcome model"))?
        .predictions;
    let mu0 = outcomes
        .pop()
        .ok_or_else(|| EstimationError::data_msg("missing control outcome model"))?
        .predictions;
    cancelled(ctx)?;
    let diagnostics = trial_nuisance_diagnostics(input, &membership.predictions, &mu0, &mu1)?;
    Ok(TrialAipwEstimate {
        diagnostics,
        estimate: effect
            .aipw
            .ok_or_else(|| EstimationError::data_msg("trial AIPW required outcome regressions"))?,
        interval: None,
        uncertainty_reason: None,
        replicates: vec![],
        failures: 0,
        overlap: effect.overlap,
        membership: membership.predictions,
        mu0,
        mu1,
        provenance,
    })
}

/// Check the supported binary estimand and sampling declarations.
/// # Errors
/// Unsupported estimand, target sample, dependence, or weights.
pub fn validate_trial_query(
    query: &antecedent_core::TransportQuery,
) -> Result<(), EstimationError> {
    query.validate().map_err(|e| EstimationError::data_msg(e.to_string()))?;
    match &query.response.functional {
        antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } => match &treatment.grid {
            antecedent_core::GridSpec::Values(values) if values.as_ref() == [0.0, 1.0] => {}
            _ => {
                return Err(EstimationError::data_msg(
                    "trial AIPW requires the binary [0,1] treatment contrast",
                ));
            }
        },
        _ => return Err(EstimationError::data_msg("trial AIPW requires a binary mean contrast")),
    }
    if let Some(catalog) = &query.catalog {
        if catalog.target_sampling != Some(antecedent_core::TargetSampling::RepresentativeSample) {
            return Err(EstimationError::data_msg(
                "trial AIPW requires representative target sampling",
            ));
        }
        if catalog.bindings.iter().any(|b| {
            b.sampling != antecedent_core::SamplingDesign::Independent
                || b.dependence != antecedent_core::DependenceGroup::IndependentStudies
                || b.weights.is_some()
        }) {
            return Err(EstimationError::data_msg(
                "unsupported trial sampling dependence or weights",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{NonZeroThreadCount, Parallelism};
    use antecedent_identify::{PopulationFactor, TransportCertificate};
    use std::sync::Arc;

    fn direct_certificate() -> TransportIdentification {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(PopulationFactor {
                population: Arc::from("target"),
                regime: None,
                variables: Arc::from([]),
                conditioned_on: Arc::from([]),
                interventions: Arc::from([]),
            }),
            certificate: TransportCertificate {
                rule: Arc::from("test"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        }
    }

    /// Randomized source trial (rows 0..80) plus target nonparticipants (rows 80..120).
    fn intercept_only_input(sampling: TrialSampling) -> TrialAipwInput {
        let n = 120u32;
        TrialAipwInput {
            features: vec![],
            covariates: vec![],
            outcome: (0..n).map(|i| 1.0 + f64::from(i % 2) + (f64::from(i) * 0.9).sin()).collect(),
            treatment: (0..n).map(|i| i % 2 == 1).collect(),
            source: (0..n).map(|i| i < 80).collect(),
            randomization: vec![0.5; n as usize],
            sampling,
        }
    }

    #[test]
    fn bootstrap_replicates_do_not_depend_on_the_thread_budget() {
        let options = TrialAipwOptions { bootstrap: 12, ..TrialAipwOptions::default() };
        for sampling in [TrialSampling::NestedCohort, TrialSampling::IndependentSamples] {
            let input = intercept_only_input(sampling);
            let serial_ctx = ExecutionContext::for_tests(11);
            let mut parallel_ctx = ExecutionContext::for_tests(11);
            parallel_ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
            let serial =
                estimate_trial_aipw(&direct_certificate(), &input, &options, &serial_ctx).unwrap();
            let parallel =
                estimate_trial_aipw(&direct_certificate(), &input, &options, &parallel_ctx)
                    .unwrap();
            assert_eq!(serial.replicates.len() + serial.failures as usize, 12);
            assert!(serial.replicates.windows(2).all(|pair| pair[0].0 < pair[1].0));
            assert_eq!(serial.replicates, parallel.replicates);
            assert_eq!(serial.failures, parallel.failures);
            assert_eq!(serial.interval, parallel.interval);
            assert_eq!(serial.estimate, parallel.estimate);
        }
    }
}
