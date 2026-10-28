//! Supplied-nuisance augmented two-period panel ATT.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Point result and overlap diagnostics. No interval is calibrated here.
#[derive(Clone, Debug, PartialEq)]
pub struct AugmentedPanelDidFit {
    /// Treated-group augmented ATT.
    pub effect: f64,
    /// Count of treated subjects.
    pub treated_subjects: usize,
    /// Count of comparison subjects.
    pub control_subjects: usize,
    /// Minimum supplied propensity.
    pub propensity_min: f64,
    /// Maximum supplied propensity.
    pub propensity_max: f64,
    /// Kish effective sample size of weighted controls.
    pub effective_control_sample_size: f64,
}

/// Estimate the treated-group ATT from supplied propensity and untreated-change predictions.
pub fn estimate(
    pre: &[f64],
    post: &[f64],
    treated: &[bool],
    propensity: &[f64],
    prediction: &[f64],
) -> Result<AugmentedPanelDidFit, &'static str> {
    let n = pre.len();
    if n == 0
        || post.len() != n
        || treated.len() != n
        || propensity.len() != n
        || prediction.len() != n
    {
        return Err(
            "pre/post outcomes, treatment, propensity, and outcome predictions must have equal non-zero length",
        );
    }
    if pre.iter().chain(post).chain(prediction).any(|value| !value.is_finite()) {
        return Err("outcomes and nuisance predictions must be finite");
    }
    if propensity.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0) {
        return Err(
            "augmented DiD overlap failure: propensity scores must be strictly between zero and one",
        );
    }
    let treated_subjects = treated.iter().filter(|&&value| value).count();
    let control_subjects = n - treated_subjects;
    if treated_subjects == 0 || control_subjects == 0 {
        return Err("augmented DiD requires treated and control subjects");
    }
    let mut treated_change = 0.0;
    let mut predicted_counterfactual = 0.0;
    let mut control_residual_correction = 0.0;
    let mut control_weight = 0.0;
    let mut propensity_min = f64::INFINITY;
    let mut propensity_max: f64 = 0.0;
    let mut squared_weights = 0.0;
    for i in 0..n {
        let p = propensity[i];
        propensity_min = propensity_min.min(p);
        propensity_max = propensity_max.max(p);
        let change = post[i] - pre[i];
        if treated[i] {
            treated_change += change;
            predicted_counterfactual += prediction[i];
        } else {
            let weight = p / (1.0 - p);
            control_weight += weight;
            squared_weights += weight * weight;
            control_residual_correction += weight * (change - prediction[i]);
        }
    }
    let denominator = treated_subjects as f64;
    // The comparison term is the odds-weighted mean of the untreated-change residual with the
    // weights normalised to sum to one (the improved doubly robust estimator): a constant
    // error in the supplied prediction then cancels between the treated and comparison terms
    // instead of leaving the unit-weight mismatch (sum of odds over n_treated) minus one.
    let effect = treated_change / denominator
        - predicted_counterfactual / denominator
        - control_residual_correction / control_weight;
    let effective_control_sample_size = control_weight.powi(2) / squared_weights;
    if !effect.is_finite() || !effective_control_sample_size.is_finite() {
        return Err("augmented DiD estimate overflowed finite precision");
    }
    Ok(AugmentedPanelDidFit {
        effect,
        treated_subjects,
        control_subjects,
        propensity_min,
        propensity_max,
        effective_control_sample_size,
    })
}

/// The augmented ATT of [`estimate`] with each subject weighted by `weights`; `None` when the
/// weighted sample has no treated or no comparison subject. Weights multiply the odds, so an
/// integer weight is the same estimate as repeating the subject.
fn weighted_effect(
    change: &[f64],
    treated: &[bool],
    propensity: &[f64],
    prediction: &[f64],
    weights: &[f64],
) -> Option<f64> {
    let mut treated_weight = 0.0;
    let mut treated_term = 0.0;
    let mut correction = 0.0;
    let mut control_weight = 0.0;
    for i in 0..change.len() {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        if treated[i] {
            treated_weight += w;
            treated_term += w * (change[i] - prediction[i]);
        } else {
            let odds = propensity[i] / (1.0 - propensity[i]);
            control_weight += w * odds;
            correction += w * odds * (change[i] - prediction[i]);
        }
    }
    (treated_weight > 0.0 && control_weight > 0.0)
        .then(|| treated_term / treated_weight - correction / control_weight)
        .filter(|effect| effect.is_finite())
}

/// Whole-cluster resample of the augmented ATT with the supplied nuisance predictions held
/// at their values: every replicate redraws (or reweights) whole clusters and recomputes the
/// estimator. The supplied predictions' own estimation error is not propagated; the interval
/// describes the sampling of subjects given those predictions. Its coverage is unmeasured.
///
/// # Errors
///
/// The inputs fail [`estimate`], fewer than two clusters, or the resample is refused
/// (cancellation, admission).
#[allow(
    clippy::too_many_arguments,
    reason = "the five aligned columns, the clusters and the resample are distinct inputs"
)]
pub fn resample_augmented_panel(
    pre: &[f64],
    post: &[f64],
    treated: &[bool],
    propensity: &[f64],
    prediction: &[f64],
    clusters: &[std::sync::Arc<str>],
    weighting: crate::rollout_resampling::UnitWeighting,
    replicates: u32,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    (AugmentedPanelDidFit, crate::rollout_resampling::ReplicateSummary),
    crate::EstimationError,
> {
    let fit = estimate(pre, post, treated, propensity, prediction)
        .map_err(|message| crate::EstimationError::refused("invalid_argument", message))?;
    if clusters.len() != pre.len() {
        return Err(crate::EstimationError::refused(
            "invalid_argument",
            "augmented panel clusters must align with the subject rows",
        ));
    }
    let mut labels: std::collections::BTreeMap<&str, usize> =
        clusters.iter().map(|label| (label.as_ref(), 0)).collect();
    for (position, slot) in labels.values_mut().enumerate() {
        *slot = position;
    }
    if labels.len() < 2 {
        return Err(crate::EstimationError::refused(
            "option_not_applicable",
            "a whole-cluster resample needs at least two clusters; next attempt: the point \
             estimate without an interval",
        ));
    }
    let index: Vec<usize> = clusters.iter().map(|label| labels[label.as_ref()]).collect();
    let change: Vec<f64> = post.iter().zip(pre).map(|(after, before)| after - before).collect();
    let summary = crate::rollout_resampling::resample_procedure(
        ctx,
        0x4155_4744_0001,
        replicates,
        &[fit.effect],
        |rng| {
            let cluster_weights =
                crate::rollout_resampling::unit_weights(weighting, labels.len(), rng);
            let weights: Vec<f64> = index.iter().map(|c| cluster_weights[*c]).collect();
            weighted_effect(&change, treated, propensity, prediction, &weights)
                .map(|effect| vec![effect])
        },
    )?;
    Ok((fit, summary))
}

#[cfg(test)]
mod resample_tests {
    use std::sync::Arc;

    use super::{estimate, resample_augmented_panel, weighted_effect};
    use crate::rollout_resampling::UnitWeighting;
    use crate::splitmix::FixtureRng;

    #[test]
    fn the_augmented_resample_reproduces_the_point_and_is_thread_invariant() {
        let mut rng = FixtureRng::new(9);
        let (mut pre, mut post, mut treated, mut propensity, mut prediction, mut clusters) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..400 {
            let x = rng.uniform();
            let p = 0.25 + 0.5 * x;
            let d = rng.uniform() < p;
            let base = rng.uniform();
            pre.push(base);
            post.push(base + x + if d { 1.5 } else { 0.0 } + 0.2 * (rng.uniform() - 0.5));
            treated.push(d);
            propensity.push(p);
            prediction.push(x);
            clusters.push(Arc::<str>::from(format!("c{}", i / 4)));
        }
        let run = |ctx: &antecedent_core::ExecutionContext| {
            resample_augmented_panel(
                &pre,
                &post,
                &treated,
                &propensity,
                &prediction,
                &clusters,
                UnitWeighting::Resample,
                199,
                ctx,
            )
            .unwrap()
        };
        let (fit, serial) = run(&antecedent_core::ExecutionContext::for_tests(3));
        let (_, parallel) = run(&antecedent_core::ExecutionContext::production(3, 4));
        assert_eq!(serial, parallel);
        assert!((fit.effect - 1.5).abs() < 0.1, "{}", fit.effect);
        let interval = serial.quantities[0].interval.unwrap();
        assert!(interval[0] < fit.effect && fit.effect < interval[1]);
    }

    #[test]
    fn an_integer_weight_is_the_same_estimate_as_repeating_the_subject() {
        let pre = [1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.0];
        let post = [3.0, 2.5, 2.0, 3.5, 2.0, 3.5, 1.0];
        let treated = [true, true, false, false, true, false, false];
        let propensity = [0.4, 0.7, 0.2, 0.9, 0.5, 0.05, 0.6];
        let prediction = [0.3, 0.1, 0.2, 0.0, 0.4, 0.1, 0.2];
        let change: Vec<f64> = post.iter().zip(&pre).map(|(a, b)| a - b).collect();
        let weights = [2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0];
        let mut rep = (vec![], vec![], vec![], vec![], vec![]);
        for (i, &w) in weights.iter().enumerate() {
            for _ in 0..w as usize {
                rep.0.push(pre[i]);
                rep.1.push(post[i]);
                rep.2.push(treated[i]);
                rep.3.push(propensity[i]);
                rep.4.push(prediction[i]);
            }
        }
        let repeated = estimate(&rep.0, &rep.1, &rep.2, &rep.3, &rep.4).unwrap().effect;
        let weighted = weighted_effect(&change, &treated, &propensity, &prediction, &weights);
        assert!((weighted.unwrap() - repeated).abs() < 1e-12, "{weighted:?} {repeated}");
    }
}
