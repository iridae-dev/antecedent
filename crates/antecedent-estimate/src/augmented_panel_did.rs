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
    let effect = treated_change / denominator
        - predicted_counterfactual / denominator
        - control_residual_correction / denominator;
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
