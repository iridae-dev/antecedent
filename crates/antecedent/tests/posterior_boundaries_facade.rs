//! The typed boundaries of two posteriors without a same-target counterpart, through the
//! facade.
//!
//! * The temporal response-curve posterior is tempered and generalized; an ordinary posterior
//!   of a joint law of the series is refused unless independent innovations are attested.
//! * A trajectory survival contrast under a prescribed regime has a subject-bootstrap interval
//!   conditional on the supplied weights, whose width is checked against the independent
//!   two-sample binomial standard error, and no credible interval: that needs a joint
//!   likelihood the weights do not supply.
//!
//! Every refusal carries the registered `cell_not_licensed` code, the missing premise and the
//! next attempt.
#![allow(clippy::cast_precision_loss, reason = "fixture counts are far below 2^52")]

use antecedent::analysis::posterior_boundaries::{
    SerialErrorModel, TemporalCurvePosteriorMeaning, TrajectoryRows,
    require_ordinary_temporal_curve_posterior, temporal_curve_posterior_meaning,
    trajectory_hazard_credible_interval, trajectory_survival_bootstrap,
    trajectory_survival_contrast,
};

#[test]
fn unmodeled_serial_dependence_refuses_the_ordinary_curve_posterior_with_the_next_attempt() {
    let refused =
        require_ordinary_temporal_curve_posterior(SerialErrorModel::Unmodeled).unwrap_err();
    assert_eq!(refused.reason_code(), Some("cell_not_licensed"));
    let text = refused.to_string();
    assert!(text.contains("temporal_curve.ordinary_posterior_not_licensed"), "{text}");
    assert!(text.contains("next attempt") && text.contains("autoregressive innovation law"));
    assert_eq!(
        temporal_curve_posterior_meaning(SerialErrorModel::Unmodeled),
        TemporalCurvePosteriorMeaning::TemperedGeneralized
    );
    assert!(
        require_ordinary_temporal_curve_posterior(SerialErrorModel::IndependentInnovations).is_ok()
    );
}

struct OnePeriod {
    treatment: Vec<bool>,
    event: Vec<bool>,
    at_risk: Vec<bool>,
    probability: Vec<f64>,
    censoring: Vec<f64>,
}

impl OnePeriod {
    fn new(subjects: usize) -> Self {
        // A deterministic low-discrepancy event draw, independent of the implementation.
        let draw = |i: usize| ((i as f64 + 1.0) * 0.618_033_988_749_895).fract();
        let treatment: Vec<bool> = (0..subjects).map(|i| i % 2 == 0).collect();
        let event =
            (0..subjects).map(|i| draw(i) < if treatment[i] { 0.25 } else { 0.40 }).collect();
        Self {
            treatment,
            event,
            at_risk: vec![true; subjects],
            probability: vec![0.5; subjects],
            censoring: vec![1.0; subjects],
        }
    }

    fn rows(&self) -> TrajectoryRows<'_> {
        TrajectoryRows {
            subjects: self.treatment.len(),
            periods: 1,
            treatment: &self.treatment,
            event: &self.event,
            at_risk: &self.at_risk,
            treatment_probability: &self.probability,
            censoring_weight: &self.censoring,
            minimum_probability: 0.2,
        }
    }

    fn share(&self, arm: bool) -> (f64, f64) {
        let members = self.treatment.iter().filter(|value| **value == arm).count() as f64;
        let events = self
            .treatment
            .iter()
            .zip(&self.event)
            .filter(|(value, hit)| **value == arm && **hit)
            .count() as f64;
        (members, events / members)
    }
}

#[test]
fn trajectory_bootstrap_interval_matches_the_binomial_standard_error() {
    let data = OnePeriod::new(600);
    let intervals =
        trajectory_survival_bootstrap(&data.rows(), &[true], &[false], 999, 17).unwrap();
    let (n1, p1) = data.share(true);
    let (n0, p0) = data.share(false);
    let standard_error = (p1 * (1.0 - p1) / n1 + p0 * (1.0 - p0) / n0).sqrt();
    let point = trajectory_survival_contrast(&data.rows(), &[true], &[false]).unwrap();
    assert!((point.survival_difference - ((1.0 - p1) - (1.0 - p0))).abs() < 1e-12);
    let [lower, upper] = intervals.survival_difference;
    assert!(lower < point.survival_difference && point.survival_difference < upper);
    let reference = 2.0 * 1.959_964 * standard_error;
    assert!(((upper - lower) / reference - 1.0).abs() < 0.15, "{lower} {upper} {reference}");
    assert_eq!(intervals.replicates_requested, 999);
    assert!(intervals.replicates_ok >= 899);
}

#[test]
fn trajectory_credible_interval_always_refuses_naming_the_missing_likelihood() {
    let data = OnePeriod::new(200);
    let refused = trajectory_hazard_credible_interval(&data.rows(), &[true], &[false]).unwrap_err();
    assert_eq!(refused.reason_code(), Some("cell_not_licensed"));
    let text = refused.to_string();
    assert!(text.contains("trajectory_survival.credible_interval_not_licensed"), "{text}");
    assert!(text.contains("joint likelihood") && text.contains("g-formula"), "{text}");
    assert!(text.contains("next attempt"), "{text}");
    // Invalid trajectories are a data error, never the boundary.
    let invalid =
        trajectory_hazard_credible_interval(&data.rows(), &[false], &[false]).unwrap_err();
    assert_ne!(invalid.reason_code(), Some("cell_not_licensed"));
    // The replicate floor is a data error of the bootstrap.
    let few = trajectory_survival_bootstrap(&data.rows(), &[true], &[false], 10, 1).unwrap_err();
    assert!(few.to_string().contains("199 to 100000"), "{few}");
}
