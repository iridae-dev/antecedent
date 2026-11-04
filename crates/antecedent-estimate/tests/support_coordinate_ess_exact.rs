//! Per-dose support labels from the Kish effective sample size of the Gaussian local weights,
//! against the closed form and pinned reference numbers.
//!
//! The route is `dose_support_report`, which fits the local weights `w_i = exp(-(d_i - at)^2 /
//! (2 h^2))` itself and labels every requested dose from their Kish size `(sum w)^2 / sum w^2`.
//! The expected size is computed here from the dose levels grouped by value, `n_j` rows at level
//! `d_j` with weight `q_j = exp(-(d_j - at)^2 / (2 h^2))`, so that `ESS = (sum n_j q_j)^2 /
//! (sum n_j q_j^2)`, and compared with numbers pinned from an independent Python evaluation of the
//! same definition (no Antecedent code):
//!
//!     levels = [(0, 10), (1, 5), (2, 1), (3, 1), (5, 8)]; h = 1
//!     ESS(0) = 14.647646929243267, ESS(1) = 15.385863459672898, ESS(5) = 8.279976655743262
//!
//! The labels are then derived from the threshold: floor 10 labels the doses (0, 5, 1) as
//! (supported, weak overlap, supported), a floor at the exact Kish size flips only that dose, the
//! summary is the worst label, and a dose outside the observed range is labelled outside the
//! empirical support whatever its weights. The tolerance 1e-10 absorbs only libm `exp` rounding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{DiagnosticScope, SupportReport, SupportStatus};
use antecedent_estimate::dose_grid_functional::dose_support_report;

const LEVELS: [(f64, usize); 5] = [(0.0, 10), (1.0, 5), (2.0, 1), (3.0, 1), (5.0, 8)];
const BANDWIDTH: f64 = 1.0;
const TOLERANCE: f64 = 1e-10;

fn dose() -> Vec<f64> {
    LEVELS.iter().flat_map(|(level, count)| std::iter::repeat(*level).take(*count)).collect()
}

/// Closed-form Kish size at `at`, from the grouped levels.
fn kish_closed_form(at: f64) -> f64 {
    let (mut sum, mut sum_squares) = (0.0, 0.0);
    for (level, count) in LEVELS {
        let q = (-(level - at).powi(2) / (2.0 * BANDWIDTH * BANDWIDTH)).exp();
        sum += count as f64 * q;
        sum_squares += count as f64 * q * q;
    }
    sum * sum / sum_squares
}

fn local_ess(report: &SupportReport) -> Vec<f64> {
    let diagnostic = report
        .diagnostics
        .iter()
        .find(|d| &*d.id == "response.local_ess")
        .expect("local ess diagnostic");
    assert_eq!(diagnostic.scope, DiagnosticScope::PerCoordinate);
    diagnostic.values.to_vec()
}

fn labels(points: &[f64], floor: f64) -> Vec<SupportStatus> {
    dose_support_report(&dose(), points, BANDWIDTH, floor).unwrap().point_status.unwrap().to_vec()
}

#[test]
fn local_ess_is_the_closed_form_kish_size_of_the_gaussian_weights() {
    let pinned = [
        (0.0, 14.647_646_929_243_267),
        (1.0, 15.385_863_459_672_898),
        (5.0, 8.279_976_655_743_262),
    ];
    for (at, reference) in pinned {
        assert!((kish_closed_form(at) - reference).abs() < TOLERANCE, "closed form at {at}");
    }
    let points = [0.0, 5.0, 1.0];
    let report = dose_support_report(&dose(), &points, BANDWIDTH, 10.0).unwrap();
    let ess = local_ess(&report);
    assert_eq!(ess.len(), 3);
    for ((got, at), (_, reference)) in ess.iter().zip(points).zip([pinned[0], pinned[2], pinned[1]])
    {
        assert!((got - reference).abs() < TOLERANCE, "ess at {at}: {got} vs {reference}");
        assert!((got - kish_closed_form(at)).abs() < TOLERANCE, "ess at {at}");
    }
}

#[test]
fn labels_follow_the_threshold_applied_to_the_kish_size_in_dose_order() {
    // ESS (0, 5, 1) = (14.65, 8.28, 15.39): only the middle dose falls under a floor of 10.
    let points = [0.0, 5.0, 1.0];
    let expected = [SupportStatus::Supported, SupportStatus::WeakOverlap, SupportStatus::Supported];
    assert_eq!(labels(&points, 10.0), expected);
    let report = dose_support_report(&dose(), &points, BANDWIDTH, 10.0).unwrap();
    assert_eq!(report.status, SupportStatus::WeakOverlap, "summary is the worst label");
    // Reversing the dose order reverses the labels with the sizes.
    let reversed = [1.0, 5.0, 0.0];
    assert_eq!(labels(&reversed, 10.0), expected);
    assert_eq!(
        local_ess(&dose_support_report(&dose(), &reversed, BANDWIDTH, 10.0).unwrap())
            .iter()
            .map(|v| (v * 1e6).round())
            .collect::<Vec<_>>(),
        [15.385_863, 8.279_977, 14.647_647]
            .iter()
            .map(|v| (v * 1e6_f64).round())
            .collect::<Vec<_>>()
    );
    // A floor of 15 puts the first dose under it too; a floor of 8 leaves every dose supported.
    assert_eq!(
        labels(&points, 15.0),
        [SupportStatus::WeakOverlap, SupportStatus::WeakOverlap, SupportStatus::Supported]
    );
    assert_eq!(labels(&points, 8.0), [SupportStatus::Supported; 3]);
}

#[test]
fn a_floor_at_the_exact_kish_size_flips_exactly_that_dose() {
    for at in [0.0, 1.0, 5.0] {
        let ess = kish_closed_form(at);
        assert_eq!(labels(&[at], ess * (1.0 + 1e-9)), [SupportStatus::WeakOverlap], "above, {at}");
        assert_eq!(labels(&[at], ess * (1.0 - 1e-9)), [SupportStatus::Supported], "below, {at}");
    }
}

#[test]
fn a_dose_outside_the_observed_range_is_outside_whatever_its_weights() {
    // At 5.5 the weights of the eight rows at 5 give a large Kish size, yet the dose is past the
    // observed maximum 5, so its label is the empirical-support failure, not a size label.
    assert!(kish_closed_form(5.5) > 3.0);
    assert_eq!(labels(&[5.5], 3.0), [SupportStatus::OutsideEmpiricalSupport]);
    assert_eq!(labels(&[-0.25], 3.0), [SupportStatus::OutsideEmpiricalSupport]);
}
