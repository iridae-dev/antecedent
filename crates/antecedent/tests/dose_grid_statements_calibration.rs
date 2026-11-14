//! Coverage of the further statements of the randomized dose-grid local quadratic: the
//! bias-aware pointwise interval under a declared bound on the third derivative of the dose
//! response, and the Gaussian multiplier sup-t simultaneous band over a grid of doses.
//!
//! Every design draws independent rows with a dose uniform on `[0, 4]` and unit Gaussian
//! outcome noise, and fits at the fixed bandwidth 0.6 on the grid `1, 1.5, 2, 2.5, 3`:
//!
//! * Under the exact quadratic mean `1 + .5 d + .25 d^2` and a declared bound of zero the
//!   smoothing bias is zero, the bias-aware interval is the plain interval and the band is the
//!   sup-t band alone: both are held two-sided at nominal. The band is scored as one joint
//!   event (every grid dose inside its envelope).
//! * Under the cubic mean `1 + .5 d + .25 d^2 + (M / 6) (d - 2)^3`, whose third derivative is
//!   `M` everywhere, the declared bound `M` is exact. The interval and the band widen by the
//!   worst smoothing bias over every curve within the bound, so they are held to at least
//!   nominal for this one curve.
//!
//! The truths are the closed forms of the mean, its derivative and the contrast of two levels.
//!
//! The engine keys no bias-aware statement at run time, so the records name the construction
//! here, on the axes of the family's measured records (query `DoseResponse`, structure `fixed`,
//! validation `not_applicable`). A refused fit is a skip; a fit that publishes no band where one
//! is scored is a miss.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "test scaffolding converts small row counts to floating point"
)]

mod common;

use antecedent_core::ExecutionContext;
use antecedent_estimate::dose_grid_functional::{
    BiasAwareDoseGrid, ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest,
    estimate_dose_grid_bias_aware,
};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian,
    grid_n, map_replicates, n_sim, stream_seed, uniform,
};

/// Fixed Gaussian kernel bandwidth, the one the family's measured records ran at.
const BANDWIDTH: f64 = 0.6;

/// Doses are uniform on `[0, DOSE_TOP]`.
const DOSE_TOP: f64 = 4.0;

/// Rows at the base grid point.
const BASE_ROWS: usize = 500;

/// Doses the level, the derivative and the band are evaluated at.
const GRID: [f64; 5] = [1.0, 1.5, 2.0, 2.5, 3.0];

/// Doses of the contrast, `m(to) - m(from)`.
const CONTRAST: (f64, f64) = (1.0, 3.0);

/// Gaussian multiplier replicates behind each band.
const BAND_REPLICATES: usize = 999;

/// Third derivative of the cubic dose response, and the bound declared for it.
const CURVATURE: f64 = 1.5;

/// Interval method of a pointwise interval.
const POINTWISE: &str = "analytic_se";

/// Interval method of the simultaneous band.
const BAND: &str = "simultaneous_band";

/// What a bias-aware record under a non-zero bound covers.
const BIAS_AWARE_TARGET: &str =
    "the dose response covered at least nominally under the declared third-derivative bound";

/// The exact quadratic mean.
fn quadratic_level(d: f64) -> f64 {
    1.0 + 0.5 * d + 0.25 * d * d
}

fn quadratic_derivative(d: f64) -> f64 {
    0.5 + 0.5 * d
}

/// The quadratic mean plus a cubic term whose third derivative is [`CURVATURE`] everywhere.
fn cubic_level(d: f64) -> f64 {
    quadratic_level(d) + CURVATURE / 6.0 * (d - 2.0).powi(3)
}

fn cubic_derivative(d: f64) -> f64 {
    quadratic_derivative(d) + CURVATURE / 2.0 * (d - 2.0).powi(2)
}

/// One measured statement: which functional, under which mean and declared bound.
#[derive(Clone, Copy)]
struct Design {
    test: &'static str,
    dgp: &'static str,
    functional: DoseFunctional,
    /// The generating mean.
    mean: fn(f64) -> f64,
    /// Declared bound on the absolute third derivative of the mean.
    bound: f64,
    seed_base: u64,
}

impl Design {
    fn name(self) -> &'static str {
        match self.functional {
            DoseFunctional::Level => "level",
            DoseFunctional::Derivative => "derivative",
            DoseFunctional::Contrast { .. } => "contrast",
        }
    }

    /// The construction of a statement of this design, named here.
    fn construction(self, interval: &str, se_kind: &str) -> Construction {
        Construction {
            query: "DoseResponse".into(),
            graph_class: "Dag".into(),
            structure: "fixed".into(),
            modality: "tabular".into(),
            inference: "Frequentist".into(),
            estimator: "dose_grid_bias_aware".into(),
            interval_method: interval.into(),
            se_kind: se_kind.into(),
            dependence: "iid".into(),
            posterior: String::new(),
            functional: format!(
                "{}.bandwidth:{:016x}.curvature_bound:{:016x}",
                self.name(),
                BANDWIDTH.to_bits(),
                self.bound.to_bits()
            ),
            identification: "point".into(),
            reported_level: REPORTED_LEVEL,
        }
    }

    /// Fit one independent sample; `None` when the engine refuses it.
    fn fit(self, rows: usize, seed: u64) -> Option<BiasAwareDoseGrid> {
        let mut unit = uniform(seed, 0xD05E);
        let mut noise = gaussian(stream_seed(seed, 0xD05F));
        let (mut dose, mut outcome) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
        for _ in 0..rows {
            let d = DOSE_TOP * unit();
            dose.push(d);
            outcome.push((self.mean)(d) + noise());
        }
        let derivative = matches!(self.functional, DoseFunctional::Derivative);
        let grid = !matches!(self.functional, DoseFunctional::Contrast { .. });
        let request = DoseGridRequest {
            dose: &dose,
            outcome: &outcome,
            grid: &GRID,
            bandwidth: BANDWIDTH,
            bandwidth_range: (0.1, 1.0),
            minimum_local_ess: 30.0,
            functional: self.functional,
            claims: ClaimSet { pointwise_level: !derivative, derivative, simultaneous_band: grid },
            design: DoseDesign::RandomizedDose,
        };
        estimate_dose_grid_bias_aware(
            &request,
            self.bound,
            BAND_REPLICATES,
            &ExecutionContext::for_tests(seed),
        )
        .ok()
    }
}

/// The band envelope and truth of the coordinate that decides joint coverage: the first
/// coordinate the band misses, or, when it covers every one, the coordinate whose truth sits
/// closest to an edge. `None` when the fit carries no band over exactly `truths`.
fn deciding_band_coordinate(fit: &BiasAwareDoseGrid, truths: &[f64]) -> Option<((f64, f64), f64)> {
    let band = fit.band.as_ref()?;
    if band.lower.len() != truths.len() || band.upper.len() != truths.len() {
        return None;
    }
    let mut closest: Option<(f64, (f64, f64), f64)> = None;
    for (slot, &truth) in truths.iter().enumerate() {
        let (lower, upper) = (band.lower[slot], band.upper[slot]);
        if !(lower <= truth && truth <= upper) {
            return Some(((lower, upper), truth));
        }
        let margin = (truth - lower).min(upper - truth) / (upper - lower);
        if closest.is_none_or(|(least, _, _)| margin < least) {
            closest = Some((margin, (lower, upper), truth));
        }
    }
    closest.map(|(_, envelope, truth)| (envelope, truth))
}

/// Measure `design`: one pointwise tally per named coordinate and, when `band` is set, the
/// joint coverage of the simultaneous band. `conservative` names the target of a statement
/// that is at least nominal by construction; such a tally is held to the one-sided floor.
fn measure(
    design: Design,
    targets: &[(String, f64)],
    band: bool,
    conservative: Option<&'static str>,
) {
    let rows = grid_n(BASE_ROWS);
    let scope = ScopeFacts {
        row_count: rows as u64,
        replicates_ok: None,
        posterior_draws: None,
        unidentified_mass: 0.0,
    };
    let fits = map_replicates(n_sim(), |rep| design.fit(rows, stream_seed(design.seed_base, rep)));
    let pointwise_key = RecordKey { test: design.test, dgp: design.dgp, interval: POINTWISE };
    let pointwise = design.construction(POINTWISE, "local_quadratic_HC0_bias_aware");
    let mut tallies: Vec<CoverageTally> = targets
        .iter()
        .enumerate()
        .map(|(index, (name, truth))| {
            let mut tally =
                CoverageTally::for_record(pointwise_key, REPORTED_LEVEL).labelled(name.clone());
            for fit in &fits {
                let Some(fit) = fit else {
                    tally.skip();
                    continue;
                };
                tally.bind_validation(&pointwise, scope, "not_applicable");
                tally.record(
                    fit.intervals.get(index).map(|interval| (interval.lower, interval.upper)),
                    *truth,
                );
            }
            tally
        })
        .collect();
    if band {
        let truths: Vec<f64> = targets.iter().map(|(_, truth)| *truth).collect();
        let key = RecordKey { test: design.test, dgp: design.dgp, interval: BAND };
        let construction = design.construction(BAND, "sup_t_gaussian_multiplier_bias_aware");
        let band_scope = ScopeFacts {
            replicates_ok: Some(u32::try_from(BAND_REPLICATES).expect("a small count")),
            ..scope
        };
        let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).labelled("band");
        for fit in &fits {
            let Some(fit) = fit else {
                tally.skip();
                continue;
            };
            tally.bind_validation(&construction, band_scope, "not_applicable");
            match deciding_band_coordinate(fit, &truths) {
                Some((envelope, truth)) => tally.record(Some(envelope), truth),
                // A fit that publishes no band over the grid is a miss.
                None => tally.record(None, truths[0]),
            }
        }
        tallies.push(tally);
    }
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| match conservative {
                Some(target) => tally.assert_one_sided(CoverageSide::Upper, target),
                None => tally.assert(),
            })
            .err()
            .map(|error| {
                error
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| error.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                    .unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}: {}", design.test, failures.join("; "));
}

/// One named truth per grid dose.
fn grid_targets(label: &str, truth: fn(f64) -> f64) -> Vec<(String, f64)> {
    GRID.iter().map(|&at| (format!("{label}@{at}"), truth(at))).collect()
}

/// The exact quadratic mean under a declared bound of zero: the level at every grid dose and
/// the sup-t band over the grid, two-sided at nominal.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_grid_zero_bound_level_intervals_and_band() {
    measure(
        Design {
            test: "dose_grid_zero_bound_level_intervals_and_band",
            dgp: "randomized_uniform_quadratic_mean_gaussian_errors_zero_curvature_bound",
            functional: DoseFunctional::Level,
            mean: quadratic_level,
            bound: 0.0,
            seed_base: 0xD6_0001,
        },
        &grid_targets("level", quadratic_level),
        true,
        None,
    );
}

/// The exact quadratic mean under a declared bound of zero: the derivative at every grid dose
/// and the sup-t band over the grid, two-sided at nominal.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_grid_zero_bound_derivative_intervals_and_band() {
    measure(
        Design {
            test: "dose_grid_zero_bound_derivative_intervals_and_band",
            dgp: "randomized_uniform_quadratic_mean_gaussian_errors_zero_curvature_bound",
            functional: DoseFunctional::Derivative,
            mean: quadratic_level,
            bound: 0.0,
            seed_base: 0xD6_0002,
        },
        &grid_targets("derivative", quadratic_derivative),
        true,
        None,
    );
}

/// The cubic mean under its exact third-derivative bound: the bias-aware interval of the level
/// at every grid dose and the bias-aware band, at least nominal.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_grid_bias_aware_level_intervals_and_band() {
    measure(
        Design {
            test: "dose_grid_bias_aware_level_intervals_and_band",
            dgp: "randomized_uniform_cubic_mean_gaussian_errors_exact_curvature_bound",
            functional: DoseFunctional::Level,
            mean: cubic_level,
            bound: CURVATURE,
            seed_base: 0xD6_0003,
        },
        &grid_targets("level", cubic_level),
        true,
        Some(BIAS_AWARE_TARGET),
    );
}

/// The cubic mean under its exact third-derivative bound: the bias-aware interval of the
/// derivative at every grid dose and the bias-aware band, at least nominal.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_grid_bias_aware_derivative_intervals_and_band() {
    measure(
        Design {
            test: "dose_grid_bias_aware_derivative_intervals_and_band",
            dgp: "randomized_uniform_cubic_mean_gaussian_errors_exact_curvature_bound",
            functional: DoseFunctional::Derivative,
            mean: cubic_level,
            bound: CURVATURE,
            seed_base: 0xD6_0004,
        },
        &grid_targets("derivative", cubic_derivative),
        true,
        Some(BIAS_AWARE_TARGET),
    );
}

/// The cubic mean under its exact third-derivative bound: the bias-aware interval of the
/// contrast of two levels, at least nominal. A contrast has no band.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_grid_bias_aware_contrast_interval() {
    let (from, to) = CONTRAST;
    measure(
        Design {
            test: "dose_grid_bias_aware_contrast_interval",
            dgp: "randomized_uniform_cubic_mean_gaussian_errors_exact_curvature_bound",
            functional: DoseFunctional::Contrast { from, to },
            mean: cubic_level,
            bound: CURVATURE,
            seed_base: 0xD6_0005,
        },
        &[(format!("contrast:{from}-{to}"), cubic_level(to) - cubic_level(from))],
        false,
        Some(BIAS_AWARE_TARGET),
    );
}
