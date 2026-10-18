//! Time-uniform confidence sequence for the mean of a `[0, 1]`-bounded arm.
//!
//! Let `X_1, X_2, ...` be the outcomes of one arm in the order that arm was
//! pulled, independent with common mean `mu` and bounded in `[0, 1]`. By
//! Hoeffding's lemma `X_t - mu` is sub-Gaussian with variance proxy
//! `sigma^2 = 1/4`. With `S_m = sum_{t <= m} (X_t - mu)`, the normal mixture
//! over `lambda ~ N(0, tau^2)` of `exp(lambda S_m - lambda^2 m sigma^2 / 2)`
//! is a nonnegative supermartingale with initial value one (Robbins 1970;
//! Howard, Ramdas, `McAuliffe` and Sekhon 2021, normal mixture boundary), so by
//! Ville's inequality, with probability at least `1 - alpha`, simultaneously
//! for all `m >= 1`,
//!
//! ```text
//! |S_m| < sqrt( (1 + m sigma^2 tau^2) / tau^2 * ln( (1 + m sigma^2 tau^2) / alpha^2 ) ).
//! ```
//!
//! Parameterizing `tau^2 = 1 / (sigma^2 n0)` makes `n0` the sample size at
//! which the boundary is tightest and gives the radius of the mean
//! `sigma sqrt((m + n0) ln((m + n0) / (n0 alpha^2))) / m`. The sequence is the
//! running intersection of those intervals with `[0, 1]`.
//!
//! # Predictable plug-in empirical Bernstein sequence
//!
//! The Hoeffding proxy `sigma = 1/2` is the worst case over all `[0, 1]`
//! laws, so the mixture sequence above does not narrow when the outcomes vary
//! little. [`SequenceKind::PredictablePlugInBernstein`] is the variance-adaptive
//! betting sequence of Waudby-Smith and Ramdas (2024, *Journal of the Royal
//! Statistical Society B*, "Estimating means of bounded random variables by
//! betting", the predictable plug-in empirical Bernstein interval). With
//! predictable tilts `lambda_t in [0, 1)` (functions of `X_1, ..., X_{t-1}`
//! only) and `v_t = 4 (X_t - mu_hat_{t-1})^2`, the process
//! `exp( sum_i lambda_i (X_i - m) - v_i psi_E(lambda_i) )`, with
//! `psi_E(lambda) = (-ln(1 - lambda) - lambda) / 4`, is a nonnegative
//! supermartingale at the true mean `m = mu` (Fan, Grama and Liu 2015), and
//! Ville's inequality applied to it and to the reflected outcomes gives, with
//! probability at least `1 - alpha`, simultaneously for all `t >= 1`,
//!
//! ```text
//! | sum_i lambda_i X_i / sum_i lambda_i - mu |
//!     <= ( ln(2 / alpha) + sum_i v_i psi_E(lambda_i) ) / sum_i lambda_i .
//! ```
//!
//! The tuning is the paper's: `mu_hat_t = (1/2 + sum_{i <= t} X_i) / (t + 1)`,
//! `sigma_hat_t^2 = (1/4 + sum_{i <= t} (X_i - mu_hat_i)^2) / (t + 1)` and
//! `lambda_t = min( sqrt( 2 ln(2 / alpha) / (sigma_hat_{t-1}^2 t ln(1 + t)) ),
//! c )` with the truncation `c =` [`PLUG_IN_LAMBDA_CAP`] (any `c < 1` keeps the
//! theorem). Nothing is tuned to a sample size: the tilt follows the running
//! variance, so the sequence is as tight as the data's own spread allows at
//! whatever size the arm has when it is read. The sequence is again the
//! running intersection of the intervals with `[0, 1]`.
//!
//! The guarantee is a finite-time theorem under the stated premise (the
//! observations of the arm, in pull order, are independent with a common mean:
//! the "stack of rewards" model of a stochastic bandit, with outcome delays
//! that do not depend on the outcomes). It is *not* a measured coverage
//! record of any whole procedure, and the callers label the sequence
//! uncalibrated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AdaptiveInferenceError as Error;

/// Most observations one sequence accepts.
pub const MAX_SEQUENCE_OBSERVATIONS: usize = 50_000_000;

/// Sub-Gaussian standard deviation proxy of a `[0, 1]` variable (Hoeffding).
const SUB_GAUSSIAN_SD: f64 = 0.5;
/// Truncation `c` of the predictable plug-in tilt; the theorem holds for any
/// `c` in `(0, 1)`.
pub const PLUG_IN_LAMBDA_CAP: f64 = 0.75;
const CANCEL_STRIDE: usize = 4096;

/// The construction of a confidence sequence. Artifacts record the kind they
/// were produced under, so a retained sequence replays under its own
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SequenceKind {
    /// The normal-mixture Hoeffding sequence tuned to a sample size.
    MixtureHoeffding,
    /// The predictable plug-in empirical Bernstein betting sequence.
    PredictablePlugInBernstein,
}

impl SequenceKind {
    /// Stable label of the construction, recorded in artifacts.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MixtureHoeffding => "mixture_hoeffding",
            Self::PredictablePlugInBernstein => "predictable_plug_in_empirical_bernstein",
        }
    }

    /// The construction a recorded label names.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        [Self::MixtureHoeffding, Self::PredictablePlugInBernstein]
            .into_iter()
            .find(|kind| kind.label() == label)
    }
}

/// The confidence sequence of one arm at its final sample size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSequence {
    /// Observations of the arm.
    pub count: u64,
    /// Sample mean; `NaN` when the arm has no observation, in which case the
    /// sequence is the whole unit interval.
    pub mean: f64,
    /// Lower end of the running intersection.
    pub lower: f64,
    /// Upper end of the running intersection.
    pub upper: f64,
}

fn check_level(alpha: f64, tuning_samples: f64) -> Result<(), Error> {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the miscoverage level must lie in (0, 1)"));
    }
    if !(tuning_samples.is_finite() && tuning_samples > 0.0) {
        return Err(Error::InvalidParameter("the tuning sample size must be finite and positive"));
    }
    Ok(())
}

/// Radius of the mixture-Hoeffding interval for the mean after `count`
/// observations, at two-sided miscoverage `alpha`, tuned for `tuning_samples`.
///
/// # Errors
/// `count` of zero, `alpha` outside `(0, 1)`, or a non-positive tuning size.
pub fn mixture_hoeffding_radius(count: u64, alpha: f64, tuning_samples: f64) -> Result<f64, Error> {
    check_level(alpha, tuning_samples)?;
    if count == 0 {
        return Err(Error::Shape("a radius needs at least one observation"));
    }
    let m = count as f64;
    let total = m + tuning_samples;
    let boundary =
        SUB_GAUSSIAN_SD * (total * (total / (tuning_samples * alpha * alpha)).ln()).sqrt();
    Ok(boundary / m)
}

/// The running intersection of one arm's confidence intervals, updated one
/// observation at a time.
///
/// Pushing the outcomes of an arm in pull order and reading the state after
/// any prefix gives the same sequence as [`bounded_mean_confidence_sequence`]
/// or [`plug_in_confidence_sequence`] over that prefix, so a procedure that
/// looks at the arm several times reads the sequence at each look without
/// recomputing it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceState {
    alpha: f64,
    boundary: Boundary,
    count: u64,
    sum: f64,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Boundary {
    MixtureHoeffding { tuning_samples: f64 },
    PlugIn(PlugIn),
}

/// Running sums of the predictable plug-in sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PlugIn {
    /// `sum (X_i - mu_hat_i)^2`, the numerator of the variance estimate.
    residual_ss: f64,
    /// `sum lambda_i`.
    tilt_sum: f64,
    /// `sum lambda_i X_i`.
    tilted_sum: f64,
    /// `sum v_i psi_E(lambda_i)`.
    penalty_sum: f64,
}

impl PlugIn {
    /// Take in the `count`-th outcome `value` (`previous_sum` and `sum` are
    /// the outcome sums before and after it) and return the centre and radius
    /// of the interval at this size.
    fn step(
        &mut self,
        alpha: f64,
        count: u64,
        previous_sum: f64,
        sum: f64,
        value: f64,
    ) -> (f64, f64) {
        let t = count as f64;
        let previous_mean = (0.5 + previous_sum) / t;
        let previous_variance = (0.25 + self.residual_ss) / t;
        let level = (2.0 / alpha).ln();
        let tilt =
            (2.0 * level / (previous_variance * t * (1.0 + t).ln())).sqrt().min(PLUG_IN_LAMBDA_CAP);
        let deviation = value - previous_mean;
        let psi = 0.25 * (-(-tilt).ln_1p() - tilt);
        self.tilt_sum += tilt;
        self.tilted_sum += tilt * value;
        self.penalty_sum += 4.0 * deviation * deviation * psi;
        let mean = (0.5 + sum) / (t + 1.0);
        self.residual_ss += (value - mean) * (value - mean);
        (self.tilted_sum / self.tilt_sum, (level + self.penalty_sum) / self.tilt_sum)
    }
}

impl SequenceState {
    /// An empty mixture-Hoeffding sequence at two-sided miscoverage `alpha`,
    /// tuned for `tuning_samples`.
    ///
    /// # Errors
    /// `alpha` outside `(0, 1)` or a non-positive tuning size.
    pub fn new(alpha: f64, tuning_samples: f64) -> Result<Self, Error> {
        Self::of_kind(SequenceKind::MixtureHoeffding, alpha, tuning_samples)
    }

    /// An empty predictable plug-in empirical Bernstein sequence at two-sided
    /// miscoverage `alpha`. It is tuned by the data, not to a sample size.
    ///
    /// # Errors
    /// `alpha` outside `(0, 1)`.
    pub fn predictable_plug_in(alpha: f64) -> Result<Self, Error> {
        Self::of_kind(SequenceKind::PredictablePlugInBernstein, alpha, 1.0)
    }

    /// An empty sequence of the given construction. `tuning_samples` is read
    /// by the mixture-Hoeffding sequence only, but is validated for both.
    ///
    /// # Errors
    /// `alpha` outside `(0, 1)` or a non-positive tuning size.
    pub fn of_kind(kind: SequenceKind, alpha: f64, tuning_samples: f64) -> Result<Self, Error> {
        check_level(alpha, tuning_samples)?;
        let boundary = match kind {
            SequenceKind::MixtureHoeffding => Boundary::MixtureHoeffding { tuning_samples },
            SequenceKind::PredictablePlugInBernstein => Boundary::PlugIn(PlugIn::default()),
        };
        Ok(Self { alpha, boundary, count: 0, sum: 0.0, lower: 0.0, upper: 1.0 })
    }

    /// Add the next outcome of the arm.
    ///
    /// # Errors
    /// An outcome outside `[0, 1]` or not finite.
    pub fn push(&mut self, value: f64) -> Result<(), Error> {
        if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
            return Err(Error::InvalidParameter("a bounded outcome must lie in [0, 1]"));
        }
        let previous_sum = self.sum;
        self.sum += value;
        self.count += 1;
        let (centre, radius) = match &mut self.boundary {
            Boundary::MixtureHoeffding { tuning_samples } => (
                self.sum / self.count as f64,
                mixture_hoeffding_radius(self.count, self.alpha, *tuning_samples)?,
            ),
            Boundary::PlugIn(plug_in) => {
                plug_in.step(self.alpha, self.count, previous_sum, self.sum, value)
            }
        };
        self.lower = self.lower.max(centre - radius);
        self.upper = self.upper.min(centre + radius);
        Ok(())
    }

    /// Outcomes pushed so far.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// The sequence at the current sample size.
    ///
    /// # Errors
    /// An empty intersection (evidence that the bounded independent premise
    /// failed).
    pub fn reading(&self) -> Result<ArmSequence, Error> {
        if self.count == 0 {
            return Ok(ArmSequence { count: 0, mean: f64::NAN, lower: 0.0, upper: 1.0 });
        }
        if self.lower > self.upper {
            return Err(Error::Numerical(
                "the intersected confidence sequence is empty: the bounded independent premise failed",
            ));
        }
        Ok(ArmSequence {
            count: self.count,
            mean: self.sum / self.count as f64,
            lower: self.lower,
            upper: self.upper,
        })
    }
}

fn read_sequence(
    mut state: SequenceState,
    values: &[f64],
    cancelled: &dyn Fn() -> bool,
) -> Result<ArmSequence, Error> {
    if values.len() > MAX_SEQUENCE_OBSERVATIONS {
        return Err(Error::Limit("more than fifty million observations"));
    }
    for (index, &x) in values.iter().enumerate() {
        if index % CANCEL_STRIDE == 0 && cancelled() {
            return Err(Error::Cancelled);
        }
        state.push(x)?;
    }
    state.reading()
}

/// The running-intersection mixture-Hoeffding confidence sequence of a
/// bounded arm, evaluated at the final observation, from outcomes in pull
/// order.
///
/// # Errors
/// An outcome outside `[0, 1]` or not finite, a bad level, more than
/// [`MAX_SEQUENCE_OBSERVATIONS`] observations, cancellation, or an empty
/// intersection (evidence that the bounded independent premise failed).
pub fn bounded_mean_confidence_sequence(
    values: &[f64],
    alpha: f64,
    tuning_samples: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<ArmSequence, Error> {
    read_sequence(SequenceState::new(alpha, tuning_samples)?, values, cancelled)
}

/// The running-intersection predictable plug-in empirical Bernstein
/// confidence sequence of a bounded arm, evaluated at the final observation,
/// from outcomes in pull order.
///
/// # Errors
/// As [`bounded_mean_confidence_sequence`], without the tuning size.
pub fn plug_in_confidence_sequence(
    values: &[f64],
    alpha: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<ArmSequence, Error> {
    read_sequence(SequenceState::predictable_plug_in(alpha)?, values, cancelled)
}

/// Bounds for the difference of two arm means, `a - b`, from two sequences
/// each constructed at `alpha / 2` (a union bound over the two arms).
#[must_use]
pub fn contrast_bounds(a: &ArmSequence, b: &ArmSequence) -> (f64, f64) {
    (a.lower - b.upper, a.upper - b.lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform stream (`SplitMix64`) for the seeded checks.
    fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / 9_007_199_254_740_992.0
        }
    }

    /// Success proportions of participants with six responses each at rate `p`.
    fn proportions(p: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut next = uniform_stream(seed);
        (0..n).map(|_| (0..6).filter(|_| next() < p).count() as f64 / 6.0).collect()
    }

    fn read(kind: SequenceKind, alpha: f64, values: &[f64]) -> ArmSequence {
        let mut state = SequenceState::of_kind(kind, alpha, 100.0).unwrap();
        for &x in values {
            state.push(x).unwrap();
        }
        state.reading().unwrap()
    }

    #[test]
    fn radius_matches_the_closed_form() {
        // sigma = 1/2, n0 = 100, alpha = 0.05, m = 100:
        // 0.5 sqrt(200 ln(200 / (100 * 0.0025))) / 100.
        let r = mixture_hoeffding_radius(100, 0.05, 100.0).unwrap();
        let expected = 0.5 * (200.0_f64 * (800.0_f64).ln()).sqrt() / 100.0;
        assert!((r - expected).abs() < 1e-15);
        assert!((r * 100.0 - 18.281_974_356_819_244).abs() < 1e-9);
    }

    #[test]
    fn a_value_outside_the_unit_interval_is_refused() {
        let outcome = bounded_mean_confidence_sequence(&[0.2, 1.2], 0.05, 10.0, &|| false);
        assert!(matches!(outcome, Err(Error::InvalidParameter(_))));
        let outcome = plug_in_confidence_sequence(&[0.2, -0.1], 0.05, &|| false);
        assert!(matches!(outcome, Err(Error::InvalidParameter(_))));
        assert!(matches!(SequenceState::predictable_plug_in(1.0), Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn kinds_round_trip_through_their_labels() {
        for kind in [SequenceKind::MixtureHoeffding, SequenceKind::PredictablePlugInBernstein] {
            assert_eq!(SequenceKind::from_label(kind.label()), Some(kind));
        }
        assert_eq!(SequenceKind::from_label("hoeffding"), None);
    }

    #[test]
    fn a_constant_half_outcome_has_the_hand_computed_plug_in_interval() {
        // Every outcome is 1/2: mu_hat is 1/2 throughout, so every deviation and v_i are zero,
        // sigma_hat_{t-1}^2 = 1/(4 t) and the tilt sqrt(8 ln(2 / alpha) / ln(1 + t)) exceeds the
        // cap, which binds. The interval is 1/2 +- ln(2 / alpha) / (c t) with c = 3/4, so at
        // t = 20 and alpha = 0.05 its radius is ln(40) / 15.
        let values = [0.5; 20];
        let sequence = plug_in_confidence_sequence(&values, 0.05, &|| false).unwrap();
        let radius = 40.0_f64.ln() / 15.0;
        assert_eq!(sequence.count, 20);
        assert!((sequence.lower - (0.5 - radius)).abs() < 1e-12, "{}", sequence.lower);
        assert!((sequence.upper - (0.5 + radius)).abs() < 1e-12, "{}", sequence.upper);
        assert!((radius - 0.245_925_296_940_929_07).abs() < 1e-12);
    }

    #[test]
    fn the_plug_in_sequence_matches_an_independent_computation() {
        // Alternating 0, 1 at alpha = 0.5, computed outside this crate from the paper's
        // formulas (the formulas are restated in the provenance record): the tilt is capped through t = 10 and
        // free at t = 20 (lambda_20 = 0.456783...).
        let values: Vec<f64> = (0..20).map(|i| f64::from(i % 2)).collect();
        let at_ten = read(SequenceKind::PredictablePlugInBernstein, 0.5, &values[..10]);
        assert!((at_ten.lower - 0.046_872_831_518_211).abs() < 1e-12, "{}", at_ten.lower);
        assert!((at_ten.upper - 0.919_403_662_129_577).abs() < 1e-12, "{}", at_ten.upper);
        let at_twenty = read(SequenceKind::PredictablePlugInBernstein, 0.5, &values);
        assert!((at_twenty.lower - 0.179_034_331_079_232).abs() < 1e-12, "{}", at_twenty.lower);
        assert!((at_twenty.upper - 0.800_763_004_780_992).abs() < 1e-12, "{}", at_twenty.upper);
        assert!((at_twenty.mean - 0.5).abs() < 1e-15);
    }

    #[test]
    fn the_plug_in_sequence_is_narrower_than_hoeffding_on_low_variance_data() {
        let values: Vec<f64> = (0..400).map(|i| 0.1 + 0.02 * f64::from(i % 3 - 1)).collect();
        let plug_in = read(SequenceKind::PredictablePlugInBernstein, 0.05, &values);
        let hoeffding = read(SequenceKind::MixtureHoeffding, 0.05, &values);
        let (narrow, wide) = (plug_in.upper - plug_in.lower, hoeffding.upper - hoeffding.lower);
        assert!(narrow < 0.25 * wide, "{narrow} versus {wide}");
        // Both contain the mean of the outcomes.
        assert!(plug_in.lower < 0.1 && 0.1 < plug_in.upper);
    }

    #[test]
    fn the_running_intersection_only_narrows_and_prefixes_agree_with_the_batch_form() {
        let values = proportions(0.4, 300, 11);
        for kind in [SequenceKind::MixtureHoeffding, SequenceKind::PredictablePlugInBernstein] {
            let mut state = SequenceState::of_kind(kind, 0.05, 40.0).unwrap();
            let (mut lower, mut upper) = (0.0, 1.0);
            for (index, &x) in values.iter().enumerate() {
                state.push(x).unwrap();
                let reading = state.reading().unwrap();
                assert!(reading.lower >= lower && reading.upper <= upper, "{kind:?} at {index}");
                (lower, upper) = (reading.lower, reading.upper);
                if index % 50 == 49 {
                    let prefix = &values[..=index];
                    let batch = match kind {
                        SequenceKind::MixtureHoeffding => {
                            bounded_mean_confidence_sequence(prefix, 0.05, 40.0, &|| false)
                        }
                        SequenceKind::PredictablePlugInBernstein => {
                            plug_in_confidence_sequence(prefix, 0.05, &|| false)
                        }
                    }
                    .unwrap();
                    assert_eq!(batch, reading);
                }
            }
        }
    }

    #[test]
    fn the_plug_in_sequence_covers_the_known_mean_at_every_size_over_a_seed_set() {
        // 300 seeded streams of participant proportions, the sequence read after every
        // observation: a miss at any size is a miss. The guarantee is conservative (at most
        // alpha = 0.05), so the observed miss count sits far below the 15 that alpha allows.
        for (p, n) in [(0.15, 250), (0.4, 250)] {
            let mut misses = 0;
            for seed in 0..300_u64 {
                let values = proportions(p, n, seed + 1);
                let mut state = SequenceState::predictable_plug_in(0.05).unwrap();
                let mut missed = false;
                for &x in &values {
                    state.push(x).unwrap();
                    match state.reading() {
                        Ok(r) => missed |= !(r.lower <= p && p <= r.upper),
                        Err(_) => missed = true,
                    }
                }
                misses += usize::from(missed);
            }
            assert!(misses <= 15, "{misses} of 300 sequences missed the mean {p}");
        }
    }
}
