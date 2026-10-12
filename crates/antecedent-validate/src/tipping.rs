//! Shared tipping arithmetic.
//!
//! The marginal-sensitivity and joint-mechanism brackets walk the same midpoint.
//! The one-factor kernel reports its crossing in closed form, with two domain
//! policies that the callers already differed on. Roots stay the arithmetic of
//! the loop or formula that used to be inlined.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Smallest bisection tolerance either sensitivity contract admits.
pub(crate) const MIN_TOLERANCE: f64 = 1e-12;

/// Where the first crossing of a monotone predicate landed.
pub(crate) enum Crossing {
    /// `reached(origin)`.
    AtOrigin,
    /// The predicate is false at `extent` (or the interval is empty).
    NotInInterval,
    /// `reached` is false at `lower` and true at `upper`.
    Bracket {
        /// Largest evaluated point that does not reach the threshold.
        lower: f64,
        /// Smallest evaluated point that reaches the threshold.
        upper: f64,
        /// Midpoint evaluations performed.
        iterations: usize,
    },
}

/// Bisect the first point of `[origin, extent]` at which `reached` is true.
///
/// `before_midpoint` runs after the iteration counter increments and before the
/// midpoint is formed, so a budget charge can stop the search without moving
/// the bracket. `max_iterations` is the safety cap; pass `usize::MAX` when the
/// tolerance alone bounds the loop.
///
/// # Errors
///
/// Returns the error `before_midpoint` returns.
pub(crate) fn first_crossing<E>(
    origin: f64,
    extent: f64,
    tolerance: f64,
    max_iterations: usize,
    mut reached: impl FnMut(f64) -> bool,
    mut before_midpoint: impl FnMut(usize) -> Result<(), E>,
) -> Result<Crossing, E> {
    if reached(origin) {
        return Ok(Crossing::AtOrigin);
    }
    if extent <= origin || !reached(extent) {
        return Ok(Crossing::NotInInterval);
    }
    let (mut lower, mut upper) = (origin, extent);
    let mut iterations = 0usize;
    while upper - lower > tolerance && iterations < max_iterations {
        iterations += 1;
        before_midpoint(iterations)?;
        let middle = lower + (upper - lower) / 2.0;
        if middle <= lower || middle >= upper {
            break;
        }
        if reached(middle) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    Ok(Crossing::Bracket { lower, upper, iterations })
}

/// `(threshold - baseline) / slope`, or `None` when the slope is zero.
pub(crate) fn slope_fraction(baseline: f64, threshold: f64, slope: f64) -> Option<f64> {
    (slope != 0.0).then(|| (threshold - baseline) / slope)
}

/// Keep `fraction` when it lies in `[0, max_fraction]`.
pub(crate) fn fraction_in_closed_domain(fraction: f64, max_fraction: f64) -> Option<f64> {
    (fraction >= 0.0 && fraction <= max_fraction).then_some(fraction)
}

/// Keep `fraction` when it lies in `[0, max_fraction + 1e-12]`, then clamp it
/// into `[0, max_fraction]`. The conditional-mechanism route uses this slop;
/// the kernel route uses [`fraction_in_closed_domain`].
pub(crate) fn fraction_in_sloppy_domain(fraction: f64, max_fraction: f64) -> Option<f64> {
    (fraction >= 0.0 && fraction <= max_fraction + 1e-12)
        .then_some(fraction.clamp(0.0, max_fraction))
}
