//! Actual successful posterior solves and emitted aligned draw rows.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::Cell;
/// Posterior engine work; nested observers include the same actual operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BayesianWorkCounts {
    /// Successful Gaussian posterior parameter solves, including actual auxiliary fits.
    pub model_fits: u64,
    /// Successfully emitted aligned posterior rows, including actual auxiliary sampling.
    pub posterior_draws: u64,
}
thread_local! {
    static COUNTS: Cell<BayesianWorkCounts> = const {
        Cell::new(BayesianWorkCounts { model_fits: 0, posterior_draws: 0 })
    };
}
/// Observe source engine operations, including nested observers.
pub fn count_bayesian_work<R>(work: impl FnOnce() -> R) -> (R, BayesianWorkCounts) {
    let before = COUNTS.with(Cell::get);
    let result = work();
    let after = COUNTS.with(Cell::get);
    (
        result,
        BayesianWorkCounts {
            model_fits: after.model_fits.saturating_sub(before.model_fits),
            posterior_draws: after.posterior_draws.saturating_sub(before.posterior_draws),
        },
    )
}
/// Record a successful posterior solve at the executing numerical engine.
pub fn note_posterior_fit() {
    COUNTS.with(|counts| {
        let mut value = counts.get();
        value.model_fits = value.model_fits.saturating_add(1);
        counts.set(value);
    });
}
/// Record an aligned draw row after the engine has emitted all its coordinates.
pub fn note_posterior_draw() {
    COUNTS.with(|counts| {
        let mut value = counts.get();
        value.posterior_draws = value.posterior_draws.saturating_add(1);
        counts.set(value);
    });
}
