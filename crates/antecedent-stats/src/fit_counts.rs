//! Scoped observation of successfully completed dense least-squares solves.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

thread_local! {
    static SOLVES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Observe actual successful Faer least-squares solves, including nested scopes.
///
/// This observes this thread's executing backend only. It does not infer work from
/// an estimator declaration and does not count failed solves or other backends.
pub fn count_least_squares_solves<R>(work: impl FnOnce() -> R) -> (R, u64) {
    let before = SOLVES.with(std::cell::Cell::get);
    let result = work();
    (result, SOLVES.with(std::cell::Cell::get).saturating_sub(before))
}

pub(crate) fn completed_solve() {
    SOLVES.with(|count| count.set(count.get().saturating_add(1)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

    #[test]
    fn attempt_observer_keeps_actual_rank_failure_separate_from_successful_solves() {
        use antecedent_core::execution_attempt::{Operation, observe_execution};
        let x = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let (attempt, successes) = count_least_squares_solves(|| {
            observe_execution(|| {
                FaerBackend.least_squares(&x, 4, 2, &y, &mut LeastSquaresWorkspace::default())?;
                FaerBackend.least_squares(
                    &[1.0; 8],
                    4,
                    2,
                    &y,
                    &mut LeastSquaresWorkspace::default(),
                )
            })
        });
        assert!(attempt.result.is_err());
        assert_eq!(successes, 1);
        let counts = attempt.report.counts(Operation::LeastSquaresSolve);
        assert_eq!((counts.attempted, counts.completed, counts.failed), (2, 1, 1));
        assert!(attempt.report.is_complete());
    }

    #[test]
    fn observer_counts_real_successes_nested_scopes_and_excludes_rank_failure() {
        let x = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let ((fit, inner), outer) = count_least_squares_solves(|| {
            let fit = FaerBackend
                .least_squares(&x, 4, 2, &y, &mut LeastSquaresWorkspace::default())
                .unwrap();
            let (_, inner) = count_least_squares_solves(|| {
                FaerBackend
                    .least_squares(&x, 4, 2, &y, &mut LeastSquaresWorkspace::default())
                    .unwrap()
            });
            assert!(
                FaerBackend
                    .least_squares(&[1.0; 8], 4, 2, &y, &mut LeastSquaresWorkspace::default())
                    .is_err()
            );
            (fit, inner)
        });
        assert_eq!((inner, outer), (1, 2));
        assert!((fit.coefficients[0] - 1.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-10);
    }
}
