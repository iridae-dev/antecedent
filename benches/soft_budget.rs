//! Wall-clock soft budgets of the Criterion smoke (`cargo bench -- --test`).
//!
//! Each budget is measured on the reference development machine, so it fails a miss there.
//! Other hardware, such as CI runners, cannot hold those wall times; with
//! `ANTECEDENT_ADVISORY_SOFT_BUDGETS` set, a miss is reported as a warning instead.
//! Relative regressions on any hardware are the bench-regression job's to judge.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Fail when `within` is false, or warn when budgets are advisory on this machine.
pub fn check(within: bool, message: impl FnOnce() -> String) {
    if within {
        return;
    }
    let message = message();
    if std::env::var_os("ANTECEDENT_ADVISORY_SOFT_BUDGETS").is_some() {
        println!("::warning::{message}");
    } else {
        panic!("{message}");
    }
}
