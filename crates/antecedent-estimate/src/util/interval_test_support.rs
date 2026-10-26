//! Shared assertions of the estimator-level interval-procedure tests: a request whose tolerances
//! are never met reproduces the fixed-count bootstrap, and one met early stops on a wave boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::IntervalRequest;
use crate::adaptive_resample_waves::StopReason;
use crate::effect_estimate::EffectEstimate;

/// Tolerances no finite run can meet, so the procedure runs to the replicate cap.
pub(crate) fn never_met(wave_size: u32) -> IntervalRequest {
    IntervalRequest {
        level: 0.95,
        wave_size,
        min_replicates: 0,
        se_rel_tol: Some(1e-12),
        half_width_rel_tol: Some(1e-12),
    }
}

/// Tolerances any run meets, so the procedure stops at the first wave that has
/// `min_replicates` successes.
pub(crate) fn met_early(wave_size: u32, min_replicates: u32) -> IntervalRequest {
    IntervalRequest {
        level: 0.95,
        wave_size,
        min_replicates,
        se_rel_tol: Some(1e9),
        half_width_rel_tol: Some(1e9),
    }
}

/// `never` reproduces `fixed` bit for bit and uses the whole cap in whole waves; `early`
/// stops before the cap on a wave boundary with at least the requested successes.
pub(crate) fn assert_procedure(
    fixed: &EffectEstimate,
    never: &EffectEstimate,
    early: &EffectEstimate,
    cap: u32,
    wave: u32,
) {
    assert!(fixed.diagnostics.resampling.is_none(), "a fixed count reports no procedure");
    let never_report =
        never.diagnostics.resampling.as_ref().expect("a requested interval is reported");
    assert_eq!(never_report.stop, StopReason::MaxReplicates);
    assert_eq!(never_report.attempted, cap);
    assert_eq!(never_report.waves.len() as u32, cap / wave);
    assert_eq!(
        never.se_bootstrap().map(f64::to_bits),
        fixed.se_bootstrap().map(f64::to_bits),
        "an unmet tolerance is the fixed-count bootstrap"
    );
    let early_report =
        early.diagnostics.resampling.as_ref().expect("a requested interval is reported");
    assert_ne!(early_report.stop, StopReason::MaxReplicates);
    assert!(early_report.attempted < cap, "a met tolerance shortens the run");
    assert_eq!(early_report.attempted % wave, 0, "the run ends on a wave boundary");
    assert!(early.se_bootstrap().is_some_and(f64::is_finite));
}
