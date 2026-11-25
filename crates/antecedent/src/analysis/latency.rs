//! Latency tiers and compute budgets for interactive / standard / report execute.
//!
//! Mapping is known-equivalent and never silently changes science defaults when
//! unset. Explicit builder knobs always win over a mode's mapped values.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::BayesianBackendKind;

use crate::error::CausalError;
use crate::inference::InferenceMode;

use super::builder::RefuteSuite;

/// Interactive latency profile for button-click estimate paths.
pub const INTERACTIVE_N_DRAWS: usize = 64;
/// Standard (current science default) posterior draws.
pub const STANDARD_N_DRAWS: usize = 1000;
/// Report-tier posterior draws.
pub const REPORT_N_DRAWS: usize = 4000;
/// Interactive bootstrap replicates (analytic / Laplace only).
pub const INTERACTIVE_BOOTSTRAP: u32 = 0;
/// Standard bootstrap replicates (coverage-gated circular-block count).
pub const STANDARD_BOOTSTRAP: u32 = 199;
/// Report-tier bootstrap replicates.
pub const REPORT_BOOTSTRAP: u32 = 200;
/// Identified atoms that fill the default execution memory cap.
///
/// Not a live cap. A posterior of this many identified atoms or fewer keeps the
/// set it fits today. A larger posterior admits as many atoms as the execution
/// budget pays for, at [`INTERACTIVE_ENVELOPE_ATOM_BYTES`] each.
pub const INTERACTIVE_MAX_ENVELOPE_GRAPHS: usize = 16;
/// Bytes quoted for one identified atom before the interactive envelope allocates
/// the subsample that keeps it. [`INTERACTIVE_MAX_ENVELOPE_GRAPHS`] of these fill
/// [`antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES`], the cap when the execution
/// pool sets no limit of its own.
pub const INTERACTIVE_ENVELOPE_ATOM_BYTES: u64 =
    antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES / (INTERACTIVE_MAX_ENVELOPE_GRAPHS as u64);

/// Identified atoms `memory_limit_bytes` pays for, at
/// [`INTERACTIVE_ENVELOPE_ATOM_BYTES`] each.
///
/// `None` is the default search memory cap. The result is never larger than
/// `n_identified`. Zero means one atom's quote does not fit that cap. A posterior
/// that already fits in [`INTERACTIVE_MAX_ENVELOPE_GRAPHS`] atoms does not consult
/// this: that set stays the one it fits today.
#[must_use]
pub const fn interactive_admitted_envelope_atoms(
    n_identified: usize,
    memory_limit_bytes: Option<u64>,
) -> usize {
    let cap = match memory_limit_bytes {
        Some(bytes) => bytes,
        None => antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES,
    };
    let paid = cap / INTERACTIVE_ENVELOPE_ATOM_BYTES;
    if paid >= n_identified as u64 { n_identified } else { paid as usize }
}

/// Interactive prior / posterior predictive-check replicates.
pub const INTERACTIVE_PREDICTIVE_SIMS: u32 = 50;
/// Standard predictive-check replicates (the pre-tier default).
pub const STANDARD_PREDICTIVE_SIMS: u32 = 200;
/// Report-tier predictive-check replicates.
pub const REPORT_PREDICTIVE_SIMS: u32 = 400;

/// Predictive-check replicates for a latency tier; an unset tier keeps the
/// Standard count.
#[must_use]
pub const fn predictive_check_sims(mode: Option<LatencyMode>) -> u32 {
    match mode {
        Some(LatencyMode::Interactive) => INTERACTIVE_PREDICTIVE_SIMS,
        Some(LatencyMode::Report) => REPORT_PREDICTIVE_SIMS,
        Some(LatencyMode::Standard) | None => STANDARD_PREDICTIVE_SIMS,
    }
}

/// Latency tier controlling known-equivalent compute budgets.
///
/// Same estimand, ID status, and assumption recording across tiers; only sample
/// size / backend / validator depth change — and must be visible on the result.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LatencyMode {
    /// Analytic SE or conjugate/Laplace + few draws; no bootstrap; cheap refute; no HMC.
    Interactive,
    /// Current science defaults (`bootstrap=199`, `n_draws=1000`, placebo+RCC).
    Standard,
    /// More posterior draws and the full validation suite; HMC allowed. The bootstrap
    /// count (200) is effectively the Standard count (199), not a larger resample.
    Report,
}

impl LatencyMode {
    /// Stable wire / diagnostics label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Standard => "standard",
            Self::Report => "report",
        }
    }

    /// Parse a wire label (`interactive` / `standard` / `report`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(Self::Interactive),
            "standard" => Some(Self::Standard),
            "report" => Some(Self::Report),
            _ => None,
        }
    }
}

/// Explicit compute budget overrides (field-by-field; `None` keeps mode/default).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ComputeBudget {
    /// Soft wall-clock budget in milliseconds (advisory; not yet enforced as a hard stop).
    pub wall_ms: Option<u64>,
    /// Bootstrap replicate count override.
    pub bootstrap: Option<u32>,
    /// Posterior draw count override (Bayesian paths).
    pub n_draws: Option<usize>,
    /// Refute suite override.
    pub validators: Option<RefuteSuite>,
}

impl ComputeBudget {
    /// Empty overrides.
    #[must_use]
    pub const fn new() -> Self {
        Self { wall_ms: None, bootstrap: None, n_draws: None, validators: None }
    }
}

/// Resolved knobs after applying [`LatencyMode`] + optional [`ComputeBudget`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedLatencyBudget {
    /// Effective latency mode (always set when resolution ran from a mode).
    pub mode: LatencyMode,
    /// Bootstrap replicates.
    pub bootstrap: u32,
    /// Refute suite.
    pub refute: RefuteSuite,
    /// Posterior draws when Bayesian.
    pub n_draws: usize,
    /// Advisory wall budget in ms, if any.
    pub wall_ms: Option<u64>,
}

impl ResolvedLatencyBudget {
    /// Map a latency mode to its known-equivalent defaults (before explicit overrides).
    #[must_use]
    pub const fn from_mode(mode: LatencyMode) -> Self {
        match mode {
            LatencyMode::Interactive => Self {
                mode,
                bootstrap: INTERACTIVE_BOOTSTRAP,
                refute: RefuteSuite::Cheap,
                n_draws: INTERACTIVE_N_DRAWS,
                wall_ms: None,
            },
            LatencyMode::Standard => Self {
                mode,
                bootstrap: STANDARD_BOOTSTRAP,
                refute: RefuteSuite::PlaceboAndRcc,
                n_draws: STANDARD_N_DRAWS,
                wall_ms: None,
            },
            LatencyMode::Report => Self {
                mode,
                bootstrap: REPORT_BOOTSTRAP,
                refute: RefuteSuite::Full,
                n_draws: REPORT_N_DRAWS,
                wall_ms: None,
            },
        }
    }

    /// Apply field-level [`ComputeBudget`] overrides.
    #[must_use]
    pub const fn with_overrides(mut self, budget: ComputeBudget) -> Self {
        if let Some(b) = budget.bootstrap {
            self.bootstrap = b;
        }
        if let Some(n) = budget.n_draws {
            self.n_draws = n;
        }
        if let Some(v) = budget.validators {
            self.refute = v;
        }
        if budget.wall_ms.is_some() {
            self.wall_ms = budget.wall_ms;
        }
        self
    }
}

/// Refuse HMC outside Report tier (Interactive/Standard stay Laplace/conjugate).
///
/// # Errors
///
/// [`CausalError::Unsupported`] when a non-Report mode pairs with HMC.
pub fn refuse_non_report_hmc(
    mode: LatencyMode,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    if mode == LatencyMode::Report {
        return Ok(());
    }
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    if matches!(cfg.backend, BayesianBackendKind::Hmc) {
        return Err(CausalError::Unsupported {
            message: "HMC requires LatencyMode::Report; use Laplace/conjugate for Interactive/Standard",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::BayesianConfig;

    #[test]
    fn mode_labels_roundtrip() {
        for mode in [LatencyMode::Interactive, LatencyMode::Standard, LatencyMode::Report] {
            assert_eq!(LatencyMode::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(LatencyMode::parse("nope"), None);
    }

    #[test]
    fn explicit_budget_overrides_mode() {
        let resolved = ResolvedLatencyBudget::from_mode(LatencyMode::Interactive).with_overrides(
            ComputeBudget {
                wall_ms: Some(250),
                bootstrap: Some(10),
                n_draws: Some(128),
                validators: Some(RefuteSuite::None),
            },
        );
        assert_eq!(resolved.bootstrap, 10);
        assert_eq!(resolved.n_draws, 128);
        assert_eq!(resolved.refute, RefuteSuite::None);
        assert_eq!(resolved.wall_ms, Some(250));
        assert_eq!(resolved.mode, LatencyMode::Interactive);
    }

    #[test]
    fn non_report_refuses_hmc() {
        let cfg = BayesianConfig::hmc();
        let err =
            refuse_non_report_hmc(LatencyMode::Interactive, &InferenceMode::Bayesian(cfg.clone()))
                .unwrap_err();
        assert!(matches!(err, CausalError::Unsupported { .. }));
        let err_std = refuse_non_report_hmc(LatencyMode::Standard, &InferenceMode::Bayesian(cfg))
            .unwrap_err();
        assert!(matches!(err_std, CausalError::Unsupported { .. }));
        assert!(
            refuse_non_report_hmc(
                LatencyMode::Interactive,
                &InferenceMode::Bayesian(BayesianConfig::laplace()),
            )
            .is_ok()
        );
        assert!(
            refuse_non_report_hmc(
                LatencyMode::Report,
                &InferenceMode::Bayesian(BayesianConfig::hmc()),
            )
            .is_ok()
        );
    }

    #[test]
    fn default_budget_pays_for_the_atoms_that_already_fit() {
        let cap = antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES;
        assert_eq!(INTERACTIVE_ENVELOPE_ATOM_BYTES * INTERACTIVE_MAX_ENVELOPE_GRAPHS as u64, cap);
        assert_eq!(interactive_admitted_envelope_atoms(10, None), 10);
        assert_eq!(interactive_admitted_envelope_atoms(INTERACTIVE_MAX_ENVELOPE_GRAPHS, None), 16);
        assert_eq!(
            interactive_admitted_envelope_atoms(INTERACTIVE_MAX_ENVELOPE_GRAPHS + 1, None),
            INTERACTIVE_MAX_ENVELOPE_GRAPHS
        );
        let wider = cap + INTERACTIVE_ENVELOPE_ATOM_BYTES;
        assert_eq!(interactive_admitted_envelope_atoms(40, Some(wider)), 17);
        assert_eq!(interactive_admitted_envelope_atoms(40, Some(1)), 0);
        assert_eq!(
            interactive_admitted_envelope_atoms(40, Some(INTERACTIVE_ENVELOPE_ATOM_BYTES - 1)),
            0
        );
    }
}
