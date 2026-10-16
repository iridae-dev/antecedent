//! Prior/posterior predictive checks, prior sensitivity, MCMC diagnostics, and SBC.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod calibration;
mod mcmc;
mod predictive;
mod prior_sensitivity;
mod sbc;

pub(crate) use mcmc::McmcDiagnosticsCheck;
pub use predictive::{
    PosteriorPredictiveCheck, PredictiveCheckKind, PredictiveCheckReport, PredictiveNoise,
    PriorPredictiveCheck, ResidualDiscrepancy, ResidualStatistic, SerialDiscrepancy,
};
pub use prior_sensitivity::{ExternalAlphaSensitivity, PriorSensitivity, with_prior_sensitivity};
pub use sbc::{SbcReport, SimulationBasedCalibration};

#[cfg(test)]
use calibration::summarize_check;
#[cfg(test)]
use mcmc::mcmc_report;
#[cfg(test)]
use predictive::strided_draw_index;
#[cfg(test)]
use sbc::{
    GeneratingVariance, SBC_BINS, SBC_CHI2_CRITICAL_9DF_P99, draw_generating_parameters, sbc_bin,
    sbc_rank_draws,
};

#[cfg(test)]
mod tests;
