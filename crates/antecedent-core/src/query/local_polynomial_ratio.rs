//! Declared local-polynomial fuzzy discontinuity and regression kink design.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::LocalPolynomialRatioFault;
use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_variables, flag};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// How the local window is chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LocalBandwidthRule {
    /// The declared bandwidth.
    #[default]
    Declared,
    /// The MSE-optimal plug-in bandwidth, capped at the declared bandwidth.
    MseOptimal,
}

/// Design options of a local ratio: bandwidth selection, a calendar-time running variable and
/// manipulation and balance diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalRatioOptions {
    /// Window rule; the declared bandwidth caps a selected one.
    pub bandwidth_rule: LocalBandwidthRule,
    /// Block-bootstrap replicates when the running variable is calendar time; `None` for
    /// independent rows.
    pub temporal_replicates: Option<u32>,
    /// Run the density test of the running variable at the cutoff.
    pub density_test: bool,
    /// Baseline covariates whose jump at the cutoff is reported as a balance diagnostic.
    pub balance_covariates: std::sync::Arc<[VariableId]>,
}

/// Every field is written; covariates keep their declared order.
impl CanonicalEncode for LocalRatioOptions {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let rule = match self.bandwidth_rule {
            LocalBandwidthRule::Declared => 1,
            LocalBandwidthRule::MseOptimal => 2,
        };
        writer.word(rule);
        match self.temporal_replicates {
            Some(replicates) => {
                writer.word(1).word(u64::from(replicates));
            }
            None => {
                writer.word(0);
            }
        }
        writer.word(flag(self.density_test));
        encode_variables(writer, &self.balance_covariates);
    }
}

/// Marker word ahead of the options, written only when they are present so a query without
/// them keeps its canonical bytes.
const OPTIONS_FIELD: u64 = 0x4c52_4f50_5400;

/// A fixed-cutoff, fixed-bandwidth local ratio with observed treatment uptake.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPolynomialRatioQuery {
    /// Continuous outcome variable.
    pub outcome: VariableId,
    /// Observed treatment receipt or dose.
    pub treatment: VariableId,
    /// Running variable that determines threshold exposure.
    pub running: VariableId,
    /// Prespecified threshold.
    pub cutoff: f64,
    /// Prespecified local fitting window radius.
    pub bandwidth: f64,
    /// Estimate a slope kink instead of a level jump.
    pub kink: bool,
    /// Bandwidth selection, calendar-time running variable and diagnostics.
    pub options: Option<LocalRatioOptions>,
}

/// Every field is written; reals as their exact bits.
impl CanonicalEncode for LocalPolynomialRatioQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        self.treatment.encode(writer);
        self.running.encode(writer);
        writer.real(self.cutoff).real(self.bandwidth).word(flag(self.kink));
        if let Some(options) = &self.options {
            writer.word(OPTIONS_FIELD);
            options.encode(writer);
        }
    }
}

impl LocalPolynomialRatioQuery {
    /// Validate the declared local design before compiling a study.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidLocalPolynomialRatio`] when the outcome,
    /// treatment, and running variables are not distinct, or the cutoff is
    /// non-finite or the bandwidth is not positive.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcome == self.treatment
            || self.outcome == self.running
            || self.treatment == self.running
        {
            return Err(QueryError::InvalidLocalPolynomialRatio(LocalPolynomialRatioFault::VariablesNotDistinct));
        }
        if !self.cutoff.is_finite() || !self.bandwidth.is_finite() || self.bandwidth <= 0.0 {
            return Err(QueryError::InvalidLocalPolynomialRatio(LocalPolynomialRatioFault::CutoffOrBandwidth));
        }
        if let Some(options) = &self.options {
            if options.temporal_replicates.is_some_and(|replicates| replicates < 2) {
                return Err(QueryError::InvalidLocalPolynomialRatio(LocalPolynomialRatioFault::BootstrapReplicates));
            }
            let distinct =
                options.balance_covariates.iter().collect::<std::collections::BTreeSet<_>>();
            if distinct.len() != options.balance_covariates.len()
                || options.balance_covariates.iter().any(|covariate| {
                    [self.outcome, self.treatment, self.running].contains(covariate)
                })
            {
                return Err(QueryError::InvalidLocalPolynomialRatio(LocalPolynomialRatioFault::BalanceCovariates));
            }
        }
        Ok(())
    }
}
