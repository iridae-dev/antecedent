//! Match key and scope facts of one reported interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Construction and scope facts of one reported interval: the calibration
/// match key (`antecedent_io::calibration::CalibrationKeyWire`) and the
/// execution facts its record's scope is checked against.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CalibrationBasis {
    /// Executed support validation mode, empty when not established.
    pub validation: Arc<str>,
    /// Support-matrix query axis name.
    pub query: Arc<str>,
    /// Support-matrix graph axis.
    pub graph_class: Arc<str>,
    /// `fixed` or `graph_posterior`.
    pub structure: Arc<str>,
    /// Data modality the execution ran on.
    pub modality: Arc<str>,
    /// `Frequentist` or `Bayesian`.
    pub inference: Arc<str>,
    /// Resolved plan estimator; empty when none.
    pub estimator: Arc<str>,
    /// Interval method name.
    pub interval_method: Arc<str>,
    /// Analytic SE kind; empty when not analytic.
    pub se_kind: Arc<str>,
    /// Dependence rule.
    pub dependence: Arc<str>,
    /// Posterior construction; empty for Frequentist executions.
    pub posterior: Arc<str>,
    /// Population / functional / contrast / horizon label.
    pub functional: Arc<str>,
    /// Nominal level of the reported interval.
    pub level: f64,
    /// `point` or `partial`.
    pub identification: Arc<str>,
    /// Data-snapshot rows.
    pub row_count: u64,
    /// Resampling replicates that succeeded.
    pub replicates_ok: Option<u32>,
    /// Posterior draws.
    pub posterior_draws: Option<u32>,
    /// Structural mass that is not identified.
    pub unidentified_mass: f64,
}

impl CalibrationBasis {
    /// Bind the validation coordinate established by the originating execution.
    #[must_use]
    pub fn with_validation(mut self, validation: impl Into<Arc<str>>) -> Self {
        self.validation = validation.into();
        self
    }

    /// Construct from every field, in declaration order.
    #[must_use]
    pub fn new(
        key: [Arc<str>; 11],
        level: f64,
        identification: Arc<str>,
        row_count: u64,
        replicates_ok: Option<u32>,
        posterior_draws: Option<u32>,
        unidentified_mass: f64,
    ) -> Self {
        let [
            query,
            graph_class,
            structure,
            modality,
            inference,
            estimator,
            interval_method,
            se_kind,
            dependence,
            posterior,
            functional,
        ] = key;
        Self {
            validation: Arc::from(""),
            query,
            graph_class,
            structure,
            modality,
            inference,
            estimator,
            interval_method,
            se_kind,
            dependence,
            posterior,
            functional,
            level,
            identification,
            row_count,
            replicates_ok,
            posterior_draws,
            unidentified_mass,
        }
    }
}
