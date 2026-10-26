//! Observational survival effects at a declared horizon.
//!
//! The caller supplies follow-up durations, event indicators, an observed binary treatment
//! and the covariates they declare sufficient for adjustment, and declares three things that
//! have no default: the horizon `t*`, the active and control treatment levels, and the
//! censoring assumption. The route then estimates the survival probability difference
//! `S_1(t*) - S_0(t*)` and the restricted mean survival time difference with a cross-fitted
//! augmented inverse-probability and censoring-weighted estimator (see
//! [`antecedent_estimate::observational_survival`]).
//!
//! No graph is identified or checked: the caller owns that the declared covariates are a valid
//! adjustment set. The report lists the assumptions the numbers rest on. The standard errors
//! are influence-function standard errors whose calibration is **unmeasured**, and no interval
//! is produced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::observational_survival::CensoringAssumption;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::observational_survival::{
    ArmSurvival, ObservationalSurvival as Estimator, SurvivalData,
};
use serde::{Deserialize, Serialize};

use super::multi_level_treatment::AdjustmentColumn;
use crate::error::CausalError;

/// Calibration standing of the reported standard errors.
pub const OBSERVATIONAL_SURVIVAL_CALIBRATION: &str = "unmeasured";

/// The declared horizon, levels and censoring assumption.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationalSurvivalSpecWire {
    /// Treatment level of the minuend arm.
    pub active: f64,
    /// Treatment level of the subtrahend arm.
    pub control: f64,
    /// The horizon both estimands are evaluated at; required.
    #[serde(default)]
    pub horizon: Option<f64>,
    /// `independent_within_arm` or `at_random_given_covariates`; required.
    #[serde(default)]
    pub censoring: Option<String>,
    /// Cross-fit folds; `None` uses the library default.
    #[serde(default)]
    pub folds: Option<usize>,
    /// Propensity floor; `None` uses the library default.
    #[serde(default)]
    pub min_propensity: Option<f64>,
    /// Floor on the probability of remaining uncensored through the horizon.
    #[serde(default)]
    pub min_censoring_survival: Option<f64>,
    /// Seed of the cross-fit fold plan.
    #[serde(default)]
    pub seed: u64,
}

/// A survival-effect request.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalSurvivalRequest {
    /// Observed follow-up time of each row.
    pub duration: Vec<f64>,
    /// One for an observed event, zero for censoring.
    pub event: Vec<f64>,
    /// Observed treatment level.
    pub treatment: Vec<f64>,
    /// Declared adjustment columns.
    pub adjustment: Vec<AdjustmentColumn>,
    /// Horizon, levels and censoring assumption.
    pub spec: ObservationalSurvivalSpecWire,
}

/// One arm's summaries, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ArmSurvivalWire {
    /// Treatment level of the arm.
    pub level: f64,
    /// Doubly robust survival probability at the horizon.
    pub survival: f64,
    /// Influence-function standard error of `survival`.
    pub survival_se: f64,
    /// Doubly robust restricted mean survival time through the horizon.
    pub rmst: f64,
    /// Influence-function standard error of `rmst`.
    pub rmst_se: f64,
    /// Complete-case rows in the arm.
    pub rows: u64,
    /// Observed events in the arm.
    pub events: u64,
    /// Smallest out-of-fold propensity of the arm.
    pub propensity_min: f64,
    /// Smallest out-of-fold probability of remaining uncensored through the horizon.
    pub censoring_survival_min: f64,
}

/// An active-minus-control contrast, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContrastWire {
    /// Contrast value.
    pub estimate: f64,
    /// Influence-function standard error.
    pub se: f64,
}

/// The self-describing result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ObservationalSurvivalReportWire {
    /// The horizon the estimands refer to.
    pub horizon: f64,
    /// The censoring assumption the estimate was computed under.
    pub censoring: String,
    /// The estimands, in words.
    pub estimands: Vec<String>,
    /// The assumptions the numbers rest on; the caller owns the untestable ones.
    pub assumptions: Vec<String>,
    /// The active arm.
    pub active: ArmSurvivalWire,
    /// The control arm.
    pub control: ArmSurvivalWire,
    /// `S_active(t*) - S_control(t*)`.
    pub survival_difference: ContrastWire,
    /// Restricted mean survival time difference through the horizon.
    pub rmst_difference: ContrastWire,
    /// Complete-case rows used.
    pub n_obs: u64,
    /// Rows dropped for a missing cell.
    pub rows_dropped_missing: u64,
    /// Original row index of each used row.
    pub row_index: Vec<u64>,
    /// Per-row score of the survival difference.
    pub survival_difference_scores: Vec<f64>,
    /// Per-row score of the restricted mean survival time difference.
    pub rmst_difference_scores: Vec<f64>,
    /// Calibration standing of the standard errors.
    pub calibration: String,
}

/// Observational survival effects.
pub struct ObservationalSurvivalEffect;

impl ObservationalSurvivalEffect {
    /// Estimate the declared survival effects.
    ///
    /// # Errors
    /// `required_option_missing` for an absent horizon or censoring assumption;
    /// `invalid_argument` for malformed columns or declarations; `treatment_level_undeclared`
    /// for a row at neither level; `positivity_violated` for a horizon beyond follow-up or a
    /// propensity or censoring probability below its floor; `arm_not_populated`; and the
    /// estimator's GLM and Cox refusals.
    pub fn evaluate(
        request: &ObservationalSurvivalRequest,
    ) -> Result<ObservationalSurvivalReportWire, CausalError> {
        let refused = |code: &'static str, detail: &str, message: &str| {
            CausalError::Estimate(EstimationError::refused_detail(code, detail, message))
        };
        let spec = &request.spec;
        let Some(horizon) = spec.horizon else {
            return Err(refused(
                antecedent_core::reason_code!("required_option_missing"),
                "survival_effect.horizon_missing",
                "survival at an observational horizon needs the horizon t* declared; the survival curve is not defined without it",
            ));
        };
        let censoring = match spec.censoring.as_deref() {
            None => {
                return Err(refused(
                    antecedent_core::reason_code!("required_option_missing"),
                    "survival_effect.censoring_assumption_missing",
                    "censoring must be declared ignorable: independent_within_arm, or at_random_given_covariates (the censoring hazard depends on the arm and the declared covariates only). Informative censoring is not identified from these data",
                ));
            }
            Some("independent_within_arm") => CensoringAssumption::IndependentWithinArm,
            Some("at_random_given_covariates") => CensoringAssumption::AtRandomGivenCovariates,
            Some(other) => {
                return Err(refused(
                    antecedent_core::reason_code!("invalid_argument"),
                    "survival_effect.censoring_assumption_unknown",
                    &format!(
                        "unknown censoring assumption {other:?}; use independent_within_arm or at_random_given_covariates"
                    ),
                ));
            }
        };
        let rows = request.duration.len();
        let mut names: Vec<&str> = request.adjustment.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if request.adjustment.iter().any(|c| c.values.len() != rows)
            || names.iter().any(|n| n.is_empty())
            || names.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(refused(
                antecedent_core::reason_code!("invalid_argument"),
                "survival_effect.invalid_data",
                "adjustment columns must match the rows and have distinct, non-empty names",
            ));
        }
        let mut estimator = Estimator::new(spec.active, spec.control, horizon, censoring);
        if let Some(folds) = spec.folds {
            estimator.folds = folds;
        }
        if let Some(floor) = spec.min_propensity {
            estimator.min_propensity = floor;
        }
        if let Some(floor) = spec.min_censoring_survival {
            estimator.min_censoring_survival = floor;
        }
        let covariates: Vec<&[f64]> =
            request.adjustment.iter().map(|c| c.values.as_slice()).collect();
        let ctx = ExecutionContext::production_default(spec.seed);
        let estimate = estimator
            .estimate(
                &SurvivalData {
                    duration: &request.duration,
                    event: &request.event,
                    treatment: &request.treatment,
                    covariates: &covariates,
                },
                &ctx,
            )
            .map_err(CausalError::from)?;
        let arm = |a: &ArmSurvival| ArmSurvivalWire {
            level: a.level,
            survival: a.survival,
            survival_se: a.survival_se,
            rmst: a.rmst,
            rmst_se: a.rmst_se,
            rows: a.rows,
            events: a.events,
            propensity_min: a.propensity_min,
            censoring_survival_min: a.censoring_survival_min,
        };
        let contrast = |c: antecedent_estimate::observational_survival::SurvivalContrast| {
            ContrastWire { estimate: c.estimate, se: c.se }
        };
        Ok(ObservationalSurvivalReportWire {
            horizon,
            censoring: censoring.as_str().into(),
            estimands: estimands(horizon),
            assumptions: assumptions(censoring),
            active: arm(&estimate.active),
            control: arm(&estimate.control),
            survival_difference: contrast(estimate.survival_difference),
            rmst_difference: contrast(estimate.rmst_difference),
            n_obs: estimate.n_obs,
            rows_dropped_missing: estimate.rows_dropped_missing,
            row_index: estimate.row_index.iter().map(|id| id.get()).collect(),
            survival_difference_scores: estimate.survival_difference_scores,
            rmst_difference_scores: estimate.rmst_difference_scores,
            calibration: OBSERVATIONAL_SURVIVAL_CALIBRATION.into(),
        })
    }
}

fn estimands(horizon: f64) -> Vec<String> {
    vec![
        format!(
            "survival probability difference S_active({horizon}) - S_control({horizon}), with S_a(t) = P(T(a) > t)"
        ),
        format!(
            "restricted mean survival time difference through {horizon}, with R_a = E[min(T(a), {horizon})]"
        ),
    ]
}

fn assumptions(censoring: CensoringAssumption) -> Vec<String> {
    let censoring = match censoring {
        CensoringAssumption::IndependentWithinArm => {
            "censoring is independent of the event time and of the covariates within each arm"
        }
        CensoringAssumption::AtRandomGivenCovariates => {
            "coarsening at random: given the arm and the declared covariates, the censoring hazard does not depend on the event time"
        }
    };
    vec![
        "conditional exchangeability: the declared covariates are a sufficient adjustment set (declared, not checked)".into(),
        "positivity: both treatment levels have propensity above the floor at every row".into(),
        "consistency and no interference between units".into(),
        censoring.into(),
        "censoring positivity: the probability of remaining uncensored through the horizon is above the floor".into(),
        "the nuisance Cox proportional-hazards models for the event and censoring hazards, or the propensity model, are correctly specified (double robustness needs one side)".into(),
    ]
}
