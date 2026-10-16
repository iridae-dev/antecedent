//! Observational contrasts between levels of a multi-level treatment, with the population the
//! contrast speaks for declared and reported.
//!
//! A treatment with three or more levels has no single average effect, so the caller chooses
//! the semantics and it is never inferred:
//!
//! * `pairwise_subpopulation` contrasts two declared levels on the rows at one of the two; rows
//!   at other levels are excluded and counted, and the result states that population.
//! * `full_population` standardizes every declared level over all complete-case rows with a
//!   cross-fitted multinomial-logit generalized propensity and per-level outcome regressions, so
//!   any two levels contrast over the same population. Every row must be at a declared level.
//!
//! No graph is identified or checked: the caller owns that the declared adjustment columns are
//! a valid adjustment set. The point estimate and the influence-function standard error are
//! reported; the standard error's calibration is **unmeasured** and no interval is produced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::{LevelMean, LevelRows, MultiLevelPopulation, MultiLevelTarget};

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{EstimationError, MultiLevelAipw, MultiLevelEstimate};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde::{Deserialize, Serialize};

use crate::error::CausalError;

/// Calibration standing of the reported standard error.
pub const MULTI_LEVEL_TREATMENT_CALIBRATION: &str = "unmeasured";

/// Which population semantics a request asks for.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiLevelTargetWire {
    /// The two declared levels on the rows at one of them.
    PairwiseSubpopulation,
    /// All declared levels standardized over all complete-case rows.
    FullPopulation,
}

/// The declared contrast and semantics.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MultiLevelSpecWire {
    /// Population semantics.
    pub target: MultiLevelTargetWire,
    /// Level of the minuend arm.
    pub active: f64,
    /// Level of the subtrahend arm.
    pub control: f64,
    /// The declared level set (required for `full_population`, ignored otherwise).
    #[serde(default)]
    pub levels: Option<Vec<f64>>,
    /// Cross-fit folds of the full-population fit; `None` uses the library default.
    #[serde(default)]
    pub folds: Option<usize>,
    /// Seed of the cross-fit fold plan.
    #[serde(default)]
    pub seed: u64,
}

/// One named adjustment column.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjustmentColumn {
    /// Column name.
    pub name: String,
    /// Values, one per row; `NaN` is a missing cell.
    pub values: Vec<f64>,
}

/// A multi-level contrast request.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiLevelTreatmentRequest {
    /// Outcome, one value per row.
    pub outcome: Vec<f64>,
    /// Observed treatment level, one value per row.
    pub treatment: Vec<f64>,
    /// Declared adjustment columns.
    pub adjustment: Vec<AdjustmentColumn>,
    /// Contrast and semantics.
    pub spec: MultiLevelSpecWire,
}

/// Rows at one level, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct LevelRowsWire {
    /// Treatment level.
    pub level: f64,
    /// Complete-case rows at it.
    pub rows: u64,
}

/// A standardized level mean, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct LevelMeanWire {
    /// Treatment level.
    pub level: f64,
    /// Cross-fitted AIPW mean outcome of the level over the full population.
    pub mean: f64,
    /// Influence-function standard error of the mean.
    pub se: f64,
    /// Smallest out-of-fold generalized propensity of the level.
    pub propensity_min: f64,
    /// Largest out-of-fold generalized propensity of the level.
    pub propensity_max: f64,
}

/// The self-describing result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MultiLevelReportWire {
    /// Semantics the estimate was computed under.
    pub target: String,
    /// Population the contrast is an average over.
    pub target_population: String,
    /// Active level.
    pub active: f64,
    /// Control level.
    pub control: f64,
    /// Active-minus-control contrast.
    pub effect: f64,
    /// Influence-function (analytic) standard error of the contrast.
    pub se_analytic: f64,
    /// Rows the estimate was computed on.
    pub n_obs: u64,
    /// Complete-case rows at each declared level.
    pub rows_by_level: Vec<LevelRowsWire>,
    /// Complete-case rows at no declared level that the target excluded.
    pub excluded_rows: u64,
    /// Standardized mean per declared level (full-population target only).
    pub level_means: Vec<LevelMeanWire>,
    /// Calibration standing of the standard error.
    pub calibration: String,
}

/// Multi-level observational contrasts.
pub struct MultiLevelTreatment;

impl MultiLevelTreatment {
    /// Estimate the declared contrast.
    ///
    /// # Errors
    /// `invalid_argument` for malformed columns or levels; `treatment_level_undeclared` for a full-population
    /// row at an undeclared level; `arm_not_populated` for an empty or too sparse level; and the
    /// estimator's overlap, GLM and regression refusals.
    pub fn evaluate(
        request: &MultiLevelTreatmentRequest,
    ) -> Result<MultiLevelReportWire, CausalError> {
        let invalid = |detail: &str, message: &str| {
            CausalError::Estimate(EstimationError::refused(
                antecedent_core::reason_code!("invalid_argument"),
                format!("{detail}: {message}"),
            ))
        };
        let rows = request.outcome.len();
        if rows == 0 || request.treatment.len() != rows {
            return Err(invalid(
                "treatment_levels.invalid_data",
                "outcome and treatment must be non-empty and of equal length",
            ));
        }
        let columns = request.adjustment.iter();
        if columns.clone().any(|c| c.values.len() != rows)
            || request.treatment.iter().chain(&request.outcome).any(|v| v.is_infinite())
            || columns.flat_map(|c| &c.values).any(|v| v.is_infinite())
        {
            return Err(invalid(
                "treatment_levels.invalid_data",
                "adjustment columns must match the rows and no value may be infinite",
            ));
        }
        let mut names: Vec<&str> = request.adjustment.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if names.iter().any(|n| n.is_empty() || matches!(*n, "treatment" | "outcome"))
            || names.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(invalid(
                "treatment_levels.invalid_data",
                "adjustment names must be distinct, non-empty and not 'treatment' or 'outcome'",
            ));
        }
        let table = TabularData::from_f64_columns(
            [("treatment", request.treatment.as_slice()), ("outcome", request.outcome.as_slice())]
                .into_iter()
                .chain(request.adjustment.iter().map(|c| (c.name.as_str(), c.values.as_slice()))),
        )
        .map_err(EstimationError::from)?;
        let adjustment: Vec<VariableId> = (0..request.adjustment.len())
            .map(|j| VariableId::from_raw(2 + u32::try_from(j).unwrap_or(u32::MAX)))
            .collect();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        );
        let spec = &request.spec;
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            spec.control,
            spec.active,
        );
        let estimator = match spec.target {
            MultiLevelTargetWire::PairwiseSubpopulation => MultiLevelAipw::pairwise_subpopulation(),
            MultiLevelTargetWire::FullPopulation => {
                let levels = spec.levels.clone().ok_or_else(|| {
                    invalid(
                        "treatment_levels.levels_missing",
                        "the full-population target declares its levels",
                    )
                })?;
                MultiLevelAipw::full_population(levels)
            }
        };
        let estimator = match spec.folds {
            Some(folds) => estimator.with_folds(folds),
            None => estimator,
        };
        let ctx = ExecutionContext::production_default(spec.seed);
        let estimate = estimator
            .fit(&table, &estimand, &query, &ctx, AssumptionSet::new())
            .map_err(CausalError::from)?;
        Ok(report(&estimate, spec))
    }
}

fn report(estimate: &MultiLevelEstimate, spec: &MultiLevelSpecWire) -> MultiLevelReportWire {
    MultiLevelReportWire {
        target: estimate.target.as_str().into(),
        target_population: estimate.target_population.as_str().into(),
        active: spec.active,
        control: spec.control,
        effect: estimate.effect.ate,
        se_analytic: estimate.effect.se_analytic(),
        n_obs: estimate.effect.provenance.n_obs.unwrap_or(0),
        rows_by_level: estimate
            .rows_by_level
            .iter()
            .map(|r| LevelRowsWire { level: r.level, rows: r.rows })
            .collect(),
        excluded_rows: estimate.excluded_rows,
        level_means: estimate
            .level_means
            .iter()
            .map(|m| LevelMeanWire {
                level: m.level,
                mean: m.mean,
                se: m.se,
                propensity_min: m.propensity_range.0,
                propensity_max: m.propensity_range.1,
            })
            .collect(),
        calibration: MULTI_LEVEL_TREATMENT_CALIBRATION.into(),
    }
}
