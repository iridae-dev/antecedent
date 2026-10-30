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
//! always reported. The 95% normal interval `effect ± z · se_analytic` is published only when a
//! current attesting coverage record measured this construction (the target, the number of
//! declared levels, the folds and the row count are part of the key); otherwise it is withheld
//! with status `scope_not_assessed` and the standard error is labelled **unmeasured**. Only the
//! full-population target has a measurement design; the pairwise subpopulation target has none
//! and stays withheld.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::{LevelMean, LevelRows, MultiLevelPopulation, MultiLevelTarget};

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{EstimationError, MultiLevelAipw, MultiLevelEstimate};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_io::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use serde::{Deserialize, Serialize};

use crate::error::CausalError;

/// Calibration standing of the reported standard error while no coverage record governs it.
pub const MULTI_LEVEL_TREATMENT_CALIBRATION: &str = "unmeasured";
/// Calibration standing once a current attesting coverage record governs the interval.
pub const MULTI_LEVEL_TREATMENT_CALIBRATED: &str = "calibrated";
/// Interval status when a governing coverage record measured the construction at this scope.
pub const MULTI_LEVEL_INTERVAL_CALIBRATED: &str = "calibrated";
/// Interval status when no current attesting record covers this construction and scope.
pub const MULTI_LEVEL_INTERVAL_WITHHELD: &str = "scope_not_assessed";
/// Reason when the contrast has no positive finite standard error to build an interval from.
pub const MULTI_LEVEL_NO_STANDARD_ERROR: &str = "no_positive_standard_error";

/// Query axis of the multi-level contrast coverage records.
pub const MULTI_LEVEL_QUERY: &str = "MultiLevelContrast";
/// Estimator label of the cross-fitted multinomial generalized-propensity AIPW.
pub const MULTI_LEVEL_ESTIMATOR: &str = "propensity.generalized_aipw";
/// Standard-error kind: the plug-in iid influence-function variance of the cross-fitted scores.
pub const MULTI_LEVEL_SE_KIND: &str = "influence";

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

/// The published 95% interval of the contrast, bound to its governing coverage record.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MultiLevelIntervalWire {
    /// Lower end, `effect - z · se_analytic`.
    pub lower: f64,
    /// Upper end, `effect + z · se_analytic`.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Identity of the governing coverage record.
    pub coverage_record: String,
}

fn interval_status_default() -> String {
    MULTI_LEVEL_INTERVAL_WITHHELD.into()
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
    /// Calibration standing of the standard error: `calibrated` when a coverage record governs
    /// the interval, `unmeasured` otherwise.
    pub calibration: String,
    /// The 95% interval `effect ± z · se_analytic`; present only when a current attesting
    /// coverage record measured this construction at this row count.
    #[serde(default)]
    pub interval: Option<MultiLevelIntervalWire>,
    /// `calibrated` when the interval is published, `scope_not_assessed` when it is withheld.
    #[serde(default = "interval_status_default")]
    pub interval_status: String,
    /// Why the interval is withheld (the matcher's reason code), when it is.
    #[serde(default)]
    pub interval_reason: Option<String>,
}

/// Calibration match key and scope of the interval a request would publish.
///
/// `levels` is the size of the declared level set (full-population target) and `folds` the
/// cross-fit fold count; both change the construction, so a record binds only the values it
/// measured. The measurement design builds the same key.
#[must_use]
pub fn multi_level_interval_basis(
    target: MultiLevelTargetWire,
    levels: usize,
    folds: usize,
    row_count: u64,
) -> CalibrationBasisWire {
    let functional = match target {
        MultiLevelTargetWire::FullPopulation => {
            format!("multi_level.full_population.contrast;levels={levels};folds={folds}")
        }
        MultiLevelTargetWire::PairwiseSubpopulation => {
            "multi_level.pairwise_subpopulation.contrast".to_owned()
        }
    };
    CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: MULTI_LEVEL_QUERY.to_owned(),
            graph_class: "NotApplicable".to_owned(),
            structure: "fixed".to_owned(),
            modality: "tabular".to_owned(),
            inference: "Frequentist".to_owned(),
            estimator: MULTI_LEVEL_ESTIMATOR.to_owned(),
            interval_method: antecedent_core::IntervalMethod::AnalyticSe.as_str().to_owned(),
            se_kind: MULTI_LEVEL_SE_KIND.to_owned(),
            dependence: "iid".to_owned(),
            posterior: String::new(),
            functional,
            level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
            identification: "point".to_owned(),
        },
        scope: CalibrationScopeWire {
            row_count,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    }
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
            CausalError::Estimate(EstimationError::refused_detail(
                antecedent_core::reason_code!("invalid_argument"),
                detail,
                message,
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
        Ok(report(&estimate, spec, estimator.folds))
    }
}

/// The 95% normal interval `effect ± z · se` at the reported level, when the standard error is
/// positive and finite.
///
/// The one construction a coverage record measures: the publishing path and the measurement
/// design both form the interval here, so a record describes exactly what is published.
#[must_use]
pub fn multi_level_normal_interval(effect: f64, se: f64) -> Option<(f64, f64)> {
    (effect.is_finite() && se.is_finite() && se > 0.0).then(|| {
        let half = crate::result::reported_se_interval_z() * se;
        (effect - half, effect + half)
    })
}

/// The interval, its status and the withholding reason of one estimate.
fn publish_interval(
    estimate: &MultiLevelEstimate,
    spec: &MultiLevelSpecWire,
    folds: usize,
    n_obs: u64,
) -> (Option<MultiLevelIntervalWire>, &'static str, Option<String>) {
    let levels = spec.levels.as_ref().map_or(0, Vec::len);
    let slot = calibration_slot(&multi_level_interval_basis(spec.target, levels, folds, n_obs));
    let interval = multi_level_normal_interval(estimate.effect.ate, estimate.effect.se_analytic());
    match (slot.record_id, interval) {
        (Some(record), Some((lower, upper))) if slot.status == "calibrated" => {
            let level = crate::result::REPORTED_SE_INTERVAL_LEVEL;
            let wire = MultiLevelIntervalWire { lower, upper, level, coverage_record: record };
            (Some(wire), MULTI_LEVEL_INTERVAL_CALIBRATED, None)
        }
        (Some(_), None) if slot.status == "calibrated" => {
            (None, MULTI_LEVEL_INTERVAL_WITHHELD, Some(MULTI_LEVEL_NO_STANDARD_ERROR.into()))
        }
        _ => (None, MULTI_LEVEL_INTERVAL_WITHHELD, slot.reason),
    }
}

fn report(
    estimate: &MultiLevelEstimate,
    spec: &MultiLevelSpecWire,
    folds: usize,
) -> MultiLevelReportWire {
    let n_obs = estimate.effect.provenance.n_obs.unwrap_or(0);
    let (interval, interval_status, interval_reason) =
        publish_interval(estimate, spec, folds, n_obs);
    MultiLevelReportWire {
        target: estimate.target.as_str().into(),
        target_population: estimate.target_population.as_str().into(),
        active: spec.active,
        control: spec.control,
        effect: estimate.effect.ate,
        se_analytic: estimate.effect.se_analytic(),
        n_obs,
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
        calibration: if interval.is_some() {
            MULTI_LEVEL_TREATMENT_CALIBRATED
        } else {
            MULTI_LEVEL_TREATMENT_CALIBRATION
        }
        .into(),
        interval,
        interval_status: interval_status.into(),
        interval_reason,
    }
}
