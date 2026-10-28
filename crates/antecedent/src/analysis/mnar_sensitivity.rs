//! Missing-not-at-random (MNAR) sensitivity of a two-arm mean contrast.
//!
//! The facade routes and owns no scientific rule of its own:
//!
//! * [`mnar_sensitivity`] re-runs the declared arm-mean contrast across a declared bounded
//!   grid of a delta-adjusted pattern-mixture shift or an exponential-tilting selection model
//!   of the outcome-missingness mechanism, with the tipping points of the sign, of excluding
//!   the null and of every declared decision;
//! * [`mnar_sensitivity_for_table`] reads the treatment, outcome and covariate columns of a
//!   table (an invalid outcome cell is a missing outcome) and runs the same analysis;
//! * [`mnar_sensitivity_for_study`] starts from a built [`Study`]: it takes the treatment and
//!   outcome of the study's average-effect query and its data, so the sensitivity analysis
//!   re-runs the very estimand the study declared. This is the route the refusal of a
//!   `missing_not_at_random` multiple-imputation declaration names as the next attempt.
//!
//! The adjusted estimators (`regression_adjusted`, `ipw`, `aipw`) impute each missing outcome
//! from the arm's outcome regression plus the departure (the shift, or the mean residual under
//! tilted weights), re-run the estimator on the completed table at every grid point and report
//! a seeded bootstrap sampling interval.
//!
//! Every effect over the range is an assumption-dependent range, never a confidence interval
//! and never empirically established: the departure from missing at random is recorded as an
//! assumption with status `untestable`. The sampling interval at each grid point is conditional
//! on the parameter and is reported beside, never merged with, the range.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{reason_code, CausalQuery, VariableId};
use antecedent_data::TableView;
pub use antecedent_validate::mnar_sensitivity::{
    assumptions as mnar_assumptions, ArmLoading, MnarAssumptionRange, MnarAssumptionRecord,
    MnarConclusionKind, MnarData, MnarDecision, MnarDecisionReport, MnarEstimator, MnarMechanism,
    MnarModel, MnarPoint, MnarSamplingInterval, MnarSensitivityError, MnarSensitivityResult,
    MnarSensitivitySpec, MnarTipping, MnarTippingSide, MNAR_ASSIGNMENT_ASSUMPTION_ID,
    MNAR_ASSUMPTION_RANGE_CLAIM, MNAR_DEPARTURE_ASSUMPTION_ID, MNAR_DEPARTURE_ASSUMPTION_STATUS,
    MNAR_EMPIRICAL_STATUS, MNAR_FAMILY, MNAR_INTERPRETATION, MNAR_MAX_BOOTSTRAP,
    MNAR_MAX_COVARIATES, MNAR_MAX_DECISIONS, MNAR_MAX_DELTA, MNAR_MAX_GRID_POINTS, MNAR_MAX_ROWS,
    MNAR_MAX_TILT, MNAR_MAX_TOLERANCE, MNAR_METHOD, MNAR_MIN_BOOTSTRAP, MNAR_MIN_TOLERANCE,
    MNAR_SAMPLING_BASIS, MNAR_TARGET,
};

use super::builder::DataInput;
use super::execute::Study;

/// The declared estimand re-run across the declared bounded grid of an MNAR departure.
///
/// # Errors
///
/// The typed [`MnarSensitivityError`] of the contract (`mechanism_not_declared`,
/// `unsupported_estimator`, `unbounded_range`, `range_excludes_mar`, `composition_not_licensed`,
/// `invalid_data`, `invalid_loading`, `invalid_level`, `invalid_decision`, `invalid_tolerance`,
/// `invalid_bootstrap`, `covariates_required`, `covariates_not_used`, `design_singular`,
/// `insufficient_observed`, `bounds_exceeded`).
pub fn mnar_sensitivity(
    data: &MnarData<'_>,
    spec: &MnarSensitivitySpec,
) -> Result<MnarSensitivityResult, MnarSensitivityError> {
    antecedent_validate::mnar_sensitivity::mnar_sensitivity(data, spec)
}

fn unreadable(message: String) -> MnarSensitivityError {
    MnarSensitivityError {
        code: reason_code!("invalid_argument"),
        detail: "mnar_sensitivity.invalid_data",
        message,
    }
}

fn read_column<T: TableView>(
    table: &T,
    id: VariableId,
    name: &str,
) -> Result<Vec<f64>, MnarSensitivityError> {
    table
        .float64_values(id)
        .map_err(|error| unreadable(format!("column `{name}` is not numeric: {error}")))
}

fn run_ids<T: TableView>(
    table: &T,
    treatment: VariableId,
    outcome: VariableId,
    covariates: &[VariableId],
    spec: &MnarSensitivitySpec,
) -> Result<MnarSensitivityResult, MnarSensitivityError> {
    let name_of = |id: VariableId| {
        table.schema().get(id).map_or_else(|_| format!("{id:?}"), |v| v.name.to_string())
    };
    let treatment = read_column(table, treatment, &name_of(treatment))?;
    let outcome: Vec<Option<f64>> = read_column(table, outcome, &name_of(outcome))?
        .into_iter()
        .map(|v| (!v.is_nan()).then_some(v))
        .collect();
    let columns = covariates
        .iter()
        .map(|&id| read_column(table, id, &name_of(id)))
        .collect::<Result<Vec<_>, _>>()?;
    let views: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    mnar_sensitivity(&MnarData::new(&treatment, &outcome).with_covariates(&views), spec)
}

/// The analysis of the named `treatment`, `outcome` and `covariates` columns of `table`.
///
/// The treatment column must be coded 0 and 1 with no invalid cell; an invalid cell of the
/// outcome column is a missing outcome.
///
/// # Errors
///
/// `mnar_sensitivity.invalid_data` for an unknown or non-numeric column, otherwise the refusals
/// of [`mnar_sensitivity`].
pub fn mnar_sensitivity_for_table<T: TableView>(
    table: &T,
    treatment: &str,
    outcome: &str,
    covariates: &[&str],
    spec: &MnarSensitivitySpec,
) -> Result<MnarSensitivityResult, MnarSensitivityError> {
    let id_of = |name: &str| {
        table.schema().id_of(name).map_err(|error| unreadable(format!("column `{name}`: {error}")))
    };
    let covariates =
        covariates.iter().map(|name| id_of(name)).collect::<Result<Vec<VariableId>, _>>()?;
    run_ids(table, id_of(treatment)?, id_of(outcome)?, &covariates, spec)
}

/// The analysis of a built study's average-effect query on the study's own tabular data.
///
/// The study's treatment and outcome are the analysed columns and `adjustment` (the study's
/// adjustment set) the covariates of an adjusted estimator; use the difference in means with
/// an empty adjustment set. A study whose query is not an average effect, or whose data are
/// not a table, is refused with `mnar_sensitivity.unsupported_query`.
///
/// # Errors
///
/// `mnar_sensitivity.unsupported_query`, otherwise the refusals of [`mnar_sensitivity`].
pub fn mnar_sensitivity_for_study(
    study: &Study,
    adjustment: &[VariableId],
    spec: &MnarSensitivitySpec,
) -> Result<MnarSensitivityResult, MnarSensitivityError> {
    let (CausalQuery::AverageEffect(query), DataInput::Tabular(table)) =
        (&study.query, &study.data)
    else {
        return Err(MnarSensitivityError {
            code: reason_code!("route_not_supported"),
            detail: "mnar_sensitivity.unsupported_query",
            message: "the sensitivity analysis re-runs an average-effect query on tabular data"
                .to_owned(),
        });
    };
    run_ids(table, query.treatment, query.outcome, adjustment, spec)
}
