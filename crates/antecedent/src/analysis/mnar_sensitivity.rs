//! Missing-not-at-random (MNAR) sensitivity of a two-arm mean contrast.
//!
//! The facade routes and owns no scientific rule of its own:
//!
//! * [`mnar_sensitivity`] re-runs the declared arm-mean contrast across a declared bounded
//!   grid of a delta-adjusted pattern-mixture shift or an exponential-tilting selection model
//!   of the outcome-missingness mechanism, with tipping points of the sign and of every declared decision;
//! * [`mnar_sensitivity_for_table`] reads the treatment, outcome and covariate columns of a
//!   table (an invalid outcome cell is a missing outcome) and runs the same analysis;
//! * [`mnar_sensitivity_for_study`] starts from a built [`Study`]: it binds the study's complete
//!   average-effect target (intervention levels, target population, outcome functional and
//!   effect modifiers) and checks the adjustment against the study's graph, refusing every
//!   coordinate the arm-mean engine does not compute. This is the route the refusal of a
//!   `missing_not_at_random` multiple-imputation declaration names as the next attempt.
//!
//! The adjusted estimators (`regression_adjusted`, `ipw`, `aipw`) impute each missing outcome
//! from the arm's outcome regression plus the departure (the shift, or the mean residual under
//! tilted weights) and re-run the estimator on the completed table at every grid point.
//!
//! Every effect over the range is an assumption-dependent range, never a confidence interval
//! and never empirically established: the departure from missing at random is recorded as an
//! assumption with status `untestable`. Sampling intervals and their derived tipping conclusions
//! are reported with their calibration status and are never merged with the range.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Intervention, OutcomeFunctional, TargetPopulation, VariableId,
    reason_code,
};
use antecedent_data::TableView;
use antecedent_identify::backdoor::check_joint_adjustment;
pub use antecedent_validate::mnar_sensitivity::{
    ArmLoading, MNAR_ASSIGNMENT_ASSUMPTION_ID, MNAR_ASSUMPTION_RANGE_CLAIM,
    MNAR_DEPARTURE_ASSUMPTION_ID, MNAR_DEPARTURE_ASSUMPTION_STATUS, MNAR_EMPIRICAL_STATUS,
    MNAR_FAMILY, MNAR_INTERPRETATION, MNAR_MAX_BOOTSTRAP, MNAR_MAX_COVARIATES, MNAR_MAX_DECISIONS,
    MNAR_MAX_DELTA, MNAR_MAX_GRID_POINTS, MNAR_MAX_ROWS, MNAR_MAX_TILT, MNAR_MAX_TOLERANCE,
    MNAR_METHOD, MNAR_MIN_BOOTSTRAP, MNAR_MIN_TOLERANCE, MNAR_SAMPLING_BASIS, MNAR_TARGET,
    MnarAssumptionRange, MnarAssumptionRecord, MnarConclusionKind, MnarData, MnarDecision,
    MnarDecisionReport, MnarEstimator, MnarMechanism, MnarModel, MnarPoint, MnarSamplingInterval,
    MnarSensitivityError, MnarSensitivityResult, MnarSensitivitySpec, MnarTipping, MnarTippingSide,
};

use super::builder::DataInput;
use super::execute::Study;

/// The declared estimand re-run across the declared bounded grid of an MNAR departure.
///
/// # Errors
///
/// The typed [`MnarSensitivityError`] of the contract (`mechanism_not_declared`,
/// `unsupported_estimator`, `unbounded_range`, `range_excludes_mar`, `composition_not_supported`,
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
        detail: antecedent_core::reason_details::mnar_sensitivity::INVALID_DATA,
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

fn target_refusal(coordinates: &[&str]) -> MnarSensitivityError {
    MnarSensitivityError {
        code: reason_code!("route_not_supported"),
        detail: antecedent_core::reason_details::mnar_sensitivity::UNSUPPORTED_TARGET,
        message: format!(
            "the sensitivity analysis computes the full-population difference of arm means \
             between treatment levels 1 and 0; the study's {} differ{} from that target",
            coordinates.join(", "),
            if coordinates.len() == 1 { "s" } else { "" }
        ),
    }
}

/// The coordinates of the study's average-effect target that the arm-mean engine does not
/// compute: it contrasts the arms coded 1 and 0 over all observed rows, as a mean, with no
/// effect modifiers.
fn unsupported_coordinates(query: &AverageEffectQuery) -> Vec<&'static str> {
    let level = |intervention: &Intervention, expected: f64| {
        matches!(
            intervention,
            Intervention::Set { variable, value, .. }
                if *variable == query.treatment && value.as_f64() == Some(expected)
        )
    };
    let mut refused = Vec::new();
    if !level(&query.control, 0.0) || !level(&query.active, 1.0) {
        refused.push("intervention levels");
    }
    if !matches!(query.target_population, TargetPopulation::AllObserved) {
        refused.push("target population");
    }
    if !matches!(query.outcome_functional, OutcomeFunctional::Mean) {
        refused.push("outcome functional");
    }
    if !query.effect_modifiers.is_empty() {
        refused.push("effect modifiers");
    }
    refused
}

/// The sensitivity of a built study's average-effect target on the study's own tabular data.
///
/// The study's complete target is bound before anything runs. The engine contrasts the arm
/// means of a treatment coded 0 and 1 over all observed rows, so the study's control and
/// active interventions must be `do(treatment := 0)` and `do(treatment := 1)`, its target
/// population all observed units, its outcome functional the mean and its effect modifiers
/// empty. Any other coordinate is refused with `mnar_sensitivity.unsupported_target`, which
/// names the refused coordinates; none is silently replaced by the full-population mean
/// contrast.
///
/// `adjustment` (the covariates of an adjusted estimator; empty for the difference in means)
/// is checked against the study's graph with the joint sufficient backdoor criterion: a set
/// that contains a descendant of the treatment, or leaves a backdoor path open, is refused
/// with `mnar_sensitivity.adjustment_not_valid`. A study declared on a graph posterior or on
/// a graph that is not a DAG has no single graph to check against and is refused with
/// `mnar_sensitivity.graph_not_supported`. A study whose query is not an average effect, or
/// whose data are not a table, is refused with `mnar_sensitivity.unsupported_query`.
///
/// # Errors
///
/// `mnar_sensitivity.unsupported_query`, `mnar_sensitivity.unsupported_target`,
/// `mnar_sensitivity.graph_not_supported`, `mnar_sensitivity.adjustment_not_valid`,
/// otherwise the refusals of [`mnar_sensitivity`].
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
            detail: antecedent_core::reason_details::mnar_sensitivity::UNSUPPORTED_QUERY,
            message: "the sensitivity analysis re-runs an average-effect query on tabular data"
                .to_owned(),
        });
    };
    let refused = unsupported_coordinates(query);
    if !refused.is_empty() {
        return Err(target_refusal(&refused));
    }
    let (None, Some(dag)) = (&study.graph_posterior, study.graph.as_dag()) else {
        return Err(MnarSensitivityError {
            code: reason_code!("route_not_supported"),
            detail: antecedent_core::reason_details::mnar_sensitivity::GRAPH_NOT_SUPPORTED,
            message: "the adjustment is checked against a single DAG; the study declares a \
                      graph posterior or a graph that is not a DAG"
                .to_owned(),
        });
    };
    let not_valid = |message: String| MnarSensitivityError {
        code: reason_code!("invalid_argument"),
        detail: antecedent_core::reason_details::mnar_sensitivity::ADJUSTMENT_NOT_VALID,
        message,
    };
    match check_joint_adjustment(dag, &[query.treatment], query.outcome, adjustment) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return Err(not_valid(
                "the adjustment set is not a valid backdoor adjustment in the study's graph: it \
                 contains a descendant of the treatment or leaves a backdoor path open"
                    .to_owned(),
            ));
        }
        Err(error) => return Err(not_valid(format!("the adjustment set is invalid: {error}"))),
    }
    run_ids(table, query.treatment, query.outcome, adjustment, spec)
}
