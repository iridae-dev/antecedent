//! Composition across the horizons of one temporal claim.
//!
//! A quantity built from several horizons of a temporal response (a cumulative effect, a
//! contrast between horizons, a peak) has the uncertainty of the realizations that refit all
//! its horizons together, not of the horizons taken one by one. The estimator publishes those
//! joint realizations on request (`TemporalResponseEstimator::with_joint_replicates`): each row
//! is one resample or one block-weight draw under which every horizon was refit, so the rows are
//! aligned joint draws and the composed quantity's spread is read row by row.
//!
//! [`horizon_draws`] reads them from a response; [`compose_horizons`] checks the composition at
//! its horizon-indexed coordinate and forms the composed draws. A response whose horizons were
//! fit one by one (the Bayesian conjugate curve) retains no joint realizations and is refused:
//! its per-horizon draws are independent marginals, and pairing them by draw index would assert
//! a dependence the fit does not carry.
//!
//! The result is a labelled spread (`uncalibrated_spread`); an interval at the composed
//! coordinate stays closed until a coverage record is measured at it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ComposedCoordinate, ExternalRefusal, JOINT_LAW_STAGE, PropagationOperation, PropagationPlan,
    RequestedReading, UNCALIBRATED_SPREAD_LABEL, check_horizon_composition,
};
use antecedent_core::{CausalResponse, ExecutionContext, ResponseFunctional};
use antecedent_estimate::{JOINT_HORIZON_LAYOUT, JOINT_HORIZON_REPLICATES};
use antecedent_stats::{DrawSpread, aligned_linear_draws, aligned_map_draws, draw_spread};

/// The draws of one dose's cells across the horizons of a temporal response, one column per
/// horizon, row `r` of every column belonging to the same joint realization.
#[derive(Clone, Debug, PartialEq)]
pub struct HorizonDraws {
    /// The horizons, in the order of the columns.
    pub horizons: Vec<u32>,
    /// The point value of each horizon's cell.
    pub center: Vec<f64>,
    /// `columns[k][r]` is replicate `r` of horizon `horizons[k]`.
    pub columns: Vec<Vec<f64>>,
    /// The identity of the joint law the rows realize: the response's provenance, the number of
    /// replicates and a digest of the values.
    pub alignment_id: String,
    /// The evidence the draws rest on (the response's provenance).
    pub evidence_id: String,
}

/// A composed quantity across horizons.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedHorizons {
    /// The checked plan.
    pub plan: PropagationPlan,
    /// The composed coordinate, carrying the horizon axis.
    pub coordinate: ComposedCoordinate,
    /// The composed value at each joint realization.
    pub draws: Vec<f64>,
    /// The spread of the composed draws.
    pub spread: DrawSpread,
    /// The label of the result: a spread with no coverage statement.
    pub label: &'static str,
}

fn refuse(
    code: &'static str,
    detail: &str,
    offending: impl Into<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, JOINT_LAW_STAGE, detail)
        .with_offending(offending)
        .with_remedy(remedy)
}

fn diagnostic<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
    response.support.diagnostics.iter().find(|d| d.id.as_ref() == id).map(|d| d.values.as_ref())
}

/// A count stored as a diagnostic value.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the layout diagnostic stores non-negative whole counts as floats"
)]
fn recorded_count(value: f64) -> usize {
    value as usize
}

/// The joint draws of the cells of dose number `dose_index` across the horizons of `response`.
///
/// # Errors
///
/// A response that retains no joint realizations (its horizons were fit one by one, or the
/// estimator was not asked to retain them), a response that is not a temporal mean curve, or a
/// dose outside the curve's grid.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn horizon_draws(
    response: &CausalResponse,
    dose_index: usize,
) -> Result<HorizonDraws, ExternalRefusal> {
    let (Some(layout), Some(values)) = (
        diagnostic(response, JOINT_HORIZON_LAYOUT),
        diagnostic(response, JOINT_HORIZON_REPLICATES),
    ) else {
        return Err(refuse(
            antecedent_core::reason_code!("joint_law_required"),
            "joint_law_propagation.horizon_draws_not_joint",
            "response",
            "the response retains no draws coupled across its horizons: horizons fit one by \\
             one are independent marginals and are never paired by draw index; run the \\
             temporal response with joint replicates retained (a block bootstrap or a block \\
             weight posterior over the whole surface)",
        ));
    };
    let (ResponseFunctional::MeanCurve { .. } | ResponseFunctional::InterventionResponse { .. }) =
        &response.estimand
    else {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "joint_law_propagation.horizon_draws_mean_curve",
            "estimand",
            "compose the horizons of a temporal mean curve or intervention response",
        ));
    };
    let Some(horizons) = response.horizon_identification.as_ref() else {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "joint_law_propagation.horizon_draws_mean_curve",
            "horizon_identification",
            "compose the horizons of a temporal response",
        ));
    };
    let (replicates, cells) = match layout {
        [replicates, cells, ..] => (recorded_count(*replicates), recorded_count(*cells)),
        _ => (0, 0),
    };
    let n_horizons = horizons.len();
    if n_horizons == 0
        || cells == 0
        || cells % n_horizons != 0
        || replicates.checked_mul(cells) != Some(values.len())
    {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_draws_layout",
            "layout",
            "the retained replicates do not match the response's horizons",
        ));
    }
    let doses = cells / n_horizons;
    if dose_index >= doses {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_draws_dose",
            format!("dose[{dose_index}]"),
            "name a dose inside the curve's grid",
        ));
    }
    let center = center_of(response)?;
    if center.len() < cells {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_draws_layout",
            "estimate",
            "the point surface does not match the retained replicate cells",
        ));
    }
    let first = dose_index * n_horizons;
    let columns: Vec<Vec<f64>> = (0..n_horizons)
        .map(|k| (0..replicates).map(|r| values[r * cells + first + k]).collect())
        .collect();
    let digest = values.iter().fold(0xCBF2_9CE4_8422_2325_u64, |acc, value| {
        (acc ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01B3)
    });
    Ok(HorizonDraws {
        horizons: horizons.iter().map(|h| h.horizon).collect(),
        center: center[first..first + n_horizons].to_vec(),
        columns,
        alignment_id: format!("{}#joint:{replicates}:{digest:016x}", response.provenance_id),
        evidence_id: response.provenance_id.to_string(),
    })
}

/// Compose the horizons of `draws` as the linear combination `Σ coefficients[k] · horizon k`
/// (a cumulative effect, a contrast between horizons), checked at its horizon-indexed
/// coordinate.
///
/// # Errors
///
/// A coefficient count that is not the horizon count, a composition the propagation rule
/// refuses, or non-finite draws.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn compose_horizons(
    draws: &HorizonDraws,
    coefficients: &[f64],
    operation: PropagationOperation,
) -> Result<ComposedHorizons, ExternalRefusal> {
    if coefficients.len() != draws.horizons.len() {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            "coefficients",
            "give one coefficient per horizon of the draws",
        ));
    }
    let (plan, coordinate) = check_horizon_composition(
        operation,
        &draws.horizons,
        &draws.evidence_id,
        &draws.alignment_id,
        RequestedReading::Spread,
    )?;
    let columns: Vec<&[f64]> = draws.columns.iter().map(Vec::as_slice).collect();
    let composed = aligned_linear_draws(coefficients, &columns).map_err(|error| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            error.to_string(),
            "compose finite draws of equal length",
        )
    })?;
    let spread = draw_spread(&composed, None).map_err(|error| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            error.to_string(),
            "compose at least two joint realizations",
        )
    })?;
    Ok(ComposedHorizons {
        plan,
        coordinate,
        draws: composed,
        spread,
        label: UNCALIBRATED_SPREAD_LABEL,
    })
}

/// Compose the horizons of `draws` by an arbitrary map of one realization's horizon values (a
/// peak, a time to threshold, a ratio of horizons), checked at its horizon-indexed coordinate.
///
/// `map` receives one realization's values in the order of [`HorizonDraws::horizons`]. The
/// operation must be a nonlinear one (a product, a nonlinear functional or a chain of
/// transformations); a linear combination goes through [`compose_horizons`].
///
/// # Errors
///
/// A linear operation, a composition the propagation rule refuses, non-finite draws or composed
/// values, or cancellation.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn compose_horizons_with(
    draws: &HorizonDraws,
    operation: PropagationOperation,
    map: &dyn Fn(&[f64]) -> f64,
    ctx: &ExecutionContext,
) -> Result<ComposedHorizons, ExternalRefusal> {
    if operation.class() != antecedent_contract::PropagationClass::Nonlinear {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "joint_law_propagation.horizon_composition",
            operation.as_str(),
            "compose a linear combination of horizons with compose_horizons; this call takes a \
             product, a nonlinear functional or a chain",
        ));
    }
    let (plan, coordinate) = check_horizon_composition(
        operation,
        &draws.horizons,
        &draws.evidence_id,
        &draws.alignment_id,
        RequestedReading::Spread,
    )?;
    let columns: Vec<&[f64]> = draws.columns.iter().map(Vec::as_slice).collect();
    let composed = aligned_map_draws(&columns, map, &ctx.cancellation).map_err(|error| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            error.to_string(),
            "compose finite draws of equal length with a finite map",
        )
    })?;
    let spread = draw_spread(&composed, None).map_err(|error| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            error.to_string(),
            "compose at least two joint realizations",
        )
    })?;
    Ok(ComposedHorizons {
        plan,
        coordinate,
        draws: composed,
        spread,
        label: UNCALIBRATED_SPREAD_LABEL,
    })
}

/// The point surface of a response.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn center_of(response: &CausalResponse) -> Result<Vec<f64>, ExternalRefusal> {
    use antecedent_core::{ResponseIdentification, ResponseValue};
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            Ok(mean.to_vec())
        }
        _ => Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "joint_law_propagation.horizon_draws_mean_curve",
            "estimate",
            "compose the horizons of a point-identified temporal surface",
        )),
    }
}
