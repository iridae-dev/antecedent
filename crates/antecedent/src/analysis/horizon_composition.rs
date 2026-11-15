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
//! The composed draws are a labelled spread (`uncalibrated_spread`). A linear composition also
//! has a 95% interval ([`horizon_composition_interval`]): the composed point value with the
//! spread of the composed replicates under the dispersion of the composition's own influence,
//! read against the fixed-b normal reference (bootstrap replicates) or the batch-means Student-t
//! reference (block-weight draws). A nonlinear composition has the spread only.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ComposedCoordinate, ExternalRefusal, JOINT_LAW_STAGE, PropagationOperation, PropagationPlan,
    RequestedReading, UNCALIBRATED_SPREAD_LABEL, check_horizon_composition,
};
use antecedent_core::{CausalResponse, ExecutionContext, ResponseFunctional};
use antecedent_estimate::{
    JOINT_HORIZON_DISPERSION, JOINT_HORIZON_INFLUENCE, JOINT_HORIZON_LAYOUT,
    JOINT_HORIZON_REPLICATES,
};
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

/// Interval construction of a linear horizon composition over joint circular-block bootstrap
/// replicates: the fixed-b normal interval at the dispersion of the composed influence.
pub const HORIZON_BLOCK_BOOTSTRAP_INTERVAL: &str = "horizon_block_bootstrap_fixed_b";

/// Interval construction of a linear horizon composition over joint block-weight draws: the
/// batch-means Student-t interval at the dispersion of the composed influence.
pub const HORIZON_BLOCK_WEIGHT_INTERVAL: &str = "horizon_block_weight_student_t";

/// Level of a horizon composition interval. The fixed-b ratio carried by the replicates is the
/// 95% one, so no other level has a construction.
pub const HORIZON_INTERVAL_LEVEL: f64 = 0.95;

/// The 95% interval of a linear composition across horizons.
#[derive(Clone, Debug, PartialEq)]
pub struct HorizonCompositionInterval {
    /// The checked composition and its composed draws.
    pub composed: ComposedHorizons,
    /// The composed point value `Σ coefficients[k] · horizon k`.
    pub center: f64,
    /// The standard error the interval is built on.
    pub standard_error: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Nominal level ([`HORIZON_INTERVAL_LEVEL`]).
    pub level: f64,
    /// [`HORIZON_BLOCK_BOOTSTRAP_INTERVAL`] or [`HORIZON_BLOCK_WEIGHT_INTERVAL`].
    pub construction: &'static str,
    /// Kernel-bias factor of the composed influence at the block length.
    pub kernel_factor: f64,
    /// Lag-aligned rows every horizon fits (the resampled window).
    pub rows: u64,
    /// Joint realizations composed.
    pub replicates: usize,
    /// Effective number of weighted blocks behind the Student-t reference; `None` for
    /// bootstrap replicates.
    pub effective_blocks: Option<f64>,
}

/// The 95% interval of the linear combination `Σ coefficients[k] · horizon k` of the cells of
/// dose number `dose_index` of `response`.
///
/// The published replicates scale each cell's deviation by that cell's kernel-bias factor. A
/// combination of cells has its own influence (the combination of the cells' influences), and a
/// contrast of two serially dependent levels is often close to white: the per-cell factors are
/// removed and the factor of the composed influence applied. Bootstrap replicates already carry
/// the fixed-b ratio of the circular-block variance, so `center ± z·se` is the fixed-b interval.
/// Block-weight draws weight `B` consecutive blocks by Dirichlet(1, …, 1): their variance is
/// `(B − 1)/(B + 1)` of the batch-means variance of the estimate, whose studentized error is
/// Student-t on `B − 1` degrees of freedom, so the fixed-b ratio is replaced by
/// `t_{B−1} / z · sqrt((B + 1)/(B − 1))` with `B` the effective block count.
///
/// # Errors
///
/// What [`horizon_draws`] and [`compose_horizons`] refuse, a nonlinear operation, a response
/// that carries no dispersion or influence readings (a series with missing time points), and
/// fewer than three effective blocks under block-weight draws.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
#[allow(clippy::too_many_lines, reason = "the route's steps read in order in one place")]
pub fn horizon_composition_interval(
    response: &CausalResponse,
    dose_index: usize,
    coefficients: &[f64],
    operation: PropagationOperation,
) -> Result<HorizonCompositionInterval, ExternalRefusal> {
    if operation.class() == antecedent_contract::PropagationClass::Nonlinear {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "joint_law_propagation.horizon_composition",
            operation.as_str(),
            "a nonlinear composition of horizons has a spread and no interval; ask for the \
             interval of a linear combination",
        ));
    }
    let draws = horizon_draws(response, dose_index)?;
    let composed = compose_horizons(&draws, coefficients, operation)?;
    let unreadable = || {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_draws_layout",
            "dispersion",
            "the response carries no dispersion and influence readings matching its joint \
             replicates (a series with missing time points publishes none): the composed spread \
             is the result",
        )
    };
    let (Some(layout), Some(dispersion), Some(influence)) = (
        diagnostic(response, JOINT_HORIZON_LAYOUT),
        diagnostic(response, JOINT_HORIZON_DISPERSION),
        diagnostic(response, JOINT_HORIZON_INFLUENCE),
    ) else {
        return Err(unreadable());
    };
    let n_horizons = draws.horizons.len();
    let cells = layout.get(1).map_or(0, |count| recorded_count(*count));
    let first = dose_index * n_horizons;
    let [block, rows, _, fixed_b, weighted, effective_blocks, kernel @ ..] = dispersion else {
        return Err(unreadable());
    };
    let (block, rows) = (recorded_count(*block), recorded_count(*rows));
    let cell_factors = kernel.get(first..first + n_horizons).unwrap_or(&[]);
    if kernel.len() != cells
        || cell_factors.len() != n_horizons
        || rows < 2
        || rows.checked_mul(cells) != Some(influence.len())
        || cell_factors.iter().any(|factor| !(factor.is_finite() && *factor > 0.0))
    {
        return Err(unreadable());
    }
    let composed_influence: Vec<f64> = (0..rows)
        .map(|t| {
            coefficients
                .iter()
                .enumerate()
                .map(|(k, coefficient)| coefficient * influence[(first + k) * rows + t])
                .sum::<f64>()
        })
        .collect();
    let kernel_factor = antecedent_estimate::kernel_bias_factor(&composed_influence, block);
    let replicates = draws.columns.first().map_or(0, Vec::len);
    let deviations: Vec<f64> = (0..replicates)
        .map(|r| {
            coefficients
                .iter()
                .enumerate()
                .map(|(k, coefficient)| {
                    coefficient * (draws.columns[k][r] - draws.center[k]) / cell_factors[k]
                })
                .sum::<f64>()
        })
        .collect();
    let spread = draw_spread(&deviations, None).map_err(|error| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_law_propagation.horizon_composition",
            error.to_string(),
            "compose at least two joint realizations",
        )
    })?;
    let z = antecedent_stats::normal_ppf(0.5 + HORIZON_INTERVAL_LEVEL / 2.0);
    let block_weights = *weighted > 0.5;
    let reference = if block_weights {
        let blocks = *effective_blocks;
        if !(blocks.is_finite() && blocks >= 3.0 && fixed_b.is_finite() && *fixed_b > 0.0) {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "joint_law_propagation.horizon_composition",
                "blocks",
                "block-weight draws over fewer than three effective blocks carry no interval; \
                 use a longer series",
            ));
        }
        let quantile =
            antecedent_stats::student_t_ppf(0.5 + HORIZON_INTERVAL_LEVEL / 2.0, blocks - 1.0);
        quantile / z * ((blocks + 1.0) / (blocks - 1.0)).sqrt() / fixed_b
    } else {
        1.0
    };
    let standard_error = spread.standard_deviation * kernel_factor * reference;
    let center: f64 = coefficients
        .iter()
        .zip(&draws.center)
        .map(|(coefficient, value)| coefficient * value)
        .sum();
    if !(standard_error.is_finite() && center.is_finite()) {
        return Err(unreadable());
    }
    Ok(HorizonCompositionInterval {
        composed,
        center,
        standard_error,
        lower: center - z * standard_error,
        upper: center + z * standard_error,
        level: HORIZON_INTERVAL_LEVEL,
        construction: if block_weights {
            HORIZON_BLOCK_WEIGHT_INTERVAL
        } else {
            HORIZON_BLOCK_BOOTSTRAP_INTERVAL
        },
        kernel_factor,
        rows: u64::try_from(rows).unwrap_or(u64::MAX),
        replicates,
        effective_blocks: block_weights.then_some(*effective_blocks),
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
