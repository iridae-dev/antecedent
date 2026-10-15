//! Assumption range for a violated transport premise, from a transported response.
//!
//! A transported response (a value carried to a target population, with the
//! receipt that says how) is only as good as the premise the transport rests on.
//! This module builds the [`TransportSensitivitySpec`] from the response's own
//! point value, coordinate and receipt, runs the range through
//! `antecedent_design::transport_sensitivity`, and hands the sealed
//! [`SensitivityArtifact`] to the one sensitivity surface
//! ([`super::sensitivity_decision`]: `decide`, `export`, `consume`). No second
//! sensitivity artifact or decision path exists.
//!
//! What the range is: an assumption range at the declared grid points only. The
//! bound is the caller's declaration of how far the premise may fail; nothing in
//! the data licenses it. It is never a confidence interval, a credible interval
//! or a posterior, no probability attaches to a shift, and a sampling interval is
//! withheld with its registered reason.
//!
//! # Lifecycle
//!
//! * **Refresh.** Outcome-only data cannot move an assumption range: the range is
//!   a function of the point value, the declared family and bound, and the
//!   transport identity. New data produces a new transported response; calling
//!   [`transport_sensitivity_artifact`] again **keeps** the declared family,
//!   bound and grid and **recomputes** the baseline and every grid number. The
//!   range is bound to the response's target population and quantity through the
//!   spec identity, so a range is never inherited by another target: the same
//!   family applied to a response about another population is a different range
//!   with a different identity.
//! * **Refuse.** A law-valued response (the point value is not one number), a
//!   structural premise (no scalar family), and a route that the response does
//!   not name and the caller did not declare.
//! * **Export / consume.** Through [`super::sensitivity_decision::export`] and
//!   [`super::sensitivity_decision::consume`]; consumption recomputes identity
//!   digests and the decision outcome and refuses a resealed mutation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::design_exports::design_crate::transport_sensitivity::{
    REFERENCE_ACTION_ID, TRANSPORT_ASSUMPTION_RANGE_INTERPRETATION, TRANSPORT_SENSITIVITY_VERSION,
    TRANSPORTED_ACTION_ID, TransportAssumptionRange, TransportPremise, TransportRoute,
    TransportSensitivityRefusal, TransportSensitivitySpec, ViolationFamily,
    transport_assumption_range,
};
use antecedent_contract::ScientificQuantity;
use antecedent_design::decision_transported_source::{
    TransportedResponse, TransportedRoute, TransportedValue,
};

use super::sensitivity_decision::SensitivityArtifact;

/// Detail: the response is a law, so it has no single point value to shift.
pub const DETAIL_LAW_RESPONSE: &str = "transport_sensitivity.law_response_not_scalar";
/// Detail: the response's route does not name a sensitivity route and none was declared.
pub const DETAIL_ROUTE_UNDECLARED: &str = "transport_sensitivity.route_undeclared";
/// Detail: the coordinate index is outside the response.
pub const DETAIL_COORDINATE_RANGE: &str = "transport_sensitivity.coordinate_out_of_range";

/// Why a transported response was not given an assumption range.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TransportSensitivityEntryRefusal {
    /// The response is a transported law.
    LawResponse,
    /// The response's route is `declared` and no sensitivity route was named.
    RouteUndeclared,
    /// The coordinate index does not address a coordinate of the response.
    CoordinateOutOfRange {
        /// The index asked for.
        index: usize,
        /// Coordinates the response has.
        available: usize,
    },
    /// The range itself was refused.
    Range(TransportSensitivityRefusal),
}

impl TransportSensitivityEntryRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::LawResponse | Self::RouteUndeclared => {
                antecedent_core::reason_code!("route_not_supported")
            }
            Self::CoordinateOutOfRange { .. } => antecedent_core::reason_code!("invalid_argument"),
            Self::Range(inner) => inner.reason_code(),
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::LawResponse => DETAIL_LAW_RESPONSE,
            Self::RouteUndeclared => DETAIL_ROUTE_UNDECLARED,
            Self::CoordinateOutOfRange { .. } => DETAIL_COORDINATE_RANGE,
            Self::Range(inner) => inner.detail(),
        }
    }

    /// What the caller can do about it.
    #[must_use]
    pub const fn remedy(&self) -> &'static str {
        match self {
            Self::LawResponse => {
                "shift the transported mean: build a transported point from the law's mean, or \
                 state the sensitivity on the functional the decision reads"
            }
            Self::RouteUndeclared => {
                "name the route that produced the value (TransportRoute); a declared transport \
                 carries no route of its own"
            }
            Self::CoordinateOutOfRange { .. } => "address one of the response's coordinates",
            Self::Range(inner) => inner.remedy(),
        }
    }
}

impl std::fmt::Display for TransportSensitivityEntryRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Range(inner) => write!(f, "{inner}"),
            Self::CoordinateOutOfRange { index, available } => {
                write!(f, "{}: coordinate {index} of {available}", self.detail())
            }
            _ => write!(f, "{}", self.detail()),
        }
    }
}

impl std::error::Error for TransportSensitivityEntryRefusal {}

impl From<TransportSensitivityRefusal> for TransportSensitivityEntryRefusal {
    fn from(value: TransportSensitivityRefusal) -> Self {
        Self::Range(value)
    }
}

/// The sensitivity route a transported response's own route implies, when it
/// implies one. A `declared` transport names none.
#[must_use]
pub const fn route_of(route: TransportedRoute) -> Option<TransportRoute> {
    TransportRoute::of_response_route(route)
}

/// The request for the assumption range of one coordinate of a response.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportSensitivityRequest {
    /// Index of the coordinate of the response whose point value is shifted.
    pub coordinate: usize,
    /// The route's sensitivity route; `None` takes it from the response.
    pub route: Option<TransportRoute>,
    /// The violation family and its declared bound.
    pub family: ViolationFamily,
    /// Grid points on `[0, bound]`, at least two.
    pub grid_points: usize,
}

/// Build the spec of a response's point value, quantity and receipt.
///
/// The source population is the receipt's source populations (sorted, joined by
/// `,`), the query binding is the receipt's source identity, and the snapshot and
/// causal-contract identities are those the value was computed under.
///
/// # Errors
/// [`TransportSensitivityEntryRefusal::LawResponse`], a route that is neither
/// named nor implied, and a coordinate outside the response.
pub fn transport_sensitivity_spec(
    response: &TransportedResponse,
    request: &TransportSensitivityRequest,
) -> Result<TransportSensitivitySpec, TransportSensitivityEntryRefusal> {
    let TransportedValue::Means(mean) = response.value() else {
        return Err(TransportSensitivityEntryRefusal::LawResponse);
    };
    let route = request
        .route
        .or_else(|| route_of(response.receipt().route))
        .ok_or(TransportSensitivityEntryRefusal::RouteUndeclared)?;
    let (Some(quantity), Some(point)) =
        (mean.coordinates.get(request.coordinate), mean.means.get(request.coordinate))
    else {
        return Err(TransportSensitivityEntryRefusal::CoordinateOutOfRange {
            index: request.coordinate,
            available: mean.coordinates.len(),
        });
    };
    let receipt = response.receipt();
    let mut sources = receipt.source_populations.clone();
    sources.sort_unstable();
    sources.dedup();
    Ok(TransportSensitivitySpec {
        route,
        quantity: quantity.clone(),
        point_value: *point,
        family: request.family,
        grid_points: request.grid_points,
        source_population: sources.join(","),
        target_population: receipt.target_population.clone(),
        query_binding: receipt.source_identity.clone(),
        provider_snapshot: mean.snapshot_id.clone(),
        causal_contract_id: mean.causal_contract_id.clone(),
    })
}

/// The assumption range of a response's point value.
///
/// # Errors
/// As [`transport_sensitivity_spec`] and [`transport_assumption_range`].
pub fn transport_range(
    response: &TransportedResponse,
    request: &TransportSensitivityRequest,
) -> Result<TransportAssumptionRange, TransportSensitivityEntryRefusal> {
    let spec = transport_sensitivity_spec(response, request)?;
    Ok(transport_assumption_range(&spec)?)
}

/// The sealed [`SensitivityArtifact`] of a decision between acting on the
/// transported quantity and keeping `reference_value` of `reference` (same
/// population and units). Decide, export and consume it through
/// [`super::sensitivity_decision`].
///
/// # Errors
/// As [`transport_range`], a non-finite reference and any refusal of the
/// sensitivity surface (a reference in another population or units).
pub fn transport_sensitivity_artifact(
    response: &TransportedResponse,
    request: &TransportSensitivityRequest,
    reference: &ScientificQuantity,
    reference_value: f64,
) -> Result<SensitivityArtifact, TransportSensitivityEntryRefusal> {
    let range = transport_range(response, request)?;
    Ok(range.to_artifact(reference, reference_value)?)
}
