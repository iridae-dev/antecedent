//! Coordinates, horizons and interval admission for composed laws.

use super::{
    ExternalRefusal, InputAlignment, InputRole, MEASURED_INTERVAL_LABEL, PropagationDependence,
    PropagationInput, PropagationOperation, PropagationPlan, PropagationRequest, PropagationRoute,
    RequestedReading, blank, check_propagation, invalid, refuse,
};

/// The coordinate at which a composed interval's coverage is a property: the
/// operation, the route, whether any input is transported and the sorted
/// multiset of input roles. A coverage record is about one coordinate and says
/// nothing about another; a component interval's record is never a composed
/// coordinate's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposedCoordinate {
    /// The operation.
    pub operation: PropagationOperation,
    /// The supported route.
    pub route: PropagationRoute,
    /// Sorted role names of the inputs (one entry per input).
    pub roles: Vec<&'static str>,
    /// Sorted distinct horizons of the one temporal claim whose horizons the inputs are; empty
    /// when the inputs are not horizons of a single claim. A record measured over one set of
    /// horizons says nothing about another.
    pub horizons: Vec<u32>,
}

impl ComposedCoordinate {
    /// The coordinate of a checked composition.
    #[must_use]
    pub fn of(request: &PropagationRequest, plan: &PropagationPlan) -> Self {
        let mut roles: Vec<&'static str> =
            request.inputs.iter().map(|input| input.role.as_str()).collect();
        roles.sort_unstable();
        Self { operation: plan.operation, route: plan.route, roles, horizons: Vec::new() }
    }

    /// The coordinate over the horizons of one temporal claim.
    #[must_use]
    pub fn over_horizons(mut self, horizons: &[u32]) -> Self {
        let mut horizons = horizons.to_vec();
        horizons.sort_unstable();
        horizons.dedup();
        self.horizons = horizons;
        self
    }

    /// Stable identity string: `operation|route|role,role,...`, followed by
    /// `|horizons=h,h,...` when the inputs are horizons of one temporal claim. A coordinate
    /// without horizons keeps the identity it always had.
    #[must_use]
    pub fn id(&self) -> String {
        let base =
            format!("{}|{}|{}", self.operation.as_str(), self.route.as_str(), self.roles.join(","));
        if self.horizons.is_empty() {
            return base;
        }
        let horizons: Vec<String> = self.horizons.iter().map(u32::to_string).collect();
        format!("{base}|horizons={}", horizons.join(","))
    }
}

/// The inputs of a composition across the horizons of one temporal claim: one per horizon,
/// all resting on the same evidence and all rows of the one joint law `alignment_id` names (a
/// replicate or a draw index that refit every horizon together).
///
/// Horizons fit one by one are independent marginals, not these: their draws carry no common
/// alignment identity, and [`check_horizon_composition`] refuses them.
#[must_use]
pub fn horizon_inputs(
    horizons: &[u32],
    evidence_id: &str,
    alignment_id: &str,
) -> Vec<PropagationInput> {
    horizons
        .iter()
        .map(|horizon| PropagationInput {
            id: format!("horizon:{horizon}"),
            evidence_id: evidence_id.to_owned(),
            role: InputRole::Estimate,
            alignment: InputAlignment::Aligned { alignment_id: alignment_id.to_owned() },
        })
        .collect()
}

/// Check a composition across the horizons of one temporal claim and name its coordinate.
///
/// The inputs are the claim's horizons, coupled by the joint law `alignment_id` (see
/// [`horizon_inputs`]); the dependence is that aligned joint law. The result is the checked
/// plan and the coordinate, carrying the horizon axis.
///
/// # Errors
/// No horizons, repeated horizons, a blank evidence or alignment identity, or a request the
/// propagation rule refuses (for example an operation that needs more inputs than horizons).
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_horizon_composition(
    operation: PropagationOperation,
    horizons: &[u32],
    evidence_id: &str,
    alignment_id: &str,
    reading: RequestedReading,
) -> Result<(PropagationPlan, ComposedCoordinate), ExternalRefusal> {
    let mut sorted = horizons.to_vec();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    if sorted.is_empty() || sorted.len() != before {
        return Err(invalid("horizons", "name each horizon of the claim once, and at least one"));
    }
    let request = PropagationRequest {
        operation,
        inputs: horizon_inputs(&sorted, evidence_id, alignment_id),
        dependence: PropagationDependence::AlignedJointLaw,
        reading,
    };
    let plan = check_propagation(&request)?;
    let coordinate = ComposedCoordinate::of(&request, &plan).over_horizons(&sorted);
    Ok((plan, coordinate))
}

/// A coverage record offered for a composed coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageRecord {
    /// Identity of the record (the attested measurement).
    pub record_id: String,
    /// The [`ComposedCoordinate::id`] the record was measured at.
    pub coordinate_id: String,
    /// The interval construction the record was measured for (a percentile
    /// interval of aligned draws, a normal approximation to the propagated
    /// variance, ...). A record for one construction is not another's.
    pub construction: String,
    /// The nominal level the record was measured at.
    pub nominal_level: f64,
    /// Whether the record is a completed, attested measurement (as opposed to
    /// a wired but `not_measured` placeholder).
    pub measured: bool,
}

/// What a passed composed-interval check records.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedIntervalPlan {
    /// The propagation plan of the composition (with its dependence route).
    pub propagation: PropagationPlan,
    /// The coordinate the coverage record was matched against.
    pub coordinate: ComposedCoordinate,
    /// The coverage record that governs the interval, when one matched. Calibration
    /// annotates the interval: its absence or non-attestation never withholds it.
    pub record_id: Option<String>,
    /// The construction the interval must be built with.
    pub construction: String,
    /// The nominal level.
    pub nominal_level: f64,
    /// The label of the result.
    pub label: &'static str,
}

/// Nominal level of a composed interval when no coverage record names one.
pub const COMPOSED_DEFAULT_LEVEL: f64 = 0.95;

/// Check a composed interval: the composition's dependence route must support
/// its spread. A coverage record for exactly this composed coordinate and
/// construction annotates the interval (its identity and nominal level); without
/// one, or with a wired but `not_measured` one, the interval is still planned at
/// [`COMPOSED_DEFAULT_LEVEL`] and the caller reports its calibration slot as
/// `not_measured` or `outside_measured_scope`. A record for another coordinate
/// or construction is a mismatched request and is refused.
///
/// # Errors
/// An invalid request, an unsupported dependence route (the refusals of
/// [`check_propagation`] under the spread reading), a request whose reading is
/// not an interval, a missing, `not_measured` or mismatched coverage record, or a
/// nominal level outside `(0, 1)`.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_composed_interval(
    request: &PropagationRequest,
    record: Option<&CoverageRecord>,
    construction: &str,
) -> Result<ComposedIntervalPlan, ExternalRefusal> {
    if request.reading != RequestedReading::Interval {
        return Err(invalid(
            request.reading.as_str(),
            "ask for the interval reading, or use check_propagation for the point, spread or envelope",
        ));
    }
    if blank(construction) {
        return Err(invalid("construction", "name the interval construction the record covers"));
    }
    let spread = PropagationRequest { reading: RequestedReading::Spread, ..request.clone() };
    let propagation = check_propagation(&spread)?;
    let coordinate = ComposedCoordinate::of(request, &propagation);
    let Some(record) = record else {
        return Ok(ComposedIntervalPlan {
            propagation,
            coordinate,
            record_id: None,
            construction: construction.to_owned(),
            nominal_level: COMPOSED_DEFAULT_LEVEL,
            label: MEASURED_INTERVAL_LABEL,
        });
    };
    let code = antecedent_core::reason_code!("quantity_semantics_mismatch");
    if record.coordinate_id != coordinate.id() {
        return Err(refuse(
            code,
            "joint_law_propagation.coverage_record_coordinate_mismatch",
            record.record_id.clone(),
            "a coverage record is about one composed coordinate (operation, route and input roles); measure coverage at this coordinate, or report the labelled spread",
        )
        .with_expected(coordinate.id())
        .with_supplied(record.coordinate_id.clone()));
    }
    if record.construction != construction {
        return Err(refuse(
            code,
            "joint_law_propagation.coverage_record_construction_mismatch",
            record.record_id.clone(),
            "a coverage record is about one interval construction; build the interval with the construction the record was measured for",
        )
        .with_expected(record.construction.clone())
        .with_supplied(construction.to_owned()));
    }
    if !(record.nominal_level.is_finite()
        && 0.0 < record.nominal_level
        && record.nominal_level < 1.0)
    {
        return Err(invalid(
            "nominal_level",
            "the record's nominal level is a probability strictly between zero and one",
        ));
    }
    Ok(ComposedIntervalPlan {
        propagation,
        coordinate,
        record_id: Some(record.record_id.clone()),
        construction: construction.to_owned(),
        nominal_level: record.nominal_level,
        label: MEASURED_INTERVAL_LABEL,
    })
}
