//! Assumption ranges for a violated transport premise.
//!
//! Every transported route rests on a premise that the data cannot test: the
//! selection diagram's invariance (the mechanism behind the transported quantity does
//! not differ between source and target beyond what the diagram declares), a
//! measurement or mechanism invariance, a bridge that holds in the target, a temporal
//! invariance across periods, or a provider's attested invariance. This module states,
//! for the point value a route returns, what the value becomes if that premise fails by
//! a declared amount.
//!
//! The violation family has one parameter: the transported quantity in the target
//! differs from the point value by a shift `s` with `0 <= s <= bound`, either additive
//! (in the quantity's own units) or proportional (a fraction of the point value). At
//! shift `s` the assumption range is `[v - s, v + s]` (additive) or
//! `[v - s |v|, v + s |v|]` (proportional). The ranges are built into the existing
//! [`SensitivityArtifact`] surface, so the decision consumption, tipping and wire
//! machinery of [`crate::sensitivity_decision`] apply unchanged.
//!
//! What the statement is and is not:
//!
//! * it is an assumption range over a declared coordinate set, evaluated at the declared
//!   grid points only. It is never a confidence interval, a credible interval or a
//!   posterior, and no probability attaches to any shift;
//! * the `bound` is the caller's declaration of how far the premise may fail; nothing in
//!   the data licenses it, and the module does not choose it;
//! * a sampling interval is not composed into the range (the surface withholds one with
//!   its registered reason), and exact finite-law arithmetic supplies no calibration;
//! * a route whose premise is a structural statement (a discovered graph) has no
//!   one-parameter scalar family and is refused with a remedy rather than given an
//!   invented range.
//!
//! # Refresh
//!
//! Refresh with outcome-only data. The range is a function of the
//! transported point value and the declared violation family, never of the
//! data directly. New outcome-only data changes the baseline point value: the
//! baseline and every range endpoint are recomputed, and the grid and the
//! declared specification are kept. The range is never inherited by another
//! target or another route: a different transported value is a different
//! statement and is built again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::{CanonicalWriter, SemanticDigest};
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_io_core::sensitivity_artifact::MAX_SENSITIVITY_GRID_POINTS;

use crate::sensitivity_decision::{
    ASSUMPTION_RANGE_STATEMENT, ActionUtility, AssumptionCoordinate, AssumptionRangeStatement,
    PointSupport, SamplingStatus, SensitivityArtifact, SensitivityParts, SensitivityProvenance,
    SurfaceQuantity, UncertaintyRelationship, UtilityTerm,
};

/// Version word written into the identity of a transport assumption range.
pub const TRANSPORT_SENSITIVITY_VERSION: u64 = 1;

/// What a transport assumption range can and cannot claim.
pub const TRANSPORT_ASSUMPTION_RANGE_INTERPRETATION: &str = "assumption range for a violated transport premise at the declared shift grid points only; the shift bound is declared by the caller, the range is not a probability, not a confidence or credible interval and not a sampling interval, and nothing between grid points is evaluated";

/// Action id of the decision that acts on the transported quantity.
pub const TRANSPORTED_ACTION_ID: &str = "act_on_transported_quantity";
/// Action id of the decision that keeps the reference.
pub const REFERENCE_ACTION_ID: &str = "keep_reference";

const SOURCE_KIND: &str = "transport_premise_violation";

/// A transported route that states a premise.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TransportRoute {
    /// A transported cohort decision.
    CohortDecision,
    /// A transported effect through a statistical selection-diagram identification.
    StatisticalTransport,
    /// A transported effect through a z-transport plan.
    ZTransport,
    /// A transported path-specific effect.
    PathSpecific,
    /// A transported measurement (indicator to construct) model.
    MeasurementTransport,
    /// A transported fitted mechanism.
    MechanismTransport,
    /// A transported evidence synthesis.
    SynthesisTransport,
    /// A transported scientific bridge.
    BridgeTransport,
    /// A transported policy value.
    PolicyTransport,
    /// A policy transported across periods.
    TemporalPolicyTransport,
    /// An adaptive trial used as a transport source.
    AdaptiveTrialSource,
    /// A discovered structure transported to a target population.
    DiscoveredStructure,
    /// A provider output admitted as transport source evidence.
    ProviderHandoff,
}

/// The premise a route rests on, which the violation family perturbs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TransportPremise {
    /// The selection diagram's invariance of the transported mechanism.
    SelectionDiagramInvariance,
    /// The measurement model relates indicator to construct alike in both populations.
    MeasurementInvariance,
    /// The fitted mechanism is shared by source and target.
    MechanismInvariance,
    /// The bridge's functional form holds in the target.
    BridgeInvariance,
    /// The dynamics the policy is evaluated under do not change across periods.
    TemporalInvariance,
    /// A provider's attestation that its output is invariant to the target.
    AttestedInvariance,
    /// The discovered structure itself holds in the target.
    StructuralInvariance,
}

impl TransportPremise {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SelectionDiagramInvariance => "selection_diagram_invariance",
            Self::MeasurementInvariance => "measurement_invariance",
            Self::MechanismInvariance => "mechanism_invariance",
            Self::BridgeInvariance => "bridge_invariance",
            Self::TemporalInvariance => "temporal_invariance",
            Self::AttestedInvariance => "attested_invariance",
            Self::StructuralInvariance => "structural_invariance",
        }
    }
}

impl TransportRoute {
    /// Every route, in declaration order.
    pub const ALL: [Self; 13] = [
        Self::CohortDecision,
        Self::StatisticalTransport,
        Self::ZTransport,
        Self::PathSpecific,
        Self::MeasurementTransport,
        Self::MechanismTransport,
        Self::SynthesisTransport,
        Self::BridgeTransport,
        Self::PolicyTransport,
        Self::TemporalPolicyTransport,
        Self::AdaptiveTrialSource,
        Self::DiscoveredStructure,
        Self::ProviderHandoff,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CohortDecision => "cohort_decision",
            Self::StatisticalTransport => "statistical_transport",
            Self::ZTransport => "z_transport",
            Self::PathSpecific => "path_specific",
            Self::MeasurementTransport => "measurement_transport",
            Self::MechanismTransport => "mechanism_transport",
            Self::SynthesisTransport => "synthesis_transport",
            Self::BridgeTransport => "bridge_transport",
            Self::PolicyTransport => "policy_transport",
            Self::TemporalPolicyTransport => "temporal_policy_transport",
            Self::AdaptiveTrialSource => "adaptive_trial_source",
            Self::DiscoveredStructure => "discovered_structure",
            Self::ProviderHandoff => "provider_handoff",
        }
    }

    /// The premise the route rests on.
    #[must_use]
    pub const fn premise(self) -> TransportPremise {
        match self {
            Self::CohortDecision
            | Self::StatisticalTransport
            | Self::ZTransport
            | Self::PathSpecific
            | Self::SynthesisTransport
            | Self::PolicyTransport
            | Self::AdaptiveTrialSource => TransportPremise::SelectionDiagramInvariance,
            Self::MeasurementTransport => TransportPremise::MeasurementInvariance,
            Self::MechanismTransport => TransportPremise::MechanismInvariance,
            Self::BridgeTransport => TransportPremise::BridgeInvariance,
            Self::TemporalPolicyTransport => TransportPremise::TemporalInvariance,
            Self::ProviderHandoff => TransportPremise::AttestedInvariance,
            Self::DiscoveredStructure => TransportPremise::StructuralInvariance,
        }
    }

    /// Whether the premise has a one-parameter scalar violation family. A structural
    /// premise has none: whether an edge holds in the target is not a shift of a number.
    #[must_use]
    pub const fn has_scalar_family(self) -> bool {
        !matches!(self.premise(), TransportPremise::StructuralInvariance)
    }
}

/// The one-parameter violation family: how the transported quantity may differ from the
/// point value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ViolationFamily {
    /// The target quantity lies within `bound` of the point value, in the quantity's
    /// units.
    AdditiveShift {
        /// Largest shift; finite and positive.
        bound: f64,
    },
    /// The target quantity lies within `bound |value|` of the point value; `bound` is a
    /// fraction. Refused at a point value of zero.
    ProportionalShift {
        /// Largest relative shift; finite and positive.
        bound: f64,
    },
}

impl ViolationFamily {
    /// The declared bound.
    #[must_use]
    pub const fn bound(self) -> f64 {
        match self {
            Self::AdditiveShift { bound } | Self::ProportionalShift { bound } => bound,
        }
    }

    /// Stable `snake_case` name of the coordinate scale.
    #[must_use]
    pub const fn scale(self) -> &'static str {
        match self {
            Self::AdditiveShift { .. } => "additive_shift",
            Self::ProportionalShift { .. } => "proportional_shift",
        }
    }
}

/// A request for the assumption range of one route's point value.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportSensitivitySpec {
    /// The route that produced the point value.
    pub route: TransportRoute,
    /// The transported quantity, in the target population.
    pub quantity: ScientificQuantity,
    /// The route's point value.
    pub point_value: f64,
    /// The violation family and its declared bound.
    pub family: ViolationFamily,
    /// Number of grid points on `[0, bound]`, at least two; the endpoints are exact.
    pub grid_points: usize,
    /// Identity of the source population (or period, or study).
    pub source_population: String,
    /// Identity of the target population; must be the quantity's population.
    pub target_population: String,
    /// Identity of the checked derivation the point value came from.
    pub query_binding: String,
    /// Data or provider snapshot behind the point value.
    pub provider_snapshot: String,
    /// Independently checked causal-contract identity.
    pub causal_contract_id: String,
}

/// Why no assumption range is stated.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TransportSensitivityRefusal {
    /// The route's premise is structural; there is no scalar shift family.
    NoScalarFamily {
        /// The route.
        route: TransportRoute,
    },
    /// The point value is not finite.
    NonFinitePoint,
    /// The bound is not finite and positive.
    InvalidBound,
    /// The grid has fewer than two or more than the maximum number of points.
    InvalidGrid {
        /// Grid points requested.
        points: usize,
    },
    /// A proportional shift of a zero point value is no range.
    ProportionalAtZero,
    /// Source and target are the same population, so nothing is transported.
    NotTransported,
    /// The quantity is not stated in the target population.
    QuantityNotInTarget {
        /// The quantity's population.
        quantity_population: String,
        /// The declared target population.
        target_population: String,
    },
    /// A blank identity field.
    BlankIdentity(&'static str),
    /// The sensitivity surface refused the construction.
    Surface {
        /// Registered reason code of the surface refusal.
        code: &'static str,
        /// Its message.
        message: String,
    },
}

impl TransportSensitivityRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::NoScalarFamily { .. } => antecedent_core::reason_code!("route_not_supported"),
            Self::QuantityNotInTarget { .. } => {
                antecedent_core::reason_code!("quantity_semantics_mismatch")
            }
            Self::Surface { code, .. } => *code,
            Self::NonFinitePoint
            | Self::InvalidBound
            | Self::InvalidGrid { .. }
            | Self::ProportionalAtZero
            | Self::NotTransported
            | Self::BlankIdentity(_) => antecedent_core::reason_code!("invalid_argument"),
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::NoScalarFamily { .. } => "transport_sensitivity.no_scalar_family",
            Self::NonFinitePoint => "transport_sensitivity.non_finite_point",
            Self::InvalidBound => "transport_sensitivity.invalid_bound",
            Self::InvalidGrid { .. } => "transport_sensitivity.invalid_grid",
            Self::ProportionalAtZero => "transport_sensitivity.proportional_at_zero",
            Self::NotTransported => "transport_sensitivity.not_transported",
            Self::QuantityNotInTarget { .. } => "transport_sensitivity.quantity_not_in_target",
            Self::BlankIdentity(_) => "transport_sensitivity.blank_identity",
            Self::Surface { .. } => "transport_sensitivity.surface_refused",
        }
    }

    /// What the caller can do about it.
    #[must_use]
    pub const fn remedy(&self) -> &'static str {
        match self {
            Self::NoScalarFamily { .. } => {
                "state the sensitivity on the transported effect that is estimated with the \
                 discovered structure, and declare the shift bound there; a structural \
                 premise is repaired by new evidence in the target, not by a range"
            }
            Self::NonFinitePoint => "supply the finite point value the route returned",
            Self::InvalidBound => "declare a finite, positive bound for the shift",
            Self::InvalidGrid { .. } => {
                "request at least two and at most the maximum number of grid points"
            }
            Self::ProportionalAtZero => "declare an additive shift in the quantity's units instead",
            Self::NotTransported => {
                "name the target population the claim is transported to; a claim about the \
                 source population needs no transport premise"
            }
            Self::QuantityNotInTarget { .. } => {
                "state the quantity in the target population the route transports to"
            }
            Self::BlankIdentity(_) => "supply the identity field",
            Self::Surface { .. } => {
                "declare a reference quantity in the same population and units as the \
                 transported quantity"
            }
        }
    }
}

impl std::fmt::Display for TransportSensitivityRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.detail())?;
        match self {
            Self::NoScalarFamily { route } => {
                write!(f, ": route `{}` has a structural premise", route.as_str())
            }
            Self::InvalidGrid { points } => write!(f, ": {points} grid points"),
            Self::QuantityNotInTarget { quantity_population, target_population } => write!(
                f,
                ": quantity is in `{quantity_population}`, the target is `{target_population}`"
            ),
            Self::BlankIdentity(what) => write!(f, ": `{what}` is blank"),
            Self::Surface { message, .. } => write!(f, ": {message}"),
            _ => Ok(()),
        }
    }
}

impl std::error::Error for TransportSensitivityRefusal {}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// The assumption range of one route's point value over the shift grid.
///
/// The range at grid point `k` is `[lower[k], upper[k]]`. It is an assumption range:
/// there is no probability on the shifts and no sampling interval inside it.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportAssumptionRange {
    route: TransportRoute,
    quantity: ScientificQuantityWire,
    point_value: f64,
    family: ViolationFamily,
    coordinate: AssumptionCoordinate,
    grid: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    source_population: String,
    target_population: String,
    query_binding: String,
    provider_snapshot: String,
    causal_contract_id: String,
}

/// Compute the assumption range of a route's point value for a bounded shift of the
/// transported quantity.
///
/// The computation is exact arithmetic on the declared family; nothing is estimated and
/// no calibration is claimed.
///
/// # Errors
/// A structural-premise route ([`TransportSensitivityRefusal::NoScalarFamily`]), a
/// non-finite point or bound, a grid outside `2..=MAX_SENSITIVITY_GRID_POINTS`, a
/// proportional shift of zero, a source equal to the target, a quantity outside the
/// target population or a blank identity.
pub fn transport_assumption_range(
    spec: &TransportSensitivitySpec,
) -> Result<TransportAssumptionRange, TransportSensitivityRefusal> {
    use TransportSensitivityRefusal as R;
    if !spec.route.has_scalar_family() {
        return Err(R::NoScalarFamily { route: spec.route });
    }
    for (name, value) in [
        ("source_population", &spec.source_population),
        ("target_population", &spec.target_population),
        ("query_binding", &spec.query_binding),
        ("provider_snapshot", &spec.provider_snapshot),
        ("causal_contract_id", &spec.causal_contract_id),
    ] {
        if blank(value) {
            return Err(R::BlankIdentity(name));
        }
    }
    if !spec.point_value.is_finite() {
        return Err(R::NonFinitePoint);
    }
    let bound = spec.family.bound();
    if !bound.is_finite() || bound <= 0.0 {
        return Err(R::InvalidBound);
    }
    if spec.grid_points < 2 || spec.grid_points > MAX_SENSITIVITY_GRID_POINTS {
        return Err(R::InvalidGrid { points: spec.grid_points });
    }
    if matches!(spec.family, ViolationFamily::ProportionalShift { .. }) && spec.point_value == 0.0 {
        return Err(R::ProportionalAtZero);
    }
    if spec.source_population == spec.target_population {
        return Err(R::NotTransported);
    }
    if spec.quantity.population_id != spec.target_population {
        return Err(R::QuantityNotInTarget {
            quantity_population: spec.quantity.population_id.clone(),
            target_population: spec.target_population.clone(),
        });
    }
    let last = spec.grid_points - 1;
    let grid: Vec<f64> = (0..spec.grid_points)
        .map(|k| if k == last { bound } else { bound * k as f64 / last as f64 })
        .collect();
    let scale = match spec.family {
        ViolationFamily::AdditiveShift { .. } => 1.0,
        ViolationFamily::ProportionalShift { .. } => spec.point_value.abs(),
    };
    let lower = grid.iter().map(|s| spec.point_value - s * scale).collect();
    let upper = grid.iter().map(|s| spec.point_value + s * scale).collect();
    let units = match spec.family {
        ViolationFamily::AdditiveShift { .. } => spec.quantity.units.clone(),
        ViolationFamily::ProportionalShift { .. } => "fraction".to_owned(),
    };
    let coordinate = AssumptionCoordinate {
        id: format!("{}.shift", spec.route.premise().as_str()),
        scale: spec.family.scale().to_owned(),
        units,
        minimum: 0.0,
        maximum: bound,
    };
    Ok(TransportAssumptionRange {
        route: spec.route,
        quantity: ScientificQuantityWire::from(&spec.quantity),
        point_value: spec.point_value,
        family: spec.family,
        coordinate,
        grid,
        lower,
        upper,
        source_population: spec.source_population.clone(),
        target_population: spec.target_population.clone(),
        query_binding: spec.query_binding.clone(),
        provider_snapshot: spec.provider_snapshot.clone(),
        causal_contract_id: spec.causal_contract_id.clone(),
    })
}

impl TransportAssumptionRange {
    /// The route.
    #[must_use]
    pub const fn route(&self) -> TransportRoute {
        self.route
    }

    /// The premise the shift violates.
    #[must_use]
    pub const fn premise(&self) -> TransportPremise {
        self.route.premise()
    }

    /// The point value the range is about.
    #[must_use]
    pub const fn point_value(&self) -> f64 {
        self.point_value
    }

    /// The violation family and its declared bound.
    #[must_use]
    pub const fn family(&self) -> ViolationFamily {
        self.family
    }

    /// The shift coordinate.
    #[must_use]
    pub const fn coordinate(&self) -> &AssumptionCoordinate {
        &self.coordinate
    }

    /// Grid of shifts, ascending, from `0` to the bound.
    #[must_use]
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Lower end of the assumption range at each grid point.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Upper end of the assumption range at each grid point.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// The range at the declared bound.
    #[must_use]
    pub fn range_at_bound(&self) -> (f64, f64) {
        (self.lower[self.lower.len() - 1], self.upper[self.upper.len() - 1])
    }

    /// The statement that must accompany the range.
    #[must_use]
    pub const fn statement(&self) -> &'static str {
        TRANSPORT_ASSUMPTION_RANGE_INTERPRETATION
    }

    /// The smallest shift at which the range reaches `reference` (its lower end drops to
    /// `reference` for a point value above it, its upper end rises to it for a point
    /// value below it), or `None` when the shift needed exceeds the declared bound or
    /// cannot be reached.
    ///
    /// This is a closed form on the declared family, not an evaluation between grid
    /// points of the artifact.
    #[must_use]
    pub fn reference_reached_at(&self, reference: f64) -> Option<f64> {
        if !reference.is_finite() {
            return None;
        }
        let gap = (self.point_value - reference).abs();
        let shift = match self.family {
            ViolationFamily::AdditiveShift { .. } => gap,
            ViolationFamily::ProportionalShift { .. } => gap / self.point_value.abs(),
        };
        (shift <= self.family.bound()).then_some(shift)
    }

    /// Identity of the range: route, premise, family, populations, derivation and every
    /// grid number, length-delimited.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut w = CanonicalWriter::new("antecedent.transport_sensitivity.range");
        w.word(TRANSPORT_SENSITIVITY_VERSION)
            .text(self.route.as_str())
            .text(self.premise().as_str())
            .text(self.family.scale())
            .real(self.family.bound())
            .real(self.point_value)
            .text(&self.quantity.variable_id)
            .text(&self.quantity.units)
            .text(&self.quantity.population_id)
            .text(&self.quantity.regime_id)
            .word(u64::from(self.quantity.horizon))
            .text(&self.quantity.functional_id)
            .text(&self.quantity.transform_id)
            .text(&self.source_population)
            .text(&self.target_population)
            .text(&self.query_binding)
            .text(&self.provider_snapshot)
            .text(&self.causal_contract_id)
            .reals(&self.grid)
            .reals(&self.lower)
            .reals(&self.upper);
        SemanticDigest::from_bytes(w.finish())
    }

    /// Build the sealed sensitivity artifact for a decision between acting on the
    /// transported quantity and keeping a reference value of the same units in the same
    /// population. The decision classification (invariant action, assumption-dependent
    /// switch, no robust action) then falls out of the existing surface.
    ///
    /// # Errors
    /// A non-finite reference or a surface refusal (a reference in another population or
    /// units, an id collision).
    pub fn to_artifact(
        &self,
        reference: &ScientificQuantity,
        reference_value: f64,
    ) -> Result<SensitivityArtifact, TransportSensitivityRefusal> {
        if !reference_value.is_finite() {
            return Err(TransportSensitivityRefusal::NonFinitePoint);
        }
        let n = self.grid.len();
        let transported = SurfaceQuantity {
            quantity: self.quantity.clone(),
            lower: self.lower.clone(),
            upper: self.upper.clone(),
        };
        let reference = SurfaceQuantity {
            quantity: ScientificQuantityWire::from(reference),
            lower: vec![reference_value; n],
            upper: vec![reference_value; n],
        };
        let actions = vec![
            ActionUtility {
                id: TRANSPORTED_ACTION_ID.to_owned(),
                utility: UtilityTerm::quantity(&transported.quantity.variable_id),
            },
            ActionUtility {
                id: REFERENCE_ACTION_ID.to_owned(),
                utility: UtilityTerm::quantity(&reference.quantity.variable_id),
            },
        ];
        let parts = SensitivityParts {
            coordinate: self.coordinate.clone(),
            grid: self.grid.clone(),
            support: vec![PointSupport::Supported; n],
            quantities: vec![transported, reference],
            actions,
            uncertainty: UncertaintyRelationship {
                assumption_range: AssumptionRangeStatement {
                    kind: antecedent_io_core::sensitivity_artifact::SENSITIVITY_INFERENCE_CLAIM
                        .to_owned(),
                    interpretation: format!(
                        "{ASSUMPTION_RANGE_STATEMENT}; {}",
                        TRANSPORT_ASSUMPTION_RANGE_INTERPRETATION
                    ),
                },
                identified_bound: None,
                sampling: SamplingStatus::Withheld {
                    reason_code: antecedent_core::reason_code!("cell_not_licensed").to_owned(),
                    detail: "transport_sensitivity.interval_not_composed".to_owned(),
                },
            },
            provenance: SensitivityProvenance {
                source_kind: SOURCE_KIND.to_owned(),
                query_binding: self.query_binding.clone(),
                provider_snapshot: self.provider_snapshot.clone(),
                source_regime: format!("population:{}", self.source_population),
                method: format!(
                    "{} shift of the transported quantity in `{}` (route `{}`, premise `{}`)",
                    self.family.scale(),
                    self.target_population,
                    self.route.as_str(),
                    self.premise().as_str()
                ),
                causal_contract_id: self.causal_contract_id.clone(),
                decision_threshold: Some(reference_value),
                source_tipping: Vec::new(),
            },
        };
        SensitivityArtifact::new(parts).map_err(|error| {
            let (code, message) = match error {
                IoError::Refused { code, message } => (code, message),
                other => (antecedent_core::reason_code!("invalid_argument"), other.to_string()),
            };
            TransportSensitivityRefusal::Surface { code, message }
        })
    }
}
