//! Scientific binding for one piece of evidence, with uncertainty kept on separate axes.
//!
//! The envelope binds a [`ScientificQuantity`] (quantity, units, population,
//! regime, horizon, and conditioning) to the world and time that coordinate
//! does not name, plus structural premises, origins, support, and calibration.
//! Parameter, sampling, predictive, structural, measurement, identification,
//! and Monte Carlo uncertainty are different fields. An empty field is
//! absence. A constructor or conversion that copies one axis onto another is
//! refused, so a sampling interval cannot become a parameter posterior.
//!
//! This record does not replace [`crate::evidence_ledger::EvidenceLedger`],
//! [`ScientificQuantity`], or [`crate::ClaimEnvelope`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::text::is_blank;
use antecedent_core::{CanonicalWriter, SemanticDigest, SupportStatus};

use crate::{QuantityMismatch, ScientificQuantity};

const IDENTITY_TAG: &str = "antecedent.contract.evidence_envelope.v1";

/// Which uncertainty statement a number belongs to.
///
/// The seven axes are not interchangeable. Declaration order is the canonical
/// order used by [`EvidenceEnvelope::identity`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UncertaintyAxis {
    /// Parameter-posterior uncertainty. Not a sampling distribution.
    Parameter,
    /// Repeated-sampling uncertainty of an estimator. Not a parameter posterior.
    Sampling,
    /// Outcome or interventional predictive uncertainty. Not parameter uncertainty.
    Predictive,
    /// Uncertainty from the structural model or mechanism class. Not sampling error.
    Structural,
    /// Uncertainty from the measurement process. Not a parameter posterior.
    Measurement,
    /// Partial-identification or identified-set uncertainty. Not a posterior probability.
    Identification,
    /// Numerical Monte Carlo error. Not scientific uncertainty about the estimand.
    MonteCarlo,
}

impl UncertaintyAxis {
    /// Every axis, in canonical order.
    pub const ALL: [Self; 7] = [
        Self::Parameter,
        Self::Sampling,
        Self::Predictive,
        Self::Structural,
        Self::Measurement,
        Self::Identification,
        Self::MonteCarlo,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::Sampling => "sampling",
            Self::Predictive => "predictive",
            Self::Structural => "structural",
            Self::Measurement => "measurement",
            Self::Identification => "identification",
            Self::MonteCarlo => "monte_carlo",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|axis| axis.as_str() == name)
    }

    /// Domain tag written into the canonical identity. Zero is reserved for absence.
    const fn wire(self) -> u64 {
        match self {
            Self::Parameter => 1,
            Self::Sampling => 2,
            Self::Predictive => 3,
            Self::Structural => 4,
            Self::Measurement => 5,
            Self::Identification => 6,
            Self::MonteCarlo => 7,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Parameter => 0,
            Self::Sampling => 1,
            Self::Predictive => 2,
            Self::Structural => 3,
            Self::Measurement => 4,
            Self::Identification => 5,
            Self::MonteCarlo => 6,
        }
    }
}

/// A finite closed interval bound to exactly one [`UncertaintyAxis`].
///
/// A zero-width interval, including `[0, 0]`, is a present statement. An axis
/// with no interval is an empty slot on [`UncertaintyAxes`], not this value.
/// The axis tag is fixed at construction and cannot be rewritten. Endpoints are
/// normalised so `-0.0` is stored as `0.0`: equality and the digest then agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisInterval {
    axis: UncertaintyAxis,
    lower: f64,
    upper: f64,
}

impl AxisInterval {
    /// Bind `lower <= upper` to `axis`.
    ///
    /// # Errors
    ///
    /// Returns an error when either endpoint is non-finite or `lower > upper`.
    #[must_use = "an unordered or non-finite interval is refused"]
    pub const fn try_new(
        axis: UncertaintyAxis,
        lower: f64,
        upper: f64,
    ) -> Result<Self, EvidenceEnvelopeError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(EvidenceEnvelopeError::bounds(axis));
        }
        // Adding zero turns `-0.0` into `0.0`; every other finite value is unchanged.
        Ok(Self { axis, lower: lower + 0.0, upper: upper + 0.0 })
    }

    /// Axis this interval was constructed for.
    #[must_use]
    pub const fn axis(self) -> UncertaintyAxis {
        self.axis
    }

    /// Finite lower endpoint.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// Finite upper endpoint.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.upper
    }
}

/// The seven uncertainty axes. Each slot is independently present or absent.
///
/// `None` is explicit absence. It is not the interval `[0, 0]`, and it is not
/// a statement copied from another axis. Slots are private so an interval can
/// be stored only under the axis it was constructed for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UncertaintyAxes {
    slots: [Option<AxisInterval>; 7],
}

impl UncertaintyAxes {
    /// Every axis absent.
    ///
    /// Absence is not a zero interval on any axis.
    #[must_use]
    pub const fn absent() -> Self {
        Self { slots: [None; 7] }
    }

    /// The interval stored on `axis`, or absence.
    #[must_use]
    pub const fn get(&self, axis: UncertaintyAxis) -> Option<AxisInterval> {
        self.slots[axis.index()]
    }

    /// Store `interval` on the axis it already names, replacing a previous
    /// statement of that same axis.
    ///
    /// The interval is never retagged: a sampling interval cannot be placed on
    /// the parameter axis, so it cannot be presented as a parameter posterior.
    #[must_use]
    pub const fn insert(mut self, interval: AxisInterval) -> Self {
        self.slots[interval.axis().index()] = Some(interval);
        self
    }

    /// The interval on `axis`.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is empty. The error is not the interval
    /// `[0, 0]`.
    #[must_use = "an empty axis is absence, not a zero interval"]
    pub const fn require_interval(
        &self,
        axis: UncertaintyAxis,
    ) -> Result<AxisInterval, EvidenceEnvelopeError> {
        match self.get(axis) {
            Some(interval) => Ok(interval),
            None => Err(EvidenceEnvelopeError::absent(axis)),
        }
    }
}

/// Where the evidence came from. Each origin is its own id.
///
/// `None` means that origin was not recorded. An empty string is not absence
/// and is rejected by [`EvidenceBinding::validate`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceOrigins {
    /// Observation or lineage identity of the source.
    pub source: Option<Arc<str>>,
    /// Selection history that produced the analyzed rows.
    pub selection: Option<Arc<str>>,
    /// Training or fitting history, when a fit was used.
    pub training: Option<Arc<str>>,
    /// Prior identity, when a prior contributed.
    pub prior: Option<Arc<str>>,
}

/// World and time the evidence is about.
///
/// These are not [`ScientificQuantity::regime_id`] and not
/// [`ScientificQuantity::horizon`]. A regime names the intervention. A world
/// names the potential-outcome or scenario world. A time id names the decision
/// time or period; the horizon count stays on the quantity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTime {
    /// Potential-outcome or scenario world.
    pub world_id: Arc<str>,
    /// Time index, such as a decision time or period.
    pub time_id: Arc<str>,
}

/// Support of the bound statement.
///
/// The status is the empirical-support label. The domain id names where that
/// label was assessed. Neither field is an uncertainty axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceSupport {
    /// Empirical support label.
    pub status: SupportStatus,
    /// Domain the status was assessed on.
    pub domain_id: Arc<str>,
}

/// Calibration of the bound statement.
///
/// This is the envelope's calibration binding, not a [`crate::CalibrationView`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceCalibration {
    /// A named calibration record covers this binding.
    Calibrated {
        /// Identity of the governing calibration record.
        record_id: Arc<str>,
    },
    /// No calibration record was assessed for this scope.
    ScopeNotAssessed,
    /// Calibration cannot be stated.
    Unavailable {
        /// Stable reason id.
        reason: Arc<str>,
    },
}

/// Scientific binding held fixed while uncertainty axes vary.
///
/// Quantity, units, population, regime, horizon, and conditioning live on
/// [`ScientificQuantity`]. [`WorldTime`] names the world and time those fields
/// do not. Premises, origins, support, and calibration complete the binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceBinding {
    /// Quantity coordinate, including units, population, regime, horizon, and conditioning.
    pub quantity: ScientificQuantity,
    /// World and time.
    pub world: WorldTime,
    /// Structural premises, in strictly increasing order.
    ///
    /// Empty means no premise was declared. It does not mean every premise holds.
    pub premises: Arc<[Arc<str>]>,
    /// Source, selection, training, and prior origins.
    pub origins: EvidenceOrigins,
    /// Support of this binding.
    pub support: EvidenceSupport,
    /// Calibration of this binding.
    pub calibration: EvidenceCalibration,
}

impl EvidenceBinding {
    /// Validate and store a binding.
    ///
    /// # Errors
    ///
    /// Returns an error when the quantity coordinate is invalid, the world or
    /// time id is blank, premises are blank or not strictly increasing, an
    /// origin is a blank string, the support domain is blank, or a calibration
    /// id is blank.
    #[must_use = "an invalid binding is refused rather than stored"]
    pub fn try_new(
        quantity: ScientificQuantity,
        world: WorldTime,
        premises: Arc<[Arc<str>]>,
        origins: EvidenceOrigins,
        support: EvidenceSupport,
        calibration: EvidenceCalibration,
    ) -> Result<Self, EvidenceEnvelopeError> {
        let binding = Self { quantity, world, premises, origins, support, calibration };
        binding.validate()?;
        Ok(binding)
    }

    /// Whether this binding names a complete scientific coordinate.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::try_new`].
    #[must_use = "an invalid binding is refused"]
    pub fn validate(&self) -> Result<(), EvidenceEnvelopeError> {
        self.quantity.validate().map_err(quantity_error)?;
        if is_blank(&self.world.world_id) {
            return Err(binding_error("evidence_envelope.world", "a world id is required"));
        }
        if is_blank(&self.world.time_id) {
            return Err(binding_error("evidence_envelope.time", "a time id is required"));
        }
        if premises_not_canonical(&self.premises) {
            return Err(binding_error(
                "evidence_envelope.premises",
                "premises are declared in strictly increasing order, without blanks",
            ));
        }
        self.origins.check()?;
        if is_blank(&self.support.domain_id) {
            return Err(binding_error(
                "evidence_envelope.support",
                "support names the domain it was assessed on",
            ));
        }
        self.calibration.check()
    }
}

impl EvidenceOrigins {
    fn check(&self) -> Result<(), EvidenceEnvelopeError> {
        let origins = [
            self.source.as_deref(),
            self.selection.as_deref(),
            self.training.as_deref(),
            self.prior.as_deref(),
        ];
        if origins.into_iter().any(|origin| origin.is_some_and(is_blank)) {
            return Err(binding_error(
                "evidence_envelope.origin",
                "an origin is either absent or a non-blank id",
            ));
        }
        Ok(())
    }
}

impl EvidenceCalibration {
    fn check(&self) -> Result<(), EvidenceEnvelopeError> {
        match self {
            Self::Calibrated { record_id } if is_blank(record_id) => Err(binding_error(
                "evidence_envelope.calibration",
                "a calibrated envelope names a non-blank calibration record",
            )),
            Self::Unavailable { reason } if is_blank(reason) => Err(binding_error(
                "evidence_envelope.calibration",
                "an unavailable calibration names a non-blank reason",
            )),
            Self::Calibrated { .. } | Self::ScopeNotAssessed | Self::Unavailable { .. } => Ok(()),
        }
    }
}

/// One evidence envelope: a scientific binding plus seven uncertainty axes.
///
/// Equality compares the binding and every axis. Two envelopes with the same
/// binding and different axis contents are not equal. Changing population,
/// world, or conditioning changes the binding, and therefore equality.
///
/// [`Self::identity`] writes that same content, including every absent axis,
/// through [`CanonicalWriter`]. Interval endpoints are normalised at construction
/// (`-0.0` is stored as `0.0`), so equality and the digest agree.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceEnvelope {
    binding: EvidenceBinding,
    axes: UncertaintyAxes,
}

impl EvidenceEnvelope {
    /// Bind `axes` to a validated `binding`.
    ///
    /// # Errors
    ///
    /// Returns an error when `binding` fails [`EvidenceBinding::validate`].
    /// Axis contents are not rewritten and are not required to be non-empty.
    #[allow(
        clippy::large_types_passed_by_value,
        reason = "the envelope stores the binding it is given"
    )]
    #[must_use = "an invalid binding is refused rather than wrapped"]
    pub fn try_new(
        binding: EvidenceBinding,
        axes: UncertaintyAxes,
    ) -> Result<Self, EvidenceEnvelopeError> {
        binding.validate()?;
        Ok(Self { binding, axes })
    }

    /// Scientific binding, validated at construction. It cannot be changed afterwards.
    /// Axis contents are not part of it.
    #[must_use]
    pub const fn binding(&self) -> &EvidenceBinding {
        &self.binding
    }

    /// Uncertainty axes. Empty slots are absence.
    #[must_use]
    pub const fn axes(&self) -> UncertaintyAxes {
        self.axes
    }

    /// Canonical identity of the binding and every uncertainty axis.
    ///
    /// The stream starts with `antecedent.contract.evidence_envelope.v1`, then
    /// the quantity, world, time, premises, the four origins, support,
    /// calibration, and the seven axes in [`UncertaintyAxis::ALL`] order. Each
    /// axis is named even when absent.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        self.write_identity(&mut writer);
        SemanticDigest::from_bytes(writer.finish())
    }

    fn write_identity(&self, writer: &mut CanonicalWriter) {
        write_binding(writer, &self.binding);
        write_axes(writer, &self.axes);
    }
}

/// A refused envelope construction or axis conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct EvidenceEnvelopeError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `evidence_envelope.*` detail.
    pub detail: &'static str,
    /// What was refused.
    pub message: &'static str,
    /// Axis the refusal is about, when it is about an axis.
    pub axis: Option<UncertaintyAxis>,
}

impl EvidenceEnvelopeError {
    /// The refusal as `(code, detail, message)`.
    #[must_use]
    pub const fn refusal(self) -> (&'static str, &'static str, &'static str) {
        (self.code, self.detail, self.message)
    }

    const fn new(
        code: &'static str,
        detail: &'static str,
        message: &'static str,
        axis: Option<UncertaintyAxis>,
    ) -> Self {
        Self { code, detail, message, axis }
    }

    const fn bounds(axis: UncertaintyAxis) -> Self {
        Self::new(
            antecedent_core::reason_code!("invalid_argument"),
            "evidence_envelope.interval_bounds",
            "an axis interval needs finite lower <= upper",
            Some(axis),
        )
    }

    const fn absent(axis: UncertaintyAxis) -> Self {
        Self::new(
            antecedent_core::reason_code!("invalid_argument"),
            "evidence_envelope.absent_axis",
            "an empty uncertainty axis is absence, not a zero interval",
            Some(axis),
        )
    }
}

fn binding_error(detail: &'static str, message: &'static str) -> EvidenceEnvelopeError {
    EvidenceEnvelopeError::new(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        message,
        None,
    )
}

fn quantity_error(mismatch: QuantityMismatch) -> EvidenceEnvelopeError {
    let message = match mismatch {
        QuantityMismatch::InvalidCoordinate => {
            "the quantity coordinate is incomplete or its conditioning is not canonical"
        }
        QuantityMismatch::Variable => "the quantity variable is not identified",
        QuantityMismatch::Units => "the quantity units are not identified",
        QuantityMismatch::Population => "the quantity population is not identified",
        QuantityMismatch::Regime => "the quantity regime is not identified",
        QuantityMismatch::Horizon => "the quantity horizon is not identified",
        QuantityMismatch::Functional => "the quantity functional is not identified",
        QuantityMismatch::Conditioning => "the quantity conditioning is not identified",
        QuantityMismatch::Transform => "the quantity transform is not identified",
    };
    binding_error("evidence_envelope.quantity", message)
}

fn premises_not_canonical(premises: &[Arc<str>]) -> bool {
    premises.iter().any(|premise| is_blank(premise))
        || premises.windows(2).any(|pair| pair[0].as_ref() >= pair[1].as_ref())
}

fn write_binding(writer: &mut CanonicalWriter, binding: &EvidenceBinding) {
    binding.quantity.write_canonical(writer);
    writer.text(&binding.world.world_id).text(&binding.world.time_id);
    writer.texts(binding.premises.as_ref());
    writer
        .opt_text(binding.origins.source.as_deref())
        .opt_text(binding.origins.selection.as_deref())
        .opt_text(binding.origins.training.as_deref())
        .opt_text(binding.origins.prior.as_deref());
    writer.text(binding.support.status.as_str()).text(&binding.support.domain_id);
    write_calibration(writer, &binding.calibration);
}

fn write_calibration(writer: &mut CanonicalWriter, calibration: &EvidenceCalibration) {
    match calibration {
        EvidenceCalibration::Calibrated { record_id } => {
            writer.word(1).text(record_id);
        }
        EvidenceCalibration::ScopeNotAssessed => {
            writer.word(2);
        }
        EvidenceCalibration::Unavailable { reason } => {
            writer.word(3).text(reason);
        }
    }
}

fn write_axes(writer: &mut CanonicalWriter, axes: &UncertaintyAxes) {
    writer.word(UncertaintyAxis::ALL.len() as u64);
    for axis in UncertaintyAxis::ALL {
        writer.word(axis.wire());
        match axes.get(axis) {
            None => {
                writer.word(0);
            }
            Some(interval) => {
                writer.word(1).real(interval.lower()).real(interval.upper());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::SupportStatus;

    use super::*;
    use crate::{QuantityCondition, QuantityRole, ScientificQuantity};

    fn quantity() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "outcome".into(),
            role: QuantityRole::Outcome,
            units: "kg".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 2,
            functional_id: "mean".into(),
            conditioning: vec![QuantityCondition {
                variable_id: "schema:z".into(),
                value_id: "high".into(),
            }],
            transform_id: "identity".into(),
        }
    }

    fn binding() -> EvidenceBinding {
        EvidenceBinding::try_new(
            quantity(),
            WorldTime { world_id: "world:do(a=1)".into(), time_id: "period:2".into() },
            Arc::from([Arc::from("conditional_exchangeability")]),
            EvidenceOrigins {
                source: Some("source:trial-1".into()),
                selection: Some("selection:enrolled".into()),
                training: Some("training:fold-0".into()),
                prior: Some("prior:weakly_informative".into()),
            },
            EvidenceSupport {
                status: SupportStatus::Supported,
                domain_id: "support:dose-grid".into(),
            },
            EvidenceCalibration::Calibrated { record_id: "calibration:record-1".into() },
        )
        .unwrap()
    }

    fn envelope(axes: UncertaintyAxes) -> EvidenceEnvelope {
        EvidenceEnvelope::try_new(binding(), axes).unwrap()
    }

    #[test]
    fn same_binding_with_different_axes_is_not_equal() {
        let sampling = AxisInterval::try_new(UncertaintyAxis::Sampling, -0.2, 0.5).unwrap();
        let left = envelope(UncertaintyAxes::absent());
        let right = envelope(UncertaintyAxes::absent().insert(sampling));
        assert_eq!(left.binding, right.binding);
        assert_ne!(left, right);
        assert_ne!(left.identity(), right.identity());
        assert!(left.axes().get(UncertaintyAxis::Sampling).is_none());
        assert_eq!(right.axes().get(UncertaintyAxis::Sampling), Some(sampling));
        assert_eq!(left, left.clone());
    }

    #[test]
    fn population_world_and_conditioning_change_equality() {
        let base = envelope(UncertaintyAxes::absent());

        let mut population = base.clone();
        population.binding.quantity.population_id = "other-population".into();
        assert_ne!(base, population);
        assert_ne!(base.identity(), population.identity());

        let mut world = base.clone();
        world.binding.world.world_id = "world:do(a=0)".into();
        assert_ne!(base, world);
        assert_ne!(base.identity(), world.identity());
        assert_eq!(base.binding.quantity, world.binding.quantity);

        let mut conditioning = base.clone();
        conditioning.binding.quantity.conditioning[0].value_id = "low".into();
        assert_ne!(base, conditioning);
        assert_ne!(base.identity(), conditioning.identity());
    }

    #[test]
    fn sampling_interval_is_not_a_parameter_posterior() {
        let sampling = AxisInterval::try_new(UncertaintyAxis::Sampling, 0.1, 0.9).unwrap();
        assert_eq!(sampling.axis(), UncertaintyAxis::Sampling);
        // An interval is stored only under the axis it names.
        let placed = UncertaintyAxes::absent().insert(sampling);
        assert_eq!(placed.get(UncertaintyAxis::Sampling), Some(sampling));
        assert!(placed.get(UncertaintyAxis::Parameter).is_none());

        assert!(AxisInterval::try_new(UncertaintyAxis::Sampling, 1.0, 0.0).is_err());
        assert!(AxisInterval::try_new(UncertaintyAxis::Sampling, f64::NAN, 1.0).is_err());
    }

    #[test]
    fn negative_zero_endpoints_equal_positive_zero_and_share_an_identity() {
        let positive = AxisInterval::try_new(UncertaintyAxis::Parameter, 0.0, 0.0).unwrap();
        let negative = AxisInterval::try_new(UncertaintyAxis::Parameter, -0.0, -0.0).unwrap();
        assert_eq!(positive, negative);
        assert_eq!(negative.lower().to_bits(), 0.0_f64.to_bits());
        assert_eq!(negative.upper().to_bits(), 0.0_f64.to_bits());
        let left = envelope(UncertaintyAxes::absent().insert(positive));
        let right = envelope(UncertaintyAxes::absent().insert(negative));
        assert_eq!(left, right);
        assert_eq!(left.identity(), right.identity());
        assert_eq!(left.binding().world.world_id, right.binding().world.world_id);
    }

    #[test]
    fn absent_axis_is_not_a_zero_interval() {
        let absent = UncertaintyAxes::absent();
        assert!(absent.get(UncertaintyAxis::Parameter).is_none());
        let missing = absent.require_interval(UncertaintyAxis::Parameter).unwrap_err();
        assert_eq!(missing.detail, "evidence_envelope.absent_axis");
        assert_eq!(missing.axis, Some(UncertaintyAxis::Parameter));
        assert_eq!(missing.message, "an empty uncertainty axis is absence, not a zero interval");

        let zero = AxisInterval::try_new(UncertaintyAxis::Parameter, 0.0, 0.0).unwrap();
        let present = absent.insert(zero);
        let empty_envelope = envelope(absent);
        let zero_envelope = envelope(present);
        assert_eq!(empty_envelope.binding, zero_envelope.binding);
        assert_ne!(empty_envelope, zero_envelope);
        assert_ne!(empty_envelope.identity(), zero_envelope.identity());
    }

    #[test]
    fn identity_includes_the_binding_and_every_axis() {
        assert_eq!(UncertaintyAxis::ALL.map(UncertaintyAxis::wire), [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(UncertaintyAxis::ALL.map(UncertaintyAxis::index), [0, 1, 2, 3, 4, 5, 6]);

        let base = envelope(UncertaintyAxes::absent());
        assert_eq!(base.identity(), envelope(UncertaintyAxes::absent()).identity());

        let mut seen = vec![base.identity()];
        for axis in UncertaintyAxis::ALL {
            let interval = AxisInterval::try_new(axis, 0.25, 0.75).unwrap();
            let id = envelope(UncertaintyAxes::absent().insert(interval)).identity();
            let name = axis.as_str();
            assert!(!seen.contains(&id), "{name}");
            seen.push(id);
        }

        let sampling = AxisInterval::try_new(UncertaintyAxis::Sampling, 0.25, 0.75).unwrap();
        let parameter = AxisInterval::try_new(UncertaintyAxis::Parameter, 0.25, 0.75).unwrap();
        assert_ne!(
            envelope(UncertaintyAxes::absent().insert(sampling)).identity(),
            envelope(UncertaintyAxes::absent().insert(parameter)).identity(),
        );

        let mut prior = base.clone();
        prior.binding.origins.prior = Some(Arc::from("prior:other"));
        assert_ne!(base.identity(), prior.identity());

        let mut premises = base.clone();
        premises.binding.premises = Arc::from([]);
        assert_ne!(base.identity(), premises.identity());

        let mut support = base.clone();
        support.binding.support.status = SupportStatus::MissingEvidence;
        assert_ne!(base.identity(), support.identity());

        let mut calibration = base.clone();
        calibration.binding.calibration = EvidenceCalibration::ScopeNotAssessed;
        assert_ne!(base.identity(), calibration.identity());
    }

    #[test]
    fn blank_world_and_unsorted_premises_are_refused() {
        let mut world = binding();
        world.world.world_id = "  ".into();
        assert_eq!(
            EvidenceEnvelope::try_new(world, UncertaintyAxes::absent()).unwrap_err().detail,
            "evidence_envelope.world",
        );

        let mut premises = binding();
        premises.premises = Arc::from([Arc::from("b"), Arc::from("a")]);
        assert_eq!(
            EvidenceEnvelope::try_new(premises, UncertaintyAxes::absent()).unwrap_err().detail,
            "evidence_envelope.premises",
        );
    }
}
