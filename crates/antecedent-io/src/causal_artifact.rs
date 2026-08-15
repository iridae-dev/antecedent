//! Standalone artifacts for query and result wire payloads.
//!
//! This module is deliberately a container around the existing wire types.  It does
//! not introduce another JSON or CBOR representation of causal queries/results.

use std::collections::HashSet;

use antecedent_core::{Assumption, AssumptionScope, ResponseQuery};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactKind, ArtifactManifest, CausalQueryWire, CausalResponseWire, EncodedArtifact,
    InterferenceEstimateWire, IoError, ProvenanceWire, STABLE_FORMAT, SectionBytes,
    SemanticVersion, TransportEffectEstimateWire, TransportIdentificationWire,
    causal_query_from_wire, causal_response_from_wire, from_cbor, read_and_migrate,
    section_descriptor, to_cbor, transport_effect_from_wire,
};

const HEADER_SECTION: &str = "causal_payload.header";
const PAYLOAD_SECTION: &str = "causal_payload.body";
const ARTIFACT_KIND: &str = "causal_payload";

/// The existing wire type stored in a standalone causal-payload artifact.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CausalPayloadKind {
    /// Any causal query, including response, transport, and interference queries.
    Query,
    /// A causal-response result.
    ResponseResult,
    /// A structural transport identification result.
    TransportIdentification,
    /// A trial-to-target statistical estimate.
    TransportEstimate,
    /// A randomized-interference estimate.
    InterferenceEstimate,
}

/// Small self-describing header; variable names map stable wire ids to language names.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalPayloadHeader {
    /// Type of the payload section.
    pub payload_kind: CausalPayloadKind,
    /// Variable names in raw-id order.
    pub variable_names: Vec<String>,
}

/// Decoded standalone causal payload, represented by the canonical existing wire type.
#[derive(Clone, Debug, PartialEq)]
pub enum CausalPayloadWire {
    /// Query wire.
    Query(Box<CausalQueryWire>),
    /// Response-result wire.
    ResponseResult(Box<CausalResponseWire>),
    /// Transport-identification wire.
    TransportIdentification(Box<TransportIdentificationWire>),
    /// Trial transport-estimate wire.
    TransportEstimate(Box<TransportEffectEstimateWire>),
    /// Interference-estimate wire.
    InterferenceEstimate(Box<InterferenceEstimateWire>),
}

impl CausalPayloadWire {
    /// Payload discriminator.
    #[must_use]
    pub const fn kind(&self) -> CausalPayloadKind {
        match self {
            Self::Query(_) => CausalPayloadKind::Query,
            Self::ResponseResult(_) => CausalPayloadKind::ResponseResult,
            Self::TransportIdentification(_) => CausalPayloadKind::TransportIdentification,
            Self::TransportEstimate(_) => CausalPayloadKind::TransportEstimate,
            Self::InterferenceEstimate(_) => CausalPayloadKind::InterferenceEstimate,
        }
    }
}

/// Encode one existing causal wire payload as a format-0.3 artifact.
///
/// # Errors
///
/// Returns an error if header/payload CBOR serialization or the library semantic
/// version conversion fails.
pub fn encode_causal_payload_artifact(
    payload: &CausalPayloadWire,
    variable_names: Vec<String>,
    artifact_id: &str,
) -> Result<EncodedArtifact, IoError> {
    let header = CausalPayloadHeader { payload_kind: payload.kind(), variable_names };
    validate_header(&header)?;
    validate_payload(payload, header.variable_names.len())?;
    let header_bytes = to_cbor(&header)?;
    let payload_bytes = match payload {
        CausalPayloadWire::Query(value) => to_cbor(value)?,
        CausalPayloadWire::ResponseResult(value) => to_cbor(value)?,
        CausalPayloadWire::TransportIdentification(value) => to_cbor(value)?,
        CausalPayloadWire::TransportEstimate(value) => to_cbor(value)?,
        CausalPayloadWire::InterferenceEstimate(value) => to_cbor(value)?,
    };
    let sections = vec![
        SectionBytes::new(HEADER_SECTION, header_bytes.clone()),
        SectionBytes::new(PAYLOAD_SECTION, payload_bytes.clone()),
    ];
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(antecedent_core::VERSION)?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(HEADER_SECTION, "application/cbor", &header_bytes),
                section_descriptor(PAYLOAD_SECTION, "application/cbor", &payload_bytes),
            ],
            provenance: ProvenanceWire { note: "canonical causal wire payload".into() },
        },
        sections,
    })
}

/// Decode a standalone causal payload artifact after applying supported migrations.
///
/// # Errors
///
/// Returns an error for the wrong artifact kind, missing sections, invalid CBOR, or
/// an unsupported source format.
pub fn decode_causal_payload_artifact(
    bytes: &[u8],
) -> Result<(EncodedArtifact, CausalPayloadHeader, CausalPayloadWire), IoError> {
    let artifact = read_and_migrate(bytes)?;
    if artifact.manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into()) {
        return Err(IoError::Convert("expected a causal_payload artifact".into()));
    }
    let header_bytes = artifact
        .sections
        .iter()
        .find(|section| section.id == HEADER_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{HEADER_SECTION}`")))?;
    let payload_bytes = artifact
        .sections
        .iter()
        .find(|section| section.id == PAYLOAD_SECTION)
        .ok_or_else(|| IoError::Convert(format!("missing section `{PAYLOAD_SECTION}`")))?;
    let header: CausalPayloadHeader = from_cbor(&header_bytes.data)?;
    let payload = match header.payload_kind {
        CausalPayloadKind::Query => {
            CausalPayloadWire::Query(Box::new(from_cbor(&payload_bytes.data)?))
        }
        CausalPayloadKind::ResponseResult => {
            CausalPayloadWire::ResponseResult(Box::new(from_cbor(&payload_bytes.data)?))
        }
        CausalPayloadKind::TransportIdentification => {
            CausalPayloadWire::TransportIdentification(Box::new(from_cbor(&payload_bytes.data)?))
        }
        CausalPayloadKind::TransportEstimate => {
            CausalPayloadWire::TransportEstimate(Box::new(from_cbor(&payload_bytes.data)?))
        }
        CausalPayloadKind::InterferenceEstimate => {
            CausalPayloadWire::InterferenceEstimate(Box::new(from_cbor(&payload_bytes.data)?))
        }
    };
    validate_header(&header)?;
    validate_payload(&payload, header.variable_names.len())?;
    Ok((artifact, header, payload))
}

fn validate_header(header: &CausalPayloadHeader) -> Result<(), IoError> {
    let mut names = HashSet::with_capacity(header.variable_names.len());
    for name in &header.variable_names {
        if name.trim().is_empty() {
            return Err(IoError::Convert("causal payload variable names must be non-blank".into()));
        }
        if !names.insert(name.as_str()) {
            return Err(IoError::Convert(format!(
                "duplicate causal payload variable name `{name}`"
            )));
        }
    }
    Ok(())
}

fn validate_payload(payload: &CausalPayloadWire, variable_count: usize) -> Result<(), IoError> {
    match payload {
        CausalPayloadWire::Query(wire) => {
            validate_query_ids(wire, variable_count)?;
            causal_query_from_wire(wire)?
                .validate()
                .map_err(|error| IoError::Convert(error.to_string()))?;
        }
        CausalPayloadWire::ResponseResult(wire) => {
            validate_response_functional_ids(&wire.estimand, variable_count)?;
            let response = causal_response_from_wire(wire)?;
            ResponseQuery::new(response.estimand.clone())
                .validate()
                .map_err(|error| IoError::Convert(error.to_string()))?;
            for record in &response.assumptions.entries {
                if let Assumption::ExclusionRestriction { instrument } = record.assumption {
                    validate_id(instrument.raw(), variable_count)?;
                }
                if let AssumptionScope::Variables { variables } = &record.scope {
                    validate_ids(variables.iter().map(|variable| variable.raw()), variable_count)?;
                }
            }
            validate_response_result(wire)?;
        }
        CausalPayloadWire::TransportIdentification(wire) => {
            validate_transport_identification_ids(wire, variable_count)?;
            // transport_identification_from_wire is infallible: it has no fallible decode
            // step to check here, unlike transport_effect_from_wire below.
            validate_transport_identification(wire)?;
        }
        CausalPayloadWire::TransportEstimate(wire) => {
            let _ = transport_effect_from_wire(wire)?;
            validate_transport_estimate(wire)?;
        }
        CausalPayloadWire::InterferenceEstimate(wire) => {
            // interference_estimate_from_wire is infallible: it has no fallible decode
            // step to check here, unlike transport_effect_from_wire above.
            validate_interference_estimate(wire)?;
        }
    }
    Ok(())
}

fn validate_id(raw: u32, variable_count: usize) -> Result<(), IoError> {
    if usize::try_from(raw).map_or(true, |id| id >= variable_count) {
        return Err(IoError::Convert(format!(
            "variable id {raw} is outside causal payload header bounds (len={variable_count})"
        )));
    }
    Ok(())
}

fn validate_ids(ids: impl IntoIterator<Item = u32>, variable_count: usize) -> Result<(), IoError> {
    for id in ids {
        validate_id(id, variable_count)?;
    }
    Ok(())
}

fn validate_intervention_ids(
    intervention: &crate::InterventionWire,
    variable_count: usize,
) -> Result<(), IoError> {
    match intervention {
        crate::InterventionWire::Set { variable, .. }
        | crate::InterventionWire::Shift { variable, .. }
        | crate::InterventionWire::Stochastic { variable, .. }
        | crate::InterventionWire::Soft { variable, .. } => validate_id(*variable, variable_count),
        crate::InterventionWire::Sequence { steps } => {
            for step in steps {
                validate_intervention_ids(&step.intervention, variable_count)?;
            }
            Ok(())
        }
    }
}

fn validate_response_functional_ids(
    functional: &crate::ResponseFunctionalWire,
    variable_count: usize,
) -> Result<(), IoError> {
    use crate::ResponseFunctionalWire;
    match functional {
        ResponseFunctionalWire::MeanCurve { outcome, treatment } => {
            validate_ids([*outcome, treatment.variable], variable_count)
        }
        ResponseFunctionalWire::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctionalWire::PointDerivative { outcome, treatment, .. } => {
            validate_ids([*outcome, *treatment], variable_count)
        }
        ResponseFunctionalWire::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctionalWire::Jacobian { outcomes, treatments, .. } => {
            validate_ids(outcomes.iter().chain(treatments).copied(), variable_count)
        }
        ResponseFunctionalWire::InterventionResponse { outcome, interventions } => {
            validate_id(*outcome, variable_count)?;
            for intervention in interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            Ok(())
        }
    }
}

fn validate_response_query_ids(
    query: &crate::ResponseQueryWire,
    variable_count: usize,
) -> Result<(), IoError> {
    use crate::ObservationSpecWire;

    validate_response_functional_ids(&query.functional, variable_count)?;
    match &query.observation {
        ObservationSpecWire::Complete => {}
        ObservationSpecWire::RightCensored { latent, observed, censoring, event }
        | ObservationSpecWire::LeftCensored { latent, observed, censoring, event } => {
            validate_ids([*latent, *observed, *censoring, *event], variable_count)?;
        }
        ObservationSpecWire::IntervalCensored { latent, lower, upper } => {
            validate_ids([*latent, *lower, *upper], variable_count)?;
        }
        ObservationSpecWire::Truncated { latent, observed, lower, upper } => {
            validate_ids([*latent, *observed], variable_count)?;
            validate_ids(lower.iter().chain(upper).copied(), variable_count)?;
        }
        ObservationSpecWire::Selected { latent, observed, indicator } => {
            validate_ids([*latent, *observed, *indicator], variable_count)?;
        }
    }
    for assumption in &query.observation_assumptions {
        match assumption {
            crate::ObservationAssumptionWire::IndependentGiven(ids)
            | crate::ObservationAssumptionWire::OutcomeIndependentGiven(ids) => {
                validate_ids(ids.iter().copied(), variable_count)?;
            }
            crate::ObservationAssumptionWire::Structural(_) => {}
        }
    }
    Ok(())
}

fn validate_query_ids(query: &CausalQueryWire, variable_count: usize) -> Result<(), IoError> {
    use CausalQueryWire as Q;
    match query {
        Q::AverageEffect { treatment, outcome, effect_modifiers, control, active, .. } => {
            validate_ids(
                [*treatment, *outcome].into_iter().chain(effect_modifiers.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::TemporalEffect { treatment, outcome, control, active, .. } => {
            validate_ids([*treatment, *outcome], variable_count)?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::Counterfactual { outcomes, interventions, .. } => {
            validate_ids(outcomes.iter().copied(), variable_count)?;
            for intervention in interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            Ok(())
        }
        Q::AnomalyAttribution { targets, .. } | Q::MechanismChange { targets, .. } => {
            validate_ids(targets.iter().copied(), variable_count)
        }
        Q::ChangeAttribution { outcome, .. } | Q::UnitChange { outcome, .. } => {
            validate_id(*outcome, variable_count)
        }
        Q::Mediation { treatment, outcome, mediators, control, active, .. } => {
            validate_ids(
                [*treatment, *outcome].into_iter().chain(mediators.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::ConditionalEffect { inner } => validate_query_ids(inner, variable_count),
        Q::Distribution(wire) => {
            validate_ids(wire.outcomes.iter().chain(&wire.conditioning).copied(), variable_count)?;
            for intervention in &wire.interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            Ok(())
        }
        Q::PathSpecific(wire) => {
            validate_ids(
                [wire.treatment, wire.outcome].into_iter().chain(wire.path_nodes.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(&wire.control, variable_count)?;
            validate_intervention_ids(&wire.active, variable_count)
        }
        Q::Response(wire) => validate_response_query_ids(wire, variable_count),
        Q::Transport(wire) => {
            validate_response_query_ids(&wire.response, variable_count)?;
            validate_ids(wire.source_experiments.iter().copied(), variable_count)
        }
        Q::Interference(wire) => {
            let crate::InterferenceFunctionalWire::ExposureContrast { outcome, .. } =
                &wire.functional;
            validate_id(*outcome, variable_count)
        }
    }
}

fn require_finite(values: impl IntoIterator<Item = f64>, context: &str) -> Result<(), IoError> {
    if values.into_iter().any(|value| !value.is_finite()) {
        return Err(IoError::Convert(format!("{context} must be finite")));
    }
    Ok(())
}

fn validate_response_value(value: &crate::ResponseValueWire) -> Result<usize, IoError> {
    use crate::ResponseValueWire;
    match value {
        ResponseValueWire::Scalar(value) => {
            require_finite([*value], "response scalar")?;
            Ok(1)
        }
        ResponseValueWire::Vector(values) => {
            if values.is_empty() {
                return Err(IoError::Convert("response vector must not be empty".into()));
            }
            require_finite(values.iter().copied(), "response vector")?;
            Ok(values.len())
        }
        ResponseValueWire::Surface { grid, dimension, mean } => {
            let dimension = usize::try_from(*dimension).map_err(|_| IoError::TooLarge)?;
            if dimension == 0 || grid.is_empty() || grid.len() % dimension != 0 {
                return Err(IoError::Convert(
                    "response surface grid requires complete rows with positive dimension".into(),
                ));
            }
            let rows = grid.len() / dimension;
            if mean.len() != rows {
                return Err(IoError::Convert(
                    "response surface mean length must equal the number of grid rows".into(),
                ));
            }
            require_finite(grid.iter().chain(mean).copied(), "response surface")?;
            Ok(rows)
        }
        ResponseValueWire::Jacobian { outcomes, treatments, values } => {
            let outcomes = usize::try_from(*outcomes).map_err(|_| IoError::TooLarge)?;
            let treatments = usize::try_from(*treatments).map_err(|_| IoError::TooLarge)?;
            let cells = outcomes.checked_mul(treatments).ok_or(IoError::TooLarge)?;
            if outcomes == 0 || treatments == 0 || values.len() != cells {
                return Err(IoError::Convert(
                    "response Jacobian values must match positive matrix dimensions".into(),
                ));
            }
            require_finite(values.iter().copied(), "response Jacobian")?;
            Ok(cells)
        }
        ResponseValueWire::Envelope(envelope) => {
            let dimension = usize::try_from(envelope.dimension).map_err(|_| IoError::TooLarge)?;
            // `dimension == 0` is the coordinate-free envelope: the identified set of a
            // functional that has no grid of its own, such as an average derivative or a
            // binary-IV ATE bound. The estimand already says what the rows mean, exactly as
            // it does for `Scalar` and `Vector`, so carrying an invented coordinate here
            // would be a second, contradictable source of truth.
            let rows = if dimension == 0 {
                if !envelope.grid.is_empty() || envelope.lower.is_empty() {
                    return Err(IoError::Convert(
                        "a coordinate-free response envelope carries no grid and at least one bound"
                            .into(),
                    ));
                }
                envelope.lower.len()
            } else {
                if envelope.grid.is_empty() || envelope.grid.len() % dimension != 0 {
                    return Err(IoError::Convert(
                        "response envelope grid requires complete rows with positive dimension"
                            .into(),
                    ));
                }
                envelope.grid.len() / dimension
            };
            if envelope.lower.len() != rows
                || envelope.upper.len() != rows
                || envelope.lower.iter().zip(&envelope.upper).any(|(lower, upper)| lower > upper)
            {
                return Err(IoError::Convert(
                    "response envelope bounds must be ordered and match the grid rows".into(),
                ));
            }
            require_finite(
                envelope.grid.iter().chain(&envelope.lower).chain(&envelope.upper).copied(),
                "response envelope",
            )?;
            Ok(rows)
        }
    }
}

fn validate_response_result(wire: &CausalResponseWire) -> Result<(), IoError> {
    let value_len = match &wire.estimate {
        crate::ResponseIdentificationWire::PointIdentified(value) => {
            let length = validate_response_value(value)?;
            validate_value_for_estimand(value, &wire.estimand, ValueRole::Determinate)?;
            Some(length)
        }
        crate::ResponseIdentificationWire::PartiallyIdentified(value) => {
            let length = validate_response_value(value)?;
            validate_value_for_estimand(value, &wire.estimand, ValueRole::IdentifiedSet)?;
            Some(length)
        }
        crate::ResponseIdentificationWire::GraphDependent(values) => {
            if values.is_empty() {
                return Err(IoError::Convert(
                    "graph-dependent response must contain at least one graph value".into(),
                ));
            }
            let mut graph_ids = HashSet::with_capacity(values.len());
            let mut length = None;
            for (graph_id, value) in values {
                if !graph_ids.insert(*graph_id) {
                    return Err(IoError::Convert(format!(
                        "duplicate graph-dependent response id {graph_id}"
                    )));
                }
                let current = validate_response_value(value)?;
                validate_value_for_estimand(value, &wire.estimand, ValueRole::Determinate)?;
                if length.replace(current).is_some_and(|previous| previous != current) {
                    return Err(IoError::Convert(
                        "graph-dependent response values must have a common shape".into(),
                    ));
                }
            }
            length
        }
        crate::ResponseIdentificationWire::Unidentified { certificate } => {
            if certificate.trim().is_empty() {
                return Err(IoError::Convert(
                    "unidentified response certificate must be non-blank".into(),
                ));
            }
            None
        }
    };
    let status_matches = matches!(
        (&wire.identification_status, &wire.estimate),
        (
            crate::IdentificationStatusWire::NonparametricallyIdentified
                | crate::IdentificationStatusWire::IdentifiedUnderParametricRestrictions
                | crate::IdentificationStatusWire::IdentifiedUnderPriorRestrictions,
            crate::ResponseIdentificationWire::PointIdentified(_)
        ) | (
            crate::IdentificationStatusWire::PartiallyIdentified,
            crate::ResponseIdentificationWire::PartiallyIdentified(_)
        ) | (
            crate::IdentificationStatusWire::GraphDependent,
            crate::ResponseIdentificationWire::GraphDependent(_)
        ) | (
            crate::IdentificationStatusWire::NotIdentified,
            crate::ResponseIdentificationWire::Unidentified { .. }
        )
    );
    if !status_matches {
        return Err(IoError::Convert(
            "response identification status does not match its identification payload".into(),
        ));
    }
    validate_uncertainty(&wire.uncertainty, value_len)?;
    validate_support(&wire.support, support_dimension(&wire.estimand))?;
    if wire.provenance_id.trim().is_empty() {
        return Err(IoError::Convert("response provenance id must be non-blank".into()));
    }
    Ok(())
}

/// Materialize the declared estimand grid exactly as the estimator does.
///
/// The grid stored in a response value is produced by [`GridSpec::values`], and the
/// comparison below is bit-exact. Re-deriving a linspace with a different but
/// algebraically equivalent expression rounds differently and would reject valid
/// artifacts, so this delegates to the same routine rather than reimplementing it.
fn estimand_grid(grid: &crate::GridSpecWire) -> Result<Vec<f64>, IoError> {
    crate::response_wire::grid_from_wire(grid)?
        .values()
        .map_err(|error| IoError::Convert(error.to_string()))
}

/// Which identification reading a response value sits under.
///
/// An envelope is a set-valued answer, so it is the payload of a *partially identified*
/// response and of nothing else. A point-identified or per-graph value arriving as an
/// envelope would report an interval where its own status promises a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueRole {
    /// Point-identified, or one graph's value in a graph-dependent set.
    Determinate,
    /// The single payload of a partially identified response.
    IdentifiedSet,
}

fn validate_value_for_estimand(
    value: &crate::ResponseValueWire,
    estimand: &crate::ResponseFunctionalWire,
    role: ValueRole,
) -> Result<(), IoError> {
    use crate::{ResponseFunctionalWire as F, ResponseValueWire as V};
    if matches!(value, V::Envelope(_)) && role != ValueRole::IdentifiedSet {
        return Err(IoError::Convert(
            "a response envelope is the payload of a partially identified response only".into(),
        ));
    }
    let matches = match (estimand, value) {
        (F::MeanCurve { treatment, .. }, V::Surface { grid, dimension: 1, mean }) => {
            let expected = estimand_grid(&treatment.grid)?;
            grid == &expected && mean.len() == expected.len()
        }
        (F::MeanCurve { treatment, .. }, V::Envelope(envelope)) => {
            let expected = estimand_grid(&treatment.grid)?;
            envelope.dimension == 1
                && envelope.grid == expected
                && envelope.lower.len() == expected.len()
                && envelope.upper.len() == expected.len()
        }
        (
            F::AverageDerivative { .. }
            | F::PointDerivative { .. }
            | F::InterventionResponse { .. },
            V::Scalar(_),
        ) => true,
        // The identified set of a scalar functional: one coordinate-free interval. This is
        // the shape a bound such as Balke-Pearl produces, and format 0.3 stores it rather
        // than forcing a later format bump to say "this effect lies in [l, u]".
        (
            F::AverageDerivative { .. }
            | F::PointDerivative { .. }
            | F::InterventionResponse { .. },
            V::Envelope(envelope),
        ) => envelope.dimension == 0 && envelope.lower.len() == 1,
        (F::DirectionalDerivative { outcomes, .. }, V::Vector(values)) => {
            values.len() == outcomes.len()
        }
        (F::DirectionalDerivative { outcomes, .. }, V::Envelope(envelope)) => {
            envelope.dimension == 0 && envelope.lower.len() == outcomes.len()
        }
        (
            F::Jacobian { outcomes, treatments, .. },
            V::Jacobian { outcomes: value_outcomes, treatments: value_treatments, .. },
        ) => {
            usize::try_from(*value_outcomes).ok() == Some(outcomes.len())
                && usize::try_from(*value_treatments).ok() == Some(treatments.len())
        }
        // A Jacobian envelope is deliberately not accepted. `ResponseEnvelopeWire` has no
        // outcome/treatment extents, so its rows could only be checked against a row-major
        // convention this format has never stated and no estimator has ever written. That
        // needs a wire shape of its own, not an unvalidated reinterpretation of this one.
        _ => false,
    };
    if !matches {
        return Err(IoError::Convert(
            "response value shape/type does not match the declared estimand".into(),
        ));
    }
    Ok(())
}

fn support_dimension(estimand: &crate::ResponseFunctionalWire) -> usize {
    match estimand {
        crate::ResponseFunctionalWire::DirectionalDerivative { treatments, .. }
        | crate::ResponseFunctionalWire::Jacobian { treatments, .. } => treatments.len(),
        crate::ResponseFunctionalWire::InterventionResponse { interventions, .. } => {
            interventions.len()
        }
        _ => 1,
    }
}

fn validate_uncertainty(
    uncertainty: &crate::ResponseUncertaintyWire,
    value_len: Option<usize>,
) -> Result<(), IoError> {
    use crate::ResponseUncertaintyWire;
    let validate_band = |level: f64, lower: &[f64], upper: &[f64]| -> Result<(), IoError> {
        if !level.is_finite() || !(0.0..1.0).contains(&level) || level == 0.0 {
            return Err(IoError::Convert(
                "response uncertainty level must lie strictly inside (0,1)".into(),
            ));
        }
        if lower.is_empty()
            || lower.len() != upper.len()
            || value_len.is_some_and(|expected| expected != lower.len())
            || lower.iter().zip(upper).any(|(lo, hi)| lo > hi)
        {
            return Err(IoError::Convert(
                "response uncertainty bounds must be ordered and match the response shape".into(),
            ));
        }
        require_finite(lower.iter().chain(upper).copied(), "response uncertainty bounds")
    };
    match uncertainty {
        ResponseUncertaintyWire::None => Ok(()),
        ResponseUncertaintyWire::Scalar { standard_error, level, lower, upper } => {
            if value_len != Some(1) || !standard_error.is_finite() || *standard_error < 0.0 {
                return Err(IoError::Convert(
                    "scalar uncertainty requires a scalar response and non-negative finite standard error"
                        .into(),
                ));
            }
            validate_band(*level, &[*lower], &[*upper])
        }
        ResponseUncertaintyWire::PointwiseBand { level, lower, upper }
        | ResponseUncertaintyWire::IdentifiedEnvelopeBand {
            level,
            lower_outer: lower,
            upper_outer: upper,
        } => validate_band(*level, lower, upper),
        ResponseUncertaintyWire::SimultaneousBand { level, lower, upper, replicates } => {
            if *replicates == 0 {
                return Err(IoError::Convert(
                    "simultaneous response band requires positive replicates".into(),
                ));
            }
            validate_band(*level, lower, upper)
        }
        ResponseUncertaintyWire::Posterior { artifact_id } => {
            if artifact_id.trim().is_empty() {
                return Err(IoError::Convert(
                    "response posterior artifact id must be non-blank".into(),
                ));
            }
            Ok(())
        }
    }
}

fn validate_support(
    support: &crate::SupportReportWire,
    expected_dimension: usize,
) -> Result<(), IoError> {
    let minima = &support.query_region.minima;
    let maxima = &support.query_region.maxima;
    if minima.len() != expected_dimension
        || maxima.len() != expected_dimension
        || minima.iter().zip(maxima).any(|(minimum, maximum)| minimum > maximum)
    {
        return Err(IoError::Convert(
            "response support bounds must be ordered and have equal dimensions".into(),
        ));
    }
    require_finite(minima.iter().chain(maxima).copied(), "response support bounds")?;
    for diagnostic in &support.diagnostics {
        if diagnostic.id.trim().is_empty() || diagnostic.detail.trim().is_empty() {
            return Err(IoError::Convert(
                "response support diagnostics require non-blank id and detail".into(),
            ));
        }
        require_finite(diagnostic.values.iter().copied(), "response support diagnostic")?;
    }
    Ok(())
}

fn validate_factor_ids(
    factor: &crate::PopulationFactorWire,
    variable_count: usize,
) -> Result<(), IoError> {
    if factor.population.trim().is_empty() {
        return Err(IoError::Convert("transport factor population must be non-blank".into()));
    }
    validate_ids(
        factor.variables.iter().chain(&factor.conditioned_on).chain(&factor.interventions).copied(),
        variable_count,
    )
}

fn validate_transport_identification_ids(
    identification: &TransportIdentificationWire,
    variable_count: usize,
) -> Result<(), IoError> {
    match identification {
        TransportIdentificationWire::Transportable { formula, certificate } => {
            use crate::TransportFormulaWire;
            match formula.as_ref() {
                TransportFormulaWire::Direct(factor) => {
                    validate_factor_ids(factor, variable_count)?;
                }
                TransportFormulaWire::Standardize { over, source_response, target_law } => {
                    validate_ids(over.iter().copied(), variable_count)?;
                    validate_factor_ids(source_response, variable_count)?;
                    validate_factor_ids(target_law, variable_count)?;
                }
                TransportFormulaWire::RecursiveFactorization { sum_out, factors } => {
                    validate_ids(sum_out.iter().copied(), variable_count)?;
                    for factor in factors {
                        validate_factor_ids(factor, variable_count)?;
                    }
                }
            }
            validate_ids(certificate.selection_targets.iter().copied(), variable_count)
        }
        TransportIdentificationWire::NotCertified(certificate) => {
            validate_ids(certificate.witness.iter().copied(), variable_count)
        }
    }
}

fn validate_transport_identification(
    identification: &TransportIdentificationWire,
) -> Result<(), IoError> {
    match identification {
        TransportIdentificationWire::Transportable { certificate, .. } => {
            if certificate.rule.trim().is_empty()
                || certificate.premises.iter().any(|premise| premise.trim().is_empty())
            {
                return Err(IoError::Convert(
                    "transport certificate rule and premises must be non-blank".into(),
                ));
            }
        }
        TransportIdentificationWire::NotCertified(certificate) => {
            if certificate.reason.trim().is_empty() || certificate.message.trim().is_empty() {
                return Err(IoError::Convert(
                    "non-transportable certificate reason and message must be non-blank".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_overlap(diagnostic: crate::TransportOverlapDiagnosticWire) -> Result<(), IoError> {
    require_finite(
        [diagnostic.probability_min, diagnostic.probability_max, diagnostic.effective_sample_size],
        "transport overlap diagnostic",
    )?;
    if diagnostic.probability_min < 0.0
        || diagnostic.probability_max > 1.0
        || diagnostic.probability_min > diagnostic.probability_max
        || diagnostic.effective_sample_size < 0.0
    {
        return Err(IoError::Convert(
            "transport overlap probabilities/ESS are outside their valid domains".into(),
        ));
    }
    Ok(())
}

fn validate_transport_estimate(estimate: &TransportEffectEstimateWire) -> Result<(), IoError> {
    require_finite(
        [Some(estimate.ipw), estimate.aipw].into_iter().flatten(),
        "transport estimate",
    )?;
    validate_overlap(estimate.selection_overlap)?;
    validate_overlap(estimate.treatment_overlap)
}

fn validate_probability_method(
    method: crate::ExposureProbabilityMethodWire,
) -> Result<(), IoError> {
    if let crate::ExposureProbabilityMethodWire::MonteCarlo { draws, .. } = method {
        if draws == 0 {
            return Err(IoError::Convert(
                "interference Monte Carlo probability method requires positive draws".into(),
            ));
        }
    }
    Ok(())
}

fn validate_interference_estimate(estimate: &InterferenceEstimateWire) -> Result<(), IoError> {
    require_finite(
        [
            estimate.contrast.horvitz_thompson,
            estimate.contrast.hajek,
            estimate.contrast.conservative_variance,
            estimate.minimum_exposure_probability,
        ],
        "interference estimate",
    )?;
    if estimate.contrast.conservative_variance < 0.0
        || estimate.minimum_exposure_probability <= 0.0
        || estimate.minimum_exposure_probability > 1.0
    {
        return Err(IoError::Convert(
            "interference variance/probability are outside their valid domains".into(),
        ));
    }
    validate_probability_method(estimate.from_probability_method)?;
    validate_probability_method(estimate.to_probability_method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FormatVersion, IdentificationStatusWire, ResponseFunctionalWire,
        ResponseIdentificationWire, ResponseQueryWire, ResponseUncertaintyWire, ResponseValueWire,
        SupportRegionWire, SupportReportWire, SupportStatusWire,
    };

    fn response_query() -> CausalQueryWire {
        CausalQueryWire::Response(ResponseQueryWire {
            functional: ResponseFunctionalWire::AverageDerivative {
                outcome: 1,
                treatment: 0,
                weighting: crate::DerivativeWeightingWire::Observed,
            },
            target_population: crate::TargetPopulationWire::AllObserved,
            observation: crate::ObservationSpecWire::Complete,
            observation_assumptions: Vec::new(),
        })
    }

    fn unchecked_artifact(
        payload: &CausalPayloadWire,
        header: &CausalPayloadHeader,
    ) -> EncodedArtifact {
        let valid_payload = CausalPayloadWire::Query(Box::new(response_query()));
        let mut artifact = encode_causal_payload_artifact(
            &valid_payload,
            vec!["a".into(), "y".into()],
            "malformed",
        )
        .unwrap();
        let header_bytes = to_cbor(&header).unwrap();
        let payload_bytes = match payload {
            CausalPayloadWire::Query(value) => to_cbor(value).unwrap(),
            CausalPayloadWire::ResponseResult(value) => to_cbor(value).unwrap(),
            CausalPayloadWire::TransportIdentification(value) => to_cbor(value).unwrap(),
            CausalPayloadWire::TransportEstimate(value) => to_cbor(value).unwrap(),
            CausalPayloadWire::InterferenceEstimate(value) => to_cbor(value).unwrap(),
        };
        artifact.sections = vec![
            SectionBytes::new(HEADER_SECTION, header_bytes.clone()),
            SectionBytes::new(PAYLOAD_SECTION, payload_bytes.clone()),
        ];
        artifact.manifest.sections = vec![
            section_descriptor(HEADER_SECTION, "application/cbor", &header_bytes),
            section_descriptor(PAYLOAD_SECTION, "application/cbor", &payload_bytes),
        ];
        artifact
    }

    fn decode(artifact: &EncodedArtifact) -> Result<CausalPayloadWire, IoError> {
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        decode_causal_payload_artifact(&bytes).map(|(_, _, payload)| payload)
    }

    fn container_pass_through_preserves_current_query(source_minor: u16) {
        let query = CausalQueryWire::Response(ResponseQueryWire {
            functional: ResponseFunctionalWire::AverageDerivative {
                outcome: 1,
                treatment: 0,
                weighting: crate::DerivativeWeightingWire::Observed,
            },
            target_population: crate::TargetPopulationWire::AllObserved,
            observation: crate::ObservationSpecWire::Complete,
            observation_assumptions: Vec::new(),
        });
        let payload = CausalPayloadWire::Query(Box::new(query.clone()));
        let mut artifact =
            encode_causal_payload_artifact(&payload, vec!["a".into(), "y".into()], "q").unwrap();
        artifact.manifest.format_version = FormatVersion { major: 0, minor: source_minor };
        artifact.manifest.minimum_reader_version = FormatVersion { major: 0, minor: source_minor };
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (migrated, header, decoded) = decode_causal_payload_artifact(&bytes).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(header.variable_names, ["a", "y"]);
        assert_eq!(decoded, CausalPayloadWire::Query(Box::new(query)));
    }

    #[test]
    fn current_query_survives_synthetic_0_1_container_pass_through() {
        container_pass_through_preserves_current_query(1);
    }

    #[test]
    fn current_query_survives_synthetic_0_2_container_pass_through() {
        container_pass_through_preserves_current_query(2);
    }

    #[test]
    fn decode_rejects_blank_and_duplicate_variable_names() {
        let payload = CausalPayloadWire::Query(Box::new(response_query()));
        for names in [vec!["a".into(), " ".into()], vec!["a".into(), "a".into()]] {
            let artifact = unchecked_artifact(
                &payload,
                &CausalPayloadHeader {
                    payload_kind: CausalPayloadKind::Query,
                    variable_names: names,
                },
            );
            assert!(matches!(decode(&artifact), Err(IoError::Convert(_))));
        }
    }

    #[test]
    fn decode_rejects_out_of_header_variable_id() {
        let query = CausalQueryWire::Response(ResponseQueryWire {
            functional: ResponseFunctionalWire::AverageDerivative {
                outcome: 2,
                treatment: 0,
                weighting: crate::DerivativeWeightingWire::Observed,
            },
            target_population: crate::TargetPopulationWire::AllObserved,
            observation: crate::ObservationSpecWire::Complete,
            observation_assumptions: Vec::new(),
        });
        let artifact = unchecked_artifact(
            &CausalPayloadWire::Query(Box::new(query)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::Query,
                variable_names: vec!["a".into(), "y".into()],
            },
        );
        assert!(decode(&artifact).unwrap_err().to_string().contains("outside"));
    }

    #[test]
    fn decode_rejects_domain_invalid_query() {
        let query = CausalQueryWire::Response(ResponseQueryWire {
            functional: ResponseFunctionalWire::AverageDerivative {
                outcome: 0,
                treatment: 0,
                weighting: crate::DerivativeWeightingWire::Observed,
            },
            target_population: crate::TargetPopulationWire::AllObserved,
            observation: crate::ObservationSpecWire::Complete,
            observation_assumptions: Vec::new(),
        });
        let artifact = unchecked_artifact(
            &CausalPayloadWire::Query(Box::new(query)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::Query,
                variable_names: vec!["a".into()],
            },
        );
        assert!(matches!(decode(&artifact), Err(IoError::Convert(_))));
    }

    #[test]
    fn linspace_response_surface_round_trips_bit_exactly() {
        // The grid a response carries is `GridSpec::values()` output. The validator
        // must materialize the identical floats, not an algebraically equal formula:
        // `start + i * step` and `start + (end - start) * i / (points - 1)` disagree
        // in the last ulp for most triples, and the comparison is exact.
        let points = 7usize;
        let grid =
            antecedent_core::GridSpec::Linspace { start: 0.0, end: 1.0, points }.values().unwrap();
        let response = CausalResponseWire {
            estimand: ResponseFunctionalWire::MeanCurve {
                outcome: 1,
                treatment: crate::ContinuousDomainWire {
                    variable: 0,
                    grid: crate::GridSpecWire::Linspace {
                        start: 0.0,
                        end: 1.0,
                        points: points as u64,
                    },
                },
            },
            identification_status: IdentificationStatusWire::NonparametricallyIdentified,
            estimate: ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
                grid: grid.clone(),
                dimension: 1,
                mean: vec![0.0; points],
            }),
            uncertainty: ResponseUncertaintyWire::None,
            support: SupportReportWire {
                status: SupportStatusWire::Supported,
                query_region: SupportRegionWire { minima: vec![0.0], maxima: vec![1.0] },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
            },
            assumptions: Vec::new(),
            provenance_id: "estimate.response.kennedy_dr".into(),
        };
        let payload = CausalPayloadWire::ResponseResult(Box::new(response));
        let artifact = encode_causal_payload_artifact(
            &payload,
            vec!["a".into(), "y".into()],
            "estimate.response.kennedy_dr",
        )
        .unwrap();
        assert_eq!(decode(&artifact).unwrap(), payload);
    }

    fn partially_identified_scalar(value: ResponseValueWire) -> CausalResponseWire {
        CausalResponseWire {
            estimand: ResponseFunctionalWire::AverageDerivative {
                outcome: 1,
                treatment: 0,
                weighting: crate::DerivativeWeightingWire::Observed,
            },
            identification_status: IdentificationStatusWire::PartiallyIdentified,
            estimate: ResponseIdentificationWire::PartiallyIdentified(value),
            uncertainty: ResponseUncertaintyWire::None,
            support: SupportReportWire {
                status: SupportStatusWire::Supported,
                query_region: SupportRegionWire { minima: vec![0.0], maxima: vec![1.0] },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
            },
            assumptions: Vec::new(),
            provenance_id: "identify.binary_iv_bounds".into(),
        }
    }

    fn scalar_interval(lower: f64, upper: f64) -> ResponseValueWire {
        ResponseValueWire::Envelope(crate::ResponseEnvelopeWire {
            grid: Vec::new(),
            dimension: 0,
            lower: vec![lower],
            upper: vec![upper],
        })
    }

    #[test]
    fn partially_identified_scalar_functional_stores_a_coordinate_free_interval() {
        // The shape a bound such as Balke-Pearl produces. Format 0.3 has to be able to say
        // "this effect lies in [l, u]" for a scalar functional, or saying it later costs a
        // format version and a migration.
        let payload = CausalPayloadWire::ResponseResult(Box::new(partially_identified_scalar(
            scalar_interval(-0.19, 0.01),
        )));
        let artifact = encode_causal_payload_artifact(
            &payload,
            vec!["a".into(), "y".into()],
            "identify.binary_iv_bounds",
        )
        .unwrap();
        assert_eq!(decode(&artifact).unwrap(), payload);
    }

    #[test]
    fn point_identified_response_may_not_carry_an_envelope() {
        // An interval under a status that promises a number is incoherent regardless of
        // which functional it decorates.
        let mut wire = partially_identified_scalar(scalar_interval(-0.19, 0.01));
        wire.identification_status = IdentificationStatusWire::NonparametricallyIdentified;
        wire.estimate = ResponseIdentificationWire::PointIdentified(scalar_interval(-0.19, 0.01));
        let artifact = unchecked_artifact(
            &CausalPayloadWire::ResponseResult(Box::new(wire)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::ResponseResult,
                variable_names: vec!["a".into(), "y".into()],
            },
        );
        assert!(
            decode(&artifact)
                .unwrap_err()
                .to_string()
                .contains("partially identified response only")
        );
    }

    #[test]
    fn a_coordinate_free_envelope_still_needs_a_bound_and_no_grid() {
        for broken in [
            crate::ResponseEnvelopeWire {
                grid: vec![0.0],
                dimension: 0,
                lower: vec![-1.0],
                upper: vec![1.0],
            },
            crate::ResponseEnvelopeWire {
                grid: Vec::new(),
                dimension: 0,
                lower: Vec::new(),
                upper: Vec::new(),
            },
        ] {
            let wire = partially_identified_scalar(ResponseValueWire::Envelope(broken));
            let artifact = unchecked_artifact(
                &CausalPayloadWire::ResponseResult(Box::new(wire)),
                &CausalPayloadHeader {
                    payload_kind: CausalPayloadKind::ResponseResult,
                    variable_names: vec!["a".into(), "y".into()],
                },
            );
            assert!(matches!(decode(&artifact), Err(IoError::Convert(_))));
        }
    }

    #[test]
    fn jacobian_envelopes_remain_unrepresentable() {
        // Recorded boundary, not an oversight: the envelope wire has no outcome/treatment
        // extents, so its rows cannot be checked against the matrix the estimand declares.
        let mut wire = partially_identified_scalar(scalar_interval(-1.0, 1.0));
        wire.estimand = ResponseFunctionalWire::Jacobian {
            outcomes: vec![1],
            treatments: vec![0],
            at: vec![0.0],
            scale: crate::DerivativeScaleWire::Identity,
        };
        let artifact = unchecked_artifact(
            &CausalPayloadWire::ResponseResult(Box::new(wire)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::ResponseResult,
                variable_names: vec!["a".into(), "y".into()],
            },
        );
        assert!(decode(&artifact).unwrap_err().to_string().contains("does not match"));
    }

    #[test]
    fn decode_rejects_response_shape_mismatch() {
        let response = CausalResponseWire {
            estimand: ResponseFunctionalWire::MeanCurve {
                outcome: 1,
                treatment: crate::ContinuousDomainWire {
                    variable: 0,
                    grid: crate::GridSpecWire::Values(vec![0.0, 1.0]),
                },
            },
            identification_status: IdentificationStatusWire::NonparametricallyIdentified,
            estimate: ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
                grid: vec![0.0, 1.0],
                dimension: 1,
                mean: vec![1.0],
            }),
            uncertainty: ResponseUncertaintyWire::None,
            support: SupportReportWire {
                status: SupportStatusWire::Supported,
                query_region: SupportRegionWire { minima: vec![0.0], maxima: vec![1.0] },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
            },
            assumptions: Vec::new(),
            provenance_id: "estimate.response.kennedy_dr".into(),
        };
        let artifact = unchecked_artifact(
            &CausalPayloadWire::ResponseResult(Box::new(response)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::ResponseResult,
                variable_names: vec!["a".into(), "y".into()],
            },
        );
        assert!(decode(&artifact).unwrap_err().to_string().contains("surface mean length"));
    }

    #[test]
    fn decode_rejects_transport_certificate_id_outside_header() {
        let identification =
            TransportIdentificationWire::NotCertified(crate::NonTransportableCertificateWire {
                reason: "not_certified".into(),
                witness: vec![2],
                message: "implemented rules did not certify transport".into(),
            });
        let artifact = unchecked_artifact(
            &CausalPayloadWire::TransportIdentification(Box::new(identification)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::TransportIdentification,
                variable_names: vec!["a".into(), "y".into()],
            },
        );
        assert!(decode(&artifact).unwrap_err().to_string().contains("outside"));
    }

    #[test]
    fn decode_rejects_invalid_interference_estimate() {
        let estimate = InterferenceEstimateWire {
            contrast: crate::RandomizationContrastWire {
                horvitz_thompson: 1.0,
                hajek: 1.0,
                conservative_variance: -1.0,
            },
            from_probability_method: crate::ExposureProbabilityMethodWire::Exact,
            to_probability_method: crate::ExposureProbabilityMethodWire::Exact,
            minimum_exposure_probability: 0.2,
        };
        let artifact = unchecked_artifact(
            &CausalPayloadWire::InterferenceEstimate(Box::new(estimate)),
            &CausalPayloadHeader {
                payload_kind: CausalPayloadKind::InterferenceEstimate,
                variable_names: Vec::new(),
            },
        );
        assert!(matches!(decode(&artifact), Err(IoError::Convert(_))));
    }
}
