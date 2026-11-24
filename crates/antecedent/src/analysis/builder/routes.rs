//! Builder admission for transport, interference and panel routes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, ResponseQuery, TemporalEffectQuery};
use antecedent_data::{DiscoveryEstimationSplit, PanelData};

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;

/// Refuse a transport construction outside the supported cell.
///
/// The supported `TransportQuery` cell transports a mean `ResponseCurve` on the
/// complete, all-observed, static response by binary trial-to-target IPW. A
/// derivative, a non-mean outcome functional, an embedded population, an
/// observation mechanism or a temporal attachment would be labelled with the
/// executed binary contrast it is not, so it is refused.
pub(in crate::analysis) fn refuse_unsupported_transport(
    query: &antecedent_core::TransportQuery,
) -> Result<(), CausalError> {
    let response = &query.response;
    let supported =
        matches!(response.functional, antecedent_core::ResponseFunctional::MeanCurve { .. })
            && response.outcome_functional.is_mean()
            && matches!(response.target_population, antecedent_core::TargetPopulation::AllObserved)
            && matches!(response.observation, antecedent_core::ObservationSpec::Complete)
            && response.observation_assumptions.is_empty()
            && response.temporal.is_none();
    if supported {
        Ok(())
    } else {
        Err(crate::unsupported_reason!(
            "construction_not_supported",
            "TransportQuery is supported for a mean ResponseCurve on the complete, all-observed \
             static response, transported by binary trial-to-target IPW"
        ))
    }
}

/// Refuse interference designs without a checked execution path.
///
/// Bernoulli and complete randomization run the design-based Horvitz-Thompson / Hajek contrast
/// with its Young bound under every built-in exposure mapping. The Bayesian route models only
/// Bernoulli assignment with `NeighborCount` exposure, so the other designs refuse at build
/// under Bayesian inference, as before.
pub(in crate::analysis) fn refuse_unsupported_interference(
    query: &antecedent_core::InterferenceQuery,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    use antecedent_core::{AssignmentDesign, ExposureMapping};
    let built_in_mapping = matches!(
        query.exposure,
        ExposureMapping::OwnTreatment
            | ExposureMapping::NeighborCount
            | ExposureMapping::NeighborFraction
            | ExposureMapping::WeightedNeighborExposure
    );
    let bernoulli_count = matches!(query.assignment, AssignmentDesign::Bernoulli { .. })
        && matches!(query.exposure, ExposureMapping::NeighborCount);
    let design_based = matches!(
        query.assignment,
        AssignmentDesign::Bernoulli { .. } | AssignmentDesign::CompleteRandomization { .. }
    ) && built_in_mapping
        && (bernoulli_count || inference.is_frequentist());
    let neighbor_mapping = matches!(
        query.exposure,
        ExposureMapping::NeighborCount
            | ExposureMapping::NeighborFraction
            | ExposureMapping::WeightedNeighborExposure
    );
    let supported = design_based
        || (matches!(query.assignment, AssignmentDesign::ClusterRandomization { .. })
            && matches!(query.exposure, ExposureMapping::NeighborFraction))
        || (matches!(query.assignment, AssignmentDesign::TwoStageSaturation { .. })
            && neighbor_mapping)
        || (matches!(query.assignment, AssignmentDesign::ObservedExposure { .. })
            && neighbor_mapping);
    if supported {
        Ok(())
    } else {
        Err(crate::unsupported_reason!(
            "construction_not_supported",
            "InterferenceQuery supports Bernoulli or complete randomization with a built-in exposure mapping (Bayesian: Bernoulli/NeighborCount), cluster total effects, exact two-stage saturation, or supplied-propensity observational exposure with built-in neighbor mappings"
        ))
    }
}

/// Refusal for panel response on a non-temporal or static graph.
pub(crate) const PANEL_RESPONSE_CLASS_REFUSAL: &str = concat!(
    "panel ResponseCurve / InterventionResponse is supported on a supplied ",
    "TemporalDag, TemporalCpdag, or TemporalPag",
);

/// Refusal for a panel class multi-step Sustained run under a discovery split.
pub(crate) const PANEL_CLASS_SEQUENTIAL_SPLIT_REFUSAL: &str =
    "class-aware multi-step sustained requires no discovery-estimation split";

/// Refusal for a transferred or informative prior on panel class multi-step Sustained.
pub(crate) const PANEL_CLASS_SEQUENTIAL_PRIOR_REFUSAL: &str =
    "multi-step Sequence transfer stays refused on incomplete classes";

/// Single owner for panel data-route supports. Build, prepare, compile, and
/// execute consult this; they do not restate the same refusals.
///
/// Every panel route requires one sampling regularity across units: a horizon of
/// `h` steps must mean the same duration in every unit.
pub(crate) fn refuse_unsupported_panel_route(
    query: &CausalQuery,
    class: GraphClass,
    inference: &InferenceMode,
    panel: &PanelData,
    split: Option<&DiscoveryEstimationSplit>,
) -> Result<(), CausalError> {
    panel_shared_regularity(panel)?;
    match query {
        CausalQuery::Response(query) => refuse_unsupported_panel_response(query, class),
        CausalQuery::TemporalEffect(query) => {
            refuse_unsupported_panel_effect(query, class, inference, split)
        }
        _ => Ok(()),
    }
}

/// The one time-index regularity every panel unit shares.
///
/// # Errors
///
/// When the panel is empty or its units disagree on regularity.
pub(crate) fn panel_shared_regularity(
    panel: &PanelData,
) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first = &panel.unit(0).map_err(CausalError::rejected)?.series.time_index().regularity;
    if panel.units().iter().any(|unit| &unit.series.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "panel analysis requires every panel unit to share one time-index \
                      regularity: a horizon step would otherwise mean a different duration in \
                      different units; align the units first"
                .into(),
        });
    }
    Ok(first.clone())
}

fn refuse_unsupported_panel_response(
    query: &ResponseQuery,
    class: GraphClass,
) -> Result<(), CausalError> {
    if !query.is_temporal()
        || !matches!(
            class,
            GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
        )
    {
        return Err(CausalError::Unsupported { message: PANEL_RESPONSE_CLASS_REFUSAL });
    }
    Ok(())
}

fn refuse_unsupported_panel_effect(
    query: &TemporalEffectQuery,
    class: GraphClass,
    inference: &InferenceMode,
    split: Option<&DiscoveryEstimationSplit>,
) -> Result<(), CausalError> {
    if !class.is_incomplete_temporal() || !query.is_multi_step_sustained() {
        return Ok(());
    }
    // Panel class multi-step Sustained fits every unit's full series; a split's
    // discovery rows would be reused for estimation, and the per-unit sequential
    // g-computation has no mapping for a transferred coefficient prior.
    if split.is_some() {
        return Err(CausalError::Unsupported { message: PANEL_CLASS_SEQUENTIAL_SPLIT_REFUSAL });
    }
    if let InferenceMode::Bayesian(cfg) = inference
        && (cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some())
    {
        return Err(CausalError::Unsupported { message: PANEL_CLASS_SEQUENTIAL_PRIOR_REFUSAL });
    }
    Ok(())
}
