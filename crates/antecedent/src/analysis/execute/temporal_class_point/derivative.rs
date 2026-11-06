//! Temporal class derivative evaluation and response assembly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, CausalQuery, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EstimatorId, GraphClass, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    SkippedCompletions, SlopeAtom, SlopeSource, TemporalEffectQuery, TimeSeriesData,
    derivative_response_finite, envelope_to_identification_result_for,
    identification_status_ok_for_case, identify_cached_diagnostic, mix_derivative_responses,
    point_uncertainty, refuse_unread_series_weights, require_identified_mass, select_estimand,
};

/// The diagnostics of a class derivative response: the envelope, cache
/// provenance, the derivative note and the support warnings.
pub(super) fn class_derivative_diagnostics(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
    graph_class: GraphClass,
    identify_cached: bool,
    response: &CausalResponse,
) -> Vec<Diagnostic> {
    let mut diagnostics = vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
        envelope,
        graph_class,
    )];
    if identify_cached {
        diagnostics.push(identify_cached_diagnostic());
    }
    diagnostics.push(derivative_note());
    diagnostics.extend(response.support.warnings.iter().cloned());
    diagnostics
}

pub(super) fn derivative_note() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal.linear_derivative",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted slope of the linear unfolded temporal fit; \
         its band refits every completion's outcome regressions",
    )
}

/// The weighted completion derivative responses with their recorded regressions,
/// the typed values by completion and the first completion's estimand.
pub(super) struct ClassDerivatives<'a> {
    pub(super) atoms: Vec<SlopeAtom<'a>>,
    pub(super) responses: Vec<(f64, CausalResponse)>,
    pub(super) values: Vec<Option<ResponseValue>>,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) skipped: SkippedCompletions,
}

/// The linear slope of each identified completion with its weight, the
/// finite typed values by completion, and the first completion's estimand.
pub(super) fn class_derivative_responses<'a>(
    data: &'a TimeSeriesData,
    query: &ResponseQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    offsets: (i32, i32),
    (policy, prior_scale): (antecedent_core::KernelPolicy, Option<f64>),
) -> Result<ClassDerivatives<'a>, CausalError> {
    refuse_unread_series_weights(data, "the class-aware derivative")?;
    let envelope = &bundle.envelope;
    require_identified_mass(envelope)?;
    let mut atoms = Vec::new();
    let mut responses = Vec::new();
    let mut values = vec![None; envelope.cases.len()];
    let mut primary_estimand = None;
    let mut skipped = SkippedCompletions::default();
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let atom = SlopeAtom::new(data, &estimand, indexer, offsets, policy, prior_scale);
        let response = match super::super::temporal_path::temporal_linear_derivative_response(
            data,
            query,
            &case.result,
            &atom,
        ) {
            Ok(response) => response,
            Err(error) => {
                skipped.record(error)?;
                continue;
            }
        };
        values[index] = match &response.estimate {
            ResponseIdentification::PointIdentified(value)
                if derivative_response_finite(&response) =>
            {
                Some(value.clone())
            }
            _ => None,
        };
        if primary_estimand.is_none() {
            primary_estimand = Some(estimand);
        }
        atoms.push(atom);
        responses.push((case.weight.0, response));
    }
    let estimand = primary_estimand.ok_or_else(|| skipped.none_estimable())?;
    Ok(ClassDerivatives { atoms, responses, values, estimand, skipped })
}

/// Retain each completion's actual derivative shape. The class mass accounting
/// is shared with scalar effects; its scalar marker is replaced before the
/// mixture is published and is never used as a bound for a vector response.
pub(super) fn derivative_structural_mixture(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
    values: &[Option<ResponseValue>],
) -> crate::result::StructuralResponseMixture {
    let scalar: Vec<Option<f64>> = values
        .iter()
        .map(|value| match value {
            Some(ResponseValue::Scalar(value)) => Some(*value),
            Some(ResponseValue::Vector(_) | ResponseValue::Jacobian { .. }) => Some(0.0),
            _ => None,
        })
        .collect();
    let mut mixture = super::super::temporal_path::temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &scalar,
    );
    let vector = values
        .iter()
        .flatten()
        .any(|value| matches!(value, ResponseValue::Vector(_) | ResponseValue::Jacobian { .. }));
    for (atom, value) in mixture.atoms.iter_mut().zip(values) {
        atom.value.clone_from(value);
    }
    if vector {
        mixture.identified_set = None;
    }
    mixture
}

/// A graph-dependent status cannot carry a point payload. Retain each
/// completion's value instead of attributing their weighted mean to all graphs.
pub(super) fn retain_graph_dependent_derivatives(
    response: &mut CausalResponse,
    structural: &crate::result::StructuralResponseMixture,
) {
    if response.identification_status == IdentificationStatus::GraphDependent {
        response.estimate = ResponseIdentification::GraphDependent(
            structural
                .atoms
                .iter()
                .filter_map(|atom| atom.value.clone().map(|value| (atom.graph_key, value)))
                .collect(),
        );
        response.uncertainty = ResponseUncertainty::None;
    }
}

/// The derivative of every replicate or draw of one completion.
pub(super) fn derivative_evaluator(
    functional: &ResponseFunctional,
) -> impl Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError> + '_ {
    move |_: usize, source: &dyn SlopeSource| {
        let value =
            super::super::temporal_path::temporal_linear_derivative_value(functional, source)?;
        point_uncertainty::response_values(&value).ok_or_else(|| CausalError::Compile {
            message: "temporal derivative replicate has no numeric value".into(),
        })
    }
}

/// The flattened point of a mixed derivative response.
pub(super) fn response_center(response: &CausalResponse) -> Result<Vec<f64>, CausalError> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(value) => point_uncertainty::response_values(value),
        _ => None,
    }
    .ok_or_else(|| CausalError::Compile {
        message: "temporal class derivative point has no numeric value".into(),
    })
}

pub(super) fn derivative_probe(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) =
        super::super::response_path::response_primary_pair(&query.functional)?;
    Ok(super::super::temporal_path::unit_contrast_query(
        temporal,
        (treatment, outcome),
        temporal.horizons[0],
        query.target_population.clone(),
    ))
}

/// One class envelope's mixed derivative with its completions' recorded
/// regressions and weights, estimand, identification, diagnostics, structural
/// mixture and status.
pub(super) struct ClassDerivative<'a> {
    pub(super) response: CausalResponse,
    pub(super) atoms: Vec<SlopeAtom<'a>>,
    pub(super) weights: Vec<f64>,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) identification: IdentificationResult,
    pub(super) diagnostics: Vec<Diagnostic>,
    pub(super) structural: crate::result::StructuralResponseMixture,
    pub(super) status: IdentificationStatus,
    pub(super) skipped: SkippedCompletions,
}

pub(super) fn derivative_on_envelope<'a>(
    data: &'a TimeSeriesData,
    query: &ResponseQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    graph_class: GraphClass,
    (offsets, policy, prior_scale): ((i32, i32), antecedent_core::KernelPolicy, Option<f64>),
) -> Result<ClassDerivative<'a>, CausalError> {
    let envelope = &bundle.envelope;
    let ClassDerivatives { atoms, responses, values, estimand, skipped } =
        class_derivative_responses(data, query, bundle, offsets, (policy, prior_scale))?;
    let weights: Vec<f64> = responses.iter().map(|(weight, _)| *weight).collect();
    let response = mix_derivative_responses(&responses)?;
    let mut diagnostics = vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
        envelope,
        graph_class,
    )];
    diagnostics.push(derivative_note());
    diagnostics.extend(response.support.warnings.iter().cloned());
    let identification =
        envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
    let structural = derivative_structural_mixture(envelope, &values);
    Ok(ClassDerivative {
        response,
        atoms,
        weights,
        estimand,
        identification,
        diagnostics,
        structural,
        status: envelope.status,
        skipped,
    })
}
