//! Reasoning slots of a contract and of an executed result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

/// Uncertainty component of a response interval by what its level means: a credible
/// interval is posterior parameter uncertainty, a confidence interval is sampling
/// uncertainty.
pub(super) fn response_interval_component(
    interpretation: IntervalInterpretation,
    credible: &'static str,
    confidence: &'static str,
) -> (UncertaintySource, &'static str) {
    match interpretation {
        IntervalInterpretation::Credible => (UncertaintySource::Parameter, credible),
        IntervalInterpretation::Confidence => (UncertaintySource::Sampling, confidence),
    }
}

pub(super) fn reasoning_view(
    study: &Study,
    prepared: Option<&PreparedStudy>,
    cached: Option<&IdentificationResult>,
    search_capped: bool,
) -> ReasoningView {
    let support = SupportSlot::new(
        study.support_status.map_or("off_axis", CellStatus::as_str),
        matrix_coordinate(study).map(Arc::from),
        SlotAvailability::unavailable("not_evaluated"),
    );
    let mut obligations = obligations_from_set(cached.map(|result| &result.required_assumptions));
    obligations.extend(observation_obligations(&study.query));
    let assumptions = AssumptionSlot::new(obligations);
    if prepared.is_some() && cached.is_none() && identifies_per_execution(study) {
        // Sharp RD identifies from the running-variable design on every
        // estimate click (ADR 0020 / 0022), so a prepared program has no
        // identification product to report; each executed claim carries it.
        let mut view = ReasoningView::structural(support, assumptions);
        view.identification = SlotAvailability::unavailable(IDENTIFIED_PER_EXECUTION);
        return view;
    }
    if matches!(study.query, CausalQuery::Association(_)) {
        // A description of the observed law claims no identification; the slot says so with
        // the stable reason the independent consumer reads.
        return ReasoningView::new(
            SlotAvailability::unavailable(IDENTIFICATION_NOT_CLAIMED),
            SlotAvailability::Available(support),
            SlotAvailability::unavailable("execution_specific"),
            SlotAvailability::Available(assumptions),
        );
    }
    let (Some(_), Some(result)) = (prepared, cached) else {
        return ReasoningView::structural(support, assumptions);
    };
    let slot = identification_slot_from_prepared(study, result, search_capped);
    ReasoningView::new(
        SlotAvailability::Available(slot),
        SlotAvailability::Available(support),
        SlotAvailability::unavailable("execution_specific"),
        SlotAvailability::Available(assumptions),
    )
}

/// Identification-slot reason on a prepared program that re-identifies on
/// every execution instead of caching an identification product.
pub const IDENTIFIED_PER_EXECUTION: &str = "identified_per_execution";

/// Whether this study is the identify-per-click exception (sharp RD).
pub(super) fn identifies_per_execution(study: &Study) -> bool {
    study.declared.estimator == Some(crate::EstimatorId::RdSharp)
        || study.declared.identifier == Some(crate::IdentifierId::RdSharp)
}

pub(crate) fn matrix_coordinate(study: &Study) -> Option<String> {
    // A lag-0 AverageEffect on a temporal graph runs as its pulse; its cell is the AverageEffect.
    let requested = study.lag_zero_average_effect.clone().map(CausalQuery::AverageEffect);
    let cell = crate::support::support_cell_named(
        requested.as_ref().unwrap_or(&study.query),
        if study.graph_posterior.is_some() {
            study.graph.class().as_str()
        } else {
            crate::support::matrix_graph_class(
                &study.graph,
                &study.query,
                study.declared.tiered.as_ref(),
            )
        },
        study.structure_source,
        &study.settings.inference,
        study.settings.refute,
    )?;
    Some(crate::support::cell_coordinate(cell))
}

/// Maps the matrix cell status to its semantic applicability and the blockers it carries.
pub(super) fn matrix_applicability(
    status: Option<CellStatus>,
    cell: Option<crate::support::SupportCell>,
) -> (SemanticApplicability, Vec<antecedent_contract::BlockedOperation>) {
    match status {
        Some(CellStatus::Licensed) => (SemanticApplicability::Licensed, Vec::new()),
        Some(CellStatus::Refused) => {
            let reason = cell.map_or(
                "cell is not licensed (parity/support_licensed.toml) and is not n/a; it is refused.",
                crate::support::refused_message,
            );
            (
                SemanticApplicability::Unlicensed,
                vec![antecedent_contract::BlockedOperation::refused(reason)],
            )
        }
        Some(CellStatus::NotApplicable { reason }) => (
            SemanticApplicability::Impossible,
            vec![antecedent_contract::BlockedOperation::not_applicable(reason)],
        ),
        Some(CellStatus::Allowlisted { reason, .. }) => (
            SemanticApplicability::Unlicensed,
            vec![antecedent_contract::BlockedOperation::refused(reason)],
        ),
        None => (
            SemanticApplicability::Unknown,
            vec![antecedent_contract::BlockedOperation::off_axis()],
        ),
    }
}

pub(super) fn operation_report(
    contract: &CausalContract,
    operation: OperationKind,
) -> OperationReport {
    let coordinate =
        contract.reasoning.support.as_ref().and_then(|slot| slot.matrix_coordinate.clone());
    let cell = coordinate.as_deref().and_then(crate::support::support_cell_from_coordinate);
    let unmet: Vec<ObligationRecord> = contract
        .reasoning
        .assumptions
        .as_ref()
        .map(|slot| slot.obligations.iter().filter(|o| o.is_unresolved()).cloned().collect())
        .unwrap_or_default();
    let (applicability, mut blockers) = matrix_applicability(contract.support_status, cell);
    let neighbors = match contract.support_status {
        Some(CellStatus::Refused | CellStatus::NotApplicable { .. }) => {
            cell.map(crate::support::licensed_neighbors).unwrap_or_default()
        }
        _ => Vec::new(),
    };

    if !operation.uses_matrix_row() && applicability != SemanticApplicability::Licensed {
        let reason = match operation {
            OperationKind::RankDesigns => crate::error::RANK_DESIGNS_REQUIRES_LICENSE,
            OperationKind::Retarget => crate::error::RETARGET_REQUIRES_LICENSE,
            OperationKind::Export => crate::error::EXPORT_REQUIRES_LICENSE,
            _ => crate::error::OPERATION_REQUIRES_LICENSE,
        };
        blockers = vec![antecedent_contract::BlockedOperation::operation_unlicensed(reason)];
    } else if operation == OperationKind::RankDesigns
        && contract.identities.identification_product.is_none()
    {
        blockers.push(antecedent_contract::BlockedOperation::binding("identification_product"));
    }

    let readiness = if !blockers.is_empty() {
        if blockers.iter().any(|blocker| blocker.id.as_ref().starts_with("binding.")) {
            Some(OperationReadiness::BindingMissing)
        } else if applicability == SemanticApplicability::Unknown {
            Some(OperationReadiness::UnknownSupport)
        } else {
            None
        }
    } else if unmet.iter().any(|o| o.kind == ObligationKind::FailedCheck) {
        Some(OperationReadiness::EmpiricalSupportFailed)
    } else if unmet.iter().any(|o| o.kind == ObligationKind::CheckNotRun) {
        Some(OperationReadiness::EmpiricalCheckPending)
    } else {
        Some(OperationReadiness::Executable)
    };

    let mut next_actions: Vec<NextAction> =
        if applicability == SemanticApplicability::Licensed && blockers.is_empty() {
            Vec::new()
        } else if neighbors.is_empty() {
            vec![NextAction::new(
                "no_licensed_neighbor",
                "no justified licensed neighbor; do not relabel the graph class",
            )]
        } else {
            vec![NextAction::new(
                "consider_licensed_neighbor",
                "choose an explicit licensed neighbor; never an automatic fallback",
            )]
        };
    // Evidence-gathering actions follow the existing ones; a contract without a
    // not-identified product holds none.
    next_actions.extend(contract.evidence_actions.iter().cloned());

    let (preserved, invalidated) =
        if applicability == SemanticApplicability::Licensed && blockers.is_empty() {
            (
                vec![
                    SemanticLayer::Target,
                    SemanticLayer::Identification,
                    SemanticLayer::Program,
                    SemanticLayer::Support,
                ],
                Vec::new(),
            )
        } else {
            (
                vec![SemanticLayer::Target],
                vec![SemanticLayer::Support, SemanticLayer::Identification, SemanticLayer::Results],
            )
        };

    OperationReport::new(
        operation,
        applicability,
        readiness,
        coordinate,
        blockers,
        unmet,
        preserved,
        invalidated,
        next_actions,
        neighbors,
    )
}

pub(super) fn observation_obligations(
    query: &antecedent_core::CausalQuery,
) -> Vec<ObligationRecord> {
    observation_tags(query)
        .into_iter()
        .enumerate()
        .map(|(index, tag)| {
            ObligationRecord::new(
                format!("observation.{index}"),
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::UserAssertion,
                AssumptionStatus::Declared,
                tag,
            )
        })
        .collect()
}

pub(super) fn obligations_from_set(set: Option<&AssumptionSet>) -> Vec<ObligationRecord> {
    let Some(set) = set else {
        return Vec::new();
    };
    set.entries
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let kind = match record.status {
                AssumptionStatus::Declared => ObligationKind::CheckNotRun,
                AssumptionStatus::Supported => ObligationKind::EmpiricalDiagnostic,
                AssumptionStatus::Contradicted => ObligationKind::FailedCheck,
                AssumptionStatus::Untestable => ObligationKind::Uncheckable,
            };
            let kind = match (&record.source, kind) {
                (AssumptionSource::UserDeclared, ObligationKind::CheckNotRun) => {
                    ObligationKind::UserAssertion
                }
                (AssumptionSource::AlgorithmDefault { .. }, ObligationKind::CheckNotRun) => {
                    ObligationKind::GraphicalImplication
                }
                _ => kind,
            };
            ObligationRecord::new(
                format!("assumption.{index}"),
                ObligationScope::Program,
                record.source.clone(),
                kind,
                record.status,
                assumption_label(&record.assumption),
            )
            .with_assumption(record.assumption.clone())
        })
        .collect()
}

/// Stable label of an assumption, as contracts and plan views write it.
pub(crate) fn assumption_label(assumption: &Assumption) -> String {
    match assumption {
        Assumption::CausalMarkov => "causal_markov".into(),
        Assumption::Faithfulness => "faithfulness".into(),
        Assumption::CausalSufficiency => "causal_sufficiency".into(),
        Assumption::Consistency => "consistency".into(),
        Assumption::Positivity => "positivity".into(),
        Assumption::NoInterference => "no_interference".into(),
        Assumption::Stationarity => "stationarity".into(),
        Assumption::PiecewiseStationarity => "piecewise_stationarity".into(),
        Assumption::NoSelectionBias => "no_selection_bias".into(),
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction".into(),
        Assumption::Monotonicity => "monotonicity".into(),
        Assumption::ParametricRestriction(restriction) => {
            format!("parametric:{}", restriction.id)
        }
        Assumption::PriorRestriction(restriction) => format!("prior:{}", restriction.id),
        Assumption::Custom { id, .. } => format!("custom:{id}"),
    }
}

/// Adds the response's own interval or posterior artifact as an uncertainty component, whatever the inference mode.
pub(super) fn push_response_uncertainty(
    result: &StudyResult,
    components: &mut Vec<UncertaintyComponent>,
) {
    // A function-valued posterior can carry its band directly on the response,
    // without a scalar posterior or SE on StudyResult. Its portable reasoning
    // must not call that published parameter uncertainty "omitted". This runs for
    // every inference kind: the interval's own tag decides the uncertainty source
    // below, so a Frequentist response confidence band is exactly as available
    // here as a Bayesian credible band — there is nothing left for an
    // `inference == "bayesian"` gate to guard.
    {
        if let Some(response) = result.response() {
            // The interval's own tag decides the source: a credible interval is posterior
            // parameter uncertainty; a confidence interval (a Wald interval on influence
            // scores, say) is sampling uncertainty even when the fit was Bayesian.
            let target = match &response.uncertainty {
                ResponseUncertainty::None => None,
                ResponseUncertainty::Scalar { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_interval",
                        "response_confidence_interval",
                    ))
                }
                ResponseUncertainty::PointwiseBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_pointwise_band",
                        "response_pointwise_confidence_band",
                    ))
                }
                ResponseUncertainty::SimultaneousBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_simultaneous_band",
                        "response_simultaneous_confidence_band",
                    ))
                }
                ResponseUncertainty::IdentifiedEnvelopeBand { interpretation, .. } => {
                    Some(response_interval_component(
                        *interpretation,
                        "posterior_envelope_band",
                        "response_envelope_confidence_band",
                    ))
                }
                ResponseUncertainty::Posterior { .. } => {
                    Some((UncertaintySource::Parameter, "posterior_artifact"))
                }
            };
            if let Some((source, target)) = target {
                components.push(UncertaintyComponent::new(source, target, false));
            }
        }
    }
}

pub(super) fn result_reasoning(
    result: &StudyResult,
    prepared: &ReasoningView,
    body: &AnalysisResultWire,
) -> Result<ReasoningView, CausalError> {
    if let Some(association) = result.association() {
        return Ok(association_reasoning(association, prepared));
    }
    let identification = identification_slot_from_result(result)?;
    let mut components = Vec::new();
    if let Some(effect) = result.estimate.as_effect() {
        let published = crate::PublishedScalarUncertainty::select(effect);
        match published.method {
            antecedent_core::IntervalMethod::AnalyticSe => components
                .push(UncertaintyComponent::new(UncertaintySource::Sampling, "analytic_se", false)),
            antecedent_core::IntervalMethod::BootstrapSe => components.push(
                UncertaintyComponent::new(UncertaintySource::Sampling, "bootstrap_se", false),
            ),
            antecedent_core::IntervalMethod::AndersonRubin => components.push(
                UncertaintyComponent::new(UncertaintySource::Sampling, "anderson_rubin", false),
            ),
            _ => {}
        }
    }
    if result.posterior.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Parameter,
            "posterior",
            false,
        ));
    }
    if let Some(policy) = &body.policy_value {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            format!("policy_value_{}", policy.uncertainty),
            false,
        ));
    }
    if let Some(dose) = &body.continuous_dose_response {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            format!("continuous_dose_{}", dose.uncertainty),
            false,
        ));
    }
    push_response_uncertainty(result, &mut components);
    if result.structural_response.is_some()
        || identification.weight_basis.is_some()
        || matches!(
            identification.status,
            IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
        )
    {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Structural,
            "structural_mixture",
            false,
        ));
    }
    let uncertainty = if components.is_empty() {
        SlotAvailability::unavailable("omitted")
    } else {
        SlotAvailability::Available(UncertaintySlot::new(components))
    };
    let mut support = prepared.support.clone();
    if let SlotAvailability::Available(slot) = &mut support {
        // Only overlap / positivity checks speak to support, and only a pass
        // supports it: a failed check is recorded as `failed:<refuter>`.
        slot.empirical = antecedent_io::support_empirical(&body.refutations).map_or_else(
            || SlotAvailability::unavailable("not_evaluated"),
            |label| SlotAvailability::Available(Arc::from(label)),
        );
    }
    Ok(ReasoningView::new(
        SlotAvailability::Available(identification),
        support,
        uncertainty,
        prepared.assumptions.clone(),
    ))
}

/// The reasoning of an executed association: identification not claimed, the uncertainty the
/// description reports, and the premises of a description.
pub(super) fn association_reasoning(
    association: &crate::analysis::association::AssociationPayload,
    prepared: &ReasoningView,
) -> ReasoningView {
    let mut components = Vec::new();
    if association.estimate.resampling.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            "association_resampling_standard_error",
            false,
        ));
    } else if association.estimate.terms.iter().any(|term| term.standard_error.is_some()) {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Sampling,
            "association_analytic_standard_error",
            false,
        ));
    }
    if association.estimate.posterior.is_some() {
        components.push(UncertaintyComponent::new(
            UncertaintySource::Parameter,
            "association_posterior",
            false,
        ));
    }
    let uncertainty = if components.is_empty() {
        SlotAvailability::unavailable("omitted")
    } else {
        SlotAvailability::Available(UncertaintySlot::new(components))
    };
    // No overlap or positivity check speaks to a description: the empirical support is
    // not evaluated.
    let mut support = prepared.support.clone();
    if let SlotAvailability::Available(slot) = &mut support {
        slot.empirical = SlotAvailability::unavailable("not_evaluated");
    }
    ReasoningView::new(
        SlotAvailability::unavailable(IDENTIFICATION_NOT_CLAIMED),
        support,
        uncertainty,
        prepared.assumptions.clone(),
    )
}
