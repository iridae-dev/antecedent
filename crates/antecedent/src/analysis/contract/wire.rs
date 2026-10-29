//! Wire sections of the contract body and its identification slots.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

pub(super) fn executed_scalar(result: &StudyResult) -> Option<f64> {
    if result.response().is_some() {
        return None;
    }
    let scalar = result.effect();
    scalar.is_finite().then_some(scalar)
}

/// The window namespace of a mediation under a temporal ADMG or PAG.
pub(super) fn window_namespace(study: &Study) -> Option<Vec<HorizonAdjustmentNodeWire>> {
    let (DataInput::Temporal(series) | DataInput::Event(series)) = &study.data else {
        return None;
    };
    let count = u32::try_from(series.schema().len()).ok()?;
    // A dynamic graph posterior answered atom by atom identifies in line (no cached product),
    // so its result's identification is declared on the unfolded window of the template it
    // names.
    if let Some(nodes) = crate::analysis::execute::dbn_route_window_namespace(study) {
        return Some(
            nodes
                .into_iter()
                .map(|(variable, offset)| HorizonAdjustmentNodeWire { variable, offset })
                .collect(),
        );
    }
    crate::analysis::execute::mediation_window_namespace(
        study.graph.graph_ref(),
        &study.query,
        count,
    )
    .map(|nodes| {
        nodes
            .into_iter()
            .map(|(variable, offset)| HorizonAdjustmentNodeWire { variable, offset })
            .collect()
    })
}

pub(super) fn body_frame(
    query: &antecedent_core::CausalQuery,
    temporal: Option<&CachedTemporalIdentification>,
    cached: Option<&IdentificationResult>,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Result<BodyFrame, CausalError> {
    Ok(BodyFrame {
        query: causal_query_to_wire_with_registry(query, registry).map_err(|err| io_err(&err))?,
        identification: cached
            .map(|result| identification_to_wire_with_registry(result, registry))
            .transpose()
            .map_err(|err| io_err(&err))?,
        identification_status: cached.map(|result| result.status),
        temporal_identification: temporal_identification_wires(temporal, registry)?,
        window_variables: None,
        registry: registry.cloned(),
    })
}

/// Executed body: the program's frame plus everything the result reports.
pub(super) fn body_for(
    frame: &BodyFrame,
    result: &StudyResult,
) -> Result<AnalysisResultWire, CausalError> {
    let mut identification = match &frame.identification {
        Some(identification) => identification.clone(),
        None => {
            identification_to_wire_with_registry(&result.identification, frame.registry.as_ref())
                .map_err(|err| io_err(&err))?
        }
    };
    let temporal_identification = frame.temporal_identification.clone();
    let identification_variables = temporal_identification
        .iter()
        .find(|entry| entry.identification.query == identification.query)
        .or_else(|| temporal_identification.first())
        .map(|entry| entry.variables.clone())
        .or_else(|| frame.window_variables.clone());
    identification.query = frame.query.clone();
    let effect = result.estimate.as_effect();
    let published = effect.map(crate::PublishedScalarUncertainty::select);
    let executed_estimate = effect.and_then(|_| executed_scalar(result));
    let support = result.support_status;
    let mut wire = AnalysisResultWire {
        native_execution: result
            .performance
            .native_execution
            .as_ref()
            .and_then(antecedent_io::native_execution_wire::NativeExecutionWire::from_facts),
        query: frame.query.clone(),
        identification,
        identification_variables,
        temporal_identification,
        estimate: executed_estimate,
        policy_value: None,
        continuous_dose_response: None,
        panel_did: None,
        synthetic_control: None,
        synthetic_did: None,
        local_polynomial_ratio: None,
        randomized_effect: None,
        survival: None,
        posterior_row: None,
        longitudinal_regime: None,
        sequential_contrast: None,
        history_match: None,
        quasi_rollout: None,
        temporal_design: None,
        iv_bounds: None,
        association: None,
        interference_inference: None,
        interventional_distribution: None,
        // A standard error describes the published scalar; none rides without it.
        standard_error: executed_estimate.and(published.as_ref().and_then(|p| p.standard_error)),
        interval_lower: published.as_ref().and_then(|p| p.lower),
        interval_upper: published.as_ref().and_then(|p| p.upper),
        joint_covariance: effect.and_then(|estimate| estimate.joint.covariance.as_ref()).map(
            |covariance| antecedent_io::JointCovarianceWire {
                dim: covariance.dim,
                values: covariance.values.to_vec(),
            },
        ),
        simultaneous_interval: effect.and_then(|estimate| estimate.family.simultaneous_interval),
        adjusted_p_values: effect.and_then(|estimate| estimate.family.adjusted_p_values),
        family_contrast: effect.and_then(|estimate| estimate.family.contrast),
        family_contrast_interval: effect.and_then(|estimate| estimate.family.contrast_interval),
        assumptions: antecedent_io::assumptions_to_wire(
            effect
                .map_or(&result.identification.required_assumptions, |e| &e.provenance.assumptions),
        ),
        diagnostics: result.diagnostics.iter().map(antecedent_io::diagnostic_to_wire).collect(),
        refutations: result.refutations.iter().map(antecedent_io::refutation_to_wire).collect(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        counterfactual_contrasts: None,
        cate: effect.and_then(|v| v.heterogeneity.cate.clone()),
        fitted_effect: effect.and_then(|v| v.heterogeneity.fitted_effect.clone()),
        cate_se: effect.and_then(|v| v.heterogeneity.cate_se.clone()),
        cate_leaf_dispersion: effect.and_then(|v| v.heterogeneity.cate_leaf_dispersion.clone()),
        outcome_oof_r2: effect.and_then(|v| v.crossfit.outcome_oof_r2),
        treatment_oof_logloss: effect.and_then(|v| v.crossfit.treatment_oof_logloss),
        crossfit_folds: effect.and_then(|v| v.crossfit.folds),
        crossfit_seed: effect.and_then(|v| v.crossfit.seed),
        learner_provenance: effect
            .into_iter()
            .flat_map(|v| v.crossfit.learner_provenance.iter())
            .map(|p| (p.spec.clone(), p.implementation.clone(), p.version.clone()))
            .collect(),
    };
    fill_family_section(&mut wire, result, support);
    result.fill_analysis_result_payloads(&mut wire, "execution-posterior")?;
    Ok(wire)
}

/// Whether the result family reports a calibrated scalar effect interval.
///
/// Exhaustive over [`ResultPayload`]: a new family must state its policy here.
pub(super) fn reports_calibrated_effect_interval(payload: &ResultPayload) -> bool {
    match payload {
        ResultPayload::RandomizedEffect(_)
        | ResultPayload::SyntheticControl(_)
        | ResultPayload::SyntheticDid(_)
        | ResultPayload::LocalPolynomialRatio(_)
        | ResultPayload::PolicyValue(_)
        | ResultPayload::ContinuousDoseResponse(_)
        | ResultPayload::Survival(_)
        | ResultPayload::PosteriorRow(_)
        | ResultPayload::LongitudinalRegime(_)
        | ResultPayload::SequentialContrast(_)
        | ResultPayload::HistoryMatch(_)
        | ResultPayload::QuasiRollout(_)
        | ResultPayload::TemporalDesign(_)
        | ResultPayload::IvBounds(_)
        | ResultPayload::Association(_) => false,
        ResultPayload::Scalar
        | ResultPayload::Response(_)
        | ResultPayload::Distribution(_)
        | ResultPayload::Mediation { .. }
        | ResultPayload::Counterfactual(_)
        | ResultPayload::CounterfactualContrasts(_)
        | ResultPayload::Anomaly(_)
        | ResultPayload::ChangeAttribution(_)
        | ResultPayload::MechanismChange(_)
        | ResultPayload::UnitChange(_)
        | ResultPayload::Transport(_)
        | ResultPayload::Interference { .. }
        | ResultPayload::PanelDid(_)
        | ResultPayload::Discovery(_)
        | ResultPayload::Attribution(_)
        | ResultPayload::Simulation(_) => true,
    }
}

/// Write the family-specific section of the body for the result's one payload.
///
/// Exhaustive over [`ResultPayload`]: a new family must choose its section (or
/// state that it has none) here.
pub(super) fn fill_family_section(
    wire: &mut AnalysisResultWire,
    result: &StudyResult,
    support: Option<CellStatus>,
) {
    match &result.payload {
        ResultPayload::PolicyValue(v) => wire.policy_value = Some(v.to_wire(support)),
        ResultPayload::ContinuousDoseResponse(v) => {
            wire.continuous_dose_response = Some(v.to_wire(support));
        }
        ResultPayload::PanelDid(v) => wire.panel_did = Some(v.to_wire(support)),
        ResultPayload::SyntheticControl(v) => wire.synthetic_control = Some(v.into()),
        ResultPayload::SyntheticDid(v) => wire.synthetic_did = Some(v.into()),
        ResultPayload::LocalPolynomialRatio(v) => {
            wire.local_polynomial_ratio = Some(v.to_wire(support));
        }
        ResultPayload::RandomizedEffect(v) => wire.randomized_effect = Some(v.to_wire(support)),
        ResultPayload::Survival(v) => wire.survival = Some(v.to_wire(support)),
        ResultPayload::PosteriorRow(v) => wire.posterior_row = Some(v.clone()),
        ResultPayload::LongitudinalRegime(v) => wire.longitudinal_regime = Some(v.into()),
        ResultPayload::SequentialContrast(v) => wire.sequential_contrast = Some(v.into()),
        ResultPayload::HistoryMatch(v) => wire.history_match = Some(v.into()),
        ResultPayload::QuasiRollout(v) => wire.quasi_rollout = Some(v.as_ref().into()),
        ResultPayload::TemporalDesign(v) => wire.temporal_design = Some(v.as_ref().into()),
        ResultPayload::IvBounds(v) => wire.iv_bounds = Some(result_wire::iv_bounds_wire(v)),
        ResultPayload::Association(v) => wire.association = Some(result_wire::association_wire(v)),
        ResultPayload::Interference { inference, .. } => {
            wire.interference_inference = inference.as_ref().map(Into::into);
        }
        ResultPayload::Distribution(v) => {
            wire.interventional_distribution = Some(distribution_wire(v));
        }
        // Families whose answer is carried by the response, posterior, mediation grid or
        // unit-effects payloads filled afterwards, or by no body section at all.
        ResultPayload::Scalar
        | ResultPayload::Response(_)
        | ResultPayload::Mediation { .. }
        | ResultPayload::Counterfactual(_)
        | ResultPayload::CounterfactualContrasts(_)
        | ResultPayload::Anomaly(_)
        | ResultPayload::ChangeAttribution(_)
        | ResultPayload::MechanismChange(_)
        | ResultPayload::UnitChange(_)
        | ResultPayload::Transport(_)
        | ResultPayload::Discovery(_)
        | ResultPayload::Attribution(_)
        | ResultPayload::Simulation(_) => {}
    }
}

pub(super) fn temporal_identification_wires(
    temporal: Option<&CachedTemporalIdentification>,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Result<Vec<TemporalIdentificationWire>, CausalError> {
    let Some(temporal) = temporal else {
        return Ok(Vec::new());
    };
    temporal
        .by_horizon
        .iter()
        .map(|entry| {
            let variables = (0..entry.indexer.dense_len())
                .map(|dense| -> Result<HorizonAdjustmentNodeWire, CausalError> {
                    let key = entry.indexer.key_of(u32::try_from(dense).map_err(|_| {
                        CausalError::Compile { message: "temporal dense id exceeds u32".into() }
                    })?)?;
                    Ok(HorizonAdjustmentNodeWire {
                        variable: key.variable.raw(),
                        offset: key.offset,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TemporalIdentificationWire {
                horizon: entry.horizon,
                variables,
                identification: identification_to_wire_with_registry(
                    &entry.identification,
                    registry,
                )
                .map_err(|err| io_err(&err))?,
            })
        })
        .collect()
}

pub(in crate::analysis) fn reasoning_section(view: &ReasoningView) -> ReasoningSectionWire {
    ReasoningSectionWire {
        identification: slot_section_from_availability(&view.identification, |slot| {
            IdentificationSlotWire {
                status: slot.status.as_str().into(),
                identified_mass: slot.identified_mass,
                unidentified_mass: slot.unidentified_mass,
                unevaluable_mass: slot.unevaluable_mass,
                incomplete_search_mass: slot.incomplete_search_mass,
                full_mass_scope: slot.full_mass_scope,
                search_capped: slot.search_capped,
                weight_basis: slot.weight_basis.as_ref().map(std::string::ToString::to_string),
            }
        }),
        support: slot_section_from_availability(&view.support, |slot| SupportSlotWire {
            matrix_status: slot.matrix_status.to_string(),
            matrix_coordinate: slot
                .matrix_coordinate
                .as_ref()
                .map(std::string::ToString::to_string),
            empirical: slot.empirical.label(std::string::ToString::to_string),
        }),
        uncertainty: slot_section_from_availability(&view.uncertainty, |slot| {
            UncertaintySlotWire {
                components: slot
                    .components
                    .iter()
                    .map(|component| UncertaintyComponentWire {
                        source: component.source.as_str().into(),
                        target: component.target.to_string(),
                        omitted: component.omitted,
                    })
                    .collect(),
            }
        }),
        assumptions: slot_section_from_availability(&view.assumptions, |slot| AssumptionSlotWire {
            obligations: slot
                .obligations
                .iter()
                .map(|obligation| ObligationSectionWire {
                    id: obligation.id.to_string(),
                    scope: obligation.scope.as_str().into(),
                    kind: obligation.kind.as_str().into(),
                    status: obligation.status.as_str().into(),
                })
                .collect(),
        }),
    }
}

pub(super) fn slot_section_from_availability<T, U>(
    slot: &SlotAvailability<T>,
    map: impl FnOnce(&T) -> U,
) -> SlotSectionWire<U> {
    match slot {
        SlotAvailability::Available(value) => {
            SlotSectionWire { value: Some(map(value)), unavailable: None }
        }
        SlotAvailability::Unavailable { reason } => {
            SlotSectionWire { value: None, unavailable: Some(reason.to_string()) }
        }
        _ => SlotSectionWire { value: None, unavailable: Some("unknown".into()) },
    }
}

pub(super) fn preview_obligation(
    id: &'static str,
    kind: ObligationKind,
    check: &'static str,
    message: &'static str,
) -> ObligationRecord {
    ObligationRecord::new(
        id,
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        kind,
        AssumptionStatus::Declared,
        message,
    )
    .with_required_check(check)
}

pub(super) fn identification_slot_from_result(
    result: &StudyResult,
) -> Result<IdentificationSlot, CausalError> {
    if let Some(mixture) = &result.structural_response {
        antecedent_io::validate_mixture_masses(
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
        )
        .map_err(|err| io_err(&err))?;
        return Ok(IdentificationSlot::new(
            result.identification.status,
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            mixture.subsampled_out_mass,
            mixture.full_mass_scope,
            Some(Arc::from(mixture.weight_basis.as_str())),
            mixture.truncated_atoms > 0,
        ));
    }
    if let Some((unidentified, incomplete, basis)) = projected_mixture_masses(result) {
        let identified = (1.0 - unidentified - incomplete).max(0.0);
        antecedent_io::validate_mixture_masses(identified, unidentified, 0.0, incomplete)
            .map_err(|err| io_err(&err))?;
        return Ok(IdentificationSlot::new(
            result.identification.status,
            identified,
            unidentified,
            0.0,
            incomplete,
            incomplete == 0.0,
            Some(Arc::from(basis)),
            incomplete > 0.0,
        ));
    }
    Ok(IdentificationSlot::identified_singleton(result.identification.status))
}

pub(super) fn identification_slot_from_prepared(
    study: &Study,
    result: &IdentificationResult,
    search_capped: bool,
) -> IdentificationSlot {
    if let Some(cache) = study.temporal_class_cache() {
        return slot_from_envelope(
            result.status,
            &cache.envelope.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.cpdag_cache() {
        return slot_from_envelope(
            cache.identification.status,
            &cache.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.pag_cache() {
        return slot_from_envelope(
            cache.identification.status,
            &cache.envelope,
            "completion_enumeration",
            search_capped,
        );
    }
    if let Some(cache) = study.graph_posterior_cache() {
        return slot_from_graph_samples(result.status, &cache.graphs, search_capped);
    }
    if let Some(cache) = study.dbn_posterior_cache() {
        return slot_from_graph_samples(result.status, &cache.graphs, search_capped);
    }
    let mut slot = IdentificationSlot::identified_singleton(result.status);
    slot.search_capped = search_capped;
    if search_capped {
        slot.full_mass_scope = false;
    }
    slot
}

/// Project an enumerated class envelope onto the identification slot.
///
/// Weight whose search was capped before it could decide is incomplete search,
/// not proof of non-identification (ADR 0022). Zero examined mass is the same
/// kind of ignorance: it is never reported as fully unidentified.
pub(super) fn slot_from_envelope<G>(
    status: IdentificationStatus,
    envelope: &IdentificationEnvelope<G>,
    basis: &'static str,
    search_capped: bool,
) -> IdentificationSlot {
    let identified = envelope.identified_weight.0;
    let unidentified = envelope.unidentified_weight.0;
    let truncated = envelope.truncated_weight().clamp(0.0, unidentified.max(0.0));
    let total = identified + unidentified;
    let (identified_mass, unidentified_mass, incomplete_mass) = if total > 0.0 {
        (identified / total, (unidentified - truncated) / total, truncated / total)
    } else {
        (0.0, 0.0, 1.0)
    };
    let capped = search_capped || envelope.truncated_completions > 0;
    IdentificationSlot::new(
        status,
        identified_mass,
        unidentified_mass,
        0.0,
        incomplete_mass,
        total > 0.0 && !capped,
        Some(Arc::from(basis)),
        capped,
    )
}

pub(super) fn slot_from_graph_samples(
    status: IdentificationStatus,
    graphs: &antecedent_prob::WeightedGraphSamples,
    search_capped: bool,
) -> IdentificationSlot {
    let identified = graphs.identified_mass();
    let unidentified = graphs.unidentified_mass();
    let total = identified + unidentified;
    // No examined mass tells us nothing; it is not a fully unidentified class.
    let (identified_mass, unidentified_mass, incomplete_mass) =
        if total > 0.0 { (identified / total, unidentified / total, 0.0) } else { (0.0, 0.0, 1.0) };
    IdentificationSlot::new(
        status,
        identified_mass,
        unidentified_mass,
        0.0,
        incomplete_mass,
        total > 0.0 && !search_capped,
        Some(Arc::from("posterior_probability")),
        search_capped,
    )
}

pub(super) fn projected_mixture_masses(result: &StudyResult) -> Option<(f64, f64, &'static str)> {
    let basis = match result.structure_source {
        StructureSource::GraphPosterior => "posterior_probability",
        _ => "completion_enumeration",
    };
    if let Some(posterior) = &result.posterior {
        if posterior.unidentified_mass > 0.0
            || posterior.subsampled_out_mass > 0.0
            || matches!(
                result.identification.status,
                IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
            )
        {
            return Some((posterior.unidentified_mass, posterior.subsampled_out_mass, basis));
        }
    }
    if matches!(
        result.identification.status,
        IdentificationStatus::GraphDependent | IdentificationStatus::PartiallyIdentified
    ) {
        let unidentified = unidentified_mass_from_diagnostics(&result.diagnostics)?;
        return Some((unidentified, 0.0, basis));
    }
    None
}

/// Identification mass carried on a diagnostic's structured fields.
///
/// Prose is never parsed: a message that mentions only `unidentified_mass`
/// once read as 50% identified.
pub(super) fn unidentified_mass_from_diagnostics(
    diagnostics: &[antecedent_core::Diagnostic],
) -> Option<f64> {
    diagnostics.iter().find_map(|diagnostic| {
        let unidentified = mass_field(diagnostic, "unidentified_mass")?;
        if let Some(identified) = mass_field(diagnostic, "identified_mass") {
            let total = identified + unidentified;
            if total > 0.0 {
                return Some(unidentified / total);
            }
        }
        Some(unidentified)
    })
}

pub(super) fn mass_field(diagnostic: &antecedent_core::Diagnostic, field: &str) -> Option<f64> {
    diagnostic
        .fields
        .iter()
        .find(|(key, _)| key.as_ref() == field)
        .and_then(|(_, value)| value.parse().ok())
}

pub(super) fn identification_status_wire(
    status: antecedent_core::IdentificationStatus,
) -> antecedent_io::IdentificationStatusWire {
    match status {
        antecedent_core::IdentificationStatus::NonparametricallyIdentified => {
            antecedent_io::IdentificationStatusWire::NonparametricallyIdentified
        }
        antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions => {
            antecedent_io::IdentificationStatusWire::IdentifiedUnderParametricRestrictions
        }
        antecedent_core::IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            antecedent_io::IdentificationStatusWire::IdentifiedUnderPriorRestrictions
        }
        antecedent_core::IdentificationStatus::PartiallyIdentified => {
            antecedent_io::IdentificationStatusWire::PartiallyIdentified
        }
        antecedent_core::IdentificationStatus::GraphDependent => {
            antecedent_io::IdentificationStatusWire::GraphDependent
        }
        antecedent_core::IdentificationStatus::NotIdentified => {
            antecedent_io::IdentificationStatusWire::NotIdentified
        }
    }
}

pub(super) fn mediation_grid_wire(
    grid: &crate::estimate::TemporalMediationGrid,
) -> antecedent_io::TemporalMediationGridWire {
    let interval = |summary: antecedent_estimate::MediationPosteriorSummary| {
        antecedent_io::MediationPosteriorSummaryWire {
            mean: summary.mean,
            standard_deviation: summary.standard_deviation,
            q025: summary.q025,
            q975: summary.q975,
        }
    };
    antecedent_io::TemporalMediationGridWire {
        slices: grid
            .slices
            .iter()
            .map(|slice| {
                let uncertainty = match &slice.uncertainty {
                    antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                        standard_error,
                    }
                    | antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                        requested: standard_error,
                        ..
                    } => antecedent_io::TemporalMediationUncertaintyWire::FrequentistPointwise {
                        standard_error: *standard_error,
                    },
                    antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                        requested,
                        total,
                        direct,
                        mediated,
                        n_draws,
                        backend,
                    } => antecedent_io::TemporalMediationUncertaintyWire::BayesianPointwise {
                        requested: interval(*requested),
                        total: interval(*total),
                        direct: interval(*direct),
                        mediated: interval(*mediated),
                        n_draws: u64::try_from(*n_draws).unwrap_or(u64::MAX),
                        backend: backend.to_string(),
                    },
                    _ => antecedent_io::TemporalMediationUncertaintyWire::Unavailable,
                };
                antecedent_io::TemporalMediationSliceWire {
                    horizon: slice.horizon,
                    identification_status: identification_status_wire(slice.identification_status),
                    method: slice.method.to_string(),
                    adjustment: slice
                        .adjustment
                        .iter()
                        .map(|key| antecedent_io::HorizonAdjustmentNodeWire {
                            variable: key.variable.raw(),
                            offset: key.offset,
                        })
                        .collect(),
                    effect: slice.estimate.effect.ate,
                    total: slice.estimate.total,
                    direct: slice.estimate.direct,
                    mediated: slice.estimate.mediated,
                    uncertainty,
                    identified_set: slice
                        .identified_set
                        .map(|identified| [identified.lower, identified.upper]),
                    diagnostics: slice
                        .diagnostics
                        .iter()
                        .map(antecedent_io::diagnostic_to_wire)
                        .collect(),
                }
            })
            .collect(),
        joint_posterior: grid.joint_posterior,
    }
}

pub(super) fn structural_response_wire(
    mixture: &crate::result::StructuralResponseMixture,
) -> Result<antecedent_io::StructuralResponseMixtureWire, CausalError> {
    let weight_basis = antecedent_io::StructuralWeightBasisWire::from(mixture.weight_basis);
    // Core completion weights may be counts, or be normalized separately at
    // each horizon. The portable container requires unit total while retaining
    // the weight basis and every relative weight; no weights are inferred.
    let total: f64 = mixture.atoms.iter().map(|atom| atom.weight).sum();
    if !mixture.atoms.is_empty() && (!total.is_finite() || total <= 0.0) {
        return Err(CausalError::Compile {
            message: "cannot export structural atoms with invalid total weight".into(),
        });
    }
    Ok(antecedent_io::StructuralResponseMixtureWire {
        weight_basis,
        atoms: mixture
            .atoms
            .iter()
            .map(|atom| {
                Ok(antecedent_io::StructuralResponseAtomWire {
                    graph_key: atom.graph_key,
                    weight: atom.weight / total,
                    identification_status: identification_status_wire(atom.status),
                    value: atom.value.as_ref().map(antecedent_io::response_value_to_wire),
                    posterior_artifact: atom
                        .posterior
                        .as_ref()
                        .map(|posterior| {
                            antecedent_io::encode_causal_posterior_bytes(
                                posterior,
                                "structural_atom",
                            )
                        })
                        .transpose()
                        .map_err(|err| io_err(&err))?,
                    response: atom
                        .response
                        .as_ref()
                        .map(antecedent_io::causal_response_to_wire)
                        .transpose()
                        .map_err(|err| io_err(&err))?,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?,
        identified_mass: mixture.identified_mass,
        unidentified_mass: mixture.unidentified_mass,
        unevaluable_mass: mixture.unevaluable_mass,
        subsampled_out_mass: mixture.subsampled_out_mass,
        identified_set: mixture.identified_set.as_ref().map(|envelope| {
            antecedent_io::ResponseEnvelopeWire {
                grid: envelope.grid.to_vec(),
                dimension: u64::try_from(envelope.dimension).unwrap_or(u64::MAX),
                lower: envelope.lower.to_vec(),
                upper: envelope.upper.to_vec(),
            }
        }),
        identified_set_interval: mixture
            .identified_set_interval
            .as_ref()
            .map(antecedent_io::identified_set_interval_to_wire)
            .transpose()
            .map_err(|err| io_err(&err))?,
        conditional_on_identified: mixture
            .conditional_on_identified
            .as_ref()
            .map(antecedent_io::response_value_to_wire),
        full_mass_scope: mixture.full_mass_scope,
        truncated_atoms: u64::try_from(mixture.truncated_atoms).unwrap_or(u64::MAX),
    })
}
