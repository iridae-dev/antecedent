//! TemporalCpdag/Pag graph-posterior response: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{
    mix_response_values, mix_support_reports, response_envelope_from_weighted,
    response_identified_value, response_primary_pair, response_scalar_summary,
};
use super::temporal_posterior_response::dbn_posterior_response_supported;
use super::*;
use crate::result::{StructuralResponseAtom, StructuralResponseMixture, StructuralWeightBasis};
use antecedent_discovery::{temporal_cpdag_from_dbn_masks, temporal_pag_from_dbn_masks};

impl super::Study {
    /// Mix TemporalCpdag/Pag posterior atoms through the existing class response envelope.
    pub(super) fn execute_temporal_class_graph_posterior_response(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        dbn_posterior_response_supported(query)?;
        query
            .require_licensed_temporal_observation()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let class_tag = match gp.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => "temporal_cpdag",
            antecedent_discovery::GraphPosteriorAtomKind::Pag => "temporal_pag",
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class graph-posterior response requires Cpdag or Pag \
                              atom_kind"
                        .into(),
                });
            }
        };
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let lag_masks = gp.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior missing per-atom lag masks".into(),
        })?;
        let max_lag = gp.max_lag.ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior missing max_lag".into(),
        })?;

        let mut weighted = Vec::new();
        let mut atoms = Vec::new();
        let mut contributing = Vec::new();
        let mut failed_mass = 0.0;
        let mut unidentified_mass = 0.0;
        let mut any_partial = false;
        let mut primary = None;
        for i in 0..gp.n_graphs {
            let key = u64::try_from(i).map_err(|_| CausalError::Compile {
                message: "temporal class graph-posterior response: too many atoms".into(),
            })?;
            let weight = gp.weights[i];
            let mark = gp.mark_masks.as_ref().map(|marks| marks[i]).unwrap_or(0);
            let reconstructed = reconstruct_temporal_class_atom(
                gp.atom_kind,
                gp.adjacency[i],
                lag_masks[i],
                mark,
                gp.n_vars,
                max_lag,
                &vars,
            );
            let Ok(graph) = reconstructed else {
                unidentified_mass += weight;
                atoms.push(StructuralResponseAtom {
                    posterior: None,
                    response: None,
                    graph_key: key,
                    weight,
                    status: IdentificationStatus::NotIdentified,
                    value: None,
                });
                continue;
            };
            let mut atom_study = self.clone();
            atom_study.graph_posterior = None;
            atom_study.graph = graph;
            atom_study.refute = RefuteSuite::None;
            atom_study.custom_validators.clear();
            atom_study.temporal_class_identification_cache = None;
            atom_study.temporal_class_posterior_identification_cache = None;
            match atom_study.execute_temporal_class_response(data, query, physical, ctx) {
                Ok(result) => {
                    let Some(response) = result.response else {
                        failed_mass += weight;
                        atoms.push(StructuralResponseAtom {
                            posterior: None,
                            response: None,
                            graph_key: key,
                            weight,
                            status: result.identification.status,
                            value: None,
                        });
                        continue;
                    };
                    let Some(value) =
                        response_identified_value(&response).map(collapse_singleton_envelope)
                    else {
                        failed_mass += weight;
                        atoms.push(StructuralResponseAtom {
                            posterior: None,
                            response: Some(response),
                            graph_key: key,
                            weight,
                            status: result.identification.status,
                            value: None,
                        });
                        continue;
                    };
                    any_partial |= matches!(value, ResponseValue::Envelope(_))
                        || matches!(
                            result.identification.status,
                            IdentificationStatus::PartiallyIdentified
                        ) && !matches!(
                            value,
                            ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
                        );
                    if primary.is_none() {
                        primary = Some((result.identification.clone(), result.estimand.clone()));
                    }
                    contributing.push(result.estimand);
                    atoms.push(StructuralResponseAtom {
                        posterior: None,
                        response: Some(response.clone()),
                        graph_key: key,
                        weight,
                        status: result.identification.status,
                        value: Some(value),
                    });
                    weighted.push((key, weight, response));
                }
                Err(CausalError::Unsupported { message }) => {
                    return Err(CausalError::Unsupported { message });
                }
                Err(_) => {
                    unidentified_mass += weight;
                    atoms.push(StructuralResponseAtom {
                        posterior: None,
                        response: None,
                        graph_key: key,
                        weight,
                        status: IdentificationStatus::NotIdentified,
                        value: None,
                    });
                }
            }
        }

        let (mut identification, estimand) = primary.ok_or_else(|| CausalError::Compile {
            message: "temporal class graph-posterior response has no evaluable identified atom"
                .into(),
        })?;
        let identified_mass = weighted.iter().map(|(_, weight, _)| *weight).sum::<f64>();
        let total_mass = gp.weights.iter().sum::<f64>();
        let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
        let aggregation_policy = if refs.is_empty() && any_partial {
            StructuralAggregationPolicy::IdentifiedSetEnvelope
        } else {
            resolve_structural_aggregation(&refs, any_partial)
        };
        let conditional_values = atoms
            .iter()
            .filter_map(|atom| atom.value.clone().map(|value| (atom.weight, value)))
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let mixable =
            matches!(aggregation_policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
                && !conditional_refs.is_empty()
                && conditional_refs.iter().all(|(_, value)| {
                    matches!(value, ResponseValue::Scalar(_) | ResponseValue::Surface { .. })
                });
        let conditional =
            if mixable { Some(mix_response_values(&conditional_refs)?) } else { None };
        let structural_set = union_identified_values(
            &conditional_values.iter().map(|(_, value)| value).collect::<Vec<_>>(),
        )
        .or_else(|| response_envelope_from_weighted(&weighted));
        let first = &weighted[0].2;
        let graph_dependent = unidentified_mass > 0.0
            || failed_mass > 0.0
            || matches!(aggregation_policy, StructuralAggregationPolicy::GraphDependentAtoms);
        if graph_dependent {
            identification.status = IdentificationStatus::GraphDependent;
        } else if matches!(aggregation_policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            identification.status = IdentificationStatus::PartiallyIdentified;
        } else if mixable {
            identification.status = IdentificationStatus::NonparametricallyIdentified;
        }
        let uncertainty =
            if weighted.len() == 1 { first.uncertainty.clone() } else { ResponseUncertainty::None };
        let mut assumptions = first.assumptions.clone();
        for (_, _, response) in weighted.iter().skip(1) {
            assumptions.extend_unique(&response.assumptions.entries);
        }
        let response = antecedent_core::CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: if graph_dependent {
                ResponseIdentification::GraphDependent(
                    weighted
                        .iter()
                        .filter_map(|(key, _, response)| {
                            determinate_response_value(response).map(|value| (*key, value))
                        })
                        .collect(),
                )
            } else if matches!(
                aggregation_policy,
                StructuralAggregationPolicy::IdentifiedSetEnvelope
            ) {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    structural_set.clone().ok_or_else(|| CausalError::Compile {
                        message: "temporal class graph-posterior identified set missing envelope"
                            .into(),
                    })?,
                ))
            } else if mixable {
                ResponseIdentification::PointIdentified(
                    conditional.clone().expect("mixable response"),
                )
            } else {
                ResponseIdentification::GraphDependent(
                    weighted
                        .iter()
                        .filter_map(|(key, _, response)| {
                            determinate_response_value(response).map(|value| (*key, value))
                        })
                        .collect(),
                )
            },
            uncertainty,
            support: mix_support_reports(
                &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
            ),
            assumptions,
            provenance_id: Arc::from("estimate.response.temporal_class_graph_posterior"),
            horizon_identification: first.horizon_identification.as_ref().map(|horizons| {
                horizons
                    .iter()
                    .map(|horizon| antecedent_core::HorizonIdentification {
                        status: identification.status,
                        ..horizon.clone()
                    })
                    .collect()
            }),
            interaction_structurally_zero: first.interaction_structurally_zero,
        };
        identification.status = response.identification_status;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );

        let mut diagnostics = vec![Diagnostic::new(
            "estimate.response.temporal_class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "each {class_tag} posterior atom was evaluated with the existing temporal class \
                 response envelope; outer weights are frozen posterior probabilities"
            ),
        )];
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            aggregation_policy,
            identified_mass / total_mass,
            unidentified_mass / total_mass,
            failed_mass / total_mass,
            0.0,
        );
        if weighted.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "multi-atom temporal class-posterior response withholds an aggregate band: \
                 atom surfaces are mixed by frozen graph weight",
            ));
        }

        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) = if is_intervention
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            refute_temporal_class_graph_posterior_intervention(
                self,
                data,
                query,
                gp,
                &vars,
                &weighted,
                aggregation_policy,
                ctx,
            )?
        } else {
            if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.temporal_response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued temporal response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diags);
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect \
                 summary is not applicable and the result is carried by the response payload",
            ));
        }

        let mixture = StructuralResponseMixture {
            weight_basis: StructuralWeightBasis::PosteriorProbability,
            atoms,
            identified_mass: identified_mass / total_mass,
            unidentified_mass: unidentified_mass / total_mass,
            unevaluable_mass: failed_mass / total_mass,
            subsampled_out_mass: 0.0,
            identified_set: structural_set,
            identified_set_interval: None,
            conditional_on_identified: mixable
                .then(|| conditional.clone().expect("mixable response")),
            full_mass_scope: (unidentified_mass + failed_mass) == 0.0,
            truncated_atoms: 0,
        };
        let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::TemporalResponseBayesian
        } else {
            EstimatorId::TemporalResponseGcomp
        };
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior_response",
                    aggregation_policy.as_str(),
                )),
                response: Some(response),
                structural_response: Some(mixture),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }
}

fn collapse_singleton_envelope(value: ResponseValue) -> ResponseValue {
    let ResponseValue::Envelope(envelope) = &value else {
        return value;
    };
    if envelope.lower.len() != envelope.upper.len()
        || envelope
            .lower
            .iter()
            .zip(envelope.upper.iter())
            .any(|(lo, hi)| lo.to_bits() != hi.to_bits())
    {
        return value;
    }
    ResponseValue::Surface {
        grid: Arc::clone(&envelope.grid),
        dimension: envelope.dimension,
        mean: Arc::clone(&envelope.lower),
    }
}

fn determinate_response_value(response: &CausalResponse) -> Option<ResponseValue> {
    match collapse_singleton_envelope(response_identified_value(response)?) {
        value @ (ResponseValue::Scalar(_) | ResponseValue::Surface { .. }) => Some(value),
        _ => None,
    }
}

fn union_identified_values(values: &[&ResponseValue]) -> Option<antecedent_core::ResponseEnvelope> {
    let mut envelope: Option<antecedent_core::ResponseEnvelope> = None;
    for value in values {
        let next = match value {
            ResponseValue::Envelope(item) => item.clone(),
            ResponseValue::Surface { grid, dimension, mean } => antecedent_core::ResponseEnvelope {
                grid: Arc::clone(grid),
                dimension: *dimension,
                lower: Arc::clone(mean),
                upper: Arc::clone(mean),
            },
            ResponseValue::Scalar(item) => scalar_identified_set(*item, *item),
            _ => return None,
        };
        envelope = Some(match envelope {
            None => next,
            Some(mut acc) => {
                if acc.grid.as_ref() != next.grid.as_ref()
                    || acc.dimension != next.dimension
                    || acc.lower.len() != next.lower.len()
                {
                    return None;
                }
                let lower = acc
                    .lower
                    .iter()
                    .zip(next.lower.iter())
                    .map(|(a, b)| a.min(*b))
                    .collect::<Vec<_>>();
                let upper = acc
                    .upper
                    .iter()
                    .zip(next.upper.iter())
                    .map(|(a, b)| a.max(*b))
                    .collect::<Vec<_>>();
                acc.lower = Arc::from(lower);
                acc.upper = Arc::from(upper);
                acc
            }
        });
    }
    envelope
}

fn reconstruct_temporal_class_atom(
    kind: antecedent_discovery::GraphPosteriorAtomKind,
    adjacency: u64,
    lag_mask: u64,
    mark: u64,
    n_vars: usize,
    max_lag: u32,
    variables: &[VariableId],
) -> Result<AcceptedGraph, CausalError> {
    match kind {
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
            let cpdag =
                temporal_cpdag_from_dbn_masks(adjacency, lag_mask, n_vars, max_lag, variables)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            Ok(AcceptedGraph::from(cpdag))
        }
        antecedent_discovery::GraphPosteriorAtomKind::Pag => {
            let pag =
                temporal_pag_from_dbn_masks(adjacency, lag_mask, mark, n_vars, max_lag, variables)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            Ok(AcceptedGraph::from(pag))
        }
        _ => Err(CausalError::Compile {
            message: "temporal class graph-posterior response requires Cpdag or Pag atom_kind"
                .into(),
        }),
    }
}

fn response_pulse_witness(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message:
            "temporal class graph-posterior InterventionResponse requires TemporalResponseSpec"
                .into(),
    })?;
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let horizon = *temporal.horizons.first().ok_or_else(|| CausalError::Compile {
        message:
            "temporal class graph-posterior InterventionResponse requires at least one horizon"
                .into(),
    })?;
    let active = match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            interventions.first().cloned().ok_or_else(|| CausalError::Compile {
                message: "InterventionResponse has no intervention coordinate".into(),
            })?
        }
        _ => Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
    };
    Ok(TemporalEffectQuery {
        treatment,
        outcome,
        policy: temporal.policy.clone(),
        control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
        active,
        horizon_steps: horizon,
        max_history_lag: temporal.max_history_lag,
        target_population: query.target_population.clone(),
    })
}

fn refute_temporal_class_graph_posterior_intervention(
    study: &Study,
    data: &TimeSeriesData,
    query: &ResponseQuery,
    gp: &GraphPosterior,
    vars: &[VariableId],
    weighted: &[(u64, f64, CausalResponse)],
    policy: StructuralAggregationPolicy,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    if matches!(policy, StructuralAggregationPolicy::GraphDependentAtoms) {
        return Ok((
            Vec::new(),
            vec![Diagnostic::new(
                "refute.envelope.temporal_class_graph_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "outer scalar mix skipped because StructuralAggregationPolicy is \
                 graph_dependent_atoms — per-atom class envelopes are retained",
            )],
        ));
    }
    let pulse = response_pulse_witness(query)?;
    let lag_masks = gp.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = gp.max_lag.ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    let mut refute_atoms = Vec::new();
    for (key, weight, _) in weighted {
        let i = usize::try_from(*key).map_err(|_| CausalError::Compile {
            message: "temporal class graph-posterior IR refuter: atom key overflow".into(),
        })?;
        let mark = gp.mark_masks.as_ref().map(|marks| marks[i]).unwrap_or(0);
        let Ok(graph) = reconstruct_temporal_class_atom(
            gp.atom_kind,
            gp.adjacency[i],
            lag_masks[i],
            mark,
            gp.n_vars,
            max_lag,
            vars,
        ) else {
            continue;
        };
        let mut atom_study = study.clone();
        atom_study.graph_posterior = None;
        atom_study.graph = graph;
        atom_study.query = CausalQuery::TemporalEffect(pulse.clone());
        atom_study.temporal_class_identification_cache = None;
        atom_study.temporal_class_posterior_identification_cache = None;
        let Ok(bundle) =
            atom_study.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &pulse)
        else {
            continue;
        };
        for (case_index, (case, indexer)) in
            bundle.envelope.envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let Ok(estimand) = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
            else {
                continue;
            };
            let Ok(design) = TemporalAtomDesign::linear(
                data,
                &estimand,
                &pulse,
                indexer,
                study.split.as_ref(),
                ctx,
            ) else {
                continue;
            };
            refute_atoms.push(EnvelopeRefuteAtom {
                key: (*key << 32) | u64::try_from(case_index).unwrap_or(u64::MAX),
                weight: *weight * case.weight.0,
                estimand,
                indexer: Some(indexer.clone()),
                original: design.effect_estimate(case.result.required_assumptions.clone()),
            });
        }
    }
    if refute_atoms.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let tabular = TabularData::new(data.storage().clone());
    let ate = AverageEffectQuery::binary_ate(pulse.treatment, pulse.outcome);
    let (reports, mut notes) = run_envelope_effect_refuters(
        &tabular,
        &ate,
        &refute_atoms,
        &mut EstimationWorkspace::default(),
        ctx,
        study.refute,
        EstimatorId::TemporalLinearAdjustment.as_str(),
        &study.custom_validators,
        Some(&pulse),
        study.split.as_ref(),
        Some(data.time_index()),
    )?;
    notes.insert(
        0,
        Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "InterventionResponse cheap/full run the TemporalDag Pulse-native suite on each \
             contributing class-posterior atom against that atom's own Pulse witness and mix \
             by frozen graph weight; the mixed check passes only if every atom passes",
        ),
    );
    Ok((reports, notes))
}
