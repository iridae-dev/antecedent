//! ADMG graph-posterior response surfaces: general.id + functional.effect per atom.
//!
//! Atoms are single ADMGs, not MEC completions. Function-valued surfaces mix by
//! frozen posterior weight when [`StructuralAggregationPolicy`] is
//! `SameEstimandWeightedMean`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{
    estimate_general_id_response, graph_posterior_response_supported, mix_response_values,
    mix_support_reports, response_envelope_from_weighted, response_identified_value,
    response_primary_pair, response_scalar_summary,
};
use super::*;
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

struct AdmgPosteriorResponseAtom {
    estimand: IdentifiedEstimand,
}

impl super::Study {
    /// Mix ADMG posterior atoms through `identify_admg_query` + `functional.effect`.
    pub(super) fn execute_admg_graph_posterior_response(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if !matches!(gp.atom_kind, antecedent_discovery::GraphPosteriorAtomKind::Admg) {
            return Err(CausalError::Compile {
                message: "execute_admg_graph_posterior_response requires Admg posterior atoms"
                    .into(),
            });
        }
        graph_posterior_response_supported(query)?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let (identified, identify_cached) = if let Some(cache) =
            self.graph_posterior_identification_cache.as_deref()
        {
            (cache.clone(), true)
        } else {
            (
                    crate::analysis::prepared::build_admg_graph_posterior_response_identification_cache(
                        gp, query, ctx,
                    )?,
                    false,
                )
        };
        if identified.atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "ADMG graph-posterior response: no identified ADMG atoms".into(),
            });
        }

        let mut atoms = identified
            .graphs
            .graph_keys
            .iter()
            .zip(identified.graphs.weights.iter())
            .zip(identified.graphs.identified.iter())
            .map(|((&graph_key, &weight), flag)| StructuralResponseAtom {
                posterior: None,
                response: None,
                graph_key,
                weight,
                status: if *flag == GraphIdentFlag::Identified {
                    IdentificationStatus::NonparametricallyIdentified
                } else {
                    IdentificationStatus::NotIdentified
                },
                value: None,
            })
            .collect::<Vec<_>>();
        let mut weighted = Vec::new();
        let mut evals = Vec::new();
        let mut primary = None;
        let mut failed_mass = 0.0;
        let work: Vec<_> = identified
            .atoms
            .iter()
            .filter(|atom| identified_weight_for_key(&identified.graphs, atom.key) > 0.0)
            .collect();
        let fitted = ctx.map_indexed(work.len(), |i, inner| {
            let atom = work[i];
            let mask = mask_for_atom_key(gp, atom.key)?;
            let response = estimate_admg_posterior_atom_response(
                self,
                data,
                query,
                mask,
                gp.n_vars,
                &atom.identification,
                &atom.estimand,
                inner,
            );
            Ok::<_, CausalError>((atom, response))
        })?;
        for (atom, response) in fitted {
            let weight = identified_weight_for_key(&identified.graphs, atom.key);
            let Ok(response) = response else {
                failed_mass += weight;
                for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                    slot.status = atom.identification.status;
                }
                continue;
            };
            let value =
                response_identified_value(&response).ok_or_else(|| CausalError::Compile {
                    message: "identified ADMG graph-posterior response atom had no numerical value"
                        .into(),
                })?;
            for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                slot.status = atom.identification.status;
                slot.value = Some(value.clone());
                slot.response = Some(response.clone());
            }
            if primary.is_none() {
                primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            weighted.push((atom.key, weight, response.clone()));
            evals.push(AdmgPosteriorResponseAtom { estimand: atom.estimand.clone() });
        }

        let (estimand, mut identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "ADMG graph-posterior response has no evaluable identified atom".into(),
        })?;
        let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
        let total_mass = identified.graphs.total_weight();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let contributing: Vec<&IdentifiedEstimand> =
            evals.iter().map(|eval| &eval.estimand).collect();
        let any_partial = weighted.iter().any(|(key, _, response)| {
            identified.atoms.iter().find(|atom| atom.key == *key).is_some_and(|atom| {
                matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified)
            }) || response.identification_status == IdentificationStatus::PartiallyIdentified
        });
        let aggregation_policy = resolve_structural_aggregation(&contributing, any_partial);
        let conditional_values = weighted
            .iter()
            .filter_map(|(_, weight, response)| {
                response_identified_value(response).map(|value| (*weight, value))
            })
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let conditional = mix_response_values(&conditional_refs)?;
        let structural_set = response_envelope_from_weighted(&weighted);
        let first = &weighted[0].2;
        let graph_dependent = unidentified_mass > 0.0
            || failed_mass > 0.0
            || matches!(aggregation_policy, StructuralAggregationPolicy::GraphDependentAtoms);
        if graph_dependent {
            identification.status = IdentificationStatus::GraphDependent;
        } else if matches!(aggregation_policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            identification.status = IdentificationStatus::PartiallyIdentified;
        }
        let uncertainty =
            if weighted.len() == 1 { first.uncertainty.clone() } else { ResponseUncertainty::None };
        let mut assumptions = first.assumptions.clone();
        for (_, _, response) in weighted.iter().skip(1) {
            assumptions.extend_unique(&response.assumptions.entries);
        }
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: if graph_dependent {
                ResponseIdentification::GraphDependent(
                    weighted
                        .iter()
                        .filter_map(|(key, _, response)| {
                            response_identified_value(response).map(|value| (*key, value))
                        })
                        .collect(),
                )
            } else {
                ResponseIdentification::PointIdentified(conditional.clone())
            },
            uncertainty,
            support: mix_support_reports(
                &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
            ),
            assumptions,
            provenance_id: Arc::from("estimate.response.admg_graph_posterior"),
            horizon_identification: None,
            interaction_structurally_zero: first.interaction_structurally_zero,
        };
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );

        let mut diagnostics = vec![
            Diagnostic::new(
                "estimate.response.admg_graph_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each ADMG posterior atom was identified with general.id and estimated with \
                     functional.effect; outer weights are posterior probabilities; ADMG atoms are \
                     single graphs, not MEC completions; identified_mass={}; \
                     unidentified_mass={}; unevaluable_mass={}",
                    identified_mass / total_mass,
                    unidentified_mass / total_mass,
                    failed_mass / total_mass,
                ),
            )
            .with_fields(mass_fields(
                Some(identified_mass / total_mass),
                unidentified_mass / total_mass,
            )),
            Diagnostic::new(
                "identify.response.general_id",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "intervention mean identified by Shpitser–Pearl ID; levels are the discrete \
                 functional.effect plug-in, not an adjustment g-formula",
            ),
        ];
        if weighted.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.response.admg_graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if matches!(self.inference, InferenceMode::Bayesian(_)) {
                    "multi-atom Bayesian ADMG graph-posterior response withholds an aggregate \
                     credible band: graph-specific dispersion and a frozen-weight aggregate are \
                     different objects"
                } else {
                    "multi-atom Frequentist ADMG graph-posterior response withholds an aggregate \
                     band: atom surfaces are mixed by frozen graph weight and their marginal \
                     bands are not combined as independent"
                },
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            aggregation_policy,
            identified_mass / total_mass,
            unidentified_mass / total_mass,
            failed_mass / total_mass,
            0.0,
        );

        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diagnostics) = if is_intervention
            && scalar.is_finite()
            && matches!(aggregation_policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     identified intervention mean",
                ));
            }
            run_plugin_level_refuters(
                data,
                &estimand,
                &ate_query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                EstimatorId::FunctionalEffect.as_str(),
                &self.custom_validators,
            )?
        } else {
            if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if matches!(aggregation_policy, StructuralAggregationPolicy::GraphDependentAtoms)
                    {
                        "query-native validation does not compare refuters against a withheld \
                         aggregate under GraphDependentAtoms"
                    } else {
                        "scalar ATE refuters are not applicable to a function-valued ADMG \
                         graph-posterior response"
                    },
                ));
            }
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diagnostics);
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect \
                 summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralId,
            estimator_id: EstimatorId::FunctionalEffect,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.admg_graph_posterior",
                    aggregation_policy.as_str(),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                structural_response: Some(StructuralResponseMixture {
                    weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
                    atoms,
                    identified_mass: identified_mass / total_mass,
                    unidentified_mass: unidentified_mass / total_mass,
                    unevaluable_mass: failed_mass / total_mass,
                    subsampled_out_mass: 0.0,
                    identified_set: structural_set,
                    identified_set_interval: None,
                    conditional_on_identified: Some(conditional),
                    full_mass_scope: true,
                    truncated_atoms: 0,
                }),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }
}

fn mask_for_atom_key(gp: &GraphPosterior, key: u64) -> Result<u64, CausalError> {
    gp.graph_keys
        .iter()
        .zip(gp.adjacency.iter())
        .find(|(candidate, _)| **candidate == key)
        .map(|(_, mask)| *mask)
        .ok_or_else(|| CausalError::Compile {
            message: format!("ADMG graph-posterior atom missing adjacency mask for key {key:x}"),
        })
}

fn estimate_admg_posterior_atom_response(
    study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    mask: u64,
    n_vars: usize,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let bayesian = matches!(study.inference, InferenceMode::Bayesian(_));
    match &query.functional {
        ResponseFunctional::InterventionResponse { .. } => {
            let (mut response, _) =
                estimate_general_id_response(data, query, identification, estimand, ctx)?;
            if bayesian {
                let (estimate, _) = study.estimate_functional_effect(
                    data,
                    estimand,
                    identification,
                    &[treatment, outcome],
                    ctx,
                )?;
                if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(slot)) =
                    &mut response.estimate
                {
                    *slot = estimate.ate;
                }
            }
            Ok(response)
        }
        ResponseFunctional::MeanCurve { outcome, treatment: tx } => {
            let admg = antecedent_discovery::admg_from_adjacency_mask(mask, n_vars)?;
            let levels =
                tx.grid.values().map_err(|e| CausalError::Compile { message: e.to_string() })?;
            let mut means = Vec::with_capacity(levels.len());
            let mut grid = Vec::with_capacity(levels.len());
            let mut support = None;
            for level in levels {
                let level_query = admg_response_at_level(query, tx.variable, *outcome, level);
                let (level_id, level_est) = if means.is_empty() {
                    (identification.clone(), estimand.clone())
                } else {
                    let level_id = identify_admg_query(
                        IdentifierId::GeneralId,
                        &admg,
                        &CausalQuery::Response(level_query.clone()),
                    )?;
                    let level_est = select_estimand(&level_id, EstimatorId::FunctionalEffect)?;
                    (level_id, level_est)
                };
                let mean = if bayesian {
                    let (estimate, _) = study.estimate_functional_effect(
                        data,
                        &level_est,
                        &level_id,
                        &[tx.variable, *outcome],
                        ctx,
                    )?;
                    estimate.ate
                } else {
                    let (response, _) = estimate_general_id_response(
                        data,
                        &level_query,
                        &level_id,
                        &level_est,
                        ctx,
                    )?;
                    let (scalar, _) = response_scalar_summary(&response);
                    if support.is_none() {
                        support = Some(response.support);
                    }
                    scalar
                };
                grid.push(level);
                means.push(mean);
            }
            let mut response_support = support.unwrap_or_else(|| {
                support_from_functional_eval(None).expect("empty functional eval publishes support")
            });
            let (min, max) = (
                grid.iter().copied().fold(f64::INFINITY, f64::min),
                grid.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            response_support.query_region = antecedent_core::SupportRegion {
                minima: Arc::from(vec![min]),
                maxima: Arc::from(vec![max]),
            };
            Ok(CausalResponse {
                estimand: query.functional.clone(),
                identification_status: identification.status,
                estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                    grid: Arc::from(grid),
                    dimension: 1,
                    mean: Arc::from(means),
                }),
                uncertainty: ResponseUncertainty::None,
                support: response_support,
                assumptions: identification.required_assumptions.clone(),
                provenance_id: Arc::from("estimate.response.admg_graph_posterior"),
                horizon_identification: None,
                interaction_structurally_zero: false,
            })
        }
        _ => Err(CausalError::Unsupported {
            message: "ADMG graph-posterior response is licensed for InterventionResponse and \
                      ResponseCurve",
        }),
    }
}

fn admg_response_at_level(
    query: &ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    level: f64,
) -> ResponseQuery {
    let mut level_query = query.clone();
    level_query.functional = ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from([Intervention::set(
            treatment,
            antecedent_core::Value::f64(level),
        )]),
    };
    level_query
}

