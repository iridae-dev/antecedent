// SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::mix_support_reports;
use super::temporal_path::{
    aggregate_temporal_horizon_evidence, enforce_temporal_response_memory_budget,
    horizon_adjustment_sets_differ, resolve_temporal_response_prior,
};
use super::*;

/// TemporalDag graph-posterior response: complete-observation MeanCurve or
/// one-coordinate InterventionResponse. Sequence overlays stay on the explicit path.
pub(crate) fn dbn_posterior_response_supported(query: &ResponseQuery) -> Result<(), CausalError> {
    if query.temporal.is_none() {
        return Err(CausalError::Unsupported {
            message: "DBN-posterior response requires TemporalResponseSpec",
        });
    }
    if query.observation != ObservationSpec::Complete {
        return Err(CausalError::Unsupported {
            message: "DBN-posterior response currently requires complete observations",
        });
    }
    if matches!(
        &query.functional,
        ResponseFunctional::InterventionResponse { interventions, .. } if interventions.len() != 1
    ) {
        return Err(CausalError::Unsupported {
            message: "graph-posterior InterventionResponse currently requires one intervention coordinate",
        });
    }
    if matches!(
        antecedent_estimate::plan_from_response_query(query),
        Ok(Some(
            antecedent_estimate::TemporalInterventionPlan::Sequential { .. }
                | antecedent_estimate::TemporalInterventionPlan::Mechanisms { .. }
        ))
    ) {
        return Err(CausalError::Unsupported {
            message: "DBN-posterior response supports TemporalResponseEstimator surfaces; \
                      Sequence overlays remain on the TemporalDag explicit path",
        });
    }
    Ok(())
}

impl super::Study {
    pub(super) fn execute_dbn_posterior_response(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if !matches!(gp.atom_kind, antecedent_discovery::GraphPosteriorAtomKind::Dag) {
            return Err(CausalError::Unsupported {
                message: "DBN-posterior response is licensed for TemporalDag atoms; \
                          TemporalCpdag/TemporalPag posterior atoms use the class-aware combiner",
            });
        }
        dbn_posterior_response_supported(query)?;
        query
            .require_licensed_temporal_observation()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "DBN-posterior response requires TemporalResponseSpec".into(),
        })?;
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::TemporalResponseBayesian
        } else {
            EstimatorId::TemporalResponseGcomp
        };
        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_dbn_posterior_response_identification_cache(
                        gp,
                        &vars,
                        query,
                        estimator_id,
                        ctx,
                    )?,
                    false,
                )
            };

        let mut atoms = identified
            .graphs
            .graph_keys
            .iter()
            .zip(identified.graphs.weights.iter())
            .zip(identified.graphs.identified.iter())
            .map(|((&graph_key, &weight), flag)| crate::result::StructuralResponseAtom {
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
        let mut primary = None;
        let mut failed_mass = 0.0;
        let mut conflict_summary = None;
        for atom in identified.atoms.iter() {
            let weight = identified_weight_for_key(&identified.graphs, atom.key);
            if weight <= 0.0 {
                continue;
            }
            let horizons = atom.horizons.as_ref().ok_or_else(|| CausalError::Compile {
                message: "DBN-posterior response atom missing per-horizon identification".into(),
            })?;
            let aligned = temporal
                .horizons
                .iter()
                .map(|&horizon| {
                    horizons.get(horizon).ok_or_else(|| CausalError::Compile {
                        message: format!("DBN-posterior response missing horizon {horizon}"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (aggregate_status, aggregate_assumptions) = aggregate_temporal_horizon_evidence(
                aligned.iter().map(|entry| &entry.identification),
            )?;
            let identifications: Vec<_> =
                aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();
            let mut estimator = TemporalResponseEstimator::new();
            estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
            let response = if let InferenceMode::Bayesian(cfg) = &self.inference {
                let mut bayes = bayesian_gcomp(cfg, ctx);
                let (resolved, conflict) = resolve_temporal_response_prior(
                    cfg, data, temporal, &aligned, treatment, outcome, ctx,
                )?;
                if conflict_summary.is_none() {
                    conflict_summary = conflict;
                }
                bayes.prior = resolved;
                estimator.estimate_bayesian(
                    data,
                    &identifications,
                    query,
                    aggregate_status,
                    aggregate_assumptions,
                    &bayes,
                    ctx,
                )
            } else {
                estimator.estimate(
                    data,
                    &identifications,
                    query,
                    aggregate_status,
                    aggregate_assumptions,
                    ctx,
                )
            };
            if let Err(err @ antecedent_estimate::EstimationError::Refused { .. }) = response {
                return Err(err.into());
            }
            let Ok(response) = response else {
                failed_mass += weight;
                for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                    slot.status = atom.identification.status;
                }
                continue;
            };
            let value =
                super::response_path::response_identified_value(&response).ok_or_else(|| {
                    CausalError::Compile {
                        message: "identified DBN-posterior response atom had no numerical value"
                            .into(),
                    }
                })?;
            for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                slot.status = atom.identification.status;
                slot.value = Some(value.clone());
                slot.response = Some(response.clone());
            }
            if primary.is_none() {
                primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            weighted.push((atom.key, weight, response));
        }

        let (estimand, mut identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "DBN-posterior response has no evaluable identified atom".into(),
        })?;
        let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
        let total_mass = identified.graphs.total_weight();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let contributing: Vec<&IdentifiedEstimand> = weighted
            .iter()
            .filter_map(|(key, _, _)| {
                identified.atoms.iter().find(|atom| atom.key == *key).and_then(|atom| {
                    atom.horizons.as_ref()?.get(temporal.horizons[0]).map(|entry| &entry.estimand)
                })
            })
            .collect();
        let any_partial = weighted.iter().any(|(key, _, response)| {
            identified.atoms.iter().find(|atom| atom.key == *key).is_some_and(|atom| {
                atom.horizons.as_ref().is_some_and(|horizons| {
                    temporal.horizons.iter().any(|&horizon| {
                        horizons.get(horizon).is_some_and(|entry| {
                            entry.identification.status == IdentificationStatus::PartiallyIdentified
                        })
                    })
                })
            }) || response.identification_status == IdentificationStatus::PartiallyIdentified
        });
        let aggregation_policy = resolve_structural_aggregation(&contributing, any_partial);
        let conditional_values = weighted
            .iter()
            .filter_map(|(_, weight, response)| {
                super::response_path::response_identified_value(response)
                    .map(|value| (*weight, value))
            })
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let conditional = super::response_path::mix_response_values(&conditional_refs)?;
        let structural_set = super::response_path::response_envelope_from_weighted(&weighted);
        let first = &weighted[0].2;
        let graph_dependent = unidentified_mass > 0.0 || failed_mass > 0.0;
        if graph_dependent {
            identification.status = IdentificationStatus::GraphDependent;
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
                            super::response_path::response_identified_value(response)
                                .map(|value| (*key, value))
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
            provenance_id: Arc::from("estimate.response.dbn_posterior"),
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
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );

        let mut diagnostics = vec![
            Diagnostic::new(
                "estimate.response.dbn_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "frozen posterior_probability weights; identified_mass={}, \
                     unidentified_mass={}; unevaluable_mass={}; TemporalDag atoms, not \
                     completion enumeration",
                    identified_mass / total_mass,
                    unidentified_mass / total_mass,
                    failed_mass / total_mass
                ),
            )
            .with_fields(super::mass_fields(
                Some(identified_mass / total_mass),
                unidentified_mass / total_mass,
            )),
        ];
        let aligned_all = identified
            .atoms
            .iter()
            .filter_map(|atom| atom.horizons.as_ref())
            .flat_map(|horizons| {
                temporal.horizons.iter().filter_map(|&horizon| horizons.get(horizon))
            })
            .collect::<Vec<_>>();
        if horizon_adjustment_sets_differ(&aligned_all) {
            diagnostics.push(Diagnostic::new(
                "identify.temporal_response.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons or atoms; each cell uses I(h) \
                 identified for that atom and horizon",
            ));
        }
        if weighted.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.response.dbn_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if matches!(self.inference, InferenceMode::Bayesian(_)) {
                    "multi-atom Bayesian DBN-posterior response withholds an aggregate \
                     credible band: graph-specific dispersion and a frozen-weight aggregate \
                     are different objects"
                } else {
                    "multi-atom Frequentist DBN-posterior response withholds an aggregate \
                     band: atom surfaces are mixed by frozen graph weight and their \
                     marginal bands are not combined as independent"
                },
            ));
        }
        if let Some(summary) = conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, summary);
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
        let (refutations, predictive_checks, refute_diags) = if is_intervention
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            self.refute_dbn_posterior_intervention(
                data,
                query,
                &identified,
                &weighted,
                gp,
                &vars,
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
            (Vec::new(), Vec::new(), Vec::new())
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
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::TemporalBackdoorUnfolded,
            estimator_id,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                structural_response: Some(crate::result::StructuralResponseMixture {
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
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    fn refute_dbn_posterior_intervention(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
        weighted: &[(u64, f64, CausalResponse)],
        gp: &GraphPosterior,
        vars: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<
        (
            Vec<antecedent_validate::RefutationReport>,
            Vec<antecedent_validate::PredictiveCheckReport>,
            Vec<Diagnostic>,
        ),
        CausalError,
    > {
        let pulse = response_pulse_witness(query)?;
        let ate_query = AverageEffectQuery::binary_ate(pulse.treatment, pulse.outcome);
        let tabular = TabularData::new(data.storage().clone());
        let mut notes = vec![Diagnostic::new(
            "refute.dbn_posterior.intervention_pulse_suite",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "InterventionResponse cheap/full run the TemporalDag Pulse-native suite on each \
             contributing atom against that atom's own Pulse witness and mix by frozen graph \
             weight; the mixed check passes only if every atom passes",
        )];
        match &self.inference {
            InferenceMode::Frequentist => {
                let mut refute_atoms = Vec::new();
                for (key, weight, _) in weighted {
                    let atom = identified
                        .atoms
                        .iter()
                        .find(|candidate| candidate.key == *key)
                        .ok_or_else(|| CausalError::Compile {
                            message: "DBN-posterior IR refuter missing contributing atom".into(),
                        })?;
                    let (estimand, indexer) = pulse_witness_design(atom, pulse.horizon_steps)?;
                    let design = TemporalAtomDesign::linear(
                        data,
                        estimand,
                        &pulse,
                        indexer,
                        self.split.as_ref(),
                        ctx,
                    )?;
                    refute_atoms.push(EnvelopeRefuteAtom {
                        key: *key,
                        weight: *weight,
                        estimand: estimand.clone(),
                        indexer: Some(indexer.clone()),
                        original: design
                            .effect_estimate(atom.identification.required_assumptions.clone()),
                    });
                }
                let (reports, mix_notes) = run_envelope_effect_refuters(
                    &tabular,
                    &ate_query,
                    &refute_atoms,
                    &mut EstimationWorkspace::default(),
                    ctx,
                    self.refute,
                    "temporal.linear.adjustment",
                    &self.custom_validators,
                    Some(&pulse),
                    self.split.as_ref(),
                    Some(data.time_index()),
                )?;
                notes.extend(mix_notes);
                Ok((reports, Vec::new(), notes))
            }
            InferenceMode::Bayesian(cfg) => {
                let bayes = bayesian_temporal_gcomp(cfg, ctx);
                let mut ws = BayesianGCompWorkspace::default();
                let mut fits = Vec::new();
                let mut draws = Vec::new();
                for (key, weight, _) in weighted {
                    let atom = identified
                        .atoms
                        .iter()
                        .find(|candidate| candidate.key == *key)
                        .ok_or_else(|| CausalError::Compile {
                            message: "DBN-posterior IR refuter missing contributing atom".into(),
                        })?;
                    let (estimand, indexer) = pulse_witness_design(atom, pulse.horizon_steps)?;
                    let mut temporal_est = TemporalLinearAdjustment::new();
                    temporal_est.inner.overlap = OverlapPolicy::ExplicitOverride;
                    let prep = temporal_est
                        .prepare(
                            data,
                            estimand,
                            &pulse,
                            indexer,
                            self.split.as_ref(),
                            &ctx.kernel_policy,
                        )
                        .map_err(CausalError::from)?;
                    let names = antecedent_estimate::temporal_coefficient_names(
                        data, estimand, &pulse, indexer,
                    )
                    .map_err(CausalError::from)?;
                    let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                        .map_err(CausalError::from)?;
                    let posterior = bayes.fit(&bprep, atom.identification.status, &mut ws, ctx)?;
                    let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                        message: "Bayesian Pulse witness posterior missing effect column".into(),
                    })?;
                    let effect = posterior
                        .draws
                        .column(col)
                        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                    draws.push(GraphEffectDraws {
                        graph_key: *key,
                        effect_draws: Arc::from(effect.to_vec()),
                    });
                    fits.push(EnvelopeAtomFit {
                        key: *key,
                        prep: bprep,
                        posterior,
                        status: atom.identification.status,
                        weight: *weight,
                        estimand: estimand.clone(),
                        indexer: Some(indexer.clone()),
                        prior: bayes.inner.prior.clone(),
                    });
                }
                let (reports, mix_notes) = run_envelope_effect_refuters(
                    &tabular,
                    &ate_query,
                    &envelope_refute_atoms(&fits)?,
                    &mut EstimationWorkspace::default(),
                    ctx,
                    self.refute,
                    EstimatorId::BayesianTemporalGcomp.as_str(),
                    &self.custom_validators,
                    Some(&pulse),
                    self.split.as_ref(),
                    Some(data.time_index()),
                )?;
                notes.extend(mix_notes);
                let flags = weighted
                    .iter()
                    .map(|(key, _, _)| (*key, GraphIdentFlag::Identified))
                    .collect::<Vec<_>>();
                let graphs = WeightedGraphSamples::new(
                    weighted.iter().map(|(_, weight, _)| *weight).collect::<Vec<_>>(),
                    flags.iter().map(|(_, flag)| *flag).collect::<Vec<_>>(),
                    weighted.iter().map(|(key, _, _)| *key).collect::<Vec<_>>(),
                )
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                let mut posterior = aggregate_effect_envelope(
                    &graphs,
                    &draws,
                    InferenceDiagnostics::analytic("dbn_posterior_response_pulse_witness"),
                    EnvelopeOptions::default(),
                )
                .map_err(CausalError::from)?;
                let mut refutations = reports;
                let pulse_mean = posterior
                    .effect_column()
                    .and_then(|eq| posterior.summaries.mean.get(eq).copied())
                    .unwrap_or(f64::NAN);
                let predictive = run_envelope_bayesian_full_validation(
                    self.refute,
                    cfg,
                    &bayes.inner,
                    &fits,
                    &mut posterior,
                    pulse_mean,
                    ctx,
                    super::super::latency::predictive_check_sims(self.latency_mode),
                    &mut refutations,
                    &mut notes,
                )?;
                let _ = (gp, vars);
                Ok((refutations, predictive, notes))
            }
        }
    }
}

fn response_pulse_witness(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN-posterior InterventionResponse requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
    let horizon = *temporal.horizons.first().ok_or_else(|| CausalError::Compile {
        message: "DBN-posterior InterventionResponse requires at least one horizon".into(),
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

fn pulse_witness_design(
    atom: &crate::analysis::prepared::CachedDbnPosteriorAtomIdentification,
    horizon: u32,
) -> Result<(&IdentifiedEstimand, &antecedent_data::TemporalIndexer), CausalError> {
    if let Some(horizons) = atom.horizons.as_ref() {
        if let Some(entry) = horizons.get(horizon) {
            return Ok((&entry.estimand, &entry.indexer));
        }
    }
    Ok((&atom.estimand, &atom.indexer))
}
