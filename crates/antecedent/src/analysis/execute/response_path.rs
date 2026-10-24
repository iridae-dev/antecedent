// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitKey, CompletionFitMemo};
use super::{
    bayesian_gcomp, elapsed_ns, envelope_se_omits_between_atom_variance,
    envelope_to_identification_result_for, functional_cell_unevaluable,
    identification_from_cache_or, identification_status_ok_for_case,
    identified_set_interval_diagnostics, identify_cached_diagnostic, identify_static_query,
    interactive_subsample_graphs_accounted, provenance_ids, report_identify_compute,
    resolve_posterior_status, response_functional_is_derivative, scalar_identified_set,
    support_from_functional_eval, Arc, AverageEffectQuery, CausalError, CausalQuery,
    CausalResponse, ClassEnvelopeRun, ContinuousResponseEstimator, Dag, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimandMethod, EstimationWorkspace, EstimatorId,
    EvalError, ExecuteRun, ExecutionContext, FunctionalDistributionWorkspace, FunctionalEffect,
    GraphClass, GraphIdentFlag, GraphPosterior, IdentificationResult, IdentificationStatus,
    IdentifiedClaim, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant, Intervention,
    ObservationMechanismEstimator, ObservationSpec, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, RefutationRun, RefuteSuite, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView,
    TabularData, VariableId, WeightedGraphSamples,
};
use crate::planner::RouteDefaults;

/// Treatment coordinates and outcome of a cell.aipw joint-arm query.
#[derive(Clone, Copy)]
struct CellAipwTarget<'a> {
    treatments: &'a [VariableId],
    outcome: VariableId,
}

/// Point-identified scalar response of a cell.aipw joint arm.
fn cell_aipw_response(
    query: &ResponseQuery,
    identification: &IdentificationResult,
    scalar: f64,
    se: f64,
    n_treatments: usize,
    requested_arm: u32,
) -> CausalResponse {
    CausalResponse {
        estimand: query.functional.clone(),
        identification_status: identification.status,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Scalar(scalar)),
        uncertainty: ResponseUncertainty::Scalar {
            standard_error: se,
            lower: scalar - crate::result::reported_se_interval_z() * se,
            upper: scalar + crate::result::reported_se_interval_z() * se,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        },
        support: antecedent_core::SupportReport {
            status: antecedent_core::SupportStatus::Supported,
            query_region: antecedent_core::SupportRegion {
                minima: Arc::from(
                    (0..n_treatments)
                        .map(|j| f64::from((requested_arm >> j) & 1))
                        .collect::<Vec<_>>(),
                ),
                maxima: Arc::from(
                    (0..n_treatments)
                        .map(|j| f64::from((requested_arm >> j) & 1))
                        .collect::<Vec<_>>(),
                ),
            },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            // One scalar answers the whole joint arm: one coordinate, one label (the
            // summary itself).
            point_status: Some(Arc::from([antecedent_core::SupportStatus::Supported])),
        },
        assumptions: identification.required_assumptions.clone(),
        provenance_id: Arc::from("estimate.cell.aipw"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

/// Borrowed inputs of the plugin-level refutation of a scalar response.
#[derive(Clone, Copy)]
struct ResponseRefuteInput<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    estimand: &'a IdentifiedEstimand,
    estimate: &'a EffectEstimate,
    scalar: f64,
    estimator_id: EstimatorId,
    treatment: VariableId,
    outcome: VariableId,
}

/// Borrowed inputs shared by the phases of a graph-posterior response execution.
#[derive(Clone, Copy)]
struct GraphPosteriorResponseRun<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// Atom fits of a graph-posterior response with the masses pooling accounts for.
struct GraphPosteriorResponseFits {
    graphs: WeightedGraphSamples,
    unidentified_mass: f64,
    subsampled_out_mass: f64,
    subsample_notes: Vec<Diagnostic>,
    atoms: Vec<crate::result::StructuralResponseAtom>,
    weighted: Vec<(u64, f64, CausalResponse)>,
    estimand: IdentifiedEstimand,
    failed_mass: f64,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
    options: antecedent_estimate::ContinuousResponseOptions,
}

/// The pooled response of a graph posterior with the values its diagnostics read.
struct GraphPosteriorPooled {
    identification: IdentificationResult,
    conditional: ResponseValue,
    mixed_if_se: Option<f64>,
    response: CausalResponse,
    estimate: EffectEstimate,
    scalar: f64,
    standard_error: f64,
    identified_mass: f64,
    total_mass: f64,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
}

/// Predictive checks and the posterior refutation reports added to a Bayesian level.
type PosteriorLevelChecks =
    (Vec<antecedent_validate::PredictiveCheckReport>, Vec<antecedent_validate::RefutationReport>);

/// Structural atoms of a graph posterior, one per sample, before any fit fills their values.
fn graph_posterior_structural_atoms(
    graphs: &WeightedGraphSamples,
) -> Vec<crate::result::StructuralResponseAtom> {
    // Atom statuses come from the full ensemble: a subsampled-out atom keeps
    // its identified status and carries no value.
    graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
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
        .collect()
}

impl super::Study {
    #[allow(
        clippy::too_many_lines,
        reason = "one graph-posterior response: every atom's fit, the mixture and the identified finish share the frozen posterior"
    )]
    pub(super) fn execute_graph_posterior_response(
        &self,
        data: &TabularData,
        graph_posterior: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        graph_posterior_response_supported(query)?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let identification_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let (identified, identify_cached) = if let Some(cache) =
            self.graph_posterior_identification_cache.as_ref()
        {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    graph_posterior,
                    &identification_query,
                    ctx,
                )?),
                false,
            )
        };
        let run = GraphPosteriorResponseRun { data, query, treatment, outcome, ctx };
        let (mut fits, identification) =
            self.fit_graph_posterior_response_atoms(&run, &identified)?;
        // The primary atom's own identification, before pooling may mark the
        // mixture graph-dependent: the Bayesian refuter state refits that atom.
        let primary_status = identification.status;
        let primary_assumptions = identification.required_assumptions.clone();
        let pooled = self.pool_graph_posterior_response(&run, &fits, identification)?;
        let estimator_id = if self.inference.is_bayesian() {
            EstimatorId::ResponseBayesian
        } else {
            EstimatorId::default_for_response(&query.functional)
        };
        let mut diagnostics = self.graph_posterior_response_diagnostics(&fits, &pooled);
        let (mut refutations, refute_diags) = self.refute_graph_posterior_response(
            &run,
            estimator_id,
            &fits.estimand,
            &pooled,
            &mut diagnostics,
        )?;
        diagnostics.extend(refute_diags);
        let (predictive_checks, posterior_reports) = self.bayesian_graph_posterior_level_state(
            &run,
            &fits,
            (primary_status, &primary_assumptions),
            &mut diagnostics,
        )?;
        refutations.extend(posterior_reports);
        diagnostics.extend(std::mem::take(&mut fits.subsample_notes));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let GraphPosteriorPooled {
            identification,
            conditional,
            response,
            estimate,
            identified_mass,
            total_mass,
            structural_set,
            ..
        } = pooled;
        let GraphPosteriorResponseFits {
            atoms,
            estimand,
            unidentified_mass,
            failed_mass,
            subsampled_out_mass,
            ..
        } = fits;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                response: Some(response),
                structural_response: Some(crate::result::StructuralResponseMixture {
                    weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
                    atoms,
                    identified_mass: identified_mass / total_mass,
                    unidentified_mass: unidentified_mass / total_mass,
                    unevaluable_mass: failed_mass / total_mass,
                    subsampled_out_mass: subsampled_out_mass / total_mass,
                    identified_set: structural_set,
                    identified_set_interval: None,
                    conditional_on_identified: Some(conditional),
                    full_mass_scope: true,
                    truncated_atoms: 0,
                }),
                diagnostics: Some(diagnostics),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Bayesian additions to the plugin-level graph-posterior refutation: the predictive
    /// checks and, under `full`, level-native prior sensitivity and MCMC diagnostics,
    /// evaluated on the primary identified atom (the atom whose estimand the plugin-level
    /// suite reads). Returns the predictive checks and the extra refutation reports;
    /// Frequentist studies and `none` return nothing.
    fn bayesian_graph_posterior_level_state(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        fits: &GraphPosteriorResponseFits,
        primary: (IdentificationStatus, &antecedent_core::AssumptionSet),
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<PosteriorLevelChecks, CausalError> {
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Ok((Vec::new(), Vec::new()));
        };
        if !matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full)
            || !matches!(run.query.functional, ResponseFunctional::InterventionResponse { .. })
        {
            return Ok((Vec::new(), Vec::new()));
        }
        let Some((_, _, primary_response)) = fits.weighted.first() else {
            return Ok((Vec::new(), Vec::new()));
        };
        let (level, level_sd) = response_scalar_summary(primary_response);
        let mut fitter =
            ContinuousResponseEstimator::new(Arc::clone(&fits.estimand.adjustment_set));
        fitter.options = fits.options.clone();
        let mut estimator = bayesian_gcomp(config, run.ctx);
        estimator.prior.clone_from(&config.prior);
        let (status, assumptions) = primary;
        let state = super::bayesian_level_refutation::refute_bayesian_intervention_level(
            &super::bayesian_level_refutation::BayesianLevelRefutationInput {
                data: run.data,
                query: run.query,
                estimand: &fits.estimand,
                fitter: &fitter,
                estimator: &estimator,
                status,
                assumptions,
                level,
                level_sd,
                treatment: run.treatment,
                outcome: run.outcome,
                suite: self.refute,
                latency: self.latency_mode,
                // The plugin-level suite already ran on the pooled level; only the
                // posterior checks are added here.
                plugin_level: false,
                summarize: response_scalar_summary,
                target_label: "intervention-level",
            },
            run.ctx,
        )?;
        diagnostics.extend(state.diagnostics);
        diagnostics.push(Diagnostic::new(
            "refute.response.graph_posterior.bayesian_primary_atom",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the predictive checks and prior sensitivity of this Bayesian graph-posterior \
             intervention level are evaluated on the primary identified atom, the atom whose \
             estimand the plugin-level suite reads; they are not mixed across atoms",
        ));
        Ok((state.predictive_checks, state.refutations))
    }

    /// Fits each identified atom of the graph posterior and collects the weighted responses.
    fn fit_graph_posterior_response_atoms(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        identified: &crate::analysis::prepared::CachedGraphPosteriorIdentification,
    ) -> Result<(GraphPosteriorResponseFits, IdentificationResult), CausalError> {
        let GraphPosteriorResponseRun { data, query, ctx, .. } = *run;
        // Same interactive budget as the graph-posterior ATE path: identified
        // atoms left out of the stratified subsample are never fit and never
        // renormalized away; their mass is reported as not evaluated.
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            identified.graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let mut weighted = Vec::new();
        let mut atoms = graph_posterior_structural_atoms(&identified.graphs);
        let mut primary = None;
        let mut failed_mass = 0.0;
        // Atom influence columns keep their own complete-case row indices; they
        // are aligned on the shared data rows before mixing.
        let mut atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)> = Vec::new();
        let options = self.response_options.clone().unwrap_or_default();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        let work: Vec<_> =
            identified.atoms.iter().filter(|atom| key_weights.get(atom.key) > 0.0).collect();
        let fitted = ctx.map_indexed(work.len(), |i, inner| {
            let atom = work[i];
            let mut estimator =
                ContinuousResponseEstimator::new(Arc::clone(&atom.estimand.adjustment_set));
            estimator.options = options.clone();
            let scored = if let InferenceMode::Bayesian(cfg) = &self.inference {
                let mut bayes = bayesian_gcomp(cfg, inner);
                bayes.prior.clone_from(&cfg.prior);
                estimator
                    .estimate_bayesian(
                        data,
                        query,
                        atom.identification.status,
                        atom.identification.required_assumptions.clone(),
                        &bayes,
                        inner,
                    )
                    .map(|response| (response, None))
            } else {
                estimator.estimate_identified_scored(
                    data,
                    query,
                    atom.identification.status,
                    atom.identification.required_assumptions.clone(),
                )
            };
            Ok::<_, CausalError>((atom, scored))
        })?;
        for (atom, response) in fitted {
            let weight = key_weights.get(atom.key);
            let response = match response {
                Err(err @ antecedent_estimate::EstimationError::Refused { .. }) => {
                    return Err(err.into());
                }
                Err(_) => {
                    failed_mass += weight;
                    for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key)
                    {
                        slot.status = atom.identification.status;
                    }
                    continue;
                }
                Ok((response, scores)) => {
                    if let Some(scores) = scores.filter(|s| !s.columns.is_empty()) {
                        atom_scores.push((weight, scores));
                    }
                    response
                }
            };
            let value =
                response_identified_value(&response).ok_or_else(|| CausalError::Compile {
                    message: "identified graph-posterior response atom had no numerical value"
                        .into(),
                })?;
            for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                slot.status = atom.identification.status;
                slot.value = Some(value.clone());
            }
            if primary.is_none() {
                primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            weighted.push((atom.key, weight, response));
        }
        let (estimand, identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior response has no evaluable identified atom".into(),
        })?;
        let unidentified_mass = identified.graphs.unidentified_mass();
        Ok((
            GraphPosteriorResponseFits {
                graphs,
                unidentified_mass,
                subsampled_out_mass: subsample_drop.mass,
                subsample_notes,
                atoms,
                weighted,
                estimand,
                failed_mass,
                atom_scores,
                options,
            },
            identification,
        ))
    }

    /// Pools the atom responses into the published response, estimate and mass accounting.
    fn pool_graph_posterior_response(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        fits: &GraphPosteriorResponseFits,
        mut identification: IdentificationResult,
    ) -> Result<GraphPosteriorPooled, CausalError> {
        let GraphPosteriorResponseRun { data, query, .. } = *run;
        let GraphPosteriorResponseFits { graphs, weighted, atom_scores, options, .. } = fits;
        let failed_mass = fits.failed_mass;
        let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
        let total_mass = graphs.total_weight();
        // The subsampled ensemble flags dropped atoms Unidentified; the full
        // ensemble's unidentified mass is the structural one.
        let unidentified_mass = fits.unidentified_mass;
        let subsampled_out_mass = fits.subsampled_out_mass;
        let conditional_values = weighted
            .iter()
            .filter_map(|(_, weight, response)| {
                response_identified_value(response).map(|value| (*weight, value))
            })
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let conditional = mix_response_values(&conditional_refs)?;
        let structural_set = response_envelope_from_weighted(weighted);
        let first = &weighted[0].2;
        // Unevaluable and subsampled-out mass are as incomplete as unidentified
        // mass: the published value covers only the evaluated atoms.
        let graph_dependent = resolve_posterior_status(
            identification.status,
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                failed: failed_mass,
                subsampled: subsampled_out_mass,
                ..PosteriorMassSummary::default()
            },
        ) == IdentificationStatus::GraphDependent;
        if graph_dependent {
            identification.status = IdentificationStatus::GraphDependent;
        }
        // Only a scalar aggregate takes the joint-IF SE: for a curve the first
        // score column is the influence of the first grid point alone.
        let mixed_if_se = (weighted.len() > 1
            && self.inference.is_frequentist()
            && matches!(conditional, ResponseValue::Scalar(_))
            && atom_scores.len() == weighted.len())
        .then(|| mixed_response_scalar_se(atom_scores, data.row_count()))
        .flatten();
        let uncertainty = if weighted.len() == 1 {
            first.uncertainty.clone()
        } else if let (Some(se), ResponseValue::Scalar(value)) = (mixed_if_se, &conditional) {
            let level = options.confidence_level;
            let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
            ResponseUncertainty::Scalar {
                standard_error: se,
                level,
                lower: value - z * se,
                upper: value + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        } else {
            ResponseUncertainty::None
        };
        let response = antecedent_core::CausalResponse {
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
            assumptions: first.assumptions.clone(),
            provenance_id: Arc::from("estimate.response.graph_posterior"),
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
        Ok(GraphPosteriorPooled {
            identification,
            conditional,
            mixed_if_se,
            response,
            estimate,
            scalar,
            standard_error,
            identified_mass,
            total_mass,
            structural_set,
        })
    }

    /// Mass-accounting and uncertainty diagnostics of a pooled graph-posterior response.
    fn graph_posterior_response_diagnostics(
        &self,
        fits: &GraphPosteriorResponseFits,
        pooled: &GraphPosteriorPooled,
    ) -> Vec<Diagnostic> {
        let GraphPosteriorPooled { conditional, mixed_if_se, identified_mass, total_mass, .. } =
            pooled;
        let (identified_mass, total_mass) = (*identified_mass, *total_mass);
        let (unidentified_mass, failed_mass, subsampled_out_mass) =
            (fits.unidentified_mass, fits.failed_mass, fits.subsampled_out_mass);
        let weighted = &fits.weighted;
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.response.graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "posterior_probability weights; identified_mass={}, unidentified_mass={}; \
                 unevaluable_mass={}; subsampled_out_mass={}; failed estimation and \
                 atoms the Interactive tier did not evaluate are not mixed into \
                 unidentified mass",
                identified_mass / total_mass,
                unidentified_mass / total_mass,
                failed_mass / total_mass,
                subsampled_out_mass / total_mass
            ),
        )
        .with_fields(super::mass_fields(
            Some(identified_mass / total_mass),
            unidentified_mass / total_mass,
        ))];
        if weighted.len() > 1 && mixed_if_se.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if self.inference.is_bayesian() {
                    "multi-atom Bayesian graph-posterior response withholds an aggregate \
                     credible interval: graph-specific dispersion and uncertainty of a \
                     frozen-weight aggregate are different objects"
                } else if !matches!(conditional, ResponseValue::Scalar(_)) {
                    "multi-atom Frequentist graph-posterior curve withholds an aggregate \
                     band: the joint-IF SE is published for scalar aggregates only, and \
                     marginal atom bands are not combined as independent"
                } else {
                    "multi-atom Frequentist graph-posterior response withholds an aggregate \
                     interval because aligned atom influences were unavailable; marginal \
                     atom SEs are not combined as independent"
                },
            ));
        } else if mixed_if_se.is_some() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.joint_if_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "SE of the frozen-weight aggregate from the joint influence-function \
                 covariance of identified atoms on the shared sample; simultaneous bands \
                 are not claimed",
            ));
        }
        diagnostics
    }

    /// Plugin-level refutation of a scalar graph-posterior intervention response.
    fn refute_graph_posterior_response(
        &self,
        run: &GraphPosteriorResponseRun<'_>,
        estimator_id: EstimatorId,
        estimand: &IdentifiedEstimand,
        pooled: &GraphPosteriorPooled,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let GraphPosteriorResponseRun { data, query, treatment, outcome, ctx, .. } = *run;
        let GraphPosteriorPooled { conditional, scalar, standard_error, response, .. } = pooled;
        let (scalar, standard_error) = (*scalar, *standard_error);
        let plugin_scalar = match &conditional {
            ResponseValue::Scalar(value) => *value,
            _ => scalar,
        };
        let scalar_intervention = plugin_scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        // A Bayesian pooled level is the posterior-mean level: the plugin-level suite reads
        // it exactly as it reads the Frequentist level; the posterior checks are added by
        // `bayesian_graph_posterior_level_state`.
        let (refutations, refute_diags) = if scalar_intervention
            && (!matches!(self.refute, RefuteSuite::None)
                || (!self.custom_validators.is_empty() && self.inference.is_frequentist()))
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            let plugin_estimate = EffectEstimate::new(
                plugin_scalar,
                standard_error,
                response.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     g-comp level",
                ));
            }
            RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate: &plugin_estimate,
                suite: self.refute,
                estimator: estimator_id.as_str(),
                custom: &self.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, ctx)?
        } else {
            if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued \
                     graph-posterior response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        Ok((refutations, refute_diags))
    }

    /// Identify and estimate a continuous response, including licensed observation correction.
    pub(super) fn execute_response(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        checked_curve: Option<&super::super::prepared::CheckedStaticResponseCurve>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, identify_cached, identifier_id, estimator_id) =
            self.identify_static_response(graph, query, physical, checked_curve, ctx)?;
        self.finish_static_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
        )
    }

    /// Derivative or Jacobian on a latent-confounded ADMG, identified by one common adjustment
    /// set of the ADMG and estimated by the continuous response estimator on it.
    ///
    /// The adjustment set is read from the ADMG's proper back-door graph, so a bidirected edge
    /// keeps its latent common cause; nothing is coerced to a DAG. Identification that adjustment
    /// cannot reach is refused with the hedge certificate or with the ID-only boundary.
    pub(super) fn execute_admg_derivative(
        &self,
        data: &TabularData,
        admg: &antecedent_graph::Admg,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: EstimatorId::static_response_for(&query.functional, &self.inference),
        })?;
        report_identify_compute(ctx);
        let identification = crate::strategy_table::identify_admg_derivative(admg, query)?;
        let identify_cached = false;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            CausalError::Compile { message: "response identifier returned no estimand".into() }
        })?;
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }
        self.finish_static_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
        )
    }

    /// The derivative of a front-door-identified response: the additive-outcome front-door
    /// estimator on the estimand's mediators, with its whole-procedure bootstrap interval
    /// (Frequentist) or Bayesian-bootstrap credible interval, resampling `block` consecutive
    /// rows together.
    ///
    /// # Errors
    ///
    /// The estimator's refusals.
    pub(super) fn estimate_front_door_derivative_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        claim: FrontDoorClaim<'_>,
        block: usize,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        let mut estimator =
            ContinuousResponseEstimator::new(Arc::clone(&claim.estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            estimator.options = options.clone();
        }
        let inference = match &self.inference {
            InferenceMode::Frequentist => {
                antecedent_estimate::FrontDoorDerivativeInference::Frequentist {
                    replicates: self.bootstrap_replicates,
                    block,
                }
            }
            InferenceMode::Bayesian(config) => {
                antecedent_estimate::FrontDoorDerivativeInference::Bayesian {
                    draws: config.n_draws,
                    block,
                }
            }
        };
        estimator
            .front_door_derivative(
                data,
                &query.functional,
                &claim.estimand.mediators,
                claim.status,
                claim.assumptions,
                inference,
                ctx,
            )
            .map_err(CausalError::from)
    }

    /// Result of a derivative identified by the front-door criterion.
    fn finish_front_door_derivative(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let front_door = FrontDoorClaim {
            estimand: &estimand,
            status: identification.status,
            assumptions: identification.required_assumptions.clone(),
        };
        let response = self.estimate_front_door_derivative_response(
            &data_est,
            query,
            front_door.clone(),
            1,
            ctx,
        )?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.extend(response.support.warnings.iter().cloned());
        // A front-door derivative's cheap/full suite is its own refuter state (unlicensed).
        let (refutations, refute_notes) = self.refute_front_door_derivative_response(
            (&data_est, query),
            &front_door,
            &response,
            1,
            ctx,
        )?;
        diagnostics.extend(refute_notes);
        drop(front_door);
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.front_door_derivative",
                    "estimate.response.front_door_derivative",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Estimates and finishes a response whose static identification is already settled.
    #[allow(
        clippy::too_many_lines,
        reason = "one continuous response execution: observation correction, estimation, refutation and finish run in sequence"
    )]
    fn finish_static_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let cell_aipw = matches!(estimator_id, EstimatorId::CellAipw);

        if response_functional_is_derivative(&query.functional)
            && estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
        {
            return self.finish_front_door_derivative(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }

        if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
            || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
        {
            return self.finish_general_id_response(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }

        if cell_aipw {
            return self.execute_cell_aipw_response(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
                None,
            );
        }

        let (_, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;

        let (response, response_scores) =
            self.estimate_continuous_response(&data_est, query, &estimand, &identification, ctx)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let mut estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut estimate, response_scores.as_ref())?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut predictive_checks = Vec::new();
        let (refutations, refute_diags) = match &self.inference {
            inference
                if super::derivative_refutation::is_derivative(&query.functional)
                    && !matches!(self.refute, RefuteSuite::None) =>
            {
                let mut fitter =
                    ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
                if let Some(options) = &self.response_options {
                    fitter.options = options.clone();
                }
                let bayesian = match inference {
                    InferenceMode::Bayesian(config) => Some(
                        super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                            &fitter, &data_est, query, config, ctx,
                        )?,
                    ),
                    InferenceMode::Frequentist => None,
                };
                let (reports, checks, notes) = super::derivative_refutation::refute_derivative(
                    &super::derivative_refutation::DerivativeRefutationInput {
                        data: &data_est,
                        query,
                        estimand: &estimand,
                        fitter: &fitter,
                        inference,
                        bayesian: bayesian.as_ref(),
                        status: identification.status,
                        assumptions: &identification.required_assumptions,
                        response: &response,
                        suite: self.refute,
                        latency: self.latency_mode,
                        replicates: self.bootstrap_replicates,
                    },
                    ctx,
                )?;
                predictive_checks = checks;
                (reports, notes)
            }
            InferenceMode::Bayesian(config)
                if scalar.is_finite()
                    && matches!(
                        query.functional,
                        ResponseFunctional::InterventionResponse { .. }
                    )
                    && !matches!(self.refute, RefuteSuite::None) =>
            {
                let mut fitter =
                    ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
                if let Some(options) = &self.response_options {
                    fitter.options = options.clone();
                }
                let estimator =
                    super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                        &fitter, &data_est, query, config, ctx,
                    )?;
                let state = super::bayesian_level_refutation::refute_bayesian_intervention_level(
                    &super::bayesian_level_refutation::BayesianLevelRefutationInput {
                        data: &data_est,
                        query,
                        estimand: &estimand,
                        fitter: &fitter,
                        estimator: &estimator,
                        status: identification.status,
                        assumptions: &identification.required_assumptions,
                        level: scalar,
                        level_sd: standard_error,
                        treatment,
                        outcome,
                        suite: self.refute,
                        latency: self.latency_mode,
                        plugin_level: true,
                        summarize: response_scalar_summary,
                        target_label: "intervention-level",
                    },
                    ctx,
                )?;
                predictive_checks = state.predictive_checks;
                (state.refutations, state.diagnostics)
            }
            _ => self.refute_response_plugin_level(
                ResponseRefuteInput {
                    data,
                    query,
                    estimand: &estimand,
                    estimate: &estimate,
                    scalar,
                    estimator_id,
                    treatment,
                    outcome,
                },
                &mut diagnostics,
                ctx,
            )?,
        };
        diagnostics.extend(refute_diags);
        if !scalar.is_finite() {
            // `result.effect` exists for every plan, but a curve, vector, Jacobian or
            // identified set has no scalar reading. The NaN is a "not applicable", not a
            // failed computation, and the caller must read `result.response` instead.
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let estimate_provenance = if query.observation == ObservationSpec::Complete {
            provenance_ids(Arc::clone(&response.provenance_id), Arc::clone(&response.provenance_id))
        } else {
            provenance_ids(
                "estimate.response.observation_adjusted",
                "estimate.response.observation_adjusted",
            )
        };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(estimate_provenance),
                diagnostics: Some(diagnostics),
                response: Some(response),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Resolves the response procedure and its identified claim, refusing mixed adjustment sets.
    fn identify_static_response(
        &self,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        checked_curve: Option<&super::super::prepared::CheckedStaticResponseCurve>,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentificationResult, IdentifiedEstimand, bool, IdentifierId, EstimatorId),
        CausalError,
    > {
        let binding = physical.binding();
        let checked_identity = checked_curve
            .map(|plan| {
                plan.checked_route(query).map(|(identification, estimand)| {
                    let (identifier, estimator) = plan.procedure();
                    (identification.clone(), estimand.clone(), identifier, estimator)
                })
            })
            .transpose()?;
        let (identifier_id, estimator_id) = match &checked_identity {
            Some((_, _, identifier, estimator)) => (*identifier, *estimator),
            None => binding.resolve(RouteDefaults::RESPONSE)?,
        };
        let (identification, estimand, identify_cached) =
            if let Some((identification, estimand, _, _)) = checked_identity {
                (identification, estimand, true)
            } else {
                let reused = self.reuse_static_identification_if(physical, |_| {
                    binding.resolves_to(RouteDefaults::STATIC_ATE, identifier_id, estimator_id)
                });
                identification_from_cache_or(ctx, reused, || {
                    let identification = identify_static_query(
                        identifier_id,
                        graph,
                        &CausalQuery::Response(query.clone()),
                    )?;
                    let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                        CausalError::Compile {
                            message: "response identifier returned no estimand".into(),
                        }
                    })?;
                    Ok((identification, estimand))
                })?
            };
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }

        Ok((identification, estimand, identify_cached, identifier_id, estimator_id))
    }

    /// Result of a response identified by general ID or the front-door, estimated as a discrete plug-in.
    fn finish_general_id_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let (response, _) = estimate_general_id_response(
            data,
            query,
            &identification,
            &estimand,
            FunctionalBootstrap::rows(0),
            ctx,
        )?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "identify.response.general_id",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "intervention mean identified by Shpitser–Pearl ID; levels are the \
             discrete functional.effect plug-in, not an adjustment g-formula",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.general_id",
                    "estimate.response.general_id",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Fits the continuous response on the projected table under the study's inference mode.
    fn estimate_continuous_response(
        &self,
        data_est: &TabularData,
        query: &ResponseQuery,
        estimand: &IdentifiedEstimand,
        identification: &IdentificationResult,
        ctx: &ExecutionContext,
    ) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let mut response_scores = None;
        let response = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.has_transferred_source() && cfg.prior_mapping.is_none() && cfg.prior.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian response prior transfer requires a response-specific \
                              mapping (dose/intervention coordinate to outcome-regression \
                              coefficients); incompatible catalogs must not fall back to \
                              isotropic priors",
                });
            }
            let mut bayes = bayesian_gcomp(cfg, ctx);
            if cfg.has_transferred_source() {
                let (outcome, treatments) = response_prior_targets(&query.functional)?;
                let prep = response_estimator
                    .prepare_linear_response_problem(data_est, outcome, &treatments)
                    .map_err(CausalError::from)?;
                let (resolved, _) =
                    crate::inference::resolve_bayesian_prior_with_conflict(cfg, &prep, Some(ctx))?;
                bayes.prior = resolved;
            } else {
                bayes.prior.clone_from(&cfg.prior);
            }
            response_estimator.estimate_bayesian(
                data_est,
                query,
                identification.status,
                identification.required_assumptions.clone(),
                &bayes,
                ctx,
            )
        } else if query.observation == ObservationSpec::Complete {
            let (response, scores) = response_estimator
                .estimate_identified_scored(
                    data_est,
                    query,
                    identification.status,
                    identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            response_scores = scores;
            Ok(response)
        } else {
            ObservationMechanismEstimator::new(self.observation_options).estimate_mean_curve(
                &response_estimator,
                data_est,
                query,
                self.observation_delayed_entry,
                identification.status,
                identification.required_assumptions.clone(),
            )
        }
        .map_err(CausalError::from)?;
        Ok((response, response_scores))
    }

    /// Plugin-level refutation of a scalar intervention response; notes why it was skipped otherwise.
    fn refute_response_plugin_level(
        &self,
        input: ResponseRefuteInput<'_>,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let ResponseRefuteInput {
            data,
            query,
            estimand,
            estimate,
            scalar,
            estimator_id,
            treatment,
            outcome,
        } = input;
        let scalar_intervention = scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) = if scalar_intervention
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; cheap runs overlap only and full runs overlap plus sampling-stability of the g-comp level. cell.aipw cheap uses the cell-versus-control contrast.",
                ));
            }
            RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate,
                suite: self.refute,
                estimator: estimator_id.as_str(),
                custom: &self.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, ctx)?
        } else {
            if !matches!(self.refute, RefuteSuite::None) || !scalar_intervention {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        Ok((refutations, refute_diags))
    }

    pub(super) fn execute_codetermined_joint_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let background = self.tiered.as_ref().ok_or_else(|| CausalError::Compile {
            message: "CoDetermined joint execute requires a tiered background".into(),
        })?;
        let bayesian = self.inference.is_bayesian();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: if bayesian { EstimatorId::ResponseBayesian } else { EstimatorId::CellAipw },
        })?;
        let expected = if bayesian { EstimatorId::ResponseBayesian } else { EstimatorId::CellAipw };
        if estimator_id != expected {
            return Err(CausalError::Unsupported {
                message: "CoDetermined joint cells require estimator cell.aipw (Frequentist) or \
                          response.bayesian (Bayesian)",
            });
        }
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let identification = match self.graph.as_admg() {
                    Some(admg) => {
                        antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                    }
                    None => antecedent_identify::identify_tiered_joint(
                        background,
                        data.schema(),
                        query,
                    )?,
                };
                let estimand =
                    identification.estimands.first().cloned().ok_or(CausalError::Unsupported {
                        message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                    })?;
                Ok((identification, estimand))
            })?;
        if bayesian {
            return self.execute_codetermined_bayesian_joint(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }
        self.execute_cell_aipw_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
            None,
        )
    }

    /// Bayesian joint level on a `CoDetermined` tier closure: response.bayesian over the
    /// closure's generalized adjustment set, one posterior over every coordinate of the
    /// joint arm. The outcome regression is additive in the treatments, so the level is
    /// that model's joint level, not the saturated cell mean cell.aipw estimates. The
    /// route executes and is unlicensed.
    fn execute_codetermined_bayesian_joint(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::unsupported_inference(
                "the CoDetermined Bayesian joint level requires inference=Bayesian",
            ));
        };
        let mut fitter = ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            fitter.options = options.clone();
        }
        let estimator = super::bayesian_level_refutation::bayesian_response_estimator_in_force(
            &fitter, data, query, config, ctx,
        )?;
        let response = fitter
            .estimate_bayesian(
                data,
                query,
                identification.status,
                identification.required_assumptions.clone(),
                &estimator,
                ctx,
            )
            .map_err(CausalError::from)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.codetermined.bayesian_joint",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Bayesian joint level on the CoDetermined tier closure: response.bayesian over the \
             closure's generalized adjustment set, one posterior over every coordinate of the \
             joint arm. The outcome regression is additive in the treatments, so this is that \
             model's joint level, not the saturated cell mean of cell.aipw. It executes \
             unlicensed: no coverage or evidence record names it.",
        ));
        diagnostics.extend(response.support.warnings.iter().cloned());
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                strategy,
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Reads the binary joint `Set` levels of a cell.aipw intervention response.
    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn parse_cell_aipw_interventions(
        interventions: &[Intervention],
    ) -> Result<(Vec<VariableId>, u32), CausalError> {
        let mut treatments = Vec::new();
        let mut requested_arm = 0u32;
        for (j, iv) in interventions.iter().enumerate() {
            let Intervention::Set { variable, value } = iv else {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires every intervention to be Set",
                });
            };
            let level = value.as_f64().ok_or(CausalError::Unsupported {
                message: "cell.aipw requires binary numeric Set levels",
            })?;
            if level != 0.0 && level != 1.0 {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires binary 0/1 Set levels",
                });
            }
            if j >= antecedent_estimate::cell_aipw::MAX_JOINT_BINARY
                || treatments.contains(variable)
            {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires at most three distinct coordinates",
                });
            }
            treatments.push(*variable);
            requested_arm |= u32::from(level == 1.0) << j;
        }
        Ok((treatments, requested_arm))
    }

    /// Cell-saturated AIPW score table for the joint arms, unless one was precomputed.
    fn cell_aipw_score_table(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        target: &CellAipwTarget<'_>,
        estimand: &IdentifiedEstimand,
        precomputed_table: Option<antecedent_estimate::ScoreTable>,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::ScoreTable, CausalError> {
        let CellAipwTarget { treatments, outcome } = *target;
        let est =
            antecedent_estimate::CellSaturatedAipw::new().with_fold_seed(ctx.rng.master_seed());
        let continuous = None;
        // Folds are deliberately not shared here: `fit_scores_with_assignment` leaves
        // them unset so the cell path draws its own cell-stratified
        // `crossfit_fold_plan` (keyed by `est.fold_seed`), reproducing the solo fold
        // plan for this query's own cells bit-for-bit instead of a private,
        // unstratified batch shuffle. See `SharedBatchDesign` docs.
        let fold_ids: Option<Vec<u32>> = None;
        let design = match self.shared_batch_design.as_ref() {
            Some(shared) => {
                let ids: Vec<_> = treatments
                    .iter()
                    .copied()
                    .chain(std::iter::once(outcome))
                    .chain(estimand.adjustment_set.iter().copied())
                    .collect();
                let row_index = match data.complete_case_mask(&ids) {
                    Ok(mask) => mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| {
                            keep.then_some(antecedent_data::GlobalId::new(
                                u64::try_from(i).expect("row id"),
                            ))
                        })
                        .collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                if row_index.is_empty() {
                    None
                } else {
                    shared.design_for(&estimand.adjustment_set, &row_index)?
                }
            }
            None => None,
        };
        let table = if let Some(table) = precomputed_table {
            table
        } else {
            est.fit_scores_with_assignment(
                data,
                &antecedent_estimate::CellScoreSpec {
                    treatments,
                    outcome,
                    adjustment: &estimand.adjustment_set,
                    functional: &query.outcome_functional,
                    continuous,
                },
                antecedent_estimate::SharedCellInputs {
                    fold_ids: fold_ids.as_deref(),
                    design: design.as_deref(),
                },
            )
            .map_err(CausalError::from)?
        };
        Ok(table)
    }

    /// Level, standard error and influence column of the requested cell.
    #[allow(clippy::float_cmp)] // Exact equality of duplicate threshold values.
    fn cell_aipw_level(
        table: &antecedent_estimate::ScoreTable,
        summary: &antecedent_estimate::ScoreSummary,
        query: &ResponseQuery,
        requested_arm: u32,
        functional_diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(f64, f64, Arc<[f64]>), CausalError> {
        let n_thresholds = {
            let mut t: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t.dedup_by(|a, b| *a == *b);
            t.len()
        };
        let column = table
            .columns
            .iter()
            .position(|c| {
                c.arm == requested_arm
                    && (n_thresholds <= 1
                        || c.threshold == table.columns.first().and_then(|c| c.threshold))
            })
            .or_else(|| table.columns.iter().position(|c| c.arm == requested_arm))
            .ok_or(CausalError::Unsupported { message: "unsupported requested cell" })?;
        let level = antecedent_estimate::LinearContrast {
            value: summary.means[column],
            se: summary.covariance.se(column),
        };
        let quantile = query
            .outcome_functional
            .quantile_level()
            .map(|tau| antecedent_estimate::quantile::quantile_arm(table, None, tau, requested_arm))
            .transpose()?;
        let (scalar, se) = if let Some(q) = &quantile {
            functional_diagnostics.push(super::super::helpers::quantile_scope_diagnostic());
            (q.value, antecedent_estimate::joint_influence_covariance(&[&q.influence], None)?.se(0))
        } else if n_thresholds > 1 {
            functional_diagnostics.push(Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ));
            (f64::NAN, f64::NAN)
        } else {
            (level.value, level.se)
        };
        let influence = match quantile {
            Some(q) => Arc::from(q.influence),
            None => Arc::from(table.column(column)?),
        };
        Ok((scalar, se, influence))
    }

    #[allow(clippy::float_cmp)] // Exact comparison of the requested cell with the control contrast.
    pub(super) fn execute_cell_aipw_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
        precomputed_table: Option<antecedent_estimate::ScoreTable>,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
        else {
            return Err(CausalError::Unsupported {
                message: "cell.aipw is licensed for discrete joint InterventionResponse",
            });
        };
        let (treatments, requested_arm) = Self::parse_cell_aipw_interventions(interventions)?;
        if self.continuous_cell.is_some() {
            return Err(CausalError::Unsupported {
                message: antecedent_estimate::POINT_CDE_UNLICENSED,
            });
        }
        if treatments.len() < 2 {
            return Err(CausalError::Unsupported {
                message: "cell.aipw requires at least two binary Set interventions",
            });
        }
        let target = CellAipwTarget { treatments: &treatments, outcome: *outcome };
        let table =
            self.cell_aipw_score_table(data, query, &target, &estimand, precomputed_table, ctx)?;
        let (summary, monotone_rearranged, mut functional_diagnostics) =
            antecedent_estimate::summarize_functional(&table, None)?;
        let (scalar, se, influence) = Self::cell_aipw_level(
            &table,
            &summary,
            query,
            requested_arm,
            &mut functional_diagnostics,
        )?;
        let cdf = antecedent_estimate::exceedance_cdf_values(&summary, &table);
        let response =
            cell_aipw_response(query, &identification, scalar, se, treatments.len(), requested_arm);
        let score_inference = table.inference(None)?;
        let mut estimate = EffectEstimate::new(
            scalar,
            se,
            identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(Some(table))
        .with_influence(if scalar.is_finite() { Some(influence) } else { None })
        .with_exceedance_cdf(cdf)
        .with_joint_covariance(Some(summary.covariance))
        .with_monotone_rearranged(monotone_rearranged);
        estimate.scores.inference = Some(score_inference);
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(functional_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let refute_estimate = cell_contrast_for_refute(&estimate, requested_arm);
        if !matches!(self.refute, RefuteSuite::None) && refute_estimate.ate != estimate.ate {
            diagnostics.push(Diagnostic::new(
                "refute.cell_aipw.contrast",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cell.aipw cheap/full apply overlap and E-value to the requested-cell minus control-cell contrast, not the intervention level",
            ));
        }
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, refute_diags) = RefutationRun {
            data,
            estimand: &estimand,
            query: &ate_query,
            estimate: &refute_estimate,
            suite: self.refute,
            estimator: "cell.aipw",
            custom: &self.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        diagnostics.extend(refute_diags);
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }
}

pub(super) fn response_scalar_summary(response: &antecedent_core::CausalResponse) -> (f64, f64) {
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => value,
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            return (f64::NAN, f64::NAN);
        }
    };
    let scalar = match value {
        ResponseValue::Scalar(value) => *value,
        ResponseValue::Surface { .. }
        | ResponseValue::Vector(_)
        | ResponseValue::Jacobian { .. }
        | ResponseValue::Envelope(_) => f64::NAN,
    };
    let se = match response.uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } => standard_error,
        _ => f64::NAN,
    };
    (scalar, se)
}

pub(super) fn response_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    crate::analysis::proof_support::require_primary_pair(functional)
}

pub(super) fn response_prior_targets(
    functional: &ResponseFunctional,
) -> Result<(VariableId, Vec<VariableId>), CausalError> {
    match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            Ok((*outcome, vec![treatment.variable]))
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            let treatments = interventions
                .iter()
                .map(|intervention| {
                    intervention.primary_variable().ok_or_else(|| CausalError::Compile {
                        message: "response prior transfer requires an intervention coordinate"
                            .into(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((*outcome, treatments))
        }
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            Ok((*outcome, vec![*treatment]))
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            let outcome = *outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "derivative prior transfer requires an outcome".into(),
            })?;
            Ok((outcome, treatments.to_vec()))
        }
    }
}

pub(crate) fn class_aware_response_supported(query: &ResponseQuery) -> bool {
    query.temporal.is_none()
        && !query.functional.treatment_ids().is_empty()
        && query.observation == ObservationSpec::Complete
        && matches!(
            query.functional,
            ResponseFunctional::MeanCurve { .. }
                | ResponseFunctional::InterventionResponse { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. }
        )
}

pub(crate) fn response_witness_ate(
    query: &ResponseQuery,
) -> Result<AverageEffectQuery, CausalError> {
    if !class_aware_response_supported(query) {
        return Err(CausalError::Unsupported {
            message: "class-aware response identification requires at least one intervention on complete-observation static data",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    Ok(AverageEffectQuery::binary_ate(treatment, outcome))
}

impl super::Study {
    /// Cpdag/Pag response via the same generalized-adjustment envelope as ATE.
    pub(super) fn execute_class_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !class_aware_response_supported(query) {
            return Err(CausalError::Compile {
                message: "Cpdag/Pag response supports complete-observation MeanCurve, \
                          InterventionResponse and derivative functionals only"
                    .into(),
            });
        }
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            estimator: EstimatorId::default_for_response(&query.functional),
            ..RouteDefaults::PAG
        })?;
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG response execute missing resolved static PAG".into(),
                })?;
                let (envelope, identify_cached) =
                    if let Some((cache, prepared)) = self.reuse_pag_identification(physical) {
                        (Arc::clone(&cache.envelope), prepared)
                    } else {
                        report_identify_compute(ctx);
                        (
                            Arc::new(crate::strategy_table::identify_pag_response(
                                identifier_id,
                                pag,
                                query,
                            )?),
                            false,
                        )
                    };
                let envelope_diag = super::pag_path::pag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: envelope_diag,
                    },
                    Strategy::new(identifier_id, estimator_id),
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            GraphClass::Cpdag => {
                let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "CPDAG response execute missing supplied graph".into(),
                })?;
                let (envelope, identify_cached) =
                    if let Some((cache, prepared)) = self.reuse_cpdag_identification(physical) {
                        (Arc::clone(&cache.envelope), prepared)
                    } else {
                        report_identify_compute(ctx);
                        (
                            Arc::new(crate::strategy_table::identify_cpdag_response(
                                identifier_id,
                                cpdag,
                                query,
                            )?),
                            false,
                        )
                    };
                let envelope_diag = super::pag_path::cpdag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: envelope_diag,
                    },
                    Strategy::new(identifier_id, estimator_id),
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class-aware response execute requires a Cpdag or Pag",
            }),
        }
    }

    fn finish_class_response<G>(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ClassEnvelopeRun<'_, G>,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic: envelope_diag,
        } = run;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "class-aware response not identified (no identified mass in envelope)",
            ));
        }
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let inputs = ClassResponseInputs { data, data_est: &data_est, query, ctx };
        let atoms = self.evaluate_class_response_atoms(&inputs, envelope)?;
        let mixed = self.mix_class_response(data, envelope, query, atoms)?;
        let mut diagnostics =
            self.class_response_diagnostics(envelope, envelope_diag, &mixed, identify_cached);
        // A class-aware intervention level runs its per-completion refuter state
        // (unlicensed); a curve keeps the skipped note.
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        if !matches!(self.refute, RefuteSuite::None)
            && response_functional_is_derivative(&query.functional)
        {
            let (reports, checks, notes) = self.refute_class_derivative_atoms(
                &data_est,
                query,
                envelope,
                &mixed.weighted,
                ctx,
            )?;
            refutations = reports;
            predictive_checks = checks;
            diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != "refute.response.skipped");
            diagnostics.extend(notes);
        } else if !matches!(self.refute, RefuteSuite::None)
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. })
        {
            let (reports, notes) =
                super::class_response_refutation::refute_class_intervention_atoms(
                    self,
                    data,
                    envelope,
                    &mixed.weighted,
                    (treatment, outcome),
                    estimator_id,
                    ctx,
                )?;
            refutations = reports;
            diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != "refute.response.skipped");
            diagnostics.extend(notes);
        }
        let MixedClassResponse {
            identification,
            estimand,
            estimate,
            response: mixed,
            structural_response,
            ..
        } = mixed;

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&mixed.provenance_id),
                    Arc::clone(&mixed.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(mixed),
                structural_response: Some(structural_response),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Estimates each identified completion of the class envelope, memoising shared fits.
    fn evaluate_class_response_atoms<G>(
        &self,
        inputs: &ClassResponseInputs<'_>,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
    ) -> Result<ClassResponseAtoms, CausalError> {
        let structural_atoms = envelope
            .cases
            .iter()
            .enumerate()
            .map(|(index, case)| crate::result::StructuralResponseAtom {
                posterior: None,
                response: None,
                graph_key: u64::try_from(index).unwrap_or(u64::MAX),
                weight: case.weight.0,
                status: case.result.status,
                value: None,
            })
            .collect::<Vec<_>>();
        let mut atoms = ClassResponseAtoms {
            weighted: Vec::new(),
            structural_atoms,
            atom_scores: Vec::new(),
            total_w: 0.0,
            unestimated_id_mass: 0.0,
            unevaluable_general_id: None,
            primary_estimand: None,
        };
        let mut completion_fits: CompletionFitMemo<(
            CausalResponse,
            Option<antecedent_estimate::ResponseInfluence>,
        )> = CompletionFitMemo::default();
        for (case_index, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = case.result.estimands[0].clone();
            if response_functional_is_derivative(&inputs.query.functional)
                && estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
            {
                // Identified through a front door: the additive-outcome derivative, with its
                // own whole-procedure interval (no row scores to mix).
                let response = self.estimate_front_door_derivative_response(
                    inputs.data_est,
                    inputs.query,
                    FrontDoorClaim {
                        estimand: &estimand,
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                    },
                    1,
                    inputs.ctx,
                )?;
                let w = case.weight.0;
                atoms.total_w += w;
                if atoms.primary_estimand.is_none() {
                    atoms.primary_estimand = Some(estimand);
                }
                atoms.structural_atoms[case_index].value = response_identified_value(&response);
                atoms.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
                continue;
            }
            if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
                || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
            {
                atoms.add_general_id_case(
                    inputs,
                    case_index,
                    &case.result,
                    case.weight.0,
                    estimand,
                )?;
                continue;
            }
            // Completions that share an adjustment set, status and assumptions
            // run one fit: every other input is fixed for this execution.
            let fit_key = CompletionFitKey::new(
                &estimand.adjustment_set,
                case.result.status,
                &case.result.required_assumptions,
            );
            let (response, scores) = if let Some((response, scores)) = completion_fits.get(&fit_key)
            {
                (response.clone(), scores.clone())
            } else {
                let fitted = self.fit_class_completion(inputs, &case.result, &estimand)?;
                completion_fits.insert(fit_key, (fitted.0.clone(), fitted.1.clone()));
                fitted
            };
            let w = case.weight.0;
            atoms.total_w += w;
            if atoms.primary_estimand.is_none() {
                atoms.primary_estimand = Some(estimand);
            }
            if let Some(scores) = scores {
                atoms.atom_scores.push((w, scores));
            }
            atoms.structural_atoms[case_index].value = response_identified_value(&response);
            atoms.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
        }
        Ok(atoms)
    }

    /// Fits one completion's response, frequentist or Bayesian, before the shared-fit memo stores it.
    fn fit_class_completion(
        &self,
        inputs: &ClassResponseInputs<'_>,
        result: &IdentificationResult,
        estimand: &IdentifiedEstimand,
    ) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let (response, scores) = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.has_transferred_source() && cfg.prior_mapping.is_none() && cfg.prior.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian class-aware response prior transfer requires a \
                              response-specific mapping; mappings are not a class prior \
                              and must not fall back to isotropic coefficients",
                });
            }
            let mut bayes = bayesian_gcomp(cfg, inputs.ctx);
            if cfg.has_transferred_source() {
                let (outcome, treatments) = response_prior_targets(&inputs.query.functional)?;
                let prep = response_estimator
                    .prepare_linear_response_problem(inputs.data_est, outcome, &treatments)
                    .map_err(CausalError::from)?;
                let (resolved, _) = crate::inference::resolve_bayesian_prior_with_conflict(
                    cfg,
                    &prep,
                    Some(inputs.ctx),
                )?;
                bayes.prior = resolved;
            } else {
                bayes.prior.clone_from(&cfg.prior);
            }
            (
                response_estimator
                    .estimate_bayesian(
                        inputs.data_est,
                        inputs.query,
                        result.status,
                        result.required_assumptions.clone(),
                        &bayes,
                        inputs.ctx,
                    )
                    .map_err(CausalError::from)?,
                None,
            )
        } else {
            let (response, scores) = response_estimator
                .estimate_identified_scored(
                    inputs.data_est,
                    inputs.query,
                    result.status,
                    result.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            // The derivative routes publish their own interval but no row scores; the class
            // mixture needs the scores to place completions on shared rows.
            let scores = if scores.is_none()
                && response_functional_is_derivative(&inputs.query.functional)
            {
                response_estimator
                    .derivative_influence(inputs.data_est, inputs.query)
                    .map_err(CausalError::from)?
            } else {
                scores
            };
            (response, scores)
        };
        Ok((response, scores))
    }

    /// Mixes the completion responses into the envelope response, estimate and mass accounting.
    fn mix_class_response<G>(
        &self,
        data: &TabularData,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        query: &ResponseQuery,
        atoms: ClassResponseAtoms,
    ) -> Result<MixedClassResponse, CausalError> {
        let ClassResponseAtoms {
            mut weighted,
            structural_atoms,
            atom_scores,
            total_w,
            unestimated_id_mass,
            unevaluable_general_id,
            primary_estimand,
        } = atoms;
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            if let Some(response) = unevaluable_general_id {
                // Sole general-ID atom could not evaluate a required cell.
                // Publish the support report; do not invent a number.
                weighted.push((0, 1.0, response));
            } else if response_functional_is_derivative(&query.functional)
                && unestimated_id_mass > 0.0
            {
                return Err(CausalError::Unsupported {
                    message: "every identified completion identifies the derivative through a \
                              functional that is not an adjustment (front-door or general ID); \
                              no continuous-treatment estimator evaluates such a functional",
                });
            } else {
                return Err(CausalError::Compile {
                    message: "class-aware response envelope had no estimable identified cases"
                        .into(),
                });
            }
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "class-aware response envelope missing estimand".into(),
        })?;
        let mut mixed = mix_class_responses(&weighted, envelope.status)?;
        let total_mass = envelope.identified_weight.0 + envelope.unidentified_weight.0;
        let structural_response = crate::result::StructuralResponseMixture {
            weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
            atoms: structural_atoms,
            identified_mass: total_w / total_mass.max(f64::EPSILON),
            unidentified_mass: envelope.unidentified_weight.0 / total_mass.max(f64::EPSILON),
            unevaluable_mass: unestimated_id_mass / total_mass.max(f64::EPSILON),
            subsampled_out_mass: 0.0,
            identified_set: response_envelope_from_weighted(&weighted),
            identified_set_interval: None,
            conditional_on_identified: None,
            full_mass_scope: envelope.truncated_completions == 0,
            truncated_atoms: envelope.truncated_completions,
        };
        let identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
        // A multi-completion envelope must not keep a single atom's plugin
        // interval. Mix IFs only when every contributing identified atom
        // supplies one and every completion is identified. Unidentified
        // mass is retained on status; it is not a license to publish the
        // identified-atom SE as the envelope.
        if envelope.cases.len() > 1 {
            mixed.uncertainty = ResponseUncertainty::None;
        }
        // Bayesian completions carry no influence scores to mix. When every
        // completion is identified and each returns the same response (they
        // share the adjustment set, so their posteriors are one posterior),
        // any mixture of them is that posterior: its band is the class band.
        if self.inference.is_bayesian()
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
        {
            if let Some(shared) = shared_completion_uncertainty(&weighted) {
                mixed.uncertainty = shared;
            }
        }
        let mixed_scores = if atom_scores.len() == weighted.len()
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
        {
            mix_response_influences(&atom_scores, data.row_count())
        } else {
            None
        };
        if let Some(scores) = mixed_scores.as_ref() {
            self.attach_class_response_band(&mut mixed, scores)?;
        }
        // Completions of a derivative that disagree publish the identified set; the Imbens-Manski
        // interval of the set, from the completions' joint row influence, is its sampling band.
        let mut identified_set_banded = false;
        let mut structural_response = structural_response;
        if response_functional_is_derivative(&query.functional)
            && matches!(
                &mixed.estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
            )
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
        {
            let level = self.response_options.clone().unwrap_or_default().confidence_level;
            if let Some((band, interval)) =
                derivative_identified_set_band(&weighted, &atom_scores, data.row_count(), level)
            {
                mixed.uncertainty = band;
                structural_response.identified_set_interval = interval;
                identified_set_banded = true;
            }
        }
        let (scalar, mixed_se) = response_scalar_summary(&mixed);
        let mut estimate = EffectEstimate::new(
            scalar,
            mixed_se,
            mixed.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        // Completions that disagree publish the completion identified set, not a
        // mass-weighted summary, so the frozen-weight joint-IF SE has no reported
        // point to attach to. Attaching it anyway left `se_analytic` describing a
        // mixture the result never states.
        let reports_mixed_point = matches!(
            &mixed.estimate,
            ResponseIdentification::PointIdentified(
                ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
            ) | ResponseIdentification::PartiallyIdentified(
                ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
            )
        );
        let identified_set_unbanded = !reports_mixed_point
            && !identified_set_banded
            && matches!(
                &mixed.estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
            );
        if reports_mixed_point {
            attach_response_influence(&mut estimate, mixed_scores.as_ref())?;
        }
        Ok(MixedClassResponse {
            identification,
            estimand,
            estimate,
            response: mixed,
            structural_response,
            identified_set_unbanded,
            weighted,
        })
    }

    /// Attaches the joint-influence band to a mixed response that published none.
    fn attach_class_response_band(
        &self,
        mixed: &mut CausalResponse,
        scores: &antecedent_estimate::ResponseInfluence,
    ) -> Result<(), CausalError> {
        if matches!(mixed.uncertainty, ResponseUncertainty::None) {
            let cols: Vec<&[f64]> = scores.columns.iter().map(Vec::as_slice).collect();
            let cov = antecedent_estimate::joint_influence_covariance(&cols, None)?;
            let options = self.response_options.clone().unwrap_or_default();
            let level = options.confidence_level;
            let value = match &mixed.estimate {
                ResponseIdentification::PointIdentified(value)
                | ResponseIdentification::PartiallyIdentified(value) => Some(value),
                _ => None,
            };
            match value {
                Some(ResponseValue::Scalar(v)) => {
                    let se = cov.se(0);
                    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                    mixed.uncertainty = ResponseUncertainty::Scalar {
                        standard_error: se,
                        level,
                        lower: v - z * se,
                        upper: v + z * se,
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
                    };
                }
                Some(
                    ResponseValue::Vector(coordinates)
                    | ResponseValue::Jacobian { values: coordinates, .. },
                ) if coordinates.len() == cov.dim => {
                    // A derivative vector or Jacobian: one coordinate per column, pointwise.
                    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                    let lower =
                        coordinates.iter().enumerate().map(|(j, v)| v - z * cov.se(j)).collect();
                    let upper =
                        coordinates.iter().enumerate().map(|(j, v)| v + z * cov.se(j)).collect();
                    mixed.uncertainty = ResponseUncertainty::PointwiseBand {
                        level,
                        lower,
                        upper,
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
                    };
                }
                Some(ResponseValue::Surface { mean, .. }) if mean.len() == cov.dim => {
                    let critical = match options.simultaneous_replicates {
                        Some(reps) => antecedent_estimate::max_t_critical(
                            &cov,
                            level,
                            reps,
                            options.multiplier_seed,
                        )?,
                        None => antecedent_stats::normal_ppf(0.5 + level / 2.0),
                    };
                    let lower =
                        mean.iter().enumerate().map(|(j, v)| v - critical * cov.se(j)).collect();
                    let upper =
                        mean.iter().enumerate().map(|(j, v)| v + critical * cov.se(j)).collect();
                    mixed.uncertainty = match options.simultaneous_replicates {
                        Some(replicates) => ResponseUncertainty::SimultaneousBand {
                            level,
                            lower,
                            upper,
                            replicates,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        },
                        None => ResponseUncertainty::PointwiseBand {
                            level,
                            lower,
                            upper,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                            draws: None,
                        },
                    };
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Disclosure diagnostics of a mixed class-aware response.
    fn class_response_diagnostics<G>(
        &self,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        envelope_diag: Diagnostic,
        mixed_class: &MixedClassResponse,
        identify_cached: bool,
    ) -> Vec<Diagnostic> {
        let MixedClassResponse { response: mixed, estimate, weighted, .. } = mixed_class;
        let identified_set_unbanded = mixed_class.identified_set_unbanded;
        let (scalar, _) = response_scalar_summary(mixed);
        let mut diagnostics = vec![envelope_diag];
        if identified_set_unbanded {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_identified_set_unbanded",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completions disagree, so the response is the completion identified set \
                 (pointwise lower/upper over identified completions); no sampling band and no \
                 scalar SE are published for it. Per-completion values and enumeration weights \
                 are in structural_response; a frozen-weight mixture is not reported",
            ));
        }
        if let Some(interval) = &mixed_class.structural_response.identified_set_interval {
            diagnostics.extend(identified_set_interval_diagnostics(interval));
        }
        if response_functional_is_derivative(&mixed.estimand)
            && !matches!(mixed.uncertainty, ResponseUncertainty::None)
            && weighted.iter().all(|(_, _, r)| matches!(r.uncertainty, ResponseUncertainty::None))
        {
            diagnostics.push(Diagnostic::new(
                "estimate.response.derivative_class_band_unmeasured",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the band is formed from the completions' joint row influence (the additive-GAM \
                 coefficient sandwich for a Jacobian or directional derivative, the robust \
                 bias-corrected local fit for a point derivative); no coverage record measures it \
                 and the cell is unlicensed",
            ));
        }
        if envelope.cases.iter().any(|c| {
            c.result
                .estimands
                .iter()
                .any(|e| e.method_kind().ok() == Some(EstimandMethod::GeneralId))
        }) {
            diagnostics.push(Diagnostic::new(
                "identify.response.general_id",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "at least one MAG/CPDAG completion identified the response by \
                 Shpitser–Pearl ID after generalized adjustment failed",
            ));
        }
        if !estimate.se_analytic().is_finite() && !identified_set_unbanded {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        // Disclose only when the envelope actually dropped per-completion
        // posterior uncertainty; a single contributing completion keeps its own
        // band and nothing is omitted.
        let atom_uncertainty_dropped = matches!(mixed.uncertainty, ResponseUncertainty::None)
            && weighted.iter().any(|(_, _, r)| !matches!(r.uncertainty, ResponseUncertainty::None));
        if self.inference.is_bayesian() && atom_uncertainty_dropped {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-completion posterior draws are not assigned structural probabilities; \
                 completion-enumeration responses remain an identified set and uncertainty is \
                 omitted when multiple atoms contribute",
            ));
        }
        if matches!(envelope.status, IdentificationStatus::GraphDependent) {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_graph_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "response payload retains graph-conditional atom values; completion enumeration \
                 is not treated as a posterior distribution and unidentified mass remains explicit",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued response",
        ));
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &mixed.support.warnings {
            diagnostics.push(warning.clone());
        }
        diagnostics
    }
}

/// What a front-door derivative estimates from: the estimand, its status and its premises.
#[derive(Clone)]
pub(super) struct FrontDoorClaim<'a> {
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: antecedent_core::AssumptionSet,
}

/// Borrowed inputs shared by every completion of a class-aware response.
#[derive(Clone, Copy)]
struct ClassResponseInputs<'a> {
    data: &'a TabularData,
    data_est: &'a TabularData,
    query: &'a ResponseQuery,
    ctx: &'a ExecutionContext,
}

/// Per-completion results of a class-aware response, in envelope order.
struct ClassResponseAtoms {
    weighted: Vec<(u64, f64, CausalResponse)>,
    structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
    total_w: f64,
    unestimated_id_mass: f64,
    unevaluable_general_id: Option<CausalResponse>,
    primary_estimand: Option<IdentifiedEstimand>,
}

impl ClassResponseAtoms {
    /// Adds a completion identified by general ID or the front-door, or accounts its lost mass.
    fn add_general_id_case(
        &mut self,
        inputs: &ClassResponseInputs<'_>,
        case_index: usize,
        result: &IdentificationResult,
        weight: f64,
        estimand: IdentifiedEstimand,
    ) -> Result<(), CausalError> {
        if response_functional_is_derivative(&inputs.query.functional) {
            // Identified through a functional that is not an adjustment (front-door or a
            // general ID expression). The discrete plug-in evaluates probabilities at
            // observed levels; it has no derivative with respect to a continuous treatment,
            // so the completion's mass is accounted as unevaluable and never mixed.
            self.unestimated_id_mass += weight;
            if self.primary_estimand.is_none() {
                self.primary_estimand = Some(estimand);
            }
            return Ok(());
        }
        match estimate_general_id_response(
            inputs.data,
            inputs.query,
            result,
            &estimand,
            FunctionalBootstrap::rows(0),
            inputs.ctx,
        ) {
            Ok((response, scores)) => {
                if matches!(response.estimate, ResponseIdentification::Unidentified { .. }) {
                    // Required CPT cell was not evaluable: do not mix a number.
                    self.unestimated_id_mass += weight;
                    if self.primary_estimand.is_none() {
                        self.primary_estimand = Some(estimand);
                    }
                    if self.unevaluable_general_id.is_none() {
                        self.unevaluable_general_id = Some(response);
                    }
                    return Ok(());
                }
                let w = weight;
                self.total_w += w;
                if self.primary_estimand.is_none() {
                    self.primary_estimand = Some(estimand);
                }
                if let Some(scores) = scores {
                    self.atom_scores.push((w, scores));
                }
                self.structural_atoms[case_index].value = response_identified_value(&response);
                self.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
            }
            Err(err) if is_discrete_plug_in_limit(&err) => {
                // Identified by ID, unevaluable with the discrete plug-in.
                // Keep the mass out of the mix and refuse a subset SE.
                self.unestimated_id_mass += weight;
            }
            Err(err) => return Err(err),
        }

        Ok(())
    }
}

/// Whether `err` is the discrete plug-in refusing a variable past its level cap.
///
/// A high-cardinality or continuous column fails closed as that same
/// `EstimationError::Unsupported` message. Any other refusal, including one
/// whose text merely contains "continuous", propagates.
fn is_discrete_plug_in_limit(err: &CausalError) -> bool {
    matches!(
        err.peeled(),
        CausalError::Estimate(antecedent_estimate::EstimationError::Unsupported { message })
            if *message == "variable exceeds discrete level cap for functional.distribution"
    )
}

/// The class-aware response mixed over completions, with the values its disclosures read.
struct MixedClassResponse {
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    response: CausalResponse,
    structural_response: crate::result::StructuralResponseMixture,
    identified_set_unbanded: bool,
    weighted: Vec<(u64, f64, CausalResponse)>,
}

/// The uncertainty every contributing completion shares, when they all
/// return the same estimate and the same published band; `None` when any two
/// differ or none publishes a band.
fn shared_completion_uncertainty(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
) -> Option<ResponseUncertainty> {
    let ((_, _, first), rest) = weighted.split_first()?;
    if matches!(first.uncertainty, ResponseUncertainty::None) {
        return None;
    }
    rest.iter()
        .all(|(_, _, r)| r.estimate == first.estimate && r.uncertainty == first.uncertainty)
        .then(|| first.uncertainty.clone())
}

pub(super) fn response_identified_value(
    response: &antecedent_core::CausalResponse,
) -> Option<ResponseValue> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => Some(value.clone()),
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            None
        }
    }
}

pub(super) fn response_envelope_from_weighted(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
) -> Option<antecedent_core::ResponseEnvelope> {
    let values = weighted
        .iter()
        .filter_map(|(_, _, response)| response_identified_value(response))
        .collect::<Vec<_>>();
    let first = values.first()?;
    match first {
        ResponseValue::Scalar(value) => {
            let mut lower = *value;
            let mut upper = *value;
            for candidate in values.iter().skip(1) {
                let ResponseValue::Scalar(candidate) = candidate else {
                    return None;
                };
                lower = lower.min(*candidate);
                upper = upper.max(*candidate);
            }
            Some(scalar_identified_set(lower, upper))
        }
        ResponseValue::Surface { grid, dimension, mean } => {
            let mut lower = mean.to_vec();
            let mut upper = mean.to_vec();
            for candidate in values.iter().skip(1) {
                let ResponseValue::Surface {
                    grid: candidate_grid,
                    dimension: candidate_dimension,
                    mean: candidate_mean,
                } = candidate
                else {
                    return None;
                };
                if candidate_grid.as_ref() != grid.as_ref()
                    || candidate_dimension != dimension
                    || candidate_mean.len() != mean.len()
                {
                    return None;
                }
                for (index, value) in candidate_mean.iter().enumerate() {
                    lower[index] = lower[index].min(*value);
                    upper[index] = upper[index].max(*value);
                }
            }
            Some(antecedent_core::ResponseEnvelope {
                grid: Arc::clone(grid),
                dimension: *dimension,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            })
        }
        ResponseValue::Vector(_) | ResponseValue::Jacobian { .. } => {
            // A derivative vector or Jacobian: one coordinate per entry, no grid.
            let first_entries = coordinate_entries(first)?;
            let mut lower = first_entries.to_vec();
            let mut upper = first_entries.to_vec();
            for candidate in values.iter().skip(1) {
                let entries = coordinate_entries(candidate)?;
                if entries.len() != lower.len() || !same_coordinate_shape(first, candidate) {
                    return None;
                }
                for (index, value) in entries.iter().enumerate() {
                    lower[index] = lower[index].min(*value);
                    upper[index] = upper[index].max(*value);
                }
            }
            Some(antecedent_core::ResponseEnvelope {
                grid: Arc::from([]),
                dimension: 0,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            })
        }
        _ => None,
    }
}

/// The coordinates of a vector or Jacobian payload, `None` for any other shape.
fn coordinate_entries(value: &ResponseValue) -> Option<&[f64]> {
    match value {
        ResponseValue::Vector(entries) => Some(entries),
        ResponseValue::Jacobian { values, .. } => Some(values),
        _ => None,
    }
}

/// Whether two coordinate payloads are the same kind with the same matrix shape.
fn same_coordinate_shape(first: &ResponseValue, other: &ResponseValue) -> bool {
    match (first, other) {
        (ResponseValue::Vector(a), ResponseValue::Vector(b)) => a.len() == b.len(),
        (
            ResponseValue::Jacobian { outcomes: ao, treatments: at, .. },
            ResponseValue::Jacobian { outcomes: bo, treatments: bt, .. },
        ) => ao == bo && at == bt,
        _ => false,
    }
}

fn identified_set_is_singleton(envelope: &antecedent_core::ResponseEnvelope) -> bool {
    envelope.lower.len() == envelope.upper.len()
        && envelope
            .lower
            .iter()
            .zip(envelope.upper.iter())
            .all(|(lo, hi)| lo.to_bits() == hi.to_bits())
}

fn singleton_response_value(
    items: &[(f64, &ResponseValue)],
    envelope: &antecedent_core::ResponseEnvelope,
) -> Result<ResponseValue, CausalError> {
    let Some((_, first)) = items.first() else {
        return Err(CausalError::Compile {
            message: "class-aware response mix requires at least one identified case".into(),
        });
    };
    match first {
        ResponseValue::Scalar(_) => Ok(ResponseValue::Scalar(envelope.lower[0])),
        ResponseValue::Surface { .. } => Ok(ResponseValue::Surface {
            grid: Arc::clone(&envelope.grid),
            dimension: envelope.dimension,
            mean: Arc::clone(&envelope.lower),
        }),
        ResponseValue::Vector(_) => Ok(ResponseValue::Vector(Arc::clone(&envelope.lower))),
        ResponseValue::Jacobian { outcomes, treatments, .. } => Ok(ResponseValue::Jacobian {
            outcomes: *outcomes,
            treatments: *treatments,
            values: Arc::clone(&envelope.lower),
        }),
        _ => Err(CausalError::Compile {
            message: "class-aware response only publishes scalar, surface, vector and Jacobian \
                      identified sets"
                .into(),
        }),
    }
}

pub(super) fn mix_class_responses(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
    envelope_status: IdentificationStatus,
) -> Result<antecedent_core::CausalResponse, CausalError> {
    if weighted
        .iter()
        .all(|(_, _, r)| matches!(r.estimate, ResponseIdentification::Unidentified { .. }))
    {
        return Ok(weighted[0].2.clone());
    }
    let first = &weighted[0].2;
    let items: Vec<(f64, &ResponseValue)> = weighted
        .iter()
        .map(|(_, w, r)| {
            let value = match &r.estimate {
                ResponseIdentification::PointIdentified(v)
                | ResponseIdentification::PartiallyIdentified(v) => v,
                ResponseIdentification::GraphDependent(_)
                | ResponseIdentification::Unidentified { .. } => {
                    return Err(CausalError::Compile {
                        message: "class-aware response cannot mix unidentified case payloads"
                            .into(),
                    });
                }
            };
            Ok((*w, value))
        })
        .collect::<Result<_, _>>()?;
    let mixed_uncertainty = mix_response_uncertainty(
        &weighted.iter().map(|(_, w, r)| (*w, &r.uncertainty)).collect::<Vec<_>>(),
    );
    let mixed_support =
        mix_support_reports(&weighted.iter().map(|(_, _, r)| &r.support).collect::<Vec<_>>());
    let mut assumptions = first.assumptions.clone();
    for (_, _, response) in weighted.iter().skip(1) {
        assumptions.extend_unique(&response.assumptions.entries);
    }
    let (identification_status, estimate) =
        if matches!(envelope_status, IdentificationStatus::GraphDependent) {
            let atoms = weighted
                .iter()
                .filter_map(|(key, _, response)| {
                    response_identified_value(response).map(|value| (*key, value))
                })
                .collect();
            (IdentificationStatus::GraphDependent, ResponseIdentification::GraphDependent(atoms))
        } else {
            let envelope =
                response_envelope_from_weighted(weighted).ok_or_else(|| CausalError::Compile {
                    message:
                        "class-aware response could not construct a common identified envelope"
                            .into(),
                })?;
            if identified_set_is_singleton(&envelope)
                && matches!(
                    envelope_status,
                    IdentificationStatus::NonparametricallyIdentified
                        | IdentificationStatus::IdentifiedUnderParametricRestrictions
                        | IdentificationStatus::IdentifiedUnderPriorRestrictions
                )
            {
                let value = singleton_response_value(&items, &envelope)?;
                (envelope_status, ResponseIdentification::PointIdentified(value))
            } else {
                (
                    IdentificationStatus::PartiallyIdentified,
                    ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope)),
                )
            }
        };
    Ok(antecedent_core::CausalResponse {
        estimand: first.estimand.clone(),
        identification_status,
        estimate,
        uncertainty: mixed_uncertainty,
        support: mixed_support,
        assumptions,
        provenance_id: Arc::clone(&first.provenance_id),
        horizon_identification: None,
        interaction_structurally_zero: first.interaction_structurally_zero,
    })
}

pub(super) fn mix_response_values(
    items: &[(f64, &ResponseValue)],
) -> Result<ResponseValue, CausalError> {
    let Some((_, first)) = items.first() else {
        return Err(CausalError::Compile {
            message: "class-aware response mix requires at least one identified case".into(),
        });
    };
    match first {
        ResponseValue::Scalar(_) => {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Scalar(x) = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                acc += *w * *x;
                wsum += *w;
            }
            Ok(ResponseValue::Scalar(acc / wsum))
        }
        ResponseValue::Surface { grid, dimension, mean } => {
            let n = mean.len();
            let mut acc = vec![0.0; n];
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Surface { grid: g, dimension: d, mean: m } = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                if g.as_ref() != grid.as_ref() || *d != *dimension || m.len() != n {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned incompatible grids".into(),
                    });
                }
                for (i, point) in m.iter().enumerate() {
                    acc[i] += *w * *point;
                }
                wsum += *w;
            }
            for point in &mut acc {
                *point /= wsum;
            }
            Ok(ResponseValue::Surface {
                grid: Arc::clone(grid),
                dimension: *dimension,
                mean: Arc::from(acc),
            })
        }
        ResponseValue::Vector(_) | ResponseValue::Jacobian { .. } => {
            let mut acc = vec![0.0; coordinate_entries(first).map_or(0, <[f64]>::len)];
            let mut wsum = 0.0;
            for (w, value) in items {
                let entries = coordinate_entries(value).filter(|entries| {
                    entries.len() == acc.len() && same_coordinate_shape(first, value)
                });
                let Some(entries) = entries else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                for (slot, entry) in acc.iter_mut().zip(entries) {
                    *slot += *w * *entry;
                }
                wsum += *w;
            }
            for entry in &mut acc {
                *entry /= wsum;
            }
            Ok(match first {
                ResponseValue::Jacobian { outcomes, treatments, .. } => ResponseValue::Jacobian {
                    outcomes: *outcomes,
                    treatments: *treatments,
                    values: Arc::from(acc),
                },
                _ => ResponseValue::Vector(Arc::from(acc)),
            })
        }
        _ => Err(CausalError::Compile {
            message: "class-aware response only mixes scalar, surface, vector and Jacobian \
                      payloads"
                .into(),
        }),
    }
}

fn mix_response_uncertainty(items: &[(f64, &ResponseUncertainty)]) -> ResponseUncertainty {
    // A single atom retains its valid uncertainty. Averaging confidence limits
    // does not produce confidence limits for the weighted mean (the fits share
    // observations), nor quantiles of a distribution over graphs.
    if let [(weight, uncertainty)] = items {
        if weight.is_finite() && *weight > 0.0 {
            return (*uncertainty).clone();
        }
    }
    // Markov-equivalent completions of a fully identified class produce the same
    // adjustment, hence the same estimate and the same interval (for a Bayesian
    // response, the same retained draws). Their weighted posterior is that one
    // distribution, so the class interval is the shared interval — collapsing
    // agreeing completions invents nothing, unlike averaging distinct components.
    if let Some((first_weight, first)) = items.first() {
        if first_weight.is_finite()
            && *first_weight > 0.0
            && !matches!(**first, ResponseUncertainty::None)
            && items.iter().all(|(weight, u)| weight.is_finite() && *weight > 0.0 && *u == *first)
        {
            return (*first).clone();
        }
    }
    ResponseUncertainty::None
}

pub(super) fn mix_support_reports(
    reports: &[&antecedent_core::SupportReport],
) -> antecedent_core::SupportReport {
    let first = reports[0];
    let mut status = first.status;
    let mut warnings = first.warnings.clone();
    for report in reports.iter().skip(1) {
        if support_rank(report.status) > support_rank(status) {
            status = report.status;
        }
        for warning in &report.warnings {
            if !warnings
                .iter()
                .any(|seen| seen.code == warning.code && seen.message == warning.message)
            {
                warnings.push(warning.clone());
            }
        }
    }
    // Per-cell status is the most conservative over the mixed reports whenever
    // they share one cell layout; a cell outside any report's support is outside
    // the mixture's.
    let point_status = match first.point_status.as_deref() {
        Some(cells)
            if reports
                .iter()
                .all(|r| r.point_status.as_deref().is_some_and(|p| p.len() == cells.len())) =>
        {
            Some(
                (0..cells.len())
                    .map(|cell| {
                        reports
                            .iter()
                            .filter_map(|r| r.point_status.as_deref().map(|p| p[cell]))
                            .max_by_key(|status| support_rank(*status))
                            .unwrap_or(cells[cell])
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        }
        _ => first.point_status.clone(),
    };
    let mut mixed = antecedent_core::SupportReport {
        status,
        query_region: first.query_region.clone(),
        diagnostics: first.diagnostics.clone(),
        warnings,
        point_status,
    };
    // A simultaneous band belongs to one atom's surface; it stays on that atom and
    // never describes the mixed class report.
    antecedent_estimate::clear_simultaneous_band(&mut mixed);
    mixed
}

const fn support_rank(status: antecedent_core::SupportStatus) -> u8 {
    status.severity()
}

/// E-value / overlap on cell.aipw use the requested cell minus the all-zero control cell.
fn cell_contrast_for_refute(estimate: &EffectEstimate, requested_arm: u32) -> EffectEstimate {
    let Some(table) = estimate.scores.table.as_ref() else {
        return estimate.clone();
    };
    let Ok((contrast, _)) = antecedent_estimate::cell_minus_control_contrast(table, requested_arm)
    else {
        return estimate.clone();
    };
    let mut out = estimate.clone();
    out.ate = contrast.value;
    out.uncertainty.analytic.se = contrast.se;
    out
}

pub(super) fn attach_response_influence(
    estimate: &mut EffectEstimate,
    scores: Option<&antecedent_estimate::ResponseInfluence>,
) -> Result<(), CausalError> {
    let Some(scores) = scores else {
        return Ok(());
    };
    if scores.columns.is_empty() || scores.columns.iter().any(|c| c.len() < 2) {
        return Ok(());
    }
    let cols: Vec<&[f64]> = scores.columns.iter().map(Vec::as_slice).collect();
    match antecedent_estimate::joint_influence_covariance(&cols, None) {
        Ok(cov) => {
            // Column 0's SE describes the reported scalar only. A surface response publishes no
            // scalar (`ate` is NaN), so a standard error here would sit beside no estimate and
            // describe one unnamed grid cell; the joint covariance below carries every cell.
            if estimate.se_analytic().is_nan() && estimate.ate.is_finite() {
                estimate.uncertainty.analytic.se = cov.se(0);
            }
            estimate.joint.covariance = Some(cov);
        }
        Err(err) if cols.len() > 1 => {
            return Err(CausalError::from(err));
        }
        Err(_) => {}
    }
    estimate.scores.influence = scores.columns.first().map(|c| Arc::from(c.as_slice()));
    Ok(())
}

/// SE of the frozen-weight mixture of scalar atom responses from the joint
/// influence of the atoms on the shared data rows. Each atom's influence is
/// placed on its own complete-case rows ([`mix_response_influences`]), so atoms
/// that drop different rows are aligned by row index, not by position.
pub(super) fn mixed_response_scalar_se(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<f64> {
    let mixed = mix_response_influences(atoms, full_n)?;
    let column = mixed.columns.first()?;
    antecedent_estimate::joint_influence_covariance(&[column.as_slice()], None)
        .ok()
        .map(|cov| cov.se(0))
        .filter(|se| se.is_finite())
}

/// The refusals every graph-posterior response shares (fresh run and prepared
/// handle): a static response on complete observations, and one intervention
/// coordinate for [`ResponseFunctional::InterventionResponse`].
pub(crate) fn graph_posterior_response_supported(query: &ResponseQuery) -> Result<(), CausalError> {
    if query.temporal.is_some() || query.observation != ObservationSpec::Complete {
        return Err(CausalError::Unsupported {
            message: "static graph-posterior response requires complete observations",
        });
    }
    if matches!(
        &query.functional,
        ResponseFunctional::InterventionResponse { interventions, .. }
            if interventions.len() != 1
    ) {
        return Err(CausalError::Unsupported {
            message: "graph-posterior InterventionResponse currently requires one intervention coordinate",
        });
    }
    Ok(())
}

/// The coordinates of a scalar, vector or Jacobian payload.
fn derivative_coordinates(value: &ResponseValue) -> Option<Vec<f64>> {
    match value {
        ResponseValue::Scalar(x) => Some(vec![*x]),
        ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => {
            Some(values.to_vec())
        }
        _ => None,
    }
}

/// Every atom's influence columns on the full row set, centred over the atom's own rows and
/// scaled by `full_n / rows` so a missing row contributes zero (the alignment
/// [`mix_response_influences`] mixes with). `aligned[atom][column]`.
fn aligned_atom_columns(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<Vec<Vec<Vec<f64>>>> {
    if atoms.is_empty() || full_n < 2 {
        return None;
    }
    let n_cols = atoms[0].1.columns.len();
    let mut aligned = Vec::with_capacity(atoms.len());
    for (_, scores) in atoms {
        let n = scores.row_index.len();
        if n < 2 || scores.columns.len() != n_cols || scores.columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut columns = Vec::with_capacity(n_cols);
        for col in &scores.columns {
            let center = col.iter().sum::<f64>() / n as f64;
            let mut full = vec![0.0; full_n];
            for (&row, &v) in scores.row_index.iter().zip(col) {
                *full.get_mut(usize::try_from(row.get()).ok()?)? =
                    (v - center) * full_n as f64 / n as f64;
            }
            columns.push(full);
        }
        aligned.push(columns);
    }
    Some(aligned)
}

/// The sampling band of a derivative whose completions disagree: per coordinate, the
/// Imbens-Manski interval of the completions' identified set with standard deviations from
/// their joint row influence. One coordinate publishes a scalar interval (and the interval
/// itself for the structural mixture); several publish a pointwise band.
fn derivative_identified_set_band(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
    atom_scores: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
    level: f64,
) -> Option<(ResponseUncertainty, Option<antecedent_estimate::IdentifiedSetInterval>)> {
    if weighted.len() < 2 || atom_scores.len() != weighted.len() {
        return None;
    }
    let coordinates: Vec<Vec<f64>> = weighted
        .iter()
        .map(|(_, _, response)| {
            response_identified_value(response).as_ref().and_then(derivative_coordinates)
        })
        .collect::<Option<_>>()?;
    let dim = coordinates[0].len();
    if coordinates.iter().any(|c| c.len() != dim) {
        return None;
    }
    let aligned = aligned_atom_columns(atom_scores, full_n)?;
    if aligned.iter().any(|columns| columns.len() != dim) {
        return None;
    }
    let mut lower = Vec::with_capacity(dim);
    let mut upper = Vec::with_capacity(dim);
    let mut intervals = Vec::with_capacity(dim);
    for j in 0..dim {
        let points: Vec<f64> = coordinates.iter().map(|c| c[j]).collect();
        let columns: Vec<&[f64]> = aligned.iter().map(|atom| atom[j].as_slice()).collect();
        let interval = antecedent_estimate::imbens_manski_influence(&points, &columns, level)?;
        lower.push(interval.lower);
        upper.push(interval.upper);
        intervals.push(interval);
    }
    let interpretation = antecedent_core::IntervalInterpretation::Confidence;
    if dim == 1 {
        let interval = intervals.remove(0);
        let standard_error = interval.lower_se.max(interval.upper_se);
        Some((
            ResponseUncertainty::Scalar {
                standard_error,
                level,
                lower: interval.lower,
                upper: interval.upper,
                interpretation,
                draws: None,
            },
            Some(interval),
        ))
    } else {
        Some((
            ResponseUncertainty::PointwiseBand {
                level,
                lower: lower.into(),
                upper: upper.into(),
                interpretation,
                draws: None,
            },
            None,
        ))
    }
}

fn mix_response_influences(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<antecedent_estimate::ResponseInfluence> {
    if atoms.is_empty() {
        return None;
    }
    let n_cols = atoms[0].1.columns.len();
    if full_n < 2 || n_cols == 0 {
        return None;
    }
    let mut aligned = Vec::with_capacity(atoms.len());
    for (_, scores) in atoms {
        let n = scores.row_index.len();
        if n < 2 || scores.columns.len() != n_cols || scores.columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut seen = vec![false; full_n];
        for &row in scores.row_index.iter() {
            let slot = seen.get_mut(usize::try_from(row.get()).expect("row id"))?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        let mut columns = Vec::new();
        for col in &scores.columns {
            let center = col.iter().sum::<f64>() / n as f64;
            let mut full = vec![0.0; full_n];
            for (&row, &v) in scores.row_index.iter().zip(col) {
                full[usize::try_from(row.get()).expect("row id")] =
                    (v - center) * full_n as f64 / n as f64;
            }
            columns.push(full);
        }
        aligned.push(columns);
    }
    let weights: Vec<f64> = atoms.iter().map(|(w, _)| *w).collect();
    let mut columns = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let refs: Vec<&[f64]> = aligned.iter().map(|cols| cols[j].as_slice()).collect();
        columns.push(antecedent_estimate::frozen_weight_mixture_scores(&refs, &weights).ok()?);
    }
    Some(antecedent_estimate::ResponseInfluence {
        columns,
        row_index: (0..full_n)
            .map(|i| u64::try_from(i).map(antecedent_data::GlobalId::new))
            .collect::<Result<Vec<_>, _>>()
            .ok()?
            .into(),
    })
}

#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn disjoint_atom_intervals_are_not_averaged_into_a_confidence_interval() {
        let a = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: -0.2,
            upper: 0.2,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        let b = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: 9.8,
            upper: 10.2,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        assert!(matches!(
            mix_response_uncertainty(&[(0.5, &a), (0.5, &b)]),
            ResponseUncertainty::None
        ));
        assert!(matches!(
            mix_response_uncertainty(&[(1.0, &a)]),
            ResponseUncertainty::Scalar { .. }
        ));
    }
}

/// Normal interval from the front-door plug-in bootstrap SE of a general-ID response, when one is licensed.
fn general_id_bootstrap_interval(
    est: &FunctionalEffect,
    prepared: &antecedent_estimate::PreparedFunctionalEffect,
    estimate: &ResponseIdentification,
    support: &mut antecedent_core::SupportReport,
    bootstrap_replicates: u32,
    ctx: &ExecutionContext,
) -> ResponseUncertainty {
    // The general-ID front-door functional exposes no closed-form influence on
    // this route, so its frequentist interval is the normal interval formed from
    // the plug-in bootstrap SE (the same 199-replicate front-door resample the
    // sibling AverageEffect / InterventionalDistribution ADMG cells use). It is
    // published only when a positive-finite SE returns; otherwise the route fails
    // open to `None` rather than panicking.
    let mut uncertainty = ResponseUncertainty::None;
    if bootstrap_replicates > 0 {
        if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(point)) = estimate {
            let point = *point;
            let mut ws = FunctionalDistributionWorkspace::default();
            if let Ok(effect) = est.estimate(prepared, &mut ws, ctx) {
                if let Some(se) = effect.se_bootstrap() {
                    if se.is_finite() && se > 0.0 && point.is_finite() {
                        let z = crate::result::reported_se_interval_z();
                        uncertainty = ResponseUncertainty::Scalar {
                            standard_error: se,
                            lower: point - z * se,
                            upper: point + z * se,
                            level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                            draws: None,
                        };
                        support.diagnostics.push(antecedent_core::SupportDiagnostic {
                            id: Arc::from(crate::result::RESPONSE_BOOTSTRAP_SE),
                            values: Arc::from([
                                f64::from(bootstrap_replicates),
                                f64::from(effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0)),
                            ]),
                            detail: Arc::from(
                                "general-ID functional.effect response interval is a normal \
                                 interval from the front-door plug-in bootstrap SE \
                                 [requested replicates, successful replicates]",
                            ),
                            scope: antecedent_core::DiagnosticScope::Global,
                        });
                    }
                }
            }
        }
    }
    uncertainty
}

/// The replicate budget and row resampling of a general-ID functional's bootstrap.
#[derive(Clone, Copy, Debug)]
pub(super) struct FunctionalBootstrap {
    pub(super) replicates: u32,
    pub(super) resampling: antecedent_estimate::RowResampling,
    pub(super) design: antecedent_estimate::util::SamplingDeclaration,
}

impl FunctionalBootstrap {
    /// Independent rows, no declared design: the static default.
    pub(super) const fn rows(replicates: u32) -> Self {
        Self {
            replicates,
            resampling: antecedent_estimate::RowResampling::Rows,
            design: antecedent_estimate::util::SamplingDeclaration::Undeclared,
        }
    }
}

impl super::Study {
    /// The functional bootstrap of this study with `replicates` replicates.
    ///
    /// # Errors
    ///
    /// The declared design refuses the draw ([`Self::functional_row_resampling`]).
    pub(super) fn functional_bootstrap(
        &self,
        replicates: u32,
    ) -> Result<FunctionalBootstrap, CausalError> {
        let (resampling, design) = self.functional_row_resampling()?;
        Ok(FunctionalBootstrap { replicates, resampling, design })
    }
}

pub(super) fn estimate_general_id_response(
    data: &TabularData,
    query: &ResponseQuery,
    identification: &antecedent_identify::IdentificationResult,
    estimand: &IdentifiedEstimand,
    bootstrap: FunctionalBootstrap,
    ctx: &ExecutionContext,
) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
    if !matches!(query.functional, ResponseFunctional::InterventionResponse { .. }) {
        return Err(CausalError::Unsupported {
            message: "general-ID MAG/PAG/DAG response atoms are licensed for \
                      InterventionResponse Set means; MeanCurve stays on the \
                      adjustment envelope",
        });
    }
    if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        || !matches!(
            query.outcome_functional,
            antecedent_core::OutcomeFunctional::Mean
                | antecedent_core::OutcomeFunctional::Exceedance(_)
        )
    {
        return Err(CausalError::Unsupported {
            message: "general-ID response estimation supports AllObserved scalar mean or exceedance outcomes",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let data_est = super::super::helpers::apply_scalar_outcome_functional(
        data,
        outcome,
        &query.outcome_functional,
    )?;
    let bootstrap_replicates = bootstrap.replicates;
    let est = FunctionalEffect { bootstrap_replicates, ..FunctionalEffect::new() }
        .with_row_resampling(bootstrap.resampling, bootstrap.design);
    let prepared = est
        .prepare(
            &data_est,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            &[treatment, outcome],
        )
        .map_err(CausalError::from)?;
    let eval = prepared.evaluate(&prepared.provider);
    let map_eval = |e: EvalError| {
        CausalError::from(antecedent_estimate::EstimationError::data_msg(e.to_string()))
    };
    let (estimate, mut support) = match eval {
        Ok(ate) => (
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(ate)),
            support_from_functional_eval(None).map_err(map_eval)?,
        ),
        Err(err) if functional_cell_unevaluable(&err) => (
            ResponseIdentification::Unidentified {
                certificate: Arc::from("estimate.response.general_id.unevaluable_cell"),
            },
            support_from_functional_eval(Some(&err)).map_err(map_eval)?,
        ),
        Err(err) => return Err(map_eval(err)),
    };
    if let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional {
        let levels = interventions
            .iter()
            .map(|intervention| match intervention {
                Intervention::Set { value, .. } => value.as_f64().ok_or(CausalError::Unsupported {
                    message: "general-ID response requires numeric Set levels",
                }),
                _ => Err(CausalError::Unsupported {
                    message: "general-ID response requires Set interventions",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        support.query_region = antecedent_core::SupportRegion {
            minima: Arc::from(levels.clone()),
            maxima: Arc::from(levels),
        };
    }
    let uncertainty = general_id_bootstrap_interval(
        &est,
        &prepared,
        &estimate,
        &mut support,
        bootstrap_replicates,
        ctx,
    );
    Ok((
        CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate,
            uncertainty,
            support,
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        },
        None,
    ))
}

#[cfg(test)]
mod influence_review_tests {
    use super::*;

    fn influence(columns: Vec<Vec<f64>>) -> antecedent_estimate::ResponseInfluence {
        antecedent_estimate::ResponseInfluence {
            columns,
            row_index: [0u64, 1, 2, 3].map(antecedent_data::GlobalId::new).into(),
        }
    }

    #[test]
    fn surface_response_influence_attaches_no_scalar_standard_error() {
        let scores = influence(vec![vec![1.0, -1.0, 2.0, -2.0], vec![0.5, 0.5, -0.5, -0.5]]);
        let mut surface = EffectEstimate::new(
            f64::NAN,
            f64::NAN,
            antecedent_core::AssumptionSet::new(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut surface, Some(&scores)).unwrap();
        assert!(surface.se_analytic().is_nan(), "se {}", surface.se_analytic());
        assert!(surface.joint.covariance.is_some());
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the scalar standard error is copied from the same joint-covariance entry, so it must match bit for bit"
    )]
    fn scalar_response_influence_still_attaches_its_standard_error() {
        let scores = influence(vec![vec![1.0, -1.0, 2.0, -2.0]]);
        let mut scalar = EffectEstimate::new(
            0.3,
            f64::NAN,
            antecedent_core::AssumptionSet::new(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut scalar, Some(&scores)).unwrap();
        let expected = scalar.joint.covariance.as_ref().unwrap().se(0);
        assert!(expected.is_finite() && expected > 0.0, "se {expected}");
        assert_eq!(scalar.se_analytic(), expected);
    }

    #[test]
    fn response_mixture_aligns_original_row_ids() {
        let a = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-1.0, 1.0]],
            row_index: [0u64, 2].map(antecedent_data::GlobalId::new).into(),
        };
        let b = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-2.0, 2.0]],
            row_index: [1u64, 3].map(antecedent_data::GlobalId::new).into(),
        };
        let mixed = mix_response_influences(&[(0.5, a), (0.5, b)], 4).unwrap();
        assert_eq!(mixed.row_index.iter().map(|id| id.get()).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(mixed.columns[0], vec![-1.0, -2.0, 1.0, 2.0]);
    }
}
