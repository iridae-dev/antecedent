// SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::mix_support_reports;
use super::temporal_path::{
    aggregate_temporal_horizon_evidence, enforce_temporal_response_memory_budget,
    horizon_adjustment_sets_differ, resolve_temporal_response_prior,
};
use super::{
    Arc, AverageEffectQuery, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte,
    BayesianValidation, CausalError, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeAtomFit, EnvelopeOptions, EnvelopeRefutation,
    EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId, ExecutionContext, GraphEffectDraws,
    GraphIdentFlag, GraphPosterior, GraphPosteriorScope, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceDiagnostics, InferenceMode, Instant, Intervention,
    ObservationSpec, OverlapPolicy, PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, Strategy,
    StudyResult, TableView, TabularData, TemporalAtomDesign, TemporalEffectQuery,
    TemporalLinearAdjustment, TemporalRefutation, TemporalResponseEstimator, TimeSeriesData,
    ValidationSink, WeightedGraphSamples, aggregate_effect_envelope, bayesian_gcomp,
    bayesian_temporal_gcomp, elapsed_ns, envelope_refute_atoms, identify_cached_diagnostic,
    provenance_ids, push_conflict_diagnostics,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
    resolve_structural_aggregation, run_envelope_bayesian_full_validation,
};

type PulseWitnessReports = (
    Vec<antecedent_validate::RefutationReport>,
    Vec<antecedent_validate::PredictiveCheckReport>,
    Vec<Diagnostic>,
);

/// `TemporalDag` graph-posterior response: complete-observation `MeanCurve` or
/// one-coordinate `InterventionResponse`. Sequence overlays stay on the explicit path.
/// Collapse a single-coordinate credible pointwise band that retains its draws
/// into a `Scalar` credible interval (posterior SD as the standard error, the
/// same draws so the published `[lower, upper]` is reconstructable). Any other
/// uncertainty — a multi-cell band, a frequentist band, or a band without
/// retained draws — is returned unchanged.
fn collapse_single_coordinate_band_to_scalar(
    uncertainty: ResponseUncertainty,
) -> ResponseUncertainty {
    let ResponseUncertainty::PointwiseBand { level, lower, upper, interpretation, draws } =
        &uncertainty
    else {
        return uncertainty;
    };
    if lower.len() != 1
        || upper.len() != 1
        || *interpretation != antecedent_core::IntervalInterpretation::Credible
    {
        return uncertainty;
    }
    let Some(column) = draws.as_ref().filter(|d| d.n_coordinates() == 1).and_then(|d| d.column(0))
    else {
        return uncertainty;
    };
    let n = column.len();
    if n < 2 {
        return uncertainty;
    }
    let standard_error = antecedent_stats::sample_std(column);
    ResponseUncertainty::Scalar {
        standard_error,
        level: *level,
        lower: lower[0],
        upper: upper[0],
        interpretation: *interpretation,
        draws: Some(antecedent_core::CredibleDraws::scalar(column.to_vec())),
    }
}

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
        let temporal = self.licensed_dbn_posterior_spec(gp, query, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let estimator_id = EstimatorId::temporal_response_for(&self.inference);
        let (identified, identify_cached) =
            self.dbn_response_posterior_identification(gp, &vars, query, physical, ctx)?;
        let inputs = DbnPosteriorInputs { data, query, temporal, treatment, outcome, ctx };
        let mut atoms = dbn_posterior_structural_atoms(&identified);
        let fits = self.fit_dbn_posterior_atoms(&inputs, &identified, &mut atoms)?;
        let DbnAtomFits { weighted, primary, failed_mass, conflict_summary } = fits;
        let (estimand, identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "DBN-posterior response has no evaluable identified atom".into(),
        })?;
        let mixed = mix_dbn_posterior_atoms(
            &inputs,
            &identified,
            &weighted,
            (identification, failed_mass),
        )?;
        let (scalar, standard_error) =
            super::response_path::response_scalar_summary(&mixed.response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            mixed.response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = self.dbn_posterior_diagnostics(
            &inputs,
            (&identified, &weighted),
            &mixed,
            (conflict_summary.as_ref(), identify_cached),
        );
        let (refutations, predictive_checks) = self.refute_dbn_posterior_response(
            &inputs,
            (&identified, &weighted),
            GraphPosteriorScope { posterior: gp, vars: &vars },
            &mut diagnostics,
        )?;
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect \
                 summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &mixed.response.support.warnings {
            diagnostics.push(warning.clone());
        }
        let DbnMixedResponse {
            response, identification, conditional, structural_set, masses, ..
        } = mixed;

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::TemporalBackdoorUnfolded, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
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
                structural_response: Some(dbn_posterior_structural_mixture(
                    atoms,
                    masses,
                    structural_set,
                    conditional,
                )),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// The prepared DBN-posterior identification, the product the plan's lowered
    /// program retained at compile, or a build from the posterior when the study
    /// carries neither.
    fn dbn_response_posterior_identification(
        &self,
        gp: &GraphPosterior,
        vars: &[antecedent_core::VariableId],
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<crate::analysis::prepared::CachedDbnPosteriorIdentification>, bool), CausalError>
    {
        Ok(if let Some(cache) = self.dbn_posterior_cache() {
            (Arc::clone(cache), true)
        } else if let Some(cache) =
            crate::analysis::lowering::plan_retained_dbn_posterior(self, physical)
        {
            (cache, false)
        } else {
            (
                Arc::new(
                    crate::analysis::prepared::build_dbn_posterior_response_identification_cache(
                        gp,
                        vars,
                        query,
                        EstimatorId::temporal_response_for(&self.inference),
                        ctx,
                    )?,
                ),
                false,
            )
        })
    }

    /// The temporal specification of a DBN-posterior response, after the atom
    /// kind, query, observation and memory-budget licenses.
    fn licensed_dbn_posterior_spec<'q>(
        &self,
        gp: &GraphPosterior,
        query: &'q ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<&'q antecedent_core::TemporalResponseSpec, CausalError> {
        if !matches!(gp.atom_kind(), antecedent_discovery::GraphPosteriorAtomKind::Dag) {
            return Err(CausalError::Unsupported {
                message: "DBN-posterior response is licensed for TemporalDag atoms; \
                          TemporalCpdag/TemporalPag posterior atoms use the class-aware combiner",
            });
        }
        dbn_posterior_response_supported(query)?;
        query.require_licensed_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "DBN-posterior response requires TemporalResponseSpec".into(),
        })?;
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        Ok(temporal)
    }

    /// Fits each positive-weight identified atom's temporal response on the
    /// context's workers and folds the results into the structural atoms; an
    /// atom whose estimation fails (without a refusal) counts as failed mass.
    fn fit_dbn_posterior_atoms(
        &self,
        inputs: &DbnPosteriorInputs<'_>,
        identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
        atoms: &mut [crate::result::StructuralResponseAtom],
    ) -> Result<DbnAtomFits, CausalError> {
        let DbnPosteriorInputs { data, query, temporal, treatment, outcome, ctx } = *inputs;
        let mut fits = DbnAtomFits {
            weighted: Vec::new(),
            primary: None,
            failed_mass: 0.0,
            conflict_summary: None,
        };
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        let work: Vec<_> =
            identified.atoms.iter().filter(|atom| key_weights.get(atom.key) > 0.0).collect();
        let fitted = ctx.map_indexed(work.len(), |i, inner| {
            let atom = work[i];
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
            estimator.inner.inference.bootstrap_replicates = self.bootstrap_replicates;
            let (response, conflict) = if let InferenceMode::Bayesian(cfg) = &self.inference {
                let mut bayes = bayesian_gcomp(cfg, inner);
                let (resolved, conflict) = resolve_temporal_response_prior(
                    cfg, data, temporal, &aligned, treatment, outcome, inner,
                )?;
                bayes.prior = resolved;
                (
                    estimator.estimate_bayesian(
                        data,
                        antecedent_estimate::TemporalIdentification {
                            horizons: &identifications,
                            status: aggregate_status,
                            assumptions: aggregate_assumptions,
                        },
                        query,
                        &bayes,
                        inner,
                    ),
                    conflict,
                )
            } else {
                (
                    estimator.estimate(
                        data,
                        &identifications,
                        query,
                        aggregate_status,
                        aggregate_assumptions,
                        inner,
                    ),
                    None,
                )
            };
            if let Err(err @ antecedent_estimate::EstimationError::Refused { .. }) = response {
                return Err(err.into());
            }
            Ok::<_, CausalError>((atom, response, conflict))
        })?;
        for (atom, response, conflict) in fitted {
            if fits.conflict_summary.is_none() {
                fits.conflict_summary = conflict;
            }
            let weight = key_weights.get(atom.key);
            let Ok(response) = response else {
                fits.failed_mass += weight;
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
            if fits.primary.is_none() {
                fits.primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            fits.weighted.push((atom.key, weight, response));
        }
        Ok(fits)
    }

    /// The scalar-free diagnostics of a DBN-posterior response up to the
    /// structural aggregation note: mass split, horizon-dependent adjustment,
    /// withheld multi-atom uncertainty, prior conflict and cache provenance.
    fn dbn_posterior_diagnostics(
        &self,
        inputs: &DbnPosteriorInputs<'_>,
        (identified, weighted): (
            &crate::analysis::prepared::CachedDbnPosteriorIdentification,
            &[(u64, f64, CausalResponse)],
        ),
        mixed: &DbnMixedResponse,
        (conflict_summary, identify_cached): (Option<&antecedent_prob::ConflictSummary>, bool),
    ) -> Vec<Diagnostic> {
        let DbnMasses { identified: identified_mass, unidentified, failed } = mixed.masses;
        let mut diagnostics = vec![
            Diagnostic::new(
                "estimate.response.dbn_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "frozen posterior_probability weights; identified_mass={identified_mass}, \
                     unidentified_mass={unidentified}; unevaluable_mass={failed}; TemporalDag \
                     atoms, not completion enumeration",
                ),
            )
            .with_fields(super::mass_fields(Some(identified_mass), unidentified)),
        ];
        let aligned_all = identified
            .atoms
            .iter()
            .filter_map(|atom| atom.horizons.as_ref())
            .flat_map(|horizons| {
                inputs.temporal.horizons.iter().filter_map(|&horizon| horizons.get(horizon))
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
                if self.inference.is_bayesian() {
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
        if let Some(summary) = conflict_summary {
            push_conflict_diagnostics(&mut diagnostics, summary);
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            mixed.aggregation_policy,
            identified_mass,
            unidentified,
            failed,
            0.0,
        );
        diagnostics
    }

    /// Runs the Pulse-witness refutation suite for an `InterventionResponse`
    /// when the study refutes, otherwise notes that scalar refuters are skipped.
    fn refute_dbn_posterior_response(
        &self,
        inputs: &DbnPosteriorInputs<'_>,
        (identified, weighted): (
            &crate::analysis::prepared::CachedDbnPosteriorIdentification,
            &[(u64, f64, CausalResponse)],
        ),
        scope: GraphPosteriorScope<'_>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (
            Vec<antecedent_validate::RefutationReport>,
            Vec<antecedent_validate::PredictiveCheckReport>,
        ),
        CausalError,
    > {
        let DbnPosteriorInputs { data, query, ctx, .. } = *inputs;
        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, predictive_checks, refute_diags) = if is_intervention
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            self.refute_dbn_posterior_intervention(data, query, identified, weighted, scope, ctx)?
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
        Ok((refutations, predictive_checks))
    }

    fn refute_dbn_posterior_intervention(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
        weighted: &[(u64, f64, CausalResponse)],
        scope: GraphPosteriorScope<'_>,
        ctx: &ExecutionContext,
    ) -> Result<PulseWitnessReports, CausalError> {
        let GraphPosteriorScope { posterior: gp, vars } = scope;
        let pulse = response_pulse_witness(query)?;
        let ate_query = AverageEffectQuery::binary_ate(pulse.treatment, pulse.outcome);
        let tabular = TabularData::new(data.storage().clone());
        let notes = vec![Diagnostic::new(
            "refute.dbn_posterior.intervention_pulse_suite",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "InterventionResponse cheap/full run the TemporalDag Pulse-native suite on each \
             contributing atom against that atom's own Pulse witness and mix by frozen graph \
             weight; the mixed check passes only if every atom passes",
        )];
        let witness = DbnPulseWitness {
            data,
            identified,
            weighted,
            pulse: &pulse,
            ate_query: &ate_query,
            tabular: &tabular,
            ctx,
        };
        match &self.inference {
            InferenceMode::Frequentist => self.refute_dbn_pulse_frequentist(&witness, notes),
            InferenceMode::Bayesian(cfg) => {
                let reports = self.refute_dbn_pulse_bayesian(&witness, cfg, notes);
                let _ = (gp, vars);
                reports
            }
        }
    }

    /// The Pulse-native refutation suite on each contributing atom's linear
    /// witness design, mixed by frozen graph weight.
    fn refute_dbn_pulse_frequentist(
        &self,
        witness: &DbnPulseWitness<'_>,
        mut notes: Vec<Diagnostic>,
    ) -> Result<PulseWitnessReports, CausalError> {
        let DbnPulseWitness { data, identified, weighted, pulse, ate_query, tabular, ctx } =
            *witness;
        let mut refute_atoms = Vec::new();
        for (key, weight, _) in weighted {
            let atom = identified.atoms.iter().find(|candidate| candidate.key == *key).ok_or_else(
                || CausalError::Compile {
                    message: "DBN-posterior IR refuter missing contributing atom".into(),
                },
            )?;
            let (estimand, indexer) = pulse_witness_design(atom, pulse.horizon_steps);
            let design = TemporalAtomDesign::linear(
                data,
                estimand,
                pulse,
                indexer,
                self.split.as_ref(),
                ctx,
            )?;
            refute_atoms.push(EnvelopeRefuteAtom {
                key: *key,
                weight: *weight,
                estimand: estimand.clone(),
                indexer: Some(indexer.clone()),
                original: design.effect_estimate(atom.identification.required_assumptions.clone()),
            });
        }
        let (reports, mix_notes) = EnvelopeRefutation {
            data: tabular,
            query: ate_query,
            suite: self.refute,
            estimator: "temporal.linear.adjustment",
            custom: &self.custom_validators,
            temporal: Some(TemporalRefutation {
                query: pulse,
                split: self.split.as_ref(),
                time_index: data.time_index(),
                gaps: false,
            }),
        }
        .run(&refute_atoms, &mut EstimationWorkspace::default(), ctx)?;
        notes.extend(mix_notes);
        Ok((reports, Vec::new(), notes))
    }

    /// The Pulse-native refutation suite and full Bayesian validation on each
    /// contributing atom's Bayesian witness fit, aggregated by frozen weight.
    fn refute_dbn_pulse_bayesian(
        &self,
        witness: &DbnPulseWitness<'_>,
        cfg: &BayesianConfig,
        mut notes: Vec<Diagnostic>,
    ) -> Result<PulseWitnessReports, CausalError> {
        let DbnPulseWitness { data, identified, weighted, pulse, ate_query, tabular, ctx } =
            *witness;
        let bayes = bayesian_temporal_gcomp(cfg, ctx);
        let mut ws = BayesianGCompWorkspace::default();
        let mut fits = Vec::new();
        let mut draws = Vec::new();
        for (key, weight, _) in weighted {
            let atom = identified.atoms.iter().find(|candidate| candidate.key == *key).ok_or_else(
                || CausalError::Compile {
                    message: "DBN-posterior IR refuter missing contributing atom".into(),
                },
            )?;
            let (estimand, indexer) = pulse_witness_design(atom, pulse.horizon_steps);
            let mut temporal_est = TemporalLinearAdjustment::new();
            temporal_est.inner.overlap = OverlapPolicy::ExplicitOverride;
            let prep = temporal_est
                .prepare(data, estimand, pulse, indexer, self.split.as_ref(), &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let names =
                antecedent_estimate::temporal_coefficient_names(data, estimand, pulse, indexer)
                    .map_err(CausalError::from)?;
            let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                .map_err(CausalError::from)?;
            let posterior = bayes.fit(&bprep, atom.identification.status, &mut ws, ctx)?;
            let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                message: "Bayesian Pulse witness posterior missing effect column".into(),
            })?;
            let effect = posterior.draws.column(col)?;
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
        let (reports, mix_notes) = EnvelopeRefutation {
            data: tabular,
            query: ate_query,
            suite: self.refute,
            estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
            custom: &self.custom_validators,
            temporal: Some(TemporalRefutation {
                query: pulse,
                split: self.split.as_ref(),
                time_index: data.time_index(),
                gaps: false,
            }),
        }
        .run(&envelope_refute_atoms(&fits)?, &mut EstimationWorkspace::default(), ctx)?;
        notes.extend(mix_notes);
        let flags = weighted
            .iter()
            .map(|(key, _, _)| (*key, GraphIdentFlag::Identified))
            .collect::<Vec<_>>();
        let graphs = WeightedGraphSamples::new(
            weighted.iter().map(|(_, weight, _)| *weight).collect::<Vec<_>>(),
            flags.iter().map(|(_, flag)| *flag).collect::<Vec<_>>(),
            weighted.iter().map(|(key, _, _)| *key).collect::<Vec<_>>(),
        )?;
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
            BayesianValidation { suite: self.refute, cfg, estimator: &bayes.inner },
            &fits,
            &mut posterior,
            pulse_mean,
            ctx,
            super::super::latency::predictive_check_sims(self.latency_mode),
            ValidationSink { refutations: &mut refutations, diagnostics: &mut notes },
        )?;
        Ok((refutations, predictive, notes))
    }
}

/// The series, query, specification and context of a DBN-posterior response.
#[derive(Clone, Copy)]
struct DbnPosteriorInputs<'a> {
    data: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    ctx: &'a ExecutionContext,
}

/// The fitted atoms of a DBN-posterior response: each evaluated atom's key,
/// frozen weight and response, the first atom's identification products, the
/// failed mass and the first prior-conflict summary.
struct DbnAtomFits {
    weighted: Vec<(u64, f64, CausalResponse)>,
    primary: Option<(IdentifiedEstimand, IdentificationResult)>,
    failed_mass: f64,
    conflict_summary: Option<antecedent_prob::ConflictSummary>,
}

/// Identified, unidentified and unevaluable shares of the posterior mass.
#[derive(Clone, Copy)]
struct DbnMasses {
    identified: f64,
    unidentified: f64,
    failed: f64,
}

/// The mixed DBN-posterior response with the products its diagnostics and
/// structural mixture are built from.
struct DbnMixedResponse {
    response: CausalResponse,
    identification: IdentificationResult,
    conditional: antecedent_core::ResponseValue,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
    aggregation_policy: crate::result::StructuralAggregationPolicy,
    masses: DbnMasses,
}

/// The structural mixture of a DBN-posterior response: the fitted atoms under
/// frozen posterior-probability weights.
fn dbn_posterior_structural_mixture(
    atoms: Vec<crate::result::StructuralResponseAtom>,
    masses: DbnMasses,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
    conditional: antecedent_core::ResponseValue,
) -> crate::result::StructuralResponseMixture {
    crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass: masses.identified,
        unidentified_mass: masses.unidentified,
        unevaluable_mass: masses.failed,
        subsampled_out_mass: 0.0,
        identified_set: structural_set,
        identified_set_interval: None,
        conditional_on_identified: Some(conditional),
        full_mass_scope: true,
        truncated_atoms: 0,
    }
}

/// One structural atom per posterior graph, identified or not, before fitting.
pub(super) fn dbn_posterior_structural_atoms(
    identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
) -> Vec<crate::result::StructuralResponseAtom> {
    identified
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
        .collect::<Vec<_>>()
}

/// The structural aggregation policy of the contributing atoms' estimands at
/// the first horizon, with partial identification at any requested horizon.
fn dbn_posterior_aggregation_policy(
    identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
    weighted: &[(u64, f64, CausalResponse)],
    temporal: &antecedent_core::TemporalResponseSpec,
) -> crate::result::StructuralAggregationPolicy {
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
    resolve_structural_aggregation(&contributing, any_partial)
}

/// Mixes the evaluated atoms' responses by frozen graph weight into the
/// DBN-posterior response, and resolves its identification status.
fn mix_dbn_posterior_atoms(
    inputs: &DbnPosteriorInputs<'_>,
    identified: &crate::analysis::prepared::CachedDbnPosteriorIdentification,
    weighted: &[(u64, f64, CausalResponse)],
    (mut identification, failed_mass): (IdentificationResult, f64),
) -> Result<DbnMixedResponse, CausalError> {
    let DbnPosteriorInputs { query, temporal, .. } = *inputs;
    let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
    let total_mass = identified.graphs.total_weight();
    let unidentified_mass = identified.graphs.unidentified_mass();
    let aggregation_policy = dbn_posterior_aggregation_policy(identified, weighted, temporal);
    let conditional_values = weighted
        .iter()
        .filter_map(|(_, weight, response)| {
            super::response_path::response_identified_value(response).map(|value| (*weight, value))
        })
        .collect::<Vec<_>>();
    let conditional_refs =
        conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
    let conditional = super::response_path::mix_response_values(&conditional_refs)?;
    let structural_set = super::response_path::response_envelope_from_weighted(weighted);
    let first = &weighted[0].2;
    let graph_dependent = resolve_posterior_status(
        identification.status,
        crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
        PosteriorMassSummary {
            unidentified: unidentified_mass,
            failed: failed_mass,
            ..PosteriorMassSummary::default()
        },
    ) == IdentificationStatus::GraphDependent;
    if graph_dependent {
        identification.status = IdentificationStatus::GraphDependent;
    }
    let uncertainty = if weighted.len() == 1 {
        let base = first.uncertainty.clone();
        // A single-atom, single-coordinate InterventionResponse is a scalar
        // estimand: surface its per-atom credible band as a Scalar so the
        // facade publishes scalar uncertainty (a MeanCurve stays a band).
        if matches!(query.functional, ResponseFunctional::InterventionResponse { .. }) {
            collapse_single_coordinate_band_to_scalar(base)
        } else {
            base
        }
    } else {
        ResponseUncertainty::None
    };
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
    Ok(DbnMixedResponse {
        response,
        identification,
        conditional,
        structural_set,
        aggregation_policy,
        masses: DbnMasses {
            identified: identified_mass / total_mass,
            unidentified: unidentified_mass / total_mass,
            failed: failed_mass / total_mass,
        },
    })
}

/// The series, atoms, Pulse witness and ATE query a DBN-posterior
/// intervention refutation runs against.
#[derive(Clone, Copy)]
struct DbnPulseWitness<'a> {
    data: &'a TimeSeriesData,
    identified: &'a crate::analysis::prepared::CachedDbnPosteriorIdentification,
    weighted: &'a [(u64, f64, CausalResponse)],
    pulse: &'a TemporalEffectQuery,
    ate_query: &'a AverageEffectQuery,
    tabular: &'a TabularData,
    ctx: &'a ExecutionContext,
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
        active,
        ..super::temporal_path::unit_contrast_query(
            temporal,
            (treatment, outcome),
            horizon,
            query.target_population.clone(),
        )
    })
}

fn pulse_witness_design(
    atom: &crate::analysis::prepared::CachedDbnPosteriorAtomIdentification,
    horizon: u32,
) -> (&IdentifiedEstimand, &antecedent_data::TemporalIndexer) {
    if let Some(horizons) = atom.horizons.as_ref() {
        if let Some(entry) = horizons.get(horizon) {
            return (&entry.estimand, &entry.indexer);
        }
    }
    (&atom.estimand, &atom.indexer)
}
