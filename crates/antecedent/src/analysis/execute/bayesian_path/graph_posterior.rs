//! Static graph-posterior effect execution.

use super::{
    Arc, AverageEffectQuery, BayesianGComputationAte, BayesianGraphPosteriorPriorAnchor,
    BayesianValidation, CausalError, CausalPosterior, CausalQuery, ConditionalLinearAdjustment,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeAtomFit,
    EnvelopeOptions, EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId,
    ExecutionContext, GraphEffectDraws, GraphPosterior, GraphPosteriorAtomValue,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId, InferenceDiagnostics,
    InferenceMode, Instant, InteractiveSubsampleDrop, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, PredictiveCheckReport, PreparedBayesianProblem, PriorSet, RefuteSuite,
    StaticEstimateInput, StaticEstimateWorkspaces, Strategy, StructuralAggregationPolicy,
    StudyResult, TabularData, ValidationSink, WeightedGraphSamples, aggregate_effect_envelope,
    bayesian_gcomp, bayesian_gcomp_estimator, effect_from_posterior, elapsed_ns,
    envelope_mass_diagnostic, envelope_refute_atoms, envelope_se_omission_diagnostic,
    estimate_static_effect, identified_envelope_keys, interactive_subsample_graphs_accounted,
    mass_fields, mix_graph_posterior_identified_atoms, mix_static_envelope_se,
    mix_weighted_analytic_se, mixed_static_influence, overlap_diagnostic, overlap_limit_diagnostic,
    provenance_ids, push_conflict_diagnostics, report_subsampled_out_mass,
    resolve_posterior_status, retain_envelope_assumptions, run_envelope_bayesian_full_validation,
    static_aligned_influence, with_conflict_summary,
};

/// Atoms of a Bayesian graph-posterior envelope after preparation and Interactive subsampling.
struct GraphPosteriorBayesianPrepared {
    graphs: WeightedGraphSamples,
    subsample_notes: Vec<Diagnostic>,
    subsample_drop: InteractiveSubsampleDrop,
    kept: Vec<(u64, IdentifiedEstimand, IdentificationResult)>,
    prepared: std::collections::HashMap<u64, PreparedBayesianProblem>,
    anchor: BayesianGraphPosteriorPriorAnchor,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Fitted atoms of a Bayesian graph-posterior envelope and its primary estimand.
struct GraphPosteriorBayesianFits {
    per_graph: Vec<GraphEffectDraws>,
    atoms: Vec<EnvelopeAtomFit>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
}

/// Borrowed inputs of the Bayesian graph-posterior envelope mix.
#[derive(Clone, Copy)]
struct GraphPosteriorBayesianMixInput<'a> {
    graphs: &'a WeightedGraphSamples,
    identified_graphs: &'a WeightedGraphSamples,
    subsample_drop: &'a InteractiveSubsampleDrop,
}

/// Mixed Bayesian graph-posterior envelope: posterior, scalar estimate, and the mix policy.
struct GraphPosteriorBayesianMix {
    posterior: CausalPosterior,
    estimate: EffectEstimate,
    mixed: super::super::MixedGraphPosteriorPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    unidentified_mass: f64,
}

/// Borrowed inputs of the Bayesian graph-posterior validate stage.
struct GraphPosteriorBayesianValidationInput<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    estimate_ate: f64,
    graph_dependent: bool,
}

/// Envelope aggregation of the fitted atoms plus the identified-atom scalar mix.
fn mix_graph_posterior_bayesian_envelope(
    input: &GraphPosteriorBayesianMixInput<'_>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
    fitted: &mut GraphPosteriorBayesianFits,
) -> Result<GraphPosteriorBayesianMix, CausalError> {
    let GraphPosteriorBayesianMixInput { graphs, identified_graphs, subsample_drop } = *input;
    let GraphPosteriorBayesianFits { per_graph, atoms, primary_estimand, primary_identification } =
        fitted;
    let (primary_estimand, primary_identification) =
        (primary_estimand.take(), primary_identification.take());
    let mut posterior = aggregate_effect_envelope(
        graphs,
        per_graph,
        InferenceDiagnostics::analytic("graph_posterior_envelope"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, identified_graphs, subsample_drop);
    retain_envelope_assumptions(&mut posterior, atoms);
    if let Some(summary) = envelope_conflict {
        posterior = with_conflict_summary(posterior, summary);
    }
    let mut estimate = effect_from_posterior(&posterior)?;
    let atom_values: Vec<GraphPosteriorAtomValue> = atoms
        .iter()
        .filter_map(|atom| {
            effect_from_posterior(&atom.posterior).ok().map(|summary| GraphPosteriorAtomValue {
                key: atom.key,
                weight: atom.weight,
                status: atom.status,
                estimand: atom.estimand.clone(),
                value: summary.ate,
            })
        })
        .collect();
    let unidentified_mass = identified_graphs.unidentified_mass();
    let mixed = mix_graph_posterior_identified_atoms(
        identified_graphs,
        &atom_values,
        unidentified_mass,
        0.0,
        subsample_drop.mass,
        0,
    );
    if mixed.mixable_scalar {
        estimate.ate = mixed.ate;
    } else {
        estimate.ate = f64::NAN;
        estimate.uncertainty.analytic.se = f64::NAN;
    }
    let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: no identified graph atoms".into(),
    })?;
    identification.status = resolve_posterior_status(
        identification.status,
        mixed.policy,
        PosteriorMassSummary {
            unidentified: unidentified_mass.max(posterior.unidentified_mass),
            unevaluable: posterior.unevaluable_mass,
            subsampled: subsample_drop.mass.max(posterior.subsampled_out_mass),
            ..PosteriorMassSummary::default()
        },
    );
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: missing estimand".into(),
    })?;
    Ok(GraphPosteriorBayesianMix {
        posterior,
        estimate,
        mixed,
        identification,
        estimand,
        unidentified_mass,
    })
}

/// Diagnostics of the Bayesian graph-posterior envelope: identification's, subsample notes,
/// structural aggregation, overlap, mass, and prior conflict.
fn graph_posterior_bayesian_diagnostics(
    mix: &GraphPosteriorBayesianMix,
    subsample_notes: Vec<Diagnostic>,
    subsample_mass: f64,
) -> Vec<Diagnostic> {
    let unidentified_mass = mix.unidentified_mass;
    let mut diagnostics = mix.identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; subsampled_out_mass={}",
                mix.mixed.policy.as_str(),
                mix.mixed.mixture.identified_mass,
                subsample_mass
            ),
        )
        .with_fields(mass_fields(Some(mix.mixed.mixture.identified_mass), mix.unidentified_mass)),
    );
    diagnostics.push(overlap_diagnostic(mix.estimate.provenance.overlap));
    diagnostics.extend(overlap_limit_diagnostic(mix.estimate.diagnostics.overlap_report.as_ref()));
    diagnostics.push(envelope_mass_diagnostic("estimate.graph_posterior.envelope", &mix.posterior));
    if let Some(cs) = mix.posterior.conflict_summary.as_ref() {
        push_conflict_diagnostics(&mut diagnostics, cs);
    }
    diagnostics
}

/// The run-level settings a Frequentist graph-posterior effect reads: the study's own when the
/// study runs it, the sealed operation's when a checked operation does. Resolved once, so the
/// estimate and refute phases never re-decide which side to read.
struct EffectRunConfig<'a> {
    latency_mode: Option<super::super::LatencyMode>,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<&'a super::super::PopulationRegistry>,
    /// A sealed operation carries no custom validators.
    custom_validators: &'a [std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>],
    /// The estimator procedure every atom is fitted with.
    spec: crate::estimator_spec::EstimatorSpec,
}

impl<'a> EffectRunConfig<'a> {
    fn resolve(
        study: &'a super::super::Study,
        checked: Option<&'a super::super::super::CheckedGraphPosteriorEffect>,
        estimator_id: EstimatorId,
    ) -> Self {
        match checked {
            Some(operation) => Self {
                latency_mode: operation.latency_mode(),
                refute: operation.validation(),
                bootstrap_replicates: operation.bootstrap_replicates(),
                overlap_policy: Some(operation.overlap()),
                population_registry: operation
                    .population_registry()
                    .or(study.declared.population_registry.as_ref()),
                custom_validators: &[],
                spec: operation.procedure().clone(),
            },
            None => Self {
                latency_mode: study.latency_mode,
                refute: study.settings.refute,
                bootstrap_replicates: study.settings.bootstrap_replicates,
                overlap_policy: study.declared.overlap_policy,
                population_registry: study.declared.population_registry.as_ref(),
                custom_validators: &study.declared.custom_validators,
                spec: study
                    .declared
                    .estimator_spec
                    .as_deref()
                    .cloned()
                    .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id)),
            },
        }
    }
}

/// Resolved inputs shared by the Frequentist graph-posterior estimate and refute phases.
#[derive(Clone, Copy)]
struct GraphPosteriorFrequentistRoute<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    run: &'a EffectRunConfig<'a>,
    estimator_id: EstimatorId,
    conditional: bool,
}

/// Per-atom estimates of a Frequentist graph-posterior mixture before mixing.
struct GraphPosteriorFrequentistAtoms {
    se_items: Vec<(f64, f64)>,
    atom_ifs: Vec<Vec<f64>>,
    atom_weights: Vec<f64>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
    assumptions: antecedent_core::AssumptionSet,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

/// Mixed Frequentist graph-posterior estimate with the mass and SE bookkeeping it reports.
struct GraphPosteriorFrequentistMix {
    mixed: super::super::MixedGraphPosteriorPolicy,
    estimate: EffectEstimate,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    unidentified_mass: f64,
    subsampled_out_mass: f64,
    n_contributing: usize,
    joint: bool,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

/// Frozen-weight mix of the per-atom estimates: scalar, SE (joint influence-function
/// covariance when every contributing atom aligned), and the resolved identification status.
fn mix_graph_posterior_frequentist_atoms(
    identified_graphs: &WeightedGraphSamples,
    subsampled_out_mass: f64,
    atoms: GraphPosteriorFrequentistAtoms,
) -> Result<GraphPosteriorFrequentistMix, CausalError> {
    let GraphPosteriorFrequentistAtoms {
        se_items,
        atom_ifs,
        atom_weights,
        primary_estimand,
        primary_identification,
        assumptions,
        refute_atoms,
    } = atoms;
    if !matches!(
        refute_atoms.iter().map(|a| a.weight).sum::<f64>().partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(CausalError::Compile {
            message: "graph-posterior envelope: no identified graph atoms".into(),
        });
    }
    let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: no identified graph atoms".into(),
    })?;
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: missing estimand".into(),
    })?;
    let atom_values: Vec<GraphPosteriorAtomValue> = refute_atoms
        .iter()
        .map(|atom| GraphPosteriorAtomValue {
            key: atom.key,
            weight: atom.weight,
            status: IdentificationStatus::NonparametricallyIdentified,
            estimand: atom.estimand.clone(),
            value: atom.original.ate,
        })
        .collect();
    let unidentified_mass = identified_graphs.unidentified_mass();
    let mixed = mix_graph_posterior_identified_atoms(
        identified_graphs,
        &atom_values,
        unidentified_mass,
        0.0,
        subsampled_out_mass,
        0,
    );
    let n_contributing = se_items.len();
    let joint = mixed.mixable_scalar && n_contributing > 1 && atom_ifs.len() == n_contributing;
    let se = if mixed.mixable_scalar {
        if joint {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        }
    } else {
        f64::NAN
    };
    let mut estimate =
        EffectEstimate::new(mixed.ate, se, assumptions, OverlapPolicy::ExplicitOverride);
    if joint {
        if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
            estimate.scores.influence = Some(inf);
        }
    }
    identification.status = resolve_posterior_status(
        identification.status,
        mixed.policy,
        PosteriorMassSummary {
            unidentified: unidentified_mass,
            subsampled: subsampled_out_mass,
            ..PosteriorMassSummary::default()
        },
    );
    Ok(GraphPosteriorFrequentistMix {
        mixed,
        estimate,
        identification,
        estimand,
        unidentified_mass,
        subsampled_out_mass,
        n_contributing,
        joint,
        refute_atoms,
    })
}

/// Diagnostics of the Frequentist graph-posterior mixture: identification's, subsample
/// notes, structural aggregation, overlap, SE provenance, and the envelope summary.
fn graph_posterior_frequentist_diagnostics(
    mix: &GraphPosteriorFrequentistMix,
    subsample_notes: Vec<Diagnostic>,
) -> Vec<Diagnostic> {
    let GraphPosteriorFrequentistMix {
        mixed,
        estimate,
        identification,
        unidentified_mass,
        subsampled_out_mass,
        n_contributing,
        joint,
        refute_atoms,
        ..
    } = mix;
    let (unidentified_mass, subsampled_out_mass, n_contributing, joint) =
        (*unidentified_mass, *subsampled_out_mass, *n_contributing, *joint);
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; subsampled_out_mass={subsampled_out_mass}",
                mixed.policy.as_str(),
                mixed.mixture.identified_mass,
            ),
        )
        .with_fields(mass_fields(Some(mixed.mixture.identified_mass), unidentified_mass)),
    );
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
    diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
    if joint && estimate.se_analytic().is_finite() {
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.joint_if_se",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "SE of E[τ | identified] from the joint influence-function covariance of \
                 {n_contributing} atoms fit on shared rows, with frozen graph weights; \
                 graph-weight uncertainty and unidentified mass are not in the SE; the \
                 interval is for the reported aggregate, not a distribution over \
                 graph-specific effects"
            ),
        ));
    }
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            if mixed.mixable_scalar {
                format!(
                    "published effect is E[τ | identified]; identified_mass={}, \
                     unidentified_mass={unidentified_mass}, subsampled_out_mass={subsampled_out_mass}, \
                     atoms={}",
                    mixed.mixture.identified_mass,
                    refute_atoms.len()
                )
            } else {
                format!(
                    "scalar withheld: contributing atoms disagree on estimand identity; \
                     identified_mass={}, unidentified_mass={unidentified_mass}, \
                     subsampled_out_mass={subsampled_out_mass}, atoms={}",
                    mixed.mixture.identified_mass,
                    refute_atoms.len()
                )
            },
        )
        .with_fields(mass_fields(
            Some(mixed.mixture.identified_mass),
            unidentified_mass,
        )),
    );
    diagnostics
}

impl super::super::Study {
    /// Execute a supplied static graph posterior into a Bayesian effect envelope.
    ///
    /// Identifies and fits each posterior atom independently; atoms that fail to
    /// identify (or fail to build a valid DAG from their adjacency mask) are marked
    /// [`GraphIdentFlag::Unidentified`] and contribute their posterior weight to
    /// [`antecedent_estimate::CausalPosterior::unidentified_mass`] rather than being
    /// dropped or having their mass redistributed onto the identified atoms.
    /// Identified atoms the Interactive tier leaves out of its subsample are
    /// reported as [`antecedent_estimate::CausalPosterior::subsampled_out_mass`]
    /// instead. Either kind of uncovered mass makes the result graph-dependent.
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist`, or any per-graph identification/estimation
    /// infrastructure failure (envelope aggregation, missing effect column, …).
    pub(in crate::analysis) fn execute_graph_posterior_bayesian(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.settings.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "graph-posterior discovery requires inference=Bayesian for effect mixture",
                ));
            }
        };
        let est = bayesian_gcomp(&cfg, ctx);
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));

        let (identified, identify_cached) =
            self.graph_posterior_identification_for_bayesian(gp, query, physical, ctx)?;
        let GraphPosteriorBayesianPrepared {
            graphs,
            subsample_notes,
            subsample_drop,
            kept,
            mut prepared,
            anchor,
            envelope_conflict,
        } =
            self.prepare_graph_posterior_bayesian_atoms(data, query, &identified, &est, &cfg, ctx)?;
        let mut fitted = Self::fit_graph_posterior_bayesian_atoms(
            &est,
            &graphs,
            &kept,
            &mut prepared,
            anchor,
            ctx,
        )?;
        let mix = mix_graph_posterior_bayesian_envelope(
            &GraphPosteriorBayesianMixInput {
                graphs: &graphs,
                identified_graphs: &identified.graphs,
                subsample_drop: &subsample_drop,
            },
            envelope_conflict,
            &mut fitted,
        )?;
        let mut diagnostics =
            graph_posterior_bayesian_diagnostics(&mix, subsample_notes, subsample_drop.mass);
        let GraphPosteriorBayesianMix {
            mut posterior,
            estimate,
            mixed,
            identification,
            estimand,
            ..
        } = mix;

        let (refutations, predictive_checks) = self.validate_graph_posterior_bayesian(
            &GraphPosteriorBayesianValidationInput {
                data,
                query,
                cfg: &cfg,
                est: &est,
                estimate_ate: estimate.ate,
                graph_dependent: matches!(
                    mixed.policy,
                    StructuralAggregationPolicy::GraphDependentAtoms
                ),
            },
            &fitted.atoms,
            &mut posterior,
            &mut diagnostics,
            ctx,
        )?;

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::BackdoorAdjustment,
                    bayesian_gcomp_estimator(conditional),
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_gcomp",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        ))
    }

    /// The frozen graph-posterior identification: the prepared cache, the product the
    /// plan's lowered program retained at compile, or a fresh build.
    fn graph_posterior_identification_for_bayesian(
        &self,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<
        (Arc<super::super::super::prepared::CachedGraphPosteriorIdentification>, bool),
        CausalError,
    > {
        Ok(if let Some(cache) = self.graph_posterior_cache() {
            (Arc::clone(cache), true)
        } else if let Some(cache) =
            crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
        {
            (cache, false)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
                false,
            )
        })
    }

    /// Prepare every identified atom's design once, resolve the shared prior anchor on the
    /// first atom, then Interactive-subsample and keep the atoms that still contribute draws.
    fn prepare_graph_posterior_bayesian_atoms(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        identified: &super::super::super::prepared::CachedGraphPosteriorIdentification,
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorBayesianPrepared, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let graphs = identified.graphs.clone();
        let fit_atoms: Vec<_> = identified
            .atoms
            .iter()
            .map(|atom| (atom.key, atom.estimand.clone(), atom.identification.clone()))
            .collect();
        let original_anchor_key = fit_atoms.first().map(|(key, _, _)| *key);
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

        // Prepare once before Interactive subsample, stash
        // so kept atoms are not prepared a second time. Keys may collide when
        // several atoms share an adjacency mask — keep the first prep per key.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        let preps = ctx.map_indexed(fit_atoms.len(), |i, _inner| {
            let (key, estimand, _) = &fit_atoms[i];
            let prep = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                est.prepare_conditional(data, estimand, &q)
            } else {
                est.prepare(data, estimand, query)
            }
            .map_err(CausalError::from)?;
            Ok::<_, CausalError>((*key, prep))
        })?;
        for (i, (key, prep)) in preps.into_iter().enumerate() {
            if i == 0 {
                let anchor =
                    super::super::BayesianGraphPosteriorPriorAnchor::resolve(key, cfg, &prep, ctx)?;
                envelope_prior = anchor.prior;
                envelope_conflict = anchor.conflict;
            }
            prepared.entry(key).or_insert(prep);
        }
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            graphs,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut seen_keys = std::collections::HashSet::new();
        let kept: Vec<_> = fit_atoms
            .into_iter()
            .filter(|(key, _, _)| {
                keep.contains(key) && prepared.contains_key(key) && seen_keys.insert(*key)
            })
            .collect();
        let Some(anchor_key) = original_anchor_key else {
            return Err(CausalError::Compile {
                message:
                    "Bayesian graph-posterior envelope has no identified atom for prior anchoring"
                        .into(),
            });
        };
        let anchor = super::super::BayesianGraphPosteriorPriorAnchor {
            key: anchor_key,
            prior: envelope_prior.clone(),
            conflict: envelope_conflict.clone(),
        };
        Ok(GraphPosteriorBayesianPrepared {
            graphs,
            subsample_notes,
            subsample_drop,
            kept,
            prepared,
            anchor,
            envelope_conflict,
        })
    }

    /// Fit the kept atoms under the original prior anchor; the public estimand and
    /// identification come from the first atom that actually contributes draws.
    fn fit_graph_posterior_bayesian_atoms(
        est: &BayesianGComputationAte,
        graphs: &WeightedGraphSamples,
        kept: &[(u64, IdentifiedEstimand, IdentificationResult)],
        prepared: &mut std::collections::HashMap<u64, PreparedBayesianProblem>,
        anchor: BayesianGraphPosteriorPriorAnchor,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorBayesianFits, CausalError> {
        let anchor_key = anchor.key;
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let key_weights = IdentifiedKeyWeights::new(graphs);
        let inputs = kept
            .iter()
            .map(|(key, estimand, identification)| {
                let prep = prepared.remove(key).ok_or_else(|| CausalError::Compile {
                    message: "kept Bayesian graph-posterior key has no prepared design".into(),
                })?;
                Ok(super::super::BayesianGraphPosteriorAtomInput {
                    key: *key,
                    weight: key_weights.get(*key),
                    status: identification.status,
                    estimand: estimand.clone(),
                    prepared: prep,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?;
        let fitted =
            super::super::BayesianGraphPosteriorAtomFits::fit(est.clone(), anchor, inputs, ctx)?;
        if fitted.prior_anchor.key != anchor_key {
            return Err(CausalError::Compile {
                message: "Bayesian graph-posterior fit changed its original prior anchor".into(),
            });
        }
        let mut per_graph = Vec::new();
        let mut atoms = Vec::new();
        per_graph.extend(fitted.draw_columns.iter().cloned());
        for atom in fitted.atoms {
            let key = atom.key;
            let Some((_, estimand, identification)) =
                kept.iter().find(|(found, _, _)| *found == key)
            else {
                return Err(CausalError::Compile {
                    message: "Bayesian graph-posterior fit returned an unknown atom key".into(),
                });
            };
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(identification.clone());
            }
            atoms.push(EnvelopeAtomFit {
                key,
                prep: atom.prepared,
                posterior: atom.posterior,
                status: atom.status,
                weight: atom.weight,
                estimand: atom.estimand,
                indexer: None,
                prior: atom.prior,
            });
        }
        Ok(GraphPosteriorBayesianFits {
            per_graph,
            atoms,
            primary_estimand,
            primary_identification,
        })
    }

    /// Validate stage of the Bayesian graph-posterior envelope: the refutation suite per atom
    /// group (graph-dependent atoms are refuted one by one) and the predictive checks.
    fn validate_graph_posterior_bayesian(
        &self,
        input: &GraphPosteriorBayesianValidationInput<'_>,
        atoms: &[EnvelopeAtomFit],
        posterior: &mut CausalPosterior,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let GraphPosteriorBayesianValidationInput {
            data,
            query,
            cfg,
            est,
            estimate_ate,
            graph_dependent,
        } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.settings.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let refute_atoms = envelope_refute_atoms(atoms)?;
                let groups: Vec<&[EnvelopeRefuteAtom]> = if graph_dependent {
                    refute_atoms.chunks(1).collect()
                } else {
                    vec![&refute_atoms]
                };
                let mut reports = Vec::new();
                for group in groups {
                    let (atom_reports, mut notes) = EnvelopeRefutation {
                        data,
                        query,
                        suite: self.settings.refute,
                        estimator: "bayesian.gcomp",
                        custom: &self.declared.custom_validators,
                        temporal: None,
                    }
                    .run(group, &mut refute_ws, ctx)?;
                    if graph_dependent {
                        notes.retain(|d| d.code.as_ref() != "refute.envelope.effect_mixture");
                    }
                    diagnostics.extend(notes);
                    reports.extend(atom_reports);
                }
                if graph_dependent {
                    diagnostics.push(Diagnostic::new(
                        "refute.envelope.graph_dependent_atoms",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "outer scalar mix skipped under graph_dependent_atoms; Bayesian atom refutations remain conditional on their own estimands",
                    ));
                }
                reports
            }
        };
        let predictive_checks = if graph_dependent {
            let mut checks = Vec::new();
            for atom in atoms {
                let mut atom_posterior = atom.posterior.clone();
                let atom_effect = effect_from_posterior(&atom_posterior)?.ate;
                checks.extend(run_envelope_bayesian_full_validation(
                    BayesianValidation { suite: self.settings.refute, cfg, estimator: est },
                    std::slice::from_ref(atom),
                    &mut atom_posterior,
                    atom_effect,
                    ctx,
                    super::super::super::latency::predictive_check_sims(self.latency_mode),
                    ValidationSink { refutations: &mut refutations, diagnostics },
                )?);
            }
            checks
        } else {
            run_envelope_bayesian_full_validation(
                BayesianValidation { suite: self.settings.refute, cfg, estimator: est },
                atoms,
                posterior,
                estimate_ate,
                ctx,
                super::super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
            )?
        };
        Ok((refutations, predictive_checks))
    }

    /// Execute a supplied static graph posterior into a Frequentist effect mixture.
    ///
    /// Same atoms and unidentified-mass rule as
    /// [`Self::execute_graph_posterior_bayesian`]: identified atoms contribute
    /// `linear.adjustment.ate` point estimates, mixed as
    /// `E[τ | identified] = Σ w_i τ_i / identified_mass`. Unidentified mass is
    /// retained in the envelope diagnostic and is not redistributed.
    ///
    /// # Errors
    ///
    /// No identified atoms, or any per-graph identification/estimation failure.
    pub(in crate::analysis::execute) fn execute_graph_posterior_frequentist(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_graph_posterior_frequentist_inner(data, gp, query, physical, None, ctx)
    }

    pub(in crate::analysis) fn execute_checked_graph_posterior_frequentist(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        operation: &super::super::super::CheckedGraphPosteriorEffect,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_graph_posterior_frequentist_inner(
            data,
            operation.posterior(),
            operation.query(),
            physical,
            Some(operation),
            ctx,
        )
    }

    fn execute_graph_posterior_frequentist_inner(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        checked: Option<&super::super::super::CheckedGraphPosteriorEffect>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if checked.is_none() && !self.settings.inference.is_frequentist() {
            return Err(CausalError::unsupported_inference(
                "execute_graph_posterior_frequentist requires inference=Frequentist",
            ));
        }
        let conditional = checked.map_or_else(
            || matches!(self.query, CausalQuery::ConditionalEffect(_)),
            |operation| operation.target().is_conditional(),
        );
        let estimator_id = if let Some(operation) = checked {
            operation.estimator()
        } else if conditional {
            EstimatorId::ConditionalLinearAdjustment
        } else {
            EstimatorId::LinearAdjustmentAte
        };
        let run = EffectRunConfig::resolve(self, checked, estimator_id);
        let (identified, identify_cached) = if let Some(operation) = checked {
            (operation.identification().clone(), true)
        } else if let Some(cache) = self.graph_posterior_cache() {
            (Arc::clone(cache), true)
        } else if let Some(cache) =
            crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
        {
            (cache, false)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
                false,
            )
        };
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            run.latency_mode,
            identified.graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let route =
            GraphPosteriorFrequentistRoute { data, query, run: &run, estimator_id, conditional };
        let atoms =
            Self::estimate_graph_posterior_frequentist_atoms(&route, &identified, &graphs, ctx)?;
        let mix =
            mix_graph_posterior_frequentist_atoms(&identified.graphs, subsample_drop.mass, atoms)?;
        let mut diagnostics = graph_posterior_frequentist_diagnostics(&mix, subsample_notes);
        let (refutations, na_diagnostics) = Self::refute_graph_posterior_frequentist(
            &route,
            matches!(mix.mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms),
            &mix.refute_atoms,
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);
        let GraphPosteriorFrequentistMix { mixed, estimate, identification, estimand, .. } = mix;

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.linear_adjustment",
                    "estimate.linear_adjustment_ate",
                )),
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        ))
    }

    /// Per-atom linear-adjustment estimates for the atoms the Interactive subsample kept,
    /// with each atom's weighted SE item and row-aligned influence function.
    fn estimate_graph_posterior_frequentist_atoms(
        route: &GraphPosteriorFrequentistRoute<'_>,
        identified: &super::super::super::prepared::CachedGraphPosteriorIdentification,
        graphs: &WeightedGraphSamples,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorFrequentistAtoms, CausalError> {
        let GraphPosteriorFrequentistRoute { data, query, run, conditional, .. } = *route;
        let keep = identified_envelope_keys(graphs);

        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let kept: Vec<_> =
            identified.atoms.iter().filter(|atom| keep.contains(&atom.key)).collect();
        let estimates = ctx.map_indexed(kept.len(), |i, inner| {
            let atom = kept[i];
            let mut case_ws = StaticEstimateWorkspaces::default();
            let estimate = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                ConditionalLinearAdjustment::new()
                    .estimate(data, &atom.estimand, &q, inner)
                    .map_err(CausalError::from)
            } else {
                estimate_static_effect(
                    &run.spec,
                    StaticEstimateInput {
                        data,
                        estimand: &atom.estimand,
                        query,
                        bootstrap_replicates: run.bootstrap_replicates,
                        overlap_policy: run.overlap_policy,
                        population_registry: run.population_registry,
                        sampling_design: None,
                    },
                    atom.identification.required_assumptions.clone(),
                    inner,
                    &mut case_ws,
                )
            }?;
            Ok::<_, CausalError>((atom, estimate))
        })?;
        let key_weights = IdentifiedKeyWeights::new(graphs);
        for (atom, estimate) in estimates {
            let w = key_weights.get(atom.key);
            se_items.push((w, estimate.se_analytic()));
            // Every atom is fit on the same rows; embed its IF in the original
            // row universe so the mixture SE carries cross-atom covariance.
            if let Some(inf) = estimate
                .scores
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &atom.estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            if primary_estimand.is_none() {
                primary_estimand = Some(atom.estimand.clone());
                primary_identification = Some(atom.identification.clone());
                assumptions = estimate.provenance.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: atom.key,
                weight: w,
                estimand: atom.estimand.clone(),
                indexer: None,
                original: estimate,
            });
        }
        Ok(GraphPosteriorFrequentistAtoms {
            se_items,
            atom_ifs,
            atom_weights,
            primary_estimand,
            primary_identification,
            assumptions,
            refute_atoms,
        })
    }

    /// Refutation suite over the Frequentist graph-posterior atoms: per atom when the mix is
    /// graph-dependent, the pooled envelope refutation otherwise.
    fn refute_graph_posterior_frequentist(
        route: &GraphPosteriorFrequentistRoute<'_>,
        graph_dependent: bool,
        refute_atoms: &[EnvelopeRefuteAtom],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let GraphPosteriorFrequentistRoute { data, query, run, estimator_id, .. } = *route;
        let estimator = estimator_id.as_str();
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = if graph_dependent {
            let mut reports = Vec::new();
            let mut notes = Vec::new();
            for atom in refute_atoms {
                if atom.weight <= 0.0 {
                    continue;
                }
                let (mut per_atom, mut per_notes) = EnvelopeRefutation {
                    data,
                    query,
                    suite: run.refute,
                    estimator,
                    custom: run.custom_validators,
                    temporal: None,
                }
                .run(std::slice::from_ref(atom), &mut refute_ws, ctx)?;
                reports.append(&mut per_atom);
                notes.append(&mut per_notes);
            }
            notes.retain(|d| d.code.as_ref() != "refute.envelope.effect_mixture");
            notes.push(Diagnostic::new(
                "refute.envelope.class_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "outer scalar mix skipped because StructuralAggregationPolicy is \
                 graph_dependent_atoms — per-atom reports are retained and are not compared \
                 against a pooled mixture ATE",
            ));
            (reports, notes)
        } else {
            EnvelopeRefutation {
                data,
                query,
                suite: run.refute,
                estimator,
                custom: run.custom_validators,
                temporal: None,
            }
            .run(refute_atoms, &mut refute_ws, ctx)?
        };
        Ok((refutations, na_diagnostics))
    }
}
