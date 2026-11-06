//! Panel class sequential effect fitting and validation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGComputationAte, CausalError, CausalPosterior, CausalQuery, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    GraphEffectDraws, GraphIdentFlag, IdentificationBasis, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, InferenceMode, Instant,
    OverlapPolicy, PANEL_INTERVAL_LEVEL, PanelClusters, PanelData, PhysicalExecutionPlan,
    PredictiveCheckReport, SequentialValidation, Strategy, StudyResult, TemporalDag,
    TemporalEffectQuery, TemporalIndexer, UNFOLDED_LINEAR_SEM, bayesian_gcomp, completion_fit_seed,
    elapsed_ns, envelope_to_identification_result_for, identification_status_ok_for_case,
    identify_cached_diagnostic, indexer_span, mag_sequential_unevaluable_diagnostic, nan_effect,
    panel_estimand_assumption, panel_unit_average_bootstrap_note, provenance_ids,
    refuse_short_panel_units, same_fitted_mechanisms, select_estimand,
    sequential_linear_sem_assumption, unit_average_estimand_diagnostic,
};

impl super::super::Study {
    /// Disclosures that open a panel class sequential effect.
    fn panel_class_sequential_disclosures(
        &self,
        envelope: &antecedent_identify::IdentificationEnvelope<
            antecedent_identify::TemporalCompletionGraph,
        >,
        panel_units: usize,
        bayesian_without_class_prior: bool,
    ) -> Vec<Diagnostic> {
        let mut diagnostics =
            vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            )];
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_effect.panel.estimand",
            panel_units,
        ));
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.sustained_window",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each completion's sequential contrast propagates through all intervened times and \
             is fit per panel unit, then averaged with equal unit weight; bidirected MAG \
             completions stay unevaluable",
        ));
        if bayesian_without_class_prior {
            diagnostics.push(super::super::temporal_path::enumeration_not_probability_diagnostic());
        }
        diagnostics
    }

    /// Fits each directed completion on every unit's own series, sharing seeds between
    /// completions whose fitted mechanisms coincide.
    fn fit_panel_sequential_completions(
        run: &PanelSequentialRun<'_>,
        envelope: &antecedent_identify::TemporalClassEnvelope,
        class_masses: Option<&[f64]>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<PanelSequentialFits, CausalError> {
        let mut fits = PanelSequentialFits {
            weights: Vec::new(),
            flags: Vec::new(),
            keys: Vec::new(),
            atom_values: vec![None; envelope.envelope.cases.len()],
            per_graph: Vec::new(),
            atoms: Vec::new(),
            primary_estimand: None,
            unevaluable_weight: 0.0,
        };
        for (i, (case, indexer)) in
            envelope.envelope.cases.iter().zip(envelope.indexers.iter()).enumerate()
        {
            let key = case.graph.fingerprint();
            fits.keys.push(key);
            let weight = class_masses.map_or(case.weight.0, |masses| masses[i]);
            fits.weights.push(weight);
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                fits.flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            let Some(dag) = case.graph.sequential_dag() else {
                fits.flags.push(GraphIdentFlag::Unidentified);
                fits.unevaluable_weight += weight;
                diagnostics.push(mag_sequential_unevaluable_diagnostic(key));
                continue;
            };
            let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
                .or_else(|_| {
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
                })?;
            fits.flags.push(GraphIdentFlag::Identified);
            let mut assumptions = case.result.required_assumptions.clone();
            assumptions.push(sequential_linear_sem_assumption(UNFOLDED_LINEAR_SEM));
            // Each completion's unit posteriors draw from that completion's seed; a
            // completion whose fitted unit mechanisms coincide with an earlier one's
            // is refit on that completion's seed, so one model carries one posterior.
            let mut seed = completion_fit_seed(run.ctx, key);
            let atom = loop {
                let atom = fit_panel_sequential_atom(
                    run.panel,
                    PanelAtomProblem { dag: &dag, indexer, estimand: &estimand, query: run.query },
                    IdentificationBasis { status: case.result.status, assumptions: &assumptions },
                    run.bayes,
                    seed,
                    run.ctx,
                )?;
                let shared = fits
                    .atoms
                    .iter()
                    .find(|(_, _, earlier)| {
                        run.bayes.is_some()
                            && earlier.seed != seed
                            && earlier.same_mechanisms(&atom)
                    })
                    .map(|(_, _, earlier)| earlier.seed);
                match shared {
                    Some(shared) => seed = shared,
                    None => break atom,
                }
            };
            fits.atom_values[i] = Some(atom.mean());
            if let Some(draws) = atom.mean_draws() {
                fits.per_graph
                    .push(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws) });
            }
            fits.primary_estimand.get_or_insert(estimand);
            fits.atoms.push((weight, key, atom));
        }
        Ok(fits)
    }

    /// Mixes the fitted completions into the panel effect: class posterior, frozen-weight
    /// unit mixture with its between-unit SE, or an unevaluable NaN effect.
    fn mix_panel_sequential_estimate(
        run: &PanelSequentialRun<'_>,
        fits: &mut PanelSequentialFits,
        class_masses_supplied: bool,
        truncated_completions: usize,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (EffectEstimate, Option<CausalPosterior>, crate::result::StructuralWeightBasis),
        CausalError,
    > {
        let total_weight: f64 = fits.weights.iter().sum();
        let unevaluable_share =
            if total_weight > 0.0 { fits.unevaluable_weight / total_weight } else { 0.0 };
        let evaluable_weight: f64 = fits.atoms.iter().map(|(weight, ..)| *weight).sum();
        let (estimate, posterior, weight_basis) = if run.bayes.is_some() {
            let (estimate, mut posterior, basis) =
                super::super::temporal_path::mix_class_effect_posteriors(
                    super::super::temporal_path::ClassPosteriorMix {
                        class_prior_supplied: class_masses_supplied,
                        truncated_completions,
                        evaluable_mass_positive: evaluable_weight > 0.0,
                        weights: std::mem::take(&mut fits.weights),
                        flags: std::mem::take(&mut fits.flags),
                        keys: std::mem::take(&mut fits.keys),
                        per_graph: &fits.per_graph,
                        conflict: None,
                        label: "panel_temporal_class_sequential",
                    },
                    std::iter::empty(),
                    diagnostics,
                )?;
            // Unevaluable (bidirected) completions are flagged unidentified for the
            // mixture, which never draws from them; their share is unevaluable mass.
            if let Some(mixed) = posterior.as_mut() {
                mixed.unidentified_mass = (mixed.unidentified_mass - unevaluable_share).max(0.0);
            }
            (estimate, posterior, basis)
        } else if evaluable_weight > 0.0 {
            // Frozen completion weights: each unit's mixture contrast, averaged over
            // units, with the between-unit SE of that per-unit mixture (it carries the
            // completions' covariance through the shared units).
            let units = run.panel.unit_count();
            let unit_mixture: Vec<f64> = (0..units)
                .map(|u| {
                    fits.atoms
                        .iter()
                        .map(|(w, _, atom)| w / evaluable_weight * atom.unit_ates[u])
                        .sum()
                })
                .collect();
            let ate = unit_mixture.iter().sum::<f64>() / units as f64;
            let se = antecedent_stats::sample_std(&unit_mixture) / (units as f64).sqrt()
                * antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL);
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_effect.panel.between_unit_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "SE is sd/sqrt(N) of the {} completions' frozen-weight mixture contrast over \
                     N={units} units, with N-1 degrees of freedom, scaled by t_(N-1)/z = {:.4} \
                     so estimate ± 1.96·SE is the 95% t_(N-1) interval",
                    fits.atoms.len(),
                    antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL),
                ),
            ));
            let mut assumptions = fits.atoms[0].2.assumptions.clone();
            assumptions.push(panel_estimand_assumption(PanelClusters::UnitAverage, units));
            (
                EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride),
                None,
                crate::result::StructuralWeightBasis::CompletionEnumeration,
            )
        } else {
            (nan_effect(), None, crate::result::StructuralWeightBasis::CompletionEnumeration)
        };
        Ok((estimate, posterior, weight_basis))
    }

    /// Panel class multi-step Sustained: every directed completion's sequential
    /// contrast is fit on each unit's own series and averaged with equal unit
    /// weight. Frequentist completions mix by completion mass, with the between-unit
    /// SE of the per-unit mixture; Bayesian completions follow the series class-prior
    /// contract. Bidirected completions are unevaluable and reported as such.
    pub(super) fn execute_panel_class_sequential(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.panel_class_bundle(physical, query, ctx)?;
        let envelope = &bundle.envelope.envelope;
        let class_masses = self
            .declared
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        // The panel support refuses caller priors here; the per-unit fits use the
        // default prior.
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let mut diagnostics = self.panel_class_sequential_disclosures(
            envelope,
            panel.unit_count(),
            bayes.is_some() && class_masses.is_none(),
        );
        let run = PanelSequentialRun { panel, query, bayes: bayes.as_ref(), ctx };
        let mut fits = Self::fit_panel_sequential_completions(
            &run,
            &bundle.envelope,
            class_masses.as_deref(),
            &mut diagnostics,
        )?;
        if fits.atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "panel class-aware multi-step envelope had no evaluable directed \
                          completions"
                    .into(),
            });
        }
        let estimand = fits.primary_estimand.take().ok_or_else(|| CausalError::Compile {
            message: "panel class-aware sequential envelope missing estimand".into(),
        })?;
        let (estimate, posterior, weight_basis) = Self::mix_panel_sequential_estimate(
            &run,
            &mut fits,
            class_masses.is_some(),
            envelope.truncated_completions,
            &mut diagnostics,
        )?;
        let PanelSequentialFits { atoms, atom_values, unevaluable_weight, .. } = fits;
        let (refutations, validation_diagnostics, predictive_checks) =
            self.panel_sequential_validation(panel, query, &atoms, bayes.as_ref(), ctx)?;
        diagnostics.extend(validation_diagnostics);
        if self.settings.bootstrap_replicates > 0 {
            diagnostics
                .extend(panel_unit_average_bootstrap_note(self.settings.bootstrap_replicates));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let structural = panel_sequential_structural_mixture(
            envelope,
            weight_basis,
            class_masses.as_deref(),
            &atom_values,
            unevaluable_weight,
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, EstimatorId::TemporalSequentialGcomp),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: (*bundle.envelope).clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                posterior,
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal.sequential.gcomp.panel",
                    "estimate.temporal_class.envelope",
                )),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Sequential validation of every completion on every unit's own series (the
    /// series validator; the units are fit separately), each report prefixed with
    /// its unit and completion.
    fn panel_sequential_validation(
        &self,
        panel: &PanelData,
        query: &TemporalEffectQuery,
        atoms: &[(f64, u64, PanelSequentialAtom)],
        bayes: Option<&BayesianGComputationAte>,
        ctx: &ExecutionContext,
    ) -> Result<PanelSequentialValidation, CausalError> {
        let mut reports = Vec::new();
        let mut diagnostics = Vec::new();
        let mut checks = Vec::new();
        for (unit_index, unit) in panel.units().iter().enumerate() {
            for (weight, key, atom) in atoms {
                let fit = &atom.units[unit_index];
                let validation_atom =
                    super::super::sequential_validation::SequentialValidationAtom {
                        weight: *weight,
                        graph: Arc::new(atom.graph.clone()),
                        _graph_lease: None,
                        indexer: atom.indexer.clone(),
                        estimand: atom.estimand.clone(),
                        status: atom.status,
                        estimate: fit.estimate.clone(),
                        mechanisms: fit.mechanisms.clone(),
                    };
                let mut posterior = fit.posterior.clone();
                let unit_bayes = bayes.map(|estimator| BayesianGComputationAte {
                    seed: fit.seed,
                    ..estimator.clone()
                });
                let (mut local, local_diagnostics, local_checks) =
                    super::super::sequential_validation::validate_sequential(
                        SequentialValidation {
                            data: &unit.series,
                            query,
                            suite: self.settings.refute,
                            custom: &self.declared.custom_validators,
                            predictive_sims: super::super::super::latency::predictive_check_sims(
                                self.latency_mode,
                            ),
                            gaps: false,
                        },
                        std::slice::from_ref(&validation_atom),
                        unit_bayes.as_ref(),
                        posterior.as_mut(),
                        fit.estimate.ate,
                        ctx,
                    )?;
                for report in &mut local {
                    report.refuter = Arc::from(format!(
                        "unit.{}.completion.{key}.{}",
                        unit.unit_id, report.refuter
                    ));
                }
                reports.extend(local);
                diagnostics.extend(local_diagnostics);
                checks.extend(local_checks);
            }
        }
        Ok((reports, diagnostics, checks))
    }
}

/// Refutation reports, diagnostics and predictive checks of a panel sequential run.
type PanelSequentialValidation =
    (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>, Vec<PredictiveCheckReport>);

/// One unit's sequential fit of one completion.
struct PanelSequentialUnitFit {
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    mechanisms: Vec<antecedent_estimate::temporal_sequential::SequentialBayesianMechanism>,
    seed: u64,
}

/// One directed completion's sequential contrast fit on every panel unit.
struct PanelSequentialAtom {
    graph: TemporalDag,
    indexer: TemporalIndexer,
    estimand: IdentifiedEstimand,
    status: IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    /// Seed the completion's unit seeds derive from.
    seed: u64,
    units: Vec<PanelSequentialUnitFit>,
    unit_ates: Vec<f64>,
}

impl PanelSequentialAtom {
    /// Equal-weight mean of the unit contrasts.
    fn mean(&self) -> f64 {
        self.unit_ates.iter().sum::<f64>() / self.unit_ates.len() as f64
    }

    /// Draws of the equal-weight unit mean: the units' independent posteriors
    /// paired by draw index (their product posterior), when every unit has draws.
    fn mean_draws(&self) -> Option<Vec<f64>> {
        let columns: Option<Vec<&[f64]>> = self
            .units
            .iter()
            .map(|fit| {
                let posterior = fit.posterior.as_ref()?;
                posterior.draws.column(posterior.effect_column()?).ok()
            })
            .collect();
        let columns = columns?;
        let n = columns.iter().map(|c| c.len()).min()?;
        (n > 0).then(|| {
            (0..n)
                .map(|k| columns.iter().map(|c| c[k]).sum::<f64>() / columns.len() as f64)
                .collect()
        })
    }

    /// Whether every unit fitted the same mechanisms as in `other`.
    fn same_mechanisms(&self, other: &Self) -> bool {
        self.units.len() == other.units.len()
            && self
                .units
                .iter()
                .zip(&other.units)
                .all(|(a, b)| same_fitted_mechanisms(&a.mechanisms, &b.mechanisms))
    }
}

/// The retained DBN, lag indexer, estimand and query one panel sequential atom fits.
#[derive(Clone, Copy)]
struct PanelAtomProblem<'a> {
    dag: &'a TemporalDag,
    indexer: &'a TemporalIndexer,
    estimand: &'a IdentifiedEstimand,
    query: &'a TemporalEffectQuery,
}

/// Fit one completion's sequential contrast on every unit's own series.
///
/// A unit that yields a non-finite contrast refuses the completion: the survivor
/// mean is not the panel effect. Under Bayesian inference each unit draws from its
/// own stream of `seed`.
fn fit_panel_sequential_atom(
    panel: &PanelData,
    atom: PanelAtomProblem<'_>,
    basis: IdentificationBasis<'_>,
    bayes: Option<&BayesianGComputationAte>,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<PanelSequentialAtom, CausalError> {
    let PanelAtomProblem { dag, indexer, estimand, query } = atom;
    let IdentificationBasis { status, assumptions } = basis;
    refuse_short_panel_units(panel, indexer_span(indexer))?;
    let mut units = Vec::with_capacity(panel.unit_count());
    let mut unit_ates = Vec::with_capacity(panel.unit_count());
    let mut last_assumptions = assumptions.clone();
    for (index, unit) in panel.units().iter().enumerate() {
        let unit_seed = completion_fit_seed(
            ctx,
            seed ^ (index as u64 + 1).wrapping_mul(antecedent_core::GOLDEN_GAMMA),
        );
        let unit_bayes =
            bayes.map(|estimator| BayesianGComputationAte { seed: unit_seed, ..estimator.clone() });
        let mut mechanisms = Vec::new();
        let (estimate, posterior) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                &antecedent_estimate::SequentialProblem {
                    data: &unit.series,
                    graph: dag,
                    indexer,
                    estimand,
                },
                query,
                antecedent_estimate::SequentialInference {
                    status,
                    assumptions: assumptions.clone(),
                    bootstrap_replicates: 0,
                    bayesian: unit_bayes.as_ref(),
                },
                ctx,
                Some(&mut mechanisms),
            )
            .map_err(CausalError::from)?;
        if !estimate.ate.is_finite() {
            return Err(CausalError::Compile {
                message: format!(
                    "panel sequential completion has no finite contrast on unit {}; the \
                     survivor mean is not the panel effect",
                    unit.unit_id
                ),
            });
        }
        unit_ates.push(estimate.ate);
        last_assumptions = estimate.provenance.assumptions.clone();
        units.push(PanelSequentialUnitFit { estimate, posterior, mechanisms, seed: unit_seed });
    }
    Ok(PanelSequentialAtom {
        graph: dag.clone(),
        indexer: indexer.clone(),
        estimand: estimand.clone(),
        status,
        assumptions: last_assumptions,
        seed,
        units,
        unit_ates,
    })
}

/// Borrowed inputs of a panel class sequential fit.
#[derive(Clone, Copy)]
struct PanelSequentialRun<'a> {
    panel: &'a PanelData,
    query: &'a TemporalEffectQuery,
    bayes: Option<&'a BayesianGComputationAte>,
    ctx: &'a ExecutionContext,
}

/// Directed completions of a panel class sequential effect, fitted on every unit.
struct PanelSequentialFits {
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    atom_values: Vec<Option<f64>>,
    per_graph: Vec<GraphEffectDraws>,
    atoms: Vec<(f64, u64, PanelSequentialAtom)>,
    primary_estimand: Option<IdentifiedEstimand>,
    unevaluable_weight: f64,
}

/// Structural mixture of a panel class sequential effect, with the unevaluable share split out.
fn panel_sequential_structural_mixture(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
    weight_basis: crate::result::StructuralWeightBasis,
    class_masses: Option<&[f64]>,
    atom_values: &[Option<f64>],
    unevaluable_weight: f64,
) -> crate::result::StructuralResponseMixture {
    let mut structural = super::super::temporal_path::temporal_class_structural_mixture(
        envelope,
        weight_basis,
        class_masses,
        atom_values,
    );
    let total: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
    if total > 0.0 {
        structural.unevaluable_mass = unevaluable_weight / total;
        structural.unidentified_mass =
            (structural.unidentified_mass - structural.unevaluable_mass).max(0.0);
    }
    structural
}
