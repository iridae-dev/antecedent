// SPDX-License-Identifier: MIT OR Apache-2.0
// Frequentist DBN-posterior temporal mediation.

use super::temporal_posterior_response::dbn_posterior_structural_atoms;
use super::{
    Arc, BootstrapAccounting, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, GraphPosterior, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, Instant, MediationRoute, OverlapPolicy,
    PhysicalExecutionPlan, PosteriorMassSummary, QueryRefutationPlan, RefuteSuite, ResponseValue,
    Strategy, StudyResult, TableView, TemporalIndexer, TemporalMediationEstimate,
    TemporalMediationEstimator, TimeSeriesData, VariableId, elapsed_ns, identify_cached_diagnostic,
    mediation_block_stream, provenance_ids, push_graph_posterior_structural_aggregation_diagnostic,
    require_identified, resolve_posterior_status, resolve_structural_aggregation,
    short_series_warning,
};

/// The first evaluable atom: its estimand and certificate head the result.
struct PrimaryAtom {
    estimand: IdentifiedEstimand,
    identification: IdentificationResult,
    indexer: TemporalIndexer,
    adjustment_keys: Arc<[antecedent_core::TemporalNodeKey]>,
    lagged: Arc<[antecedent_data::LaggedColumn]>,
}

/// The evaluable atoms of a DBN-posterior mediation: their prepared designs
/// and frozen weights, the refutation inputs, and what the aggregation policy
/// and mass split are resolved from.
struct DbnMediationFits<'a> {
    designs: Vec<antecedent_estimate::PreparedTemporalMediation>,
    weights: Vec<f64>,
    refute_atoms: Vec<(u64, f64, IdentifiedEstimand, Arc<[antecedent_data::LaggedColumn]>)>,
    contributing_estimands: Vec<&'a IdentifiedEstimand>,
    any_partial: bool,
    failed_mass: f64,
    primary: Option<PrimaryAtom>,
    distinct_sets: bool,
}

/// The weighted point estimate and shared circular-block uncertainty of the
/// evaluable atoms.
struct DbnMediationAggregate {
    point: f64,
    estimate: EffectEstimate,
    mediation: TemporalMediationEstimate,
    shared: Option<antecedent_estimate::SharedMediationBlockSe>,
    requested_se: Option<f64>,
    replicates_requested: bool,
    replicates_ok: u32,
    family: antecedent_estimate::CircularBlockFamily,
}

/// Identified, unidentified and unevaluable shares of the posterior mass.
#[derive(Clone, Copy)]
struct DbnMediationMasses {
    identified: f64,
    unidentified: f64,
    failed: f64,
}

/// Everything the DBN-posterior mediation result is finished from.
struct DbnMediationParts {
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    adjustment: Arc<[antecedent_core::TemporalNodeKey]>,
    aggregate: DbnMediationAggregate,
    diagnostics: Vec<Diagnostic>,
    refutations: Vec<antecedent_validate::RefutationReport>,
    atoms: Vec<crate::result::StructuralResponseAtom>,
    masses: DbnMediationMasses,
    atom_points: Vec<f64>,
    horizon: u32,
    identify_cached: bool,
}

impl super::Study {
    /// Frequentist DBN-posterior temporal mediation at one horizon.
    ///
    /// Each identified atom uses its own horizon-specific mediation set `S(h)`
    /// (its `I(h)` plus mediator/outcome confounders) and is prepared once on
    /// the original series. With frozen posterior weights, Total, Direct and
    /// Mediated are fixed-weight means over the evaluable atoms. When replicates
    /// are requested, one shared circular-block replicate of lag-aligned series
    /// times refits every atom's three mechanism regressions and mixes them
    /// inside the replicate, so the three SEs describe the reported aggregate and
    /// include between-atom sampling covariance. Unidentified mass is retained;
    /// failed estimation is unevaluable, not unidentified. Cheap/full run
    /// mediation refuters per contributing atom against that atom's own
    /// contrast (not the pooled mean) and mix by frozen graph weight; the mixed
    /// check passes only if every atom passes.
    pub(super) fn execute_dbn_posterior_mediation_frequentist(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if query.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                          multi-horizon grids need their own joint uncertainty contract",
            });
        }
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_ref() {
                (Arc::clone(cache), true)
            } else {
                (
                Arc::new(
                    crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                        gp, &vars, query, ctx,
                    )?,
                ),
                false,
            )
            };
        let horizon = query.horizons[0];
        let identified = identified.mediation_horizon(horizon)?;
        let mut atoms = dbn_posterior_structural_atoms(&identified);
        let est = TemporalMediationEstimator::new()
            .with_allow_natural_controlled_alias(true)
            .with_allow_gaps(self.declares_complete_case());
        let DbnMediationFits {
            designs,
            weights,
            refute_atoms,
            contributing_estimands,
            any_partial,
            failed_mass,
            primary,
            distinct_sets,
        } = fit_dbn_mediation_atoms(&est, (data, query, horizon), &identified, &mut atoms, ctx)?;
        let PrimaryAtom {
            estimand, mut identification, indexer, adjustment_keys: adjustment, ..
        } = primary.ok_or_else(|| CausalError::Compile {
            message: "Frequentist DBN-posterior mediation has no estimable identified atom".into(),
        })?;
        let total_mass = identified.graphs.total_weight();
        let identified_mass: f64 = weights.iter().sum();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let incomplete =
            dbn_mediation_graph_dependent(identification.status, unidentified_mass, failed_mass);
        if incomplete {
            identification.status = IdentificationStatus::GraphDependent;
        }
        let aggregate = self.aggregate_dbn_mediation(
            (&designs, &weights, identified_mass),
            &identification,
            query,
            horizon,
            ctx,
        );
        let masses = DbnMediationMasses {
            identified: identified_mass / total_mass,
            unidentified: unidentified_mass / total_mass,
            failed: failed_mass / total_mass,
        };
        let mut diagnostics = dbn_mediation_diagnostics(
            identification.diagnostics.clone(),
            (&identified, &aggregate, masses),
            (resolve_structural_aggregation(&contributing_estimands, any_partial), distinct_sets),
            (designs.len(), identify_cached),
        );
        let mut refutations = Vec::new();
        if self.refute != RefuteSuite::None {
            refutations = self.refute_dbn_mediation_atoms(
                (data, query),
                &refute_atoms,
                &designs,
                &mut diagnostics,
                ctx,
            )?;
        }
        let atom_points: Vec<f64> = designs.iter().map(|d| d.estimate().effect.ate).collect();
        Ok(self.finish_dbn_mediation(
            physical,
            query,
            DbnMediationParts {
                identification,
                estimand,
                indexer,
                adjustment,
                aggregate,
                diagnostics,
                refutations,
                atoms,
                masses,
                atom_points,
                horizon,
                identify_cached,
            },
            started,
            ctx,
        ))
    }

    /// The fixed-weight mean of the atoms' point, Total, Direct and Mediated
    /// effects, with the shared circular-block bootstrap of the aggregate.
    fn aggregate_dbn_mediation(
        &self,
        (designs, weights, identified_mass): (
            &[antecedent_estimate::PreparedTemporalMediation],
            &[f64],
            f64,
        ),
        identification: &IdentificationResult,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
        ctx: &ExecutionContext,
    ) -> DbnMediationAggregate {
        let mix = |pick: fn(&TemporalMediationEstimate) -> Option<f64>| -> Option<f64> {
            let mut acc = 0.0;
            for (design, weight) in designs.iter().zip(weights) {
                acc += weight * pick(design.estimate())?;
            }
            Some(acc / identified_mass)
        };
        let point = mix(|m| Some(m.effect.ate)).unwrap_or(f64::NAN);
        let refs: Vec<&antecedent_estimate::PreparedTemporalMediation> = designs.iter().collect();
        let shared = antecedent_estimate::shared_mediation_block_bootstrap(
            &refs,
            weights,
            query.contrast,
            self.bootstrap_replicates,
            mediation_block_stream(MediationRoute::Dbn, horizon),
            ctx,
        );
        let requested_se = shared.as_ref().and_then(|s| s.requested).filter(|s| s.is_finite());
        let replicates_requested = self.bootstrap_replicates > 0;
        let (replicates_ok, replicates_attempted) = shared
            .as_ref()
            .map_or((0, 0), |s| (s.block.replicates_ok, s.block.replicates_attempted));
        let mut assumptions = identification.required_assumptions.clone();
        for record in &designs[0].estimate().effect.provenance.assumptions.entries {
            assumptions.push(record.clone());
        }
        let family = shared.as_ref().map_or_else(
            || antecedent_estimate::CircularBlockFamily::for_mediation_atoms(designs.len()),
            |s| s.family,
        );
        let estimate = EffectEstimate::from_parts(
            point,
            f64::NAN,
            BootstrapAccounting {
                se: requested_se,
                replicates_ok: replicates_requested.then_some(replicates_ok),
                replicates_failed: replicates_requested
                    .then_some(replicates_attempted.saturating_sub(replicates_ok)),
                cancelled: ctx.cancellation.is_cancelled(),
                early_stopped: false,
            },
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        )
        .with_block_family(family);
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: mix(|m| m.total),
            direct: mix(|m| m.direct),
            mediated: mix(|m| m.mediated),
        };
        DbnMediationAggregate {
            point,
            estimate,
            mediation,
            shared,
            requested_se,
            replicates_requested,
            replicates_ok,
            family,
        }
    }

    /// Runs the mediation refuters on each contributing atom against its own
    /// contrast and mixes the reports by frozen graph weight.
    fn refute_dbn_mediation_atoms(
        &self,
        (data, query): (&TimeSeriesData, &antecedent_core::MediationQuery),
        refute_atoms: &[(u64, f64, IdentifiedEstimand, Arc<[antecedent_data::LaggedColumn]>)],
        designs: &[antecedent_estimate::PreparedTemporalMediation],
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
        let plan = QueryRefutationPlan::temporal_mediation(self.refute == RefuteSuite::Full);
        let mut per_atom = Vec::with_capacity(refute_atoms.len());
        for ((_, weight, estimand, lagged), design) in refute_atoms.iter().zip(designs) {
            let reports = plan
                .refute_temporal_atom(data, estimand, query, design.estimate(), lagged, ctx)
                .map_err(CausalError::from)?;
            per_atom.push((*weight, reports));
        }
        let refutations = QueryRefutationPlan::mix_weighted(per_atom);
        let atom_keys: String = refute_atoms
            .iter()
            .map(|(key, _, _, _)| format!("{key:x}"))
            .collect::<Vec<_>>()
            .join(",");
        diagnostics.push(Diagnostic::new(
            "refute.envelope.effect_mixture",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "mediation refuters evaluated each contributing graph atom [{atom_keys}] \
                 against that atom's own contrast using that atom's S(h), not the pooled \
                 mixture; reports mix by fixed graph weight and pass only if every \
                 contributing atom passes"
            ),
        ));
        Ok(refutations)
    }

    /// The identified-set mixture, mediation grid and identified-execute
    /// finish of a DBN-posterior mediation.
    fn finish_dbn_mediation(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &antecedent_core::MediationQuery,
        parts: DbnMediationParts,
        started: Instant,
        ctx: &ExecutionContext,
    ) -> StudyResult {
        let DbnMediationParts {
            identification,
            estimand,
            indexer,
            adjustment,
            aggregate,
            diagnostics,
            refutations,
            atoms,
            masses,
            atom_points,
            horizon,
            identify_cached,
        } = parts;
        let DbnMediationAggregate {
            point,
            estimate,
            mediation,
            shared,
            replicates_requested,
            replicates_ok,
            ..
        } = aggregate;
        let uncertainty = match shared {
            Some(s) if replicates_requested => {
                antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: s.requested,
                    block: s.block,
                }
            }
            _ => antecedent_estimate::TemporalMediationUncertainty::Unavailable,
        };
        let structural_response =
            dbn_mediation_structural_mixture(atoms, masses, (horizon, point), &atom_points);
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from([antecedent_estimate::TemporalMediationSlice {
                horizon,
                identification_status: identification.status,
                method: Arc::clone(&estimand.method),
                adjustment,
                estimate: mediation.clone(),
                uncertainty,
                identified_set: None,
                diagnostics: diagnostics.clone(),
            }]),
            joint_posterior: false,
        };
        let bootstrap_ok = replicates_requested.then_some(replicates_ok);
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand,
                estimate,
                Strategy::new(IdentifierId::Frontdoor, EstimatorId::TemporalMediation),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(Some(mediation))
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.dbn_posterior.mediation.frequentist",
                    "estimate.temporal_mediation",
                )),
                structural_response: Some(structural_response),
                mediation_grid: Some(mediation_grid),
                diagnostics: Some(diagnostics),
                certificate: Some(crate::Identification::Point {
                    result: identification,
                    temporal_indexer: Some(indexer),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                ..Default::default()
            }),
        )
    }
}

/// Prepares each positive-weight atom's mediation design on the original
/// series on the context's workers and collects the evaluable ones; atoms that
/// are identified but cannot be estimated add to the unevaluable mass.
fn fit_dbn_mediation_atoms<'a>(
    est: &TemporalMediationEstimator,
    (data, query, horizon): (&TimeSeriesData, &antecedent_core::MediationQuery, u32),
    identified: &'a crate::analysis::prepared::CachedDbnPosteriorIdentification,
    atoms: &mut [crate::result::StructuralResponseAtom],
    ctx: &ExecutionContext,
) -> Result<DbnMediationFits<'a>, CausalError> {
    let mut fits = DbnMediationFits {
        designs: Vec::new(),
        weights: Vec::new(),
        refute_atoms: Vec::new(),
        contributing_estimands: Vec::new(),
        any_partial: false,
        failed_mass: 0.0,
        primary: None,
        distinct_sets: false,
    };
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    let mapped = ctx.map_indexed(identified.atoms.len(), |i, inner| {
        let atom = &identified.atoms[i];
        let weight = key_weights.get(atom.key);
        if weight <= 0.0 {
            return Ok(None);
        }
        let Some(entry) = atom.horizons.as_ref().and_then(|h| h.get(horizon)) else {
            return Ok(Some((atom, weight, None, None)));
        };
        let lagged = super::temporal_path::lagged_adjustment_from_entry(entry);
        let prepared = require_identified(&entry.identification)
            .ok()
            .and_then(|()| est.prepare_shared(data, &entry.estimand, query, &lagged, inner).ok())
            .filter(|prepared| prepared.estimate().effect.ate.is_finite());
        Ok::<_, CausalError>(Some((atom, weight, Some(entry), prepared.map(|p| (lagged, p)))))
    })?;
    for mapped_atom in mapped {
        let Some((atom, weight, entry, prepared)) = mapped_atom else {
            continue;
        };
        let Some(entry) = entry else {
            fits.failed_mass += weight;
            continue;
        };
        if let Some(slot) = atoms.iter_mut().find(|candidate| candidate.graph_key == atom.key) {
            slot.status = entry.identification.status;
        }
        let Some((lagged, prepared)) = prepared else {
            // Identified but not evaluable: the atom keeps its status and
            // has no value, so its mass is unevaluable.
            fits.failed_mass += weight;
            continue;
        };
        if let Some(slot) = atoms.iter_mut().find(|candidate| candidate.graph_key == atom.key) {
            slot.value = Some(ResponseValue::Scalar(prepared.estimate().effect.ate));
        }
        let lagged_for_refute = Arc::clone(&lagged);
        match fits.primary.as_ref() {
            None => {
                let mut keys = entry
                    .estimand
                    .adjustment_set
                    .iter()
                    .filter_map(|id| entry.indexer.key_of(id.raw()).ok())
                    .collect::<Vec<_>>();
                keys.sort();
                fits.primary = Some(PrimaryAtom {
                    estimand: entry.estimand.clone(),
                    identification: entry.identification.clone(),
                    indexer: entry.indexer.clone(),
                    adjustment_keys: Arc::from(keys),
                    lagged,
                });
            }
            Some(first) if first.lagged.as_ref() != lagged.as_ref() => fits.distinct_sets = true,
            Some(_) => {}
        }
        fits.refute_atoms.push((atom.key, weight, entry.estimand.clone(), lagged_for_refute));
        fits.designs.push(prepared);
        fits.weights.push(weight);
        fits.contributing_estimands.push(&entry.estimand);
        fits.any_partial |=
            entry.identification.status == IdentificationStatus::PartiallyIdentified;
    }
    Ok(fits)
}

/// The diagnostics of a DBN-posterior mediation, in publication order: the
/// mass split, atom demotion, structural aggregation, distinct adjustment
/// sets, shared-block uncertainty and cache provenance.
fn dbn_mediation_diagnostics(
    mut diagnostics: Vec<Diagnostic>,
    (identified, aggregate, masses): (
        &crate::analysis::prepared::CachedDbnPosteriorIdentification,
        &DbnMediationAggregate,
        DbnMediationMasses,
    ),
    (aggregation_policy, distinct_sets): (crate::result::StructuralAggregationPolicy, bool),
    (atom_count, identify_cached): (usize, bool),
) -> Vec<Diagnostic> {
    diagnostics.push(
        Diagnostic::new(
            "estimate.dbn_posterior.mediation.frequentist",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "frozen posterior weights; identified_mass={}; unidentified_mass={}; \
             unevaluable_mass={}; each atom uses its own S(h) (its I(h) plus \
             mediator-outcome confounders); Total, Direct and Mediated are fixed-weight \
             means over evaluable atoms; failed estimation is not mixed into unidentified \
             mass; the mass split is in structural_response",
                masses.identified, masses.unidentified, masses.failed
            ),
        )
        .with_fields(super::mass_fields(Some(masses.identified), masses.unidentified)),
    );
    diagnostics.push(Diagnostic::new(
        "estimate.dbn_posterior.atom_demotion",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        identified.identify_demotion.summary(0, 0, 0),
    ));
    push_graph_posterior_structural_aggregation_diagnostic(
        &mut diagnostics,
        aggregation_policy,
        masses.identified,
        masses.unidentified,
        masses.failed,
        0.0,
    );
    if distinct_sets {
        diagnostics.push(Diagnostic::new(
            "identify.dbn_posterior.atom_horizon_sets_differ",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "contributing DBN atoms have distinct S(h) adjustment sets at the published \
             horizon; sets are not unioned across atoms",
        ));
    }
    diagnostics.extend(dbn_mediation_band_diagnostics(aggregate, atom_count));
    if identify_cached {
        diagnostics.push(identify_cached_diagnostic());
    }
    diagnostics
}

/// Whether the unidentified or unevaluable posterior mass makes the aggregate
/// graph dependent.
fn dbn_mediation_graph_dependent(
    status: IdentificationStatus,
    unidentified_mass: f64,
    failed_mass: f64,
) -> bool {
    resolve_posterior_status(
        status,
        crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
        PosteriorMassSummary {
            unidentified: unidentified_mass,
            failed: failed_mass,
            ..PosteriorMassSummary::default()
        },
    ) == IdentificationStatus::GraphDependent
}

/// The posterior-probability mixture of the atoms, with the range of their
/// points as the horizon's identified set.
fn dbn_mediation_structural_mixture(
    atoms: Vec<crate::result::StructuralResponseAtom>,
    masses: DbnMediationMasses,
    (horizon, point): (u32, f64),
    atom_points: &[f64],
) -> crate::result::StructuralResponseMixture {
    let lower = atom_points.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = atom_points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass: masses.identified,
        unidentified_mass: masses.unidentified,
        unevaluable_mass: masses.failed,
        subsampled_out_mass: 0.0,
        identified_set: Some(antecedent_core::ResponseEnvelope {
            grid: Arc::from([f64::from(horizon)]),
            dimension: 1,
            lower: Arc::from([lower]),
            upper: Arc::from([upper]),
        }),
        identified_set_interval: None,
        conditional_on_identified: Some(ResponseValue::Scalar(point)),
        full_mass_scope: true,
        truncated_atoms: 0,
    }
}

/// The shared-block uncertainty diagnostic (or the withheld-uncertainty
/// warning) of a DBN-posterior mediation, with its short-series warning.
fn dbn_mediation_band_diagnostics(
    aggregate: &DbnMediationAggregate,
    atom_count: usize,
) -> Vec<Diagnostic> {
    let DbnMediationAggregate { shared, requested_se, replicates_requested, family, .. } =
        aggregate;
    let (requested_se, replicates_requested, family) =
        (*requested_se, *replicates_requested, *family);
    let mut diagnostics = Vec::new();
    match shared.as_ref() {
        Some(s) if replicates_requested && requested_se.is_some() => {
            diagnostics.push(Diagnostic::new(
                "estimate.dbn_posterior.mediation.shared_block",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "iid analytic SEs are NaN for Total, Direct and Mediated; shared \
                     circular-block replicates={}; attempted={}; blocks of {} consecutive \
                     series times (dependence-aware length, at least max(span, \
                     ceil(m^(1/3)))) over the {} times where every atom's lag window is \
                     available; each atom's lag-aligned rows keep their original lag \
                     windows and every atom's three mechanism regressions are refit on the \
                     same resampled times, then mixed with the frozen weights inside the \
                     replicate; replicate SD scaled by the circular-Bartlett fixed-b factor \
                     {:.4} and the Bartlett kernel-bias factor {:.4} of the contrast and \
                     mixture scores; score effective rows {:.1}; between-atom sampling covariance \
                     included; unidentified and unevaluable mass is not mixed into the SE; \
                     the interval is for the reported aggregate, not a distribution over \
                     graph-specific effects",
                    s.block.replicates_ok,
                    s.block.replicates_attempted,
                    s.block.block_length,
                    s.block.rows,
                    antecedent_estimate::circular_fixed_b_scale(s.block.block_length, s.block.rows),
                    s.block.kernel_bias,
                    s.block.effective_rows,
                ),
            ));
            let mut warning = short_series_warning(s.block.effective_rows, family);
            if warning.is_none() && atom_count > 1 {
                warning = short_series_warning(
                    s.block.effective_rows,
                    antecedent_estimate::CircularBlockFamily::Mediation,
                );
            }
            diagnostics.extend(warning);
        }
        _ => diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.mediation.uncertainty_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            if replicates_requested {
                "the shared circular-block bootstrap did not produce a usable SE (too few \
                 successful replicates, or the atoms share no series time); the aggregate \
                 SE is withheld and iid analytic SEs are not substituted"
            } else {
                "no bootstrap replicates were requested; lagged rows of one series are \
                 serially dependent, so no iid analytic SE is published for the aggregate"
            },
        )),
    }
    diagnostics
}
