//! Shared uncertainty for cross-world graph-posterior mixtures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AcceptedGraph, Arc, AtomRun, BootstrapAccounting, CausalError, CausalPosterior, CrossWorldKind,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, ExecutionContext,
    GraphPosterior, HashMap, IdentificationStatus, LagMask, MediationQuery, MixedAtomEvals,
    OverlapPolicy, PhysicalExecutionPlan, RefuteSuite, Replicate, ReplicateSe,
    SHARED_REPLICATE_STREAM, StudyResult, TableView, TabularData, TimeSeriesData, UNCERTAINTY_CODE,
    VariableId, effect_from_posterior,
};

impl super::super::Study {
    /// Re-run every contributing temporal-PAG atom on shared resampled series and read the spread
    /// of the draw-wise mixture of their requested values.
    ///
    /// The series is resampled in circular blocks of consecutive periods, never shorter than
    /// the window a pulse and its outcome horizon span, so every atom reads the same resampled
    /// series and the windows its template needs stay intact inside a block.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the operation keeps its checked inputs and execution context together"
    )]
    pub(super) fn temporal_pag_shared_replicates(
        &self,
        series: &TimeSeriesData,
        posterior: (&GraphPosterior, &[LagMask], u32),
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        atoms: &[(u64, f64)],
        budget: &mut antecedent_core::TemporalBudget,
    ) -> Result<ReplicateSe, CausalError> {
        let (gp, _lag_masks, max_lag) = posterior;
        let requested = self.settings.bootstrap_replicates;
        let unavailable =
            |note: &str| ReplicateSe { se: None, ok: 0, failed: 0, note: note.to_owned() };
        if requested < 2 {
            return Ok(unavailable(
                "no bootstrap replicates were requested, so the mixture of several atoms \
                 publishes no interval",
            ));
        }
        if matches!(
            self.declared.sampling_design,
            Some(
                antecedent_core::DeclaredSamplingDesign::ClusteredPanel
                    | antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories
            )
        ) {
            return Ok(unavailable(
                "the declared design needs whole subjects or clusters, and one series carries \
                 no labels to draw them from",
            ));
        }
        let n = series.row_count();
        let horizon = query.horizons.iter().copied().max().unwrap_or(1);
        let window = max_lag as usize + horizon as usize + 1;
        let block = antecedent_data::circular_block_length(1, n).max(window);
        let ids: Vec<VariableId> =
            series.schema().variables().iter().map(|variable| variable.id).collect();
        let columns: Vec<Vec<f64>> =
            match ids.iter().map(|&id| series.float64_values(id)).collect::<Result<_, _>>() {
                Ok(columns) => columns,
                Err(_) => {
                    return Ok(unavailable(
                        "a column cannot be read as floats, so the series cannot be resampled",
                    ));
                }
            };
        let total: f64 = atoms.iter().map(|(_, weight)| *weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(unavailable("the contributing atoms carry no posterior mass"));
        }
        let Some(members) = self.temporal_pag_replicate_members(gp, &ids, atoms, budget, ctx)?
        else {
            return Ok(unavailable("a contributing atom no longer builds its template"));
        };
        let slot = CrossWorldKind::Mediation(query).requested();
        let tabular = TabularData::new(series.storage().clone());
        let mut values: Vec<f64> = Vec::new();
        let mut failed = 0_u32;
        'replicates: for replicate in 0..requested {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let mut rng = ctx.rng.stream_for(
                antecedent_core::StreamDomain::Resample,
                SHARED_REPLICATE_STREAM.wrapping_add(0x100 + u64::from(replicate)),
            );
            let picked = resample_rows(n, block, &mut rng);
            let resampled = resampled_series(series, &tabular, &ids, &columns, &picked);
            let Some(resampled) = resampled else {
                failed += 1;
                continue;
            };
            let mut mixed = 0.0;
            for (weight, member, _lease) in &members {
                let value = match member
                    .execute_temporal_pag_mediation_within(&resampled, query, physical, budget, ctx)
                {
                    Ok(result) => CrossWorldKind::Mediation(query).component_value(&result, slot),
                    Err(error @ (CausalError::Resource { .. } | CausalError::Cancelled { .. })) => {
                        return Err(error);
                    }
                    Err(_) => None,
                }
                .filter(|value| value.is_finite());
                let Some(value) = value else {
                    failed += 1;
                    continue 'replicates;
                };
                mixed += weight * value;
            }
            values.push(mixed / total);
        }
        let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
        let se = replicate_standard_error(&values, requested);
        Ok(ReplicateSe {
            se,
            ok,
            failed,
            note: format!(
                "every contributing temporal-PAG atom was re-run on each of {requested} shared \
                 resampled series (circular blocks of {block} consecutive periods, no shorter \
                 than the {window}-period window); the standard error is the spread of the \
                 draw-wise posterior-weighted mixture over {ok} surviving replicates"
            ),
        })
    }

    /// Resolve every contributing template before any shared resampling stream is drawn.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the operation keeps its checked inputs and execution context together"
    )]
    pub(super) fn temporal_pag_replicate_members(
        &self,
        gp: &GraphPosterior,
        ids: &[VariableId],
        atoms: &[(u64, f64)],
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<(f64, Self, antecedent_core::execution::AdmissionLease)>>, CausalError>
    {
        let mut members = Vec::with_capacity(atoms.len());
        for (key, weight) in atoms {
            let Ok(index) = usize::try_from(*key) else {
                return Ok(None);
            };
            let Some(crate::analysis::identification_cache::AdmittedTemporalClassAtom::Pag(
                template,
                lease,
            )) = crate::analysis::identification_cache::admitted_temporal_class_atom(
                gp, index, ids, budget, ctx,
            )?
            else {
                return Ok(None);
            };
            let mut member = self.clone();
            member.settings.bootstrap_replicates = 0;
            member.settings.refute = RefuteSuite::None;
            member.graph = Arc::new(AcceptedGraph::temporal_pag(template));
            members.push((*weight, member, lease));
        }
        Ok(Some(members))
    }

    /// The uncertainty of the mixture of the atoms that contribute a value.
    ///
    /// One contributing atom carries the whole conditional mixture, so its own interval and
    /// posterior are the mixture's. Several atoms: Frequentist re-runs every atom on shared
    /// resampled tables and reads the spread of the draw-wise posterior-weighted mixture;
    /// Bayesian pools the atoms' draws (all run in one context, so draw `d` of every atom is one
    /// draw of the row weights). When neither can be formed the interval is withheld and the
    /// result says why.
    pub(super) fn mixture_uncertainty(
        &self,
        mixed: &mut MixedAtomEvals,
        by_key: &HashMap<u64, &AtomRun>,
        replicates: Option<Replicate<'_>>,
    ) -> Result<Vec<Diagnostic>, CausalError> {
        let contributing: Vec<(u64, f64, &StudyResult)> = mixed
            .mixture
            .atoms
            .iter()
            .filter(|atom| atom.value.is_some())
            .filter_map(|atom| match by_key.get(&atom.graph_key).copied() {
                Some(AtomRun::Fit(result)) => Some((atom.graph_key, atom.weight, &**result)),
                _ => None,
            })
            .collect();
        let mut notes = Vec::new();
        if let [(_, _, only)] = contributing.as_slice() {
            adopt_single_atom_uncertainty(mixed, only);
            return Ok(notes);
        }
        if contributing.is_empty() {
            return Ok(notes);
        }
        if self.settings.inference.is_bayesian() {
            let items: Vec<(f64, &CausalPosterior)> = contributing
                .iter()
                .filter_map(|(_, weight, result)| result.posterior.as_ref().map(|p| (*weight, p)))
                .collect();
            let pooled = (items.len() == contributing.len())
                .then(|| {
                    pool_atom_posteriors(
                        &items,
                        mixed.identification.status,
                        (mixed.mixture.unidentified_mass, mixed.mixture.unevaluable_mass),
                    )
                })
                .flatten();
            if let Some(pooled) = pooled {
                if let Ok(pooled_estimate) = effect_from_posterior(&pooled) {
                    mixed.estimate = pooled_estimate;
                }
                mixed.posterior = Some(pooled);
                notes.push(Diagnostic::new(
                    UNCERTAINTY_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "pooled posterior: draw d of the mixture is the posterior-weighted sum of \
                     draw d of every contributing atom, all run in one context so that draw d is \
                     one draw of the row weights",
                ));
            } else {
                notes.push(withheld_uncertainty(
                    "no aggregate posterior: the atoms' draws differ in count or publish no \
                     effect column, so they are not draws of one shared run; the per-atom \
                     posteriors are retained on the structural mixture"
                        .to_owned(),
                ));
            }
            return Ok(notes);
        }
        let Some(replicate) = replicates else {
            notes.push(withheld_uncertainty(
                "the standard error of a mixture of several atoms needs the atoms re-run on \
                 shared resampled tables, which this route does not form; the per-atom intervals \
                 are retained on the structural mixture"
                    .to_owned(),
            ));
            return Ok(notes);
        };
        let atoms: Vec<(u64, f64)> =
            contributing.iter().map(|(key, weight, _)| (*key, *weight)).collect();
        let shared = replicate(&atoms)?;
        if let Some(se) = shared.se {
            mixed.estimate = EffectEstimate::from_parts(
                mixed.estimate.ate,
                f64::NAN,
                BootstrapAccounting {
                    se: Some(se),
                    replicates_ok: Some(shared.ok),
                    replicates_failed: Some(shared.failed),
                    cancelled: false,
                    early_stopped: false,
                },
                mixed.estimate.provenance.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
                None,
                None,
            );
            notes.push(Diagnostic::new(
                UNCERTAINTY_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                shared.note,
            ));
        } else {
            notes.push(withheld_uncertainty(shared.note));
        }
        Ok(notes)
    }

    /// Re-run every contributing atom on shared resampled tables and read the spread of the
    /// draw-wise mixture of their requested values.
    pub(super) fn cross_world_shared_replicates(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        atoms: &[(u64, f64)],
    ) -> Result<ReplicateSe, CausalError> {
        let requested = self.settings.bootstrap_replicates;
        let unavailable =
            |note: &str| ReplicateSe { se: None, ok: 0, failed: 0, note: note.to_owned() };
        if requested < 2 {
            return Ok(unavailable(
                "no bootstrap replicates were requested, so the mixture of several atoms \
                 publishes no interval",
            ));
        }
        let n = data.row_count();
        let block = match self.declared.sampling_design {
            Some(antecedent_core::DeclaredSamplingDesign::ClusteredPanel) => {
                return Ok(unavailable(
                    "the declared clustered design needs whole clusters, and the table carries \
                     no cluster labels to draw them from",
                ));
            }
            Some(antecedent_core::DeclaredSamplingDesign::DependentTimeSeries) => {
                antecedent_data::circular_block_length(1, n)
            }
            _ => 1,
        };
        let ids: Vec<VariableId> =
            data.schema().variables().iter().map(|variable| variable.id).collect();
        let columns: Vec<Vec<f64>> =
            match ids.iter().map(|&id| data.float64_values(id)).collect::<Result<_, _>>() {
                Ok(columns) => columns,
                Err(_) => {
                    return Ok(unavailable(
                        "a column cannot be read as floats, so the table cannot be resampled",
                    ));
                }
            };
        let total: f64 = atoms.iter().map(|(_, weight)| *weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(unavailable("the contributing atoms carry no posterior mass"));
        }
        let mut first_index: HashMap<u64, usize> = HashMap::new();
        for index in 0..gp.n_graphs() {
            first_index.entry(gp.graph_keys()[index]).or_insert(index);
        }
        let mut quiet = self.clone();
        quiet.settings.bootstrap_replicates = 0;
        quiet.settings.refute = RefuteSuite::None;
        let slot = kind.requested();
        let mut values: Vec<f64> = Vec::new();
        let mut failed = 0_u32;
        'replicates: for replicate in 0..requested {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let mut rng = ctx.rng.stream_for(
                antecedent_core::StreamDomain::Resample,
                SHARED_REPLICATE_STREAM.wrapping_add(u64::from(replicate)),
            );
            let picked = resample_rows(n, block, &mut rng);
            let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
                .iter()
                .zip(&columns)
                .map(|(&id, column)| (id, picked.iter().map(|&row| column[row]).collect()))
                .collect();
            let Ok(table) = data.with_replaced_floats(&replacements) else {
                failed += 1;
                continue;
            };
            let mut mixed = 0.0;
            for (key, weight) in atoms {
                let value = first_index
                    .get(key)
                    .and_then(|&index| {
                        quiet.cross_world_on_atom(&table, gp, index, kind, physical, ctx).ok()
                    })
                    .and_then(|result| kind.component_value(&result, slot))
                    .filter(|value| value.is_finite());
                let Some(value) = value else {
                    failed += 1;
                    continue 'replicates;
                };
                mixed += weight * value;
            }
            values.push(mixed / total);
        }
        let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
        let se = replicate_standard_error(&values, requested);
        Ok(ReplicateSe {
            se,
            ok,
            failed,
            note: format!(
                "every contributing atom was re-run on each of {requested} shared resampled \
                 tables ({} drawn in blocks of {block} row(s)); the standard error is the spread \
                 of the draw-wise posterior-weighted mixture over {ok} surviving replicates",
                if block == 1 { "rows" } else { "consecutive rows" },
            ),
        })
    }
}

/// Resample `n` rows with replacement, in circular blocks of `block` consecutive rows.
fn resample_rows(n: usize, block: usize, rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
    let block = block.clamp(1, n.max(1));
    let blocks = n.div_ceil(block);
    let mut rows = Vec::with_capacity(blocks * block);
    for _ in 0..blocks {
        let start = antecedent_kernels::unbiased_index(rng, n);
        rows.extend((0..block).map(|offset| (start + offset) % n));
    }
    rows.truncate(n);
    rows
}

/// The pooled posterior of a mixture: draw `d` is the posterior-weighted sum of draw `d` of every
/// atom. `None` when an atom publishes no effect column, or the atoms' draws differ in count so
/// they are not draws of one shared run.
fn pool_atom_posteriors(
    items: &[(f64, &CausalPosterior)],
    status: IdentificationStatus,
    (unidentified_mass, unevaluable_mass): (f64, f64),
) -> Option<CausalPosterior> {
    let (_, first) = items.first()?;
    let first_column = first.effect_column()?;
    let n_draws = first.draws.n_draws;
    let total: f64 = items.iter().map(|(weight, _)| *weight).sum();
    if n_draws < 2 || !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut pooled = vec![0.0; n_draws];
    for (weight, posterior) in items {
        if posterior.draws.n_draws != n_draws {
            return None;
        }
        let column = posterior.draws.column(posterior.effect_column()?).ok()?;
        if column.len() != n_draws || column.iter().any(|draw| !draw.is_finite()) {
            return None;
        }
        for (slot, draw) in pooled.iter_mut().zip(column) {
            *slot += weight / total * draw;
        }
    }
    let quantity = first.draws.schema.quantities.get(first_column)?.clone();
    let schema = antecedent_prob::PosteriorSchema { quantities: Arc::from([quantity]) };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_draws,
        Arc::<[f64]>::from(pooled),
    )
    .ok()?;
    let mut posterior = (*first).clone();
    posterior.summaries = draws.summarize();
    posterior.draws = draws;
    posterior.identification = status;
    posterior.prior_sensitivity = None;
    posterior.unidentified_mass = unidentified_mass;
    posterior.unevaluable_mass = unevaluable_mass;
    posterior.diagnostics.notes.push(Arc::from(
        "pooled over the graph posterior: draw d is the posterior-weighted sum of draw d of every \
         contributing atom, all run in one context so that draw d is one draw of the row weights",
    ));
    for (_, other) in items.iter().skip(1) {
        posterior.assumptions.extend_unique(&other.assumptions.entries);
    }
    Some(posterior)
}

fn replicate_standard_error(values: &[f64], requested: u32) -> Option<f64> {
    (values.len() >= 2
        && super::super::bootstrap_has_enough_successes(values.len(), requested as usize))
    .then(|| {
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance =
            values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / (count - 1.0);
        variance.sqrt()
    })
    .filter(|se| se.is_finite())
}

fn adopt_single_atom_uncertainty(mixed: &mut MixedAtomEvals, only: &StudyResult) {
    if let Some(own) = only.estimate.as_effect() {
        mixed.estimate.uncertainty = own.uncertainty;
        mixed.estimate.scores.influence.clone_from(&own.scores.influence);
    }
    if mixed.posterior.is_none() {
        mixed.posterior.clone_from(&only.posterior);
    }
    if let Some(posterior) = mixed.posterior.as_mut() {
        posterior.unidentified_mass = mixed.mixture.unidentified_mass;
        posterior.unevaluable_mass = mixed.mixture.unevaluable_mass;
    }
}

fn resampled_series(
    series: &TimeSeriesData,
    tabular: &TabularData,
    ids: &[VariableId],
    columns: &[Vec<f64>],
    picked: &[usize],
) -> Option<TimeSeriesData> {
    let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
        .iter()
        .zip(columns)
        .map(|(&id, column)| (id, picked.iter().map(|&row| column[row]).collect()))
        .collect();
    tabular.with_replaced_floats(&replacements).ok().and_then(|table| {
        TimeSeriesData::try_new(table.storage().clone(), series.time_index().clone()).ok()
    })
}

fn withheld_uncertainty(note: String) -> Diagnostic {
    Diagnostic::new(UNCERTAINTY_CODE, DiagnosticKind::Scientific, DiagnosticSeverity::Warning, note)
}
