//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// One penalized-bootstrap replicate after its indexes have been drawn.
pub(super) enum PenalizedDraw {
    Done(f64, Vec<f64>),
    Failed,
    Cancelled,
}

/// Stream tag of the penalized refit bootstrap's resample plan.
pub(super) const PENALIZED_BOOTSTRAP_STREAM: u64 = 0xA1D7_0001;

/// Draw replicate `replicate` into reused buffers on the penalized bootstrap stream.
///
/// Cluster labels draw whole clusters. A series block draws contiguous blocks.
/// Otherwise each row is one subject. `idx` follows the draw, so a cluster
/// replicate may not have length `n`.
#[allow(
    clippy::too_many_arguments,
    reason = "the draw is fixed by the replicate, the grouping, and the declared design together"
)]
pub(super) fn fill_penalized_bootstrap_indexes(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    ids: &mut Vec<antecedent_data::GlobalId>,
    idx: &mut Vec<usize>,
    clusters: Option<&[u32]>,
    series_block: Option<usize>,
    declared: crate::util::SamplingDeclaration,
) -> Result<(), EstimationError> {
    use antecedent_core::{DeclaredSamplingDesign, ResamplingRule};
    use antecedent_data::{DataError, ResamplingPlan};

    let design_err = |error: antecedent_core::SamplingDesignError| {
        EstimationError::Data(DataError::SamplingDesign(error))
    };
    let plan = if clusters.is_some() {
        // Under independent subject trajectories the labels are subject ids, and drawing
        // whole subjects is that design's own rule; every other design draws whole clusters.
        let design = declared.resolve(DeclaredSamplingDesign::ClusteredPanel);
        let rule = if matches!(design, DeclaredSamplingDesign::IndependentSubjectTrajectories) {
            ResamplingRule::RowSubjectIid
        } else {
            ResamplingRule::WholeCluster
        };
        design.apply_resampling(rule).map_err(design_err)?;
        ResamplingPlan::ClusterBootstrap { cluster: antecedent_core::VariableId::from_raw(0) }
    } else if let Some(length) = series_block {
        declared
            .resolve(DeclaredSamplingDesign::DependentTimeSeries)
            .apply_resampling(ResamplingRule::ContiguousBlock)
            .map_err(design_err)?;
        ResamplingPlan::MovingBlock { length }
    } else {
        match declared {
            crate::util::SamplingDeclaration::Undeclared => {
                DeclaredSamplingDesign::IndependentSubjectTrajectories
                    .apply_resampling(ResamplingRule::RowSubjectIid)
            }
            crate::util::SamplingDeclaration::Declared(design) => {
                design.apply_resampling(ResamplingRule::RowIid)
            }
        }
        .map_err(design_err)?;
        ResamplingPlan::IidBootstrap
    };
    antecedent_data::resample::fill_resample_index_replicate(
        plan,
        n,
        clusters,
        ctx,
        PENALIZED_BOOTSTRAP_STREAM,
        replicate,
        ids,
    )?;
    idx.clear();
    idx.try_reserve_exact(ids.len()).map_err(|e| {
        EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
    })?;
    for id in ids.iter() {
        idx.push(usize::try_from(id.get()).map_err(|_| {
            EstimationError::data_msg("a resampled row identity exceeds the addressable table size")
        })?);
    }
    Ok(())
}

/// Evaluate penalized replicates with one index buffer per worker.
///
/// Serial contexts pass `ctx` through. A parallel context passes
/// [`ExecutionContext::serial_inner`] and stops scheduling indexes above the first error,
/// matching [`ExecutionContext::map_indexed`].
pub(super) fn map_penalized_replicates<F>(
    ctx: &ExecutionContext,
    n_rep: usize,
    f: F,
) -> Result<Vec<PenalizedDraw>, EstimationError>
where
    F: Fn(
            usize,
            &ExecutionContext,
            &mut Vec<antecedent_data::GlobalId>,
            &mut Vec<usize>,
        ) -> Result<PenalizedDraw, EstimationError>
        + Sync,
{
    ctx.map_indexed_with_scratch(
        n_rep,
        || (Vec::new(), Vec::new()),
        |replicate, inner, scratch| f(replicate, inner, &mut scratch.0, &mut scratch.1),
    )
}

/// The problem a penalized refit bootstrap resamples.
///
/// The fold seed and replicate nuisance are written onto a clone only when they differ
/// from `problem`. A problem that already carries both matches that clone, so it is reused.
pub(super) fn penalized_bootstrap_base(
    problem: &PreparedPropensityProblem,
    fold_seed: u64,
    nuisance: PropensityNuisance,
) -> std::borrow::Cow<'_, PreparedPropensityProblem> {
    if problem.fold_seed == fold_seed && problem.propensity == nuisance {
        std::borrow::Cow::Borrowed(problem)
    } else {
        let mut seeded = problem.clone();
        seeded.fold_seed = fold_seed;
        seeded.propensity = nuisance;
        std::borrow::Cow::Owned(seeded)
    }
}

/// The prepared problem restricted to the resampled rows `idx`. Copies of one original row
/// keep that row's identity (and shared fold), so no unit is both in a training set and in
/// its own validation fold. `weights` are the custom-target weights (`None` = uniform).
pub(crate) fn resampled_problem(
    problem: &PreparedPropensityProblem,
    idx: &[usize],
    weights: Option<&[f64]>,
) -> PreparedPropensityProblem {
    let mut p = problem.clone();
    let mut design = Vec::new();
    select_rows_colmajor(
        &problem.design_matrix,
        problem.nrows,
        problem.design_ncols,
        idx,
        &mut design,
    );
    p.design_matrix = design.into();
    p.nrows = idx.len();
    p.treatment = gather(&problem.treatment, idx).into();
    p.outcome = gather(&problem.outcome, idx).into();
    p.row_index = idx.iter().map(|&i| problem.row_index[i]).collect::<Vec<_>>().into();
    p.fold_assignment = problem
        .fold_assignment
        .as_deref()
        .map(|ids| idx.iter().map(|&i| ids[i]).collect::<Vec<_>>().into());
    if let Some(w) = weights {
        p.target_weights = Some(gather(w, idx).into());
    }
    if let Some(sampling) = problem.sampling_weights.as_ref() {
        p.sampling_weights = Some(sampling.subset(idx));
    }
    p
}

/// Extract rows `idx` from a column-major `nrows × ncols` matrix into a fresh column-major
/// `idx.len() × ncols` buffer.
pub(crate) fn select_rows_colmajor(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    idx: &[usize],
    out: &mut Vec<f64>,
) {
    crate::util::gather_cols_into(out, matrix, nrows, ncols, idx);
}

pub(super) fn select_values(values: &[f64], idx: &[usize], out: &mut Vec<f64>) {
    crate::util::gather_into(out, values, idx);
}

impl AipwAte {
    /// Row-resample bootstrap of the cross-fitted contrast: each replicate rebuilds the score
    /// table on the resampled rows. `weights` are the custom-target weights (`None` = uniform).
    pub(super) fn crossfit_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        weights: Option<&[f64]>,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let fold_seed = ctx.rng.master_seed();
        let clusters = self.bootstrap_clusters(problem)?;
        let series_block = self.series_block(problem.nrows);
        // Each replicate clones the problem with a resampled design and four row vectors.
        let _gather = crate::util::admit_gather_scratch(
            ctx,
            problem.nrows,
            problem.design_ncols + 4,
            self.inference.bootstrap_replicates,
        )?;
        crate::util::bootstrap_outcome_by_design(
            self.inference.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.inference.sampling_design,
            },
            self.inference.bootstrap_replicates,
            ctx,
            0xA1D5,
            problem.nrows,
            || (),
            |(), idx| {
                let mut p = resampled_problem(problem, idx, weights);
                // The point is the GLM result: a replicate whose GLM fit fails is a failed
                // replicate, never silently a ridge fit.
                p.propensity = PropensityNuisance::default();
                let Ok((t, _)) = self.crossfit_table(&p, false, ctx, Some(fold_seed)) else {
                    return Ok(None);
                };
                let ss = t.summarize(p.score_weights().as_deref())?;
                Ok(Some(ss.means[1] - ss.means[0]))
            },
        )
    }

    /// The nuisance configuration a bootstrap replicate runs: the fallback destination when
    /// the point came from a fallback, the declared penalized propensity, or the plain GLM.
    pub(super) fn replicate_nuisance(&self, point: &EffectEstimate) -> PropensityNuisance {
        let fell_back =
            point.diagnostics.penalized.as_ref().is_some_and(|report| report.fallback.is_some());
        if fell_back {
            self.propensity.fallback_destination().unwrap_or_default()
        } else if self.propensity.is_penalized() {
            self.propensity.clone()
        } else {
            PropensityNuisance::default()
        }
    }

    /// Unit-resample bootstrap of the penalized cross-fitted contrast.
    ///
    /// Every replicate draws `n` rows with replacement into a reused buffer (copies keep their
    /// unit identity, so a unit is never in a training set and in its own validation fold) and
    /// re-runs the whole pipeline on the resample under `nuisance`: the arm-stratified unit
    /// fold plan, each fold's inner-CV penalty (and lasso support) selection and every nuisance
    /// fit. Replicate `r` is drawn from [`antecedent_core::StreamDomain::Resample`] at
    /// `derive_stream(PENALIZED_BOOTSTRAP_STREAM, r)`, so it is fixed by the seed and `r` alone.
    /// The penalties each successful replicate selected are recorded in replicate order, so the
    /// record replays from the seed under any thread count. A replicate that fails softly
    /// (a fold missing an arm, a failed fit, a boundary propensity) is counted, never
    /// replaced; a cancelled one stops the loop and is reported as a stop.
    pub(super) fn attach_penalized_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        nuisance: PropensityNuisance,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let base = penalized_bootstrap_base(problem, ctx.rng.master_seed(), nuisance);
        let n = base.nrows;
        let n_rep = self.inference.bootstrap_replicates as usize;
        if let Some(request) = self.inference.bootstrap_interval {
            return self.attach_penalized_waves(&base, ctx, request, point);
        }
        let outcomes = if ctx.cancellation.is_cancelled() {
            vec![PenalizedDraw::Cancelled]
        } else if n == 0 || n_rep == 0 {
            return Err(EstimationError::Data(antecedent_data::DataError::InvalidArgument {
                message: "batch resample needs n > 0 and n_replicates > 0".into(),
            }));
        } else {
            let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
            retained_index_slots(n, threads)?;
            if u64::try_from(n.saturating_sub(1)).is_err() {
                return Err(EstimationError::data_msg(
                    "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                ));
            }
            map_penalized_replicates(ctx, n_rep, |r, inner, ids, idx| {
                self.penalized_replicate(&base, ctx, inner, n, r, (ids, idx))
            })?
        };
        let mut ates = Vec::with_capacity(n_rep);
        let mut penalties = Vec::with_capacity(n_rep);
        let (mut attempted, mut cancelled) = (0u32, false);
        for (r, outcome) in outcomes.into_iter().enumerate() {
            match outcome {
                PenalizedDraw::Done(ate, lambdas) => {
                    attempted += 1;
                    ates.push(ate);
                    penalties.push(ReplicatePenalties {
                        replicate: u32::try_from(r).unwrap_or(u32::MAX),
                        lambdas,
                    });
                }
                PenalizedDraw::Failed => attempted += 1,
                PenalizedDraw::Cancelled => cancelled = true,
            }
        }
        let boot = crate::util::finalize_bootstrap_se_ex(&ates, attempted, cancelled, false);
        let influence_se = match point.scores.table.as_ref() {
            Some(table) => crossfit_influence_se(table)?,
            None => f64::NAN,
        };
        let mut report = point.diagnostics.penalized.as_deref().cloned().unwrap_or_default();
        report.variance = Some(PenalizedVarianceReport {
            uncertainty_kind: REFIT_BOOTSTRAP_UNCERTAINTY_KIND,
            refit_bootstrap_se: boot.se,
            replicates_requested: self.inference.bootstrap_replicates,
            replicates_ok: boot.replicates_ok,
            replicates_failed: boot.replicates_failed,
            cancelled: boot.cancelled,
            replicate_penalties: penalties,
            influence_se,
        });
        let mut estimate = point.with_bootstrap(Some(boot));
        estimate.diagnostics.penalized = Some(Box::new(report));
        Ok(estimate)
    }

    /// The penalized refit bootstrap as the wave procedure of a requested interval.
    ///
    /// Replicate `r` is the same penalized refit on the same rows as in the fixed-count loop
    /// (stream `derive_stream(PENALIZED_BOOTSTRAP_STREAM, r)`); waves run on the caller and stop
    /// on the request's Monte Carlo tolerances, capped by the declared replicate count. The
    /// penalties the kept replicates selected are recorded in replicate order, and the report
    /// of the procedure travels beside them.
    pub(super) fn attach_penalized_waves(
        &self,
        base: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        request: crate::util::IntervalRequest,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        use crate::adaptive_resample_waves::{DriveError, run_waves_serial};
        let n = base.nrows;
        if n == 0 || self.inference.bootstrap_replicates == 0 {
            return Err(EstimationError::Data(antecedent_data::DataError::InvalidArgument {
                message: "batch resample needs n > 0 and n_replicates > 0".into(),
            }));
        }
        retained_index_slots(n, 1)?;
        let rule = request.rule(self.inference.bootstrap_replicates);
        let mut state = (Vec::<antecedent_data::GlobalId>::new(), Vec::<usize>::new(), Vec::new());
        let run = run_waves_serial(
            &rule,
            ctx,
            PENALIZED_BOOTSTRAP_STREAM,
            &mut state,
            |(ids, idx, penalties): &mut (_, _, Vec<ReplicatePenalties>), id, _rng| {
                let replicate = usize::try_from(id).map_err(|_| {
                    EstimationError::data_msg("bootstrap replicate id exceeds usize")
                })?;
                Ok(match self.penalized_replicate(base, ctx, ctx, n, replicate, (ids, idx))? {
                    PenalizedDraw::Done(ate, lambdas) => {
                        penalties.push(ReplicatePenalties {
                            replicate: u32::try_from(id).unwrap_or(u32::MAX),
                            lambdas,
                        });
                        Some(ate)
                    }
                    // A cancelled draw is observed again before the next replicate, which
                    // discards the wave it belongs to.
                    PenalizedDraw::Failed | PenalizedDraw::Cancelled => None,
                })
            },
        );
        let run = match run {
            Ok(run) => run,
            Err(DriveError::Estimator(error)) => return Err(error),
            Err(DriveError::Wave(error)) => {
                return Err(EstimationError::data_msg(error.to_string()));
            }
        };
        // A wave discarded by cancellation leaves its penalties behind: keep exactly the
        // replicates whose estimates the run kept.
        let mut penalties = state.2;
        penalties.truncate(run.successes.len());
        let wave = crate::util::WaveBootstrap::from_run(run);
        let boot = wave.se;
        let influence_se = match point.scores.table.as_ref() {
            Some(table) => crossfit_influence_se(table)?,
            None => f64::NAN,
        };
        let mut report = point.diagnostics.penalized.as_deref().cloned().unwrap_or_default();
        report.variance = Some(PenalizedVarianceReport {
            uncertainty_kind: REFIT_BOOTSTRAP_UNCERTAINTY_KIND,
            refit_bootstrap_se: boot.se,
            replicates_requested: self.inference.bootstrap_replicates,
            replicates_ok: boot.replicates_ok,
            replicates_failed: boot.replicates_failed,
            cancelled: boot.cancelled,
            replicate_penalties: penalties,
            influence_se,
        });
        let resampling = wave.report(rule);
        let mut estimate = point.with_bootstrap(Some(boot));
        estimate.diagnostics.penalized = Some(Box::new(report));
        estimate.diagnostics.resampling = Some(resampling);
        Ok(estimate)
    }

    /// One penalized refit on the rows drawn for replicate `r`.
    ///
    /// Indexes are written into the caller's buffers. `outer` is the stream the batch used;
    /// `inner` is the context that evaluates the replicate (serial when the pool is parallel).
    pub(super) fn penalized_replicate(
        &self,
        base: &PreparedPropensityProblem,
        outer: &ExecutionContext,
        inner: &ExecutionContext,
        n: usize,
        replicate: usize,
        indexes: (&mut Vec<antecedent_data::GlobalId>, &mut Vec<usize>),
    ) -> Result<PenalizedDraw, EstimationError> {
        let (ids, idx) = indexes;
        if inner.cancellation.is_cancelled() {
            return Ok(PenalizedDraw::Cancelled);
        }
        let replicate = u64::try_from(replicate)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
        let clusters = self.bootstrap_clusters(base)?;
        let series_block = self.series_block(n);
        match fill_penalized_bootstrap_indexes(
            n,
            replicate,
            outer,
            ids,
            idx,
            clusters.as_deref(),
            series_block,
            self.inference.sampling_design,
        ) {
            Ok(()) => {}
            Err(EstimationError::Data(antecedent_data::DataError::Cancelled)) => {
                return Ok(PenalizedDraw::Cancelled);
            }
            Err(error) => return Err(error),
        }
        let p = resampled_problem(base, idx, None);
        Ok(match self.crossfit_table(&p, false, inner, None) {
            Ok((table, fit)) => match table.summarize(None) {
                Ok(summary) => {
                    let ate = summary.means[1] - summary.means[0];
                    if ate.is_finite() {
                        let lambdas = fit.selections.iter().map(|s| s.lambda).collect();
                        PenalizedDraw::Done(ate, lambdas)
                    } else {
                        PenalizedDraw::Failed
                    }
                }
                Err(_) => PenalizedDraw::Failed,
            },
            Err(_) if inner.cancellation.is_cancelled() => PenalizedDraw::Cancelled,
            Err(_) => PenalizedDraw::Failed,
        })
    }

    /// Source-table indices of the complete-case rows, converted once from their identities.
    /// Only source-row cluster labels read them, so without those labels the list is empty.
    pub(super) fn source_rows(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<Vec<usize>, EstimationError> {
        if self.inference.source_cluster_ids.is_none() {
            return Ok(Vec::new());
        }
        physical_indices(&problem.row_index)
    }

    pub(super) fn bootstrap_clusters(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<Option<std::borrow::Cow<'_, [u32]>>, EstimationError> {
        let source_rows = self.source_rows(problem)?;
        let labels = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            source_rows.iter().copied(),
        )?;
        if let Some(labels) = &labels {
            crate::se::require_clusters(Some(labels.as_ref()), problem.nrows)?;
        }
        Ok(labels)
    }

    pub(super) fn series_block(&self, n: usize) -> Option<usize> {
        self.inference.panel_times.as_ref().map(|_| {
            let span = match self.inference.se_kind {
                AnalyticSeKind::NeweyWest { lag } | AnalyticSeKind::PanelClusterHac { lag } => {
                    lag.max(1)
                }
                _ => 1,
            };
            antecedent_data::circular_block_length(span, n)
        })
    }

    pub(super) fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        _workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let clusters = self.bootstrap_clusters(problem)?;
        let series_block = self.series_block(n);
        // Each worker gathers the design, three vectors, and four trimmed copies.
        let _gather = crate::util::admit_gather_scratch(
            ctx,
            n,
            ncols + 8,
            self.inference.bootstrap_replicates,
        )?;
        // Replicates own a cold workspace; parallel workers cannot share the caller's.
        crate::util::bootstrap_outcome_by_design(
            self.inference.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.inference.sampling_design,
            },
            self.inference.bootstrap_replicates,
            ctx,
            0xA1D0_u64,
            n,
            || ReplicateScratch::new(n, ncols),
            |scratch, idx| Ok(self.replicate_ate(problem, idx, scratch)),
        )
    }

    /// One bootstrap replicate: refit the propensity and both outcome arms on the drawn rows
    /// (with the resampled sampling weights) and score the contrast. `None` is a failed
    /// replicate, counted by the caller and never a zero estimate.
    fn replicate_ate(
        &self,
        problem: &PreparedPropensityProblem,
        idx: &[usize],
        scratch: &mut ReplicateScratch,
    ) -> Option<f64> {
        let clip = clip_of(problem.overlap);
        let trim = trim_of(problem.overlap);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let ReplicateScratch {
            workspace,
            x_boot,
            t_boot,
            y_boot,
            e,
            design_trim,
            t_trim,
            y_trim,
            e_trim,
        } = scratch;
        let drawn = idx.len();
        x_boot.resize(drawn * ncols, 0.0);
        for buffer in [&mut *t_boot, &mut *y_boot, &mut *e] {
            buffer.resize(drawn, 0.0);
        }
        crate::util::gather_bootstrap_vector(t_boot, &problem.treatment, idx);
        crate::util::gather_bootstrap_vector(y_boot, &problem.outcome, idx);
        crate::util::gather_cols(x_boot, &problem.design_matrix, n, ncols, idx);
        // The resample carries each drawn row's sampling weight.
        let w_boot: Option<Vec<f64>> =
            problem.sampling_weights.as_ref().map(|s| idx.iter().map(|&i| s.weights[i]).collect());
        let propensity_fit = match w_boot.as_deref() {
            Some(w) => antecedent_stats::fit_propensity_weighted_in_place(
                x_boot,
                drawn,
                ncols,
                t_boot,
                w,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ),
            None => antecedent_stats::fit_propensity_in_place(
                x_boot,
                drawn,
                ncols,
                t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ),
        };
        if propensity_fit.is_err() {
            return None;
        }
        let raw = &workspace.propensity.scores[..drawn];
        e.copy_from_slice(raw);
        if let Some(c) = clip {
            clamp_scores(e, c);
        }
        let retained = trim_retained_rows(raw, trim).ok()?;
        let (design_used, t_used, y_used, e_used): (&[f64], &[f64], &[f64], &[f64]) =
            match &retained {
                Some(kept) => {
                    select_rows_colmajor(x_boot, drawn, ncols, kept, design_trim);
                    gather_into(t_trim, t_boot, kept);
                    gather_into(y_trim, y_boot, kept);
                    gather_into(e_trim, e, kept);
                    (design_trim, t_trim, y_trim, e_trim)
                }
                None => (x_boot, t_boot, y_boot, e),
            };
        let nrows = t_used.len();
        // Weights of the rows the contrast uses, renormalized to mean one.
        let w_used: Option<Vec<f64>> = w_boot.as_ref().map(|w| {
            let mut used: Vec<f64> = match &retained {
                Some(rows) => rows.iter().map(|&i| w[i]).collect(),
                None => w.clone(),
            };
            let scale = used.len() as f64 / used.iter().sum::<f64>();
            for value in &mut used {
                *value *= scale;
            }
            used
        });
        let (beta0, beta1) = fit_outcome_models_weighted(
            design_used,
            nrows,
            ncols,
            t_used,
            y_used,
            w_used.as_deref(),
            self.backend,
            workspace,
        )
        .ok()?;
        predict_colmajor(design_used, nrows, ncols, &beta0, &mut workspace.mu0);
        predict_colmajor(design_used, nrows, ncols, &beta1, &mut workspace.mu1);
        aipw_psi_weighted(
            t_used,
            y_used,
            e_used,
            &workspace.mu0,
            &workspace.mu1,
            &problem.target_population,
            w_used.as_deref(),
            &mut workspace.psi,
        )
        .ok()?;
        if let Some(w) = w_used.as_deref() {
            for (value, &m) in workspace.psi.iter_mut().zip(w) {
                *value *= m;
            }
        }
        let m = workspace.psi.len() as f64;
        Some(workspace.psi.iter().sum::<f64>() / m)
    }
}

/// Per-worker scratch of one bootstrap replicate: a cold workspace, the resampled design and
/// vectors, and the trimmed copies the contrast reads when a trim rule retains a subset.
pub(super) struct ReplicateScratch {
    workspace: AipwWorkspace,
    x_boot: Vec<f64>,
    t_boot: Vec<f64>,
    y_boot: Vec<f64>,
    e: Vec<f64>,
    design_trim: Vec<f64>,
    t_trim: Vec<f64>,
    y_trim: Vec<f64>,
    e_trim: Vec<f64>,
}

impl ReplicateScratch {
    /// Scratch for resamples of `n` rows of an `ncols`-column design.
    pub(super) fn new(n: usize, ncols: usize) -> Self {
        Self {
            workspace: AipwWorkspace::default(),
            x_boot: vec![0.0; n * ncols],
            t_boot: vec![0.0; n],
            y_boot: vec![0.0; n],
            e: vec![0.0; n],
            design_trim: Vec::new(),
            t_trim: Vec::new(),
            y_trim: Vec::new(),
            e_trim: Vec::new(),
        }
    }
}
