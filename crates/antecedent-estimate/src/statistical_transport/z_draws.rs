use super::{
    AliasPolicy, Arc, Assignment, CitedTables, EstimationError, ExactDistribution,
    ExactEvaluationLimits, ExecutionContext, NominalZTransportInterval, ReplicateColumns,
    ReplicatePolicy, StreamDomain, TransportIntervalReason, TransportStream,
    Z_TRANSPORT_INTERVAL_NOT_MEASURED, is_support_failure, refuse_eval, supported_iid_regimes,
};

/// What one z-transport interval route draws: its method label, RNG stream
/// tag, cancellation stage, replicate count and coverage, and whether the drawn
/// tables carry counts (a bootstrap) or probabilities only (a posterior).
pub(super) struct ZDrawSpec {
    pub(super) method: &'static str,
    pub(super) stream: TransportStream,
    pub(super) stage: &'static str,
    pub(super) replicates: u32,
    pub(super) coverage_level: f64,
    pub(super) keep_counts: bool,
}

/// The one draw loop of both z-transport interval routes.
///
/// The cited count tables are validated against the catalog once; every draw
/// then rebuilds them with `draw(counts, rng, probabilities, resampled)` per
/// dataset (returning `false` for a draw that cannot be taken), compiles the
/// checked formula against the rebuilt laws without revalidating providers,
/// and records the replicate. Each dataset reads its own RNG stream.
pub(super) fn z_interval_over_draws(
    functional: &dyn crate::transport::BoundZFormula,
    data: &antecedent_expr::ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    spec: &ZDrawSpec,
    ctx: &ExecutionContext,
    draw: impl FnMut(
        &[u64],
        &mut antecedent_core::CausalRng,
        &mut Vec<f64>,
        &mut Vec<u64>,
    ) -> Result<bool, EstimationError>,
) -> Result<Result<NominalZTransportInterval, &'static str>, EstimationError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Ok(Err("exact_supplied_law_no_sampling_uncertainty"));
    }
    validate_draw_spec(spec)?;
    let regimes = data.laws().iter().map(antecedent_expr::ExactDiscreteLaw::regime);
    if !supported_iid_regimes(functional.catalog(), regimes) {
        return Ok(Err("transport.unsupported_dependence"));
    }
    crate::transport::validate_exact_laws(functional.catalog(), data)
        .map_err(|error| refuse_eval(&error))?;
    let tables = CitedTables::new(functional, data, AliasPolicy::Identical)?
        .map_err(EstimationError::data_msg)?;
    let drawn = z_draws(
        functional,
        TableDrawer::new(&tables, data.max_support_rows()),
        std::slice::from_ref(request),
        limits,
        spec,
        ctx,
        draw,
    )?;
    Ok(finish_z_interval(
        spec.method,
        &drawn.columns[0],
        spec.replicates,
        drawn.failed,
        spec.coverage_level,
    ))
}

pub(super) fn validate_draw_spec(spec: &ZDrawSpec) -> Result<(), EstimationError> {
    if spec.replicates < 2
        || !spec.coverage_level.is_finite()
        || !(spec.coverage_level > 0.0 && spec.coverage_level < 1.0)
    {
        return Err(EstimationError::data_msg(
            "z-transport interval requires at least two replicates and coverage strictly between zero and one",
        ));
    }
    Ok(())
}

/// Replicate columns of every request, drawn jointly.
pub(super) struct ZDraws {
    /// One column set per request; replicate `r` of every set is the same draw.
    pub(super) columns: Vec<ReplicateColumns>,
    /// Draws where any request failed; such a draw is recorded for none.
    pub(super) failed: u32,
}

/// One resampled law set per replicate, from reusable per-dataset buffers.
pub(super) struct TableDrawer<'t, 'a> {
    tables: &'t CitedTables<'a>,
    max_support_rows: usize,
    resampled: Vec<Vec<u64>>,
    probabilities: Vec<Vec<f64>>,
}

impl<'t, 'a> TableDrawer<'t, 'a> {
    pub(super) fn new(tables: &'t CitedTables<'a>, max_support_rows: usize) -> Self {
        Self {
            tables,
            max_support_rows,
            resampled: tables.counts.iter().map(|c| vec![0; c.len()]).collect(),
            probabilities: tables.counts.iter().map(|c| vec![0.0; c.len()]).collect(),
        }
    }

    /// Draw every dataset once for `replicate` (each on its own stream) and
    /// rebuild the laws; `None` when a dataset draw cannot be taken.
    pub(super) fn draw(
        &mut self,
        spec: &ZDrawSpec,
        replicate: u32,
        ctx: &ExecutionContext,
        draw: &mut impl FnMut(
            &[u64],
            &mut antecedent_core::CausalRng,
            &mut Vec<f64>,
            &mut Vec<u64>,
        ) -> Result<bool, EstimationError>,
    ) -> Result<Option<antecedent_expr::ExactTransportData>, EstimationError> {
        for (dataset, counts) in self.tables.counts.iter().enumerate() {
            let mut rng =
                ctx.rng.stream_for(StreamDomain::Transport, spec.stream.index(dataset, replicate)?);
            if !draw(
                counts,
                &mut rng,
                &mut self.probabilities[dataset],
                &mut self.resampled[dataset],
            )? {
                return Ok(None);
            }
        }
        let laws = self
            .tables
            .rebuild(&self.probabilities, spec.keep_counts.then_some(&self.resampled))?;
        antecedent_expr::ExactTransportData::try_new(laws, self.max_support_rows)
            .map(Some)
            .map_err(EstimationError::from)
    }
}

/// Draw the cited tables `spec.replicates` times and evaluate every request on
/// each draw. A replicate counts only when every request evaluates, so the
/// columns stay aligned by replicate id and contrasts are within-replicate.
pub(super) fn z_draws(
    functional: &dyn crate::transport::BoundZFormula,
    mut drawer: TableDrawer<'_, '_>,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    spec: &ZDrawSpec,
    ctx: &ExecutionContext,
    mut draw: impl FnMut(
        &[u64],
        &mut antecedent_core::CausalRng,
        &mut Vec<f64>,
        &mut Vec<u64>,
    ) -> Result<bool, EstimationError>,
) -> Result<ZDraws, EstimationError> {
    let outcomes = functional.outcomes().clone();
    let mut columns = requests.iter().map(|_| ReplicateColumns::new(&outcomes)).collect::<Vec<_>>();
    let mut failed = 0u32;
    for replicate in 0..spec.replicates {
        crate::transport::refuse_cancelled(ctx, spec.stage)?;
        let Some(drawn) = drawer.draw(spec, replicate, ctx, &mut draw)? else {
            failed += 1;
            continue;
        };
        let Some(evaluated) = evaluate_z_requests(functional, drawn, requests, limits, ctx)? else {
            failed += 1;
            continue;
        };
        for (column, distribution) in columns.iter_mut().zip(&evaluated) {
            column.record(distribution)?;
        }
    }
    Ok(ZDraws { columns, failed })
}

/// Evaluate every request on one drawn law set, in request order; `None` when
/// any request is a counted support failure. Compilation consumes the law set,
/// so only the last request takes it by move and the others read a clone.
fn evaluate_z_requests(
    functional: &dyn crate::transport::BoundZFormula,
    drawn: antecedent_expr::ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<Option<Vec<ExactDistribution>>, EstimationError> {
    let mut evaluated = Vec::with_capacity(requests.len());
    let Some((last, head)) = requests.split_last() else {
        return Ok(Some(evaluated));
    };
    for request in head {
        let Some(distribution) = evaluate_z_draw(functional, drawn.clone(), request, limits, ctx)?
        else {
            return Ok(None);
        };
        evaluated.push(distribution);
    }
    let Some(distribution) = evaluate_z_draw(functional, drawn, last, limits, ctx)? else {
        return Ok(None);
    };
    evaluated.push(distribution);
    Ok(Some(evaluated))
}

/// Evaluate one resampled or drawn law set; `None` is a counted support failure.
fn evaluate_z_draw(
    functional: &dyn crate::transport::BoundZFormula,
    draw: antecedent_expr::ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<Option<ExactDistribution>, EstimationError> {
    let evaluated =
        crate::transport::compile_exact_z_transport(functional, draw, request.clone(), limits, ctx)
            .and_then(|plan| plan.evaluate(ctx));
    match evaluated {
        Ok(distribution) => Ok(Some(distribution)),
        Err(error) if is_support_failure(&error) => Ok(None),
        Err(error) => Err(refuse_eval(&error)),
    }
}

/// Summarize the recorded z-transport replicates under the bootstrap rule.
fn finish_z_interval(
    method: &'static str,
    columns: &ReplicateColumns,
    requested: u32,
    failed: u32,
    coverage_level: f64,
) -> Result<NominalZTransportInterval, &'static str> {
    ReplicatePolicy::BOOTSTRAP.decide(requested, columns.ok, failed)?;
    let intervals = columns.intervals(coverage_level).map_err(TransportIntervalReason::as_str)?;
    Ok(NominalZTransportInterval {
        method: Arc::from(method),
        reason: Arc::from(Z_TRANSPORT_INTERVAL_NOT_MEASURED),
        coverage_target: coverage_level,
        atom_intervals: intervals.atoms,
        mean_intervals: intervals.means,
        replicates_requested: requested,
        replicates_ok: columns.ok,
        replicates_failed: failed,
    })
}

/// Resample a count table by the same iid row draws without materializing indexes.
pub(super) fn resample_cited_counts(
    counts: &[u64],
    rng: &mut antecedent_core::CausalRng,
    out: &mut [u64],
    ctx: &ExecutionContext,
) -> Result<bool, EstimationError> {
    let total = counts.iter().try_fold(0u64, |n, k| n.checked_add(*k));
    let Some(total) = total.and_then(|total| usize::try_from(total).ok()) else {
        return Ok(false);
    };
    if total == 0 || out.len() != counts.len() {
        return Ok(false);
    }
    crate::transport::refuse_cancelled(ctx, "z-transport bootstrap")?;
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut running = 0u64;
    for count in counts {
        running += count;
        cumulative.push(running);
    }
    if antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories
        .apply_resampling(antecedent_core::ResamplingRule::RowSubjectIid)
        .is_err()
    {
        return Ok(false);
    }
    out.fill(0);
    for draw in 0..total {
        if draw % 1024 == 0 {
            crate::transport::refuse_cancelled(ctx, "z-transport bootstrap")?;
        }
        let index = antecedent_kernels::unbiased_index(rng, total);
        let cell = cumulative.partition_point(|end| *end <= index as u64);
        if cell >= out.len() {
            return Ok(false);
        }
        out[cell] += 1;
    }
    Ok(true)
}
