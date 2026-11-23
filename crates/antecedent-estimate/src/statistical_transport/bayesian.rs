use super::{
    Arc, Assignment, BayesianLawDrawer, BayesianTransportLawProvider, BoundTransportFunctional,
    EstimationError, ExactDistribution, ExactEvaluationLimits, ExecutionContext,
    POSTERIOR_EQUAL_TAIL, ReplicateColumns, ReplicatePolicy, StatisticalTransportInput,
    StreamDomain, TransportIntervalReason, TransportStream, VariableId, dependence_refusal,
    is_support_failure, prepare_exact_transport, refuse_eval,
};

/// Posterior summaries from coherent finite-joint law draws.
#[derive(Clone, Debug)]
pub struct BayesianStatisticalTransportEstimate {
    /// Provider identity, including its support/prior distinction.
    pub estimator: Arc<str>,
    /// Posterior interval interpretation.
    pub interval_method: Arc<str>,
    /// Complete transport distribution for every joint posterior draw.
    pub distributions: Arc<[ExactDistribution]>,
    /// Pointwise equal-tail posterior intervals for each atom, in distribution order.
    pub atom_intervals: Arc<[(f64, f64)]>,
    /// Pointwise equal-tail posterior intervals by outcome mean.
    pub mean_intervals: Arc<[(VariableId, f64, f64)]>,
    /// Why the intervals are withheld, or [`TransportIntervalReason::NotMeasured`] when they are published.
    pub interval_reason: Option<TransportIntervalReason>,
    /// Requested joint posterior draws.
    pub draws_requested: u32,
    /// Successfully evaluated draws.
    pub draws_ok: u32,
    /// Failed draws, retained in accounting.
    pub draws_failed: u32,
}

/// Settings shared by every request in a Bayesian statistical transport grid.
#[derive(Clone, Copy, Debug)]
pub struct BayesianStatisticalTransportOptions {
    /// Finite-joint posterior law provider.
    pub provider: BayesianTransportLawProvider,
    /// Number of shared posterior law draws.
    pub draws: u32,
    /// Equal-tail interval probability mass.
    pub coverage_level: f64,
    /// Maximum cells in a materialized joint law.
    pub max_joint_cells: usize,
}

/// Evaluate finite-discrete structural transport under joint Bayesian law draws.
///
/// Each independent dataset is drawn once per iteration and aliases resolve to that same
/// draw. Caller supplied exact laws are included unchanged. The two providers have
/// distinct identities and support semantics; this function makes no causal-identification
/// or repeated-sampling calibration claim.
/// # Errors
/// Invalid interval settings, cancellation, or exhausted exact-evaluation resources.
pub fn evaluate_bayesian_statistical_transport(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    options: BayesianStatisticalTransportOptions,
    ctx: &ExecutionContext,
) -> Result<BayesianStatisticalTransportEstimate, EstimationError> {
    let mut estimates = evaluate_bayesian_statistical_transport_grid(
        functional,
        input,
        std::slice::from_ref(request),
        limits,
        options,
        ctx,
    )?;
    Ok(estimates.remove(0))
}

/// Evaluate every request from one shared sequence of posterior law draws.
///
/// A draw that fails support is counted and skipped. Intervals are withheld, rather than
/// estimated from the survivors, when successful draws are fewer than two or the failure
/// fraction exceeds the bootstrap rule. Published intervals carry
/// [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`]: coverage is measured in the release calibration pass.
/// # Errors
/// Invalid interval settings, cancellation, exhausted resources, or atom support that changes
/// between successful draws.
pub fn evaluate_bayesian_statistical_transport_grid(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: BayesianStatisticalTransportOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<BayesianStatisticalTransportEstimate>, EstimationError> {
    if requests.is_empty() {
        return Err(EstimationError::data_msg("Bayesian transport grid is empty"));
    }
    if options.draws < 2
        || !options.coverage_level.is_finite()
        || !(options.coverage_level > 0.0 && options.coverage_level < 1.0)
    {
        return Err(EstimationError::data_msg(
            "Bayesian transport requires at least two draws and coverage strictly between zero and one",
        ));
    }
    let estimator: Arc<str> = Arc::from(match options.provider {
        BayesianTransportLawProvider::EmpiricalSupport => {
            crate::empirical_table::EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP
        }
        BayesianTransportLawProvider::DeclaredStateSpaceDirichlet => {
            crate::empirical_table::STATE_SPACE_DIRICHLET
        }
    });
    if let Some(reason) = dependence_refusal(functional.catalog(), &input.samples) {
        return Ok(withheld_bayesian_grid(&estimator, options.draws, 0, 0, reason, requests.len()));
    }
    let outcomes = functional.derivation().query().outcomes.clone();
    let drawer =
        BayesianLawDrawer::prepare(input, functional, options.provider, options.max_joint_cells)?;
    let mut per_request: Vec<ReplicateColumns> =
        requests.iter().map(|_| ReplicateColumns::new(&outcomes)).collect();
    let mut distributions: Vec<Vec<ExactDistribution>> =
        (0..requests.len()).map(|_| Vec::new()).collect();
    let mut failed = 0u32;
    for draw_id in 0..options.draws {
        crate::transport::refuse_cancelled(ctx, "Bayesian transport execution")?;
        let mut rng = ctx
            .rng
            .stream_for(StreamDomain::Transport, TransportStream::BayesianGrid.index(0, draw_id)?);
        let (mut laws, by_sample) = drawer.draw(&mut rng)?;
        laws.extend(by_sample.into_values());
        let data = antecedent_expr::ExactTransportData::try_new(laws, options.max_joint_cells)?;
        let mut request_distributions = Vec::with_capacity(requests.len());
        let mut draw_failed = false;
        for request in requests {
            match prepare_exact_transport(functional, data.clone(), request.clone(), limits, ctx)
                .and_then(|plan| plan.evaluate(ctx))
            {
                Ok(distribution) => request_distributions.push(distribution),
                Err(error) if is_support_failure(&error) => {
                    draw_failed = true;
                    break;
                }
                Err(error) => return Err(refuse_eval(&error)),
            }
        }
        if draw_failed {
            failed = failed.saturating_add(1);
            continue;
        }
        for ((columns, slot), distribution) in
            per_request.iter_mut().zip(&mut distributions).zip(request_distributions)
        {
            columns.record(&distribution)?;
            slot.push(distribution);
        }
    }
    let successful = options.draws.saturating_sub(failed);
    let withhold = ReplicatePolicy::BOOTSTRAP
        .decide(options.draws, successful, failed)
        .err()
        .map(|_| TransportIntervalReason::BootstrapFailureFraction);
    per_request
        .into_iter()
        .zip(distributions)
        .map(|(columns, distributions)| {
            let (atom_intervals, mean_intervals, interval_reason) = match withhold {
                Some(reason) => (Arc::from([]), Arc::from([]), reason),
                None => match columns.intervals(options.coverage_level) {
                    Ok(intervals) => {
                        (intervals.atoms, intervals.means, TransportIntervalReason::NotMeasured)
                    }
                    Err(reason) => (Arc::from([]), Arc::from([]), reason),
                },
            };
            Ok(BayesianStatisticalTransportEstimate {
                estimator: Arc::clone(&estimator),
                interval_method: Arc::from(POSTERIOR_EQUAL_TAIL),
                distributions: distributions.into(),
                atom_intervals,
                mean_intervals,
                interval_reason: Some(interval_reason),
                draws_requested: options.draws,
                draws_ok: successful,
                draws_failed: failed,
            })
        })
        .collect()
}

fn withheld_bayesian_grid(
    estimator: &Arc<str>,
    draws: u32,
    ok: u32,
    failed: u32,
    reason: &'static str,
    requests: usize,
) -> Vec<BayesianStatisticalTransportEstimate> {
    (0..requests)
        .map(|_| BayesianStatisticalTransportEstimate {
            estimator: Arc::clone(estimator),
            interval_method: Arc::from(POSTERIOR_EQUAL_TAIL),
            distributions: Arc::from([]),
            atom_intervals: Arc::from([]),
            mean_intervals: Arc::from([]),
            interval_reason: Some(TransportIntervalReason::from_dependence_refusal(reason)),
            draws_requested: draws,
            draws_ok: ok,
            draws_failed: failed,
        })
        .collect()
}
