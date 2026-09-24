//! Joint outer bootstrap for catalog-bound empirical transport tables.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{ResamplingPlan, fill_resample_indexes};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits};
use antecedent_identify::BoundTransportFunctional;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::empirical_table::{
    BayesianTransportLawProvider, EmpiricalTableOptions, StatisticalTransportInput,
    assemble_point_laws, assemble_statistical_laws, bound_sample_key, dependence_refusal,
    draw_bayesian_transport_laws, licensed_iid_dependence, validate_dataset_aliases,
    validate_options,
};
use crate::error::EstimationError;
use crate::transport::prepare_exact_transport;
use crate::util::BOOTSTRAP_MAX_FAILURE_FRAC;

/// Licensed uncertainty method for the empirical-table path.
pub const PERCENTILE_BOOTSTRAP: &str = "percentile_bootstrap";
/// Posterior interval constructor for finite-discrete structural transport.
pub const POSTERIOR_EQUAL_TAIL: &str = "posterior_equal_tail";

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
    /// Requested joint posterior draws.
    pub draws_requested: u32,
    /// Successfully evaluated draws.
    pub draws_ok: u32,
    /// Failed draws, retained in accounting.
    pub draws_failed: u32,
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
    request: Assignment,
    limits: ExactEvaluationLimits,
    provider: BayesianTransportLawProvider,
    draws: u32,
    coverage_level: f64,
    max_joint_cells: usize,
    ctx: &ExecutionContext,
) -> Result<BayesianStatisticalTransportEstimate, EstimationError> {
    if draws < 2 || !coverage_level.is_finite() || !(0.0..1.0).contains(&coverage_level) {
        return Err(EstimationError::data_msg(
            "Bayesian transport requires at least two draws and coverage strictly between zero and one",
        ));
    }
    let estimator: Arc<str> = Arc::from(match provider {
        BayesianTransportLawProvider::EmpiricalSupport => {
            crate::empirical_table::EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP
        }
        BayesianTransportLawProvider::DeclaredStateSpaceDirichlet => {
            crate::empirical_table::STATE_SPACE_DIRICHLET
        }
    });
    let mut distributions = Vec::with_capacity(draws as usize);
    for draw_id in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::data_msg("Bayesian transport execution cancelled"));
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Transport, 0xB_AE5_0000 | u64::from(draw_id));
        let (mut laws, by_sample) =
            draw_bayesian_transport_laws(input, functional, provider, &mut rng, max_joint_cells)?;
        laws.extend(by_sample.into_values());
        let data = antecedent_expr::ExactTransportData::try_new(laws, max_joint_cells)
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
        let distribution =
            match prepare_exact_transport(functional, data, request.clone(), limits, ctx)
                .and_then(|plan| plan.evaluate(ctx))
            {
                Ok(distribution) => distribution,
                Err(
                    antecedent_expr::EvalError::ExactLaw(_)
                    | antecedent_expr::EvalError::ExactRatioSupport { .. }
                    | antecedent_expr::EvalError::DivisionByZero,
                ) => {
                    continue;
                }
                Err(error) => return Err(EstimationError::data_msg(error.to_string())),
            };
        distributions.push(distribution);
    }
    let successful = u32::try_from(distributions.len()).unwrap_or(u32::MAX);
    let failed = draws.saturating_sub(successful);
    let atom_columns = distributions.iter().map(|d| d.probabilities.to_vec()).collect::<Vec<_>>();
    if atom_columns.is_empty() {
        return Err(EstimationError::data_msg("all Bayesian transport draws failed"));
    }
    let atoms = atom_columns[0].len();
    if atom_columns.iter().any(|draw| draw.len() != atoms) {
        return Err(EstimationError::data_msg(
            "Bayesian transport atom support changed between draws",
        ));
    }
    let mut atom_intervals = Vec::with_capacity(atoms);
    for atom in 0..atoms {
        let mut column = atom_columns.iter().map(|draw| draw[atom]).collect::<Vec<_>>();
        column.sort_by(f64::total_cmp);
        atom_intervals.push(equal_tail_interval_sorted(
            &column,
            coverage_level,
            QuantileRule::Interpolated,
        ));
    }
    let outcomes = functional.derivation().query().outcomes.clone();
    let mut mean_intervals = Vec::with_capacity(outcomes.len());
    for outcome in outcomes.iter().copied() {
        let mut column = distributions
            .iter()
            .map(|d| d.mean(outcome).map_err(|e| EstimationError::data_msg(e.to_string())))
            .collect::<Result<Vec<_>, _>>()?;
        column.sort_by(f64::total_cmp);
        let (lower, upper) =
            equal_tail_interval_sorted(&column, coverage_level, QuantileRule::Interpolated);
        mean_intervals.push((outcome, lower, upper));
    }
    Ok(BayesianStatisticalTransportEstimate {
        estimator,
        interval_method: Arc::from(POSTERIOR_EQUAL_TAIL),
        distributions: distributions.into(),
        atom_intervals: atom_intervals.into(),
        mean_intervals: mean_intervals.into(),
        draws_requested: draws,
        draws_ok: successful,
        draws_failed: failed,
    })
}

/// Joint bootstrap columns aligned to original replicate IDs.
pub type TransportMeanReplicates = Arc<[(VariableId, Arc<[f64]>)]>;

/// Recorded interval row. Failed replicates stay visible.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportUncertaintyRow {
    /// Estimator id.
    pub estimator: Arc<str>,
    /// Interval constructor.
    pub method: Arc<str>,
    /// Nominal coverage.
    pub coverage_target: f64,
    /// `pointwise` unless a simultaneous family is separately licensed.
    pub interval_scope: Arc<str>,
    /// Sample size per estimated snapshot.
    pub sample_sizes: Arc<[(Arc<str>, u32)]>,
    /// Support / positivity regime.
    pub support_regime: Arc<str>,
    /// Requested outer replicates.
    pub replicates_requested: u32,
    /// Successful complete-functional replicates.
    pub replicates_ok: u32,
    /// Support, numeric, or empty-sample failures. Never dropped silently.
    pub replicates_failed: u32,
    /// Master seed that produced the streams.
    pub seed: u64,
    /// Optional coverage-record binding.
    pub calibration_binding: Option<Arc<str>>,
}

/// One statistical evaluation of a certified transport functional.
#[derive(Clone, Debug)]
pub struct StatisticalTransportEstimate {
    /// Plug-in target law.
    pub distribution: ExactDistribution,
    /// Licensed interval metadata, when published.
    pub uncertainty: Option<TransportUncertaintyRow>,
    /// Pointwise percentile intervals aligned with `distribution.atoms`.
    pub atom_intervals: Option<Arc<[(f64, f64)]>>,
    /// Pointwise mean intervals per outcome.
    pub mean_intervals: Option<Arc<[(VariableId, f64, f64)]>>,
    /// Joint atom replicates (replicate × atom), retained for covariance.
    pub atom_replicates: Option<Arc<[Arc<[f64]>]>>,
    /// Per-outcome replicate means.
    pub mean_replicates: Option<TransportMeanReplicates>,
    /// Original replicate indices, shared across grid points (failures are not reindexed).
    pub replicate_ids: Option<Arc<[u32]>>,
    /// Why an interval was withheld, when the point is still published.
    pub uncertainty_reason: Option<Arc<str>>,
}

/// Evaluate the plug-in and, when licensed, the joint outer bootstrap.
///
/// # Errors
/// Provider, support, numerical, dependence-as-hard-error, or budget failure.
pub fn evaluate_statistical_transport(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    request: Assignment,
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<StatisticalTransportEstimate, EstimationError> {
    evaluate_statistical_transport_grid(functional, input, &[request], limits, options, ctx)?
        .pop()
        .ok_or_else(|| EstimationError::data_msg("empty treatment grid"))
}

/// Evaluate a finite treatment grid with one joint refit per dataset and replicate.
/// Every point retains the same successful replicate IDs; intervals are pointwise.
/// # Errors
/// Invalid requests, support failure, cancellation or exceeded resources.
pub fn evaluate_statistical_transport_grid(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    evaluate_grid(functional, input, requests, limits, options, ctx, None, None)
}

/// Evaluate retained point laws without fitting their original samples again.
///
/// The caller must retain laws fitted to exactly these immutable inputs and
/// settings. This numerical primitive creates no compiler or artifact authority.
/// Bootstrap datasets are still refitted jointly for every outer replicate.
/// # Errors
/// Invalid requests, support failure, cancellation, or exceeded resources.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_statistical_transport_grid_with_point_laws(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    point_laws: &antecedent_expr::ExactTransportData,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    evaluate_grid(functional, input, requests, limits, options, ctx, Some(point_laws), None)
}

/// [`evaluate_statistical_transport_grid_with_point_laws`] for a caller that has already
/// evaluated every request's plan on exactly `point_laws` (its eligibility pass): those
/// distributions are the points' estimates and are not computed a second time. Bootstrap
/// datasets are still refitted jointly for every outer replicate.
///
/// `distributions[i]` must be the exact evaluation of `requests[i]` on `point_laws`.
/// # Errors
/// A length mismatch, invalid requests, cancellation, or exceeded resources.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_statistical_transport_grid_with_evaluated_point_laws(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    point_laws: &antecedent_expr::ExactTransportData,
    distributions: Vec<ExactDistribution>,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    if distributions.len() != requests.len() {
        return Err(EstimationError::data_msg(
            "evaluated point laws do not align with the treatment grid",
        ));
    }
    evaluate_grid(
        functional,
        input,
        requests,
        limits,
        options,
        ctx,
        Some(point_laws),
        Some(distributions),
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_grid(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    point_laws: Option<&antecedent_expr::ExactTransportData>,
    evaluated: Option<Vec<ExactDistribution>>,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    if requests.is_empty() {
        return Err(EstimationError::data_msg("empty treatment grid"));
    }
    validate_options(options)?;
    validate_dataset_aliases(functional.catalog(), &input.samples)?;
    let mut budget_ctx = ctx.clone();
    budget_ctx.memory.hard_limit_bytes =
        ctx.memory.hard_limit_bytes.map(|n| n / requests.len() as u64);
    let (fit_cost, reserved_bytes) =
        check_statistical_resources(input, functional, options, &budget_ctx)?;
    budget_ctx.memory.hard_limit_bytes =
        budget_ctx.memory.hard_limit_bytes.map(|limit| limit.saturating_sub(reserved_bytes as u64));
    let bootstrapping = options.bootstrap_replicates > 0
        && !input.samples.is_empty()
        && licensed_iid_dependence(functional.catalog(), &input.samples);
    let evaluations = if bootstrapping { options.bootstrap_replicates as usize + 1 } else { 1 };
    let fits = fit_cost
        .checked_mul(evaluations)
        .ok_or_else(|| EstimationError::data_msg("statistical operation budget overflow"))?;
    let operations = limits
        .operations
        .checked_sub(fits)
        .and_then(|n| n.checked_div(evaluations))
        .and_then(|n| n.checked_div(requests.len()))
        .filter(|n| *n > 0)
        .ok_or_else(|| EstimationError::data_msg("statistical operation budget exceeded"))?;
    let limits = ExactEvaluationLimits { operations, ..limits };
    let data = match point_laws {
        Some(laws) => laws.clone(),
        None => assemble_point_laws(input, functional, options, &budget_ctx)?,
    }
    .with_shared_factor_cache(if requests.len() > 1 { 1024 } else { 0 });
    let distributions = match evaluated {
        Some(distributions) => distributions,
        None => requests
            .iter()
            .map(|request| {
                prepare_exact_transport(
                    functional,
                    data.clone(),
                    request.clone(),
                    limits,
                    &budget_ctx,
                )
                .and_then(|plan| plan.evaluate(&budget_ctx))
                .map_err(|e| EstimationError::data_msg(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let draws = if bootstrapping {
        Some(outer_bootstrap(functional, input, requests, limits, options, &budget_ctx)?)
    } else {
        None
    };
    distributions
        .into_iter()
        .enumerate()
        .map(|(i, distribution)| {
            summarize_bootstrap(
                distribution,
                draws.as_ref().map(|all| all[i].clone()),
                input,
                functional,
                options,
                ctx,
            )
        })
        .collect()
}

fn summarize_bootstrap(
    distribution: ExactDistribution,
    boot: Option<BootstrapDraws>,
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<StatisticalTransportEstimate, EstimationError> {
    let sample_sizes: Arc<[(Arc<str>, u32)]> = input
        .samples
        .iter()
        .map(|sample| {
            (sample.snapshot_identity.clone(), u32::try_from(sample.n()).unwrap_or(u32::MAX))
        })
        .collect();
    let mut estimate = StatisticalTransportEstimate {
        distribution,
        uncertainty: None,
        atom_intervals: None,
        mean_intervals: None,
        atom_replicates: None,
        mean_replicates: None,
        replicate_ids: None,
        uncertainty_reason: None,
    };
    if input.samples.is_empty() || options.bootstrap_replicates == 0 {
        estimate.uncertainty_reason = Some(Arc::from(if input.samples.is_empty() {
            "exact_supplied_law_no_sampling_uncertainty"
        } else {
            "bootstrap_not_requested"
        }));
        return Ok(estimate);
    }
    if !licensed_iid_dependence(functional.catalog(), &input.samples) {
        estimate.uncertainty_reason = Some(Arc::from(
            dependence_refusal(functional.catalog(), &input.samples)
                .unwrap_or("transport.unsupported_dependence"),
        ));
        return Ok(estimate);
    }
    let boot = boot.ok_or_else(|| EstimationError::data_msg("missing joint bootstrap"))?;
    estimate.uncertainty = Some(TransportUncertaintyRow {
        estimator: Arc::from(options.estimator.as_str()),
        method: Arc::from(PERCENTILE_BOOTSTRAP),
        coverage_target: options.coverage_level,
        interval_scope: Arc::from("pointwise"),
        sample_sizes,
        support_regime: Arc::from("empirical_complete_case_no_smoothing"),
        replicates_requested: options.bootstrap_replicates,
        replicates_ok: boot.ok,
        replicates_failed: boot.failed,
        seed: ctx.rng.master_seed(),
        calibration_binding: None,
    });
    estimate.replicate_ids = Some(boot.ids.clone());
    let attempted = boot.ok.saturating_add(boot.failed);
    let fail_frac =
        if attempted == 0 { 0.0 } else { f64::from(boot.failed) / f64::from(attempted) };
    if boot.ok < 2 || fail_frac > BOOTSTRAP_MAX_FAILURE_FRAC {
        estimate.uncertainty_reason = Some(Arc::from("bootstrap_failure_fraction"));
        estimate.atom_replicates = Some(boot.atom_replicates.clone());
        estimate.mean_replicates = Some(boot.mean_replicates.clone());
        return Ok(estimate);
    }
    estimate.atom_intervals =
        Some(pointwise_intervals(&boot.atom_replicates, options.coverage_level));
    estimate.mean_intervals = Some(
        boot.mean_replicates
            .iter()
            .map(|(outcome, reps)| (*outcome, percentile_interval(reps, options.coverage_level)))
            .map(|(outcome, (lo, hi))| (outcome, lo, hi))
            .collect(),
    );
    estimate.atom_replicates = Some(boot.atom_replicates);
    estimate.mean_replicates = Some(boot.mean_replicates);
    Ok(estimate)
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        Err(EstimationError::data_msg("transport statistical execution cancelled"))
    } else {
        Ok(())
    }
}

/// Preflight empirical fitting and retained bootstrap buffers with checked arithmetic.
/// Returns fitting-work and reserved-byte counts.
/// # Errors
/// Cancellation, overflow, or a memory budget smaller than the required buffers.
pub fn check_statistical_resources(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<(usize, usize), EstimationError> {
    check_cancelled(ctx)?;
    validate_options(options)?;
    let overflow = || EstimationError::data_msg("transport statistical memory budget/overflow");
    let mut bytes = 0usize;
    let mut fit_cost = 0usize;
    let outcome_cells = target_cells(input, functional)?;
    for sample in &input.samples {
        let mut cells = 1usize;
        let regime = functional
            .catalog()
            .regimes
            .iter()
            .find(|r| r.id == sample.regime)
            .ok_or_else(|| EstimationError::data_msg("unknown empirical regime"))?;
        for variable in regime
            .measured
            .iter()
            .filter(|v| !sample.interventions.iter().any(|a| a.variable == **v))
        {
            let domain = functional
                .catalog()
                .environments
                .iter()
                .find(|e| e.identity == sample.population)
                .and_then(|e| e.variables.iter().find(|v| v.variable == *variable))
                .map(|v| &v.domain);
            let cardinality = match domain {
                Some(antecedent_core::VariableDomain::Binary) => 2,
                Some(antecedent_core::VariableDomain::Categorical { cardinality }) => {
                    *cardinality as usize
                }
                _ => return Err(EstimationError::data_msg("declared finite domain required")),
            };
            cells = cells
                .checked_mul(cardinality)
                .filter(|n| *n <= options.max_joint_cells)
                .ok_or_else(overflow)?;
        }
        bytes =
            bytes.checked_add(cells.checked_mul(256).ok_or_else(overflow)?).ok_or_else(overflow)?;
        fit_cost = fit_cost
            .checked_add(cells)
            .and_then(|n| {
                sample.n().checked_mul(sample.columns.len()).and_then(|rows| n.checked_add(rows))
            })
            .ok_or_else(overflow)?;
        bytes = bytes
            .checked_add(
                sample
                    .n()
                    .checked_mul(sample.columns.len())
                    .and_then(|n| n.checked_mul(512))
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
    }
    let retained_replicates = if !input.samples.is_empty()
        && licensed_iid_dependence(functional.catalog(), &input.samples)
    {
        options.bootstrap_replicates as usize
    } else {
        0
    };
    let width = outcome_cells
        .checked_add(functional.derivation().query().outcomes.len())
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(128))
        .ok_or_else(overflow)?;
    bytes = bytes
        .checked_add(width.checked_mul(retained_replicates).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    if ctx
        .memory
        .hard_limit_bytes
        .is_some_and(|limit| u64::try_from(bytes).map_or(true, |n| n > limit))
    {
        return Err(overflow());
    }
    Ok((fit_cost, bytes))
}

fn target_cells(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
) -> Result<usize, EstimationError> {
    let overflow = || EstimationError::data_msg("transport statistical cardinality overflow");
    let mut outcome_cells = 1usize;
    for outcome in functional.derivation().query().outcomes.iter() {
        let cardinality = functional
            .catalog()
            .environments
            .iter()
            .flat_map(|e| e.variables.iter())
            .find(|v| v.variable == *outcome)
            .map_or(0, |v| match v.domain {
                antecedent_core::VariableDomain::Binary => 2,
                antecedent_core::VariableDomain::Categorical { cardinality } => {
                    cardinality as usize
                }
                _ => 0,
            });
        // Exact-only coordinates obtain their finite domain from supplied laws.
        let cardinality = if cardinality == 0 {
            input
                .supplied
                .iter()
                .flat_map(antecedent_expr::ExactDiscreteLaw::axes)
                .find(|axis| axis.variable == *outcome)
                .map_or(1, |axis| axis.values.len())
        } else {
            cardinality
        };
        outcome_cells = outcome_cells.checked_mul(cardinality).ok_or_else(overflow)?;
    }
    Ok(outcome_cells)
}

#[derive(Clone)]
struct BootstrapDraws {
    ids: Arc<[u32]>,
    ok: u32,
    failed: u32,
    atom_replicates: Arc<[Arc<[f64]>]>,
    mean_replicates: Arc<[(VariableId, Arc<[f64]>)]>,
}

#[allow(clippy::too_many_lines)] // One joint resampling transaction; no partial draws escape.
fn outer_bootstrap(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<BootstrapDraws>, EstimationError> {
    let outcomes = functional.derivation().query().outcomes.clone();
    let mut atom_reps: Vec<Vec<Arc<[f64]>>> = vec![Vec::new(); requests.len()];
    let mut mean_cols: Vec<BTreeMap<VariableId, Vec<f64>>> = (0..requests.len())
        .map(|_| outcomes.iter().copied().map(|v| (v, Vec::new())).collect())
        .collect();
    let mut ids = Vec::new();
    let mut ok = 0u32;
    let mut failed = 0u32;
    let mut samples: Vec<_> = input
        .samples
        .iter()
        .map(|sample| (bound_sample_key(functional.catalog(), sample), sample))
        .collect();
    if !functional.derivation().sources().is_empty()
        || functional.catalog().bindings.iter().any(|b| b.dataset_identity.is_some())
    {
        samples.sort_by(|a, b| a.0.cmp(&b.0));
    }
    samples.dedup_by(|a, b| a.0 == b.0);
    // Every replicate resamples each dataset on its own `(dataset, replicate)` stream, so
    // replicates are independent; they are evaluated across the context's thread budget
    // and folded in index order. `None` is a counted failed replicate.
    let attempts = ctx.map_indexed::<_, EstimationError, _>(
        options.bootstrap_replicates as usize,
        |index, ctx| {
            let replicate = u32::try_from(index).unwrap_or(u32::MAX);
            check_cancelled(ctx)?;
            let mut indexes = Vec::new();
            let mut row_indexes = BTreeMap::new();
            for (dataset, (key, sample)) in samples.iter().enumerate() {
                let n = sample.n();
                if n == 0 {
                    return Ok(None);
                }
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::Transport,
                    ((dataset as u64) << 32) | u64::from(replicate),
                );
                indexes.clear();
                if fill_resample_indexes(ResamplingPlan::IidBootstrap, n, &mut rng, &mut indexes)
                    .is_err()
                {
                    return Ok(None);
                }
                row_indexes.insert(key.clone(), indexes.clone());
            }
            let assembled =
                match assemble_statistical_laws(input, functional, options, &row_indexes, ctx) {
                    Ok(data) => {
                        data.with_shared_factor_cache(if requests.len() > 1 { 1024 } else { 0 })
                    }
                    Err(EstimationError::EmptyEmpiricalSample { .. }) => return Ok(None),
                    Err(error) => return Err(error),
                };
            let mut complete = Vec::with_capacity(requests.len());
            for request in requests {
                match prepare_exact_transport(
                    functional,
                    assembled.clone(),
                    request.clone(),
                    limits,
                    ctx,
                )
                .and_then(|plan| plan.evaluate(ctx))
                {
                    Ok(distribution) => complete.push(distribution),
                    Err(error) => {
                        check_cancelled(ctx)?;
                        if matches!(
                            error,
                            antecedent_expr::EvalError::ExactLaw(_)
                                | antecedent_expr::EvalError::ExactRatioSupport { .. }
                                | antecedent_expr::EvalError::DivisionByZero
                        ) {
                            break;
                        }
                        return Err(EstimationError::data_msg(error.to_string()));
                    }
                }
            }
            if let Some(progress) = &ctx.progress {
                progress.report(
                    f64::from(replicate + 1) / f64::from(options.bootstrap_replicates),
                    "transport.bootstrap",
                );
            }
            Ok((complete.len() == requests.len()).then_some(complete))
        },
    )?;
    for (index, attempt) in attempts.into_iter().enumerate() {
        let Some(complete) = attempt else {
            failed += 1;
            continue;
        };
        for (i, distribution) in complete.into_iter().enumerate() {
            for outcome in outcomes.iter() {
                let mean = distribution
                    .mean(*outcome)
                    .map_err(|e| EstimationError::data_msg(e.to_string()))?;
                mean_cols[i].get_mut(outcome).expect("known outcome").push(mean);
            }
            atom_reps[i].push(distribution.probabilities);
        }
        ids.push(u32::try_from(index).unwrap_or(u32::MAX));
        ok += 1;
    }
    check_cancelled(ctx)?;
    let ids: Arc<[u32]> = ids.into();
    Ok(atom_reps
        .into_iter()
        .zip(mean_cols)
        .map(|(atoms, means)| BootstrapDraws {
            ids: ids.clone(),
            ok,
            failed,
            atom_replicates: atoms.into(),
            mean_replicates: means.into_iter().map(|(k, v)| (k, Arc::from(v))).collect(),
        })
        .collect())
}

fn pointwise_intervals(replicates: &[Arc<[f64]>], level: f64) -> Arc<[(f64, f64)]> {
    let atoms = replicates.first().map_or(0, |row| row.len());
    (0..atoms)
        .map(|atom| {
            let column: Vec<f64> =
                replicates.iter().filter_map(|row| row.get(atom).copied()).collect();
            percentile_interval(&column, level)
        })
        .collect()
}

/// Linear-interpolation (type-7) percentile interval of resampled replicates.
#[must_use]
pub fn percentile_interval(values: &[f64], level: f64) -> (f64, f64) {
    if values.len() < 2
        || !level.is_finite()
        || level <= 0.0
        || level >= 1.0
        || values.iter().any(|v| !v.is_finite())
    {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    equal_tail_interval_sorted(&sorted, level, QuantileRule::Interpolated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interval_is_pointwise_and_ordered() {
        let (lo, hi) = percentile_interval(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.5);
        // Type 7 on five points: p·(n − 1) = 1 and 3.
        assert!((lo - 1.0).abs() < 1e-12 && (hi - 3.0).abs() < 1e-12);
    }
}
