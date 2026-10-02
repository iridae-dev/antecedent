//! Joint outer bootstrap for catalog-bound empirical transport tables.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    CatalogDistribution, EvidenceCatalog, EvidenceProjection, ExecutionContext, StreamDomain,
    VariableId,
};
use antecedent_data::{ResamplingPlan, fill_resample_indexes};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits};
use antecedent_identify::BoundTransportFunctional;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::empirical_table::{
    BayesianLawDrawer, BayesianTransportLawProvider, EmpiricalTableOptions,
    StatisticalTransportInput, assemble_point_laws, assemble_statistical_laws, bound_sample_key,
    dependence_refusal, dirichlet_posterior_probabilities, licensed_iid_dependence,
    licensed_iid_regimes, validate_dataset_aliases, validate_options,
};
use crate::error::EstimationError;
use crate::transport::{is_support_failure, prepare_exact_transport, refuse_budget, refuse_eval};
use crate::util::ReplicatePolicy;

/// Licensed uncertainty method for the empirical-table path.
pub const PERCENTILE_BOOTSTRAP: &str = "percentile_bootstrap";
/// Posterior interval constructor for finite-discrete structural transport.
pub const POSTERIOR_EQUAL_TAIL: &str = "posterior_equal_tail";
/// Why a published z-transport percentile interval is not a coverage claim.
pub const Z_TRANSPORT_INTERVAL_NOT_MEASURED: &str = "estimator_grid_not_measured";
/// Why an interval is withheld when a replicate summary is not a finite number.
pub const INTERVAL_NUMERICAL_FAILURE: &str = "bootstrap_numerical_failure";

/// Random-stream layout of every statistical transport draw.
///
/// `tag << 48 | dataset << 32 | draw`: the three fields are disjoint, so two
/// cited datasets at the same draw index never share a stream and a draw index
/// never reaches into the dataset field. The classical joint bootstrap and the
/// nominal z-transport bootstrap keep tag zero, which reproduces their streams
/// exactly; the Bayesian draws carry their own tags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransportStream {
    /// Row bootstrap of one cited dataset per replicate.
    Bootstrap = 0,
    /// Joint posterior law draw of a statistical grid, one stream per draw.
    BayesianGrid = 1,
    /// Posterior draw of one cited z-transport law per draw.
    ZPosterior = 2,
}

impl TransportStream {
    const DATASET_BITS: u32 = 16;

    fn index(self, dataset: usize, draw: u32) -> Result<u64, EstimationError> {
        let dataset = u16::try_from(dataset).map_err(|_| {
            EstimationError::refused(
                antecedent_core::reason_code!("invalid_argument"),
                format!(
                    "statistical transport draws at most {} cited datasets per stream",
                    1u64 << Self::DATASET_BITS
                ),
            )
        })?;
        Ok(((self as u64) << 48) | (u64::from(dataset) << 32) | u64::from(draw))
    }
}

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
    /// Why the intervals are withheld, or [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`] when they are published.
    pub interval_reason: Option<Arc<str>>,
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

/// Per-atom and per-outcome-mean replicate columns of one request.
///
/// Every interval constructor on this path records its successful replicates
/// here and summarizes them through one rule, so the classical bootstrap, the
/// nominal z-transport bootstrap, and both posterior draw paths cannot drift
/// in how they check atom support or compute an equal-tail interval.
#[derive(Clone, Debug, Default)]
struct ReplicateColumns {
    outcomes: Vec<VariableId>,
    atoms: Vec<Vec<f64>>,
    means: Vec<Vec<f64>>,
    ok: u32,
}

impl ReplicateColumns {
    fn new(outcomes: &[VariableId]) -> Self {
        Self {
            outcomes: outcomes.to_vec(),
            atoms: Vec::new(),
            means: vec![Vec::new(); outcomes.len()],
            ok: 0,
        }
    }

    /// Record one successful replicate distribution.
    fn record(&mut self, distribution: &ExactDistribution) -> Result<(), EstimationError> {
        if self.ok > 0 && self.atoms.len() != distribution.probabilities.len() {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("transport_numerical_failure"),
                "transport atom support changed between replicates",
            ));
        }
        for (column, outcome) in self.means.iter_mut().zip(&self.outcomes) {
            column.push(distribution.mean(*outcome).map_err(|error| refuse_eval(&error))?);
        }
        for (atom, probability) in distribution.probabilities.iter().enumerate() {
            if self.atoms.len() == atom {
                self.atoms.push(Vec::new());
            }
            self.atoms[atom].push(*probability);
        }
        self.ok = self.ok.saturating_add(1);
        Ok(())
    }

    /// Replicate rows (replicate × atom), in replicate order.
    fn atom_rows(&self) -> Vec<Arc<[f64]>> {
        (0..self.ok as usize)
            .map(|row| self.atoms.iter().map(|column| column[row]).collect())
            .collect()
    }

    /// Per-outcome mean replicates.
    fn mean_columns(&self) -> Arc<[(VariableId, Arc<[f64]>)]> {
        self.outcomes
            .iter()
            .zip(&self.means)
            .map(|(outcome, column)| (*outcome, Arc::from(column.as_slice())))
            .collect()
    }

    /// Equal-tail intervals of every atom and outcome mean, or the reason none
    /// can be published.
    fn intervals(&self, level: f64) -> Result<PointwiseIntervals, &'static str> {
        let atoms = self
            .atoms
            .iter()
            .map(|column| percentile_interval(column, level))
            .collect::<Option<Arc<[_]>>>()
            .ok_or(INTERVAL_NUMERICAL_FAILURE)?;
        let means = self
            .outcomes
            .iter()
            .zip(&self.means)
            .map(|(outcome, column)| {
                percentile_interval(column, level).map(|(lower, upper)| (*outcome, lower, upper))
            })
            .collect::<Option<Arc<[_]>>>()
            .ok_or(INTERVAL_NUMERICAL_FAILURE)?;
        Ok(PointwiseIntervals { atoms, means })
    }
}

/// Pointwise atom and outcome-mean intervals.
struct PointwiseIntervals {
    atoms: Arc<[(f64, f64)]>,
    means: Arc<[(VariableId, f64, f64)]>,
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
        let data = antecedent_expr::ExactTransportData::try_new(laws, options.max_joint_cells)
            .map_err(|error| EstimationError::data_msg(error.to_string()))?;
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
    let withhold = ReplicatePolicy::BOOTSTRAP.decide(options.draws, successful, failed).err();
    per_request
        .into_iter()
        .zip(distributions)
        .map(|(columns, distributions)| {
            let (atom_intervals, mean_intervals, interval_reason) = match withhold {
                Some(reason) => (Arc::from([]), Arc::from([]), reason),
                None => match columns.intervals(options.coverage_level) {
                    Ok(intervals) => {
                        (intervals.atoms, intervals.means, Z_TRANSPORT_INTERVAL_NOT_MEASURED)
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
                interval_reason: Some(Arc::from(interval_reason)),
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
            interval_reason: Some(Arc::from(reason)),
            draws_requested: draws,
            draws_ok: ok,
            draws_failed: failed,
        })
        .collect()
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
        .ok_or_else(|| refuse_budget("statistical operation budget overflow"))?;
    let operations = limits
        .operations
        .checked_sub(fits)
        .and_then(|n| n.checked_div(evaluations))
        .and_then(|n| n.checked_div(requests.len()))
        .filter(|n| *n > 0)
        .ok_or_else(|| refuse_budget("statistical operation budget exceeded"))?;
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
                .map_err(|error| refuse_eval(&error))
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
    let decided = ReplicatePolicy::BOOTSTRAP
        .decide(options.bootstrap_replicates, boot.ok, boot.failed)
        .and_then(|()| boot.columns.intervals(options.coverage_level));
    match decided {
        Ok(intervals) => {
            estimate.atom_intervals = Some(intervals.atoms);
            estimate.mean_intervals = Some(intervals.means);
        }
        Err(reason) => estimate.uncertainty_reason = Some(Arc::from(reason)),
    }
    estimate.atom_replicates = Some(boot.columns.atom_rows().into());
    estimate.mean_replicates = Some(boot.columns.mean_columns());
    Ok(estimate)
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
    crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
    validate_options(options)?;
    let overflow = || refuse_budget("transport statistical memory budget/overflow");
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
    let overflow = || refuse_budget("transport statistical cardinality overflow");
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
    columns: ReplicateColumns,
}

fn outer_bootstrap(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<BootstrapDraws>, EstimationError> {
    let outcomes = functional.derivation().query().outcomes.clone();
    let mut columns: Vec<ReplicateColumns> =
        requests.iter().map(|_| ReplicateColumns::new(&outcomes)).collect();
    let mut ids = Vec::new();
    let mut failed = 0u32;
    // Datasets are resampled in the order of their bound sample keys, whatever
    // the order the evidence was supplied in, so the same evidence always draws
    // the same replicates; forwarded aliases share one dataset.
    let mut samples: Vec<_> = input
        .samples
        .iter()
        .map(|sample| (bound_sample_key(functional.catalog(), sample), sample))
        .collect();
    samples.sort_by(|a, b| a.0.cmp(&b.0));
    samples.dedup_by(|a, b| a.0 == b.0);
    // Every replicate resamples each dataset on its own `(dataset, replicate)` stream, so
    // replicates are independent; they are evaluated across the context's thread budget
    // and folded in index order. `None` is a counted failed replicate.
    let attempts = ctx.map_indexed::<_, EstimationError, _>(
        options.bootstrap_replicates as usize,
        |index, ctx| {
            let replicate = u32::try_from(index).unwrap_or(u32::MAX);
            crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
            let mut indexes = Vec::new();
            let mut row_indexes = BTreeMap::new();
            for (dataset, (key, sample)) in samples.iter().enumerate() {
                let n = sample.n();
                if n == 0 {
                    return Ok(None);
                }
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::Transport,
                    TransportStream::Bootstrap.index(dataset, replicate)?,
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
                        crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
                        if is_support_failure(&error) {
                            break;
                        }
                        return Err(refuse_eval(&error));
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
        for (column, distribution) in columns.iter_mut().zip(&complete) {
            column.record(distribution)?;
        }
        ids.push(u32::try_from(index).unwrap_or(u32::MAX));
    }
    crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
    let ids: Arc<[u32]> = ids.into();
    let ok = u32::try_from(ids.len()).unwrap_or(u32::MAX);
    Ok(columns
        .into_iter()
        .map(|columns| BootstrapDraws { ids: Arc::clone(&ids), ok, failed, columns })
        .collect())
}

/// Equal-tail percentile interval for an empirical z-transport functional.
///
/// The point law is unchanged. The interval resamples each cited count table
/// with the same iid row bootstrap the classical joint outer bootstrap uses,
/// re-evaluates the bound formula, and takes an equal-tail percentile. It is
/// not a calibrated coverage statement.
#[derive(Clone, Debug)]
pub struct NominalZTransportInterval {
    /// `percentile_bootstrap` or `posterior_equal_tail`.
    pub method: Arc<str>,
    /// Always [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`].
    pub reason: Arc<str>,
    /// Nominal equal-tail level.
    pub coverage_target: f64,
    /// Pointwise atom intervals in distribution order.
    pub atom_intervals: Arc<[(f64, f64)]>,
    /// Pointwise outcome-mean intervals.
    pub mean_intervals: Arc<[(VariableId, f64, f64)]>,
    /// Requested replicates.
    pub replicates_requested: u32,
    /// Replicates whose formula evaluation succeeded.
    pub replicates_ok: u32,
    /// Support or numeric failures, retained in the accounting.
    pub replicates_failed: u32,
}

/// Settings for a posterior z-transport interval.
#[derive(Clone, Copy, Debug)]
pub struct BayesianZTransportIntervalOptions {
    /// Finite-joint posterior law provider.
    pub provider: BayesianTransportLawProvider,
    /// Number of posterior law draws.
    pub draws: u32,
    /// Equal-tail interval probability mass.
    pub coverage_level: f64,
}

/// How one cited law is drawn: from the table of its dataset as it is, or as
/// a recorded margin of that dataset's finest table.
struct LawSource {
    /// Index of the resampled table the law reads.
    dataset: usize,
    /// For a projected margin, the law cell of every table cell.
    projection: Option<Vec<usize>>,
}

/// The cited count tables of a z-transport functional, one per independent
/// dataset: laws whose catalog binding names the same forwarded dataset share
/// one table and one resampling stream.
struct CitedTables<'a> {
    laws: &'a [ExactDiscreteLaw],
    /// How each law is rebuilt from the resampled tables.
    sources: Vec<LawSource>,
    /// The counts of each resampled table.
    counts: Vec<&'a [u64]>,
}

/// How laws naming the same forwarded dataset are resampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AliasPolicy {
    /// Every alias carries the identical table; anything else is a data error.
    Identical,
    /// Aliases are identical or a recorded margin of the finest shared table;
    /// any other pair withholds the interval.
    JointProjection,
}

/// Forwarded dataset identity of the catalog binding of `law`'s regime.
fn forwarded_dataset(catalog: &EvidenceCatalog, law: &ExactDiscreteLaw) -> Option<Arc<str>> {
    catalog
        .bindings
        .iter()
        .find(|binding| binding.regime == law.regime())
        .and_then(|binding| binding.dataset_identity.clone())
}

/// Whether two catalog descriptors describe the same law family: everything
/// but the source entry, its snapshot and the projection trail agrees.
fn same_law_family(a: &CatalogDistribution, b: &CatalogDistribution) -> bool {
    let set = |ids: &[VariableId]| ids.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let values = |d: &CatalogDistribution| {
        let mut out =
            d.intervention_values.iter().map(|a| (a.variable, a.value.clone())).collect::<Vec<_>>();
        out.sort_by_key(|(variable, _)| *variable);
        out
    };
    a.study == b.study
        && a.population == b.population
        && a.kind == b.kind
        && a.evidence_kind == b.evidence_kind
        && a.origin == b.origin
        && a.selection == b.selection
        && set(&a.interventions) == set(&b.interventions)
        && values(a) == values(b)
        && set(&a.measured) == set(&b.measured)
        && a.availability == b.availability
        && set(&a.conditioned_on) == set(&b.conditioned_on)
        && a.dataset == b.dataset
}

/// How a law relates to the finest table of the dataset it shares.
enum Relation {
    /// The identical table.
    Identical,
    /// A recorded margin: the law cell of every root cell.
    Margin(Vec<usize>),
}

/// How `law` relates to the finest table `root` of the dataset they share:
/// [`Relation::Identical`] for the identical table, [`Relation::Margin`] for a recorded margin
/// (the catalog descriptor of `law` is `root`'s marginalized over the dropped
/// coordinates, and its counts are exactly that margin of `root`'s counts), and
/// `None` when neither holds.
fn relate_to_root(
    catalog: &EvidenceCatalog,
    root: &ExactDiscreteLaw,
    law: &ExactDiscreteLaw,
) -> Option<Relation> {
    let (Some(root_counts), Some(counts)) = (root.empirical_counts(), law.empirical_counts())
    else {
        return None;
    };
    let (Some(root_d), Some(law_d)) =
        (catalog.distribution(root.regime()), catalog.distribution(law.regime()))
    else {
        return None;
    };
    let same_world = root.population() == law.population()
        && root.interventions().len() == law.interventions().len()
        && law.interventions().iter().all(|a| root.interventions().contains(a));
    if !same_world {
        return None;
    }
    let same_axes = |a: &antecedent_expr::DiscreteAxis, b: &antecedent_expr::DiscreteAxis| {
        a.variable == b.variable && a.values == b.values
    };
    if root.axes().len() == law.axes().len()
        && root.axes().iter().zip(law.axes()).all(|(a, b)| same_axes(a, b))
    {
        return (same_law_family(&root_d, &law_d) && root_counts == counts)
            .then_some(Relation::Identical);
    }
    let drop = root_d
        .measured
        .iter()
        .filter(|variable| !law_d.measured.contains(variable))
        .copied()
        .collect::<Vec<_>>();
    let projected = root_d.project(EvidenceProjection::Marginalize { drop: drop.into() }).ok()?;
    if !same_law_family(&projected, &law_d) {
        return None;
    }
    // Each law axis is a root axis with the same levels; the law cell of a root
    // cell reads those axes' levels (last axis fastest in both layouts).
    let positions = law
        .axes()
        .iter()
        .map(|axis| root.axes().iter().position(|root_axis| same_axes(root_axis, axis)))
        .collect::<Option<Vec<_>>>()?;
    let mut map = Vec::with_capacity(root_counts.len());
    let mut level = vec![0usize; root.axes().len()];
    for _ in 0..root_counts.len() {
        map.push(
            positions
                .iter()
                .zip(law.axes())
                .fold(0usize, |cell, (position, axis)| cell * axis.values.len() + level[*position]),
        );
        for (axis, value) in root.axes().iter().zip(level.iter_mut()).rev() {
            *value += 1;
            if *value < axis.values.len() {
                break;
            }
            *value = 0;
        }
    }
    let mut margin = vec![0u64; counts.len()];
    for (cell, count) in map.iter().zip(root_counts) {
        margin[*cell] = margin[*cell].checked_add(*count)?;
    }
    (margin == counts).then_some(Relation::Margin(map))
}

impl<'a> CitedTables<'a> {
    fn new(
        functional: &dyn crate::transport::BoundZFormula,
        data: &'a antecedent_expr::ExactTransportData,
        policy: AliasPolicy,
    ) -> Result<Result<Self, &'static str>, EstimationError> {
        let laws = data.laws();
        let catalog = functional.catalog();
        let mut tables = Vec::with_capacity(laws.len());
        for law in laws {
            tables.push(law.empirical_counts().ok_or_else(|| {
                EstimationError::data_msg("z-transport interval requires counted laws")
            })?);
        }
        let identities = laws.iter().map(|law| forwarded_dataset(catalog, law)).collect::<Vec<_>>();
        // The finest law of each forwarded dataset, by cell count then order.
        let root_of = |index: usize| -> usize {
            let Some(identity) = &identities[index] else {
                return index;
            };
            match policy {
                AliasPolicy::Identical => (0..laws.len())
                    .find(|j| identities[*j].as_ref() == Some(identity))
                    .unwrap_or(index),
                AliasPolicy::JointProjection => (0..laws.len())
                    .filter(|j| identities[*j].as_ref() == Some(identity))
                    .max_by_key(|j| (tables[*j].len(), std::cmp::Reverse(*j)))
                    .unwrap_or(index),
            }
        };
        let roots = (0..laws.len()).map(root_of).collect::<Vec<_>>();
        // Resampled tables in first-appearance order; the joint policy orders
        // them by (population, regime) so its streams do not depend on the
        // order laws were supplied in.
        let mut order = Vec::new();
        for root in &roots {
            if !order.contains(root) {
                order.push(*root);
            }
        }
        if policy == AliasPolicy::JointProjection {
            order.sort_by_key(|root| (laws[*root].population(), laws[*root].regime().raw()));
        }
        let counts = order.iter().map(|root| tables[*root]).collect::<Vec<_>>();
        let mut sources = Vec::with_capacity(laws.len());
        for (index, law) in laws.iter().enumerate() {
            let root = roots[index];
            let dataset = order.iter().position(|r| *r == root).unwrap_or_default();
            let projection = if root == index {
                None
            } else {
                match policy {
                    AliasPolicy::Identical => {
                        if tables[root] != tables[index] {
                            return Err(EstimationError::data_msg(
                                "conflicting forwarded dataset aliases",
                            ));
                        }
                        None
                    }
                    AliasPolicy::JointProjection => match relate_to_root(catalog, &laws[root], law)
                    {
                        Some(Relation::Identical) => None,
                        Some(Relation::Margin(map)) => Some(map),
                        None => return Ok(Err("transport.unsupported_dependence")),
                    },
                }
            };
            sources.push(LawSource { dataset, projection });
        }
        Ok(Ok(Self { laws, sources, counts }))
    }

    /// Rebuild every cited law from one resampled table per dataset; a projected
    /// law is the margin of its dataset's table in the same replicate.
    fn rebuild(
        &self,
        tables: &[Vec<f64>],
        counts: Option<&[Vec<u64>]>,
    ) -> Result<Vec<ExactDiscreteLaw>, EstimationError> {
        self.laws
            .iter()
            .zip(&self.sources)
            .map(|(law, source)| {
                let dataset = source.dataset;
                let (probabilities, law_counts) = match &source.projection {
                    None => (tables[dataset].clone(), counts.map(|c| c[dataset].clone())),
                    Some(map) => {
                        let cells = law.probabilities().len();
                        if let Some(counts) = counts {
                            let mut margin = vec![0u64; cells];
                            for (cell, count) in map.iter().zip(&counts[dataset]) {
                                margin[*cell] += count;
                            }
                            let total = margin.iter().sum::<u64>() as f64;
                            let p = margin.iter().map(|c| *c as f64 / total).collect();
                            (p, Some(margin))
                        } else {
                            let mut p = vec![0.0; cells];
                            for (cell, mass) in map.iter().zip(&tables[dataset]) {
                                p[*cell] += mass;
                            }
                            (p, None)
                        }
                    }
                };
                let rebuilt = match law_counts {
                    Some(law_counts) => antecedent_expr::ExactDiscreteLaw::try_empirical(
                        law.population(),
                        law.regime(),
                        law.interventions().to_vec(),
                        law.axes().to_vec(),
                        probabilities,
                        law.snapshot_identity(),
                        law.tolerance(),
                    )
                    .and_then(|rebuilt| rebuilt.with_empirical_counts(law_counts)),
                    None => antecedent_expr::ExactDiscreteLaw::try_bayesian_posterior(
                        law.population(),
                        law.regime(),
                        law.interventions().to_vec(),
                        law.axes().to_vec(),
                        probabilities,
                        law.snapshot_identity(),
                        law.tolerance(),
                    ),
                };
                rebuilt.map_err(|error| EstimationError::data_msg(error.to_string()))
            })
            .collect()
    }
}

/// Publish a nominal percentile interval when every cited law has counts.
///
/// Exact laws and dependence the classical empirical path already refuses stay
/// without an interval. The withheld reason is that path's reason, and a
/// bootstrap that cannot evaluate enough replicates withholds the interval
/// with `bootstrap_failure_fraction` while the point stays published.
///
/// # Errors
/// Invalid interval settings, a cited law whose counts do not match its atoms,
/// or cancellation.
pub fn nominal_z_transport_interval(
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    replicates: u32,
    coverage_level: f64,
    ctx: &ExecutionContext,
) -> Result<Result<NominalZTransportInterval, &'static str>, EstimationError> {
    let spec = ZDrawSpec {
        method: PERCENTILE_BOOTSTRAP,
        stream: TransportStream::Bootstrap,
        stage: "z-transport bootstrap",
        replicates,
        coverage_level,
        keep_counts: true,
    };
    let mut indexes = Vec::new();
    z_interval_over_draws(
        functional,
        data,
        request,
        limits,
        &spec,
        ctx,
        |counts, rng, out, resampled| bootstrap_draw(counts, rng, &mut indexes, out, resampled),
    )
}

/// One iid row-bootstrap draw of a cited count table, as counts and frequencies.
fn bootstrap_draw(
    counts: &[u64],
    rng: &mut antecedent_core::CausalRng,
    indexes: &mut Vec<u32>,
    probabilities: &mut Vec<f64>,
    resampled: &mut [u64],
) -> bool {
    if !resample_cited_counts(counts, rng, indexes, resampled) {
        return false;
    }
    let total = resampled.iter().sum::<u64>() as f64;
    probabilities.clear();
    probabilities.extend(resampled.iter().map(|count| *count as f64 / total));
    true
}

#[cfg(feature = "calibration-internal")]
/// Joint percentile-bootstrap intervals for a multi-source mz functional.
///
/// Every replicate draws all cited tables once under one replicate id and
/// evaluates every request on that draw, so each contrast is a within-replicate
/// difference. Independent studies are resampled independently; tables that
/// declare the same forwarded dataset are resampled as one. The intervals are
/// nominal and pointwise and carry [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`]: this
/// route publishes no calibrated claim until its coverage records exist.
#[derive(Clone, Debug)]
pub struct MzTransportIntervals {
    /// `percentile_bootstrap`.
    pub method: Arc<str>,
    /// Always [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`].
    pub reason: Arc<str>,
    /// Nominal equal-tail level.
    pub coverage_target: f64,
    /// Per request, in request order.
    pub requests: Arc<[MzRequestInterval]>,
    /// `(k, outcome, lower, upper)` for the mean of request `k` minus request 0.
    pub contrasts: Arc<[(usize, VariableId, f64, f64)]>,
    /// Requested replicates.
    pub replicates_requested: u32,
    /// Replicates where every request evaluated.
    pub replicates_ok: u32,
    /// Replicates where a draw or any request failed.
    pub replicates_failed: u32,
}

#[cfg(feature = "calibration-internal")]
/// Pointwise intervals of one request of a joint mz bootstrap.
#[derive(Clone, Debug)]
pub struct MzRequestInterval {
    /// Atom intervals in distribution order.
    pub atom_intervals: Arc<[(f64, f64)]>,
    /// `(outcome, lower, upper)` per outcome mean.
    pub mean_intervals: Arc<[(VariableId, f64, f64)]>,
}

/// Whether the declared sampling of the studies a multi-source formula cites
/// licenses a joint bootstrap, or the reason the interval is withheld.
///
/// Every cited regime needs an independent-design binding without weights.
/// Each pair of cited regimes must be declared independent studies, or the same
/// forwarded dataset (resampled as one table). Linked units and shared data
/// without a forwarded dataset identity cannot be resampled jointly and are
/// refused; anything undeclared is `sampling_dependence_unknown`.
///
/// # Errors
/// The stable reason the interval is withheld; the point is unaffected.
pub fn mz_sampling_dependence(
    functional: &antecedent_identify::BoundMzTransportFunctional,
) -> Result<(), &'static str> {
    use antecedent_core::{SamplingDesign, SharedData};
    let catalog = functional.catalog();
    let cited = functional.cited_regimes();
    for regime in cited {
        let Some(binding) = catalog.bindings.iter().find(|b| b.regime == *regime) else {
            return Err("sampling_dependence_unknown");
        };
        if binding.sampling != SamplingDesign::Independent || binding.weights.is_some() {
            return Err("transport.unsupported_dependence");
        }
    }
    for (i, a) in cited.iter().enumerate() {
        for b in &cited[i + 1..] {
            match catalog.shared_data(*a, *b) {
                SharedData::IndependentStudies => {}
                SharedData::SameDataset => {
                    let dataset = |id| {
                        catalog
                            .bindings
                            .iter()
                            .find(|bd| bd.regime == id)
                            .and_then(|bd| bd.dataset_identity.clone())
                    };
                    if dataset(*a).is_none() || dataset(*a) != dataset(*b) {
                        return Err("transport.unsupported_dependence");
                    }
                }
                SharedData::LinkedUnits => return Err("transport.unsupported_dependence"),
                SharedData::Unknown => return Err("sampling_dependence_unknown"),
            }
        }
    }
    Ok(())
}

#[cfg(feature = "calibration-internal")]
/// The spec of the internal mz joint bootstrap: one row bootstrap per cited
/// dataset and replicate on the classical bootstrap stream, with counts kept.
const fn mz_bootstrap_spec(replicates: u32, coverage_level: f64) -> ZDrawSpec {
    ZDrawSpec {
        method: PERCENTILE_BOOTSTRAP,
        stream: TransportStream::Bootstrap,
        stage: "mz-transport bootstrap",
        replicates,
        coverage_level,
        keep_counts: true,
    }
}

/// Validate the laws against the catalog and group them into resampled tables,
/// or the stable reason the mz joint bootstrap is withheld before drawing.
fn mz_tables<'a>(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: &'a antecedent_expr::ExactTransportData,
) -> Result<Result<CitedTables<'a>, &'static str>, EstimationError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Ok(Err("exact_supplied_law_no_sampling_uncertainty"));
    }
    if let Err(reason) = mz_sampling_dependence(functional) {
        return Ok(Err(reason));
    }
    crate::transport::validate_exact_laws(functional.catalog(), data)
        .map_err(|error| refuse_eval(&error))?;
    CitedTables::new(functional, data, AliasPolicy::JointProjection)
}

/// Why the internal joint bootstrap of an mz functional withholds its interval
/// before drawing, or `None` when it would draw.
///
/// Exact laws have no sampling uncertainty; the catalog's declared sampling must
/// license a joint bootstrap ([`mz_sampling_dependence`]); and laws that name the
/// same forwarded dataset must be the identical table or a recorded margin of
/// the finest one (its catalog descriptor is that table's marginalization and
/// its counts are exactly that margin). Any other shared dataset cannot be
/// resampled jointly and is `transport.unsupported_dependence`. The point is
/// never affected.
///
/// # Errors
/// A provider/catalog disagreement.
pub fn mz_interval_withheld_reason(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
) -> Result<Option<&'static str>, EstimationError> {
    Ok(mz_tables(functional, data)?.err())
}

#[cfg(feature = "calibration-internal")]
/// The law sets the internal mz joint bootstrap evaluates, one per replicate
/// (`None` for a replicate whose draw failed), for inspection: laws sharing a
/// forwarded dataset are drawn from one resampled table in every replicate.
///
/// # Errors
/// Fewer than two replicates, a provider/catalog disagreement, or cancellation.
pub fn mz_transport_bootstrap_law_draws(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<Result<Vec<Option<antecedent_expr::ExactTransportData>>, &'static str>, EstimationError>
{
    let spec = mz_bootstrap_spec(replicates, 0.95);
    validate_draw_spec(&spec)?;
    let tables = match mz_tables(functional, data)? {
        Ok(tables) => tables,
        Err(reason) => return Ok(Err(reason)),
    };
    let mut indexes = Vec::new();
    let mut drawer = TableDrawer::new(&tables, data.max_support_rows());
    let mut out = Vec::with_capacity(replicates as usize);
    for replicate in 0..replicates {
        crate::transport::refuse_cancelled(ctx, spec.stage)?;
        out.push(drawer.draw(&spec, replicate, ctx, &mut |counts, rng, p, resampled| {
            bootstrap_draw(counts, rng, &mut indexes, p, resampled)
        })?);
    }
    Ok(Ok(out))
}

#[cfg(feature = "calibration-internal")]
/// The law sets of the iid row bootstrap of a single-source z functional's cited
/// count tables, one per replicate (`None` for a replicate whose draw failed).
///
/// Same stream, alias policy and dependence rule as [`nominal_z_transport_interval`];
/// the 2.2B X3 joint-sensitivity endpoint bootstrap evaluates its exact range on
/// each draw. Compiled only under `calibration-internal` (dev-dependencies alone).
///
/// # Errors
/// Fewer than two replicates, a provider/catalog disagreement, or cancellation.
#[doc(hidden)]
pub fn z_transport_bootstrap_law_draws(
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<Result<Vec<Option<antecedent_expr::ExactTransportData>>, &'static str>, EstimationError>
{
    let spec = ZDrawSpec {
        method: PERCENTILE_BOOTSTRAP,
        stream: TransportStream::Bootstrap,
        stage: "z-transport joint sensitivity bootstrap",
        replicates,
        coverage_level: 0.95,
        keep_counts: true,
    };
    validate_draw_spec(&spec)?;
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Ok(Err("exact_supplied_law_no_sampling_uncertainty"));
    }
    let regimes = data.laws().iter().map(antecedent_expr::ExactDiscreteLaw::regime);
    if !licensed_iid_regimes(functional.catalog(), regimes) {
        return Ok(Err("transport.unsupported_dependence"));
    }
    crate::transport::validate_exact_laws(functional.catalog(), data)
        .map_err(|error| refuse_eval(&error))?;
    let tables = CitedTables::new(functional, data, AliasPolicy::Identical)?
        .map_err(EstimationError::data_msg)?;
    let mut indexes = Vec::new();
    let mut drawer = TableDrawer::new(&tables, data.max_support_rows());
    let mut out = Vec::with_capacity(replicates as usize);
    for replicate in 0..replicates {
        crate::transport::refuse_cancelled(ctx, spec.stage)?;
        out.push(drawer.draw(&spec, replicate, ctx, &mut |counts, rng, p, resampled| {
            bootstrap_draw(counts, rng, &mut indexes, p, resampled)
        })?);
    }
    Ok(Ok(out))
}

#[cfg(feature = "calibration-internal")]
/// Joint percentile bootstrap of a multi-source mz functional over `requests`.
///
/// This is the internal estimator the calibration harness measures; the public
/// prepared route does not publish it until its coverage records exist.
///
/// Like the X4 learned-continuous interval, this is compiled only with the
/// `calibration-internal` feature, which only dev-dependencies (this crate's integration
/// tests, `antecedent-io`'s interval tests and the facade's calibration harness) enable;
/// no released build links it. The public prepared route withholds the interval, and
/// [`mz_interval_withheld_reason`] / [`mz_sampling_dependence`] stay public to report why.
/// Exact laws, undeclared or unsupported dependence (including shared datasets
/// that are not one table or a recorded margin of it), and a bootstrap that
/// fails too often withhold the interval with a stable reason; the point
/// estimate is never withheld by this function.
///
/// # Errors
/// Empty requests, invalid interval settings, a provider/catalog disagreement,
/// or cancellation.
#[doc(hidden)]
pub fn mz_transport_bootstrap_interval(
    functional: &antecedent_identify::BoundMzTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    replicates: u32,
    coverage_level: f64,
    ctx: &ExecutionContext,
) -> Result<Result<MzTransportIntervals, &'static str>, EstimationError> {
    if requests.is_empty() {
        return Err(EstimationError::data_msg(
            "mz-transport interval requires at least one request",
        ));
    }
    let spec = mz_bootstrap_spec(replicates, coverage_level);
    validate_draw_spec(&spec)?;
    let tables = match mz_tables(functional, data)? {
        Ok(tables) => tables,
        Err(reason) => return Ok(Err(reason)),
    };
    let mut indexes = Vec::new();
    let drawn = z_draws(
        functional,
        &tables,
        data.max_support_rows(),
        requests,
        limits,
        &spec,
        ctx,
        |counts, rng, out, resampled| bootstrap_draw(counts, rng, &mut indexes, out, resampled),
    )?;
    let ok = drawn.columns[0].ok;
    if let Err(reason) = ReplicatePolicy::BOOTSTRAP.decide(replicates, ok, drawn.failed) {
        return Ok(Err(reason));
    }
    let mut per_request = Vec::with_capacity(requests.len());
    for column in &drawn.columns {
        match column.intervals(coverage_level) {
            Ok(intervals) => per_request.push(MzRequestInterval {
                atom_intervals: intervals.atoms,
                mean_intervals: intervals.means,
            }),
            Err(reason) => return Ok(Err(reason)),
        }
    }
    let base = &drawn.columns[0];
    let mut contrasts = Vec::new();
    for (k, column) in drawn.columns.iter().enumerate().skip(1) {
        for ((outcome, base_means), means) in
            base.outcomes.iter().zip(&base.means).zip(&column.means)
        {
            let differences = means.iter().zip(base_means).map(|(m, b)| m - b).collect::<Vec<_>>();
            let Some((lower, upper)) = percentile_interval(&differences, coverage_level) else {
                return Ok(Err(INTERVAL_NUMERICAL_FAILURE));
            };
            contrasts.push((k, *outcome, lower, upper));
        }
    }
    Ok(Ok(MzTransportIntervals {
        method: Arc::from(PERCENTILE_BOOTSTRAP),
        reason: Arc::from(Z_TRANSPORT_INTERVAL_NOT_MEASURED),
        coverage_target: coverage_level,
        requests: per_request.into(),
        contrasts: contrasts.into(),
        replicates_requested: replicates,
        replicates_ok: ok,
        replicates_failed: drawn.failed,
    }))
}

/// Equal-tail posterior interval for an empirical z-transport functional.
///
/// Each cited count table is drawn from the named finite-discrete provider and the bound
/// formula is re-evaluated. The point law is unchanged. A published interval is pointwise
/// and carries [`Z_TRANSPORT_INTERVAL_NOT_MEASURED`]. Draws are withheld, rather than
/// summarized from the survivors, when fewer than two succeed or the failure fraction exceeds
/// the bootstrap rule. The analysis layer applies the percentile draw floor.
///
/// # Errors
/// Invalid interval settings, a cited law whose counts do not match its atoms, or cancellation.
pub fn bayesian_z_transport_interval(
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    options: BayesianZTransportIntervalOptions,
    ctx: &ExecutionContext,
) -> Result<Result<NominalZTransportInterval, &'static str>, EstimationError> {
    let prior = match options.provider {
        BayesianTransportLawProvider::EmpiricalSupport => 0.0,
        BayesianTransportLawProvider::DeclaredStateSpaceDirichlet => 1.0,
    };
    let spec = ZDrawSpec {
        method: POSTERIOR_EQUAL_TAIL,
        stream: TransportStream::ZPosterior,
        stage: "z-transport posterior",
        replicates: options.draws,
        coverage_level: options.coverage_level,
        keep_counts: false,
    };
    z_interval_over_draws(functional, data, request, limits, &spec, ctx, |counts, rng, out, _| {
        match dirichlet_posterior_probabilities(counts, prior, rng) {
            Some(drawn) => {
                *out = drawn;
                true
            }
            None => false,
        }
    })
}

/// What one z-transport interval route draws: its method label, RNG stream
/// tag, cancellation stage, replicate count and coverage, and whether the drawn
/// tables carry counts (a bootstrap) or probabilities only (a posterior).
struct ZDrawSpec {
    method: &'static str,
    stream: TransportStream,
    stage: &'static str,
    replicates: u32,
    coverage_level: f64,
    keep_counts: bool,
}

/// The one draw loop of both z-transport interval routes.
///
/// The cited count tables are validated against the catalog once; every draw
/// then rebuilds them with `draw(counts, rng, probabilities, resampled)` per
/// dataset (returning `false` for a draw that cannot be taken), compiles the
/// checked formula against the rebuilt laws without revalidating providers,
/// and records the replicate. Each dataset reads its own RNG stream.
fn z_interval_over_draws(
    functional: &dyn crate::transport::BoundZFormula,
    data: &antecedent_expr::ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    spec: &ZDrawSpec,
    ctx: &ExecutionContext,
    draw: impl FnMut(&[u64], &mut antecedent_core::CausalRng, &mut Vec<f64>, &mut Vec<u64>) -> bool,
) -> Result<Result<NominalZTransportInterval, &'static str>, EstimationError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Ok(Err("exact_supplied_law_no_sampling_uncertainty"));
    }
    validate_draw_spec(spec)?;
    let regimes = data.laws().iter().map(antecedent_expr::ExactDiscreteLaw::regime);
    if !licensed_iid_regimes(functional.catalog(), regimes) {
        return Ok(Err("transport.unsupported_dependence"));
    }
    crate::transport::validate_exact_laws(functional.catalog(), data)
        .map_err(|error| refuse_eval(&error))?;
    let tables = CitedTables::new(functional, data, AliasPolicy::Identical)?
        .map_err(EstimationError::data_msg)?;
    let drawn = z_draws(
        functional,
        &tables,
        data.max_support_rows(),
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

fn validate_draw_spec(spec: &ZDrawSpec) -> Result<(), EstimationError> {
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
struct ZDraws {
    /// One column set per request; replicate `r` of every set is the same draw.
    columns: Vec<ReplicateColumns>,
    /// Draws where any request failed; such a draw is recorded for none.
    failed: u32,
}

/// One resampled law set per replicate, from reusable per-dataset buffers.
struct TableDrawer<'t, 'a> {
    tables: &'t CitedTables<'a>,
    max_support_rows: usize,
    resampled: Vec<Vec<u64>>,
    probabilities: Vec<Vec<f64>>,
}

impl<'t, 'a> TableDrawer<'t, 'a> {
    fn new(tables: &'t CitedTables<'a>, max_support_rows: usize) -> Self {
        Self {
            tables,
            max_support_rows,
            resampled: tables.counts.iter().map(|c| vec![0; c.len()]).collect(),
            probabilities: tables.counts.iter().map(|c| vec![0.0; c.len()]).collect(),
        }
    }

    /// Draw every dataset once for `replicate` (each on its own stream) and
    /// rebuild the laws; `None` when a dataset draw cannot be taken.
    fn draw(
        &mut self,
        spec: &ZDrawSpec,
        replicate: u32,
        ctx: &ExecutionContext,
        draw: &mut impl FnMut(
            &[u64],
            &mut antecedent_core::CausalRng,
            &mut Vec<f64>,
            &mut Vec<u64>,
        ) -> bool,
    ) -> Result<Option<antecedent_expr::ExactTransportData>, EstimationError> {
        for (dataset, counts) in self.tables.counts.iter().enumerate() {
            let mut rng =
                ctx.rng.stream_for(StreamDomain::Transport, spec.stream.index(dataset, replicate)?);
            if !draw(
                counts,
                &mut rng,
                &mut self.probabilities[dataset],
                &mut self.resampled[dataset],
            ) {
                return Ok(None);
            }
        }
        let laws = self
            .tables
            .rebuild(&self.probabilities, spec.keep_counts.then_some(&self.resampled))?;
        antecedent_expr::ExactTransportData::try_new(laws, self.max_support_rows)
            .map(Some)
            .map_err(|error| EstimationError::data_msg(error.to_string()))
    }
}

/// Draw the cited tables `spec.replicates` times and evaluate every request on
/// each draw. A replicate counts only when every request evaluates, so the
/// columns stay aligned by replicate id and contrasts are within-replicate.
#[allow(clippy::too_many_arguments)] // The draw loop's inputs, explicitly.
fn z_draws(
    functional: &dyn crate::transport::BoundZFormula,
    tables: &CitedTables<'_>,
    max_support_rows: usize,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    spec: &ZDrawSpec,
    ctx: &ExecutionContext,
    mut draw: impl FnMut(&[u64], &mut antecedent_core::CausalRng, &mut Vec<f64>, &mut Vec<u64>) -> bool,
) -> Result<ZDraws, EstimationError> {
    let outcomes = functional.outcomes().clone();
    let mut columns = requests.iter().map(|_| ReplicateColumns::new(&outcomes)).collect::<Vec<_>>();
    let mut failed = 0u32;
    let mut drawer = TableDrawer::new(tables, max_support_rows);
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
    let intervals = columns.intervals(coverage_level)?;
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

/// Resample a count table by an iid row bootstrap of its observations.
///
/// The rows are never materialized: each drawn row index is located in the
/// cumulative count table, so the memory is one index buffer (reused across
/// datasets and replicates) plus the output cells. The draws are the same as
/// the row bootstrap of the expanded table.
fn resample_cited_counts(
    counts: &[u64],
    rng: &mut antecedent_core::CausalRng,
    indexes: &mut Vec<u32>,
    out: &mut [u64],
) -> bool {
    let total = counts.iter().try_fold(0u64, |n, k| n.checked_add(*k));
    let Some(total) = total.and_then(|total| usize::try_from(total).ok()) else {
        return false;
    };
    if total == 0 || out.len() != counts.len() {
        return false;
    }
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut running = 0u64;
    for count in counts {
        running += count;
        cumulative.push(running);
    }
    indexes.clear();
    if fill_resample_indexes(ResamplingPlan::IidBootstrap, total, rng, indexes).is_err()
        || indexes.len() != total
    {
        return false;
    }
    out.fill(0);
    for index in indexes.iter() {
        // The first cell whose cumulative count exceeds the row index holds that row.
        let cell = cumulative.partition_point(|end| *end <= u64::from(*index));
        if cell >= out.len() {
            return false;
        }
        out[cell] += 1;
    }
    true
}

/// Linear-interpolation (type-7) percentile interval of resampled replicates.
///
/// `None` when fewer than two replicates, a non-finite replicate, or an
/// invalid level leave no interval to publish; callers withhold the interval
/// with a reason instead of publishing a non-finite pair.
#[must_use]
pub fn percentile_interval(values: &[f64], level: f64) -> Option<(f64, f64)> {
    if values.len() < 2
        || !level.is_finite()
        || level <= 0.0
        || level >= 1.0
        || values.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, level, QuantileRule::Interpolated);
    (lower.is_finite() && upper.is_finite()).then_some((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interval_is_pointwise_and_ordered() {
        let (lo, hi) = percentile_interval(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.5).unwrap();
        // Type 7 on five points: p·(n − 1) = 1 and 3.
        assert!((lo - 1.0).abs() < 1e-12 && (hi - 3.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_interval_withholds_non_finite_and_short_columns() {
        assert_eq!(percentile_interval(&[0.5], 0.95), None);
        assert_eq!(percentile_interval(&[0.5, f64::NAN, 0.7], 0.95), None);
        assert_eq!(percentile_interval(&[0.5, 0.7], 1.0), None);
    }

    #[test]
    fn bayesian_draw_failure_fraction_withholds_the_interval() {
        let policy = ReplicatePolicy::BOOTSTRAP;
        assert_eq!(policy.decide(10, 0, 10), Err("bootstrap_failure_fraction"));
        assert_eq!(policy.decide(10, 4, 6), Err("bootstrap_failure_fraction"));
        assert_eq!(policy.decide(10, 6, 4), Ok(()));
    }

    #[test]
    fn transport_streams_keep_datasets_and_draws_in_disjoint_fields() {
        let a = TransportStream::ZPosterior.index(0, 1).unwrap();
        let b = TransportStream::ZPosterior.index(1, 1).unwrap();
        let c = TransportStream::ZPosterior.index(1, 0).unwrap();
        assert!(a != b && b != c && a != c);
        assert_eq!(TransportStream::Bootstrap.index(3, 7).unwrap(), (3 << 32) | 7);
        assert!(TransportStream::Bootstrap.index(1 << 16, 0).is_err());
        assert_ne!(
            TransportStream::BayesianGrid.index(0, 5).unwrap(),
            TransportStream::ZPosterior.index(0, 5).unwrap()
        );
    }

    #[test]
    fn identical_count_tables_of_different_datasets_draw_different_posteriors() {
        let ctx = ExecutionContext::for_tests(11);
        let counts = [20u64, 5, 10, 15];
        let draw = |dataset: usize| {
            let mut rng = ctx.rng.stream_for(
                StreamDomain::Transport,
                TransportStream::ZPosterior.index(dataset, 0).unwrap(),
            );
            dirichlet_posterior_probabilities(&counts, 0.0, &mut rng).unwrap()
        };
        assert_ne!(draw(0), draw(1));
        assert_ne!(draw(4), draw(5));
        assert_eq!(draw(1), draw(1));
    }

    #[test]
    fn resampled_counts_match_the_expanded_row_bootstrap() {
        let counts = [3u64, 0, 2, 5];
        let ctx = ExecutionContext::for_tests(5);
        let mut indexes = Vec::new();
        let mut out = vec![0u64; 4];
        let mut rng = ctx.rng.stream_for(StreamDomain::Transport, 9);
        assert!(resample_cited_counts(&counts, &mut rng, &mut indexes, &mut out));
        assert_eq!(out.iter().sum::<u64>(), 10);
        assert_eq!(out[1], 0);
        // Expanding the rows and drawing the same indexes gives the same table.
        let rows: Vec<usize> = counts
            .iter()
            .enumerate()
            .flat_map(|(cell, count)| std::iter::repeat_n(cell, usize::try_from(*count).unwrap()))
            .collect();
        let mut expected = vec![0u64; 4];
        let mut rng = ctx.rng.stream_for(StreamDomain::Transport, 9);
        let mut drawn = Vec::new();
        fill_resample_indexes(ResamplingPlan::IidBootstrap, 10, &mut rng, &mut drawn).unwrap();
        for index in drawn {
            expected[rows[index as usize]] += 1;
        }
        assert_eq!(out, expected);
        assert!(!resample_cited_counts(&[0, 0], &mut rng, &mut indexes, &mut out[..2]));
    }

    #[test]
    fn replicate_columns_summarize_atoms_and_means() {
        let outcome = VariableId::from_raw(1);
        let mut columns = ReplicateColumns::new(&[outcome]);
        for p in [0.2, 0.4, 0.6, 0.8] {
            let distribution = ExactDistribution {
                outcomes: Arc::from([outcome]),
                atoms: Arc::from([
                    Arc::from([antecedent_core::Value::Int64(0)]),
                    Arc::from([antecedent_core::Value::Int64(1)]),
                ]),
                probabilities: Arc::from([1.0 - p, p]),
                support: Arc::from([]),
            };
            columns.record(&distribution).unwrap();
        }
        assert_eq!(columns.ok, 4);
        let intervals = columns.intervals(0.5).unwrap();
        assert_eq!(intervals.atoms.len(), 2);
        let (_, lower, upper) = intervals.means[0];
        assert!(lower < upper && (0.2..=0.8).contains(&lower));
        assert_eq!(columns.atom_rows().len(), 4);
    }
}
