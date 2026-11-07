//! Joint outer bootstrap for catalog-bound empirical transport tables.

mod bayesian;
pub use bayesian::{
    BayesianStatisticalTransportEstimate, BayesianStatisticalTransportOptions,
    evaluate_bayesian_statistical_transport, evaluate_bayesian_statistical_transport_grid,
};
mod bootstrap;
mod cited_tables;
mod replicates;
mod z_draws;
use bootstrap::{BootstrapDraws, outer_bootstrap};
use cited_tables::{AliasPolicy, CitedTables};
use replicates::ReplicateColumns;
use z_draws::{
    TableDrawer, ZDrawSpec, resample_cited_counts, validate_draw_spec, z_draws,
    z_interval_over_draws,
};

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
use serde::{Deserialize, Serialize};

use crate::empirical_table::{
    BayesianLawDrawer, BayesianTransportLawProvider, EmpiricalTableOptions,
    StatisticalTransportInput, assemble_point_laws, assemble_statistical_laws, bound_sample_key,
    dependence_refusal, dirichlet_posterior_probabilities, supported_iid_dependence,
    supported_iid_regimes, validate_dataset_aliases, validate_options,
};
use crate::error::EstimationError;
use crate::transport::{is_support_failure, prepare_exact_transport, refuse_budget, refuse_eval};
use crate::util::ReplicatePolicy;

/// Supported uncertainty method for the empirical-table path.
pub const PERCENTILE_BOOTSTRAP: &str = "percentile_bootstrap";
/// Posterior interval constructor for finite-discrete structural transport.
pub const POSTERIOR_EQUAL_TAIL: &str = "posterior_equal_tail";
/// Why a published z-transport percentile interval is not a coverage claim.
pub const Z_TRANSPORT_INTERVAL_NOT_MEASURED: &str = "estimator_grid_not_measured";
/// Why an interval is withheld when a replicate summary is not a finite number.
pub const INTERVAL_NUMERICAL_FAILURE: &str = "bootstrap_numerical_failure";

/// Why a Bayesian transport posterior publishes, or withholds, its intervals.
///
/// The serialized spelling of each reason is the estimator's own stable string, so artifacts
/// written before the reason was a closed set keep reading; a payload naming any other reason
/// fails at deserialisation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum TransportIntervalReason {
    /// The intervals are published; coverage is measured in the release calibration pass
    /// ([`Z_TRANSPORT_INTERVAL_NOT_MEASURED`]).
    #[serde(rename = "estimator_grid_not_measured")]
    NotMeasured,
    /// The draws exceeded the bootstrap failure budget.
    #[serde(rename = "bootstrap_failure_fraction")]
    BootstrapFailureFraction,
    /// An interval endpoint was not a finite number ([`INTERVAL_NUMERICAL_FAILURE`]).
    #[serde(rename = "bootstrap_numerical_failure")]
    NumericalFailure,
    /// Fewer posterior draws than a nominal interval needs.
    #[serde(rename = "insufficient_bootstrap_replicates")]
    InsufficientDraws,
    /// The evidence's dependence structure supports no resampling.
    #[serde(rename = "transport.unsupported_dependence")]
    UnsupportedDependence,
    /// The catalog binds no estimated regime to draw.
    #[serde(rename = "no_estimated_regime")]
    NoEstimatedRegime,
}

impl TransportIntervalReason {
    /// Every reason, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::NotMeasured,
        Self::BootstrapFailureFraction,
        Self::NumericalFailure,
        Self::InsufficientDraws,
        Self::UnsupportedDependence,
        Self::NoEstimatedRegime,
    ];

    /// The stable string form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotMeasured => Z_TRANSPORT_INTERVAL_NOT_MEASURED,
            Self::BootstrapFailureFraction => "bootstrap_failure_fraction",
            Self::NumericalFailure => INTERVAL_NUMERICAL_FAILURE,
            Self::InsufficientDraws => "insufficient_bootstrap_replicates",
            Self::UnsupportedDependence => "transport.unsupported_dependence",
            Self::NoEstimatedRegime => "no_estimated_regime",
        }
    }

    /// The reason a stable string names, or `None` for a string that is not a reason.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|reason| reason.as_str() == text)
    }

    /// The reason a dependence refusal ([`dependence_refusal`]) names.
    fn from_dependence_refusal(reason: &str) -> Self {
        if reason == Self::NoEstimatedRegime.as_str() {
            Self::NoEstimatedRegime
        } else {
            Self::UnsupportedDependence
        }
    }
}

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

/// Joint bootstrap columns aligned to original replicate IDs.
pub(crate) type TransportMeanReplicates = Arc<[(VariableId, Arc<[f64]>)]>;

/// Recorded interval row. Failed replicates stay visible.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportUncertaintyRow {
    /// Estimator id.
    pub estimator: Arc<str>,
    /// Interval constructor.
    pub method: Arc<str>,
    /// Nominal coverage.
    pub coverage_target: f64,
    /// `pointwise` unless a simultaneous family is separately supported.
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
    /// Supported interval metadata, when published.
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

/// Evaluate the plug-in and, when supported, the joint outer bootstrap.
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
pub(crate) fn evaluate_statistical_transport_grid(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    evaluate_grid(functional, input, requests, limits, options, ctx, PointLaws::Fit)
}

/// Evaluate retained point laws without fitting their original samples again.
///
/// The caller must retain laws fitted to exactly these immutable inputs and
/// settings. This numerical primitive creates no compiler or artifact authority.
/// Bootstrap datasets are still refitted jointly for every outer replicate.
/// # Errors
/// Invalid requests, support failure, cancellation, or exceeded resources.
pub fn evaluate_statistical_transport_grid_with_point_laws(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    point_laws: &antecedent_expr::ExactTransportData,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    evaluate_grid(
        functional,
        input,
        requests,
        limits,
        options,
        ctx,
        PointLaws::Retained(point_laws),
    )
}

/// [`evaluate_statistical_transport_grid_with_point_laws`] for a caller that has already
/// evaluated every request's plan on exactly `point_laws` (its eligibility pass): those
/// distributions are the points' estimates and are not computed a second time. Bootstrap
/// datasets are still refitted jointly for every outer replicate.
///
/// `distributions[i]` must be the exact evaluation of `requests[i]` on `point_laws`.
/// # Errors
/// A length mismatch, invalid requests, cancellation, or exceeded resources.
pub fn evaluate_statistical_transport_grid_with_evaluated_point_laws(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    evaluated: EvaluatedPointLaws<'_>,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    if evaluated.distributions.len() != requests.len() {
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
        PointLaws::Evaluated(evaluated),
    )
}

/// Retained point laws together with their already-evaluated request distributions.
#[derive(Debug)]
pub struct EvaluatedPointLaws<'a> {
    /// The retained point laws the distributions were evaluated on.
    pub point_laws: &'a antecedent_expr::ExactTransportData,
    /// `distributions[i]` is the exact evaluation of `requests[i]` on `point_laws`.
    pub distributions: Vec<ExactDistribution>,
}

/// Where a grid evaluation gets its point laws, and whether their request distributions
/// are already known.
enum PointLaws<'a> {
    /// Fit the laws from the input samples and evaluate every request.
    Fit,
    /// Use retained laws; evaluate every request on them.
    Retained(&'a antecedent_expr::ExactTransportData),
    /// Use retained laws and their evaluated distributions.
    Evaluated(EvaluatedPointLaws<'a>),
}

fn evaluate_grid(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
    source: PointLaws<'_>,
) -> Result<Vec<StatisticalTransportEstimate>, EstimationError> {
    let (point_laws, evaluated) = match source {
        PointLaws::Fit => (None, None),
        PointLaws::Retained(laws) => (Some(laws), None),
        PointLaws::Evaluated(EvaluatedPointLaws { point_laws, distributions }) => {
            (Some(point_laws), Some(distributions))
        }
    };
    if requests.is_empty() {
        return Err(EstimationError::data_msg("empty treatment grid"));
    }
    validate_options(options)?;
    validate_dataset_aliases(functional.catalog(), &input.samples)?;
    let mut budget_ctx = ctx.clone();
    budget_ctx.set_memory_limit(ctx.memory_limit_bytes().map(|n| n / requests.len() as u64));
    let (fit_cost, reserved_bytes) =
        check_statistical_resources(input, functional, options, &budget_ctx)?;
    budget_ctx.set_memory_limit(
        budget_ctx.memory_limit_bytes().map(|limit| limit.saturating_sub(reserved_bytes as u64)),
    );
    let bootstrapping = options.bootstrap_replicates > 0
        && !input.samples.is_empty()
        && supported_iid_dependence(functional.catalog(), &input.samples);
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
    if !supported_iid_dependence(functional.catalog(), &input.samples) {
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
        .and_then(|()| {
            boot.columns.intervals(options.coverage_level).map_err(TransportIntervalReason::as_str)
        });
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
        && supported_iid_dependence(functional.catalog(), &input.samples)
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
        .memory_limit_bytes()
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
    z_interval_over_draws(
        functional,
        data,
        request,
        limits,
        &spec,
        ctx,
        |counts, rng, out, resampled| bootstrap_draw(counts, rng, out, resampled, ctx),
    )
}

/// One iid row-bootstrap draw of a cited count table, as counts and frequencies.
fn bootstrap_draw(
    counts: &[u64],
    rng: &mut antecedent_core::CausalRng,
    probabilities: &mut Vec<f64>,
    resampled: &mut [u64],
    ctx: &ExecutionContext,
) -> Result<bool, EstimationError> {
    if !resample_cited_counts(counts, rng, resampled, ctx)? {
        return Ok(false);
    }
    let total = resampled.iter().sum::<u64>() as f64;
    probabilities.clear();
    probabilities.extend(resampled.iter().map(|count| *count as f64 / total));
    Ok(true)
}

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

/// Pointwise intervals of one request of a joint mz bootstrap.
#[derive(Clone, Debug)]
pub struct MzRequestInterval {
    /// Atom intervals in distribution order.
    pub atom_intervals: Arc<[(f64, f64)]>,
    /// `(outcome, lower, upper)` per outcome mean.
    pub mean_intervals: Arc<[(VariableId, f64, f64)]>,
}

/// Whether the declared sampling of the studies a multi-source formula cites
/// supports a joint bootstrap, or the reason the interval is withheld.
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

/// The spec of the mz joint bootstrap: one row bootstrap per cited
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
/// support a joint bootstrap ([`mz_sampling_dependence`]); and laws that name the
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

/// The law sets the mz joint bootstrap evaluates, one per replicate
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
    bootstrap_law_draws(&tables, data.max_support_rows(), &spec, ctx).map(Ok)
}

/// The law sets of the iid row bootstrap of a single-source z functional's cited
/// count tables, one per replicate (`None` for a replicate whose draw failed).
///
/// Same stream, alias policy and dependence rule as [`nominal_z_transport_interval`];
/// the joint-sensitivity endpoint bootstrap evaluates its exact range on
/// each draw.
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
    if !supported_iid_regimes(functional.catalog(), regimes) {
        return Ok(Err("transport.unsupported_dependence"));
    }
    crate::transport::validate_exact_laws(functional.catalog(), data)
        .map_err(|error| refuse_eval(&error))?;
    let tables = CitedTables::new(functional, data, AliasPolicy::Identical)?
        .map_err(EstimationError::data_msg)?;
    bootstrap_law_draws(&tables, data.max_support_rows(), &spec, ctx).map(Ok)
}

// Draw each cited table once per replicate. Route-specific validation and
// dependence policies are resolved before entering this shared loop.
fn bootstrap_law_draws(
    tables: &CitedTables<'_>,
    max_support_rows: usize,
    spec: &ZDrawSpec,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<antecedent_expr::ExactTransportData>>, EstimationError> {
    let mut drawer = TableDrawer::new(tables, max_support_rows);
    let mut out = Vec::with_capacity(spec.replicates as usize);
    for replicate in 0..spec.replicates {
        crate::transport::refuse_cancelled(ctx, spec.stage)?;
        out.push(drawer.draw(spec, replicate, ctx, &mut |counts, rng, p, resampled| {
            bootstrap_draw(counts, rng, p, resampled, ctx)
        })?);
    }
    Ok(out)
}

/// Joint percentile bootstrap of a multi-source mz functional over `requests`.
///
/// The reported intervals carry a calibration annotation resolved from the construction's
/// calibration basis; calibration never gates them.
/// [`mz_interval_withheld_reason`] / [`mz_sampling_dependence`] report why an interval is
/// withheld for a data or dependence reason. Exact laws, undeclared or unsupported dependence (including shared datasets
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
    let drawn = z_draws(
        functional,
        TableDrawer::new(&tables, data.max_support_rows()),
        requests,
        limits,
        &spec,
        ctx,
        |counts, rng, out, resampled| bootstrap_draw(counts, rng, out, resampled, ctx),
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
            Err(reason) => return Ok(Err(reason.as_str())),
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
        Ok(match dirichlet_posterior_probabilities(counts, prior, rng) {
            Some(drawn) => {
                *out = drawn;
                true
            }
            None => false,
        })
    })
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
    fn interval_reasons_keep_their_stable_strings() {
        assert_eq!(
            TransportIntervalReason::BootstrapFailureFraction.as_str(),
            ReplicatePolicy::BOOTSTRAP.reason
        );
        assert_eq!(
            TransportIntervalReason::InsufficientDraws.as_str(),
            ReplicatePolicy::percentile_floor(0).reason
        );
        for reason in TransportIntervalReason::ALL {
            assert_eq!(TransportIntervalReason::parse(reason.as_str()), Some(reason));
        }
        assert_eq!(TransportIntervalReason::parse("not_a_reason"), None);
        assert_eq!(
            TransportIntervalReason::NotMeasured.as_str(),
            Z_TRANSPORT_INTERVAL_NOT_MEASURED
        );
    }

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
        let mut out = vec![0u64; 4];
        let mut rng = ctx.rng.stream_for(StreamDomain::Transport, 9);
        assert!(resample_cited_counts(&counts, &mut rng, &mut out, &ctx).unwrap());
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
            expected[rows[usize::try_from(index.get()).expect("row id")]] += 1;
        }
        assert_eq!(out, expected);
        assert!(!resample_cited_counts(&[0, 0], &mut rng, &mut out[..2], &ctx).unwrap());
        let cancelled = ExecutionContext::for_tests(5);
        cancelled.cancellation.cancel();
        let error =
            resample_cited_counts(&[usize::MAX as u64], &mut rng, &mut out[..1], &cancelled)
                .unwrap_err();
        assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
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
