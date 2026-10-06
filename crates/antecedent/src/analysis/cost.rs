//! Planning-time cost counts for prepared single and batch plans.
//!
//! [`CostEstimate`] counts what a plan will do (nuisance fits, cross-fit folds, bootstrap
//! replicates, the design matrix it materializes) from the frozen configuration. It is a
//! **planning hint, not a runtime guarantee**: the counts are exact for the configuration
//! they describe where [`CostEstimate::nuisance_fits_per_pass`] is present and absent where
//! this module has not derived them, and the bootstrap pass count is an upper bound that
//! assumes every replicate refits every nuisance.
//!
//! A wall-clock figure is given only when a named local benchmark backs it. The checked-in
//! `scripts/bench_cost_model.py` times the nuisance fits over a small `(rows, columns)` grid
//! on one machine and writes `parity/cost_model.toml`: for each estimator the coefficients of
//! a monotone model `seconds per counted fit = a + b * rows * columns^2`
//! (`a, b >= 0`), the grid, the benchmark name and a declared machine descriptor.
//! [`CostEstimate::seconds`] is `fits * (a + b * rows * columns^2)` from that file's
//! coefficients, labelled with the benchmark and machine, and stays absent (with the reason in
//! [`CostEstimate::seconds_basis`]) when the file is missing, unreadable, has no declared
//! machine, or has no coefficients for the estimator. The file is read from the path in
//! `ANTECEDENT_COST_MODEL`, else `parity/cost_model.toml` relative to the working directory.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Serialize;

use antecedent_data::TableView;
use antecedent_estimate::{
    AipwAte, DmlAte, DmlScore, DrLearner, FactorizedJointConfig, JointFitPlan, PropensityPenalty,
};

use super::batch::PreparedBatch;
use super::batch_retarget::{BatchRetargetRequest, BatchScores};
use super::builder::DataInput;
use super::execute::Study;
use super::preflight::adjustment_of;
use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{DEFAULT_ESTIMATOR_ID, EstimatorId};

/// Label on every estimate: what it is and is not.
const PLANNING_NOTE: &str = "planning hint, not a runtime guarantee: counts follow the frozen \
     configuration, the bootstrap pass count is an upper bound, and seconds appear only when a \
     named local benchmark file backs them";

/// Why no seconds are given when no model file is present.
const NO_BENCHMARK: &str = "no named local benchmark file (parity/cost_model.toml, or the path \
     in ANTECEDENT_COST_MODEL) is present, so only counts are reported";

/// Environment variable naming the cost-model file.
const COST_MODEL_ENV: &str = "ANTECEDENT_COST_MODEL";

/// Default cost-model path, relative to the working directory.
const COST_MODEL_FILE: &str = "parity/cost_model.toml";

/// `seconds per fit = a + b * rows * columns^2` for one estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FitCoefficients {
    pub(crate) a: f64,
    pub(crate) b: f64,
}

/// A fitted cost model with the benchmark and machine it was measured on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CostModel {
    pub(crate) benchmark: String,
    pub(crate) platform: String,
    pub(crate) cpu_count: u64,
    pub(crate) coefficients: Vec<(String, FitCoefficients)>,
}

/// Where the cost model came from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ModelSource {
    /// No file.
    Absent,
    /// A file that cannot back a time estimate, and why.
    Unusable(String),
    /// A usable model.
    Loaded(CostModel),
}

fn unquote(value: &str) -> Result<String, String> {
    let inner = value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .filter(|v| !v.is_empty() && !v.contains(['"', '\\']))
        .ok_or_else(|| format!("{value} is not a plain non-empty quoted string"))?;
    Ok(inner.to_string())
}

impl CostModel {
    /// Parse the strict TOML subset `scripts/bench_cost_model.py` writes: top-level
    /// `benchmark`, `[machine]` with `platform` and `cpu_count`, and one
    /// `[coefficients."<estimator id>"]` table per estimator with `a` and `b`. Other keys (the
    /// grid, sample counts) are informational. A model without a declared machine, or with a
    /// negative or non-finite coefficient (which would not be monotone), is refused.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let mut section = String::new();
        let mut benchmark = None;
        let mut platform = None;
        let mut cpu_count = None;
        let mut rows: Vec<(String, Option<f64>, Option<f64>)> = Vec::new();
        for (number, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                name.trim().clone_into(&mut section);
                if let Some(id) = section.strip_prefix("coefficients.") {
                    rows.push((id.trim_matches('"').to_string(), None, None));
                }
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .map(|(k, v)| (k.trim(), v.trim()))
                .ok_or_else(|| format!("line {}: not a key = value pair", number + 1))?;
            let float = |v: &str| {
                v.parse::<f64>().map_err(|_| format!("line {}: {v} is not a number", number + 1))
            };
            match (section.as_str(), key) {
                ("", "benchmark") => benchmark = Some(unquote(value)?),
                ("machine", "platform") => platform = Some(unquote(value)?),
                ("machine", "cpu_count") => {
                    cpu_count =
                        Some(value.parse::<u64>().map_err(|_| {
                            format!("line {}: {value} is not a CPU count", number + 1)
                        })?);
                }
                (table, "a" | "b") if table.starts_with("coefficients.") => {
                    if let Some(row) = rows.last_mut() {
                        if key == "a" {
                            row.1 = Some(float(value)?);
                        } else {
                            row.2 = Some(float(value)?);
                        }
                    }
                }
                _ => {}
            }
        }
        let benchmark = benchmark.ok_or("no benchmark name is declared")?;
        let platform = platform.ok_or("no machine platform is declared")?;
        let cpu_count = cpu_count.filter(|&n| n > 0).ok_or("no machine CPU count is declared")?;
        let mut coefficients = Vec::with_capacity(rows.len());
        for (id, a, b) in rows {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(format!("coefficients for {id} need both a and b"));
            };
            if !(a.is_finite() && b.is_finite() && a >= 0.0 && b >= 0.0) {
                return Err(format!("coefficients for {id} must be finite and non-negative"));
            }
            coefficients.push((id, FitCoefficients { a, b }));
        }
        if coefficients.is_empty() {
            return Err("no estimator coefficients are present".to_string());
        }
        Ok(Self { benchmark, platform, cpu_count, coefficients })
    }

    /// `fits * (a + b * rows * columns^2)` for the estimator, when the model covers it.
    #[allow(
        clippy::cast_precision_loss,
        reason = "row, column and fit counts sit far below 2^53 for a planning hint"
    )]
    pub(crate) fn seconds(
        &self,
        estimator: &str,
        rows: usize,
        columns: usize,
        fits: u64,
    ) -> Option<f64> {
        let (_, coefficient) = self.coefficients.iter().find(|(id, _)| id == estimator)?;
        let per_fit = coefficient.a + coefficient.b * rows as f64 * (columns as f64).powi(2);
        Some(fits as f64 * per_fit)
    }

    fn label(&self) -> String {
        format!(
            "planning hint from named local benchmark {}, machine {} ({} cpus)",
            self.benchmark, self.platform, self.cpu_count
        )
    }
}

fn load_model() -> ModelSource {
    let path = std::env::var_os(COST_MODEL_ENV)
        .map_or_else(|| PathBuf::from(COST_MODEL_FILE), PathBuf::from);
    match std::fs::read_to_string(&path) {
        Ok(text) => match CostModel::parse(&text) {
            Ok(model) => ModelSource::Loaded(model),
            Err(why) => ModelSource::Unusable(format!("{}: {why}", path.display())),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ModelSource::Absent,
        Err(error) => ModelSource::Unusable(format!("{}: {error}", path.display())),
    }
}

/// The process-wide cost model, read once.
fn default_model() -> &'static ModelSource {
    static MODEL: OnceLock<ModelSource> = OnceLock::new();
    MODEL.get_or_init(load_model)
}

/// Seconds and their basis for one plan.
fn seconds_of(
    inputs: &CostInputs,
    fits: Option<u64>,
    source: &ModelSource,
) -> (Option<f64>, String) {
    match source {
        ModelSource::Absent => (None, NO_BENCHMARK.to_string()),
        ModelSource::Unusable(why) => {
            (None, format!("the cost-model file cannot back a time estimate ({why})"))
        }
        ModelSource::Loaded(_) if inputs.shape.as_ref().is_some_and(|s| !s.benchmarked) => (
            None,
            format!(
                "no benchmark coefficients cover the route {} (the checked-in benchmark times \
                 plain logistic and OLS fits), so only counts are reported",
                inputs.shape.as_ref().map_or("", |s| s.route.as_str())
            ),
        ),
        ModelSource::Loaded(model) => {
            let (Some(rows), Some(columns), Some(fits)) =
                (inputs.rows, inputs.design_columns, fits)
            else {
                return (
                    None,
                    "rows, design columns or nuisance-fit counts are not derived for this plan, \
                     so no time is estimated"
                        .to_string(),
                );
            };
            match model.seconds(inputs.estimator.as_str(), rows, columns, fits) {
                Some(seconds) => (Some(seconds), model.label()),
                None => (
                    None,
                    format!(
                        "the benchmark {} has no coefficients for estimator {}",
                        model.benchmark,
                        inputs.estimator.as_str()
                    ),
                ),
            }
        }
    }
}

/// The inference the plan will run by default, shown before any work starts.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InferenceDefault {
    /// `frequentist`, or `bayesian(n_draws=..)`.
    pub mode: String,
    /// Bootstrap replicates the frozen plan runs (0 = none).
    pub bootstrap_replicates: u32,
    /// Plain-language warning when the default refits by resampling, else absent.
    pub refit_warning: Option<String>,
}

/// Cost counts for one prepared plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CostEstimate {
    /// Always `true`: this is a planning hint.
    pub planning_hint: bool,
    /// What the estimate is and is not.
    pub note: &'static str,
    /// Estimator id the plan runs.
    pub estimator: String,
    /// The counted fit route, read off the declared configuration: the estimator id, or for a
    /// configured cross-fitted route its penalty grid, independence unit or score (for example
    /// `aipw.ridge_logistic(penalties=3, inner_folds=5)`).
    pub fit_route: String,
    /// Active inference default.
    pub inference: InferenceDefault,
    /// Rows in the retained table (an upper bound on complete-case rows).
    pub rows: Option<usize>,
    /// Columns of the `[1 | Z]` design, when the plan retains its adjustment set.
    pub design_columns: Option<usize>,
    /// Cross-fit folds, for a cross-fitted estimator.
    pub crossfit_folds: Option<u32>,
    /// Distinct first-endpoint cluster labels a cluster-DML declaration folds by (whole
    /// clusters, never rows), when one is declared.
    pub cluster_labels: Option<usize>,
    /// Propensity fits in one pass over the data, when derived for this estimator. For a
    /// ridge-penalized propensity this is an upper bound: the folds times the penalty grid
    /// times the inner folds plus one refit per penalty.
    pub propensity_fits_per_pass: Option<u64>,
    /// Outcome-regression fits in one pass over the data, when derived for this estimator.
    pub outcome_fits_per_pass: Option<u64>,
    /// Total nuisance fits in one pass (`propensity + outcome`), when derived.
    pub nuisance_fits_per_pass: Option<u64>,
    /// Passes: the point estimate plus one per bootstrap replicate (an upper bound, assuming
    /// each replicate refits every nuisance).
    pub passes_upper_bound: u64,
    /// `nuisance_fits_per_pass * passes_upper_bound`, when the per-pass count is derived.
    pub nuisance_fits_upper_bound: Option<u64>,
    /// Refuter replicates. Not enumerated: the suite is named, its replicate counts are not
    /// read here.
    pub refuter_replicates: Option<u32>,
    /// Name of the frozen refute suite.
    pub refute_suite: String,
    /// Bytes of the `[1 | Z]` design matrix (`rows * design_columns * 8`).
    pub design_matrix_bytes: Option<u64>,
    /// Bytes of one fold's training and validation copies, which together partition the
    /// design (cross-fitted estimators only).
    pub fold_copy_bytes: Option<u64>,
    /// Bytes of the batch's shared covariate design, when this plan shares one.
    pub shared_covariate_bytes: Option<u64>,
    /// Estimated seconds. Absent unless a named local benchmark file backs it.
    pub seconds: Option<f64>,
    /// Why `seconds` is absent, or the named benchmark and machine behind it.
    pub seconds_basis: String,
}

/// Cost counts for a prepared batch.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchCostEstimate {
    /// Always `true`: this is a planning hint.
    pub planning_hint: bool,
    /// What the estimate is and is not.
    pub note: &'static str,
    /// Number of claims (plans) in the batch.
    pub claims: usize,
    /// One estimate per plan, in query order.
    pub plans: Vec<CostEstimate>,
    /// Sum of the plans' nuisance-fit upper bounds, or absent if any plan's is. An upper
    /// bound because identical cross-fitted AIPW nuisances are shared across queries.
    pub nuisance_fits_upper_bound: Option<u64>,
    /// Sum of the plans' bootstrap replicates.
    pub bootstrap_replicates_total: u64,
    /// Sum of the plans' design-matrix bytes, or absent if any plan's is.
    pub design_matrix_bytes_total: Option<u64>,
}

/// Frozen quantities a cost count is derived from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CostInputs {
    pub(crate) estimator: EstimatorId,
    pub(crate) inference_mode: String,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) refute_suite: String,
    pub(crate) rows: Option<usize>,
    pub(crate) design_columns: Option<usize>,
    pub(crate) folds: u32,
    pub(crate) shares_covariates: bool,
    /// How a configured cross-fitted route's fits scale with its folds (`None`: the id's
    /// default counts).
    pub(crate) shape: Option<CrossfitShape>,
    pub(crate) cluster_labels: Option<usize>,
}

/// How a cross-fitted route's nuisance fits scale with its folds, read off the declared
/// configuration the route executes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CrossfitShape {
    /// Label of the counted route.
    pub(crate) route: String,
    /// Propensity fits per fold.
    pub(crate) propensity_per_fold: u64,
    /// Outcome-regression fits per fold.
    pub(crate) outcome_per_fold: u64,
    /// Outcome-side fits made once after the folds (the DR-Learner's final-stage regression).
    pub(crate) outcome_once: u64,
    /// Whether the counted fits are the ones the checked-in benchmark times (a plain logistic
    /// propensity and arm-wise OLS); a seconds figure is given only then.
    pub(crate) benchmarked: bool,
}

impl CrossfitShape {
    fn new(
        route: &str,
        propensity_per_fold: u64,
        outcome_per_fold: u64,
        outcome_once: u64,
        benchmarked: bool,
    ) -> Self {
        Self {
            route: route.to_string(),
            propensity_per_fold,
            outcome_per_fold,
            outcome_once,
            benchmarked,
        }
    }
}

/// The counted shape of a configured cross-fitted route and the fold count it declares
/// (`None`: the study's fold count), from the same configuration the route executes.
fn shape_of(spec: &EstimatorSpec) -> Option<(CrossfitShape, Option<u32>)> {
    match spec {
        EstimatorSpec::Aipw(aipw) => Some((aipw_shape(aipw), None)),
        EstimatorSpec::Default(EstimatorId::Aipw) => {
            Some((CrossfitShape::new("aipw", 1, 2, 0, true), None))
        }
        EstimatorSpec::Dml(dml) => Some(dml_shape(dml)),
        EstimatorSpec::Default(EstimatorId::Dml) => Some(dml_shape(&DmlAte::new())),
        EstimatorSpec::DrLearner(dr) => Some(dr_shape(dr)),
        EstimatorSpec::Default(EstimatorId::DrLearner) => Some(dr_shape(&DrLearner::new())),
        _ => None,
    }
}

/// AIPW: one propensity fit and two arm-wise outcome regressions per fold. A ridge or lasso
/// propensity replaces the propensity fit by its penalty selection (grid times inner folds
/// plus refits), and a declared penalized fallback adds the destination's selection and a
/// second pass of the outcome models, as an upper bound (it runs only when the GLM fit fails).
fn aipw_shape(aipw: &AipwAte) -> CrossfitShape {
    let mut route = match aipw.propensity.penalty() {
        PropensityPenalty::None => "aipw".to_string(),
        PropensityPenalty::RidgeLogistic(tuning) => format!(
            "aipw.ridge_logistic(penalties={}, inner_folds={})",
            tuning.planned_penalties(),
            tuning.inner_folds()
        ),
        PropensityPenalty::Lasso(tuning) => format!(
            "aipw.lasso(penalties={}, inner_folds={})",
            tuning.planned_penalties(),
            tuning.inner_folds()
        ),
    };
    let reruns = aipw.propensity.reruns_on_fallback();
    if reruns {
        route.push_str(".glm_fallback");
    }
    if let Some(cluster) = aipw.cluster_dml {
        route = format!("{route}.cluster_dml(unit={})", cluster.independence_unit().as_str());
    }
    CrossfitShape {
        route,
        propensity_per_fold: aipw.propensity.planned_fits_per_fold(),
        outcome_per_fold: 2 * (1 + u64::from(reruns)),
        outcome_once: 0,
        benchmarked: !aipw.propensity.is_penalized() && !reruns,
    }
}

fn folds_of(folds: usize) -> u32 {
    u32::try_from(folds).unwrap_or(u32::MAX)
}

/// DML: the AIPW score fits one propensity and two arm-wise outcome models per fold, the
/// partially linear score one outcome and one treatment regression per fold.
fn dml_shape(dml: &DmlAte) -> (CrossfitShape, Option<u32>) {
    let shape = match dml.score {
        DmlScore::Aipw => CrossfitShape::new("dml(score=aipw)", 1, 2, 0, false),
        DmlScore::PartiallyLinear => {
            CrossfitShape::new("dml(score=partially_linear)", 1, 1, 0, false)
        }
    };
    (shape, Some(folds_of(dml.folds)))
}

/// DR-Learner: the cross-fitted AIPW nuisances (one propensity, two outcome models per fold)
/// and one final-stage regression of the pseudo-outcome.
fn dr_shape(dr: &DrLearner) -> (CrossfitShape, Option<u32>) {
    (CrossfitShape::new("dr_learner", 1, 2, 1, false), Some(folds_of(dr.folds)))
}

/// Nuisance fits per pass for an estimator whose count is derived, as
/// `(propensity, outcome, crossfit)`; `None` where it is not.
///
/// Derived from the estimators' fit loops: a cross-fitted route fits its shape's propensity
/// and outcome models in each fold (AIPW one propensity model and two outcome regressions, a
/// ridge-penalized propensity its penalty selection's fits instead of one); linear and GLM
/// adjustment fit one outcome model; propensity weighting, matching and stratification fit
/// one propensity model; distance matching fits none.
fn fits_per_pass(inputs: &CostInputs) -> Option<(u64, u64, bool)> {
    let folds = u64::from(inputs.folds);
    if let Some(shape) = &inputs.shape {
        return Some((
            shape.propensity_per_fold.saturating_mul(folds),
            shape.outcome_per_fold.saturating_mul(folds).saturating_add(shape.outcome_once),
            true,
        ));
    }
    match inputs.estimator {
        EstimatorId::Aipw => Some((folds, 2 * folds, true)),
        EstimatorId::LinearAdjustmentAte | EstimatorId::GlmAdjustment => Some((0, 1, false)),
        EstimatorId::PropensityWeighting
        | EstimatorId::PropensityMatching
        | EstimatorId::PropensityStratification => Some((1, 0, false)),
        EstimatorId::DistanceMatching => Some((0, 0, false)),
        _ => None,
    }
}

fn bytes(rows: Option<usize>, columns: Option<usize>) -> Option<u64> {
    let cells = rows?.checked_mul(columns?)?;
    u64::try_from(cells).ok()?.checked_mul(8)
}

/// Derive the counts and, when `source` holds a usable model, the seconds. Pure, so
/// monotonicity in folds, replicates and claims is testable without a study or a file.
pub(crate) fn cost_of(inputs: &CostInputs, source: &ModelSource) -> CostEstimate {
    let counts = fits_per_pass(inputs);
    let passes = 1 + u64::from(inputs.bootstrap_replicates);
    let nuisance_per_pass = counts.map(|(p, o, _)| p + o);
    let crossfit = counts.is_some_and(|(_, _, crossfit)| crossfit);
    let design_matrix_bytes = bytes(inputs.rows, inputs.design_columns);
    let nuisance_fits_upper_bound = nuisance_per_pass.and_then(|n| n.checked_mul(passes));
    let (seconds, seconds_basis) = seconds_of(inputs, nuisance_fits_upper_bound, source);
    CostEstimate {
        planning_hint: true,
        note: PLANNING_NOTE,
        estimator: inputs.estimator.as_str().to_string(),
        fit_route: inputs
            .shape
            .as_ref()
            .map_or_else(|| inputs.estimator.as_str().to_string(), |shape| shape.route.clone()),
        inference: InferenceDefault {
            mode: inputs.inference_mode.clone(),
            bootstrap_replicates: inputs.bootstrap_replicates,
            refit_warning: (inputs.bootstrap_replicates > 0).then(|| {
                format!(
                    "the active default resamples rows and re-runs the estimator \
                     {} times; as a planning upper bound each replicate refits every nuisance",
                    inputs.bootstrap_replicates
                )
            }),
        },
        rows: inputs.rows,
        design_columns: inputs.design_columns,
        crossfit_folds: crossfit.then_some(inputs.folds),
        cluster_labels: inputs.cluster_labels,
        propensity_fits_per_pass: counts.map(|(p, _, _)| p),
        outcome_fits_per_pass: counts.map(|(_, o, _)| o),
        nuisance_fits_per_pass: nuisance_per_pass,
        passes_upper_bound: passes,
        nuisance_fits_upper_bound,
        refuter_replicates: None,
        refute_suite: inputs.refute_suite.clone(),
        design_matrix_bytes,
        fold_copy_bytes: crossfit.then_some(design_matrix_bytes).flatten(),
        shared_covariate_bytes: inputs.shares_covariates.then_some(design_matrix_bytes).flatten(),
        seconds,
        seconds_basis,
    }
}

fn inputs_of(study: &Study) -> CostInputs {
    let estimator = study
        .estimator_spec
        .as_ref()
        .map(crate::estimator_spec::EstimatorSpec::id)
        .or(study.estimator)
        .unwrap_or(DEFAULT_ESTIMATOR_ID);
    let rows = match &study.data {
        DataInput::Tabular(data) => Some(data.row_count()),
        _ => None,
    };
    let shared = study.shared_batch_design.as_deref();
    let default_spec = study.estimator.map(EstimatorSpec::Default);
    let (shape, declared_folds) =
        study.estimator_spec.as_ref().or(default_spec.as_ref()).and_then(shape_of).unzip();
    let cluster_labels = match study.estimator_spec.as_ref() {
        Some(EstimatorSpec::Aipw(aipw)) if aipw.cluster_dml.is_some() => aipw
            .cluster_ids
            .as_deref()
            .map(|labels| labels.iter().collect::<std::collections::BTreeSet<_>>().len()),
        _ => None,
    };
    CostInputs {
        estimator,
        inference_mode: match &study.inference {
            InferenceMode::Frequentist => "frequentist".to_string(),
            InferenceMode::Bayesian(config) => format!("bayesian(n_draws={})", config.n_draws),
        },
        bootstrap_replicates: study.bootstrap_replicates,
        refute_suite: study.refute.diagnostic_label().to_string(),
        rows,
        design_columns: adjustment_of(study).map(|set| set.len() + 1),
        folds: declared_folds.flatten().unwrap_or_else(|| {
            shared.map_or_else(
                || u32::try_from(antecedent_estimate::DEFAULT_AIPW_FOLDS).unwrap_or(5),
                |design| design.n_folds,
            )
        }),
        shares_covariates: shared.is_some_and(|design| design.covariate.is_some()),
        shape,
        cluster_labels,
    }
}

impl PreparedStudy {
    /// Planning-time cost counts for this plan: nuisance fits, folds, bootstrap replicates,
    /// the design matrix it materializes, and the active inference default.
    ///
    /// A planning hint, not a runtime guarantee. Counts always; seconds only when the named
    /// local benchmark file (`parity/cost_model.toml`) backs them (see the module docs).
    ///
    /// # Errors
    ///
    /// Never today; the `Result` keeps the signature stable for plans whose cost cannot be
    /// stated.
    pub fn estimate_cost(&self) -> Result<CostEstimate, CausalError> {
        Ok(cost_of(&inputs_of(self.study()), default_model()))
    }
}

impl PreparedBatch {
    /// Planning-time cost counts for every plan and their totals (see
    /// [`PreparedStudy::estimate_cost`]). The fit total is an upper bound: identical
    /// cross-fitted AIPW nuisances are shared across queries and fit once.
    ///
    /// # Errors
    ///
    /// Never today; see [`PreparedStudy::estimate_cost`].
    pub fn estimate_cost(&self) -> Result<BatchCostEstimate, CausalError> {
        let plans: Vec<CostEstimate> =
            self.plans().iter().map(PreparedStudy::estimate_cost).collect::<Result<_, _>>()?;
        Ok(total_cost(plans))
    }
}

fn total_cost(plans: Vec<CostEstimate>) -> BatchCostEstimate {
    let sum = |pick: fn(&CostEstimate) -> Option<u64>| {
        plans.iter().try_fold(0u64, |acc, plan| acc.checked_add(pick(plan)?))
    };
    BatchCostEstimate {
        planning_hint: true,
        note: PLANNING_NOTE,
        claims: plans.len(),
        nuisance_fits_upper_bound: sum(|p| p.nuisance_fits_upper_bound),
        bootstrap_replicates_total: plans
            .iter()
            .map(|p| u64::from(p.inference.bootstrap_replicates))
            .sum(),
        design_matrix_bytes_total: sum(|p| p.design_matrix_bytes),
        plans,
    }
}

/// Why no seconds are given for a route the benchmark does not cover.
const NO_ROUTE_COEFFICIENTS: &str = "no cost-model coefficients cover this route (parity/cost_model.toml \
     times plain linear adjustment and AIPW fits only), so only counts are reported";

/// Cost counts for a batch retarget request.
///
/// A retarget reweights the retained score tables: it fits no nuisance model, so the fit
/// count is zero by construction. What it costs is reading the scores once per claim and
/// forming the Gram matrix of the weighted influence values. A planning hint, not a runtime
/// guarantee.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RetargetCostEstimate {
    /// Always `true`: this is a planning hint.
    pub planning_hint: bool,
    /// What the estimate is and is not.
    pub note: &'static str,
    /// Claims the request retargets.
    pub claims: usize,
    /// Contrasts the request forms over the claims.
    pub contrasts: usize,
    /// Rows of the common snapshot the claims reweight, absent when the scores share none.
    pub snapshot_rows: Option<usize>,
    /// Always `0`: scores are reused, nothing is refit.
    pub nuisance_fits: u64,
    /// `claims * rows`: weighted score sums, one pass of the snapshot per claim.
    pub weighted_score_reads: Option<u64>,
    /// `claims * (claims + 1) / 2 * rows`: products of the weighted influence values in the
    /// upper triangle of the family covariance.
    pub covariance_products: Option<u64>,
    /// Estimated seconds: always absent, no benchmark covers a retarget.
    pub seconds: Option<f64>,
    /// Why `seconds` is absent.
    pub seconds_basis: String,
}

impl BatchScores {
    /// Planning-time cost counts of retargeting these scores under `request`: zero nuisance
    /// fits (the scores are reused) and the claim, contrast and row counts the reweighting
    /// and the family covariance read.
    ///
    /// A planning hint, not a runtime guarantee: the request is not validated here, so a
    /// request the retarget refuses is still counted as declared.
    #[must_use]
    pub fn estimate_retarget_cost(&self, request: &BatchRetargetRequest) -> RetargetCostEstimate {
        let claims = request.claims.len();
        let snapshot_rows = self.common_rows().ok().flatten().map(|rows| rows.len());
        let pairs = u64::try_from(claims * (claims + 1) / 2).ok();
        let rows = snapshot_rows.and_then(|rows| u64::try_from(rows).ok());
        RetargetCostEstimate {
            planning_hint: true,
            note: PLANNING_NOTE,
            claims,
            contrasts: request.contrasts.len(),
            snapshot_rows,
            nuisance_fits: 0,
            weighted_score_reads: rows
                .and_then(|rows| u64::try_from(claims).ok()?.checked_mul(rows)),
            covariance_products: rows.and_then(|rows| pairs?.checked_mul(rows)),
            seconds: None,
            seconds_basis: NO_ROUTE_COEFFICIENTS.to_string(),
        }
    }
}

/// Cost counts for one factorized joint-cell fit, from the declared configuration.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointCellCostEstimate {
    /// Always `true`: this is a planning hint.
    pub planning_hint: bool,
    /// What the estimate is and is not.
    pub note: &'static str,
    /// The counted route: the ridge penalty grid, or the declared learner.
    pub fit_route: String,
    /// Treatment components of the joint cell.
    pub components: usize,
    /// Cells of the family (`2^components`).
    pub cells: u64,
    /// Orderings the family is estimated under.
    pub orderings: u64,
    /// Cross-fit folds.
    pub folds: u64,
    /// Distinct prefix-stratum conditionals of one fold across the orderings.
    pub conditional_strata: u64,
    /// Conditional fits (`folds * conditional_strata`), an upper bound: an empty or
    /// one-class stratum is not fit.
    pub conditional_fits: u64,
    /// Propensity fits: one per conditional under a declared learner, the penalty selection's
    /// fits per conditional on the ridge route (an upper bound).
    pub propensity_fits: u64,
    /// Per-cell outcome models (`cells * folds`), an upper bound.
    pub outcome_fits: u64,
    /// `propensity_fits + outcome_fits`.
    pub nuisance_fits: u64,
    /// Rows the fit would read, when the caller states them.
    pub rows: Option<usize>,
    /// Estimated seconds: always absent, no benchmark covers the route.
    pub seconds: Option<f64>,
    /// Why `seconds` is absent.
    pub seconds_basis: String,
}

/// Planning-time cost counts of a factorized joint-cell fit with this configuration,
/// `components` treatment components and these `orderings` (see
/// [`antecedent_estimate::fit_factorized_joint_cells`]). The counts come from
/// [`FactorizedJointConfig::planned_fits`], the same declaration the fit executes.
///
/// A planning hint, not a runtime guarantee.
///
/// # Errors
///
/// The refusal the fit itself raises for an invalid configuration or ordering list: no cost
/// is stated for a declaration the route refuses before any fit.
pub fn estimate_joint_cell_cost(
    config: &FactorizedJointConfig,
    components: usize,
    orderings: &[Vec<usize>],
    rows: Option<usize>,
) -> Result<JointCellCostEstimate, CausalError> {
    let JointFitPlan {
        folds,
        cells,
        orderings: ordering_count,
        conditional_strata,
        conditional_fits,
        propensity_fits,
        outcome_fits,
    } = config.planned_fits(components, orderings)?;
    let fit_route = match config.learner {
        Some(spec) => format!("joint_cells.learner({})", spec.identity()),
        None => format!(
            "joint_cells.ridge_logistic(penalties={}, inner_folds={})",
            config.tuning.planned_penalties(),
            config.tuning.inner_folds()
        ),
    };
    Ok(JointCellCostEstimate {
        planning_hint: true,
        note: PLANNING_NOTE,
        fit_route,
        components,
        cells,
        orderings: ordering_count,
        folds,
        conditional_strata,
        conditional_fits,
        propensity_fits,
        outcome_fits,
        nuisance_fits: propensity_fits.saturating_add(outcome_fits),
        rows,
        seconds: None,
        seconds_basis: NO_ROUTE_COEFFICIENTS.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use antecedent_estimate::{
        AipwAte, ClusterDml, DmlAte, DmlScore, DrLearner, NuisanceFallback, PropensityNuisance,
        RidgeTuning,
    };

    use super::{CostInputs, CostModel, FitCoefficients, ModelSource, shape_of, total_cost};
    use crate::estimator_spec::EstimatorSpec;
    use crate::strategy_table::EstimatorId;

    fn cost_of(inputs: &CostInputs) -> super::CostEstimate {
        super::cost_of(inputs, &ModelSource::Absent)
    }

    fn inputs(estimator: EstimatorId, folds: u32, bootstrap: u32) -> CostInputs {
        CostInputs {
            estimator,
            inference_mode: "frequentist".to_string(),
            bootstrap_replicates: bootstrap,
            refute_suite: "none".to_string(),
            rows: Some(1000),
            design_columns: Some(11),
            folds,
            shares_covariates: false,
            shape: None,
            cluster_labels: None,
        }
    }

    /// Cross-fitted AIPW fits one propensity and two outcome models per fold, and every
    /// bootstrap replicate adds a pass: the hand count for 5 folds and 199 replicates is
    /// `5 * 3 * 200`.
    #[test]
    fn aipw_counts_match_the_hand_derivation() {
        let cost = cost_of(&inputs(EstimatorId::Aipw, 5, 199));
        assert_eq!(cost.propensity_fits_per_pass, Some(5));
        assert_eq!(cost.outcome_fits_per_pass, Some(10));
        assert_eq!(cost.nuisance_fits_per_pass, Some(15));
        assert_eq!(cost.passes_upper_bound, 200);
        assert_eq!(cost.nuisance_fits_upper_bound, Some(3000));
        assert_eq!(cost.crossfit_folds, Some(5));
        assert_eq!(cost.design_matrix_bytes, Some(1000 * 11 * 8));
        assert_eq!(cost.fold_copy_bytes, cost.design_matrix_bytes);
        assert!(cost.inference.refit_warning.is_some());
        assert!(cost.planning_hint);
    }

    /// The count never falls as folds, replicates or claims grow.
    #[test]
    fn cost_is_monotone_in_folds_replicates_and_claims() {
        let fits = |folds, bootstrap| {
            cost_of(&inputs(EstimatorId::Aipw, folds, bootstrap)).nuisance_fits_upper_bound.unwrap()
        };
        assert!(fits(2, 0) < fits(5, 0) && fits(5, 0) < fits(10, 0));
        assert!(fits(5, 0) < fits(5, 1) && fits(5, 1) < fits(5, 199));
        let one = total_cost(vec![cost_of(&inputs(EstimatorId::Aipw, 5, 10))]);
        let two = total_cost(vec![
            cost_of(&inputs(EstimatorId::Aipw, 5, 10)),
            cost_of(&inputs(EstimatorId::Aipw, 5, 10)),
        ]);
        let three = total_cost(vec![
            cost_of(&inputs(EstimatorId::Aipw, 5, 10)),
            cost_of(&inputs(EstimatorId::Aipw, 5, 10)),
            cost_of(&inputs(EstimatorId::Aipw, 5, 10)),
        ]);
        assert!(one.nuisance_fits_upper_bound < two.nuisance_fits_upper_bound);
        assert!(two.nuisance_fits_upper_bound < three.nuisance_fits_upper_bound);
        assert_eq!(three.claims, 3);
        assert_eq!(three.bootstrap_replicates_total, 30);
    }

    /// Estimators whose fit count is not derived report it absent, never zero, and no
    /// estimate invents seconds.
    #[test]
    fn underived_counts_are_absent_and_no_seconds_are_invented() {
        let cost = cost_of(&inputs(EstimatorId::IvWald, 5, 0));
        assert_eq!(cost.nuisance_fits_per_pass, None);
        assert_eq!(cost.nuisance_fits_upper_bound, None);
        assert_eq!(cost.crossfit_folds, None);
        assert!(cost.seconds.is_none() && !cost.seconds_basis.is_empty());
        assert!(cost.inference.refit_warning.is_none());
        let batch = total_cost(vec![cost, cost_of(&inputs(EstimatorId::Aipw, 5, 0))]);
        assert_eq!(batch.nuisance_fits_upper_bound, None);
    }

    /// Linear adjustment fits one outcome model; a shared covariate design is reported only
    /// for a plan that shares one.
    #[test]
    fn simple_estimators_and_shared_design_bytes() {
        let mut shared = inputs(EstimatorId::LinearAdjustmentAte, 5, 0);
        assert_eq!(cost_of(&shared).nuisance_fits_upper_bound, Some(1));
        assert_eq!(cost_of(&shared).shared_covariate_bytes, None);
        shared.shares_covariates = true;
        assert_eq!(cost_of(&shared).shared_covariate_bytes, Some(1000 * 11 * 8));
    }

    fn model(a: f64, b: f64) -> CostModel {
        CostModel {
            benchmark: "synthetic".to_string(),
            platform: "test-platform".to_string(),
            cpu_count: 4,
            coefficients: vec![("aipw".to_string(), FitCoefficients { a, b })],
        }
    }

    /// Seconds are `fits * (a + b * rows * columns^2)`, checked by hand: AIPW with 5 folds
    /// and 9 replicates is 150 fits, and 150 * (0.01 + 1e-9 * 1000 * 121) = 150 * 0.01012...
    #[test]
    fn model_seconds_follow_the_hand_arithmetic_and_are_labelled() {
        let source = ModelSource::Loaded(model(0.01, 1e-9));
        let cost = super::cost_of(&inputs(EstimatorId::Aipw, 5, 9), &source);
        let expected = 150.0 * (0.01 + 1e-9 * 1000.0 * 121.0);
        assert!((cost.seconds.unwrap() - expected).abs() < 1e-12, "{:?}", cost.seconds);
        assert_eq!(
            cost.seconds_basis,
            "planning hint from named local benchmark synthetic, machine test-platform (4 cpus)"
        );
    }

    /// The model is monotone in folds, replicates and rows, and in columns.
    #[test]
    fn model_seconds_are_monotone_in_folds_replicates_rows_and_columns() {
        let source = ModelSource::Loaded(model(0.001, 1e-9));
        let seconds = |folds: u32, bootstrap: u32, rows: usize, columns: usize| {
            let mut input = inputs(EstimatorId::Aipw, folds, bootstrap);
            input.rows = Some(rows);
            input.design_columns = Some(columns);
            super::cost_of(&input, &source).seconds.unwrap()
        };
        assert!(seconds(2, 0, 1000, 11) < seconds(5, 0, 1000, 11));
        assert!(seconds(5, 0, 1000, 11) < seconds(10, 0, 1000, 11));
        assert!(seconds(5, 0, 1000, 11) < seconds(5, 1, 1000, 11));
        assert!(seconds(5, 1, 1000, 11) < seconds(5, 199, 1000, 11));
        assert!(seconds(5, 10, 500, 11) < seconds(5, 10, 1000, 11));
        assert!(seconds(5, 10, 1000, 6) < seconds(5, 10, 1000, 11));
    }

    /// A missing file, an unusable file, an estimator the benchmark does not cover and a plan
    /// without derived counts all leave `seconds` absent with a stated reason.
    #[test]
    fn seconds_stay_absent_without_a_usable_model() {
        let absent = cost_of(&inputs(EstimatorId::Aipw, 5, 0));
        assert!(absent.seconds.is_none() && absent.seconds_basis.contains("only counts"));

        let unusable = super::cost_of(
            &inputs(EstimatorId::Aipw, 5, 0),
            &ModelSource::Unusable("bad".to_string()),
        );
        assert!(unusable.seconds.is_none() && unusable.seconds_basis.contains("bad"));

        let loaded = ModelSource::Loaded(model(0.01, 1e-9));
        let uncovered = super::cost_of(&inputs(EstimatorId::LinearAdjustmentAte, 5, 0), &loaded);
        assert!(uncovered.seconds.is_none());
        assert!(uncovered.seconds_basis.contains("no coefficients"));

        let underived = super::cost_of(&inputs(EstimatorId::IvWald, 5, 0), &loaded);
        assert!(underived.seconds.is_none());
        assert!(underived.seconds_basis.contains("not derived"));
    }

    const FILE: &str = r#"
# written by scripts/bench_cost_model.py
benchmark = "scripts/bench_cost_model.py"
model = "seconds per fit = a + b * rows * columns^2"

[machine]
platform = "macOS-15-arm64"
cpu_count = 10

[grid]
rows = [200, 400]
columns = [3, 6]

[coefficients."aipw"]
a = 2.5e-04
b = 3.0e-10
samples = 12

[coefficients."linear.adjustment.ate"]
a = 1.0e-05
b = 1.0e-10
"#;

    /// The checked-in file format parses, and a model needs a declared machine and
    /// non-negative finite coefficients.
    #[test]
    fn the_file_format_parses_and_refuses_undeclared_or_non_monotone_models() {
        let parsed = CostModel::parse(FILE).unwrap();
        assert_eq!(parsed.benchmark, "scripts/bench_cost_model.py");
        assert_eq!(parsed.platform, "macOS-15-arm64");
        assert_eq!(parsed.cpu_count, 10);
        assert_eq!(parsed.coefficients.len(), 2);
        assert_eq!(
            parsed.coefficients[0],
            ("aipw".to_string(), FitCoefficients { a: 2.5e-4, b: 3.0e-10 })
        );
        assert_eq!(parsed.coefficients[1].0, "linear.adjustment.ate");
        let seconds = parsed.seconds("aipw", 400, 6, 10).unwrap();
        assert!((seconds - 10.0 * (2.5e-4 + 3.0e-10 * 400.0 * 36.0)).abs() < 1e-15);

        let no_machine =
            FILE.replace("[machine]\nplatform = \"macOS-15-arm64\"\ncpu_count = 10\n", "");
        assert!(CostModel::parse(&no_machine).unwrap_err().contains("platform"));
        let zero_cpus = FILE.replace("cpu_count = 10", "cpu_count = 0");
        assert!(CostModel::parse(&zero_cpus).unwrap_err().contains("CPU count"));
        let negative = FILE.replace("b = 3.0e-10", "b = -3.0e-10");
        assert!(CostModel::parse(&negative).unwrap_err().contains("non-negative"));
        assert!(CostModel::parse("").is_err());
    }
    // ---- configured cross-fitted routes: counts follow the declared configuration ----------

    fn routed(spec: &EstimatorSpec, bootstrap: u32) -> CostInputs {
        let (shape, folds) = shape_of(spec).unzip();
        CostInputs { shape, folds: folds.flatten().unwrap_or(5), ..inputs(spec.id(), 5, bootstrap) }
    }

    fn aipw_with(propensity: PropensityNuisance) -> EstimatorSpec {
        EstimatorSpec::Aipw(Box::new(AipwAte {
            propensity,
            bootstrap_replicates: 0,
            ..AipwAte::new()
        }))
    }

    fn ridge(grid: &[f64], inner_folds: usize) -> PropensityNuisance {
        PropensityNuisance::ridge_logistic(RidgeTuning::new(grid, inner_folds).unwrap())
    }

    /// A ridge grid of 3 penalties and 4 inner folds fits `3 * (4 + 1) = 15` propensity models
    /// per outer fold (each penalty on every inner split and once on the whole training set), so
    /// 5 folds give 75 propensity and 10 outcome fits, and 199 refit-bootstrap replicates each
    /// repeat the whole pipeline: `85 * 200`.
    #[test]
    fn penalized_aipw_counts_follow_the_grid_inner_folds_and_replicates() {
        let spec = aipw_with(ridge(&[1.0, 10.0, 100.0], 4));
        let cost = cost_of(&routed(&spec, 0));
        assert_eq!(cost.fit_route, "aipw.ridge_logistic(penalties=3, inner_folds=4)");
        assert_eq!(cost.crossfit_folds, Some(5));
        assert_eq!(cost.propensity_fits_per_pass, Some(75));
        assert_eq!(cost.outcome_fits_per_pass, Some(10));
        assert_eq!(cost.nuisance_fits_upper_bound, Some(85));
        let refit = cost_of(&routed(&spec, 199));
        assert_eq!(refit.passes_upper_bound, 200);
        assert_eq!(refit.nuisance_fits_upper_bound, Some(85 * 200));
    }

    /// The count never falls as the grid, the inner folds, the outer folds or the replicates
    /// grow.
    #[test]
    fn penalized_counts_are_monotone_in_grid_inner_folds_folds_and_replicates() {
        let fits = |grid: &[f64], inner, bootstrap| {
            cost_of(&routed(&aipw_with(ridge(grid, inner)), bootstrap))
                .nuisance_fits_upper_bound
                .unwrap()
        };
        assert!(fits(&[1.0], 3, 0) < fits(&[1.0, 2.0], 3, 0));
        assert!(fits(&[1.0, 2.0], 3, 0) < fits(&[1.0, 2.0, 4.0], 3, 0));
        assert!(fits(&[1.0, 2.0], 2, 0) < fits(&[1.0, 2.0], 5, 0));
        assert!(fits(&[1.0, 2.0], 5, 0) < fits(&[1.0, 2.0], 5, 1));
        assert!(fits(&[1.0, 2.0], 5, 1) < fits(&[1.0, 2.0], 5, 199));
        let folds = |folds| {
            let mut input = routed(&aipw_with(ridge(&[1.0, 2.0], 3)), 0);
            input.folds = folds;
            cost_of(&input).nuisance_fits_upper_bound.unwrap()
        };
        assert!(folds(2) < folds(5) && folds(5) < folds(10));
    }

    /// A lasso grid is counted like a ridge grid; a declared penalized fallback adds the
    /// destination's selection and a second pass of the outcome models as an upper bound.
    #[test]
    fn lasso_and_fallback_counts_are_derived_from_their_own_grids() {
        // Default lasso: 8 penalties, 5 inner folds: 8 * (5 + 1) = 48 per fold.
        let lasso = cost_of(&routed(&aipw_with(PropensityNuisance::lasso()), 0));
        assert_eq!(lasso.fit_route, "aipw.lasso(penalties=8, inner_folds=5)");
        assert_eq!(lasso.propensity_fits_per_pass, Some(5 * 48));
        assert_eq!(lasso.outcome_fits_per_pass, Some(10));
        let fallback = PropensityNuisance::default().with_fallback(
            NuisanceFallback::RidgeLogistic(RidgeTuning::new(&[1.0, 10.0], 3).unwrap()),
        );
        let cost = cost_of(&routed(&aipw_with(fallback), 0));
        assert_eq!(cost.fit_route, "aipw.glm_fallback");
        // 1 GLM fit + 2 * (3 + 1) fallback fits per fold, and the outcome models twice.
        assert_eq!(cost.propensity_fits_per_pass, Some(5 * 9));
        assert_eq!(cost.outcome_fits_per_pass, Some(20));
        // An ML fallback is closed: it adds no fit.
        let ml = PropensityNuisance::default().with_fallback(NuisanceFallback::Ml);
        assert_eq!(cost_of(&routed(&aipw_with(ml), 0)).propensity_fits_per_pass, Some(5));
    }

    /// Seconds are given only for the fits the benchmark times: plain AIPW and a cluster-DML
    /// declaration (the same logistic and OLS fits per fold), never a penalty selection.
    #[test]
    fn seconds_cover_plain_and_cluster_aipw_but_not_penalized_routes() {
        let source = ModelSource::Loaded(model(0.01, 1e-9));
        let with_model = |spec: &EstimatorSpec| super::cost_of(&routed(spec, 0), &source);
        let plain = with_model(&aipw_with(PropensityNuisance::default()));
        let cluster = with_model(&EstimatorSpec::Aipw(Box::new(AipwAte {
            bootstrap_replicates: 0,
            cluster_ids: Some(vec![0; 40]),
            cluster_dml: Some(ClusterDml::new(10).unwrap()),
            ..AipwAte::new()
        })));
        assert_eq!(cluster.fit_route, "aipw.cluster_dml(unit=cluster)");
        assert_eq!(cluster.nuisance_fits_per_pass, plain.nuisance_fits_per_pass);
        assert_eq!(cluster.seconds, plain.seconds);
        assert!(plain.seconds.is_some());
        let penalized = with_model(&aipw_with(ridge(&[1.0, 10.0], 3)));
        assert!(penalized.seconds.is_none());
        assert!(
            penalized.seconds_basis.contains("aipw.ridge_logistic"),
            "{}",
            penalized.seconds_basis
        );
        assert!(penalized.seconds_basis.contains("only counts"));
        let dyadic = ClusterDml::dyadic(10, 4).unwrap();
        let spec =
            EstimatorSpec::Aipw(Box::new(AipwAte { cluster_dml: Some(dyadic), ..AipwAte::new() }));
        assert_eq!(cost_of(&routed(&spec, 0)).fit_route, "aipw.cluster_dml(unit=dyad)");
    }

    /// DML counts use the estimator's own fold count and score: the AIPW score fits one
    /// propensity and two outcome models per fold, the partially linear score one of each, the
    /// DR-Learner the AIPW nuisances and one final-stage regression.
    #[test]
    fn dml_and_dr_counts_follow_their_folds_and_scores() {
        let dml = |score, folds| {
            EstimatorSpec::Dml(Box::new(DmlAte::new().with_score(score).with_folds(folds)))
        };
        let aipw = cost_of(&routed(&dml(DmlScore::Aipw, 7), 0));
        assert_eq!(aipw.fit_route, "dml(score=aipw)");
        assert_eq!(aipw.crossfit_folds, Some(7));
        assert_eq!(
            (aipw.propensity_fits_per_pass, aipw.outcome_fits_per_pass),
            (Some(7), Some(14))
        );
        let plr = cost_of(&routed(&dml(DmlScore::PartiallyLinear, 7), 0));
        assert_eq!((plr.propensity_fits_per_pass, plr.outcome_fits_per_pass), (Some(7), Some(7)));
        let dr = cost_of(&routed(&EstimatorSpec::DrLearner(Box::new(DrLearner::new())), 0));
        assert_eq!((dr.propensity_fits_per_pass, dr.outcome_fits_per_pass), (Some(5), Some(11)));
        let by_id = cost_of(&routed(&EstimatorSpec::Default(EstimatorId::Dml), 0));
        assert_eq!(by_id.nuisance_fits_per_pass, Some(15));
        let fits = |folds| {
            cost_of(&routed(&dml(DmlScore::Aipw, folds), 0)).nuisance_fits_upper_bound.unwrap()
        };
        assert!(fits(2) < fits(5) && fits(5) < fits(10));
        // No coefficients cover these routes, so no seconds are invented.
        let loaded = ModelSource::Loaded(model(0.01, 1e-9));
        let timed = super::cost_of(&routed(&dml(DmlScore::Aipw, 5), 0), &loaded);
        assert!(timed.seconds.is_none() && timed.seconds_basis.contains("only counts"));
    }
}
