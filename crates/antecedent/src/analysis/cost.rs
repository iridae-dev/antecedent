//! Planning-time cost counts for prepared single and batch plans.
//!
//! [`CostEstimate`] counts what a plan will do (nuisance fits, cross-fit folds, bootstrap
//! replicates, the design matrix it materializes) from the frozen configuration. It is a
//! **planning hint, not a runtime guarantee**: the counts are exact for the configuration
//! they describe where [`CostEstimate::nuisance_fits_per_pass`] is present and absent where
//! this module has not derived them, and the bootstrap pass count is an upper bound that
//! assumes every replicate refits every nuisance.
//!
//! No wall-clock figure is given. A time estimate needs a named local benchmark of the same
//! fits, and the repository carries none (the benches under `benches/baselines` measure
//! graph, counterfactual, kernel and Laplace workloads), so [`CostEstimate::seconds`] stays
//! absent and says why.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::Serialize;

use antecedent_data::TableView;

use super::batch::PreparedBatch;
use super::builder::DataInput;
use super::execute::Study;
use super::preflight::adjustment_of;
use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::{DEFAULT_ESTIMATOR_ID, EstimatorId};

/// Label on every estimate: what it is and is not.
const PLANNING_NOTE: &str = "planning hint, not a runtime guarantee: counts follow the frozen \
     configuration, the bootstrap pass count is an upper bound, and no time is estimated \
     because no named local benchmark of these fits exists in the repository";

/// Why no seconds are given.
const NO_BENCHMARK: &str = "no named local benchmark covers this estimator's fits, so only \
     counts are reported";

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
    /// Active inference default.
    pub inference: InferenceDefault,
    /// Rows in the retained table (an upper bound on complete-case rows).
    pub rows: Option<usize>,
    /// Columns of the `[1 | Z]` design, when the plan retains its adjustment set.
    pub design_columns: Option<usize>,
    /// Cross-fit folds, for a cross-fitted estimator.
    pub crossfit_folds: Option<u32>,
    /// Propensity fits in one pass over the data, when derived for this estimator.
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
    /// Estimated seconds. Absent unless a named local benchmark backs it.
    pub seconds: Option<f64>,
    /// Why `seconds` is absent (or the named benchmark behind it).
    pub seconds_basis: &'static str,
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
}

/// Nuisance fits per pass for an estimator whose count is derived, as
/// `(propensity, outcome, crossfit)`; `None` where it is not.
///
/// Derived from the estimators' fit loops: cross-fitted AIPW fits one propensity model and
/// two outcome regressions (one per arm) in each fold; linear and GLM adjustment fit one
/// outcome model; propensity weighting, matching and stratification fit one propensity
/// model; distance matching fits none.
fn fits_per_pass(estimator: EstimatorId, folds: u32) -> Option<(u64, u64, bool)> {
    let folds = u64::from(folds);
    match estimator {
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

/// Derive the counts. Pure, so monotonicity in folds, replicates and claims is testable
/// without a study.
pub(crate) fn cost_of(inputs: &CostInputs) -> CostEstimate {
    let counts = fits_per_pass(inputs.estimator, inputs.folds);
    let passes = 1 + u64::from(inputs.bootstrap_replicates);
    let nuisance_per_pass = counts.map(|(p, o, _)| p + o);
    let crossfit = counts.is_some_and(|(_, _, crossfit)| crossfit);
    let design_matrix_bytes = bytes(inputs.rows, inputs.design_columns);
    CostEstimate {
        planning_hint: true,
        note: PLANNING_NOTE,
        estimator: inputs.estimator.as_str().to_string(),
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
        propensity_fits_per_pass: counts.map(|(p, _, _)| p),
        outcome_fits_per_pass: counts.map(|(_, o, _)| o),
        nuisance_fits_per_pass: nuisance_per_pass,
        passes_upper_bound: passes,
        nuisance_fits_upper_bound: nuisance_per_pass.and_then(|n| n.checked_mul(passes)),
        refuter_replicates: None,
        refute_suite: inputs.refute_suite.clone(),
        design_matrix_bytes,
        fold_copy_bytes: crossfit.then_some(design_matrix_bytes).flatten(),
        shared_covariate_bytes: inputs.shares_covariates.then_some(design_matrix_bytes).flatten(),
        seconds: None,
        seconds_basis: NO_BENCHMARK,
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
        folds: shared.map_or_else(
            || u32::try_from(antecedent_estimate::DEFAULT_AIPW_FOLDS).unwrap_or(5),
            |design| design.n_folds,
        ),
        shares_covariates: shared.is_some_and(|design| design.covariate.is_some()),
    }
}

impl PreparedStudy {
    /// Planning-time cost counts for this plan: nuisance fits, folds, bootstrap replicates,
    /// the design matrix it materializes, and the active inference default.
    ///
    /// A planning hint, not a runtime guarantee, and counts only: no seconds are estimated
    /// because the repository has no named local benchmark of these fits.
    ///
    /// # Errors
    ///
    /// Never today; the `Result` keeps the signature stable for plans whose cost cannot be
    /// stated.
    pub fn estimate_cost(&self) -> Result<CostEstimate, CausalError> {
        Ok(cost_of(&inputs_of(self.study())))
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

#[cfg(test)]
mod tests {
    use super::{CostInputs, cost_of, total_cost};
    use crate::strategy_table::EstimatorId;

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
}
