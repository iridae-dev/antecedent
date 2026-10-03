//! Factorized-propensity AIPW for bounded discrete joint cells (2.2 E5).
//!
//! The joint treatment `T = (T_1, .., T_k)` is `k <= 3` binary coordinates, so there are at
//! most 8 cells. [`crate::cell_aipw::CellSaturatedAipw`] fits one multinomial logit over all
//! cells and refuses the whole family when any cell is empty. This route instead writes each
//! cell propensity as a product of binary conditionals under a declared ordering `o`,
//!
//! ```text
//! P(T = c | Z) = prod_j P(T_{o_j} = c_{o_j} | T_{o_1..o_{j-1}} = c_{o_1..o_{j-1}}, Z),
//! ```
//!
//! and fits each conditional with the ridge-logistic learner of
//! [`crate::propensity::PropensityNuisance`], separately inside every observed prefix
//! stratum (saturated in the prefix, ridge-penalized in `Z`). Folds are cross-fit: a
//! conditional is fit on the training rows of a fold and predicts that fold's rows, with the
//! penalty chosen on the training rows only. The per-cell outcome model is the same OLS on
//! `[1 | Z]` as the multinomial route.
//!
//! A cell is evaluated or refused **individually**: an empty cell, an unsupported
//! conditional factor (no training rows with the prefix), an outcome model without residual
//! degrees of freedom, or inverse-probability weights with too small an effective sample
//! each refuse that cell with a registered reason and leave the others. A prefix stratum
//! whose training rows are all of one class is a constant conditional (probability 0 or 1),
//! counted in [`FactorizedJointFit::degenerate_conditionals`], never a silent floor.
//!
//! Checks: the enumerated cell propensities are summed per row (normalization) and refused
//! if they exceed one or, when every cell is enumerable, differ from one beyond the declared
//! tolerance; the family is re-estimated under every declared ordering and a cell whose
//! estimates move more than the tolerance (or that an alternative ordering cannot support)
//! is flagged. The flag is a receipt about this table, never a verdict on the ordering.
//!
//! The score table keeps the declared ordering's scores for the supported cells so a family
//! contrast has aligned scores, but the propensity is penalized: its provenance carries the
//! ridge tag and no interval or joint covariance is published
//! ([`refuse_joint_inference`]).
//!
//! A declared learner ([`FactorizedJointConfig::learner`], resolved by
//! [`declared_joint_nuisance`]) replaces both the ridge-logistic conditionals and the per-cell
//! OLS outcome models with `antecedent-learn` learners under the same fold plan. Every
//! prediction is out of fold (a learner is fit on a fold's training rows and predicts only that
//! fold's rows), a failed fit refuses the cells that need it and never falls back to ridge, and
//! the score table's provenance carries the learner identity, the learners' implementations and
//! the seeds. The result stays point only: the table is marked
//! (`provenance_marks_learned`) so no interval or covariance is published
//! ([`refuse_joint_learner_inference`]), because a flexible nuisance needs its own remainder and
//! coverage license that no record grants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_learn::{
    ForestSpec, GbtSpec, LearnerFactory, LearnerProvenance, LearnerSpec, PredictionTask,
    RowSelection, TargetView, resolve_for,
};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::aipw::{predict_colmajor, select_rows_colmajor};
use crate::cell_aipw::{cell_minus_control_coefficients, interaction_coefficients};
use crate::error::EstimationError;
use crate::learn_nuisance::{crossfit_fold_plan, design_for_spec};
use crate::propensity::{
    PropensityNuisance, RidgeFoldInput, RidgeFoldSelection, RidgeTuning, fit_ridge_fold, refuse,
};
use crate::scores::{ScoreColumn, ScoreTable};

/// Most jointly intervened binary components of a factorized joint cell.
pub const MAX_FACTORIZED_COMPONENTS: usize = 3;

/// Most orderings one fit re-estimates the family under (`3! = 6`).
pub const MAX_ORDERINGS: usize = 6;

/// Largest declared number of cross-fit folds.
const MAX_FOLDS: usize = 20;

/// Provenance of the factorized joint-cell score table.
const FACTORIZED_PROVENANCE: &str = "joint_cell.factorized.crossfit.ridge_logistic_prefix.ols.v1";

/// Provenance of the score table when a declared learner supplies the nuisances.
const LEARNED_PROVENANCE: &str = "joint_cell.factorized.crossfit.learner_prefix.v1";

/// Marker that a score table's nuisances came from a declared learner.
const LEARNED_PROVENANCE_TAG: &str = ";nuisance=ml_joint_cell";

/// Whether a score table's provenance marks learner-supplied joint-cell nuisances, whose
/// remainder is not licensed: the table is a point-and-score artifact.
#[must_use]
pub(crate) fn provenance_marks_learned(provenance: &str) -> bool {
    provenance.contains(LEARNED_PROVENANCE_TAG)
}

/// Declared tuning of one factorized joint-cell fit.
#[derive(Clone, Debug)]
pub struct FactorizedJointConfig {
    /// Ridge tuning of every conditional factor (the default route; unused with a learner).
    pub tuning: RidgeTuning,
    /// Declared `antecedent-learn` learner for every prefix-stratum conditional and per-cell
    /// outcome model (see [`declared_joint_nuisance`]); `None` is the ridge-logistic plus OLS
    /// route. The spec is mapped onto each task with `LearnerSpec::for_task`; `Auto` is refused
    /// because it hides the learner actually fit.
    pub learner: Option<LearnerSpec>,
    /// Cross-fit folds, `2..=20`.
    pub folds: usize,
    /// Seed of the fold plan and of the inner penalty-selection plans.
    pub seed: u64,
    /// Floor applied to a cell propensity before it becomes an inverse-probability weight,
    /// in `(0, 0.5)`. There is no unclipped mode: a zero propensity is an infinite weight.
    pub clip: f64,
    /// A cell whose in-cell weights have a Kish effective sample size below this is refused.
    pub min_cell_ess: f64,
    /// Largest accepted deviation of a row's enumerated cell propensities from summing to one.
    pub normalization_tolerance: f64,
    /// A cell is flagged when its estimates across orderings spread by more than this
    /// multiple of the outcome's standard deviation.
    pub ordering_tolerance_sd: f64,
}

impl FactorizedJointConfig {
    /// 5 folds, clip 0.01, minimum cell ESS 10, normalization tolerance `1e-9`, ordering
    /// tolerance 0.05 outcome standard deviations, the ridge-logistic route.
    #[must_use]
    pub fn new(tuning: RidgeTuning) -> Self {
        Self {
            tuning,
            learner: None,
            folds: 5,
            seed: 0,
            clip: 0.01,
            min_cell_ess: 10.0,
            normalization_tolerance: 1e-9,
            ordering_tolerance_sd: 0.05,
        }
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let bad = |message: &str| {
            Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "joint_cells.config",
                message,
            ))
        };
        if !(2..=MAX_FOLDS).contains(&self.folds) {
            return bad("folds must lie in 2..=20");
        }
        if !(self.clip > 0.0 && self.clip < 0.5) {
            return bad(
                "clip must lie in (0, 0.5); an unclipped zero propensity is an infinite weight",
            );
        }
        if !(self.min_cell_ess.is_finite() && self.min_cell_ess >= 1.0) {
            return bad("min_cell_ess must be finite and at least 1");
        }
        if !(self.normalization_tolerance.is_finite() && self.normalization_tolerance > 0.0) {
            return bad("normalization_tolerance must be finite and positive");
        }
        if !(self.ordering_tolerance_sd.is_finite() && self.ordering_tolerance_sd >= 0.0) {
            return bad("ordering_tolerance_sd must be finite and non-negative");
        }
        if let Some(spec) = self.learner {
            if matches!(spec, LearnerSpec::Auto) || spec.validate().is_err() {
                return Err(refuse(
                    antecedent_core::reason_code!("invalid_argument"),
                    "joint_cells.learner",
                    "declare an explicit learner with valid hyperparameters; auto is refused \
                     because it hides the learner actually fit",
                ));
            }
        }
        Ok(())
    }
}

/// Validate a declared nuisance provider for joint cells and return the learner it declares.
///
/// `ridge_logistic` is the default route (`None`). `random_forest` and `gradient_boosted_trees`
/// declare that learner with its default hyperparameters for every conditional and outcome
/// model (a point-only route; see [`FactorizedJointConfig::learner`]). `lasso` is closed
/// (`selection_inference_not_licensed`). The remaining machine-learning names are not declared
/// joint-cell learners (`ml_nuisance_not_licensed`). Any other name is an invalid argument.
///
/// # Errors
///
/// A refusal for every provider that is not declared.
pub fn declared_joint_nuisance(provider: &str) -> Result<Option<LearnerSpec>, EstimationError> {
    match provider {
        "ridge_logistic" => Ok(None),
        "random_forest" => Ok(Some(LearnerSpec::RandomForest(ForestSpec::default()))),
        "gradient_boosted_trees" => Ok(Some(LearnerSpec::GradientBoostedTrees(GbtSpec::default()))),
        "lasso" => PropensityNuisance::lasso().validate_for_execution().map(|()| None),
        "ml" | "gradient_boosting" | "neural_network" | "forest" | "boosting" => Err(refuse(
            antecedent_core::reason_code!("ml_nuisance_not_licensed"),
            "joint_cells.ml_learner_not_declared",
            &format!(
                "{provider} is not a declared joint-cell learner; declare random_forest or \
                 gradient_boosted_trees (point only), or ridge_logistic"
            ),
        )),
        other => Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_cells.nuisance_provider",
            &format!(
                "unknown joint-cell nuisance provider {other:?}; declare ridge_logistic, \
                 random_forest or gradient_boosted_trees"
            ),
        )),
    }
}

/// The refusal for an interval or joint covariance over a joint-cell family.
///
/// The conditional propensities are ridge-penalized, so the nuisance remainder is not shown
/// negligible; only the point estimates and the aligned score table are published.
#[must_use]
pub fn refuse_joint_inference() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("penalized_interval_not_licensed"),
        "joint_cells.interval_withheld",
        "factorized joint cells publish point estimates and an aligned score table only; no \
         interval or joint covariance is licensed for penalized conditional propensities",
    )
}

/// The refusal for an interval or joint covariance over learner-supplied joint cells.
///
/// A flexible nuisance needs its own remainder-rate and coverage license for an interval, and no
/// record grants one on joint cells; only the point estimates and the aligned score table are
/// published.
#[must_use]
pub fn refuse_joint_learner_inference() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("ml_nuisance_not_licensed"),
        "joint_cells.ml_interval_withheld",
        "learner-supplied joint cells publish point estimates and an aligned score table only; \
         an interval or joint covariance over machine-learning nuisances needs a nuisance \
         remainder and coverage license that no record grants",
    )
}

/// Orderings to re-estimate under: the declared ordering first, then (when
/// `all_permutations`) every other permutation in lexicographic order.
///
/// # Errors
///
/// `declared` is not a permutation of `0..k`, or `k` is outside `1..=3`.
pub fn orderings_for(
    k: usize,
    declared: &[usize],
    all_permutations: bool,
) -> Result<Vec<Vec<usize>>, EstimationError> {
    check_permutation(k, declared)?;
    if !all_permutations {
        return Ok(vec![declared.to_vec()]);
    }
    let mut all = Vec::new();
    permutations(&mut (0..k).collect::<Vec<_>>(), 0, &mut all);
    all.sort();
    let mut out = vec![declared.to_vec()];
    out.extend(all.into_iter().filter(|p| p.as_slice() != declared));
    Ok(out)
}

fn permutations(items: &mut Vec<usize>, start: usize, out: &mut Vec<Vec<usize>>) {
    if start == items.len() {
        out.push(items.clone());
        return;
    }
    for i in start..items.len() {
        items.swap(start, i);
        permutations(items, start + 1, out);
        items.swap(start, i);
    }
}

fn ordering_error(message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("invalid_argument"), "joint_cells.ordering", message)
}

fn check_permutation(k: usize, ordering: &[usize]) -> Result<(), EstimationError> {
    if k == 0 || k > MAX_FACTORIZED_COMPONENTS {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "joint_cells.component_count",
            "factorized joint cells support 1 to 3 binary components",
        ));
    }
    let mut seen = vec![false; k];
    for &component in ordering {
        match seen.get_mut(component) {
            Some(slot) if !*slot => *slot = true,
            _ => return Err(ordering_error("an ordering must list every component exactly once")),
        }
    }
    if ordering.len() != k {
        return Err(ordering_error("an ordering must list every component exactly once"));
    }
    Ok(())
}

fn validate_orderings(k: usize, orderings: &[Vec<usize>]) -> Result<(), EstimationError> {
    if orderings.is_empty() || orderings.len() > MAX_ORDERINGS {
        return Err(ordering_error("declare 1 to 6 orderings; the first is the declared one"));
    }
    for (i, ordering) in orderings.iter().enumerate() {
        check_permutation(k, ordering)?;
        if orderings[..i].contains(ordering) {
            return Err(ordering_error("orderings must be distinct"));
        }
    }
    Ok(())
}

/// Why one cell was refused, with its registered reason code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail callers switch on.
    pub detail: &'static str,
    /// What was observed.
    pub message: String,
}

/// Point estimate and positivity diagnostics of one supported cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEstimate {
    /// Cross-fitted AIPW mean of `Y^{do(T = cell)}`.
    pub estimate: f64,
    /// Kish effective sample size of the in-cell inverse-probability weights.
    pub ess: f64,
    /// Smallest out-of-fold cell propensity over all rows.
    pub propensity_min: f64,
    /// Largest out-of-fold cell propensity over all rows.
    pub propensity_max: f64,
    /// Share of rows whose cell propensity fell below the clip.
    pub clipped_share: f64,
}

/// A cell's evaluation under one ordering.
#[derive(Clone, Debug, PartialEq)]
pub enum CellStatus {
    /// Evaluated.
    Supported(CellEstimate),
    /// Refused individually; the rest of the family is kept.
    Unsupported(CellRefusal),
}

/// One cell of the family under the declared ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct JointCellReport {
    /// Cell code: bit `j` is the level of component `j` (`treatments[j]`).
    pub cell: u32,
    /// Level of each component, in treatment order.
    pub levels: Vec<u8>,
    /// Complete-case rows observed in the cell.
    pub rows: usize,
    /// Evaluation under the declared ordering.
    pub status: CellStatus,
}

/// Normalization of the enumerated cell propensities under one ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizationCheck {
    /// The ordering checked.
    pub ordering: Vec<usize>,
    /// Cells whose propensity could be enumerated (all conditionals supported).
    pub cells_enumerated: usize,
    /// Largest `|sum of cell propensities - 1|` over rows; absent when some cell could not be
    /// enumerated (the sum is then only bounded above by one).
    pub max_abs_error: Option<f64>,
    /// Largest per-row sum of the enumerated cell propensities.
    pub max_row_sum: f64,
}

/// One cell's estimates under every declared ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct CellOrderingSpread {
    /// Cell code.
    pub cell: u32,
    /// Estimate under each ordering, in declared order; `None` where that ordering refused.
    pub estimates: Vec<Option<f64>>,
    /// Max minus min over the available estimates; absent with fewer than two.
    pub spread: Option<f64>,
    /// The declared ordering supports the cell and an alternative either refuses it or moves
    /// it by more than the tolerance.
    pub flagged: bool,
}

/// Sensitivity of the family to the declared ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderingSensitivity {
    /// Orderings estimated, the declared one first.
    pub orderings: Vec<Vec<usize>>,
    /// Absolute spread tolerance used (`ordering_tolerance_sd` times the outcome's SD).
    pub tolerance: f64,
    /// One entry per cell.
    pub cells: Vec<CellOrderingSpread>,
    /// Whether any cell is flagged.
    pub disagreement: bool,
}

/// A family-level contrast over supported cells.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JointContrast {
    /// Requested cell minus the all-zero cell.
    CellMinusControl(u32),
    /// `mu_00 - mu_10 - mu_01 + mu_11`; two components, every cell supported.
    Interaction,
}

/// Result of a factorized joint-cell fit.
#[derive(Clone, Debug)]
pub struct FactorizedJointFit {
    /// Treatment components, in cell-bit order.
    pub treatments: Vec<VariableId>,
    /// Complete-case rows.
    pub n_rows: usize,
    /// Cross-fit folds.
    pub folds: usize,
    /// Every cell under the declared ordering.
    pub cells: Vec<JointCellReport>,
    /// Normalization under each ordering.
    pub normalization: Vec<NormalizationCheck>,
    /// Sensitivity to the declared ordering.
    pub sensitivity: OrderingSensitivity,
    /// Conditional strata fit as a constant (all training rows one class), over every
    /// ordering.
    pub degenerate_conditionals: usize,
    /// Aligned scores of the supported cells under the declared ordering. The provenance
    /// carries the ridge tag, so no interval or covariance is published from it.
    pub scores: ScoreTable,
}

impl FactorizedJointFit {
    /// Point value of a family contrast over the supported cells.
    ///
    /// # Errors
    ///
    /// `joint_cell_unsupported` when a cell the contrast needs is not supported (or the
    /// interaction is requested for other than two components).
    pub fn contrast_point(&self, contrast: JointContrast) -> Result<f64, EstimationError> {
        let supported = |cell: u32| self.scores.columns.iter().any(|c| c.arm == cell);
        let coefficients = match contrast {
            JointContrast::CellMinusControl(cell) => {
                if !supported(cell) || !supported(0) {
                    return Err(contrast_refusal(
                        "the requested cell or the all-zero control cell is not supported",
                    ));
                }
                cell_minus_control_coefficients(&self.scores, cell)?
            }
            JointContrast::Interaction => {
                if self.treatments.len() != 2 || self.scores.n_columns() != 4 {
                    return Err(contrast_refusal(
                        "the interaction needs two components with all four cells supported",
                    ));
                }
                interaction_coefficients(&self.scores)?
            }
        };
        let row_scores = self.scores.combine_scores(&coefficients)?;
        Ok(row_scores.iter().sum::<f64>() / row_scores.len() as f64)
    }
}

fn contrast_refusal(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("joint_cell_unsupported"),
        "joint_cells.contrast_cell_unsupported",
        message,
    )
}

struct Problem {
    n: usize,
    ncols: usize,
    design: Vec<f64>,
    cell: Vec<u32>,
    outcome: Vec<f64>,
    row_index: Vec<u32>,
    adjustment_set: Arc<[VariableId]>,
}

struct FoldSplit {
    train: Vec<usize>,
    valid: Vec<usize>,
    design_valid: Vec<f64>,
}

fn prepare_problem(
    data: &TabularData,
    treatments: &[VariableId],
    outcome: VariableId,
    adjustment: &[VariableId],
) -> Result<Problem, EstimationError> {
    let invalid = |detail: &'static str, message: &str| {
        refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
    };
    let mut ids: Vec<VariableId> = treatments.to_vec();
    ids.push(outcome);
    ids.extend_from_slice(adjustment);
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            return Err(invalid(
                "joint_cells.columns",
                "treatment, outcome and adjustment columns must be distinct",
            ));
        }
    }
    let mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let outcome_values = data.float64_masked(outcome, &mask).map_err(EstimationError::from)?;
    let n = outcome_values.len();
    if n == 0 {
        return Err(invalid("joint_cells.no_rows", "no complete-case rows for joint cells"));
    }
    let mut cell = vec![0u32; n];
    for (j, &id) in treatments.iter().enumerate() {
        let column = data.float64_masked(id, &mask).map_err(EstimationError::from)?;
        for (code, &value) in cell.iter_mut().zip(&column) {
            if (value - 1.0).abs() <= 1e-12 {
                *code |= 1 << j;
            } else if value.abs() > 1e-12 {
                return Err(invalid(
                    "joint_cells.not_binary",
                    "factorized joint cells require binary 0/1 treatment components",
                ));
            }
        }
    }
    let ncols = 1 + adjustment.len();
    let mut design = vec![1.0; n * ncols];
    for (j, &z) in adjustment.iter().enumerate() {
        let column = data.float64_masked(z, &mask).map_err(EstimationError::from)?;
        design[(1 + j) * n..(2 + j) * n].copy_from_slice(&column);
    }
    let mut row_index = Vec::with_capacity(n);
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            row_index.push(u32::try_from(i).map_err(|_| {
                EstimationError::data_msg("row index exceeds the u32 row-id capacity")
            })?);
        }
    }
    Ok(Problem {
        n,
        ncols,
        design,
        cell,
        outcome: outcome_values,
        row_index,
        adjustment_set: Arc::from(adjustment.to_vec()),
    })
}

fn is_cancelled(error: &EstimationError) -> bool {
    matches!(error, EstimationError::Refused { code, .. }
        if *code == antecedent_core::reason_code!("cancelled_no_claim"))
}

fn cancelled() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "joint_cells.cancelled",
        "the factorized joint-cell fit was cancelled; no estimate is reported",
    )
}

/// Learners resolved for the conditionals (binary probability) and the per-cell outcome models
/// (regression) of a declared [`FactorizedJointConfig::learner`].
struct LearnedNuisance {
    outcome: Box<dyn LearnerFactory>,
    outcome_spec: LearnerSpec,
    propensity: Box<dyn LearnerFactory>,
    propensity_spec: LearnerSpec,
}

impl LearnedNuisance {
    fn resolve(spec: LearnerSpec) -> Result<Self, EstimationError> {
        let outcome_spec = spec.for_task(PredictionTask::Regression);
        let propensity_spec = spec.for_task(PredictionTask::BinaryProbability);
        let unavailable = |error: antecedent_learn::LearnError| {
            refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "joint_cells.learner_unavailable",
                &format!(
                    "the {} learner cannot be resolved for joint-cell nuisances: {error}",
                    spec.name()
                ),
            )
        };
        Ok(Self {
            outcome: resolve_for(outcome_spec, PredictionTask::Regression).map_err(unavailable)?,
            outcome_spec,
            propensity: resolve_for(propensity_spec, PredictionTask::BinaryProbability)
                .map_err(unavailable)?,
            propensity_spec,
        })
    }

    fn identity(&self) -> String {
        format!(
            "outcome={};propensity={}",
            self.outcome_spec.identity(),
            self.propensity_spec.identity()
        )
    }
}

fn row_ids(rows: &[usize]) -> Result<Vec<u32>, EstimationError> {
    rows.iter()
        .map(|&i| {
            u32::try_from(i)
                .map_err(|_| EstimationError::data_msg("row index exceeds the u32 row-id capacity"))
        })
        .collect()
}

/// Fit `factory` on the `train` rows of the problem's design and predict the `valid` rows.
///
/// `train` and `valid` are the two sides of one fold, so no row is both fit on and predicted.
/// Every failure is returned as text; the caller decides whether it is a cancellation.
fn learner_fold_predict(
    factory: &dyn LearnerFactory,
    view_spec: LearnerSpec,
    problem: &Problem,
    train: &[u32],
    valid: &[u32],
    y: &[f64],
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, LearnerProvenance), String> {
    let view = design_for_spec(view_spec, &problem.design, problem.n, problem.ncols)
        .map_err(|error| error.to_string())?;
    let fit_view = view.with_rows(RowSelection::new(train)).map_err(|error| error.to_string())?;
    let model =
        factory.fit(fit_view, TargetView::new(y), None, ctx).map_err(|error| error.to_string())?;
    let predict_view =
        view.with_rows(RowSelection::new(valid)).map_err(|error| error.to_string())?;
    let mut predicted = vec![0.0; valid.len()];
    model.predict(predict_view, &mut predicted, ctx).map_err(|error| error.to_string())?;
    Ok((predicted, model.provenance()))
}

/// Cache key of one conditional: fold, prefix components (bit mask), their values, target
/// component.
type ConditionalKey = (usize, u32, u32, usize);

/// Cross-fitted conditional factors, fit once per key and shared across orderings.
struct ConditionalFits<'a> {
    ctx: &'a ExecutionContext,
    config: &'a FactorizedJointConfig,
    problem: &'a Problem,
    folds: &'a [FoldSplit],
    learned: Option<&'a LearnedNuisance>,
    cache: HashMap<ConditionalKey, Result<Vec<f64>, String>>,
    selections: Vec<RidgeFoldSelection>,
    provenances: Vec<LearnerProvenance>,
    degenerate: usize,
}

impl ConditionalFits<'_> {
    /// Validation-row probabilities `P(T_target = 1 | prefix = values, Z)` of one fold, or
    /// the reason the factor is unsupported. Only cancellation is an `Err`.
    fn get(
        &mut self,
        fold: usize,
        mask: u32,
        values: u32,
        target: usize,
    ) -> Result<Result<Vec<f64>, String>, EstimationError> {
        let key = (fold, mask, values, target);
        if let Some(hit) = self.cache.get(&key) {
            return Ok(hit.clone());
        }
        if self.ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let fitted = self.fit(fold, mask, values, target)?;
        self.cache.insert(key, fitted.clone());
        Ok(fitted)
    }

    fn fit(
        &mut self,
        fold: usize,
        mask: u32,
        values: u32,
        target: usize,
    ) -> Result<Result<Vec<f64>, String>, EstimationError> {
        let problem = self.problem;
        let split = &self.folds[fold];
        let rows: Vec<usize> =
            split.train.iter().copied().filter(|&i| problem.cell[i] & mask == values).collect();
        let n_valid = split.valid.len();
        if rows.is_empty() {
            return Ok(Err("no training row has this prefix of treatment values".to_string()));
        }
        let t: Vec<f64> =
            rows.iter().map(|&i| f64::from((problem.cell[i] >> target) & 1)).collect();
        let positives = t.iter().filter(|&&v| v > 0.5).count();
        if positives == 0 || positives == rows.len() {
            self.degenerate += 1;
            return Ok(Ok(vec![if positives == 0 { 0.0 } else { 1.0 }; n_valid]));
        }
        if let Some(learned) = self.learned {
            return self.fit_learned(learned, fold, &rows, target);
        }
        let mut design_train = Vec::new();
        select_rows_colmajor(&problem.design, problem.n, problem.ncols, &rows, &mut design_train);
        let units: Vec<u32> = rows.iter().map(|&i| problem.row_index[i]).collect();
        let input = RidgeFoldInput {
            design_train: &design_train,
            n_train: rows.len(),
            design_valid: &split.design_valid,
            n_valid,
            ncols: problem.ncols,
            t_train: &t,
            units: &units,
            fold,
            seed: self.config.seed,
        };
        match fit_ridge_fold(&self.config.tuning, &input, self.ctx) {
            Ok((predicted, selection)) => {
                self.selections.push(selection);
                Ok(Ok(predicted))
            }
            Err(error) if is_cancelled(&error) => Err(error),
            Err(error) => Ok(Err(error.to_string())),
        }
    }
}

impl ConditionalFits<'_> {
    /// The declared learner's out-of-fold probabilities for one conditional: it is fit on the
    /// fold's training rows with the prefix and predicts the fold's validation rows only. A
    /// failed fit or a probability outside `[0, 1]` makes the factor unsupported (refusing the
    /// cells that need it); only cancellation is an `Err`.
    fn fit_learned(
        &mut self,
        learned: &LearnedNuisance,
        fold: usize,
        rows: &[usize],
        target: usize,
    ) -> Result<Result<Vec<f64>, String>, EstimationError> {
        let problem = self.problem;
        let train = row_ids(rows)?;
        let valid = row_ids(&self.folds[fold].valid)?;
        let t_full: Vec<f64> =
            problem.cell.iter().map(|&code| f64::from((code >> target) & 1)).collect();
        match learner_fold_predict(
            learned.propensity.as_ref(),
            learned.propensity_spec,
            problem,
            &train,
            &valid,
            &t_full,
            self.ctx,
        ) {
            Ok((predicted, _)) if predicted.iter().any(|p| !(0.0..=1.0).contains(p)) => {
                Ok(Err(format!("fold {fold}: the learner returned a probability outside [0, 1]")))
            }
            Ok((predicted, provenance)) => {
                self.provenances.push(provenance);
                Ok(Ok(predicted))
            }
            Err(_) if self.ctx.cancellation.is_cancelled() => Err(cancelled()),
            Err(reason) => Ok(Err(format!("fold {fold}: the learner failed: {reason}"))),
        }
    }
}

/// Out-of-fold predictions of one cell's outcome model and the learner provenance of its folds.
type LearnedOutcome = (Vec<f64>, Vec<LearnerProvenance>);

/// Out-of-fold outcome-model predictions of one cell under a declared learner, with the
/// learner provenance of every fold, or why the cell's model is unsupported. Only cancellation
/// is an `Err`.
fn learned_outcome_predictions(
    learned: &LearnedNuisance,
    problem: &Problem,
    folds: &[FoldSplit],
    cell: u32,
    ctx: &ExecutionContext,
) -> Result<Result<LearnedOutcome, String>, EstimationError> {
    let mut out = vec![0.0; problem.n];
    let mut provenances = Vec::with_capacity(folds.len());
    for (fold, split) in folds.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let rows: Vec<usize> =
            split.train.iter().copied().filter(|&i| problem.cell[i] == cell).collect();
        if rows.is_empty() {
            return Ok(Err(format!("fold {fold}: no training row lies in the cell")));
        }
        let train = row_ids(&rows)?;
        let valid = row_ids(&split.valid)?;
        match learner_fold_predict(
            learned.outcome.as_ref(),
            learned.outcome_spec,
            problem,
            &train,
            &valid,
            &problem.outcome,
            ctx,
        ) {
            Ok((predicted, _)) if predicted.iter().any(|p| !p.is_finite()) => {
                return Ok(Err(format!("fold {fold}: the learner returned a non-finite value")));
            }
            Ok((predicted, provenance)) => {
                provenances.push(provenance);
                for (&i, &p) in split.valid.iter().zip(&predicted) {
                    out[i] = p;
                }
            }
            Err(_) if ctx.cancellation.is_cancelled() => return Err(cancelled()),
            Err(reason) => {
                return Ok(Err(format!("fold {fold}: the outcome learner failed: {reason}")));
            }
        }
    }
    Ok(Ok((out, provenances)))
}

/// Provenance suffix of a learner-supplied table: the declared learner identity, the distinct
/// fitted implementations, and the seeds that make the fit replayable.
fn learned_suffix(
    learned: &LearnedNuisance,
    provenances: &[LearnerProvenance],
    fold_seed: u64,
    ctx_seed: u64,
) -> String {
    let implementations: std::collections::BTreeSet<String> = provenances
        .iter()
        .map(|p| format!("{}/{}/{}", p.spec, p.implementation, p.version))
        .collect();
    let implementations: Vec<String> = implementations.into_iter().collect();
    format!(
        "{LEARNED_PROVENANCE_TAG};{};learners={};fold_seed={fold_seed};ctx_seed={ctx_seed}",
        learned.identity(),
        implementations.join("|")
    )
}

/// Out-of-fold outcome-model predictions of one cell, or why the model is unsupported.
fn outcome_predictions(
    problem: &Problem,
    folds: &[FoldSplit],
    cell: u32,
) -> Result<Vec<f64>, String> {
    let mut out = vec![0.0; problem.n];
    let mut workspace = LeastSquaresWorkspace::default();
    for (fold, split) in folds.iter().enumerate() {
        let rows: Vec<usize> =
            split.train.iter().copied().filter(|&i| problem.cell[i] == cell).collect();
        if rows.len() <= problem.ncols {
            return Err(format!(
                "fold {fold}: the training cell has {} rows for {} design columns, so its \
                 outcome model has no residual degrees of freedom",
                rows.len(),
                problem.ncols
            ));
        }
        let mut design = Vec::new();
        select_rows_colmajor(&problem.design, problem.n, problem.ncols, &rows, &mut design);
        let y: Vec<f64> = rows.iter().map(|&i| problem.outcome[i]).collect();
        let fit = FaerBackend
            .least_squares(&design, rows.len(), problem.ncols, &y, &mut workspace)
            .map_err(|error| format!("fold {fold}: {error}"))?;
        let mut predicted = Vec::new();
        predict_colmajor(
            &split.design_valid,
            split.valid.len(),
            problem.ncols,
            &fit.coefficients,
            &mut predicted,
        );
        for (&i, &p) in split.valid.iter().zip(&predicted) {
            out[i] = p;
        }
    }
    Ok(out)
}

/// Out-of-fold propensity of every cell under one ordering (`Err`: the reason it cannot be
/// enumerated).
fn ordering_propensities(
    fits: &mut ConditionalFits<'_>,
    ordering: &[usize],
    n_cells: u32,
) -> Result<Vec<Result<Vec<f64>, String>>, EstimationError> {
    let folds = fits.folds;
    let n = fits.problem.n;
    let mut out = Vec::new();
    for cell in 0..n_cells {
        let mut column = vec![0.0; n];
        let mut failure: Option<String> = None;
        'folds: for (fold, split) in folds.iter().enumerate() {
            let mut product = vec![1.0; split.valid.len()];
            for (j, &target) in ordering.iter().enumerate() {
                let mask = ordering[..j].iter().fold(0u32, |m, &t| m | (1 << t));
                match fits.get(fold, mask, cell & mask, target)? {
                    Ok(p1) => {
                        let one = (cell >> target) & 1 == 1;
                        for (q, &p) in product.iter_mut().zip(&p1) {
                            *q *= if one { p } else { 1.0 - p };
                        }
                    }
                    Err(reason) => {
                        failure = Some(format!(
                            "P(component {target} | components {:?}) on fold {fold}: {reason}",
                            &ordering[..j]
                        ));
                        break 'folds;
                    }
                }
            }
            for (&i, &q) in split.valid.iter().zip(&product) {
                column[i] = q;
            }
        }
        out.push(failure.map_or(Ok(column), Err));
    }
    Ok(out)
}

fn unsupported(detail: &'static str, message: String) -> CellRefusal {
    CellRefusal { code: antecedent_core::reason_code!("joint_cell_unsupported"), detail, message }
}

/// A cell's estimate (and scores) under one ordering, or its individual refusal.
fn evaluate_cell(
    problem: &Problem,
    config: &FactorizedJointConfig,
    cell: u32,
    rows: usize,
    propensity: &Result<Vec<f64>, String>,
    mu: &Result<Vec<f64>, String>,
) -> Result<(CellEstimate, Vec<f64>), CellRefusal> {
    if rows == 0 {
        return Err(CellRefusal {
            code: antecedent_core::reason_code!("arm_not_populated"),
            detail: "joint_cells.cell_empty",
            message: format!("cell {cell} has no complete-case rows"),
        });
    }
    let mu = mu
        .as_ref()
        .map_err(|why| unsupported("joint_cells.outcome_model_unsupported", why.clone()))?;
    let e = propensity
        .as_ref()
        .map_err(|why| unsupported("joint_cells.conditional_unsupported", why.clone()))?;
    if e.iter().any(|v| !v.is_finite()) {
        return Err(unsupported(
            "joint_cells.conditional_unsupported",
            format!("cell {cell} has a non-finite fitted propensity"),
        ));
    }
    let weight = |p: f64| 1.0 / p.max(config.clip);
    let (mut sum_w, mut sum_w2) = (0.0, 0.0);
    for (i, &p) in e.iter().enumerate() {
        if problem.cell[i] == cell {
            let w = weight(p);
            sum_w += w;
            sum_w2 += w * w;
        }
    }
    let ess = sum_w * sum_w / sum_w2;
    if ess < config.min_cell_ess {
        return Err(unsupported(
            "joint_cells.cell_ess_below_minimum",
            format!(
                "cell {cell}: in-cell weights have effective sample size {ess:.3}, below the \
                 declared minimum {}",
                config.min_cell_ess
            ),
        ));
    }
    let scores: Vec<f64> = (0..problem.n)
        .map(|i| {
            let indicator = f64::from(problem.cell[i] == cell);
            mu[i] + indicator * weight(e[i]) * (problem.outcome[i] - mu[i])
        })
        .collect();
    let n = problem.n as f64;
    Ok((
        CellEstimate {
            estimate: scores.iter().sum::<f64>() / n,
            ess,
            propensity_min: e.iter().copied().fold(f64::INFINITY, f64::min),
            propensity_max: e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            clipped_share: e.iter().filter(|&&p| p < config.clip).count() as f64 / n,
        },
        scores,
    ))
}

fn normalization(
    ordering: &[usize],
    propensities: &[Result<Vec<f64>, String>],
    n: usize,
    tolerance: f64,
) -> Result<NormalizationCheck, EstimationError> {
    let enumerated: Vec<&Vec<f64>> = propensities.iter().filter_map(|p| p.as_ref().ok()).collect();
    let complete = enumerated.len() == propensities.len();
    let (mut max_row_sum, mut max_error) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let sum: f64 = enumerated.iter().map(|column| column[i]).sum();
        max_row_sum = max_row_sum.max(sum);
        max_error = max_error.max((sum - 1.0).abs());
    }
    if max_row_sum > 1.0 + tolerance || (complete && max_error > tolerance) {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "joint_cells.normalization_failed",
            &format!(
                "under ordering {ordering:?} the enumerated cell propensities sum to at most \
                 {max_row_sum} with deviation {max_error} from one, beyond the tolerance \
                 {tolerance}; no estimate is reported"
            ),
        ));
    }
    Ok(NormalizationCheck {
        ordering: ordering.to_vec(),
        cells_enumerated: enumerated.len(),
        max_abs_error: complete.then_some(max_error),
        max_row_sum,
    })
}

/// Fit the cell-saturated family with factorized propensities under every declared ordering.
///
/// `treatments` are `k <= 3` binary 0/1 columns (cell bit `j` is `treatments[j]`),
/// `adjustment` the certified adjustment set, and `orderings` the declared ordering followed by
/// any alternatives (see [`orderings_for`]); each is a permutation of `0..k`.
///
/// # Errors
///
/// An invalid declaration, a non-binary treatment, cancellation, a failed normalization
/// check, or no supported cell. A cell that is merely unsupported is reported in
/// [`FactorizedJointFit::cells`], not raised.
pub fn fit_factorized_joint_cells(
    data: &TabularData,
    treatments: &[VariableId],
    outcome: VariableId,
    adjustment: &[VariableId],
    orderings: &[Vec<usize>],
    config: &FactorizedJointConfig,
    ctx: &ExecutionContext,
) -> Result<FactorizedJointFit, EstimationError> {
    config.validate()?;
    let learned = config.learner.map(LearnedNuisance::resolve).transpose()?;
    fit_with_nuisance(
        data,
        treatments,
        outcome,
        adjustment,
        orderings,
        config,
        ctx,
        learned.as_ref(),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the public entry point's arguments plus the resolved nuisance learners"
)]
fn fit_with_nuisance(
    data: &TabularData,
    treatments: &[VariableId],
    outcome: VariableId,
    adjustment: &[VariableId],
    orderings: &[Vec<usize>],
    config: &FactorizedJointConfig,
    ctx: &ExecutionContext,
    learned: Option<&LearnedNuisance>,
) -> Result<FactorizedJointFit, EstimationError> {
    let k = treatments.len();
    check_permutation(k, &(0..k).collect::<Vec<_>>())?;
    validate_orderings(k, orderings)?;
    let problem = prepare_problem(data, treatments, outcome, adjustment)?;
    let n = problem.n;
    let n_cells = 1u32 << k;
    let plan = crossfit_fold_plan(&problem.cell, &problem.row_index, config.folds, config.seed)?;
    let folds: Vec<FoldSplit> = (0..config.folds)
        .map(|fold| {
            let train: Vec<usize> = (0..n).filter(|&i| plan[i] as usize != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|&i| plan[i] as usize == fold).collect();
            let mut design_valid = Vec::new();
            select_rows_colmajor(&problem.design, n, problem.ncols, &valid, &mut design_valid);
            FoldSplit { train, valid, design_valid }
        })
        .collect();
    let rows_in_cell: Vec<usize> =
        (0..n_cells).map(|c| problem.cell.iter().filter(|&&code| code == c).count()).collect();

    let mut mu = Vec::with_capacity(n_cells as usize);
    let mut outcome_provenances: Vec<LearnerProvenance> = Vec::new();
    for cell in 0..n_cells {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        mu.push(if rows_in_cell[cell as usize] == 0 {
            Err("the cell has no rows".to_string())
        } else if let Some(learned) = learned {
            match learned_outcome_predictions(learned, &problem, &folds, cell, ctx)? {
                Ok((predicted, provenances)) => {
                    outcome_provenances.extend(provenances);
                    Ok(predicted)
                }
                Err(why) => Err(why),
            }
        } else {
            outcome_predictions(&problem, &folds, cell)
        });
    }

    let mut fits = ConditionalFits {
        ctx,
        config,
        problem: &problem,
        folds: &folds,
        learned,
        cache: HashMap::new(),
        selections: Vec::new(),
        provenances: Vec::new(),
        degenerate: 0,
    };
    let mut evaluations = Vec::with_capacity(orderings.len());
    let mut checks = Vec::with_capacity(orderings.len());
    let mut declared_propensities = Vec::new();
    for (index, ordering) in orderings.iter().enumerate() {
        let propensities = ordering_propensities(&mut fits, ordering, n_cells)?;
        checks.push(normalization(ordering, &propensities, n, config.normalization_tolerance)?);
        let evaluated: Vec<_> = (0..n_cells)
            .map(|cell| {
                evaluate_cell(
                    &problem,
                    config,
                    cell,
                    rows_in_cell[cell as usize],
                    &propensities[cell as usize],
                    &mu[cell as usize],
                )
            })
            .collect();
        evaluations.push(evaluated);
        if index == 0 {
            declared_propensities = propensities;
        }
    }
    let ConditionalFits { selections, mut provenances, degenerate, .. } = fits;
    provenances.extend(outcome_provenances);

    let sd = {
        let mean = problem.outcome.iter().sum::<f64>() / n as f64;
        (problem.outcome.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n as f64).sqrt()
    };
    let tolerance = config.ordering_tolerance_sd * sd;
    let sensitivity = ordering_sensitivity(orderings, &evaluations, tolerance);

    let mut columns = Vec::new();
    let mut scores = Vec::new();
    let mut propensity_columns = Vec::new();
    let mut cells = Vec::with_capacity(n_cells as usize);
    for (cell, evaluated) in (0..n_cells).zip(&evaluations[0]) {
        let status = match evaluated {
            Ok((estimate, cell_scores)) => {
                columns.push(ScoreColumn { arm: cell, threshold: None });
                scores.extend_from_slice(cell_scores);
                if let Ok(e) = &declared_propensities[cell as usize] {
                    propensity_columns.extend_from_slice(e);
                }
                CellStatus::Supported(estimate.clone())
            }
            Err(refusal) => CellStatus::Unsupported(refusal.clone()),
        };
        cells.push(JointCellReport {
            cell,
            levels: (0..k).map(|j| u8::from((cell >> j) & 1 == 1)).collect(),
            rows: rows_in_cell[cell as usize],
            status,
        });
    }
    if columns.is_empty() {
        let reasons: Vec<String> = cells
            .iter()
            .filter_map(|c| match &c.status {
                CellStatus::Unsupported(r) => Some(format!("cell {}: {}", c.cell, r.message)),
                CellStatus::Supported(_) => None,
            })
            .collect();
        return Err(refuse(
            antecedent_core::reason_code!("joint_cell_unsupported"),
            "joint_cells.no_supported_cell",
            &format!("no cell of the family is supported ({})", reasons.join("; ")),
        ));
    }
    let provenance = match learned {
        Some(learned) => format!(
            "{LEARNED_PROVENANCE}{}",
            learned_suffix(learned, &provenances, config.seed, ctx.rng.master_seed())
        ),
        None => format!(
            "{FACTORIZED_PROVENANCE}{}",
            PropensityNuisance::ridge_logistic(config.tuning.clone())
                .provenance_suffix(&selections)
        ),
    };
    let table = ScoreTable {
        observed_arm: problem.cell.clone().into(),
        propensities: propensity_columns.into(),
        observed_outcome: problem.outcome.clone().into(),
        n_rows: n,
        row_index: Arc::from(problem.row_index.clone()),
        fold_ids: Arc::from(plan),
        n_folds: u32::try_from(config.folds).unwrap_or(u32::MAX),
        scores: Arc::from(scores),
        columns: Arc::from(columns),
        adjustment_set: Arc::clone(&problem.adjustment_set),
        nuisance_provenance: Arc::from(provenance),
        propensity_clip: Some(config.clip),
        treatment: treatments[0],
        intervened: Arc::from(treatments.to_vec()),
    };
    Ok(FactorizedJointFit {
        treatments: treatments.to_vec(),
        n_rows: n,
        folds: config.folds,
        cells,
        normalization: checks,
        sensitivity,
        degenerate_conditionals: degenerate,
        scores: table,
    })
}

type Evaluated = Vec<Result<(CellEstimate, Vec<f64>), CellRefusal>>;

fn ordering_sensitivity(
    orderings: &[Vec<usize>],
    evaluations: &[Evaluated],
    tolerance: f64,
) -> OrderingSensitivity {
    let n_cells = evaluations[0].len();
    let cells: Vec<CellOrderingSpread> = (0..n_cells)
        .map(|cell| {
            let estimates: Vec<Option<f64>> = evaluations
                .iter()
                .map(|evaluated| evaluated[cell].as_ref().ok().map(|(e, _)| e.estimate))
                .collect();
            let present: Vec<f64> = estimates.iter().flatten().copied().collect();
            let spread = (present.len() >= 2).then(|| {
                present.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - present.iter().copied().fold(f64::INFINITY, f64::min)
            });
            let flagged = orderings.len() > 1
                && estimates[0].is_some()
                && (present.len() < estimates.len() || spread.is_some_and(|s| s > tolerance));
            CellOrderingSpread {
                cell: u32::try_from(cell).unwrap_or(u32::MAX),
                estimates,
                spread,
                flagged,
            }
        })
        .collect();
    OrderingSensitivity {
        orderings: orderings.to_vec(),
        tolerance,
        disagreement: cells.iter().any(|c| c.flagged),
        cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported(estimate: f64) -> (CellEstimate, Vec<f64>) {
        (
            CellEstimate {
                estimate,
                ess: 100.0,
                propensity_min: 0.1,
                propensity_max: 0.9,
                clipped_share: 0.0,
            },
            Vec::new(),
        )
    }

    fn refused() -> Result<(CellEstimate, Vec<f64>), CellRefusal> {
        Err(unsupported("joint_cells.cell_ess_below_minimum", "thin".to_string()))
    }

    /// Hand-computed: cell 0 estimates 1.00, 1.02, 0.99 spread by 0.03; cell 1 is refused under
    /// the second ordering; cell 2 is refused under the declared ordering.
    #[test]
    fn the_spread_and_flags_follow_the_declared_tolerance() {
        let orderings = vec![vec![0, 1], vec![1, 0], vec![0, 1]];
        let evaluations: Vec<Evaluated> = vec![
            vec![Ok(supported(1.0)), Ok(supported(2.0)), refused()],
            vec![Ok(supported(1.02)), refused(), Ok(supported(5.0))],
            vec![Ok(supported(0.99)), Ok(supported(2.0)), Ok(supported(5.0))],
        ];
        let lax = ordering_sensitivity(&orderings, &evaluations, 0.05);
        assert!((lax.cells[0].spread.unwrap() - 0.03).abs() < 1e-12);
        assert!(!lax.cells[0].flagged);
        assert!(lax.cells[1].flagged, "a cell an alternative ordering refuses is flagged");
        assert_eq!(lax.cells[1].estimates, vec![Some(2.0), None, Some(2.0)]);
        assert!(!lax.cells[2].flagged, "a cell the declared ordering refuses is not flagged");
        assert!(lax.disagreement);

        let strict = ordering_sensitivity(&orderings, &evaluations, 0.01);
        assert!(strict.cells[0].flagged);

        let single = ordering_sensitivity(&orderings[..1], &evaluations[..1], 0.0);
        assert!(single.cells.iter().all(|c| c.spread.is_none() && !c.flagged));
        assert!(!single.disagreement);
    }

    #[test]
    fn permutations_are_complete_for_three_components() {
        let mut all = Vec::new();
        permutations(&mut vec![0, 1, 2], 0, &mut all);
        all.sort();
        assert_eq!(
            all,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0]
            ]
        );
    }

    use std::sync::Mutex;

    use antecedent_core::CausalRng;
    use antecedent_data::TableView;
    use antecedent_learn::{
        DesignView, FittedPredictor, LearnError, LearnerCapabilities, LinearSpec, LogisticSpec,
    };

    /// `(rows the model was fit on, rows it predicted)` of every fitted model, physical indices.
    type FitLog = Arc<Mutex<Vec<(Vec<usize>, Vec<usize>)>>>;
    /// A rule that makes a fit fail from the physical rows it would be fit on.
    type FailRule = Arc<dyn Fn(&[usize]) -> bool + Send + Sync>;

    fn physical_rows(x: DesignView<'_>) -> Vec<usize> {
        (0..x.nrows()).map(|i| x.physical_index(i).unwrap()).collect()
    }

    /// A real learner wrapped to record the rows it is fit and asked to predict on, and to
    /// fail on demand.
    struct Stub {
        inner: Box<dyn LearnerFactory>,
        log: FitLog,
        fail: FailRule,
    }

    struct StubModel {
        inner: Box<dyn FittedPredictor>,
        fit_rows: Vec<usize>,
        log: FitLog,
    }

    impl LearnerFactory for Stub {
        fn task(&self) -> PredictionTask {
            self.inner.task()
        }

        fn capabilities(&self) -> LearnerCapabilities {
            self.inner.capabilities()
        }

        fn fit(
            &self,
            x: DesignView<'_>,
            y: TargetView<'_>,
            weights: Option<&[f64]>,
            ctx: &ExecutionContext,
        ) -> Result<Box<dyn FittedPredictor>, LearnError> {
            let fit_rows = physical_rows(x);
            if (self.fail)(&fit_rows) {
                return Err(LearnError::Backend("injected failure".to_string()));
            }
            Ok(Box::new(StubModel {
                inner: self.inner.fit(x, y, weights, ctx)?,
                fit_rows,
                log: Arc::clone(&self.log),
            }))
        }
    }

    impl FittedPredictor for StubModel {
        fn predict(
            &self,
            x: DesignView<'_>,
            out: &mut [f64],
            ctx: &ExecutionContext,
        ) -> Result<(), LearnError> {
            self.log.lock().unwrap().push((self.fit_rows.clone(), physical_rows(x)));
            self.inner.predict(x, out, ctx)
        }

        fn provenance(&self) -> LearnerProvenance {
            LearnerProvenance {
                spec: "stub".to_string(),
                implementation: "recording".to_string(),
                version: "test".to_string(),
            }
        }
    }

    struct Fixture {
        data: TabularData,
        treatments: Vec<VariableId>,
        outcome: VariableId,
        adjustment: Vec<VariableId>,
        cell: Vec<u32>,
        t0: Vec<f64>,
    }

    /// Two binary components, `z ~ Bernoulli(1/2)`, the cell drawn from a hand-written
    /// `P(cell | z)`, `Y = 0.4 cell + 1{cell = 3} + 0.5 z + uniform noise`.
    fn fixture(n: usize, seed: u64) -> Fixture {
        let law = [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]];
        let mut rng = CausalRng::from_seed(seed);
        let (mut z, mut t0, mut t1, mut y, mut cell) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let zi = usize::from(rng.next_f64() < 0.5);
            let u = rng.next_f64();
            let mut cumulative = 0.0;
            let mut code = 3u32;
            for (c, &p) in law[zi].iter().enumerate() {
                cumulative += p;
                if u < cumulative {
                    code = u32::try_from(c).unwrap();
                    break;
                }
            }
            z.push(zi as f64);
            t0.push(f64::from(code & 1));
            t1.push(f64::from((code >> 1) & 1));
            y.push(
                0.4 * f64::from(code)
                    + f64::from(u8::from(code == 3))
                    + 0.5 * zi as f64
                    + 0.3 * (2.0 * rng.next_f64() - 1.0),
            );
            cell.push(code);
        }
        let data = TabularData::from_f64_columns(vec![
            ("t0", t0.as_slice()),
            ("t1", t1.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let id = |name: &str| data.schema().id_of(name).unwrap();
        let (treatments, outcome, adjustment) = (vec![id("t0"), id("t1")], id("y"), vec![id("z")]);
        Fixture { data, treatments, outcome, adjustment, cell, t0 }
    }

    fn stubbed(
        outcome_fail: FailRule,
        propensity_fail: FailRule,
    ) -> (LearnedNuisance, FitLog, FitLog) {
        let (outcome_log, propensity_log) = (FitLog::default(), FitLog::default());
        let outcome_spec = LearnerSpec::Linear(LinearSpec {});
        let propensity_spec = LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 });
        let learned = LearnedNuisance {
            outcome: Box::new(Stub {
                inner: resolve_for(outcome_spec, PredictionTask::Regression).unwrap(),
                log: Arc::clone(&outcome_log),
                fail: outcome_fail,
            }),
            outcome_spec,
            propensity: Box::new(Stub {
                inner: resolve_for(propensity_spec, PredictionTask::BinaryProbability).unwrap(),
                log: Arc::clone(&propensity_log),
                fail: propensity_fail,
            }),
            propensity_spec,
        };
        (learned, outcome_log, propensity_log)
    }

    fn run_stubbed(
        fixture: &Fixture,
        learned: &LearnedNuisance,
    ) -> Result<FactorizedJointFit, EstimationError> {
        let mut config = FactorizedJointConfig::new(RidgeTuning::default());
        config.seed = 5;
        fit_with_nuisance(
            &fixture.data,
            &fixture.treatments,
            fixture.outcome,
            &fixture.adjustment,
            &[vec![0, 1]],
            &config,
            &ExecutionContext::for_tests(9),
            Some(learned),
        )
    }

    fn never() -> FailRule {
        Arc::new(|_| false)
    }

    /// A recording learner proves the cross-fitting contract: every fitted model predicts one
    /// fold's rows and none of the rows it was fit on, the predicted rows are exactly that
    /// fold, and the expected number of models was fit (3 conditionals and 4 cell outcome
    /// models, each per fold).
    #[test]
    fn a_learner_never_predicts_a_row_it_was_fit_on() {
        let fixture = fixture(1_500, 21);
        let (learned, outcome_log, propensity_log) = stubbed(never(), never());
        let fit = run_stubbed(&fixture, &learned).unwrap();
        let plan = &fit.scores.fold_ids;
        for (log, expected) in [(&outcome_log, 20), (&propensity_log, 15)] {
            let log = log.lock().unwrap();
            assert_eq!(log.len(), expected);
            for (fit_rows, predicted) in log.iter() {
                let fold = plan[predicted[0]];
                assert!(predicted.iter().all(|&i| plan[i] == fold), "one fold per model");
                assert!(fit_rows.iter().all(|&i| plan[i] != fold), "train and predict overlap");
                assert!(fit_rows.iter().all(|i| !predicted.contains(i)));
                assert_eq!(predicted.len(), plan.iter().filter(|&&f| f == fold).count());
            }
        }
        assert!(fit.scores.nuisance_provenance.contains("learners=stub/recording/test"));
        assert!(provenance_marks_learned(&fit.scores.nuisance_provenance));
        assert!(fit.scores.nuisance_provenance.contains(";fold_seed=5;ctx_seed="));
    }

    /// A failing outcome learner for one cell refuses that cell alone; a failing conditional
    /// refuses the cells that need it; the survivors keep their estimates and nothing falls
    /// back to the ridge route.
    #[test]
    fn a_failing_learner_refuses_only_the_cells_that_need_it() {
        let fixture = fixture(1_500, 22);
        let cell = fixture.cell.clone();
        let outcome_fail: FailRule = Arc::new(move |rows| rows.iter().all(|&i| cell[i] == 3));
        let (learned, _, _) = stubbed(outcome_fail, never());
        let fit = run_stubbed(&fixture, &learned).unwrap();
        for (index, report) in fit.cells.iter().enumerate() {
            match (&report.status, index) {
                (CellStatus::Unsupported(refusal), 3) => {
                    assert_eq!(refusal.detail, "joint_cells.outcome_model_unsupported");
                    assert!(refusal.message.contains("outcome learner failed"), "{refusal:?}");
                }
                (CellStatus::Supported(_), 0..=2) => {}
                (status, _) => panic!("cell {index}: unexpected {status:?}"),
            }
        }
        assert_eq!(fit.scores.n_columns(), 3);

        // Under the ordering [0, 1] the factor P(t1 | t0 = 1) is fit on rows whose t0 is one
        // only; failing it refuses the cells with t0 = 1 (codes 1 and 3).
        let t0 = fixture.t0.clone();
        let propensity_fail: FailRule = Arc::new(move |rows| rows.iter().all(|&i| t0[i] > 0.5));
        let (learned, _, _) = stubbed(never(), propensity_fail);
        let fit = run_stubbed(&fixture, &learned).unwrap();
        for (index, report) in fit.cells.iter().enumerate() {
            match (&report.status, index) {
                (CellStatus::Unsupported(refusal), 1 | 3) => {
                    assert_eq!(refusal.detail, "joint_cells.conditional_unsupported");
                    assert!(refusal.message.contains("the learner failed"), "{refusal:?}");
                }
                (CellStatus::Supported(_), 0 | 2) => {}
                (status, _) => panic!("cell {index}: unexpected {status:?}"),
            }
        }
        assert!(!fit.scores.nuisance_provenance.contains(";propensity=ridge_logistic"));

        // A learner that fails everywhere leaves no supported cell.
        let everywhere: FailRule = Arc::new(|_| true);
        let (learned, _, _) = stubbed(Arc::clone(&everywhere), everywhere);
        let EstimationError::Refused { message, .. } = run_stubbed(&fixture, &learned).unwrap_err()
        else {
            panic!("expected a refusal")
        };
        assert!(message.contains("joint_cells.no_supported_cell"), "{message}");
    }
}
