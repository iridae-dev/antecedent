//! Mechanism registry and auto-assignment.
//!
//! Assignment returns candidates and scores; there is no silent default family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::manual_let_else,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_core::{CausalRng, RoleHint, ValueType, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_kernels::shuffled_fold_assignment;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    MultinomialDesignRef, fit_glm_ridge,
};
#[cfg(feature = "gaussian-process")]
use antecedent_stats::{chol_log_det, chol_solve, cholesky_spd};

use crate::basis::{ParentBasis, column_moments, spline_knots};
use crate::batch::ParentBatch;
use crate::compile::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ParentGatherPlan,
};
use crate::error::ModelError;
use crate::mechanism::log_prob_column;

/// Candidate mechanism family known to the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MechanismFamily {
    /// Linear Gaussian additive noise (invertible).
    LinearGaussian,
    /// Constant (root or intercept-only).
    Constant,
    /// Discrete categorical (unconditional root or parent-conditional softmax).
    Discrete,
    /// Linear Gaussian plus every cross-parent product (invertible).
    ///
    /// The minimum family whose per-unit treatment effects vary: with the
    /// treatment among the parents, the `treatment × parent` columns make the
    /// contrast depend on the unit's covariates.
    LinearInteractions,
    /// Linear Gaussian in an additive cubic-spline expansion with cross-parent
    /// products of every smooth column (invertible).
    LinearSpline,
    /// Parent-conditional softmax whose logits carry every cross-parent product.
    DiscreteInteractions,
    /// Parent-conditional softmax whose logits carry an additive cubic-spline
    /// expansion with cross-parent products.
    DiscreteSpline,
    /// Hierarchical linear Gaussian (EB / group partial pooling).
    HierarchicalLinear,
    /// Hierarchical Bernoulli-logit GLM (EB / group shrinkage) → [`MechanismSlot::Discrete`].
    HierarchicalGlm,
    /// Single-equation Bayesian VAR (Minnesota prior).
    Bvar,
    /// Linear Gaussian state-space observation mechanism (Kalman 1960 filter, EM fit).
    LinearGaussianStateSpace,
    /// Gaussian-process regression mechanism (feature `gaussian-process`).
    GaussianProcess,
}

impl MechanismFamily {
    /// Registry id string.
    #[must_use]
    pub const fn id(self) -> &'static str {
        mechanism_family_id(self)
    }

    /// Whether a fit of this family on a node with at least two parents can
    /// represent effect modification — a contrast that depends on the unit's
    /// other parent values.
    ///
    /// This is a property of the *family*, asked before any fit, so a selection
    /// that scored such a family and rejected it can be told apart from one
    /// that never had the option. [`MechanismSlot::admits_no_effect_modification`]
    /// answers the same question of a fitted slot.
    #[must_use]
    pub const fn can_modify_effects(self) -> bool {
        match self {
            Self::Discrete
            | Self::HierarchicalGlm
            | Self::GaussianProcess
            | Self::LinearInteractions
            | Self::LinearSpline
            | Self::DiscreteInteractions
            | Self::DiscreteSpline => true,
            Self::LinearGaussian
            | Self::Constant
            | Self::HierarchicalLinear
            | Self::Bvar
            | Self::LinearGaussianStateSpace => false,
        }
    }

    /// Whether this family's fitted mechanism keeps an additive disturbance, so
    /// counterfactual abduction inverts it exactly
    /// (`NoiseInferenceMode::Invertible`).
    ///
    /// The heterogeneity-capable continuous families answer `true`: their
    /// expansion moves the conditional mean only. The categorical families
    /// answer `false` and are recorded as such — their abduction was already a
    /// posterior draw from the observed category's CDF bin before this family
    /// existed, so selecting one is not a downgrade, but it is not exact
    /// inversion either.
    #[must_use]
    pub const fn has_additive_disturbance(self) -> bool {
        match self {
            Self::LinearGaussian
            | Self::Constant
            | Self::HierarchicalLinear
            | Self::Bvar
            | Self::GaussianProcess
            | Self::LinearInteractions
            | Self::LinearSpline => true,
            Self::Discrete
            | Self::HierarchicalGlm
            | Self::DiscreteInteractions
            | Self::DiscreteSpline
            | Self::LinearGaussianStateSpace => false,
        }
    }
}

/// Scored candidate for one node.
#[derive(Clone, Debug)]
pub struct MechanismCandidate {
    /// Family.
    pub family: MechanismFamily,
    /// Mean held-out predictive log-likelihood (nats per row, on shared cross-validation
    /// folds) less a `k ln n / 2n` parameter-count penalty; higher is better. Differences
    /// between families are free of the outcome's units. A degenerate `Constant` scores `0`.
    pub score: f64,
    /// Estimated fit cost (relative).
    pub fit_cost: f64,
    /// Estimated evaluation cost (relative).
    pub eval_cost: f64,
}

/// Result of auto-assignment for one node.
#[derive(Clone, Debug)]
pub struct MechanismAssignment {
    /// Dense node.
    pub node: DenseNodeId,
    /// Variable.
    pub variable: VariableId,
    /// All scored candidates (sorted descending by score).
    pub candidates: Arc<[MechanismCandidate]>,
    /// Selected family (must be chosen explicitly from candidates).
    pub selected: MechanismFamily,
    /// Which rule sent the node to the discrete or the continuous family list.
    pub typing: MechanismTyping,
    /// Fitted slot.
    pub fitted: MechanismSlot,
    /// Families that failed to score/fit, with error messages.
    pub failed_families: Arc<[(MechanismFamily, String)]>,
}

/// Which rule decided that a node draws its mechanism from the discrete or the
/// continuous family list.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MechanismTyping {
    /// The schema declares a discrete value type (binary, categorical or
    /// ordinal), so the node is categorical however many levels it has.
    DeclaredDiscrete,
    /// The declared type does not assert a discrete variable (continuous or
    /// count) but the column has at most [`LOW_CARDINALITY_MAX_LEVELS`] levels,
    /// each observed at least [`LOW_CARDINALITY_MIN_ROWS_PER_LEVEL`] times on
    /// average: a categorical mechanism is the better description.
    LowCardinality,
    /// Continuous mechanism families.
    Continuous,
}

/// Most distinct values a column not declared discrete may take and still be
/// typed categorical.
pub const LOW_CARDINALITY_MAX_LEVELS: usize = 8;

/// Average rows per level below which a short column is not read as
/// categorical: a level set seen once or twice per level is a small sample of
/// a continuous variable, and a categorical fit would restrict counterfactual
/// support to the handful of values observed.
pub const LOW_CARDINALITY_MIN_ROWS_PER_LEVEL: usize = 5;

/// Registry of mechanism families.
#[derive(Clone, Debug)]
pub struct MechanismRegistry {
    /// Families considered for continuous nodes.
    pub continuous: Arc<[MechanismFamily]>,
    /// Families considered for discrete / low-cardinality nodes.
    pub discrete: Arc<[MechanismFamily]>,
}

impl Default for MechanismRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

/// Registry id string per family, kept next to the family preset lists
/// ([`MechanismRegistry::standard`] / [`MechanismRegistry::with_bayesian_families`]) so
/// adding a family updates the preset(s) and this table in the same place.
///
/// Purely descriptive — [`score_family`] and [`fit_family`] stay exhaustive `match`es
/// (each family's fit is a distinct statistical procedure) and are not table-driven.
const fn mechanism_family_id(family: MechanismFamily) -> &'static str {
    match family {
        MechanismFamily::LinearGaussian => "linear_gaussian",
        MechanismFamily::Constant => "constant",
        MechanismFamily::Discrete => "discrete",
        MechanismFamily::LinearInteractions => "linear_interactions",
        MechanismFamily::LinearSpline => "linear_spline",
        MechanismFamily::DiscreteInteractions => "discrete_interactions",
        MechanismFamily::DiscreteSpline => "discrete_spline",
        MechanismFamily::HierarchicalLinear => "hierarchical_linear",
        MechanismFamily::HierarchicalGlm => "hierarchical_glm",
        MechanismFamily::Bvar => "bvar",
        MechanismFamily::LinearGaussianStateSpace => "lgssm",
        MechanismFamily::GaussianProcess => "gaussian_process",
    }
}

impl MechanismRegistry {
    /// Standard registry (core families).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            continuous: Arc::from(vec![MechanismFamily::LinearGaussian, MechanismFamily::Constant]),
            discrete: Arc::from(vec![MechanismFamily::Discrete, MechanismFamily::Constant]),
        }
    }

    /// Standard registry plus the heterogeneity-capable families.
    ///
    /// Used by the counterfactual path, where a per-unit effect that cannot
    /// vary is a property of the candidate set and not of the data. The extra
    /// families are additive-disturbance (continuous) or already-categorical
    /// (discrete), so abduction keeps the mode it had; they are scored on
    /// cross-validated error and win only where they earn it.
    #[must_use]
    pub fn with_heterogeneity_families() -> Self {
        Self {
            continuous: Arc::from(vec![
                MechanismFamily::LinearGaussian,
                MechanismFamily::LinearInteractions,
                MechanismFamily::LinearSpline,
                MechanismFamily::Constant,
            ]),
            discrete: Arc::from(vec![
                MechanismFamily::Discrete,
                MechanismFamily::DiscreteInteractions,
                MechanismFamily::DiscreteSpline,
                MechanismFamily::Constant,
            ]),
        }
    }

    /// Extended continuous registry including hierarchical / BVAR / LGSSM / GP.
    #[must_use]
    pub fn with_bayesian_families() -> Self {
        #[cfg(feature = "gaussian-process")]
        let continuous = {
            let mut continuous = vec![
                MechanismFamily::LinearGaussian,
                MechanismFamily::HierarchicalLinear,
                MechanismFamily::Bvar,
                MechanismFamily::LinearGaussianStateSpace,
                MechanismFamily::Constant,
            ];
            continuous.insert(continuous.len() - 1, MechanismFamily::GaussianProcess);
            continuous
        };
        #[cfg(not(feature = "gaussian-process"))]
        let continuous = vec![
            MechanismFamily::LinearGaussian,
            MechanismFamily::HierarchicalLinear,
            MechanismFamily::Bvar,
            MechanismFamily::LinearGaussianStateSpace,
            MechanismFamily::Constant,
        ];
        let discrete = vec![
            MechanismFamily::Discrete,
            MechanismFamily::HierarchicalGlm,
            MechanismFamily::Constant,
        ];
        Self { continuous: Arc::from(continuous), discrete: Arc::from(discrete) }
    }

    /// Refit standard mechanisms under continuous row weights, conditional on
    /// the originally selected families. This preserves the original units and
    /// discrete support instead of drawing a second random data set.
    ///
    /// # Errors
    /// Invalid weights, nonstandard assignments, or weighted fitting failures.
    pub fn refit_weighted(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        assignments: &[MechanismAssignment],
        weights: &[f64],
    ) -> Result<CompiledMechanismStore, ModelError> {
        let total: f64 = weights.iter().sum();
        if weights.len() != data.row_count()
            || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
            || !total.is_finite()
            || total <= 0.0
            || assignments.len() != model.n_nodes()
        {
            return Err(ModelError::Shape {
                message: "invalid mechanism weights or assignments".into(),
            });
        }
        let weights: Vec<f64> =
            weights.iter().map(|w| w * data.row_count() as f64 / total).collect();
        let mut slots = vec![MechanismSlot::Vacant; model.n_nodes()];
        let mut ws = LeastSquaresWorkspace::default();
        for gather in model.parent_gathers.iter() {
            let assignment =
                assignments.iter().find(|a| a.node == gather.child).ok_or_else(|| {
                    ModelError::Shape { message: "missing mechanism assignment".into() }
                })?;
            if !matches!(
                assignment.selected,
                MechanismFamily::Constant
                    | MechanismFamily::LinearGaussian
                    | MechanismFamily::Discrete
                    | MechanismFamily::LinearInteractions
                    | MechanismFamily::LinearSpline
                    | MechanismFamily::DiscreteInteractions
                    | MechanismFamily::DiscreteSpline
            ) {
                return Err(ModelError::Unsupported {
                    message: "weighted refit requires standard mechanisms".into(),
                });
            }
            let var = model.output_layout.variables[gather.child.as_usize()];
            let y = data.float64_cow(var)?;
            slots[gather.child.as_usize()] = fit_family_weighted(
                assignment.selected,
                gather,
                model,
                data,
                &y,
                FaerBackend,
                &mut ws,
                Some(&weights),
            )?;
        }
        Ok(CompiledMechanismStore { slots: Arc::from(slots) })
    }

    /// Assign and fit all nodes. Requires an explicit selection policy.
    ///
    /// # Errors
    ///
    /// Data / fit failures, or empty candidate sets.
    pub fn assign_and_fit(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        let n = model.n_nodes();
        let nrows = data.row_count();
        if nrows == 0 {
            return Err(ModelError::Shape { message: "empty data for mechanism fit".into() });
        }
        let mut slots = vec![MechanismSlot::Vacant; n];
        let mut assignments = Vec::with_capacity(n);
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();

        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            let var = model.output_layout.variables[node.as_usize()];
            let y = data.float64_cow(var).map_err(ModelError::from)?;
            let typing = classify_node(data, var, &y);
            let families: &[MechanismFamily] = if typing == MechanismTyping::Continuous {
                &self.continuous
            } else {
                &self.discrete
            };
            let frame = FitFrame::from_data(gather, model, data, y.len())?;

            let mut candidates = Vec::new();
            let mut fits: Vec<(MechanismFamily, MechanismSlot)> = Vec::new();
            let mut failed = Vec::new();
            for &family in families {
                match score_family(family, &frame, &y, backend, &mut ls_ws) {
                    // A non-finite score is not a low score: it is the family's own admissibility
                    // gate (see `CONSTANT_FAMILY_MAX_VARIANCE` above) reporting that the fit is
                    // not a candidate at all. Treat it exactly like an `Err` from the fit itself —
                    // it must never win selection just because everything else also failed.
                    Ok((c, _)) if !c.score.is_finite() => {
                        failed.push((family, format!("score not finite ({})", c.score)));
                    }
                    Ok((c, slot)) => {
                        candidates.push(c);
                        fits.push((family, slot));
                    }
                    Err(e) => failed.push((family, e.to_string())),
                }
            }
            if candidates.is_empty() {
                let detail = failed
                    .iter()
                    .map(|(f, e)| format!("{f:?}: {e}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(ModelError::Unsupported {
                    message: format!(
                        "no mechanism candidates for variable {var} (failures: {detail})"
                    ),
                });
            }
            // Every score here is finite (non-finite ones were moved to `failed` above), so
            // the order is total; exact ties keep the registry's listing order, which lists
            // the simpler family first.
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let selected = policy.select(&candidates).ok_or_else(|| ModelError::Unsupported {
                message: "selection policy produced no family".into(),
            })?;
            // The scoring fit for the selected family is the fit (same inputs,
            // deterministic solvers); reuse it instead of refitting from scratch.
            let fitted = fits
                .into_iter()
                .find_map(|(family, slot)| (family == selected).then_some(slot))
                .ok_or_else(|| ModelError::Unsupported {
                    message: "selected family missing a retained fit".into(),
                })?;
            slots[node.as_usize()] = fitted.clone();
            assignments.push(MechanismAssignment {
                node,
                variable: var,
                candidates: Arc::from(candidates),
                selected,
                typing,
                fitted,
                failed_families: Arc::from(failed),
            });
        }

        Ok((CompiledMechanismStore { slots: Arc::from(slots) }, assignments))
    }
}

/// How to pick among scored candidates (no silent fallback).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectionPolicy {
    /// Highest score: mean held-out predictive log-likelihood (every family
    /// refit on the training folds of one shared deterministic partition) less a
    /// parameter-count complexity penalty.
    BestScore,
    /// Require the named family to appear; error if missing.
    RequireFamily(MechanismFamily),
}

impl SelectionPolicy {
    /// Select a family.
    #[must_use]
    pub fn select(self, candidates: &[MechanismCandidate]) -> Option<MechanismFamily> {
        match self {
            Self::BestScore => candidates.first().map(|c| c.family),
            Self::RequireFamily(fam) => {
                candidates.iter().find(|c| c.family == fam).map(|c| c.family)
            }
        }
    }
}

/// Number of distinct values (1e-6 quantization, non-finite skipped) when it is
/// at most `max_levels`, else `None`; `None` also for an all-non-finite column.
///
/// Early-exiting scan: at most `max_levels` keys are ever retained, and the
/// (`max_levels` + 1)-th distinct key bails out immediately, so no full-column
/// sort or copy.
fn distinct_level_count(y: &[f64], max_levels: usize) -> Option<usize> {
    let mut seen: Vec<i64> = Vec::with_capacity(max_levels.min(64));
    for v in y.iter().filter(|v| v.is_finite()) {
        let key = (v * 1e6).round() as i64;
        if !seen.contains(&key) {
            if seen.len() == max_levels {
                return None;
            }
            seen.push(key);
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}

#[cfg(test)]
fn is_low_cardinality(y: &[f64], max_levels: usize) -> bool {
    distinct_level_count(y, max_levels).is_some()
}

/// Decide which family list a node draws from.
///
/// A declared discrete value type (binary, categorical, ordinal) is
/// authoritative. Otherwise the level count decides, but only when the levels
/// are observed repeatedly ([`LOW_CARDINALITY_MIN_ROWS_PER_LEVEL`]): a short
/// column of a continuous variable must not become a multinomial whose
/// counterfactual support is the few values that happened to be observed.
fn classify_node(data: &TabularData, var: VariableId, y: &[f64]) -> MechanismTyping {
    let declared_discrete = data.schema().get(var).is_ok_and(|v| {
        matches!(v.value_type, ValueType::Binary | ValueType::Categorical | ValueType::Ordinal)
    });
    if declared_discrete {
        return MechanismTyping::DeclaredDiscrete;
    }
    match distinct_level_count(y, LOW_CARDINALITY_MAX_LEVELS) {
        Some(levels) if y.len() >= levels * LOW_CARDINALITY_MIN_ROWS_PER_LEVEL => {
            MechanismTyping::LowCardinality
        }
        _ => MechanismTyping::Continuous,
    }
}

/// Spread, relative to the outcome's largest magnitude, above which
/// [`MechanismFamily::Constant`] is inadmissible.
///
/// `Constant` claims the variable is deterministic. A held-out likelihood cannot
/// score a point mass against a density, so its admissibility is decided on the
/// claim itself: any spread above numerical noise means the claim is false, and it
/// would silently strip every root of its distribution. The threshold is relative so
/// a variable measured in small units is not declared deterministic for its units.
const CONSTANT_FAMILY_MAX_RELATIVE_SD: f64 = 1e-6;

/// Folds of the deterministic cross-validation that scores every family.
pub const MECHANISM_CV_FOLDS: usize = 5;

/// Seed of the partition behind [`MECHANISM_CV_FOLDS`]. Fixed so the score of a
/// family is a function of the table alone; not a user-facing random stream.
const CV_FOLD_SEED: u64 = 0x4D45_4348_4356_0001;

/// Column view a mechanism fit reads: the parent columns of one node and the
/// optional unit grouping. Fits and their cross-validation refits go through
/// this view, so a training fold is a row subset of the same object rather than
/// a second table.
struct FitFrame<'d> {
    parents: Vec<Cow<'d, [f64]>>,
    /// Raw `UnitId` labels per row (the group structure is validated where used).
    groups: Option<Vec<u32>>,
    n: usize,
}

impl<'d> FitFrame<'d> {
    fn from_data(
        gather: &ParentGatherPlan,
        model: &CompiledCausalModel,
        data: &'d TabularData,
        n: usize,
    ) -> Result<Self, ModelError> {
        let parents = gather_parent_cols(gather, model, data)?;
        if parents.iter().any(|c| c.len() < n) {
            return Err(ModelError::Shape {
                message: "parent column shorter than the outcome".into(),
            });
        }
        Ok(Self { parents, groups: unit_id_groups(data, n), n })
    }

    fn n_parents(&self) -> usize {
        self.parents.len()
    }

    /// Owned frame over `rows` (in the given order).
    fn select(&self, rows: &[usize]) -> FitFrame<'static> {
        FitFrame {
            parents: self
                .parents
                .iter()
                .map(|c| Cow::Owned(rows.iter().map(|&r| c[r]).collect::<Vec<f64>>()))
                .collect(),
            groups: self.groups.as_ref().map(|g| rows.iter().map(|&r| g[r]).collect()),
            n: rows.len(),
        }
    }

    /// Same rows, replaced parent columns (e.g. within-group demeaned).
    fn with_parents(&self, parents: Vec<Vec<f64>>) -> FitFrame<'static> {
        FitFrame {
            parents: parents.into_iter().map(Cow::Owned).collect(),
            groups: self.groups.clone(),
            n: self.n,
        }
    }

    /// Column-major parent matrix `[parent * n + row]`.
    fn column_major(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * self.n_parents());
        for c in &self.parents {
            out.extend_from_slice(&c[..self.n]);
        }
        out
    }
}

fn family_is_categorical(family: MechanismFamily) -> bool {
    matches!(
        family,
        MechanismFamily::Discrete
            | MechanismFamily::DiscreteInteractions
            | MechanismFamily::DiscreteSpline
            | MechanismFamily::HierarchicalGlm
    )
}

/// Dense stratum id per row for a categorical outcome (quantized value order).
fn category_strata(y: &[f64]) -> Vec<u32> {
    let key = |v: f64| (v * 1e6).round() as i64;
    let mut keys: Vec<i64> = y.iter().map(|&v| key(v)).collect();
    keys.sort_unstable();
    keys.dedup();
    y.iter().map(|&v| keys.binary_search(&key(v)).map_or(0, |i| i as u32)).collect()
}

/// Seeded fold of every row, shared by every family scored on the same node so
/// candidates are compared on identical training/held-out splits. Stratified on
/// the outcome for categorical families so each training fold keeps every level.
fn cv_fold_plan(n: usize, strata: Option<&[u32]>) -> Result<Vec<usize>, ModelError> {
    let folds = MECHANISM_CV_FOLDS.min(n);
    if folds < 2 {
        return Err(ModelError::Unsupported {
            message: "not enough rows to cross-validate a mechanism family".into(),
        });
    }
    let mut rng = CausalRng::from_seed(CV_FOLD_SEED);
    Ok(shuffled_fold_assignment(&mut rng, n, folds, strata))
}

/// Mean held-out predictive log-likelihood of `family`, refit on each training
/// fold and scored on the rows it did not see.
///
/// One criterion for every family: the log-likelihood is a density, so
/// rescaling the outcome shifts every family's score by the same `ln(scale)` and
/// cannot change which family wins. Held-out rows of a categorical outcome whose
/// level is absent from a training fold are not scored (a sampling zero, not
/// evidence against the family).
fn oof_mean_loglik(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<f64, ModelError> {
    let n = y.len();
    let categorical = family_is_categorical(family);
    let strata = categorical.then(|| category_strata(y));
    let fold = cv_fold_plan(n, strata.as_deref())?;
    let folds = fold.iter().copied().max().map_or(0, |m| m + 1);
    let p = frame.n_parents();
    let mut total = 0.0;
    let mut scored = 0usize;
    let mut last_fold_error: Option<ModelError> = None;
    for k in 0..folds {
        let (train, test): (Vec<usize>, Vec<usize>) = (0..n).partition(|&r| fold[r] != k);
        if train.len() < 2 || test.is_empty() {
            continue;
        }
        let y_train: Vec<f64> = train.iter().map(|&r| y[r]).collect();
        // A fold whose training rows cannot be fit (a quasi-separated multinomial, a
        // parent constant within the fold) is not scored rather than failing the family:
        // the mechanism itself is the full-data fit, which already succeeded. If no fold
        // can be scored the family is refused with the last fit error.
        let slot =
            match fit_frame_family(family, &frame.select(&train), &y_train, backend, ls_ws, None) {
                Ok(slot) => slot,
                Err(e) => {
                    last_fold_error = Some(e);
                    continue;
                }
            };
        let held_out = frame.select(&test);
        let y_test: Vec<f64> = test.iter().map(|&r| y[r]).collect();
        let values = held_out.column_major();
        let mut lp = vec![0.0; test.len()];
        log_prob_column(
            &slot,
            &y_test,
            ParentBatch { n_rows: test.len(), n_parents: p, values: &values },
            &mut lp,
        )?;
        for v in lp {
            if categorical && v == f64::NEG_INFINITY {
                continue;
            }
            total += v;
            scored += 1;
        }
    }
    if scored == 0 {
        return Err(last_fold_error.unwrap_or_else(|| ModelError::Unsupported {
            message: "not enough rows to cross-validate a mechanism family".into(),
        }));
    }
    Ok(total / scored as f64)
}

/// Out-of-fold residuals of the ordinary-least-squares parent mean (the linear
/// part of a conditional state-space mechanism).
fn oof_linear_residuals(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<Vec<f64>, ModelError> {
    let n = y.len();
    let fold = cv_fold_plan(n, None)?;
    let folds = fold.iter().copied().max().map_or(0, |m| m + 1);
    let mut residual = vec![0.0; n];
    for k in 0..folds {
        let (train, test): (Vec<usize>, Vec<usize>) = (0..n).partition(|&r| fold[r] != k);
        if train.len() < 2 || test.is_empty() {
            continue;
        }
        let y_train: Vec<f64> = train.iter().map(|&r| y[r]).collect();
        let slot = fit_frame_family(
            MechanismFamily::LinearGaussian,
            &frame.select(&train),
            &y_train,
            backend,
            ls_ws,
            None,
        )?;
        let MechanismSlot::LinearGaussian { intercept, coeffs, .. } = slot else {
            return Err(ModelError::Unsupported { message: "linear reference fit failed".into() });
        };
        for &r in &test {
            let mut pred = intercept;
            for (c, col) in frame.parents.iter().enumerate() {
                pred += coeffs[c] * col[r];
            }
            residual[r] = y[r] - pred;
        }
    }
    Ok(residual)
}

/// Free parameters of a fitted mechanism, for the complexity penalty.
fn parameter_count(slot: &MechanismSlot, n_parents: usize) -> f64 {
    let p = n_parents;
    let count = match slot {
        MechanismSlot::LinearGaussian { .. }
        | MechanismSlot::HierarchicalLinear { .. }
        | MechanismSlot::Bvar { .. } => p + 2,
        MechanismSlot::LinearBasis { basis, .. } => basis.n_terms() + 2,
        MechanismSlot::Discrete { support, logit_coeffs, .. } => {
            (support.len().saturating_sub(1)) * if logit_coeffs.is_some() { 1 + p } else { 1 }
        }
        MechanismSlot::DiscreteBasis { support, basis, .. } => {
            support.len().saturating_sub(1) * (1 + basis.n_terms())
        }
        MechanismSlot::ConditionalLinearGaussianStateSpace { .. } => p + 1 + 4,
        MechanismSlot::LinearGaussianStateSpace { .. } => 4,
        // length scale, noise and the prior mean; the interpolant's effective
        // freedom is already charged by the held-out likelihood.
        MechanismSlot::GaussianProcess { .. } => 3,
        _ => 1,
    };
    count as f64
}

fn score_family(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<(MechanismCandidate, MechanismSlot), ModelError> {
    let fitted = fit_frame_family(family, frame, y, backend, ls_ws, None)?;
    let n = y.len();
    let p = frame.n_parents();
    // Every family is scored by the same criterion, mean held-out predictive
    // log-likelihood on one shared fold partition, less a BIC-scaled
    // parameter-count penalty (`k ln n / 2n` per row). Both terms are free of the
    // outcome's units: rescaling `y` shifts every family's log-density by the same
    // `ln(scale)`, and the penalty depends on counts only. Nothing is scored on
    // its own training rows, so a richer family (basis expansion, GP) wins only by
    // predicting rows it did not fit.
    let penalty = 0.5 * parameter_count(&fitted, p) * (n.max(2) as f64).ln() / n.max(1) as f64;
    let score = match &fitted {
        MechanismSlot::Constant { value } => {
            let sd =
                (y.iter().map(|yi| (yi - value).powi(2)).sum::<f64>() / n.max(1) as f64).sqrt();
            let max_abs = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            // A `Constant` mechanism asserts the variable is *deterministic*: it carries no
            // distribution at all. A held-out density cannot score a point mass, and squared
            // error ties the best-fitting alternative on a root (`LinearGaussian` with
            // `intercept = mean`, `sigma = SD` has exactly this MSE), which used to make every
            // root deterministic and turn interventions on a root's mechanism into no-ops. So
            // gate on the claim `Constant` is making: it is admissible only when the target
            // really is degenerate, and then its log-mass on its own single value is 0 — the
            // score a one-level discrete fit earns, so the two tie and the listing order breaks
            // the tie.
            if sd > CONSTANT_FAMILY_MAX_RELATIVE_SD * max_abs { f64::NEG_INFINITY } else { 0.0 }
        }
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean }
        | MechanismSlot::ConditionalLinearGaussianStateSpace {
            a,
            process_std,
            obs_std,
            initial_mean,
            ..
        } => {
            // The Kalman filter's one-step-ahead predictive density at row `t` uses only
            // `y_{<t}` (prequential, so out of sample in the state), and the parent mean
            // it is applied to is the out-of-fold linear prediction. Row-shuffled folds
            // cannot refit the state model because the rows are a series.
            let residual = if matches!(fitted, MechanismSlot::LinearGaussianStateSpace { .. }) {
                y.to_vec()
            } else {
                oof_linear_residuals(frame, y, backend, ls_ws)?
            };
            let state = MechanismSlot::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            };
            let mut lp = vec![0.0; n];
            log_prob_column(&state, &residual, ParentBatch::empty(n), &mut lp)?;
            lp.iter().sum::<f64>() / n.max(1) as f64 - penalty
        }
        _ => oof_mean_loglik(family, frame, y, backend, ls_ws)? - penalty,
    };
    Ok((
        MechanismCandidate { family, score, fit_cost: 1.0 + p as f64, eval_cost: 1.0 + p as f64 },
        fitted,
    ))
}

#[cfg(test)]
fn fit_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    fit_family_weighted(family, gather, model, data, y, backend, ls_ws, None)
}

#[allow(clippy::too_many_arguments)]
fn fit_family_weighted(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let frame = FitFrame::from_data(gather, model, data, y.len())?;
    fit_frame_family(family, &frame, y, backend, ls_ws, weights)
}

fn fit_frame_family(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    match family {
        MechanismFamily::Constant => {
            let mean =
                y.iter().enumerate().map(|(i, v)| v * weights.map_or(1.0, |w| w[i])).sum::<f64>()
                    / weights.map_or(n.max(1) as f64, |w| w.iter().sum());
            Ok(MechanismSlot::Constant { value: mean })
        }
        MechanismFamily::Discrete => {
            let (support, probs) = discrete_support(y, weights)?;
            let k = support.len();
            let p = frame.n_parents();
            if p == 0 {
                return Ok(MechanismSlot::Discrete {
                    support: Arc::from(support),
                    probs: Arc::from(probs),
                    logit_coeffs: None,
                });
            }
            // Parent-conditional: baseline-category multinomial logit MLE (Fisher / IRLS).
            // Coefficients are true softmax logits; category 0 is the reference (zeros).
            //
            // The IRLS is run on **standardized** parent columns and the fitted
            // logits are transformed back to the raw scale afterwards. A
            // multinomial logit is equivariant under an affine change of the
            // design, so the fitted conditional law is identical either way —
            // but its *numerics* are not: on columns that differ in scale by
            // orders of magnitude (a raw year column beside a 0–1 axis) the
            // Fisher step is ill-conditioned and the iteration stalls short of
            // the tolerance. Standardizing removes a numerical property of the
            // optimizer from the list of things that decide whether a licensed
            // cell can run at all. The stored coefficients are on the original
            // parent scale, so nothing downstream changes shape.
            let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
            let (centers, scales) = column_moments(&refs, n);
            let ncols = 1 + p;
            let mut x = vec![0.0; n * ncols];
            x[..n].fill(1.0);
            for (pi, col) in refs.iter().enumerate() {
                let base = (1 + pi) * n;
                for r in 0..n {
                    x[base + r] = (col[r] - centers[pi]) / scales[pi];
                }
            }
            let y_cat = discrete_categories(y, &support)?;
            let fit = antecedent_stats::fit_multinomial_logit_weighted(
                MultinomialDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols,
                    y_category: &y_cat,
                    n_categories: k,
                },
                weights,
                &backend,
                ls_ws,
                &GlmOptions::default(),
            )?;
            // Refuse non-converged fits; separation is allowed (near-deterministic
            // conditionals → large logits; softmax evaluation remains well-defined).
            if !fit.converged {
                return Err(ModelError::NotConverged {
                    message: format!(
                        "multinomial logit did not converge on standardized parents \
                         (iters={}, deviance={})",
                        fit.iterations, fit.deviance
                    ),
                });
            }
            // Back-transform: η = b₀ + Σ bⱼ (xⱼ − cⱼ)/sⱼ
            //                   = (b₀ − Σ bⱼ cⱼ/sⱼ) + Σ (bⱼ/sⱼ) xⱼ.
            let mut coefficients = fit.coefficients;
            for cat in 0..k {
                let base = cat * ncols;
                let mut shift = 0.0;
                for pi in 0..p {
                    let raw = coefficients[base + 1 + pi] / scales[pi];
                    shift += raw * centers[pi];
                    coefficients[base + 1 + pi] = raw;
                }
                coefficients[base] -= shift;
            }
            Ok(MechanismSlot::Discrete {
                support: Arc::from(support),
                probs: Arc::from(probs),
                logit_coeffs: Some(Arc::from(coefficients)),
            })
        }
        MechanismFamily::LinearGaussian => {
            fit_linear_gaussian_weighted(frame, y, backend, ls_ws, 0.0, weights)
        }
        MechanismFamily::LinearInteractions | MechanismFamily::LinearSpline => {
            let basis = build_basis(family, frame, n)?;
            fit_linear_basis(&basis, frame, y, backend, ls_ws, weights)
        }
        MechanismFamily::DiscreteInteractions | MechanismFamily::DiscreteSpline => {
            let basis = build_basis(family, frame, n)?;
            fit_discrete_basis(&basis, frame, y, backend, ls_ws, weights)
        }
        MechanismFamily::HierarchicalLinear => fit_hierarchical_linear(frame, y, backend, ls_ws),
        MechanismFamily::HierarchicalGlm => fit_hierarchical_glm(frame, y, backend, ls_ws),
        MechanismFamily::Bvar => fit_bvar_minnesota(frame, y, backend, ls_ws),
        MechanismFamily::LinearGaussianStateSpace => fit_lgssm_kalman_em(frame, y, backend, ls_ws),
        MechanismFamily::GaussianProcess => {
            #[cfg(feature = "gaussian-process")]
            {
                fit_gaussian_process(frame, y)
            }
            #[cfg(not(feature = "gaussian-process"))]
            {
                let _ = (frame, y, backend, ls_ws);
                Err(ModelError::Unsupported {
                    message: "GaussianProcess requires feature `gaussian-process`".into(),
                })
            }
        }
    }
}

fn gather_parent_cols<'d>(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &'d TabularData,
) -> Result<Vec<Cow<'d, [f64]>>, ModelError> {
    let mut parent_cols = Vec::with_capacity(gather.n_parents());
    for &parent in gather.parents.iter() {
        let var = model.output_layout.variables[parent.as_usize()];
        parent_cols.push(data.float64_cow(var).map_err(ModelError::from)?);
    }
    Ok(parent_cols)
}

/// Relative floor of a fitted residual standard deviation: a fraction of the
/// outcome's own RMS, so a deterministic relation measured in small units keeps a
/// σ at the data's scale instead of an absolute `1e-8` that may exceed it.
fn sigma_floor(y: &[f64]) -> f64 {
    let rms = (y.iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64).sqrt();
    (1e-8 * rms).max(f64::MIN_POSITIVE)
}

/// Residual standard deviation from a data-row residual sum of squares.
fn residual_sigma(rss: f64, n: usize, ncols: usize, y: &[f64]) -> f64 {
    (rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(sigma_floor(y))
}

/// Residual sum of squares of `[1 | parents]·(intercept, coeffs)` over the data rows
/// only (no pseudo-observations), weighted when `weights` are given.
fn data_rss(
    frame: &FitFrame<'_>,
    y: &[f64],
    intercept: f64,
    coeffs: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    let mut rss = 0.0;
    for r in 0..y.len() {
        let mut pred = intercept;
        for (c, col) in frame.parents.iter().enumerate() {
            pred += coeffs[c] * col[r];
        }
        rss += weights.map_or(1.0, |w| w[r]) * (y[r] - pred).powi(2);
    }
    rss
}

/// Whether raw `UnitId` labels define a usable grouping: at least two groups, and
/// fewer groups than rows (otherwise every group mean is its own observation).
fn group_structure_usable(groups: &[u32]) -> bool {
    let mut uniq = groups.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    uniq.len() >= 2 && uniq.len() < groups.len()
}

/// Empirical-Bayes hierarchical linear: estimate τ² / λ from OLS; with a `UnitId`
/// column, the slopes come from the within-unit (fixed-effects) regression.
fn fit_hierarchical_linear(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let ols = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let MechanismSlot::LinearGaussian { coeffs: ols_coeffs, sigma: ols_sigma, .. } = ols else {
        return Err(ModelError::Unsupported { message: "hierarchical base fit failed".into() });
    };
    let p = ols_coeffs.len();
    // Method-of-moments EB: τ² ≈ mean(β̂²) − σ²·mean(diag((X'X)^{-1})) proxy;
    // use simplified τ² = mean(β̂²) clamped, λ = σ² / τ².
    let mean_b2 =
        if p == 0 { 0.0 } else { ols_coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64 };
    let tau2 = (mean_b2 - ols_sigma * ols_sigma / n.max(1) as f64).max(1e-8);
    let mut lambda = (ols_sigma * ols_sigma / tau2).clamp(1e-6, 1e6);

    // Optional UnitId random intercept. The within estimator demeans the outcome *and*
    // every parent by unit: demeaning only `y` regresses it on a parent that still
    // carries its between-unit variance, attenuating the slope by
    // `Var_within(x)/Var(x)`. The between-unit spread of the outcome is not
    // discarded: the residual σ below is measured on the population-intercept mean
    // function, so it carries that spread, exactly as abduction (`y − mean`) and the
    // log-density will see it.
    let grouped = frame.groups.as_deref().filter(|g| group_structure_usable(g));
    let slot = if let Some(groups) = grouped {
        let y_work = demean_by_group(y, groups);
        let within = frame
            .with_parents(frame.parents.iter().map(|c| demean_by_group(&c[..n], groups)).collect());
        // Re-estimate λ on the demeaned series.
        let ols2 = fit_linear_gaussian(&within, &y_work, backend, ls_ws, 0.0)?;
        if let MechanismSlot::LinearGaussian { coeffs, sigma, .. } = ols2 {
            let mean_b2 = if coeffs.is_empty() {
                0.0
            } else {
                coeffs.iter().map(|b| b * b).sum::<f64>() / coeffs.len() as f64
            };
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            lambda = (sigma * sigma / tau2).clamp(1e-6, 1e6);
        }
        fit_linear_gaussian(&within, &y_work, backend, ls_ws, lambda)?
    } else {
        fit_linear_gaussian(frame, y, backend, ls_ws, lambda)?
    };
    let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = slot else {
        return Ok(slot);
    };
    let (intercept, sigma) = if grouped.is_some() {
        // Population intercept, then σ on the same mean function every downstream
        // consumer evaluates.
        let mean_y = y.iter().sum::<f64>() / n.max(1) as f64;
        let mut intercept = mean_y;
        for (c, col) in frame.parents.iter().enumerate() {
            intercept -= coeffs[c] * col[..n].iter().sum::<f64>() / n.max(1) as f64;
        }
        let rss = data_rss(frame, y, intercept, &coeffs, None);
        (intercept, residual_sigma(rss, n, 1 + p, y))
    } else {
        (intercept, sigma)
    };
    Ok(MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage: lambda })
}

/// Hierarchical Bernoulli logit with always-on empirical-Bayes ridge.
///
/// EB λ is estimated from linear-probability OLS moments and passed to
/// [`fit_glm_ridge`], which applies the penalty on ordinary (non-separated) data
/// as well as separated cases (MM-014). Intercept is left unpenalized.
fn fit_hierarchical_glm(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let binary = y.iter().all(|&yi| yi == 0.0 || yi == 1.0);
    if !binary {
        return Err(ModelError::Unsupported {
            message: "HierarchicalGlm requires binary {0,1} outcomes".into(),
        });
    }
    // EB λ from linear-probability OLS moments.
    let ols = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let lambda = match &ols {
        MechanismSlot::LinearGaussian { coeffs, sigma, .. } => {
            let p = coeffs.len().max(1);
            let mean_b2 = coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64;
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            (sigma * sigma / tau2).clamp(1e-4, 1e3)
        }
        _ => 1.0,
    };
    let p = frame.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let opts = GlmOptions::default();
    let fit = fit_glm_ridge(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols, y },
        &backend,
        ls_ws,
        &opts,
        lambda,
    )
    .map_err(|e| ModelError::Numerical { message: e.to_string() })?;
    if !fit.converged {
        return Err(ModelError::Numerical {
            message: "hierarchical GLM logit did not converge".into(),
        });
    }
    // Encode as 2-category Discrete with baseline-category logits (cat0 = 0, cat1 = β).
    let mut logit_coeffs = vec![0.0; 2 * ncols];
    logit_coeffs[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    let n1 = y.iter().filter(|&&yi| yi == 1.0).count() as f64;
    let p1 = n1 / n.max(1) as f64;
    Ok(MechanismSlot::Discrete {
        support: Arc::from([0.0, 1.0]),
        probs: Arc::from([1.0 - p1, p1]),
        logit_coeffs: Some(Arc::from(logit_coeffs)),
    })
}

/// Minnesota-style single-equation Bayesian regression: a zero-mean normal prior
/// on each parent's coefficient, with the intercept unpenalized.
///
/// The prior is placed on the coefficient *per standard deviation of the parent*
/// and scales with the error variance (`b'ⱼ ~ N(0, φ σ²)`, the conjugate
/// normal–inverse-gamma form), so the posterior mode is a ridge fit on
/// scale-standardized columns with pseudo-row weight `1/√φ`. That makes the fit
/// equivariant to a change of units of the outcome and of every parent, which the
/// unstandardized pseudo-rows were not. The Minnesota lag decay `φ/ℓ²` is not
/// applied: the gather plan carries no lag metadata, a parent's position in the
/// list is not a lag, and letting it set the shrinkage made the fit depend on the
/// order the parents happened to be listed in.
fn fit_bvar_minnesota(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    let ncols = 1 + p;
    let phi: f64 = 0.2; // overall tightness
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let (_, scales) = column_moments(&refs, n);
    let extra = p;
    let rows = n + extra;
    let mut x2 = vec![0.0; rows * ncols];
    let mut y2 = vec![0.0; rows];
    x2[..n].fill(1.0);
    for (j, col) in refs.iter().enumerate() {
        for r in 0..n {
            x2[(1 + j) * rows + r] = col[r] / scales[j];
        }
        x2[(1 + j) * rows + n + j] = (1.0 / phi).sqrt();
    }
    y2[..n].copy_from_slice(y);
    let fit = backend.least_squares(&x2, rows, ncols, &y2, ls_ws).map_err(ModelError::from)?;
    let intercept = fit.coefficients[0];
    let coeffs: Vec<f64> =
        fit.coefficients[1..].iter().zip(scales.iter()).map(|(b, s)| b / s).collect();
    // σ from the data rows: the augmented RSS also holds Σ b'²/φ from the prior rows.
    let rss = data_rss(frame, y, intercept, &coeffs, None);
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::Bvar { intercept, coeffs: Arc::from(coeffs), sigma })
}

/// Scalar LGSSM on parent-adjusted residuals via EM (Kalman 1960 filter / Rauch–Tung–Striebel
/// 1965 smoother).
fn fit_lgssm_kalman_em(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let lg = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let (intercept, coeffs) = match lg {
        MechanismSlot::LinearGaussian { intercept, coeffs, .. } => (intercept, coeffs),
        _ => {
            return Err(ModelError::Unsupported {
                message: "lgssm fit requires linear base".into(),
            });
        }
    };
    let mut resid = vec![0.0; y.len()];
    for r in 0..y.len() {
        let mut pred = intercept;
        for (p, col) in frame.parents.iter().enumerate() {
            pred += coeffs[p] * col[r];
        }
        resid[r] = y[r] - pred;
    }
    let (a, process_std, obs_std, initial_mean) = lgssm_em(&resid, 25);
    Ok(MechanismSlot::ConditionalLinearGaussianStateSpace {
        intercept,
        coeffs,
        a,
        process_std,
        obs_std,
        initial_mean,
    })
}

/// EM for scalar LGSSM: `x_t` = a x_{t-1} + q ε, `y_t` = `x_t` + r η.
fn lgssm_em(y: &[f64], max_iters: usize) -> (f64, f64, f64, f64) {
    // Fit in relative units so variance floors and initialization do not change
    // the model when the outcome's measurement units change.
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let normalized: Vec<f64> = y.iter().map(|v| v / scale).collect();
    let (a, process_std, obs_std, initial_mean) = lgssm_em_normalized(&normalized, max_iters);
    (a, process_std * scale, obs_std * scale, initial_mean * scale)
}

fn lgssm_em_normalized(y: &[f64], max_iters: usize) -> (f64, f64, f64, f64) {
    let n = y.len();
    if n < 3 {
        let (a, q) = fit_ar1(y);
        return (a, q, q.max(1e-8), y.first().copied().unwrap_or(0.0));
    }
    let mut a = 0.8;
    let mut q = 1.0; // process variance
    let mut r = 1.0; // obs variance
    let mut x0 = y[0];
    for _ in 0..max_iters {
        let (x_f, p_f, x_pred, p_pred) = crate::lgssm::kalman_filter(y, a, q, r, x0, q);
        let (x_s, p_s, p_lag) = crate::lgssm::rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred);
        // M-step
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 1..n {
            num += p_lag[t] + x_s[t] * x_s[t - 1];
            den += p_s[t - 1] + x_s[t - 1] * x_s[t - 1];
        }
        a = if den > 1e-12 { (num / den).clamp(-0.999, 0.999) } else { a };
        // Initial state variance is q in the emitted model as well. Its
        // expected squared residual contributes one of the n process terms.
        x0 = x_s[0];
        let mut q_acc = p_s[0];
        for t in 1..n {
            q_acc += p_s[t] + x_s[t] * x_s[t] + a * a * (p_s[t - 1] + x_s[t - 1] * x_s[t - 1])
                - 2.0 * a * (p_lag[t] + x_s[t] * x_s[t - 1]);
        }
        q = (q_acc / n as f64).max(1e-8);
        let mut r_acc = 0.0;
        for t in 0..n {
            r_acc += p_s[t] + (y[t] - x_s[t]).powi(2);
        }
        r = (r_acc / n as f64).max(1e-8);
    }
    (a, q.sqrt(), r.sqrt(), x0)
}

/// Raw `UnitId` labels per row, when a finite `UnitId` column of length `n` exists.
fn unit_id_groups(data: &TabularData, n: usize) -> Option<Vec<u32>> {
    let schema = data.schema();
    for var in schema.variables() {
        if !var.role_hints.contains(RoleHint::UnitId) {
            continue;
        }
        let Ok(col) = data.float64_cow(var.id) else {
            continue;
        };
        if col.len() != n {
            continue;
        }
        let mut groups = Vec::with_capacity(n);
        let mut ok = true;
        for &v in col.iter() {
            if !v.is_finite() {
                ok = false;
                break;
            }
            groups.push(v.round() as u32);
        }
        if ok {
            return Some(groups);
        }
    }
    None
}

/// Subtract each unit's mean from its rows.
fn demean_by_group(values: &[f64], groups: &[u32]) -> Vec<f64> {
    let mut sums = std::collections::HashMap::<u32, (f64, usize)>::new();
    for (&g, &v) in groups.iter().zip(values.iter()) {
        let e = sums.entry(g).or_insert((0.0, 0));
        e.0 += v;
        e.1 += 1;
    }
    groups
        .iter()
        .zip(values.iter())
        .map(|(&g, &v)| {
            let (s, c) = sums[&g];
            v - s / c.max(1) as f64
        })
        .collect()
}

fn fit_ar1(series: &[f64]) -> (f64, f64) {
    let n = series.len();
    if n < 3 {
        return (0.0, series.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8));
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        num += series[t] * series[t - 1];
        den += series[t - 1] * series[t - 1];
    }
    let a = if den > 1e-12 { (num / den).clamp(-0.999, 0.999) } else { 0.0 };
    let mut rss = 0.0;
    for t in 1..n {
        let e = series[t] - a * series[t - 1];
        rss += e * e;
    }
    let process_std = (rss / (n - 1) as f64).sqrt().max(1e-8);
    (a, process_std)
}

fn fit_linear_gaussian(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    ridge: f64,
) -> Result<MechanismSlot, ModelError> {
    fit_linear_gaussian_weighted(frame, y, backend, ls_ws, ridge, None)
}

#[allow(clippy::too_many_arguments)]
fn fit_linear_gaussian_weighted(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    ridge: f64,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let weighted_y;
    let y_fit = if let Some(weights) = weights {
        weighted_y = y.iter().zip(weights).map(|(v, w)| v * w.sqrt()).collect::<Vec<_>>();
        for c in 0..ncols {
            for r in 0..n {
                x[c * n + r] *= weights[r].sqrt();
            }
        }
        weighted_y.as_slice()
    } else {
        y
    };
    let fit = if ridge > 0.0 {
        // Augment with ridge rows for coefficients (not intercept).
        let extra = p;
        let mut x2 = vec![0.0; (n + extra) * ncols];
        let mut y2 = vec![0.0; n + extra];
        for c in 0..ncols {
            for r in 0..n {
                x2[c * (n + extra) + r] = x[c * n + r];
            }
        }
        y2[..n].copy_from_slice(y_fit);
        let sqrt_r = ridge.sqrt();
        for j in 0..p {
            x2[(1 + j) * (n + extra) + (n + j)] = sqrt_r;
        }
        backend.least_squares(&x2, n + extra, ncols, &y2, ls_ws).map_err(ModelError::from)?
    } else {
        backend.least_squares(&x, n, ncols, y_fit, ls_ws).map_err(ModelError::from)?
    };
    let intercept = fit.coefficients[0];
    let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
    // σ from the data rows only: a ridge fit's `fit.rss` also sums the penalty
    // pseudo-rows (`λ Σ β²`), which would inflate σ with the prior.
    let rss = if ridge > 0.0 { data_rss(frame, y, intercept, &coeffs, weights) } else { fit.rss };
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
}

// ---------------------------------------------------------------- basis families

/// Rows required per fitted column before a basis family is admissible.
///
/// A two-way interaction or spline expansion adds columns quadratically in the
/// parent count. Below this ratio the expansion is fitting noise, and the
/// cross-validated score would say so only after paying for an unstable solve;
/// refusing is cheaper and the refusal is recorded in
/// [`MechanismAssignment::failed_families`].
pub const BASIS_MIN_ROWS_PER_COLUMN: usize = 10;

/// Relative rank-deficiency floor for the spline solve, on column-scaled design
/// columns. Not a smoothing penalty: smoothing comes from the fixed three-knot
/// cubic *regression* spline, and this only keeps a near-collinear truncated
/// power basis from producing an exploding solve.
const SPLINE_CONDITIONING_RIDGE: f64 = 1e-6;

/// Build the expansion a basis family fits on.
fn build_basis(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    n: usize,
) -> Result<ParentBasis, ModelError> {
    let p = frame.n_parents();
    if p < 2 {
        return Err(ModelError::Unsupported {
            message: format!(
                "{} needs at least two parents to form a cross-parent product (got {p}); with one \
                 parent the contrast is the same for every unit under any additive-noise \
                 mechanism",
                family.id()
            ),
        });
    }
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let (centers, scales) = column_moments(&refs, n);
    let centers: Arc<[f64]> = Arc::from(centers);
    let scales: Arc<[f64]> = Arc::from(scales);
    let basis = match family {
        MechanismFamily::LinearInteractions | MechanismFamily::DiscreteInteractions => {
            ParentBasis::interactions(p, centers, scales)?
        }
        MechanismFamily::LinearSpline | MechanismFamily::DiscreteSpline => {
            let knots = spline_knots(&refs, n, &centers, &scales);
            if knots.iter().all(|k| k.is_empty()) {
                return Err(ModelError::Unsupported {
                    message: format!(
                        "{} found no parent with enough distinct values for a spline; the \
                         expansion would duplicate the interaction family",
                        family.id()
                    ),
                });
            }
            ParentBasis::spline_interactions(p, centers, scales, Arc::from(knots))?
        }
        other => {
            return Err(ModelError::Unsupported {
                message: format!("{} is not a basis family", other.id()),
            });
        }
    };
    if !basis.has_cross_parent_product() {
        return Err(ModelError::Unsupported {
            message: format!("{} produced no cross-parent product", family.id()),
        });
    }
    let ncols = 1 + basis.n_terms();
    if n < ncols.saturating_mul(BASIS_MIN_ROWS_PER_COLUMN) {
        return Err(ModelError::Unsupported {
            message: format!(
                "{} expands {p} parents to {ncols} columns and needs at least \
                 {BASIS_MIN_ROWS_PER_COLUMN} rows per column (got {n})",
                family.id()
            ),
        });
    }
    Ok(basis)
}

/// Column-major design `[1 | φ(pa)]` for a basis.
fn basis_design_matrix(
    basis: &ParentBasis,
    parent_cols: &[std::borrow::Cow<'_, [f64]>],
    n: usize,
) -> Result<Vec<f64>, ModelError> {
    let t = basis.n_terms();
    let mut x = vec![0.0; n * (1 + t)];
    x[..n].fill(1.0);
    let mut row = vec![0.0; basis.n_parents()];
    let mut expanded = vec![0.0; t];
    for r in 0..n {
        for (p, slot) in row.iter_mut().enumerate() {
            *slot = basis.standardize(p, parent_cols[p][r]);
        }
        basis.expand_standardized(&row, &mut expanded)?;
        for (c, value) in expanded.iter().enumerate() {
            x[(1 + c) * n + r] = *value;
        }
    }
    Ok(x)
}

/// Weighted ridge least squares on a column-major design, solved on
/// RMS-normalized columns and un-normalized afterwards.
///
/// Column normalization is an exact change of variables — the fitted surface is
/// identical — and it is what makes a truncated power basis solvable at all: the
/// raw columns of `z³` and `z·z³` differ in scale by orders of magnitude.
/// Returns the coefficients on the original column scale and the weighted RSS.
#[allow(clippy::too_many_arguments)]
fn solve_scaled_ridge(
    x: &[f64],
    n: usize,
    ncols: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    ridge: f64,
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<(Vec<f64>, f64), ModelError> {
    let mut scale = vec![1.0_f64; ncols];
    for (c, s) in scale.iter_mut().enumerate().skip(1) {
        let rms = (x[c * n..c * n + n].iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64).sqrt();
        *s = if rms.is_finite() && rms > 1e-12 { rms } else { 1.0 };
    }
    let extra = if ridge > 0.0 { ncols - 1 } else { 0 };
    let rows = n + extra;
    let mut design = vec![0.0; rows * ncols];
    let mut target = vec![0.0; rows];
    for c in 0..ncols {
        for r in 0..n {
            let w = weights.map_or(1.0, |w| w[r].sqrt());
            design[c * rows + r] = x[c * n + r] / scale[c] * w;
        }
    }
    for r in 0..n {
        target[r] = y[r] * weights.map_or(1.0, |w| w[r].sqrt());
    }
    if ridge > 0.0 {
        let sqrt_r = ridge.sqrt();
        for j in 1..ncols {
            design[j * rows + (n + j - 1)] = sqrt_r;
        }
    }
    let fit =
        backend.least_squares(&design, rows, ncols, &target, ls_ws).map_err(ModelError::from)?;
    let coeffs: Vec<f64> = fit.coefficients.iter().zip(&scale).map(|(b, s)| b / s).collect();
    if coeffs.iter().any(|b| !b.is_finite()) {
        return Err(ModelError::Numerical {
            message: "basis mechanism solve produced a non-finite coefficient".into(),
        });
    }
    // Residual sum of squares over the data rows only: the solver's own RSS also
    // sums the penalty pseudo-rows.
    let mut rss = 0.0;
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += coeffs[c] * x[c * n + r];
        }
        rss += weights.map_or(1.0, |w| w[r]) * (y[r] - pred).powi(2);
    }
    Ok((coeffs, rss))
}

/// Per-row ridge weight for a basis solve: the conditioning floor for a spline
/// expansion, and nothing at all for a pure interaction expansion (which is
/// ordinary least squares on the same columns `LinearGaussian` fits, plus the
/// products).
fn basis_ridge_rel(basis: &ParentBasis) -> f64 {
    if basis.knots().iter().any(|k| !k.is_empty()) { SPLINE_CONDITIONING_RIDGE } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
fn fit_linear_basis(
    basis: &ParentBasis,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let x = basis_design_matrix(basis, &frame.parents, n)?;
    let ncols = 1 + basis.n_terms();
    let ridge = basis_ridge_rel(basis) * n as f64;
    let (coeffs, rss) = solve_scaled_ridge(&x, n, ncols, y, weights, ridge, backend, ls_ws)?;
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::LinearBasis {
        intercept: coeffs[0],
        basis: basis.clone(),
        coeffs: Arc::from(coeffs[1..].to_vec()),
        sigma,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_discrete_basis(
    basis: &ParentBasis,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let (support, probs) = discrete_support(y, weights)?;
    let k = support.len();
    if k < 2 {
        return Err(ModelError::Unsupported {
            message: "discrete basis families need at least two categories".into(),
        });
    }
    let x = basis_design_matrix(basis, &frame.parents, n)?;
    let ncols = 1 + basis.n_terms();
    let y_cat = discrete_categories(y, &support)?;
    let fit = antecedent_stats::fit_multinomial_logit_weighted(
        MultinomialDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols,
            y_category: &y_cat,
            n_categories: k,
        },
        weights,
        &backend,
        ls_ws,
        &GlmOptions::default(),
    )?;
    if !fit.converged {
        return Err(ModelError::NotConverged {
            message: format!(
                "multinomial logit on a parent basis did not converge (iters={}, deviance={})",
                fit.iterations, fit.deviance
            ),
        });
    }
    Ok(MechanismSlot::DiscreteBasis {
        support: Arc::from(support),
        probs: Arc::from(probs),
        basis: basis.clone(),
        logit_coeffs: Arc::from(fit.coefficients),
    })
}

/// Weighted support and marginal probabilities of a discrete column.
fn discrete_support(
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
    let mut pairs: Vec<(i64, f64, f64)> = Vec::new();
    for (r, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            continue;
        }
        let key = (yi * 1e6).round() as i64;
        if let Some(e) = pairs.iter_mut().find(|(k, _, _)| *k == key) {
            e.2 += weights.map_or(1.0, |w| w[r]);
        } else {
            pairs.push((key, yi, weights.map_or(1.0, |w| w[r])));
        }
    }
    if pairs.is_empty() {
        return Err(ModelError::Shape { message: "no finite values for discrete fit".into() });
    }
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let total = pairs.iter().map(|(_, _, c)| *c).sum::<f64>();
    Ok((
        pairs.iter().map(|(_, v, _)| *v).collect(),
        pairs.iter().map(|(_, _, c)| *c / total).collect(),
    ))
}

/// Category index per row against a fitted support.
fn discrete_categories(y: &[f64], support: &[f64]) -> Result<Vec<u32>, ModelError> {
    let mut out = vec![0u32; y.len()];
    for (r, &yi) in y.iter().enumerate() {
        let Some(idx) = support.iter().position(|&s| (s - yi).abs() < 1e-12) else {
            return Err(ModelError::Shape {
                message: "discrete outcome not in fitted support".into(),
            });
        };
        out[r] = u32::try_from(idx)
            .map_err(|_| ModelError::Shape { message: "too many discrete categories".into() })?;
    }
    Ok(out)
}

/// Row cap for the [`MechanismFamily::GaussianProcess`] grid search.
///
/// The fit runs a 25-cell `(ℓ, σ)` grid where every cell builds an O(n²) dense
/// Gram matrix and factors it with an O(n³) Cholesky, and family scoring refits
/// it on every training fold. At n = 1 000 that is an 8 MB Gram and about six
/// full grid searches of ≈ 25 · n³/3 flops each — seconds on current hardware and
/// the last point where the exact GP is a reasonable candidate; n = 10 000
/// would already need 800 MB and minutes per node. Above the cap
/// `fit_family(GaussianProcess)` refuses with [`ModelError::Unsupported`]
/// instead of silently hanging; because model selection records failed
/// families and picks among the rest, the refusal is surfaced in
/// [`MechanismAssignment::failed_families`] and another family is selected.
#[cfg(feature = "gaussian-process")]
pub const GP_FAMILY_MAX_ROWS: usize = 1_000;

/// Hyperparameter grid of the GP fit, on the data's own scale.
///
/// Length scales are multiples of the parents' RMS spread and noise levels are
/// multiples of the outcome's SD, so the grid — and therefore the fit — is
/// equivariant to a change of units of the outcome and of every parent. The
/// signal variance is the outcome's variance and the prior mean its sample mean
/// (the fitted surface reverts to that mean off-support, not to zero).
#[cfg(feature = "gaussian-process")]
#[derive(Clone, Debug)]
pub(crate) struct GpGrid {
    pub mean: f64,
    pub variance: f64,
    pub length_scales: [f64; 5],
    pub noise_stds: [f64; 5],
}

#[cfg(feature = "gaussian-process")]
pub(crate) fn gp_grid(parents: &[&[f64]], y: &[f64]) -> Result<GpGrid, ModelError> {
    let n = y.len().max(1) as f64;
    let mean = y.iter().sum::<f64>() / n;
    let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if !(variance.is_finite() && variance > 0.0) {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess needs an outcome with positive variance".into(),
        });
    }
    let x_var = parents
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / n;
            c.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
        })
        .sum::<f64>()
        / parents.len().max(1) as f64;
    if !(x_var.is_finite() && x_var > 0.0) {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess needs at least one parent that varies".into(),
        });
    }
    let (s_x, s_y) = (x_var.sqrt(), variance.sqrt());
    Ok(GpGrid {
        mean,
        variance,
        length_scales: [0.25, 0.5, 1.0, 2.0, 4.0].map(|m| m * s_x),
        noise_stds: [0.05, 0.1, 0.2, 0.5, 1.0].map(|m| m * s_y),
    })
}

#[cfg(feature = "gaussian-process")]
/// Grid-search RBF GP hyperparameters by exact Cholesky NLML.
///
/// For each `(ℓ, σ)` cell of [`gp_grid`], form `K = k_RBF + σ²I`, factor once with
/// [`cholesky_spd`], reuse that factor for both `log|K|` ([`chol_log_det`]) and
/// `α = K⁻¹(y − ȳ)` ([`chol_solve`]). Do not proxy the determinant by `Σ log Kᵢᵢ`
/// (MM-015).
fn fit_gaussian_process(frame: &FitFrame<'_>, y: &[f64]) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    if p == 0 {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess requires at least one parent".into(),
        });
    }
    if n > GP_FAMILY_MAX_ROWS {
        return Err(ModelError::Unsupported {
            message: format!(
                "GaussianProcess is limited to {GP_FAMILY_MAX_ROWS} rows (got {n}): the exact \
                 Cholesky grid search is O(n³) per cell with an O(n²) Gram matrix; use another \
                 mechanism family (or subsample) for larger data"
            ),
        });
    }
    let mut x_train = vec![0.0; n * p];
    for r in 0..n {
        for c in 0..p {
            x_train[r * p + c] = frame.parents[c][r];
        }
    }
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let grid = gp_grid(&refs, y)?;
    let yc: Vec<f64> = y.iter().map(|v| v - grid.mean).collect();
    let variance = grid.variance;
    let mut best = None::<(f64, f64, f64, Vec<f64>)>; // (nlml, ℓ, σ, α)
    for &length_scale in &grid.length_scales {
        for &noise_std in &grid.noise_stds {
            let mut k = vec![0.0; n * n];
            let inv_l2 = 1.0 / (length_scale * length_scale);
            for i in 0..n {
                for j in i..n {
                    let mut d2 = 0.0;
                    for c in 0..p {
                        let d = x_train[i * p + c] - x_train[j * p + c];
                        d2 += d * d;
                    }
                    let kij = variance * (-0.5 * d2 * inv_l2).exp();
                    k[i * n + j] = kij;
                    k[j * n + i] = kij;
                }
                k[i * n + i] += noise_std * noise_std;
            }
            let Some(chol) = cholesky_spd(&k, n) else {
                continue;
            };
            let Some(alpha) = chol_solve(&chol, n, &yc) else {
                continue;
            };
            let mut y_alpha = 0.0;
            for i in 0..n {
                y_alpha += yc[i] * alpha[i];
            }
            let nlml = 0.5 * y_alpha
                + 0.5 * chol_log_det(&chol, n)
                + 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
            match &best {
                Some((best_nlml, ..)) if nlml >= *best_nlml => {}
                _ => best = Some((nlml, length_scale, noise_std, alpha)),
            }
        }
    }
    let (_nlml, length_scale, noise_std, alpha) = best.ok_or_else(|| ModelError::Numerical {
        message: "GP hyperparameter search failed".into(),
    })?;
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        mean: grid.mean,
        x_train: Arc::from(x_train),
        n_train: n,
        n_parents: p,
        alpha: Arc::from(alpha),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use antecedent_graph::{Dag, DenseNodeId};

    fn toy_data() -> (TabularData, Dag) {
        let n = 40usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            xv[i] = i as f64 * 0.1;
            yv[i] = 1.0 + 2.0 * xv[i];
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (TabularData::new(storage), g)
    }

    /// The early-exiting distinct scan must agree with the historical
    /// sort+dedup implementation on ties, NaN/±inf, negatives, and the
    /// quantization boundary (values closer than 1e-6 collapse to one level).
    #[test]
    fn fitted_lgssm_preserves_intercept_and_parent_effect() {
        use crate::batch::{MechanismWorkspace, ParentBatch};
        let (data, graph) = toy_data();
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let (store, _) = MechanismRegistry::with_bayesian_families()
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussianStateSpace),
            )
            .unwrap();
        let slot = store.get(DenseNodeId::from_raw(1));
        let MechanismSlot::ConditionalLinearGaussianStateSpace { intercept, coeffs, .. } = slot
        else {
            panic!("conditional LGSSM expected");
        };
        assert!((intercept - 1.0).abs() < 1e-10);
        assert!((coeffs[0] - 2.0).abs() < 1e-10);
        let parents = [0.0, 1.0, 2.0];
        let mut noise = [0.0; 3];
        crate::mechanism::sample_noise_column(
            slot,
            3,
            &mut antecedent_core::CausalRng::from_seed(12),
            &mut noise,
        )
        .unwrap();
        let mut output = [0.0; 3];
        crate::mechanism::evaluate_column(
            slot,
            ParentBatch { n_rows: 3, n_parents: 1, values: &parents },
            &noise,
            &mut output,
            &mut MechanismWorkspace::default(),
        )
        .unwrap();
        for (actual, expected) in output.iter().zip([1.0, 3.0, 5.0]) {
            assert!((actual - expected).abs() < 1e-8, "actual={actual} expected={expected}");
        }
        let MechanismSlot::ConditionalLinearGaussianStateSpace { intercept, coeffs, .. } =
            store.get(DenseNodeId::from_raw(0))
        else {
            panic!("conditional root LGSSM expected");
        };
        assert!(coeffs.is_empty());
        assert!((intercept - 1.95).abs() < 1e-10, "root mean must survive residualization");
    }

    /// When every real family fails to fit (here: collinear parents make the
    /// design matrix rank-deficient, so `LinearGaussian` errors), the registry
    /// must refuse the node rather than silently select `Constant` at its
    /// `-∞` inadmissibility score. A varying `y` is not degenerate, so a
    /// `Constant` fit is not a legitimate answer here — it must be excluded
    /// from selection along with the family that errored outright.
    #[test]
    fn assign_and_fit_refuses_when_only_admissible_candidate_is_negative_infinity() {
        let n = 20usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["p1", "p2", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut p1 = vec![0.0; n];
        let mut p2 = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            p1[i] = i as f64 * 0.1;
            p2[i] = 2.0 * p1[i]; // exactly collinear with p1 -> rank-deficient design
            yv[i] = if i % 2 == 0 { 10.0 } else { -10.0 }; // clearly not degenerate
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(p1), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(p2), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let result = reg.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore);
        assert!(
            result.is_err(),
            "must refuse to fit y when LinearGaussian is rank-deficient and Constant is \
             inadmissible, not silently select Constant at -infinity: {result:?}"
        );
    }

    #[test]
    fn lgssm_em_is_equivariant_to_measurement_units() {
        let y = [0.2, -0.4, 0.6, 0.3, -0.7, 0.5, 0.1, -0.2];
        let fitted = lgssm_em(&y, 25);
        for scale in [1e-100, 1e100] {
            let scaled: Vec<f64> = y.iter().map(|v| v * scale).collect();
            let result = lgssm_em(&scaled, 25);
            assert!((result.0 - fitted.0).abs() < 1e-10);
            assert!((result.1 / scale - fitted.1).abs() < 1e-10);
            assert!((result.2 / scale - fitted.2).abs() < 1e-10);
            assert!((result.3 / scale - fitted.3).abs() < 1e-10);
        }
    }

    #[test]
    fn lgssm_em_increases_the_emitted_models_likelihood() {
        let y = [0.2, -0.4, 0.6, 0.3, -0.7, 0.5, 0.1, -0.2];
        let mut previous = f64::NEG_INFINITY;
        for iterations in 0..25 {
            let (a, process_std, obs_std, initial_mean) = lgssm_em(&y, iterations);
            let q = process_std.powi(2);
            let r = obs_std.powi(2);
            let (_, _, means, variances) =
                crate::lgssm::kalman_filter(&y, a, q, r, initial_mean, q);
            let loglik: f64 = y
                .iter()
                .zip(means)
                .zip(variances)
                .map(|((&value, mean), variance)| {
                    let v = variance + r;
                    -0.5 * (v.ln() + (value - mean).powi(2) / v)
                })
                .sum();
            assert!(
                loglik >= previous - 1e-10,
                "iteration {iterations}: loglik {loglik} < {previous}"
            );
            previous = loglik;
        }
    }

    #[test]
    fn is_low_cardinality_matches_sort_dedup_reference() {
        fn reference(y: &[f64], max_levels: usize) -> bool {
            let mut vals: Vec<i64> =
                y.iter().filter(|v| v.is_finite()).map(|v| (v * 1e6).round() as i64).collect();
            vals.sort_unstable();
            vals.dedup();
            !vals.is_empty() && vals.len() <= max_levels
        }
        let cases: [&[f64]; 8] = [
            &[],
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            &[1.0, 1.0, 1.0, 1.0],
            &[-3.0, -3.0, 2.0, 2.0, f64::NAN, -3.0],
            &[0.0, -0.0, 1e-7, 2e-7, 5e-7], // all quantize to the same level
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[-1.5, 1.5, -1.5, 1.5, 0.25, f64::NAN, -8.75],
        ];
        for y in cases {
            for max_levels in [0usize, 1, 2, 8] {
                assert_eq!(
                    is_low_cardinality(y, max_levels),
                    reference(y, max_levels),
                    "mismatch for y={y:?} max_levels={max_levels}"
                );
            }
        }
    }

    /// Above [`GP_FAMILY_MAX_ROWS`] the GP family must refuse (recorded in
    /// `failed_families`) instead of running the multi-minute O(n³) grid, and
    /// selection must still pick another family.
    #[cfg(feature = "gaussian-process")]
    #[test]
    fn gp_family_refuses_above_row_cap_and_selection_falls_back() {
        let n = GP_FAMILY_MAX_ROWS + 1;
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let xv: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x + (x * 0.5).cos() * 0.01).collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (_, assigns) = MechanismRegistry::with_bayesian_families()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let child = assigns.iter().find(|a| a.node == DenseNodeId::from_raw(1)).unwrap();
        let gp_failure = child
            .failed_families
            .iter()
            .find(|(f, _)| *f == MechanismFamily::GaussianProcess)
            .expect("GP refusal must be recorded, not silently absent");
        assert!(gp_failure.1.contains("limited to"), "message: {}", gp_failure.1);
        assert_ne!(child.selected, MechanismFamily::GaussianProcess);
        assert!(!child.candidates.iter().any(|c| c.family == MechanismFamily::GaussianProcess));
    }

    #[test]
    fn auto_assign_linear_chain() {
        let (data, g) = toy_data();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let (store, assigns) =
            reg.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore).unwrap();
        assert_eq!(assigns.len(), 2);
        assert!(matches!(
            store.get(DenseNodeId::from_raw(1)),
            MechanismSlot::LinearGaussian { .. }
        ));
    }

    #[test]
    fn bayesian_families_fit_hierarchical_and_bvar() {
        let (data, g) = toy_data();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::with_bayesian_families();
        let (store, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::HierarchicalLinear),
            )
            .unwrap();
        assert!(matches!(
            store.get(DenseNodeId::from_raw(1)),
            MechanismSlot::HierarchicalLinear { .. }
        ));
        let (store2, _) = reg
            .assign_and_fit(&compiled, &data, SelectionPolicy::RequireFamily(MechanismFamily::Bvar))
            .unwrap();
        assert!(matches!(store2.get(DenseNodeId::from_raw(1)), MechanismSlot::Bvar { .. }));
        let (store3, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussianStateSpace),
            )
            .unwrap();
        assert!(matches!(
            store3.get(DenseNodeId::from_raw(1)),
            MechanismSlot::ConditionalLinearGaussianStateSpace { .. }
        ));
    }

    #[test]
    fn discrete_conditional_multinomial_logit_mle() {
        let n = 120usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            xv[i] = t;
            // Soft association: mostly Y=t, occasional flips (avoids complete separation).
            yv[i] = if i % 8 == 0 { 1.0 - t } else { t };
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let (store, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::Discrete),
            )
            .unwrap();
        let MechanismSlot::Discrete { support, logit_coeffs, .. } =
            store.get(DenseNodeId::from_raw(1))
        else {
            panic!("expected discrete mechanism");
        };
        let logits = logit_coeffs.as_ref().expect("parent-conditional logits");
        assert_eq!(support.len(), 2);
        assert_eq!(logits.len(), 2 * 2); // K * (1 + p)
        // Reference category pinned to zero.
        assert!(logits[0].abs() < 1e-12 && logits[1].abs() < 1e-12);
        // Positive slope for the higher class vs reference.
        assert!(logits[3] > 0.5, "slope={}", logits[3]);
    }

    #[cfg(feature = "gaussian-process")]
    fn gp_oracle_table(x_scale: f64, y_scale: f64, y_shift: f64) -> (TabularData, Vec<f64>) {
        let n = 24usize;
        let x: Vec<f64> =
            (0..n).map(|i| x_scale * (-2.4 + 4.8 * i as f64 / (n - 1) as f64)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = -2.4 + 4.8 * i as f64 / (n - 1) as f64;
                y_shift + y_scale * ((1.3 * t).sin() + 0.18 * (3.1 * t).cos())
            })
            .collect();
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        (data, y)
    }

    #[cfg(feature = "gaussian-process")]
    fn fit_gp(data: &TabularData, y: &[f64]) -> MechanismSlot {
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let gather = compiled.gather_for(DenseNodeId::from_raw(1)).unwrap();
        fit_family(
            MechanismFamily::GaussianProcess,
            gather,
            &compiled,
            data,
            y,
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap()
    }

    /// The GP hyperparameters are the exact-Cholesky NLML argmin over the data-scaled
    /// grid, checked against a frozen SciPy oracle (`conformance/gcm/gaussian_process`).
    /// The diagonal-proxy determinant the old code used (MM-015) picks a different cell.
    #[cfg(feature = "gaussian-process")]
    #[test]
    fn gaussian_process_matches_exact_logdet_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/gcm/gaussian_process/expected.json"
        ))
        .unwrap();
        let (data, y) = gp_oracle_table(1.0, 1.0, 0.0);
        let MechanismSlot::GaussianProcess { length_scale, noise_std, variance, mean, .. } =
            fit_gp(&data, &y)
        else {
            panic!("GP slot");
        };
        let reference = &fixture["reference"];
        for (got, key) in [
            (length_scale, "length_scale"),
            (noise_std, "noise_std"),
            (variance, "variance"),
            (mean, "mean"),
        ] {
            let want = reference[key].as_f64().unwrap();
            assert!(((got - want) / want).abs() < 1e-9, "{key}: {got} vs {want}");
        }
    }

    /// A GP fit must not depend on the units of the outcome or the parent: rescaling `y`
    /// by 100 (and shifting it by 1000) and `x` by 10 multiplies the chosen length scale by
    /// 10, the noise level, signal variance (by 100²) and prior mean accordingly, and
    /// leaves the dual weights `α = K⁻¹(y − ȳ)` scaled by exactly 1/100 — the fixed
    /// absolute grid the old fit used chose different cells at different units.
    #[cfg(feature = "gaussian-process")]
    #[test]
    fn gaussian_process_fit_is_equivariant_to_units() {
        let (data, y) = gp_oracle_table(1.0, 1.0, 0.0);
        let (data_s, y_s) = gp_oracle_table(10.0, 100.0, 1000.0);
        let (
            MechanismSlot::GaussianProcess {
                length_scale: l0,
                noise_std: n0,
                variance: v0,
                mean: m0,
                alpha: a0,
                ..
            },
            MechanismSlot::GaussianProcess {
                length_scale: l1,
                noise_std: n1,
                variance: v1,
                mean: m1,
                alpha: a1,
                ..
            },
        ) = (fit_gp(&data, &y), fit_gp(&data_s, &y_s))
        else {
            panic!("GP slots");
        };
        assert!((l1 / l0 - 10.0).abs() < 1e-9, "length scale ratio {}", l1 / l0);
        assert!((n1 / n0 - 100.0).abs() < 1e-9, "noise ratio {}", n1 / n0);
        assert!((v1 / v0 - 1.0e4).abs() < 1e-6, "variance ratio {}", v1 / v0);
        assert!((m1 - (1000.0 + 100.0 * m0)).abs() < 1e-9, "mean {m1}");
        for (b, a) in a1.iter().zip(a0.iter()) {
            assert!((b * 100.0 - a).abs() < 1e-8, "alpha {b} vs {a}");
        }
    }

    /// MM-A1: LGSSM/GP scoring must use a genuine fitted residual, on the same scale as the
    /// linear families — not `mean(y²)`, the raw second moment of the target. On data
    /// generated from a persistent near-random-walk LGSSM with a large offset, `mean(y²)`
    /// is dominated by `mean(y)² ≈ 25`, which was always far worse than the intercept-only
    /// `LinearGaussian` residual MSE (≈ the small variance around the slowly-drifting level)
    /// regardless of how well LGSSM actually predicts one step ahead. With the fix, LGSSM's
    /// score uses the Kalman one-step-ahead predictive residual, which is small for this
    /// series, so `BestScore` correctly prefers LGSSM.
    /// A root with real variance must be fit as a *distribution*, not a point mass.
    ///
    /// Every family is scored on conditional-mean fit. For a parentless node the
    /// `LinearGaussian` fit is `intercept = mean, coeffs = [], sigma = SD` — the correct
    /// marginal — and its residual MSE is exactly `Constant`'s MSE. `Constant` then won the
    /// tie, because `LinearGaussian` alone pays the `sigma` penalty. Every root therefore
    /// became deterministic, and swapping a root's mechanism between two populations was a
    /// no-op even when its variance had changed — which is what made
    /// `AttributionComponents::InputsAndMechanisms` unable to attribute input change at all.
    ///
    /// MSE cannot see this: a point mass predicts the mean perfectly. The scoring now gates
    /// `Constant` on the claim it is actually making — that the target is degenerate.
    #[test]
    fn root_with_variance_is_fit_as_a_distribution_not_a_constant() {
        fn fit_single_column(values: Vec<f64>) -> MechanismSlot {
            let n = values.len();
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let cols = vec![OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )];
            let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
            let data = TabularData::new(storage);
            let compiled = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
            let (store, _) = MechanismRegistry::standard()
                .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
                .unwrap();
            store.slots[0].clone()
        }

        // A root that genuinely varies: continuous ramp, sample SD ≈ 2.9.
        let spread: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
        let mean = spread.iter().sum::<f64>() / spread.len() as f64;
        let var =
            spread.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (spread.len() - 1) as f64;
        let sd = var.sqrt();
        match fit_single_column(spread) {
            MechanismSlot::LinearGaussian { intercept, ref coeffs, sigma } => {
                assert!(coeffs.is_empty(), "a root has no parents");
                assert!((intercept - mean).abs() < 1e-9, "intercept {intercept} != mean {mean}");
                assert!(
                    sigma > 0.5 * sd,
                    "root sigma {sigma} must carry the marginal spread (SD {sd}), not collapse"
                );
            }
            other => panic!("root with variance fit as {other:?}; expected a real marginal"),
        }

        // A genuinely degenerate column must still get a deterministic mechanism — the gate
        // keys on `Constant`'s claim being true, not on banning the family. A zero-variance
        // column is low-cardinality, so it is routed to the discrete family list and
        // `Discrete{support:[v], probs:[1.0]}` ties `Constant` at score 0 and wins on order.
        // That representation is an equally exact point mass, and it is what this path
        // selected before the gate existed too (mse = 0 scores identically either way), so
        // assert the invariant that matters rather than a particular family tag.
        match fit_single_column(vec![7.0; 100]) {
            MechanismSlot::Constant { value } => assert!((value - 7.0).abs() < 1e-12),
            MechanismSlot::Discrete { ref support, ref probs, logit_coeffs: None } => {
                assert_eq!(support.len(), 1, "degenerate column must have a single support point");
                assert!((support[0] - 7.0).abs() < 1e-12);
                assert!((probs[0] - 1.0).abs() < 1e-12);
            }
            other => panic!("constant column fit as {other:?}; expected a deterministic mechanism"),
        }
    }

    #[test]
    fn best_score_prefers_lgssm_over_linear_gaussian_on_lgssm_generated_data() {
        use antecedent_core::CausalRng;
        use antecedent_kernels::standard_normal;

        let n = 60usize;
        let a = 0.95_f64;
        let process_std = 0.05_f64;
        let obs_std = 0.05_f64;
        let initial_mean = 5.0_f64;
        let mut rng = CausalRng::from_seed(11);
        let mut yv = vec![0.0; n];
        let mut x = initial_mean;
        for i in 0..n {
            x = if i == 0 {
                initial_mean + process_std * standard_normal(&mut rng)
            } else {
                a * x + process_std * standard_normal(&mut rng)
            };
            yv[i] = x + obs_std * standard_normal(&mut rng);
        }

        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(yv), validity).unwrap(),
        )];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let compiled = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();

        let registry = MechanismRegistry::with_bayesian_families();
        let (_, assigns) =
            registry.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore).unwrap();
        assert_eq!(assigns.len(), 1);
        let assignment = &assigns[0];
        let lg_candidate = assignment
            .candidates
            .iter()
            .find(|c| c.family == MechanismFamily::LinearGaussian)
            .expect("LinearGaussian candidate present");
        let lgssm_candidate = assignment
            .candidates
            .iter()
            .find(|c| c.family == MechanismFamily::LinearGaussianStateSpace)
            .expect("LGSSM candidate present");
        assert!(
            lgssm_candidate.score > lg_candidate.score,
            "lgssm={} linear={}",
            lgssm_candidate.score,
            lg_candidate.score
        );
        assert_eq!(assignment.selected, MechanismFamily::LinearGaussianStateSpace);
    }

    // ------------------------------------------------------------ scoring, typing, fits

    fn single_node_assignment(registry: &MechanismRegistry, y: &[f64]) -> Vec<MechanismAssignment> {
        let data = TabularData::from_f64_columns([("y", y)]).unwrap();
        let compiled = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
        registry.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore).unwrap().1
    }

    /// Every family is scored by held-out log-likelihood less a count-based penalty, so
    /// rescaling the outcome shifts each family's score by exactly `−ln(scale)` (the
    /// change of variables of a density) and cannot change which family wins. The old
    /// `−mse − 0.01·|ln σ|` mixed a unit-squared term with a unit-free one, and picked
    /// a different family at scale 0.01 than at scale 1.
    #[test]
    fn family_scores_shift_by_minus_ln_scale_when_the_outcome_is_rescaled() {
        use antecedent_kernels::standard_normal;
        let mut rng = antecedent_core::CausalRng::from_seed(21);
        let mut y = vec![0.0; 120];
        let mut x = 0.0;
        for v in &mut y {
            x = 0.5 * x + standard_normal(&mut rng);
            *v = x;
        }
        let registry = MechanismRegistry::with_bayesian_families();
        let base = single_node_assignment(&registry, &y).remove(0);
        for scale in [0.01, 100.0] {
            let scaled_y: Vec<f64> = y.iter().map(|v| v * scale).collect();
            let scaled = single_node_assignment(&registry, &scaled_y).remove(0);
            assert_eq!(scaled.selected, base.selected, "scale {scale}");
            assert_eq!(scaled.candidates.len(), base.candidates.len());
            for c in base.candidates.iter() {
                let s = scaled
                    .candidates
                    .iter()
                    .find(|s| s.family == c.family)
                    .unwrap_or_else(|| panic!("{:?} missing at scale {scale}", c.family));
                assert!(
                    (s.score - (c.score - scale.ln())).abs() < 1e-6,
                    "{:?} at scale {scale}: {} vs {} − ln(scale)",
                    c.family,
                    s.score,
                    c.score
                );
            }
        }
    }

    /// The `LinearGaussian` score is *exactly* its mean held-out Gaussian log-likelihood on
    /// the shared folds minus `k ln(n) / 2n` with `k = p + 2` free parameters, recomputed
    /// here from first principles (training mean/SD per fold, closed-form density).
    #[test]
    fn linear_gaussian_score_is_held_out_loglik_less_the_bic_penalty() {
        let n = 25usize;
        let y: Vec<f64> =
            (0..n).map(|i| 3.0 + (0.7 * i as f64).sin() + 0.4 * (2.3 * i as f64).cos()).collect();
        let a = single_node_assignment(&MechanismRegistry::standard(), &y).remove(0);
        let candidate = a
            .candidates
            .iter()
            .find(|c| c.family == MechanismFamily::LinearGaussian)
            .expect("LinearGaussian scored");
        let fold = cv_fold_plan(n, None).unwrap();
        let (mut total, mut count) = (0.0, 0usize);
        for k in 0..MECHANISM_CV_FOLDS {
            let train: Vec<f64> = (0..n).filter(|&r| fold[r] != k).map(|r| y[r]).collect();
            let m = train.len() as f64;
            let mean = train.iter().sum::<f64>() / m;
            let rss = train.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            let sigma = (rss / (m - 1.0)).sqrt();
            for r in (0..n).filter(|&r| fold[r] == k) {
                let z = (y[r] - mean) / sigma;
                total += -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln() - 0.5 * z * z;
                count += 1;
            }
        }
        assert_eq!(count, n, "every row is held out exactly once");
        let penalty = 0.5 * 2.0 * (n as f64).ln() / n as f64;
        let expected = total / count as f64 - penalty;
        assert!(
            (candidate.score - expected).abs() < 1e-10,
            "score {} vs held-out log-likelihood {expected}",
            candidate.score
        );
    }

    fn typed_table(value_type: ValueType, y: &[f64]) -> (TabularData, VariableId) {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "y",
            value_type,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let data = TabularData::try_from_schema_f64(b.build().unwrap(), [("y", y)]).unwrap();
        (data, VariableId::from_raw(0))
    }

    /// The declared value type decides discreteness; a level count only decides for a
    /// column not declared discrete, and only when levels repeat. The old rule sent any
    /// column with ≤ 8 distinct values (including every column with n ≤ 8) to a
    /// multinomial whose counterfactual support is the observed values.
    #[test]
    fn declared_value_type_decides_discreteness_and_short_columns_stay_continuous() {
        let cycle =
            |levels: usize, n: usize| -> Vec<f64> { (0..n).map(|i| (i % levels) as f64).collect() };
        // Declared binary beats the level count: 12 levels, still categorical.
        let twelve = cycle(12, 120);
        let (data, var) = typed_table(ValueType::Binary, &twelve);
        assert_eq!(classify_node(&data, var, &twelve), MechanismTyping::DeclaredDiscrete);
        // Continuous with two repeated levels: the heuristic reads it as categorical.
        let binary = cycle(2, 40);
        let (data, var) = typed_table(ValueType::Continuous, &binary);
        assert_eq!(classify_node(&data, var, &binary), MechanismTyping::LowCardinality);
        let five = cycle(5, 40);
        let (data, var) = typed_table(ValueType::Continuous, &five);
        assert_eq!(classify_node(&data, var, &five), MechanismTyping::LowCardinality);
        // Too few rows per level: a short sample of a continuous variable.
        let short_binary = cycle(2, 4);
        let (data, var) = typed_table(ValueType::Continuous, &short_binary);
        assert_eq!(classify_node(&data, var, &short_binary), MechanismTyping::Continuous);
        let short_five = cycle(5, 12);
        let (data, var) = typed_table(ValueType::Continuous, &short_five);
        assert_eq!(classify_node(&data, var, &short_five), MechanismTyping::Continuous);
        // Many distinct values.
        let ramp: Vec<f64> = (0..40).map(f64::from).collect();
        let (data, var) = typed_table(ValueType::Continuous, &ramp);
        assert_eq!(classify_node(&data, var, &ramp), MechanismTyping::Continuous);

        // The rule that fired is recorded on the assignment.
        let (data, _) = typed_table(ValueType::Binary, &twelve);
        let compiled = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
        let (_, assignments) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        assert_eq!(assignments[0].typing, MechanismTyping::DeclaredDiscrete);
        assert!(matches!(assignments[0].fitted, MechanismSlot::Discrete { .. }));
    }

    /// σ floors are relative to the outcome's scale: a deterministic relation measured
    /// in units of 1e-6 keeps a σ at that scale, not an absolute 1e-8 that dwarfs it.
    #[test]
    fn residual_sigma_floor_is_relative_to_the_outcome_scale() {
        let n = 40usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|v| 1e-6 * (1.0 + 2.0 * v)).collect();
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let gather = compiled.gather_for(DenseNodeId::from_raw(1)).unwrap();
        let slot = fit_family(
            MechanismFamily::LinearGaussian,
            gather,
            &compiled,
            &data,
            &y,
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap();
        let MechanismSlot::LinearGaussian { sigma, .. } = slot else { panic!("linear slot") };
        assert!(sigma < 1e-9, "sigma {sigma} exceeds the data scale");
    }

    /// `HierarchicalLinear` with a `UnitId` column is a within-unit (fixed-effects)
    /// regression: parents are demeaned by unit as well as the outcome. Demeaning only
    /// `y` regresses it on a parent that still carries its between-unit variance, which
    /// attenuates the slope to `β·Var_within/Var_total`. σ is measured on the reported
    /// population-intercept mean function, so abduction and log-density see the
    /// between-unit spread instead of a σ that excludes it.
    #[test]
    fn hierarchical_linear_uses_the_within_estimator_and_sigma_keeps_between_unit_spread() {
        let (groups, per) = (8usize, 10usize);
        let n = groups * per;
        let (mut unit, mut x, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for g in 0..groups {
            for j in 0..per {
                let i = g * per + j;
                unit[i] = g as f64;
                x[i] = 1.5 * g as f64 + (0.9 * i as f64 + g as f64).sin();
                y[i] = 2.0 * x[i] + 0.3 * g as f64 + 0.05 * (1.7 * i as f64).cos();
            }
        }
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("unit", RoleHint::UnitId),
            ("x", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let data = TabularData::try_from_schema_f64(
            b.build().unwrap(),
            [("unit", unit.as_slice()), ("x", x.as_slice()), ("y", y.as_slice())],
        )
        .unwrap();
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let gather = compiled.gather_for(DenseNodeId::from_raw(2)).unwrap();
        let slot = fit_family(
            MechanismFamily::HierarchicalLinear,
            gather,
            &compiled,
            &data,
            &y,
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap();
        let MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, .. } = slot else {
            panic!("hierarchical slot");
        };

        // Closed-form within estimator, and the attenuated "demean y only" estimator.
        let demean = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; n];
            for g in 0..groups {
                let rows = g * per..(g + 1) * per;
                let mean = v[rows.clone()].iter().sum::<f64>() / per as f64;
                for i in rows {
                    out[i] = v[i] - mean;
                }
            }
            out
        };
        let (xd, yd) = (demean(&x), demean(&y));
        let beta_within = xd.iter().zip(&yd).map(|(a, b)| a * b).sum::<f64>()
            / xd.iter().map(|a| a * a).sum::<f64>();
        let x_mean = x.iter().sum::<f64>() / n as f64;
        let attenuated = x.iter().zip(&yd).map(|(a, b)| (a - x_mean) * b).sum::<f64>()
            / x.iter().map(|a| (a - x_mean).powi(2)).sum::<f64>();
        assert!(
            (coeffs[0] - beta_within).abs() < 1e-3 * beta_within.abs(),
            "slope {} vs within-unit estimator {beta_within}",
            coeffs[0]
        );
        assert!((attenuated - beta_within).abs() > 1.0, "attenuated estimator {attenuated}");

        let rss: f64 = (0..n).map(|i| (y[i] - intercept - coeffs[0] * x[i]).powi(2)).sum::<f64>();
        let expected_sigma = (rss / (n - 2) as f64).sqrt();
        assert!((sigma - expected_sigma).abs() < 1e-9, "sigma {sigma} vs {expected_sigma}");
        assert!(sigma > 0.3, "sigma {sigma} lost the between-unit spread");
    }

    /// The single-equation BVAR is a ridge fit on scale-standardized parents with the
    /// intercept unpenalized, so it is exactly equivariant to the units of the outcome
    /// and of every parent (`x → 1000x`, `y → 7y` moves the coefficient by `7/1000`,
    /// the intercept and σ by 7); and σ is the data-row residual SD of its own reported
    /// coefficients — the augmented solver RSS also holds the prior pseudo-rows.
    #[test]
    fn bvar_is_unit_equivariant_and_sigma_excludes_the_prior_rows() {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin() + 0.01 * i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.4 * (1.9 * i as f64).cos()).collect();
        let fit = |x: &[f64], y: &[f64]| -> (f64, f64, f64) {
            let data = TabularData::from_f64_columns([("x", x), ("y", y)]).unwrap();
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let compiled = CompiledCausalModel::compile(g).unwrap();
            let gather = compiled.gather_for(DenseNodeId::from_raw(1)).unwrap();
            let slot = fit_family(
                MechanismFamily::Bvar,
                gather,
                &compiled,
                &data,
                y,
                FaerBackend,
                &mut LeastSquaresWorkspace::default(),
            )
            .unwrap();
            let MechanismSlot::Bvar { intercept, coeffs, sigma } = slot else {
                panic!("bvar slot");
            };
            (intercept, coeffs[0], sigma)
        };
        let (i0, c0, s0) = fit(&x, &y);
        let x_big: Vec<f64> = x.iter().map(|v| v * 1000.0).collect();
        let y_big: Vec<f64> = y.iter().map(|v| v * 7.0).collect();
        let (i1, c1, s1) = fit(&x_big, &y_big);
        assert!((i1 - 7.0 * i0).abs() < 1e-9 * i0.abs().max(1.0), "intercept {i1} vs 7·{i0}");
        assert!((c1 - c0 * 7.0 / 1000.0).abs() < 1e-9 * c0.abs(), "coefficient {c1}");
        assert!((s1 - 7.0 * s0).abs() < 1e-9 * s0, "sigma {s1} vs 7·{s0}");

        let rss: f64 = (0..n).map(|i| (y[i] - i0 - c0 * x[i]).powi(2)).sum();
        let expected = (rss / (n - 2) as f64).sqrt();
        assert!((s0 - expected).abs() < 1e-12, "sigma {s0} vs data-row {expected}");
    }

    // ------------------------------------------------------------ basis families

    /// `z ~ N(0,1)`, `a ~ Bern(σ(0.5 z))`, `b ~ Bern(0.5)`, and
    /// `y = 0.8a + 0.5b + gain·ab + z + ε`. Graph `z → a`, `{z, a, b} → y`.
    fn interaction_table(n: usize, gain: f64, seed: u64) -> (TabularData, Dag) {
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let (mut z, mut a, mut b, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            z[i] = antecedent_kernels::standard_normal(&mut rng);
            a[i] = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-0.5 * z[i]).exp())));
            b[i] = f64::from(u8::from(rng.next_f64() < 0.5));
            y[i] = 0.8 * a[i]
                + 0.5 * b[i]
                + gain * a[i] * b[i]
                + z[i]
                + antecedent_kernels::standard_normal(&mut rng);
        }
        let data = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut g = Dag::with_variables(4);
        for (from, to) in [(0, 1), (0, 3), (1, 3), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        (data, g)
    }

    fn outcome_assignment(data: &TabularData, graph: Dag) -> MechanismAssignment {
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let (_, assignments) = MechanismRegistry::with_heterogeneity_families()
            .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
            .unwrap();
        assignments.into_iter().find(|a| a.node == DenseNodeId::from_raw(3)).unwrap()
    }

    #[test]
    fn heterogeneity_family_wins_only_where_the_data_has_an_interaction() {
        let (data, graph) = interaction_table(2500, 0.6, 7);
        let with = outcome_assignment(&data, graph);
        assert!(
            matches!(
                with.selected,
                MechanismFamily::LinearInteractions | MechanismFamily::LinearSpline
            ),
            "interaction data selected {:?}: {:?}",
            with.selected,
            with.candidates
        );
        assert!(!with.fitted.admits_no_effect_modification());

        let (data, graph) = interaction_table(2500, 0.0, 7);
        let without = outcome_assignment(&data, graph);
        assert_eq!(
            without.selected,
            MechanismFamily::LinearGaussian,
            "additive data must keep the linear family: {:?}",
            without.candidates
        );
        // The richer families were scored — the data, not the registry, rejected them.
        for family in [MechanismFamily::LinearInteractions, MechanismFamily::LinearSpline] {
            assert!(
                without.candidates.iter().any(|c| c.family == family),
                "{family:?} must have been scored: {:?}",
                without.failed_families
            );
        }
    }

    /// Rescaling the outcome must not change which family is selected, so the
    /// counterfactual contrast scales exactly with the outcome.
    #[test]
    fn family_selection_is_invariant_to_outcome_units() {
        for gain in [0.0, 0.05, 0.6] {
            let (data, graph) = interaction_table(900, gain, 13);
            let names = ["z", "a", "b", "y"];
            let cols: Vec<Vec<f64>> = (0..4)
                .map(|v| data.float64_cow(VariableId::from_raw(v)).unwrap().to_vec())
                .collect();
            for scale in [1e-3, 2.0, 1e4] {
                let y: Vec<f64> = cols[3].iter().map(|v| v * scale).collect();
                let scaled = TabularData::from_f64_columns([
                    (names[0], cols[0].as_slice()),
                    (names[1], cols[1].as_slice()),
                    (names[2], cols[2].as_slice()),
                    (names[3], y.as_slice()),
                ])
                .unwrap();
                let base = outcome_assignment(&data, graph.clone()).selected;
                let rescaled = outcome_assignment(&scaled, graph.clone()).selected;
                assert_eq!(base, rescaled, "gain={gain} scale={scale}");
            }
        }
    }

    #[test]
    fn basis_family_abduction_is_exact_inversion() {
        use crate::batch::{MechanismWorkspace, ParentBatch};
        use crate::mechanism::{NoiseInferenceMode, evaluate_column, infer_noise_column_rng};
        let (data, graph) = interaction_table(600, 0.6, 3);
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let gather = compiled.gather_for(DenseNodeId::from_raw(3)).unwrap().clone();
        let n = data.row_count();
        let y_col = data.float64_cow(VariableId::from_raw(3)).unwrap();
        let fitted = fit_family(
            MechanismFamily::LinearSpline,
            &gather,
            &compiled,
            &data,
            &y_col,
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap();
        let slot = &fitted;
        let MechanismSlot::LinearBasis { basis, .. } = slot else {
            panic!("spline family must fit a basis slot");
        };
        assert!(basis.knots()[0].len() == 3, "continuous z earns knots");
        assert!(
            basis.knots()[1].is_empty() && basis.knots()[2].is_empty(),
            "binary parents do not"
        );
        let mut parents = Vec::new();
        for v in 0..3 {
            parents.extend_from_slice(&data.float64_cow(VariableId::from_raw(v)).unwrap());
        }
        let batch = ParentBatch { n_rows: n, n_parents: 3, values: &parents };
        let y = data.float64_cow(VariableId::from_raw(3)).unwrap();
        let mut noise = vec![0.0; n];
        let mode = infer_noise_column_rng(
            slot,
            &y,
            batch,
            &mut noise,
            &mut antecedent_core::CausalRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Invertible);
        let mut rebuilt = vec![0.0; n];
        evaluate_column(slot, batch, &noise, &mut rebuilt, &mut MechanismWorkspace::default())
            .unwrap();
        for (r, (got, want)) in rebuilt.iter().zip(y.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "row {r}: {got} vs {want}");
        }
    }

    #[test]
    fn basis_families_refuse_a_single_parent_node() {
        let n = 400;
        let x: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.1).collect();
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (_, assignments) = MechanismRegistry::with_heterogeneity_families()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let y_assignment = assignments.iter().find(|a| a.node == DenseNodeId::from_raw(1)).unwrap();
        for family in [MechanismFamily::LinearInteractions, MechanismFamily::LinearSpline] {
            assert!(y_assignment.candidates.iter().all(|c| c.family != family));
            let reason =
                y_assignment.failed_families.iter().find(|(f, _)| *f == family).map_or_else(
                    || panic!("{family:?} must be recorded as failed"),
                    |(_, e)| e.as_str(),
                );
            assert!(reason.contains("at least two parents"), "{reason}");
        }
    }

    /// A parent-conditional categorical fit must be the same conditional law
    /// whether a parent is a raw year column or its z-score: the IRLS runs on
    /// standardized columns and back-transforms. Before, the raw fit could stop
    /// short of tolerance under row weights and refuse the whole analysis.
    #[test]
    fn discrete_fit_is_invariant_to_affine_parent_rescaling() {
        use crate::batch::ParentBatch;
        let n = 3000;
        let mut rng = antecedent_core::CausalRng::from_seed(11);
        let mut year = vec![0.0; n];
        let mut axis = vec![0.0; n];
        let mut lever = vec![0.0; n];
        for i in 0..n {
            year[i] = 2015.0 + (rng.next_f64() * 11.0).floor();
            axis[i] = rng.next_f64();
            let eta = 0.4 * (year[i] - 2020.0) / 3.16 + axis[i] - 0.5;
            lever[i] = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
        }
        let mean = year.iter().sum::<f64>() / n as f64;
        let sd = (year.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let year_std: Vec<f64> = year.iter().map(|v| (v - mean) / sd).collect();
        let fit = |year_col: &[f64], weights: Option<&[f64]>| {
            let data = TabularData::from_f64_columns([
                ("year", year_col),
                ("axis", axis.as_slice()),
                ("lever", lever.as_slice()),
            ])
            .unwrap();
            let mut g = Dag::with_variables(3);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
            g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
            let compiled = CompiledCausalModel::compile(g).unwrap();
            let gather = compiled.gather_for(DenseNodeId::from_raw(2)).unwrap().clone();
            let slot = fit_family_weighted(
                MechanismFamily::Discrete,
                &gather,
                &compiled,
                &data,
                &lever,
                FaerBackend,
                &mut LeastSquaresWorkspace::default(),
                weights,
            )
            .unwrap();
            let mut values = year_col.to_vec();
            values.extend_from_slice(&axis);
            let mut lp = vec![0.0; n];
            log_prob_column(
                &slot,
                &lever,
                ParentBatch { n_rows: n, n_parents: 2, values: &values },
                &mut lp,
            )
            .unwrap();
            lp
        };
        let weights: Vec<f64> =
            (0..n).map(|i| 0.2 + 1.6 * ((i * 7919) % 101) as f64 / 100.0).collect();
        for w in [None, Some(weights.as_slice())] {
            let raw = fit(&year, w);
            let std = fit(&year_std, w);
            for (r, (a, b)) in raw.iter().zip(&std).enumerate() {
                assert!((a - b).abs() < 1e-8, "row {r} weighted={}: {a} vs {b}", w.is_some());
            }
        }
    }
}
