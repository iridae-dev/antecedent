//! Mechanism registry and auto-assignment.
//!
//! Assignment returns candidates and scores; there is no silent default family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_stats::{FaerBackend, LeastSquaresWorkspace};

use crate::compile::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};
use crate::error::ModelError;
use crate::mechanism::NoiseInferenceMode;

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
    /// Gradient-boosted-tree conditional mean with homoscedastic Gaussian noise.
    ///
    /// `y = f(pa) + ε`, `ε ~ N(0, σ²)`, `f` a boosted ensemble and `σ` the
    /// root-mean-square of its out-of-fold residuals. Invertible. Needs the
    /// `ml-gbdt` feature to fit.
    TreeBoostedAdditive,
    /// Random-forest conditional mean with homoscedastic Gaussian noise; the
    /// same noise model as [`Self::TreeBoostedAdditive`]. Needs the `ml-forest`
    /// feature to fit.
    TreeForestAdditive,
    /// Categorical outcome whose class probabilities come from boosted-tree
    /// classifiers (one-vs-rest, renormalised; one classifier for a binary
    /// outcome). `U(0,1)` noise through the inverse CDF, posterior abduction.
    /// Needs the `ml-gbdt` feature to fit.
    TreeBoostedCategorical,
    /// Categorical outcome with random-forest class probabilities; the same
    /// noise model as [`Self::TreeBoostedCategorical`]. Needs the `ml-forest`
    /// feature to fit.
    TreeForestCategorical,
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
            | Self::DiscreteSpline
            | Self::TreeBoostedAdditive
            | Self::TreeForestAdditive
            | Self::TreeBoostedCategorical
            | Self::TreeForestCategorical => true,
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
            | Self::LinearSpline
            | Self::TreeBoostedAdditive
            | Self::TreeForestAdditive => true,
            Self::Discrete
            | Self::HierarchicalGlm
            | Self::DiscreteInteractions
            | Self::DiscreteSpline
            | Self::TreeBoostedCategorical
            | Self::TreeForestCategorical
            | Self::LinearGaussianStateSpace => false,
        }
    }

    /// Whether this family models a categorical outcome (a CDF-bin noise).
    #[must_use]
    pub const fn is_categorical(self) -> bool {
        matches!(
            self,
            Self::Discrete
                | Self::DiscreteInteractions
                | Self::DiscreteSpline
                | Self::HierarchicalGlm
                | Self::TreeBoostedCategorical
                | Self::TreeForestCategorical
        )
    }

    /// Whether this family models a real-valued outcome. [`Self::Constant`]
    /// fits either, so it is neither categorical nor continuous.
    #[must_use]
    pub const fn is_continuous(self) -> bool {
        !self.is_categorical() && !matches!(self, Self::Constant)
    }

    /// How counterfactual abduction recovers this family's exogenous noise.
    ///
    /// The disclosure carried by every result computed from a node of this
    /// family: [`NoiseInferenceMode::Invertible`] for an additive disturbance
    /// (exact residual), [`NoiseInferenceMode::Posterior`] for a CDF-bin noise
    /// (a draw from the observed category's bin, which rests on rank
    /// preservation of the noise across parent settings).
    #[must_use]
    pub const fn noise_inference_mode(self) -> NoiseInferenceMode {
        if self.is_categorical() {
            NoiseInferenceMode::Posterior
        } else {
            NoiseInferenceMode::Invertible
        }
    }

    /// Whether a node of this family participates in noise-based attribution
    /// (anomaly, Shapley, distribution-change): it samples, evaluates, abduces
    /// noise and names a reference noise value. True for every registered
    /// family; a categorical family is attributable under the posterior-noise
    /// assumption its [`Self::noise_inference_mode`] discloses.
    #[must_use]
    pub const fn attribution_compatible(self) -> bool {
        match self {
            Self::LinearGaussian
            | Self::Constant
            | Self::Discrete
            | Self::LinearInteractions
            | Self::LinearSpline
            | Self::DiscreteInteractions
            | Self::DiscreteSpline
            | Self::HierarchicalLinear
            | Self::HierarchicalGlm
            | Self::Bvar
            | Self::LinearGaussianStateSpace
            | Self::GaussianProcess
            | Self::TreeBoostedAdditive
            | Self::TreeForestAdditive
            | Self::TreeBoostedCategorical
            | Self::TreeForestCategorical => true,
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
    /// Mean held-out predictive log-likelihood before the complexity penalty (nats per
    /// row, on the same folds for every candidate of the node).
    pub held_out_log_likelihood: f64,
    /// The `k ln n / 2n` charge subtracted from the held-out log-likelihood to give
    /// [`Self::score`].
    pub complexity_penalty: f64,
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
    /// Candidate set, split scheme, seed and rule that produced the scores and the
    /// selection.
    pub provenance: SelectionProvenance,
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

impl MechanismTyping {
    /// Stable id of the rule that typed the node.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::DeclaredDiscrete => "declared_discrete",
            Self::LowCardinality => "low_cardinality",
            Self::Continuous => "continuous",
        }
    }
}

/// Most distinct values a column not declared discrete may take and still be
/// typed categorical.
pub(crate) const LOW_CARDINALITY_MAX_LEVELS: usize = 8;

/// Average rows per level below which a short column is not read as
/// categorical: a level set seen once or twice per level is a small sample of
/// a continuous variable, and a categorical fit would restrict counterfactual
/// support to the handful of values observed.
pub(crate) const LOW_CARDINALITY_MIN_ROWS_PER_LEVEL: usize = 5;

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
        MechanismFamily::TreeBoostedAdditive => "tree_boosted_additive",
        MechanismFamily::TreeForestAdditive => "tree_forest_additive",
        MechanismFamily::TreeBoostedCategorical => "tree_boosted_categorical",
        MechanismFamily::TreeForestCategorical => "tree_forest_categorical",
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

    /// Standard families plus the nonlinear ones: the cubic-spline expansions and
    /// the tree ensembles (boosted and forest regressions for continuous nodes,
    /// boosted and forest classifiers for categorical nodes).
    ///
    /// Candidates compete on the shared held-out criterion, so a flexible family
    /// wins only where it predicts rows it did not fit; the simpler family leads
    /// the listing and wins exact ties. The tree families need the `ml-gbdt` /
    /// `ml-forest` features to fit; without a provider they are recorded among
    /// the failed families instead of silently dropped.
    #[must_use]
    pub fn with_nonlinear_families() -> Self {
        Self {
            continuous: Arc::from(vec![
                MechanismFamily::LinearGaussian,
                MechanismFamily::LinearSpline,
                MechanismFamily::TreeBoostedAdditive,
                MechanismFamily::TreeForestAdditive,
                MechanismFamily::Constant,
            ]),
            discrete: Arc::from(vec![
                MechanismFamily::Discrete,
                MechanismFamily::DiscreteSpline,
                MechanismFamily::TreeBoostedCategorical,
                MechanismFamily::TreeForestCategorical,
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
        self.refit_weighted_polled(model, data, assignments, weights, &|| false)
    }

    /// [`Self::refit_weighted`] that asks `cancelled` before every node and stops
    /// with [`ModelError::Cancelled`] (no partial store) when it answers `true`.
    ///
    /// # Errors
    /// As [`Self::refit_weighted`], and [`ModelError::Cancelled`].
    pub fn refit_weighted_polled(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        assignments: &[MechanismAssignment],
        weights: &[f64],
        cancelled: &dyn Fn() -> bool,
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
            poll(cancelled)?;
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
                FitSource { gather, model, data },
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
        self.assign_and_fit_polled(model, data, policy, &|| false)
    }

    /// [`Self::assign_and_fit`] that asks `cancelled` before every node, before
    /// every candidate family and before every cross-validation fold fit, and
    /// stops with [`ModelError::Cancelled`] (no partial store) when it answers
    /// `true`. With `|| false` it is exactly [`Self::assign_and_fit`].
    ///
    /// # Errors
    ///
    /// As [`Self::assign_and_fit`], and [`ModelError::Cancelled`].
    pub fn assign_and_fit_polled(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        self.assign_and_fit_with_polled(
            model,
            data,
            policy,
            &SelectionOptions::default(),
            cancelled,
        )
    }

    /// [`Self::assign_and_fit`] with per-node family constraints and a declared
    /// validation-split scheme. With [`SelectionOptions::default`] it is exactly
    /// [`Self::assign_and_fit`].
    ///
    /// # Errors
    ///
    /// As [`Self::assign_and_fit`], plus [`ModelError::InvalidSelection`] for a
    /// constraint or split declaration the model or data cannot honour, and
    /// [`ModelError::RequiredFamilyUnavailable`] when a required family fails.
    pub fn assign_and_fit_with(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
        options: &SelectionOptions,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        self.assign_and_fit_with_polled(model, data, policy, options, &|| false)
    }

    /// [`Self::assign_and_fit_with`] that polls `cancelled` as
    /// [`Self::assign_and_fit_polled`] does.
    ///
    /// # Errors
    ///
    /// As [`Self::assign_and_fit_with`], and [`ModelError::Cancelled`].
    #[allow(
        clippy::too_many_lines,
        reason = "one pass per node: resolve candidates, score each, select, record provenance"
    )]
    pub fn assign_and_fit_with_polled(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
        options: &SelectionOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        let n = model.n_nodes();
        let nrows = data.row_count();
        if nrows == 0 {
            return Err(ModelError::Shape { message: "empty data for mechanism fit".into() });
        }
        validate_constraints(model, options)?;
        let mut slots = vec![MechanismSlot::Vacant; n];
        let mut assignments = Vec::with_capacity(n);
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();

        for gather in model.parent_gathers.iter() {
            poll(cancelled)?;
            let node = gather.child;
            let var = model.output_layout.variables[node.as_usize()];
            let y = data.float64_cow(var).map_err(ModelError::from)?;
            let typing = classify_node(data, var, &y);
            let registry_list: &[MechanismFamily] = if typing == MechanismTyping::Continuous {
                &self.continuous
            } else {
                &self.discrete
            };
            let constraint = options.constraint_for(var);
            let families = resolve_candidates(registry_list, typing, constraint, var)?;
            let frame =
                FitFrame::from_data(gather, model, data, y.len())?.with_splits(options.splits);
            let plan = frame.split_plan()?;

            let mut candidates = Vec::new();
            let mut fits: Vec<(MechanismFamily, MechanismSlot)> = Vec::new();
            let mut failed = Vec::new();
            for &family in &families {
                poll(cancelled)?;
                match score_family(family, &frame, &y, backend, &mut ls_ws, cancelled) {
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
                    // Cancellation is not a family failure: it must stop the whole fit.
                    Err(ModelError::Cancelled) => return Err(ModelError::Cancelled),
                    Err(e) => failed.push((family, e.to_string())),
                }
            }
            if candidates.is_empty() {
                let detail = failed
                    .iter()
                    .map(|(f, e)| format!("{}: {e}", f.id()))
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
            let (node_policy, rule) = match constraint.and_then(|c| c.required) {
                Some(family) => (
                    SelectionPolicy::RequireFamily(family),
                    SelectionRule::RequiredByNodeConstraint,
                ),
                None => match policy {
                    SelectionPolicy::BestScore => (policy, SelectionRule::BestScore),
                    SelectionPolicy::RequireFamily(_) => (policy, SelectionRule::RequiredByPolicy),
                },
            };
            let selected = node_policy.select(&candidates).ok_or_else(|| match node_policy {
                SelectionPolicy::RequireFamily(required) => {
                    let reason = failed
                        .iter()
                        .find(|(f, _)| *f == required)
                        .map_or("not among the node's candidates", |(_, reason)| reason.as_str());
                    ModelError::RequiredFamilyUnavailable {
                        message: format!(
                            "variable {var}: family {} could not be scored ({reason})",
                            required.id()
                        ),
                    }
                }
                SelectionPolicy::BestScore => ModelError::Unsupported {
                    message: "selection policy produced no family".into(),
                },
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
                provenance: SelectionProvenance {
                    splits: options.splits,
                    folds: plan.folds,
                    n_rows: nrows,
                    seed: plan.seed,
                    candidate_set: Arc::from(families),
                    rule,
                    constraint: constraint.cloned(),
                },
            });
        }

        Ok((CompiledMechanismStore { slots: Arc::from(slots) }, assignments))
    }
}

/// Refuse constraints that name a variable outside the model or repeat one.
fn validate_constraints(
    model: &CompiledCausalModel,
    options: &SelectionOptions,
) -> Result<(), ModelError> {
    for (i, (variable, _)) in options.constraints.iter().enumerate() {
        if !model.output_layout.variables.contains(variable) {
            return Err(ModelError::InvalidSelection {
                message: format!("constraint names variable {variable}, which is not in the model"),
            });
        }
        if options.constraints[..i].iter().any(|(v, _)| v == variable) {
            return Err(ModelError::InvalidSelection {
                message: format!("variable {variable} has more than one constraint"),
            });
        }
    }
    Ok(())
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

mod cv;
mod fit;
mod select;
mod selection;

pub use selection::{
    NodeFamilyConstraint, SelectionOptions, SelectionProvenance, SelectionRule, ValidationSplits,
};

use cv::{FitFrame, poll, score_family};
use fit::{FitSource, fit_family_weighted};
use select::classify_node;
use selection::resolve_candidates;

#[cfg(feature = "gaussian-process")]
pub use fit::GP_FAMILY_MAX_ROWS;

#[cfg(test)]
use cv::*;
#[cfg(test)]
use fit::*;
#[cfg(test)]
use select::*;

#[cfg(test)]
mod tests {
    use super::fit::lgssm_em;
    use super::*;
    use crate::mechanism::log_prob_column;
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
    /// grid, checked against a frozen `SciPy` oracle (`conformance/gcm/gaussian_process`).
    /// The diagonal-proxy determinant the old code used (MM-015) picks a different cell.
    #[cfg(feature = "gaussian-process")]
    #[test]
    fn gaussian_process_matches_exact_logdet_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../conformance/gcm/gaussian_process/expected.json"
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
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
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
            a[i] = f64::from(u8::from(rng.next_f64() < antecedent_kernels::sigmoid(0.5 * z[i])));
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
            lever[i] = f64::from(u8::from(rng.next_f64() < antecedent_kernels::sigmoid(eta)));
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
                FitSource { gather: &gather, model: &compiled, data: &data },
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
