//! Doubly robust value of a treatment policy from observational data.
//!
//! A policy assigns every unit one action from a declared treatment support. Its value is
//! `V(pi) = E[Y(pi(Z))]`, the mean outcome if each unit were given the action the policy
//! picks from its covariates `Z`. Under conditional exchangeability given the declared
//! covariates, positivity of every declared action, consistency and no interference, `V(pi)`
//! is identified, and the cross-fitted augmented inverse-probability score
//!
//! ```text
//! psi_i = m_{pi(Z_i)}(Z_i) + 1{T_i = pi(Z_i)} (Y_i - m_{pi(Z_i)}(Z_i)) / e_{pi(Z_i)}(Z_i)
//! ```
//!
//! has mean `V(pi)` when either the outcome regressions `m_a` or the multinomial-logit
//! propensities `e_a` are consistent. The standard error is the sample standard deviation of
//! the scores over the square root of the evaluation rows.
//!
//! The evaluation is explicit about where the policy came from, because a policy that was
//! chosen with the rows it is scored on has a value that is estimated too high:
//!
//! * [`PolicyEvaluation::Prespecified`] scores a policy fixed without these rows, on every
//!   row, with nuisance models cross-fitted over a seeded fold plan. A policy declared as
//!   learned on the evaluation rows is refused rather than scored.
//! * [`PolicyEvaluation::HeldOut`] learns the policy (the action with the best predicted
//!   outcome) and its nuisance models on a seeded learning part and scores that one policy on
//!   the independent held-out rows.
//! * [`PolicyEvaluation::CrossFitted`] is selection aware for a policy learned on the same
//!   data: for each fold the policy is learned on the other folds and scored on the fold, so
//!   every score uses a policy that never saw its row. The estimate is the value of the
//!   learning procedure fed `(K - 1) / K` of the data, not of one final policy. The
//!   standard error pools the scores and treats the fold policies as fixed: that is valid
//!   when the learned policy stabilises across folds (the cross-fitting argument of Athey
//!   and Wager, 2021; Zhou, Athey and Wager, 2023), as for a parametric rule fitted on
//!   `(K - 1) / K` of a large sample. When the learned rule varies materially between folds
//!   the pooled standard error is only approximate and carries no correction here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::identity::{CanonicalWriter, ReusableProduct};
use antecedent_core::{CausalRng, ExecutionContext, SemanticDigest};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace};

use crate::aipw::predict_colmajor;
use crate::crossfit_plan::{
    CrossfitPlan, FoldRows, FoldSource, SHARED_FOLD_ID_ERRORS, rows_colmajor,
};
use crate::error::EstimationError;
use crate::nuisance_cache::{
    FoldIdentity, NuisanceCacheKey, NuisanceKeyBuilder, NuisanceOptions, ProviderDependency,
};
use crate::nuisance_keys::{PROVIDER_VERSION, glm_numerical};
use crate::propensity::{CategoryTrainingRows, fit_category_probabilities};
use crate::util::{refuse, stats_err};

/// Smallest number of rows the value is scored on.
pub const MIN_EVALUATION_ROWS: usize = 50;

/// Default cross-fit fold count.
pub const DEFAULT_POLICY_FOLDS: usize = 5;

/// Default positivity floor on the propensity of every declared action.
pub const DEFAULT_MIN_PROPENSITY: f64 = 0.01;

/// Whether a larger or a smaller outcome is better.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyObjective {
    /// Choose the action with the largest predicted outcome.
    Maximize,
    /// Choose the action with the smallest predicted outcome.
    Minimize,
}

impl PolicyObjective {
    /// Stable spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Maximize => "maximize",
            Self::Minimize => "minimize",
        }
    }
}

/// How the declared policy relates to the rows it is scored on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyOrigin {
    /// The policy was fixed without looking at the outcomes or treatments of these rows.
    Prespecified,
    /// The policy was learned from these rows: its value cannot be scored on them.
    LearnedOnEvaluationRows,
}

/// Where the evaluated policy comes from and which rows score it.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicyEvaluation {
    /// A fixed policy: the declared action of every row, scored on all rows.
    Prespecified {
        /// The action the policy takes at each row, a level of the declared support.
        actions: Arc<[f64]>,
        /// Where the policy came from.
        origin: PolicyOrigin,
    },
    /// A policy learned on a seeded learning part and scored on the independent rest.
    HeldOut {
        /// Fraction of the rows (per treatment level) held out for scoring, in `(0, 1)`.
        evaluation_fraction: f64,
    },
    /// A policy learned on the other folds for every fold, scored on the fold.
    CrossFitted,
}

impl PolicyEvaluation {
    /// Stable spelling.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Prespecified { .. } => "prespecified",
            Self::HeldOut { .. } => "held_out",
            Self::CrossFitted => "cross_fitted_learning",
        }
    }
}

/// The value of one action given to every unit, on the scored rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionValue {
    /// Treatment level.
    pub level: f64,
    /// Doubly robust mean outcome if every scored unit were given this action.
    pub value: f64,
    /// Standard error of that mean.
    pub se: f64,
    /// Fraction of the scored rows to which the policy gives this action.
    pub policy_share: f64,
    /// Smallest out-of-sample propensity of this action over the scored rows.
    pub propensity_min: f64,
    /// Complete-case rows observed at this action.
    pub rows: u64,
}

/// The policy against a single declared action given to everyone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceComparison {
    /// The reference action.
    pub level: f64,
    /// Policy value minus the reference action's value, on the same scored rows.
    pub incremental_value: f64,
    /// Standard error of the paired difference.
    pub incremental_se: f64,
}

/// A policy value with the evidence needed to read and replay it.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValueEstimate {
    /// How the policy was obtained and scored.
    pub evaluation: PolicyEvaluation,
    /// The objective the policy optimizes (also the direction of learned policies).
    pub objective: PolicyObjective,
    /// Doubly robust policy value.
    pub value: f64,
    /// Influence-function standard error (calibration unmeasured).
    pub se: f64,
    /// The value of each declared action given to everyone, in declaration order.
    pub action_values: Vec<ActionValue>,
    /// The policy against the declared reference action, when one was declared.
    pub reference: Option<ReferenceComparison>,
    /// Rows the value is scored on.
    pub evaluation_rows: u64,
    /// Rows the policy and nuisance models were fitted on; for cross-fitted learning the
    /// smallest training set over folds.
    pub learning_rows: u64,
    /// Rows dropped for a missing or non-finite cell.
    pub rows_dropped_missing: u64,
    /// Original row index of each scored row.
    pub row_index: Vec<antecedent_data::GlobalId>,
    /// The action the policy takes at each scored row.
    pub policy_actions: Vec<f64>,
    /// The doubly robust score of each scored row.
    pub scores: Vec<f64>,
}

/// Declared assignment probabilities of a randomized or otherwise known design, which replace
/// the fitted propensity model.
#[derive(Clone, Debug, PartialEq)]
pub enum KnownPropensity {
    /// One probability per declared action, in support order, the same for every row.
    Constant(Arc<[f64]>),
    /// One column per declared action, in support order, with one probability per row. A `NaN`
    /// is a missing cell and drops the row.
    PerRow(Arc<[Arc<[f64]>]>),
}

/// Columns of an observational policy-evaluation problem. `NaN` is a missing cell.
#[derive(Clone, Copy, Debug)]
pub struct PolicyData<'a> {
    /// Outcome of each row.
    pub outcome: &'a [f64],
    /// Observed treatment level of each row.
    pub treatment: &'a [f64],
    /// Declared adjustment columns, each with one value per row.
    pub covariates: &'a [&'a [f64]],
}

/// Observational policy value under a declared treatment support.
#[derive(Clone, Debug)]
pub struct ObservationalPolicyValue {
    /// The declared treatment support: the actions the policy may take and whose propensity
    /// must be positive. Every observed treatment value must be one of them.
    pub support: Arc<[f64]>,
    /// Whether larger or smaller outcomes are better; sets the direction of learned policies.
    pub objective: PolicyObjective,
    /// Where the policy comes from and which rows score it. There is no default.
    pub evaluation: PolicyEvaluation,
    /// An action to compare the policy against, a level of the support.
    pub reference: Option<f64>,
    /// Cross-fit folds (prespecified and cross-fitted learning).
    pub folds: usize,
    /// Smallest propensity tolerated for any declared action at any scored row.
    pub min_propensity: f64,
    /// Known assignment probabilities that replace the fitted propensity model; `None`
    /// estimates them by a multinomial logit on the declared covariates.
    pub known_propensity: Option<KnownPropensity>,
    /// GLM options of the propensity model.
    pub glm_options: GlmOptions,
}

impl ObservationalPolicyValue {
    /// A request with the default folds and positivity floor and no reference action.
    #[must_use]
    pub fn new(
        support: impl Into<Arc<[f64]>>,
        objective: PolicyObjective,
        evaluation: PolicyEvaluation,
    ) -> Self {
        Self {
            support: support.into(),
            objective,
            evaluation,
            reference: None,
            folds: DEFAULT_POLICY_FOLDS,
            min_propensity: DEFAULT_MIN_PROPENSITY,
            known_propensity: None,
            glm_options: GlmOptions::default(),
        }
    }

    /// Estimate the value of the declared policy.
    ///
    /// # Errors
    ///
    /// `invalid_argument` for malformed columns, an invalid support, reference, fold count,
    /// positivity floor or held-out fraction; `treatment_level_undeclared` for a row at no
    /// declared action; `positivity_violated` for a policy action outside the support or an
    /// action whose propensity falls below the floor; `policy_learned_on_evaluation_rows` for a
    /// policy declared as learned on the rows that score it; `arm_not_populated` for an
    /// action with no or too few rows; `cancelled_no_claim`; and GLM or regression failures.
    pub fn estimate(
        &self,
        data: &PolicyData<'_>,
        ctx: &ExecutionContext,
    ) -> Result<PolicyValueEstimate, EstimationError> {
        self.validate()?;
        let prepared = Prepared::new(self, data)?;
        let k = self.support.len();
        let n = prepared.n;
        let plan = match &self.evaluation {
            PolicyEvaluation::HeldOut { evaluation_fraction } => {
                Splits::held_out(&prepared.category, *evaluation_fraction, ctx.rng.master_seed())?
            }
            PolicyEvaluation::Prespecified { .. } | PolicyEvaluation::CrossFitted => {
                Splits::folds(&prepared, self.folds, ctx.rng.master_seed())?
            }
        };
        // The split nuisances do not depend on the policy being scored, so a batch of policies
        // over the same rows, support and splits fits them once.
        let fits = crate::crossfit_cache::reuse_in_active_scope(
            || self.nuisance_key(&prepared, &plan, ctx.rng.master_seed()),
            (16 * k as u64).saturating_mul(n as u64).saturating_add(1024),
            &|| if ctx.cancellation.is_cancelled() { Err(policy_cancelled()) } else { Ok(()) },
            || self.fit_splits(&prepared, &plan, ctx),
        )?;
        let mut scored = Scored::new(n, k);
        for (split, nuisance) in plan.splits().zip(fits.nuisances.iter()) {
            scored.add_split(self, &prepared, split.valid, nuisance);
        }
        scored.flatten_action_scores();
        self.finish(&prepared, scored, fits.learning_rows)
    }

    /// Fit the propensity and outcome models of every split.
    fn fit_splits(
        &self,
        prepared: &Prepared,
        plan: &Splits,
        ctx: &ExecutionContext,
    ) -> Result<PolicyFits, EstimationError> {
        let mut learning_rows = usize::MAX;
        let mut workspace = FitWorkspace::default();
        let mut nuisances = Vec::with_capacity(plan.parts.len());
        for split in plan.splits() {
            if ctx.cancellation.is_cancelled() {
                return Err(policy_cancelled());
            }
            learning_rows = learning_rows.min(split.train.len());
            nuisances.push(self.fit_split(prepared, split.train, split.valid, &mut workspace)?);
        }
        Ok(PolicyFits { nuisances, learning_rows })
    }

    /// Key of the split nuisances, from the actual design, observed actions, outcome, declared
    /// support, realized splits, propensity inputs and seed. `None` when a part does not hash
    /// (then the fit is not shared).
    fn nuisance_key(
        &self,
        prepared: &Prepared,
        plan: &Splits,
        seed: u64,
    ) -> Option<NuisanceCacheKey> {
        let parts = plan.parts.len();
        let mut validated_in = vec![u32::try_from(parts).ok()?; prepared.n];
        let mut splits = CanonicalWriter::new("antecedent.policy.splits");
        splits.word(parts as u64);
        for (index, (train, valid)) in plan.parts.iter().enumerate() {
            for &row in valid {
                *validated_in.get_mut(row)? = u32::try_from(index).ok()?;
            }
            splits.word(train.len() as u64);
            for &row in train {
                splits.word(row as u64);
            }
            splits.word(valid.len() as u64);
            for &row in valid {
                splits.word(row as u64);
            }
        }
        let actions: Vec<f64> = prepared.category.iter().map(|&c| f64::from(c)).collect();
        let options = NuisanceOptions::new("policy_nuisance")
            .text("glm", &format!("{:?}", self.glm_options))
            .text("evaluation", self.evaluation.as_str())
            .digest(
                "support",
                Some(SemanticDigest::of_reals("antecedent.policy.support", &self.support)),
            )
            .digest(
                "known_propensity",
                prepared.known_propensity.as_deref().map(|known| {
                    SemanticDigest::of_reals("antecedent.policy.known_propensity", known)
                }),
            )
            .digest("splits", Some(SemanticDigest::from_bytes(splits.finish())));
        NuisanceKeyBuilder::new(ReusableProduct::PolicyNuisance)
            .design_matrix(prepared.n, prepared.ncols, &prepared.design)
            .rows_by_id(prepared.row_index.iter().map(|row| row.get()))
            .folds(
                FoldIdentity::of_fold_ids(
                    u32::try_from(parts).ok()?.checked_add(1)?,
                    &validated_in,
                )
                .ok()?,
            )
            .treatment_values("observed_action", &[&actions])
            .outcome(&prepared.outcome)
            .weights(None)
            .provider(ProviderDependency::named("policy_value", PROVIDER_VERSION))
            .options(&options)
            .seed(seed)
            .numerical(glm_numerical(&self.glm_options)?)
            .build()
            .ok()
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let invalid = |detail: &str, message: &str| {
            refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        let levels = &self.support;
        let mut sorted = levels.to_vec();
        sorted.sort_by(f64::total_cmp);
        #[allow(
            clippy::float_cmp,
            reason = "declared treatment levels are exact category labels, not computed values"
        )]
        let repeated = sorted.windows(2).any(|pair| pair[0] == pair[1]);
        if levels.len() < 2 || levels.iter().any(|l| !l.is_finite()) || repeated {
            return Err(invalid(
                "policy_value.invalid_support",
                "the declared treatment support must be at least two distinct finite values",
            ));
        }
        if let Some(reference) = self.reference {
            if self.index_of(reference).is_none() {
                return Err(invalid(
                    "policy_value.reference_outside_support",
                    "the reference action is not a level of the declared treatment support",
                ));
            }
        }
        if !self.min_propensity.is_finite() || !(0.0..0.5).contains(&self.min_propensity) {
            return Err(invalid(
                "policy_value.invalid_propensity_floor",
                "the positivity floor must be finite and in [0, 0.5)",
            ));
        }
        if let PolicyEvaluation::HeldOut { evaluation_fraction } = &self.evaluation {
            if !evaluation_fraction.is_finite()
                || !(0.0..1.0).contains(evaluation_fraction)
                || *evaluation_fraction == 0.0
            {
                return Err(invalid(
                    "policy_value.invalid_evaluation_fraction",
                    "the held-out evaluation fraction must be strictly between 0 and 1",
                ));
            }
        } else if self.folds < 2 {
            return Err(invalid(
                "policy_value.invalid_folds",
                "cross-fitting requires at least two folds",
            ));
        }
        if let PolicyEvaluation::Prespecified {
            origin: PolicyOrigin::LearnedOnEvaluationRows,
            ..
        } = &self.evaluation
        {
            return Err(refuse(
                antecedent_core::reason_code!("policy_learned_on_evaluation_rows"),
                "policy_value.policy_learned_on_evaluation_rows",
                "a policy learned on the rows that score it has a value estimated too high; evaluate it on independent held-out rows (held_out) or learn and score it by cross-fitting (cross_fitted_learning)",
            ));
        }
        Ok(())
    }

    #[allow(
        clippy::float_cmp,
        reason = "declared treatment levels are exact category labels, not computed values"
    )]
    fn index_of(&self, level: f64) -> Option<usize> {
        self.support.iter().position(|&l| l == level)
    }

    /// Fit the propensity and per-action outcome models on `train` and predict `valid`.
    fn fit_split(
        &self,
        prepared: &Prepared,
        train: &[usize],
        valid: &[usize],
        workspace: &mut FitWorkspace,
    ) -> Result<SplitNuisance, EstimationError> {
        let k = self.support.len();
        let (n, ncols) = (prepared.n, prepared.ncols);
        let train_category: Vec<u32> = train.iter().map(|&i| prepared.category[i]).collect();
        let mut counts = vec![0usize; k];
        for &c in &train_category {
            counts[c as usize] += 1;
        }
        if let Some(missing) = counts.iter().position(|&seen| seen == 0) {
            return Err(refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "policy_value.training_split_missing_action",
                &format!(
                    "a training split has no row at action {}; use fewer folds or a smaller held-out fraction",
                    self.support[missing]
                ),
            ));
        }
        let design_train = rows_colmajor(&prepared.design, n, ncols, train);
        let design_valid = rows_colmajor(&prepared.design, n, ncols, valid);
        let nv = valid.len();
        let propensity = if let Some(known) = &prepared.known_propensity {
            let mut out = vec![0.0; k * nv];
            for c in 0..k {
                for (r, &i) in valid.iter().enumerate() {
                    out[c * nv + r] = known[c * n + i];
                }
            }
            out
        } else {
            fit_category_probabilities(
                &CategoryTrainingRows {
                    design: &design_train,
                    category: &train_category,
                    n_categories: k,
                    ncols,
                },
                &design_valid,
                nv,
                &self.glm_options,
                &mut workspace.glm,
            )?
        };
        let mut mean = vec![0.0; k * nv];
        let mut prediction = Vec::new();
        for c in 0..k {
            let level_train: Vec<usize> =
                train.iter().copied().filter(|&i| prepared.category[i] as usize == c).collect();
            // The outcome regression needs residual degrees of freedom; an exact interpolant
            // would leave double robustness resting on the propensity alone.
            if level_train.len() <= ncols {
                return Err(refuse(
                    antecedent_core::reason_code!("arm_not_populated"),
                    "policy_value.sparse_action",
                    &format!(
                        "a training split has no more rows at action {} than design columns, so its outcome model has no residual degrees of freedom",
                        self.support[c]
                    ),
                ));
            }
            let design_level = rows_colmajor(&prepared.design, n, ncols, &level_train);
            let y_level: Vec<f64> = level_train.iter().map(|&i| prepared.outcome[i]).collect();
            let fit = FaerBackend
                .least_squares(
                    &design_level,
                    level_train.len(),
                    ncols,
                    &y_level,
                    &mut workspace.ols,
                )
                .map_err(stats_err)?;
            predict_colmajor(&design_valid, nv, ncols, &fit.coefficients, &mut prediction);
            mean[c * nv..(c + 1) * nv].copy_from_slice(&prediction);
        }
        Ok(SplitNuisance { propensity, mean, rows: nv })
    }

    fn finish(
        &self,
        prepared: &Prepared,
        scored: Scored,
        learning_rows: usize,
    ) -> Result<PolicyValueEstimate, EstimationError> {
        let k = self.support.len();
        let m = scored.rows.len();
        if m < MIN_EVALUATION_ROWS {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "policy_value.too_few_evaluation_rows",
                &format!(
                    "the value is scored on {m} rows; at least {MIN_EVALUATION_ROWS} are required"
                ),
            ));
        }
        if let Some((c, &lowest)) = scored
            .propensity_min
            .iter()
            .enumerate()
            .find(|&(_, &low)| low.is_nan() || low < self.min_propensity)
        {
            return Err(refuse(
                antecedent_core::reason_code!("positivity_violated"),
                "policy_value.action_propensity_below_floor",
                &format!(
                    "the propensity of action {} falls to {lowest:.6} on a scored row, below the declared floor {}; the doubly robust score weights by its inverse. Drop the action from the declared support, restrict the population, or lower the floor knowingly",
                    self.support[c], self.min_propensity
                ),
            ));
        }
        let mean_se = |values: &dyn Fn(usize) -> f64| {
            let mean = (0..m).map(values).sum::<f64>() / m as f64;
            let var = (0..m).map(|r| (values(r) - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
            (mean, (var / m as f64).sqrt())
        };
        let (value, se) = mean_se(&|r| scored.policy_score[r]);
        let mut action_values = Vec::with_capacity(k);
        for c in 0..k {
            let (v, s) = mean_se(&|r| scored.action_score[c * m + r]);
            action_values.push(ActionValue {
                level: self.support[c],
                value: v,
                se: s,
                policy_share: scored.policy_count[c] as f64 / m as f64,
                propensity_min: scored.propensity_min[c],
                rows: prepared.rows_by_level[c],
            });
        }
        let reference = self.reference.and_then(|level| self.index_of(level)).map(|c| {
            let (incremental_value, incremental_se) =
                mean_se(&|r| scored.policy_score[r] - scored.action_score[c * m + r]);
            ReferenceComparison { level: self.support[c], incremental_value, incremental_se }
        });
        let policy_actions =
            scored.policy_choice.iter().map(|&c| self.support[c as usize]).collect();
        let row_index = scored.rows.iter().map(|&i| prepared.row_index[i]).collect();
        Ok(PolicyValueEstimate {
            evaluation: self.evaluation.clone(),
            objective: self.objective,
            value,
            se,
            action_values,
            reference,
            evaluation_rows: m as u64,
            learning_rows: if learning_rows == usize::MAX { 0 } else { learning_rows as u64 },
            rows_dropped_missing: prepared.dropped,
            row_index,
            policy_actions,
            scores: scored.policy_score,
        })
    }
}

#[derive(Default)]
struct FitWorkspace {
    glm: LeastSquaresWorkspace,
    ols: LeastSquaresWorkspace,
}

/// The split nuisances of one policy-value request, and the fewest rows any split trained on.
struct PolicyFits {
    nuisances: Vec<SplitNuisance>,
    learning_rows: usize,
}

fn policy_cancelled() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "policy_value.cancelled",
        "the cross-fitted policy-value fit was cancelled before every split was fit; no estimate is reported",
    )
}

/// Out-of-sample propensities and outcome regressions of one split's scored rows, action `c`
/// of row `r` at `c * rows + r`.
struct SplitNuisance {
    propensity: Vec<f64>,
    mean: Vec<f64>,
    rows: usize,
}

/// Accumulated scores of every scored row, in the order the splits were scored.
struct Scored {
    rows: Vec<usize>,
    policy_choice: Vec<u32>,
    policy_score: Vec<f64>,
    /// Every action's score on each split's rows, action `c` of row `r` at `c * rows + r`.
    action_score_by_split: Vec<Vec<f64>>,
    /// The same scores over all scored rows once the splits are laid out.
    action_score: Vec<f64>,
    policy_count: Vec<usize>,
    propensity_min: Vec<f64>,
    k: usize,
}

impl Scored {
    fn new(n: usize, k: usize) -> Self {
        Self {
            rows: Vec::with_capacity(n),
            policy_choice: Vec::with_capacity(n),
            policy_score: Vec::with_capacity(n),
            action_score_by_split: Vec::new(),
            action_score: Vec::new(),
            policy_count: vec![0; k],
            propensity_min: vec![f64::INFINITY; k],
            k,
        }
    }

    /// Score the rows of one split: choose each row's action, form every action's
    /// augmented score, and keep the policy's.
    fn add_split(
        &mut self,
        request: &ObservationalPolicyValue,
        prepared: &Prepared,
        valid: &[usize],
        nuisance: &SplitNuisance,
    ) {
        let (k, nv) = (self.k, nuisance.rows);
        debug_assert_eq!(nv, valid.len());
        let mut action_scores = vec![0.0; k * nv];
        for (r, &i) in valid.iter().enumerate() {
            let observed = prepared.category[i] as usize;
            let y = prepared.outcome[i];
            for c in 0..k {
                let e = nuisance.propensity[c * nv + r];
                let mu = nuisance.mean[c * nv + r];
                self.propensity_min[c] = self.propensity_min[c].min(e);
                action_scores[c * nv + r] =
                    mu + f64::from(observed == c) * (y - mu) / e.max(f64::MIN_POSITIVE);
            }
            let choice = match &request.evaluation {
                PolicyEvaluation::Prespecified { .. } => prepared.declared_action[i] as usize,
                PolicyEvaluation::HeldOut { .. } | PolicyEvaluation::CrossFitted => {
                    best_action(request.objective, (0..k).map(|c| nuisance.mean[c * nv + r]))
                }
            };
            self.rows.push(i);
            self.policy_choice.push(u32::try_from(choice).unwrap_or(u32::MAX));
            self.policy_score.push(action_scores[choice * nv + r]);
            self.policy_count[choice] += 1;
        }
        self.action_score_by_split.push(action_scores);
    }

    /// Lay the per-split action scores out as one `k × rows` block in scored-row order.
    fn flatten_action_scores(&mut self) {
        let m = self.rows.len();
        let mut flat = vec![0.0; self.k * m];
        let mut offset = 0;
        for split in &self.action_score_by_split {
            let nv = split.len() / self.k;
            for c in 0..self.k {
                flat[c * m + offset..c * m + offset + nv]
                    .copy_from_slice(&split[c * nv..(c + 1) * nv]);
            }
            offset += nv;
        }
        self.action_score = flat;
        self.action_score_by_split.clear();
    }
}

/// Index of the best of `values` for `objective`; the first of equals.
fn best_action(objective: PolicyObjective, values: impl Iterator<Item = f64>) -> usize {
    let mut best = 0;
    let mut best_value = f64::NAN;
    for (c, v) in values.enumerate() {
        let better = match objective {
            PolicyObjective::Maximize => v > best_value,
            PolicyObjective::Minimize => v < best_value,
        };
        if c == 0 || better {
            best = c;
            best_value = v;
        }
    }
    best
}

/// The training and scored rows of each split.
struct Splits {
    parts: Vec<(Vec<usize>, Vec<usize>)>,
}

struct SplitRows<'a> {
    train: &'a [usize],
    valid: &'a [usize],
}

impl Splits {
    fn splits(&self) -> impl Iterator<Item = SplitRows<'_>> {
        self.parts.iter().map(|(train, valid)| SplitRows { train, valid })
    }

    fn folds(prepared: &Prepared, folds: usize, seed: u64) -> Result<Self, EstimationError> {
        if folds > prepared.n {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "policy_value.invalid_folds",
                "cross-fitting folds cannot exceed the complete-case rows",
            ));
        }
        let plan = CrossfitPlan::derive(
            prepared.n,
            folds,
            FoldSource::Stratified { strata: &prepared.category, units: &prepared.row_index, seed },
            &SHARED_FOLD_ID_ERRORS,
        )?;
        let parts = (0..folds)
            .map(|fold| {
                let FoldRows { train, valid, .. } = plan.rows(fold);
                (train.to_vec(), valid.to_vec())
            })
            .collect();
        Ok(Self { parts })
    }

    /// A seeded split holding out `fraction` of the rows at every treatment level.
    fn held_out(category: &[u32], fraction: f64, seed: u64) -> Result<Self, EstimationError> {
        let levels = category.iter().copied().max().map_or(0, |top| top as usize + 1);
        let mut held = vec![false; category.len()];
        for level in 0..levels {
            let mut members: Vec<usize> =
                (0..category.len()).filter(|&i| category[i] as usize == level).collect();
            let mut rng =
                CausalRng::from_seed(seed ^ (level as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for i in (1..members.len()).rev() {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a uniform draw in [0, 1) scaled by i + 1 is a non-negative index at most i"
                )]
                let j = (rng.next_f64() * (i as f64 + 1.0)) as usize;
                members.swap(i, j.min(i));
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the product is a non-negative count not above the stratum size"
            )]
            let take = ((members.len() as f64 * fraction).ceil() as usize).min(members.len());
            for &i in &members[..take] {
                held[i] = true;
            }
        }
        let (valid, train): (Vec<usize>, Vec<usize>) = (0..category.len()).partition(|&i| held[i]);
        if train.is_empty() || valid.is_empty() {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "policy_value.invalid_evaluation_fraction",
                "the held-out split leaves no learning or no evaluation rows",
            ));
        }
        Ok(Self { parts: vec![(train, valid)] })
    }
}

/// The complete-case design of a policy-evaluation problem.
struct Prepared {
    n: usize,
    ncols: usize,
    /// Column-major `[1 | Z...]`.
    design: Vec<f64>,
    /// Index into the declared support of each row's observed treatment.
    category: Vec<u32>,
    /// Index into the declared support of each row's declared action (prespecified only;
    /// zero otherwise).
    declared_action: Vec<u32>,
    /// Known assignment probabilities of the complete-case rows, action `c` of row `i` at
    /// `c * n + i`; `None` when the propensity is fitted.
    known_propensity: Option<Vec<f64>>,
    outcome: Vec<f64>,
    /// Original row index of each complete-case row.
    row_index: Vec<antecedent_data::GlobalId>,
    rows_by_level: Vec<u64>,
    dropped: u64,
}

impl Prepared {
    #[allow(
        clippy::float_cmp,
        reason = "declared treatment levels are exact category labels, not computed values"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one linear preparation: validate the request and data, then assemble the prepared rows"
    )]
    fn new(
        request: &ObservationalPolicyValue,
        data: &PolicyData<'_>,
    ) -> Result<Self, EstimationError> {
        let invalid = |detail: &str, message: &str| {
            refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        let rows = data.outcome.len();
        let declared: Option<&[f64]> = match &request.evaluation {
            PolicyEvaluation::Prespecified { actions, .. } => Some(actions),
            _ => None,
        };
        if rows == 0
            || data.treatment.len() != rows
            || data.covariates.iter().any(|c| c.len() != rows)
            || declared.is_some_and(|actions| actions.len() != rows)
        {
            return Err(invalid(
                "policy_value.invalid_data",
                "outcome, treatment, covariates and the declared policy must be non-empty and of equal length",
            ));
        }
        let infinite = |values: &[f64]| values.iter().any(|v| v.is_infinite());
        if infinite(data.outcome)
            || infinite(data.treatment)
            || data.covariates.iter().any(|c| infinite(c))
            || declared.is_some_and(infinite)
        {
            return Err(invalid("policy_value.invalid_data", "no value may be infinite"));
        }
        let k = request.support.len();
        let per_row: &[Arc<[f64]>] = match &request.known_propensity {
            Some(KnownPropensity::PerRow(columns)) => columns,
            _ => &[],
        };
        if !per_row.is_empty()
            && (per_row.len() != k || per_row.iter().any(|column| column.len() != rows))
        {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "policy_value.invalid_known_propensity",
                "per-row known propensities need one column per declared action, each with one value per row",
            ));
        }
        let complete = |i: usize| {
            !data.outcome[i].is_nan()
                && !data.treatment[i].is_nan()
                && data.covariates.iter().all(|c| !c[i].is_nan())
                && declared.is_none_or(|actions| !actions[i].is_nan())
                && per_row.iter().all(|column| !column[i].is_nan())
        };
        let keep: Vec<usize> = (0..rows).filter(|&i| complete(i)).collect();
        let n = keep.len();
        let dropped = (rows - n) as u64;
        let ncols = 1 + data.covariates.len();
        if n <= ncols * k {
            return Err(invalid(
                "policy_value.too_few_rows",
                "the complete-case rows cannot support an outcome model per action",
            ));
        }
        let mut category = Vec::with_capacity(n);
        let mut declared_action = Vec::with_capacity(n);
        let mut rows_by_level = vec![0u64; k];
        for &i in &keep {
            let Some(c) = request.index_of(data.treatment[i]) else {
                return Err(refuse(
                    antecedent_core::reason_code!("treatment_level_undeclared"),
                    "policy_value.undeclared_level",
                    &format!(
                        "row {i} has treatment {}, which is not a level of the declared treatment support; declare the level or remove the row",
                        data.treatment[i]
                    ),
                ));
            };
            category.push(u32::try_from(c).unwrap_or(u32::MAX));
            rows_by_level[c] += 1;
            if let Some(actions) = declared {
                let Some(a) = request.index_of(actions[i]) else {
                    return Err(refuse(
                        antecedent_core::reason_code!("positivity_violated"),
                        "policy_value.action_outside_support",
                        &format!(
                            "the policy takes action {} at row {i}, which is outside the declared treatment support, so its value is not identified from these data",
                            actions[i]
                        ),
                    ));
                };
                declared_action.push(u32::try_from(a).unwrap_or(u32::MAX));
            } else {
                declared_action.push(0);
            }
        }
        if let Some(empty) = rows_by_level.iter().position(|&r| r == 0) {
            return Err(refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "policy_value.action_not_populated",
                &format!("no complete-case row has the declared action {}", request.support[empty]),
            ));
        }
        let mut design = vec![0.0; n * ncols];
        design[..n].fill(1.0);
        for (j, column) in data.covariates.iter().enumerate() {
            for (r, &i) in keep.iter().enumerate() {
                design[(1 + j) * n + r] = column[i];
            }
        }
        let row_index = keep
            .iter()
            .map(|&i| {
                u64::try_from(i).map(antecedent_data::GlobalId::new).map_err(|_| {
                    EstimationError::data_msg(
                        "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let known_propensity = match &request.known_propensity {
            None => None,
            Some(known) => {
                let mut out = vec![0.0; k * n];
                for c in 0..k {
                    for (r, &i) in keep.iter().enumerate() {
                        out[c * n + r] = match known {
                            KnownPropensity::Constant(values) => values[c],
                            KnownPropensity::PerRow(columns) => columns[c][i],
                        };
                    }
                }
                let sums_to_one =
                    |r: usize| ((0..k).map(|c| out[c * n + r]).sum::<f64>() - 1.0).abs() < 1e-6;
                let constant_ok = match known {
                    KnownPropensity::Constant(values) => values.len() == k,
                    KnownPropensity::PerRow(_) => true,
                };
                if !constant_ok
                    || out.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
                    || !(0..n).all(sums_to_one)
                {
                    return Err(refuse(
                        antecedent_core::reason_code!("invalid_argument"),
                        "policy_value.invalid_known_propensity",
                        "known propensities must be one probability in (0, 1] per declared action and row, summing to one over the actions",
                    ));
                }
                Some(out)
            }
        };
        Ok(Self {
            n,
            ncols,
            design,
            category,
            declared_action,
            known_propensity,
            outcome: keep.iter().map(|&i| data.outcome[i]).collect(),
            row_index,
            rows_by_level,
            dropped,
        })
    }
}
