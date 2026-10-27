//! Cross-fitted sequential contrast of two static treatment regimes.
//!
//! Each subject is observed at two binary decisions and a terminal outcome, with no dropout.
//! The contrast is `E[Y(a0, a1)] − E[Y(b0, b1)]`. Sequential conditional exchangeability given
//! the measured history, positivity, consistency, and no interference are assumed. They are
//! not derived from a graph.
//!
//! For a regime `(a0, a1)` the score on an evaluation row is the complete-data sequential
//! augmented inverse-probability score
//!
//! ```text
//! Q0 + 1{A0 = a0} (Q1 − Q0) / π0(a0 | L0)
//!    + 1{A0 = a0, A1 = a1} (Y − Q1) / (π0(a0 | L0) π1(a1 | H1))
//! ```
//!
//! `Q1` is the least-squares regression of `Y` on the history through the second decision,
//! fit among training rows with `A1 = a1`. `Q0` is the least-squares regression of that `Q1`
//! on the baseline, fit among training rows with `A0 = a0`. Propensities are logistic.
//! Every model that enters an evaluation score is fit on the complement of that fold. The
//! nested regression's pseudo-outcome is `Q1`'s in-sample prediction on that same complement,
//! not an out-of-fold prediction of the training rows.
//!
//! A fit is stored under the certified source together with the nuisance key: target, design,
//! folds, treatment, provider, and numerical dependencies, plus the nuisance role. Lookup
//! returns that stored fit. A change to the source or to any key field does not reuse it.
//!
//! Known-randomization sequential scores are a different estimator. This module does not call
//! them and does not build a posterior. The score scale is an unmeasured diagnostic, not an
//! interval. Horizons beyond two decisions, censoring, and high-dimensional orthogonal
//! learning are outside this contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::SemanticDigest;
use antecedent_core::identity::{CanonicalWriter, ReusableProduct};

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    StatsError, fit_glm,
};
use thiserror::Error;

use crate::error::EstimationError;
use crate::nuisance_cache::{
    BlockedOutOfFoldPredictions, CertifiedSource, DesignDependency, FoldAssignment,
    NuisanceCacheError, NuisanceCacheKey, NuisanceKeyBuilder, NuisanceOptions,
    NumericalDependencies, PredictedRow, ProviderDependency, TargetDependency, TreatmentDependency,
};

impl From<EstimationError> for SequentialCrossfitError {
    /// Only the shared nuisance cache's own lock failure reaches here.
    fn from(_: EstimationError) -> Self {
        refused("the shared nuisance cache is unavailable; next attempt: fit without a cache scope")
    }
}

/// Name of the positivity floor inside [`NumericalDependencies`].
pub const MIN_PROBABILITY: &str = "min_probability";

/// Why a sequential cross-fit contrast was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SequentialCrossfitError {
    /// The certified-nuisance cache refused the fold block or the key.
    #[error(transparent)]
    Cache(#[from] NuisanceCacheError),
    /// A logistic or least-squares fit failed.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// The panel, the fold assignment, or a named dependency cannot be used.
    #[error("{message}")]
    Refused {
        /// What failed, and what to change.
        message: &'static str,
    },
    /// A training fold has no row at the action a regression subsets on.
    #[error(
        "fold {fold} has no row at action {action} for decision {decision}; \
         next attempt: keep both actions in every training fold or use fewer folds"
    )]
    MissingAction {
        /// Evaluation fold whose training complement is missing the action.
        fold: u32,
        /// `0` is the first decision and `1` is the second.
        decision: u8,
        /// Action code, `0` or `1`.
        action: u8,
    },
    /// A training subset has no more rows than columns.
    #[error(
        "fold {fold} has {rows} training rows and {columns} columns; \
         next attempt: use fewer folds or fewer covariates"
    )]
    Underdetermined {
        /// Evaluation fold.
        fold: u32,
        /// Rows in the training subset.
        rows: usize,
        /// Columns in the design, including the intercept.
        columns: usize,
    },
    /// The design of a fold is rank deficient.
    #[error("fold {fold} has a rank-deficient design; next attempt: drop a collinear covariate")]
    Rank {
        /// Evaluation fold.
        fold: u32,
    },
    /// An evaluation-row propensity is outside the declared floor.
    #[error(
        "row {row_id} at decision {decision} is outside the positivity floor; \
         next attempt: keep both actions inside the floor or enlarge the sample"
    )]
    Positivity {
        /// Evaluation row.
        row_id: u64,
        /// `0` is the first decision and `1` is the second.
        decision: u8,
    },
}

/// Binary action code.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ActionLevel {
    /// The action coded 0.
    Zero,
    /// The action coded 1.
    One,
}

impl ActionLevel {
    /// `0` or `1`.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }

    /// Numeric code used in the panel.
    #[must_use]
    pub const fn value(self) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::One => 1.0,
        }
    }
}

/// One static regime, the action at each of the two decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct StaticRegime {
    first: ActionLevel,
    second: ActionLevel,
}

impl StaticRegime {
    /// Regime that assigns `first` and then `second`.
    #[must_use]
    pub const fn new(first: ActionLevel, second: ActionLevel) -> Self {
        Self { first, second }
    }

    /// Action at the first decision.
    #[must_use]
    pub const fn first(self) -> ActionLevel {
        self.first
    }

    /// Action at the second decision.
    #[must_use]
    pub const fn second(self) -> ActionLevel {
        self.second
    }
}

/// One subject per index, observed at both decisions.
///
/// `NaN` is refused. A column shorter than the treatment is refused. Covariate order is part
/// of the design the caller binds into [`DesignDependency`].
#[derive(Clone, Copy, Debug)]
pub struct SequentialPanel<'a> {
    /// Baseline covariates, one column per slice.
    pub baseline: &'a [&'a [f64]],
    /// First-decision treatment, coded `0.0` or `1.0`.
    pub first_treatment: &'a [f64],
    /// Covariates measured after the first decision and before the second.
    pub intermediate: &'a [&'a [f64]],
    /// Second-decision treatment, coded `0.0` or `1.0`.
    pub second_treatment: &'a [f64],
    /// Terminal outcome.
    pub outcome: &'a [f64],
}

/// Dependencies that must all match before two contrasts share a fit.
///
/// The design, the actions, the outcome and the folds are derived from the panel itself by
/// [`Self::from_panel`], and [`estimate_sequential_contrast`] refuses a panel whose values
/// differ from the one the identity was derived from. There is no way to assert a design or
/// outcome identity the values do not carry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequentialIdentity {
    source: CertifiedSource,
    folds: Arc<FoldAssignment>,
    numerical: NumericalDependencies,
    panel: SemanticDigest,
    keys: [NuisanceCacheKey; ROLE_COUNT],
}

/// Roles a sequential contrast can store.
const ROLE_COUNT: usize = 8;

const fn role_index(role: SequentialRole) -> usize {
    match role {
        SequentialRole::BaselinePropensity => 0,
        SequentialRole::FollowUpPropensity => 1,
        SequentialRole::TerminalOutcome { second: ActionLevel::Zero } => 2,
        SequentialRole::TerminalOutcome { second: ActionLevel::One } => 3,
        SequentialRole::NestedOutcome { first: ActionLevel::Zero, second: ActionLevel::Zero } => 4,
        SequentialRole::NestedOutcome { first: ActionLevel::Zero, second: ActionLevel::One } => 5,
        SequentialRole::NestedOutcome { first: ActionLevel::One, second: ActionLevel::Zero } => 6,
        SequentialRole::NestedOutcome { first: ActionLevel::One, second: ActionLevel::One } => 7,
    }
}

fn every_role() -> [SequentialRole; ROLE_COUNT] {
    use ActionLevel::{One, Zero};
    [
        SequentialRole::BaselinePropensity,
        SequentialRole::FollowUpPropensity,
        SequentialRole::TerminalOutcome { second: Zero },
        SequentialRole::TerminalOutcome { second: One },
        SequentialRole::NestedOutcome { first: Zero, second: Zero },
        SequentialRole::NestedOutcome { first: Zero, second: One },
        SequentialRole::NestedOutcome { first: One, second: Zero },
        SequentialRole::NestedOutcome { first: One, second: One },
    ]
}

/// Digest of every value of a panel, in column order.
fn panel_fingerprint(panel: SequentialPanel<'_>) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.sequential.panel");
    writer.word(panel.baseline.len() as u64);
    for column in panel.baseline {
        writer.reals(column);
    }
    writer.reals(panel.first_treatment);
    writer.word(panel.intermediate.len() as u64);
    for column in panel.intermediate {
        writer.reals(column);
    }
    writer.reals(panel.second_treatment).reals(panel.outcome);
    SemanticDigest::from_bytes(writer.finish())
}

impl SequentialIdentity {
    /// Bind the dependencies and derive the design, action and outcome identities from `panel`.
    /// `numerical` must name [`MIN_PROBABILITY`] in `(0, 0.5]`. `options` are the fitting
    /// options the caller varies (the regimes' coding, a provider tuning); each role adds its
    /// own name, the column counts and the source.
    ///
    /// The cross-fit consumes no randomness: the folds are explicit, so the seed dependency
    /// is fixed at zero.
    ///
    /// # Errors
    ///
    /// [`SequentialCrossfitError::Refused`] when the floor is missing or outside that range, or
    /// the panel or the folds cannot be used; [`SequentialCrossfitError::Cache`] when a
    /// dependency does not hash.
    #[must_use = "a refused identity is not a reuse key"]
    pub fn from_panel(
        panel: SequentialPanel<'_>,
        source: CertifiedSource,
        target: TargetDependency,
        folds: FoldAssignment,
        provider: ProviderDependency,
        numerical: NumericalDependencies,
        options: &NuisanceOptions,
    ) -> Result<Self, SequentialCrossfitError> {
        let _ = floor_of(&numerical)?;
        let n = checked_panel(panel)?;
        require_panel_rows(&folds, n)?;
        let columns: Vec<&[f64]> =
            panel.baseline.iter().chain(panel.intermediate.iter()).copied().collect();
        let design = DesignDependency::from_columns(&columns)?;
        let actions = TreatmentDependency::of_values(
            "sequential_actions",
            &[panel.first_treatment, panel.second_treatment],
        )?;
        let options = options
            .clone()
            .word("baseline_columns", panel.baseline.len() as u64)
            .word("intermediate_columns", panel.intermediate.len() as u64)
            .text("source", &format!("{:?}", source.id()))
            .digest("source_content", Some(source.content()));
        let identity = folds.identity();
        let mut keys = Vec::with_capacity(ROLE_COUNT);
        for role in every_role() {
            let role_options = options.clone().text("role", &format!("{role:?}"));
            keys.push(
                NuisanceKeyBuilder::new(ReusableProduct::SequentialNuisance)
                    .target(target)
                    .design(design)
                    .rows_of_folds(&folds)
                    .folds(identity)
                    .treatment(actions.clone())
                    .outcome(panel.outcome)
                    .weights(None)
                    .provider(provider)
                    .options(&role_options)
                    .seed(0)
                    .numerical(numerical.clone())
                    .build()?,
            );
        }
        let Ok(keys) = <[NuisanceCacheKey; ROLE_COUNT]>::try_from(keys) else {
            return Err(refused("the sequential roles were not all keyed"));
        };
        Ok(Self {
            source,
            folds: Arc::new(folds),
            numerical,
            panel: panel_fingerprint(panel),
            keys,
        })
    }

    /// Certified source. A different id or content is a different fit.
    #[must_use]
    pub const fn source(&self) -> CertifiedSource {
        self.source
    }

    /// Fold assignment the contrast evaluates.
    #[must_use]
    pub fn folds(&self) -> &FoldAssignment {
        &self.folds
    }

    /// Reuse key of one nuisance role.
    #[must_use]
    pub fn fit_key(&self, role: SequentialRole) -> SequentialFitKey {
        SequentialFitKey {
            source: self.source,
            cache: self.keys[role_index(role)].clone(),
            role,
            folds: CarriedFolds(Arc::clone(&self.folds)),
        }
    }

    /// Refuse a panel whose values are not the ones this identity was derived from.
    fn require_panel(&self, panel: SequentialPanel<'_>) -> Result<(), SequentialCrossfitError> {
        if panel_fingerprint(panel) == self.panel {
            Ok(())
        } else {
            Err(refused(
                "the panel differs from the one this identity was derived from; \
                 next attempt: derive the identity from this panel with from_panel",
            ))
        }
    }
}

/// The checked fold assignment a key was built for, carried for leakage checks only. The
/// compact [`crate::nuisance_cache::FoldIdentity`] in the nuisance key is what separates two
/// fits, so this never takes part in equality or ordering.
#[derive(Clone, Debug)]
struct CarriedFolds(Arc<FoldAssignment>);

impl PartialEq for CarriedFolds {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for CarriedFolds {}

impl PartialOrd for CarriedFolds {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CarriedFolds {
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

/// Which sequential nuisance a stored fit is.
///
/// The role is not a substitute for the cache key. Two roles under one key are different
/// fits. The nested outcome carries both actions because `Q0` depends on the terminal action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SequentialRole {
    /// Logistic `P(A0 = 1 | L0)`.
    BaselinePropensity,
    /// Logistic `P(A1 = 1 | L0, A0, L1)`.
    FollowUpPropensity,
    /// Least-squares `E[Y | A1 = second, H1]`.
    TerminalOutcome {
        /// Second-decision action the regression conditions on.
        second: ActionLevel,
    },
    /// Least-squares `E[Q1 | A0 = first, L0]` for the terminal action `second`.
    NestedOutcome {
        /// First-decision action the regression conditions on.
        first: ActionLevel,
        /// Terminal action whose `Q1` is the pseudo-outcome.
        second: ActionLevel,
    },
}

/// Identity of one stored fit: certified source, nuisance key, and role.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SequentialFitKey {
    source: CertifiedSource,
    cache: NuisanceCacheKey,
    role: SequentialRole,
    folds: CarriedFolds,
}

impl SequentialFitKey {
    /// Certified source.
    #[must_use]
    pub const fn source(&self) -> CertifiedSource {
        self.source
    }

    /// Target, design, folds, treatment, provider, and numerical dependencies.
    #[must_use]
    pub const fn cache(&self) -> &NuisanceCacheKey {
        &self.cache
    }

    /// Nuisance role.
    #[must_use]
    pub const fn role(&self) -> SequentialRole {
        self.role
    }

    /// The fold assignment the fit is certified against.
    #[must_use]
    pub fn folds(&self) -> &FoldAssignment {
        &self.folds.0
    }
}

/// Coefficients of one evaluation fold, fit on that fold's complement.
#[derive(Debug)]
pub struct FoldCoefficients {
    evaluation_fold: u32,
    coefficients: Box<[f64]>,
}

impl FoldCoefficients {
    /// Finite, non-empty coefficient vector for `evaluation_fold`.
    ///
    /// # Errors
    ///
    /// [`SequentialCrossfitError::Refused`] when the vector is empty or not finite.
    #[must_use = "a refused coefficient vector is not a fold fit"]
    pub fn try_new(
        evaluation_fold: u32,
        coefficients: &[f64],
    ) -> Result<Self, SequentialCrossfitError> {
        if coefficients.is_empty() {
            return Err(refused("a fold fit has no coefficients"));
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(refused("a fold coefficient is not finite"));
        }
        Ok(Self { evaluation_fold, coefficients: coefficients.to_vec().into_boxed_slice() })
    }

    /// Fold these coefficients were held out of.
    #[must_use]
    pub const fn evaluation_fold(&self) -> u32 {
        self.evaluation_fold
    }

    /// Coefficients in design-column order.
    #[must_use]
    pub const fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn duplicate(&self) -> Self {
        Self { evaluation_fold: self.evaluation_fold, coefficients: self.coefficients.clone() }
    }
}

/// One certified sequential fit. Lookup borrows it.
///
/// The type is not [`Clone`]. Copying it out of the store is not how a fit is reused.
#[derive(Debug)]
pub struct SequentialFit {
    folds: Box<[FoldCoefficients]>,
    blocks: Box<[BlockedOutOfFoldPredictions]>,
}

impl SequentialFit {
    /// Per-fold coefficients, in evaluation-fold order.
    #[must_use]
    pub const fn folds(&self) -> &[FoldCoefficients] {
        &self.folds
    }

    /// Out-of-fold predictions, in evaluation-fold order.
    #[must_use]
    pub const fn blocks(&self) -> &[BlockedOutOfFoldPredictions] {
        &self.blocks
    }

    /// Coefficients fit on the complement of `evaluation_fold`.
    #[must_use]
    pub fn coefficients(&self, evaluation_fold: u32) -> Option<&[f64]> {
        self.folds
            .iter()
            .find(|fold| fold.evaluation_fold == evaluation_fold)
            .map(FoldCoefficients::coefficients)
    }

    /// Stored prediction for `row_id`, if this fit held that row out.
    #[must_use]
    pub fn prediction(&self, row_id: u64) -> Option<f64> {
        for block in &self.blocks {
            if let Ok(index) = block.rows().binary_search_by_key(&row_id, PredictedRow::row_id) {
                return Some(block.rows()[index].value());
            }
        }
        None
    }
}

/// Fits keyed by source, nuisance key, and role.
///
/// A hit borrows the stored fit. Certifying the same coefficients and blocks again leaves
/// that allocation in place.
#[derive(Debug)]
pub struct SequentialNuisanceStore {
    fits: BTreeMap<SequentialFitKey, Arc<SequentialFit>>,
    computed: u64,
}

impl SequentialNuisanceStore {
    /// An empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self { fits: BTreeMap::new(), computed: 0 }
    }

    /// Roles inserted. An identical re-certification does not increment this.
    #[must_use]
    pub const fn computed(&self) -> u64 {
        self.computed
    }

    /// The stored fit, borrowed. A miss is [`None`].
    #[must_use = "a cache miss is not a stored fit"]
    pub fn lookup(&self, key: &SequentialFitKey) -> Option<&SequentialFit> {
        self.fits.get(key).map(|fit| &**fit)
    }

    /// Hold a fit that was certified when it was produced, in this call or by another contrast
    /// of the cache scope. `computed` counts it as produced here.
    fn adopt(&mut self, key: SequentialFitKey, fit: Arc<SequentialFit>, computed: bool) {
        self.fits.insert(key, fit);
        if computed {
            self.computed = self.computed.saturating_add(1);
        }
    }

    /// Store out-of-fold predictions for `key`.
    ///
    /// The same coefficients and blocks may be certified again and do not replace the stored
    /// fit. A different prediction or coefficient vector for that key is refused.
    ///
    /// # Errors
    ///
    /// [`SequentialCrossfitError::Cache`] when a block leaks a training or held-out row, or
    /// when the key already stores a different fit.
    /// [`SequentialCrossfitError::Refused`] when the fold coefficients and blocks disagree.
    #[must_use = "a refused certification is not a stored fit"]
    pub fn certify(
        &mut self,
        key: SequentialFitKey,
        coefficients: &[FoldCoefficients],
        blocks: &[BlockedOutOfFoldPredictions],
    ) -> Result<(), SequentialCrossfitError> {
        let fit = assemble_fit(key.folds(), coefficients, blocks)?;
        if let Some(existing) = self.fits.get(&key) {
            if same_fit(existing, &fit) {
                return Ok(());
            }
            return Err(NuisanceCacheError::ConflictingFit.into());
        }
        self.adopt(key, Arc::new(fit), true);
        Ok(())
    }
}

impl Default for SequentialNuisanceStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Point contrast and the unmeasured scale of its cross-fit scores.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialContrast {
    /// Active-regime mean minus control-regime mean.
    pub contrast: f64,
    /// Cross-fit mean of the active regime.
    pub active_mean: f64,
    /// Cross-fit mean of the control regime.
    pub control_mean: f64,
    /// Standard deviation of the contrast scores over `sqrt(n)`.
    ///
    /// Calibration is unmeasured. This is not an interval endpoint.
    pub unmeasured_score_scale: f64,
    /// Subjects scored. Held-out ids in the fold assignment are not subjects.
    pub evaluated_rows: u64,
    /// Smallest regime-action probability used in a score.
    pub minimum_action_probability: f64,
    /// Cross-fit score of the active regime, one per subject in panel order.
    pub active_scores: Vec<f64>,
    /// Cross-fit score of the control regime, one per subject in panel order.
    pub control_scores: Vec<f64>,
    /// Active minus control score, one per subject in panel order.
    pub contrast_scores: Vec<f64>,
    /// Nuisance roles fitted during this call.
    pub fits_computed: u64,
    /// Nuisance roles borrowed from the store during this call.
    pub fits_reused: u64,
}

/// Estimate the cross-fit contrast of `active` minus `control`.
///
/// A role already stored under the identity and that role is borrowed. Its coefficients and
/// predictions are not cloned into a new fit.
///
/// # Errors
///
/// Panel, fold, rank, missing-action, positivity, and cache errors. A refusal may leave
/// earlier roles of this call in `store`.
#[must_use = "a refused contrast is not an estimate"]
pub fn estimate_sequential_contrast(
    panel: SequentialPanel<'_>,
    identity: &SequentialIdentity,
    active: StaticRegime,
    control: StaticRegime,
    store: &mut SequentialNuisanceStore,
) -> Result<SequentialContrast, SequentialCrossfitError> {
    let n = checked_panel(panel)?;
    identity.require_panel(panel)?;
    require_panel_rows(identity.folds(), n)?;
    let floor = floor_of(&identity.numerical)?;
    let mut workspace = LeastSquaresWorkspace::default();
    let mut fits_reused = 0u64;
    let computed_before = store.computed();
    let fit_bytes = (n as u64).saturating_mul(32).saturating_add(4096);
    for role in contrast_roles(active, control) {
        let key = identity.fit_key(role);
        if store.lookup(&key).is_some() {
            fits_reused = fits_reused.saturating_add(1);
            continue;
        }
        // A contrast of an active cache scope borrows a fit another contrast certified under
        // the same key; the certified leakage checks ran when it was produced, against folds of
        // the same identity.
        let produced_here = std::cell::Cell::new(false);
        let fit = crate::crossfit_cache::reuse_in_active_scope(
            || Some(key.cache().clone()),
            fit_bytes,
            &|| Ok(()),
            || {
                produced_here.set(true);
                let produced = produce_role(panel, identity, floor, role, store, &mut workspace)?;
                assemble_fit(identity.folds(), &produced.coefficients, &produced.blocks)
            },
        )?;
        if !produced_here.get() {
            fits_reused = fits_reused.saturating_add(1);
        }
        store.adopt(key, fit, produced_here.get());
    }
    let active_scores = score_regime(store, identity, panel, active, floor)?;
    let control_scores = score_regime(store, identity, panel, control, floor)?;
    let contrast_scores = contrast_scores(&active_scores.values, &control_scores.values)?;
    let evaluated_rows =
        u64::try_from(n).map_err(|_| refused("the panel does not fit this count"))?;
    Ok(SequentialContrast {
        contrast: mean(&contrast_scores)?,
        active_mean: active_scores.mean,
        control_mean: control_scores.mean,
        unmeasured_score_scale: score_scale(&contrast_scores)?,
        evaluated_rows,
        minimum_action_probability: active_scores.minimum.min(control_scores.minimum),
        active_scores: active_scores.values,
        control_scores: control_scores.values,
        contrast_scores,
        fits_computed: store.computed().saturating_sub(computed_before),
        fits_reused,
    })
}

/// Which regression a fold is fitting.
#[derive(Clone, Copy)]
enum FitFamily {
    Logistic,
    LeastSquares,
}

/// Columns beyond the intercept.
#[derive(Clone, Copy)]
enum FeatureSet {
    /// Intercept and baseline covariates.
    Baseline,
    /// Intercept, baseline, first treatment, and intermediate covariates.
    History,
}

/// How training rows are chosen.
#[derive(Clone, Copy)]
enum Subset {
    All,
    First(ActionLevel),
    Second(ActionLevel),
}

struct Produced {
    coefficients: Vec<FoldCoefficients>,
    blocks: Vec<BlockedOutOfFoldPredictions>,
}

struct RegimeScores {
    mean: f64,
    minimum: f64,
    values: Vec<f64>,
}

fn refused(message: &'static str) -> SequentialCrossfitError {
    SequentialCrossfitError::Refused { message }
}

fn floor_of(numerical: &NumericalDependencies) -> Result<f64, SequentialCrossfitError> {
    let Some(found) = numerical.dependencies().iter().find(|item| item.name() == MIN_PROBABILITY)
    else {
        return Err(refused(
            "min_probability is not a numerical dependency; next attempt: name that floor",
        ));
    };
    let floor = found.value();
    if floor.is_finite() && floor > 0.0 && floor <= 0.5 {
        Ok(floor)
    } else {
        Err(refused("min_probability must be in (0, 0.5]"))
    }
}

fn is_action(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == 1.0_f64.to_bits()
}

fn is_level(value: f64, level: ActionLevel) -> bool {
    value.to_bits() == level.value().to_bits()
}

fn checked_panel(panel: SequentialPanel<'_>) -> Result<usize, SequentialCrossfitError> {
    let n = panel.first_treatment.len();
    if n < 2 {
        return Err(refused(
            "the contrast needs at least two subjects; next attempt: pass a larger panel",
        ));
    }
    if panel.second_treatment.len() != n || panel.outcome.len() != n {
        return Err(refused(
            "panel columns differ in length; next attempt: pass one value per subject",
        ));
    }
    for column in panel.baseline.iter().chain(panel.intermediate.iter()) {
        if column.len() != n {
            return Err(refused(
                "panel columns differ in length; next attempt: pass one value per subject",
            ));
        }
        if column.iter().any(|value| !value.is_finite()) {
            return Err(refused("a covariate is not finite; next attempt: pass a finite column"));
        }
    }
    if panel.outcome.iter().any(|value| !value.is_finite()) {
        return Err(refused("an outcome is not finite; next attempt: pass a finite outcome"));
    }
    let mut treatments = panel.first_treatment.iter().chain(panel.second_treatment);
    if treatments.any(|value| !is_action(*value)) {
        return Err(refused(
            "a treatment is not 0 or 1; next attempt: code both decisions as binary actions",
        ));
    }
    Ok(n)
}

fn require_panel_rows(folds: &FoldAssignment, n: usize) -> Result<(), SequentialCrossfitError> {
    if folds.rows().len() != n {
        return Err(refused(
            "fold assignment does not cover the panel; next attempt: assign each subject once",
        ));
    }
    for (offset, row) in folds.rows().iter().enumerate() {
        let Ok(expected) = u64::try_from(offset) else {
            return Err(refused("the panel does not fit this count"));
        };
        if row.row_id() != expected {
            return Err(refused(
                "fold assignment does not cover the panel; next attempt: assign subjects 0..n",
            ));
        }
    }
    Ok(())
}

fn contrast_roles(active: StaticRegime, control: StaticRegime) -> Vec<SequentialRole> {
    let mut roles = vec![SequentialRole::BaselinePropensity, SequentialRole::FollowUpPropensity];
    push_role(&mut roles, SequentialRole::TerminalOutcome { second: active.second });
    push_role(&mut roles, SequentialRole::TerminalOutcome { second: control.second });
    push_role(
        &mut roles,
        SequentialRole::NestedOutcome { first: active.first, second: active.second },
    );
    push_role(
        &mut roles,
        SequentialRole::NestedOutcome { first: control.first, second: control.second },
    );
    roles
}

fn push_role(roles: &mut Vec<SequentialRole>, role: SequentialRole) {
    if !roles.contains(&role) {
        roles.push(role);
    }
}

fn produce_role(
    panel: SequentialPanel<'_>,
    identity: &SequentialIdentity,
    floor: f64,
    role: SequentialRole,
    store: &SequentialNuisanceStore,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<Produced, SequentialCrossfitError> {
    let (family, features, subset, decision, second) = match role {
        SequentialRole::BaselinePropensity => {
            (FitFamily::Logistic, FeatureSet::Baseline, Subset::All, 0, None)
        }
        SequentialRole::FollowUpPropensity => {
            (FitFamily::Logistic, FeatureSet::History, Subset::All, 1, None)
        }
        SequentialRole::TerminalOutcome { second } => {
            (FitFamily::LeastSquares, FeatureSet::History, Subset::Second(second), 1, None)
        }
        SequentialRole::NestedOutcome { first, second } => {
            (FitFamily::LeastSquares, FeatureSet::Baseline, Subset::First(first), 0, Some(second))
        }
    };
    let terminal =
        second.map(|action| terminal_coefficients(store, identity, action)).transpose()?;
    let request = ProduceRequest {
        panel,
        folds: identity.folds(),
        floor,
        family,
        features,
        subset,
        decision,
        terminal: terminal.as_deref(),
    };
    produce(&request, workspace)
}

/// Copy the terminal coefficients stored under this identity.
///
/// The copy is the numeric vector the nested regression regresses. It is not a second fit,
/// and a terminal fit stored for another source or key is not readable here.
fn terminal_coefficients(
    store: &SequentialNuisanceStore,
    identity: &SequentialIdentity,
    second: ActionLevel,
) -> Result<Vec<(u32, Vec<f64>)>, SequentialCrossfitError> {
    let key = identity.fit_key(SequentialRole::TerminalOutcome { second });
    let Some(found) = store.lookup(&key) else {
        return Err(refused(
            "the terminal outcome fit is missing; next attempt: fit that action before Q0",
        ));
    };
    Ok(found
        .folds()
        .iter()
        .map(|fold| (fold.evaluation_fold(), fold.coefficients().to_vec()))
        .collect())
}

struct ProduceRequest<'a> {
    panel: SequentialPanel<'a>,
    folds: &'a FoldAssignment,
    floor: f64,
    family: FitFamily,
    features: FeatureSet,
    subset: Subset,
    decision: u8,
    terminal: Option<&'a [(u32, Vec<f64>)]>,
}

fn produce(
    request: &ProduceRequest<'_>,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<Produced, SequentialCrossfitError> {
    let mut coefficients = Vec::new();
    let mut blocks = Vec::new();
    // The constant column and the fit buffers are shared by every evaluation fold.
    let intercept = vec![1.0; request.panel.first_treatment.len()];
    let mut scratch = FitScratch::default();
    for evaluation_fold in used_folds(request.folds) {
        let (coefficient, block) =
            fit_evaluation(request, evaluation_fold, &intercept, workspace, &mut scratch)?;
        coefficients.push(coefficient);
        blocks.push(block);
    }
    coefficients.sort_by_key(FoldCoefficients::evaluation_fold);
    blocks.sort_by_key(BlockedOutOfFoldPredictions::evaluation_fold);
    Ok(Produced { coefficients, blocks })
}

fn used_folds(folds: &FoldAssignment) -> Vec<u32> {
    let mut seen = Vec::new();
    for row in folds.rows() {
        if !seen.contains(&row.fold()) {
            seen.push(row.fold());
        }
    }
    seen.sort_unstable();
    seen
}

fn row_indexes(
    folds: &FoldAssignment,
    evaluation: u32,
    training: bool,
) -> Result<Vec<usize>, SequentialCrossfitError> {
    let mut rows = Vec::new();
    for row in folds.rows() {
        let in_fold = row.fold() == evaluation;
        if training == in_fold {
            continue;
        }
        rows.push(panel_index(row.row_id())?);
    }
    Ok(rows)
}

fn panel_index(row_id: u64) -> Result<usize, SequentialCrossfitError> {
    usize::try_from(row_id).map_err(|_| refused("a row id does not fit this architecture"))
}

fn fit_evaluation(
    request: &ProduceRequest<'_>,
    evaluation_fold: u32,
    intercept: &[f64],
    workspace: &mut LeastSquaresWorkspace,
    scratch: &mut FitScratch,
) -> Result<(FoldCoefficients, BlockedOutOfFoldPredictions), SequentialCrossfitError> {
    let training = row_indexes(request.folds, evaluation_fold, true)?;
    let evaluation_rows = row_indexes(request.folds, evaluation_fold, false)?;
    if training.is_empty() {
        return Err(NuisanceCacheError::EmptyTraining { fold: evaluation_fold }.into());
    }
    if evaluation_rows.is_empty() {
        return Err(refused("an evaluation fold has no rows; next attempt: assign that fold"));
    }
    let outcome = if let Some(terminal) = request.terminal {
        std::borrow::Cow::Owned(pseudo_outcomes(
            request.panel,
            intercept,
            evaluation_fold,
            terminal,
        )?)
    } else {
        std::borrow::Cow::Borrowed(outcome_column(request.panel, request.family, request.features))
    };
    let columns = feature_columns(request.features, request.panel, intercept);
    let subset = subset_rows(request.panel, &training, request.subset, evaluation_fold)?;
    if subset.len() <= columns.len() {
        return Err(SequentialCrossfitError::Underdetermined {
            fold: evaluation_fold,
            rows: subset.len(),
            columns: columns.len(),
        });
    }
    if matches!(request.family, FitFamily::Logistic) {
        require_both_actions(&outcome, &subset, evaluation_fold, request.decision)?;
    }
    let coefficients = fit_coefficients(
        request.family,
        evaluation_fold,
        &columns,
        &subset,
        &outcome,
        workspace,
        scratch,
    )?;
    let mut predicted = Vec::with_capacity(evaluation_rows.len());
    for row in &evaluation_rows {
        let eta = linear_predictor(&coefficients, &columns, *row)?;
        let value = match request.family {
            FitFamily::Logistic => GlmFamily::BinomialLogit.mean_from_eta(eta),
            FitFamily::LeastSquares => eta,
        };
        if !value.is_finite() {
            return Err(refused("a fold prediction is not finite"));
        }
        if matches!(request.family, FitFamily::Logistic) {
            require_floor(value, request.floor, row_id_of(*row)?, request.decision)?;
        }
        predicted.push(PredictedRow::try_new(row_id_of(*row)?, value)?);
    }
    let block = BlockedOutOfFoldPredictions::try_new(evaluation_fold, &predicted)?;
    Ok((FoldCoefficients::try_new(evaluation_fold, &coefficients)?, block))
}

fn outcome_column(panel: SequentialPanel<'_>, family: FitFamily, features: FeatureSet) -> &[f64] {
    match (family, features) {
        (FitFamily::Logistic, FeatureSet::Baseline) => panel.first_treatment,
        (FitFamily::Logistic, FeatureSet::History) => panel.second_treatment,
        (FitFamily::LeastSquares, _) => panel.outcome,
    }
}

fn pseudo_outcomes(
    panel: SequentialPanel<'_>,
    intercept: &[f64],
    evaluation_fold: u32,
    terminal: &[(u32, Vec<f64>)],
) -> Result<Vec<f64>, SequentialCrossfitError> {
    let Some((_, coefficients)) = terminal.iter().find(|(fold, _)| *fold == evaluation_fold) else {
        return Err(refused(
            "the terminal outcome fit is missing for a fold; next attempt: fit that fold first",
        ));
    };
    let columns = feature_columns(FeatureSet::History, panel, intercept);
    let mut values = Vec::with_capacity(panel.outcome.len());
    for row in 0..panel.outcome.len() {
        values.push(linear_predictor(coefficients, &columns, row)?);
    }
    Ok(values)
}

fn feature_columns<'a>(
    features: FeatureSet,
    panel: SequentialPanel<'a>,
    intercept: &'a [f64],
) -> Vec<&'a [f64]> {
    let mut columns = Vec::with_capacity(1 + panel.baseline.len() + panel.intermediate.len() + 1);
    columns.push(intercept);
    columns.extend(panel.baseline.iter().copied());
    if matches!(features, FeatureSet::History) {
        columns.push(panel.first_treatment);
        columns.extend(panel.intermediate.iter().copied());
    }
    columns
}

/// The training rows the fit reads: all of them (borrowed) or those at one action level.
fn subset_rows<'t>(
    panel: SequentialPanel<'_>,
    training: &'t [usize],
    subset: Subset,
    evaluation_fold: u32,
) -> Result<std::borrow::Cow<'t, [usize]>, SequentialCrossfitError> {
    let (treatment, action, decision) = match subset {
        Subset::All => return Ok(std::borrow::Cow::Borrowed(training)),
        Subset::First(action) => (panel.first_treatment, action, 0),
        Subset::Second(action) => (panel.second_treatment, action, 1),
    };
    let rows: Vec<usize> =
        training.iter().copied().filter(|&row| is_level(treatment[row], action)).collect();
    if rows.is_empty() {
        return Err(SequentialCrossfitError::MissingAction {
            fold: evaluation_fold,
            decision,
            action: action.code(),
        });
    }
    Ok(std::borrow::Cow::Owned(rows))
}

fn require_both_actions(
    treatment: &[f64],
    rows: &[usize],
    fold: u32,
    decision: u8,
) -> Result<(), SequentialCrossfitError> {
    let mut zero = false;
    let mut one = false;
    for &row in rows {
        if treatment[row].to_bits() == 0.0_f64.to_bits() {
            zero = true;
        }
        if treatment[row].to_bits() == 1.0_f64.to_bits() {
            one = true;
        }
    }
    if !zero {
        return Err(SequentialCrossfitError::MissingAction { fold, decision, action: 0 });
    }
    if !one {
        return Err(SequentialCrossfitError::MissingAction { fold, decision, action: 1 });
    }
    Ok(())
}

fn require_floor(
    probability: f64,
    floor: f64,
    row_id: u64,
    decision: u8,
) -> Result<(), SequentialCrossfitError> {
    if probability < floor || probability > 1.0 - floor {
        Err(SequentialCrossfitError::Positivity { row_id, decision })
    } else {
        Ok(())
    }
}

fn row_id_of(row: usize) -> Result<u64, SequentialCrossfitError> {
    u64::try_from(row).map_err(|_| refused("a row index does not fit this count"))
}

/// Training-design buffers reused by every fold fit of one role.
#[derive(Default)]
struct FitScratch {
    matrix: Vec<f64>,
    y: Vec<f64>,
}

fn fit_coefficients(
    family: FitFamily,
    fold: u32,
    columns: &[&[f64]],
    rows: &[usize],
    outcome: &[f64],
    workspace: &mut LeastSquaresWorkspace,
    scratch: &mut FitScratch,
) -> Result<Vec<f64>, SequentialCrossfitError> {
    column_major_into(&mut scratch.matrix, columns, rows);
    crate::util::gather_into(&mut scratch.y, outcome, rows);
    let (matrix, y) = (&scratch.matrix, &scratch.y);
    let nrows = rows.len();
    let ncols = columns.len();
    let backend = FaerBackend;
    let coefficients = match family {
        FitFamily::Logistic => {
            let fit = fit_glm(
                GlmFamily::BinomialLogit,
                GlmDesignRef { x_colmajor: matrix, nrows, ncols, y },
                &backend,
                workspace,
                &GlmOptions::new(50, 1e-8).without_separation_ridge(),
            )?;
            fit.require_ok()?;
            fit.coefficients
        }
        FitFamily::LeastSquares => {
            let fit = backend.least_squares(matrix, nrows, ncols, y, workspace)?;
            if fit.rank != ncols {
                return Err(SequentialCrossfitError::Rank { fold });
            }
            fit.coefficients
        }
    };
    if coefficients.len() != ncols || coefficients.iter().any(|value| !value.is_finite()) {
        return Err(refused("a fold coefficient is not finite"));
    }
    Ok(coefficients)
}

/// The `rows` of `columns` as one column-major matrix, written into the reused buffer.
fn column_major_into(matrix: &mut Vec<f64>, columns: &[&[f64]], rows: &[usize]) {
    let nrows = rows.len();
    matrix.clear();
    matrix.resize(nrows.saturating_mul(columns.len()), 0.0);
    if nrows == 0 {
        return;
    }
    for (col, column) in columns.iter().enumerate() {
        for (out, &row) in rows.iter().enumerate() {
            matrix[col * nrows + out] = column[row];
        }
    }
}

fn linear_predictor(
    coefficients: &[f64],
    columns: &[&[f64]],
    row: usize,
) -> Result<f64, SequentialCrossfitError> {
    if coefficients.len() != columns.len() {
        return Err(refused(
            "a stored coefficient width does not match the design; next attempt: refit that role",
        ));
    }
    let mut eta = 0.0;
    for (weight, column) in coefficients.iter().zip(columns) {
        let Some(&value) = column.get(row) else {
            return Err(refused("a design column is shorter than the panel"));
        };
        eta += weight * value;
    }
    if eta.is_finite() { Ok(eta) } else { Err(refused("a linear predictor is not finite")) }
}

fn assemble_fit(
    folds: &FoldAssignment,
    coefficients: &[FoldCoefficients],
    blocks: &[BlockedOutOfFoldPredictions],
) -> Result<SequentialFit, SequentialCrossfitError> {
    if coefficients.is_empty() || blocks.is_empty() {
        return Err(NuisanceCacheError::EmptyBlocks.into());
    }
    let width = coefficients[0].coefficients().len();
    if coefficients.iter().any(|fold| fold.coefficients().len() != width) {
        return Err(refused("fold coefficients differ in width"));
    }
    if coefficients.len() != blocks.len() {
        return Err(refused("coefficient folds and prediction blocks differ"));
    }
    let mut fold_ids = Vec::with_capacity(blocks.len());
    for (coefficient, block) in coefficients.iter().zip(blocks) {
        if coefficient.evaluation_fold() != block.evaluation_fold() {
            return Err(refused("coefficient folds and prediction blocks differ"));
        }
        if fold_ids.contains(&block.evaluation_fold()) {
            return Err(NuisanceCacheError::DuplicateFold { fold: block.evaluation_fold() }.into());
        }
        fold_ids.push(block.evaluation_fold());
    }
    let mut seen = vec![false; folds.rows().len()];
    for block in blocks {
        if block.evaluation_fold() >= folds.fold_count() {
            return Err(NuisanceCacheError::FoldOutOfRange {
                fold: block.evaluation_fold(),
                fold_count: folds.fold_count(),
            }
            .into());
        }
        for row in block.rows() {
            mark_prediction(folds, block.evaluation_fold(), row.row_id(), &mut seen)?;
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(refused(
            "a fold block is missing an evaluation row; next attempt: predict every held-out row",
        ));
    }
    let mut folds_stored: Vec<FoldCoefficients> =
        coefficients.iter().map(FoldCoefficients::duplicate).collect();
    let mut blocks_stored = blocks.to_vec();
    folds_stored.sort_by_key(FoldCoefficients::evaluation_fold);
    blocks_stored.sort_by_key(BlockedOutOfFoldPredictions::evaluation_fold);
    Ok(SequentialFit {
        folds: folds_stored.into_boxed_slice(),
        blocks: blocks_stored.into_boxed_slice(),
    })
}

fn same_fit(existing: &SequentialFit, candidate: &SequentialFit) -> bool {
    if existing.blocks != candidate.blocks || existing.folds.len() != candidate.folds.len() {
        return false;
    }
    existing.folds.iter().zip(candidate.folds.iter()).all(|(left, right)| {
        left.evaluation_fold == right.evaluation_fold
            && same_bits(left.coefficients(), right.coefficients())
    })
}

fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| left.to_bits() == right.to_bits())
}

fn mark_prediction(
    folds: &FoldAssignment,
    evaluation_fold: u32,
    row_id: u64,
    seen: &mut [bool],
) -> Result<(), SequentialCrossfitError> {
    if folds.is_held_out(row_id) {
        return Err(NuisanceCacheError::HeldOutLeakage { row_id }.into());
    }
    if folds.is_training_row(evaluation_fold, row_id) {
        return Err(NuisanceCacheError::TrainingLeakage { fold: evaluation_fold, row_id }.into());
    }
    let Some(assigned) = folds.fold_of(row_id) else {
        return Err(NuisanceCacheError::RowNotInFold { fold: evaluation_fold, row_id }.into());
    };
    if assigned != evaluation_fold {
        return Err(NuisanceCacheError::RowNotInFold { fold: evaluation_fold, row_id }.into());
    }
    let Some(index) = folds.rows().iter().position(|row| row.row_id() == row_id) else {
        return Err(NuisanceCacheError::RowNotInFold { fold: evaluation_fold, row_id }.into());
    };
    if seen[index] {
        return Err(NuisanceCacheError::DuplicateRow { row_id }.into());
    }
    seen[index] = true;
    Ok(())
}

fn score_regime(
    store: &SequentialNuisanceStore,
    identity: &SequentialIdentity,
    panel: SequentialPanel<'_>,
    regime: StaticRegime,
    floor: f64,
) -> Result<RegimeScores, SequentialCrossfitError> {
    let baseline = restored(store, &identity.fit_key(SequentialRole::BaselinePropensity), panel)?;
    let follow = restored(store, &identity.fit_key(SequentialRole::FollowUpPropensity), panel)?;
    let terminal = restored(
        store,
        &identity.fit_key(SequentialRole::TerminalOutcome { second: regime.second }),
        panel,
    )?;
    let nested = restored(
        store,
        &identity
            .fit_key(SequentialRole::NestedOutcome { first: regime.first, second: regime.second }),
        panel,
    )?;
    let mut values = Vec::with_capacity(panel.outcome.len());
    let mut minimum = 1.0_f64;
    for row in 0..panel.outcome.len() {
        let row_id = row_id_of(row)?;
        let first_probability = action_probability(baseline[row], regime.first);
        let second_probability = action_probability(follow[row], regime.second);
        require_floor(first_probability, floor, row_id, 0)?;
        require_floor(second_probability, floor, row_id, 1)?;
        minimum = minimum.min(first_probability).min(second_probability);
        values.push(augmented_score(
            nested[row],
            terminal[row],
            panel.outcome[row],
            is_level(panel.first_treatment[row], regime.first),
            is_level(panel.second_treatment[row], regime.second),
            first_probability,
            second_probability,
        )?);
    }
    let mean = mean(&values)?;
    Ok(RegimeScores { mean, minimum, values })
}

fn restored(
    store: &SequentialNuisanceStore,
    key: &SequentialFitKey,
    panel: SequentialPanel<'_>,
) -> Result<Vec<f64>, SequentialCrossfitError> {
    let Some(fit) = store.lookup(key) else {
        return Err(refused("a nuisance role is not stored; next attempt: fit that role"));
    };
    let mut values = Vec::with_capacity(panel.outcome.len());
    for row in 0..panel.outcome.len() {
        let Some(value) = fit.prediction(row_id_of(row)?) else {
            return Err(refused(
                "an evaluation row has no stored prediction; next attempt: predict that row",
            ));
        };
        values.push(value);
    }
    Ok(values)
}

fn action_probability(probability_of_one: f64, action: ActionLevel) -> f64 {
    match action {
        ActionLevel::One => probability_of_one,
        ActionLevel::Zero => 1.0 - probability_of_one,
    }
}

fn augmented_score(
    q0: f64,
    q1: f64,
    outcome: f64,
    first_matches: bool,
    second_matches: bool,
    first_probability: f64,
    second_probability: f64,
) -> Result<f64, SequentialCrossfitError> {
    let mut score = q0;
    if first_matches {
        score += (q1 - q0) / first_probability;
        if second_matches {
            score += (outcome - q1) / (first_probability * second_probability);
        }
    }
    if score.is_finite() {
        Ok(score)
    } else {
        Err(refused("a sequential score is not finite; next attempt: raise the positivity floor"))
    }
}

fn contrast_scores(active: &[f64], control: &[f64]) -> Result<Vec<f64>, SequentialCrossfitError> {
    if active.len() != control.len() {
        return Err(refused("regime scores differ in length"));
    }
    let mut scores = Vec::with_capacity(active.len());
    for (active_score, control_score) in active.iter().zip(control) {
        let score = active_score - control_score;
        if !score.is_finite() {
            return Err(refused("a contrast score is not finite"));
        }
        scores.push(score);
    }
    Ok(scores)
}

fn mean(values: &[f64]) -> Result<f64, SequentialCrossfitError> {
    if values.is_empty() {
        return Err(refused("there is no score to average"));
    }
    let value = crate::util::mean(values);
    if value.is_finite() { Ok(value) } else { Err(refused("a regime mean is not finite")) }
}

fn score_scale(scores: &[f64]) -> Result<f64, SequentialCrossfitError> {
    if scores.len() < 2 {
        return Err(refused("the score scale needs at least two subjects"));
    }
    let center = mean(scores)?;
    let mut sum_of_squares = 0.0;
    for score in scores {
        let delta = score - center;
        sum_of_squares += delta * delta;
    }
    let width =
        u64::try_from(scores.len()).map_err(|_| refused("the panel does not fit this count"))?;
    let Some(denom) = width.checked_mul(width - 1) else {
        return Err(refused("the score scale overflowed"));
    };
    let variance = sum_of_squares / denom as f64;
    if variance < 0.0 || !variance.is_finite() {
        return Err(refused("the score scale is not finite"));
    }
    Ok(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{
        ExecutionContext, NumericalPolicy, ProviderId, SemanticDigest, SourceDependencyId,
    };

    use crate::crossfit_cache::CrossfitNuisanceCache;
    use crate::nuisance_cache::{FoldRow, NumericalDependency};

    use super::{
        ActionLevel, BlockedOutOfFoldPredictions, CertifiedSource, FoldAssignment,
        FoldCoefficients, MIN_PROBABILITY, NuisanceCacheError, NuisanceOptions,
        NumericalDependencies, PredictedRow, ProviderDependency, SequentialContrast,
        SequentialCrossfitError, SequentialIdentity, SequentialNuisanceStore, SequentialPanel,
        SequentialRole, StaticRegime, TargetDependency, estimate_sequential_contrast,
    };

    const SUBJECTS: usize = 16;

    struct Marks {
        target: u8,
        treatment: &'static str,
        provider: u64,
        floor: f64,
        source: u8,
        swap_folds: bool,
    }

    impl Marks {
        fn balanced() -> Self {
            Self {
                target: 1,
                treatment: "regimes",
                provider: 4,
                floor: 0.05,
                source: 3,
                swap_folds: false,
            }
        }

        fn identity(&self) -> SequentialIdentity {
            let (first, second, outcome) = balanced_columns();
            self.identity_for(SequentialPanel {
                baseline: &[],
                first_treatment: &first,
                intermediate: &[],
                second_treatment: &second,
                outcome: &outcome,
            })
        }

        /// The identity derived from the values of `panel`.
        fn identity_for(&self, panel: SequentialPanel<'_>) -> SequentialIdentity {
            let floor = NumericalDependency::try_new(MIN_PROBABILITY, self.floor).expect("floor");
            SequentialIdentity::from_panel(
                panel,
                CertifiedSource::new(SourceDependencyId::from_raw(11), digest(self.source)),
                TargetDependency::new(digest(self.target)),
                folds(self.swap_folds),
                ProviderDependency::new(ProviderId::from_raw(self.provider), digest(6)),
                NumericalDependencies::try_new(NumericalPolicy::strict(), &[floor])
                    .expect("numerical"),
                &NuisanceOptions::new("sequential").text("treatment", self.treatment),
            )
            .expect("identity")
        }
    }

    fn digest(mark: u8) -> SemanticDigest {
        let mut bytes = [0_u8; 32];
        bytes[31] = mark;
        SemanticDigest::from_bytes(bytes)
    }

    fn folds(swap: bool) -> FoldAssignment {
        let rows: Vec<FoldRow> = (0..SUBJECTS)
            .map(|row| {
                let mut fold = u32::from(row >= SUBJECTS / 2);
                if swap && (row == 0 || row == SUBJECTS / 2) {
                    fold = 1 - fold;
                }
                FoldRow::new(u64::try_from(row).expect("row"), fold)
            })
            .collect();
        FoldAssignment::try_new(2, &rows, &[]).expect("folds")
    }

    fn balanced_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut first = Vec::with_capacity(SUBJECTS);
        let mut second = Vec::with_capacity(SUBJECTS);
        let mut outcome = Vec::with_capacity(SUBJECTS);
        for row in 0..SUBJECTS {
            let offset = row % 8;
            let a0 = f64::from(u8::from(offset >= 4));
            let a1 = f64::from(u8::from(offset % 4 >= 2));
            first.push(a0);
            second.push(a1);
            outcome.push(3.0 + a0 + 2.0 * a1);
        }
        (first, second, outcome)
    }

    fn close(actual: f64, expected: f64) {
        let gap = (actual - expected).abs();
        assert!(gap < 1.0e-6, "actual {actual} expected {expected}");
    }

    fn run(
        identity: &SequentialIdentity,
        store: &mut SequentialNuisanceStore,
        active: StaticRegime,
        control: StaticRegime,
    ) -> SequentialContrast {
        let (first, second, outcome) = balanced_columns();
        let panel = SequentialPanel {
            baseline: &[],
            first_treatment: &first,
            intermediate: &[],
            second_treatment: &second,
            outcome: &outcome,
        };
        estimate_sequential_contrast(panel, identity, active, control, store).expect("contrast")
    }

    fn treated() -> StaticRegime {
        StaticRegime::new(ActionLevel::One, ActionLevel::One)
    }

    fn untreated() -> StaticRegime {
        StaticRegime::new(ActionLevel::Zero, ActionLevel::Zero)
    }

    fn assert_resplit(second: &SequentialIdentity) {
        let first = Marks::balanced().identity();
        assert_ne!(
            first.fit_key(SequentialRole::BaselinePropensity),
            second.fit_key(SequentialRole::BaselinePropensity)
        );
        let mut store = SequentialNuisanceStore::new();
        let original = run(&first, &mut store, treated(), untreated());
        close(original.contrast, 3.0);
        let computed = store.computed();
        let rerun = run(second, &mut store, treated(), untreated());
        close(rerun.contrast, 3.0);
        assert_eq!(store.computed(), computed.saturating_mul(2));
        assert_eq!(rerun.fits_computed, computed);
        let original_fit = store.lookup(&first.fit_key(SequentialRole::BaselinePropensity));
        let other_fit = store.lookup(&second.fit_key(SequentialRole::BaselinePropensity));
        assert!(!std::ptr::eq(original_fit.expect("original"), other_fit.expect("other")));
    }

    #[test]
    fn balanced_two_decision_contrast_matches_the_linear_regime_gap() {
        let identity = Marks::balanced().identity();
        let mut store = SequentialNuisanceStore::new();
        let contrast = run(&identity, &mut store, treated(), untreated());
        close(contrast.active_mean, 6.0);
        close(contrast.control_mean, 3.0);
        close(contrast.contrast, 3.0);
        close(contrast.minimum_action_probability, 0.5);
        assert!(contrast.unmeasured_score_scale < 1.0e-5);
        assert_eq!(contrast.evaluated_rows, u64::try_from(SUBJECTS).expect("count"));
        assert_eq!(contrast.fits_computed, 6);
        assert_eq!(contrast.fits_reused, 0);
    }

    #[test]
    fn second_call_reuses_the_stored_fit() {
        let identity = Marks::balanced().identity();
        let mut store = SequentialNuisanceStore::new();
        let first = run(&identity, &mut store, treated(), untreated());
        assert_eq!(first.fits_computed, 6);
        let second = run(&identity, &mut store, treated(), untreated());
        assert_eq!(second.fits_computed, 0);
        assert_eq!(second.fits_reused, 6);
        assert_eq!(store.computed(), 6);
        close(second.contrast, 3.0);
    }

    #[test]
    fn lookup_returns_the_stored_fit_in_place() {
        let identity = Marks::balanced().identity();
        let mut store = SequentialNuisanceStore::new();
        let _ = run(&identity, &mut store, treated(), untreated());
        let key = identity.fit_key(SequentialRole::BaselinePropensity);
        let first = std::ptr::from_ref(store.lookup(&key).expect("fit"));
        let second = std::ptr::from_ref(store.lookup(&key).expect("fit"));
        assert_eq!(first, second);
        let computed = store.computed();
        let _ = run(&identity, &mut store, treated(), untreated());
        assert_eq!(store.computed(), computed);
        assert_eq!(std::ptr::from_ref(store.lookup(&key).expect("fit")), first);
    }

    #[test]
    fn changed_target_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.target = 9;
        assert_resplit(&marks.identity());
    }

    #[test]
    fn changed_design_does_not_share_a_fit() {
        let covariate: Vec<f64> =
            (0..SUBJECTS).map(|row| f64::from(u8::try_from(row).expect("row"))).collect();
        let (first, second, outcome) = balanced_columns();
        let baseline: [&[f64]; 1] = [&covariate];
        let widened = Marks::balanced().identity_for(SequentialPanel {
            baseline: &baseline,
            first_treatment: &first,
            intermediate: &[],
            second_treatment: &second,
            outcome: &outcome,
        });
        let original = Marks::balanced().identity();
        let role = SequentialRole::BaselinePropensity;
        assert_ne!(original.fit_key(role), widened.fit_key(role));
        assert_ne!(original.fit_key(role).cache().design(), widened.fit_key(role).cache().design());
        // An identity derived from other values cannot be run on this panel.
        let mut store = SequentialNuisanceStore::new();
        let error = estimate_sequential_contrast(
            SequentialPanel {
                baseline: &[],
                first_treatment: &first,
                intermediate: &[],
                second_treatment: &second,
                outcome: &outcome,
            },
            &widened,
            treated(),
            untreated(),
            &mut store,
        )
        .expect_err("a panel the identity was not derived from");
        assert!(matches!(error, SequentialCrossfitError::Refused { .. }), "{error}");
        assert_eq!(store.computed(), 0);
    }

    #[test]
    fn changed_outcome_does_not_share_a_fit() {
        let (first, second, mut outcome) = balanced_columns();
        outcome[3] += 0.5;
        let shifted = Marks::balanced().identity_for(SequentialPanel {
            baseline: &[],
            first_treatment: &first,
            intermediate: &[],
            second_treatment: &second,
            outcome: &outcome,
        });
        let original = Marks::balanced().identity();
        let role = SequentialRole::TerminalOutcome { second: ActionLevel::One };
        assert_ne!(original.fit_key(role), shifted.fit_key(role));
        assert_ne!(
            original.fit_key(role).cache().outcome(),
            shifted.fit_key(role).cache().outcome()
        );
        let mut store = SequentialNuisanceStore::new();
        let (first, second, outcome) = balanced_columns();
        let stale = estimate_sequential_contrast(
            SequentialPanel {
                baseline: &[],
                first_treatment: &first,
                intermediate: &[],
                second_treatment: &second,
                outcome: &outcome,
            },
            &shifted,
            treated(),
            untreated(),
            &mut store,
        );
        assert!(matches!(stale, Err(SequentialCrossfitError::Refused { .. })));
    }

    #[test]
    fn contrasts_in_one_cache_scope_share_certified_fits_and_a_changed_input_does_not() {
        let cache = CrossfitNuisanceCache::new();
        let identity = Marks::balanced().identity();
        let (mut a, mut b, mut c) = (
            SequentialNuisanceStore::new(),
            SequentialNuisanceStore::new(),
            SequentialNuisanceStore::new(),
        );
        let mut other = Marks::balanced();
        other.source = 9;
        let other = other.identity();
        let (first, second, third) = cache
            .scope(|| {
                (
                    run(&identity, &mut a, treated(), untreated()),
                    run(&identity, &mut b, untreated(), treated()),
                    run(&other, &mut c, treated(), untreated()),
                )
            })
            .0;
        assert_eq!((first.fits_computed, first.fits_reused), (6, 0));
        assert_eq!((second.fits_computed, second.fits_reused), (0, 6));
        close(second.contrast, -3.0);
        assert_eq!((third.fits_computed, third.fits_reused), (6, 0), "another source refits");
        let stats = cache.stats();
        assert_eq!((stats.bundle_fits, stats.bundle_reuses), (12, 6));
        // Outside a scope nothing is shared.
        let mut d = SequentialNuisanceStore::new();
        assert_eq!(run(&identity, &mut d, treated(), untreated()).fits_computed, 6);
        assert_eq!(cache.stats(), stats);
    }

    #[test]
    fn a_small_shared_pool_refuses_retention_and_each_contrast_fits_for_itself() {
        let pool = AdmissionPool::fixed(1, 64);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let cache = CrossfitNuisanceCache::charged_to(&ctx);
        let identity = Marks::balanced().identity();
        let (mut a, mut b) = (SequentialNuisanceStore::new(), SequentialNuisanceStore::new());
        let (first, second) = cache
            .scope(|| {
                (
                    run(&identity, &mut a, treated(), untreated()),
                    run(&identity, &mut b, treated(), untreated()),
                )
            })
            .0;
        assert_eq!(first.fits_computed, 6);
        assert_eq!(second.fits_computed, 6, "nothing was retained to reuse");
        close(second.contrast, first.contrast);
        assert_eq!(cache.retained_entries(), 0);
        assert!(cache.stats().refused_retentions >= 12);
        assert_eq!(pool.available().bytes(), 64);
    }

    #[test]
    fn changed_fold_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.swap_folds = true;
        assert_resplit(&marks.identity());
    }

    #[test]
    fn changed_treatment_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.treatment = "exposure";
        assert_resplit(&marks.identity());
    }

    #[test]
    fn changed_provider_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.provider = 12;
        assert_resplit(&marks.identity());
    }

    #[test]
    fn changed_numerical_dependency_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.floor = 0.2;
        assert_resplit(&marks.identity());
    }

    #[test]
    fn changed_source_does_not_share_a_fit() {
        let mut marks = Marks::balanced();
        marks.source = 9;
        assert_resplit(&marks.identity());
    }

    #[test]
    fn evaluation_block_omits_training_rows() {
        let identity = Marks::balanced().identity();
        let mut store = SequentialNuisanceStore::new();
        let _ = run(&identity, &mut store, treated(), untreated());
        let key = identity.fit_key(SequentialRole::BaselinePropensity);
        let fit = store.lookup(&key).expect("baseline");
        for block in fit.blocks() {
            for row in block.rows() {
                let assigned = key.folds().fold_of(row.row_id());
                assert_eq!(assigned, Some(block.evaluation_fold()));
                let training = key.folds().is_training_row(block.evaluation_fold(), row.row_id());
                assert!(!training);
            }
            let ids: Vec<u64> = block.rows().iter().map(PredictedRow::row_id).collect();
            if block.evaluation_fold() == 0 {
                assert!(ids.contains(&0));
                assert!(!ids.contains(&8));
            }
        }
    }

    #[test]
    fn a_fold_missing_an_action_is_refused() {
        let (first, _, outcome) = balanced_columns();
        let second = vec![1.0; SUBJECTS];
        let panel = SequentialPanel {
            baseline: &[],
            first_treatment: &first,
            intermediate: &[],
            second_treatment: &second,
            outcome: &outcome,
        };
        let identity = Marks::balanced().identity_for(panel);
        let error = estimate_sequential_contrast(
            panel,
            &identity,
            treated(),
            untreated(),
            &mut SequentialNuisanceStore::new(),
        )
        .expect_err("missing action");
        assert!(matches!(
            error,
            SequentialCrossfitError::MissingAction { fold: 0, decision: 1, action: 0 }
        ));
    }

    #[test]
    fn an_action_probability_outside_the_floor_is_refused() {
        let mut marks = Marks::balanced();
        marks.floor = 0.3;
        let (_, second, outcome) = balanced_columns();
        let first: Vec<f64> = (0..SUBJECTS).map(|row| f64::from(u8::from(row % 8 >= 2))).collect();
        let panel = SequentialPanel {
            baseline: &[],
            first_treatment: &first,
            intermediate: &[],
            second_treatment: &second,
            outcome: &outcome,
        };
        let identity = marks.identity_for(panel);
        let error = estimate_sequential_contrast(
            panel,
            &identity,
            treated(),
            untreated(),
            &mut SequentialNuisanceStore::new(),
        )
        .expect_err("positivity");
        assert!(matches!(error, SequentialCrossfitError::Positivity { decision: 0, .. }));
    }

    #[test]
    fn swapped_regimes_reuse_fits_and_negate_the_contrast() {
        let identity = Marks::balanced().identity();
        let mut store = SequentialNuisanceStore::new();
        let forward = run(&identity, &mut store, treated(), untreated());
        close(forward.contrast, 3.0);
        assert_eq!(forward.fits_computed, 6);
        let backward = run(&identity, &mut store, untreated(), treated());
        close(backward.contrast, -3.0);
        assert_eq!(backward.fits_computed, 0);
        assert_eq!(backward.fits_reused, 6);
        assert_eq!(store.computed(), 6);
    }

    #[test]
    fn conflicting_certification_keeps_the_stored_fit() {
        let folds = FoldAssignment::try_new(2, &[FoldRow::new(0, 0), FoldRow::new(1, 1)], &[])
            .expect("two rows");
        let panel = SequentialPanel {
            baseline: &[],
            first_treatment: &[0.0, 1.0],
            intermediate: &[],
            second_treatment: &[0.0, 1.0],
            outcome: &[1.0, 2.0],
        };
        let narrow = SequentialIdentity::from_panel(
            panel,
            CertifiedSource::new(SourceDependencyId::from_raw(11), digest(3)),
            TargetDependency::new(digest(1)),
            folds,
            ProviderDependency::new(ProviderId::from_raw(4), digest(6)),
            NumericalDependencies::try_new(
                NumericalPolicy::strict(),
                &[NumericalDependency::try_new(MIN_PROBABILITY, 0.05).expect("floor")],
            )
            .expect("numerical"),
            &NuisanceOptions::new("sequential"),
        )
        .expect("narrow identity");
        let key = narrow.fit_key(SequentialRole::BaselinePropensity);
        let coefficients = vec![
            FoldCoefficients::try_new(0, &[0.25]).expect("coef"),
            FoldCoefficients::try_new(1, &[0.75]).expect("coef"),
        ];
        let blocks = vec![
            BlockedOutOfFoldPredictions::try_new(
                0,
                &[PredictedRow::try_new(0, 0.25).expect("row")],
            )
            .expect("block"),
            BlockedOutOfFoldPredictions::try_new(
                1,
                &[PredictedRow::try_new(1, 0.75).expect("row")],
            )
            .expect("block"),
        ];
        let mut store = SequentialNuisanceStore::new();
        store.certify(key.clone(), &coefficients, &blocks).expect("store");
        let kept = std::ptr::from_ref(store.lookup(&key).expect("fit"));
        store.certify(key.clone(), &coefficients, &blocks).expect("same fit");
        assert_eq!(store.computed(), 1);
        assert_eq!(std::ptr::from_ref(store.lookup(&key).expect("fit")), kept);
        let other = vec![
            BlockedOutOfFoldPredictions::try_new(0, &[PredictedRow::try_new(0, 0.5).expect("row")])
                .expect("block"),
            blocks[1].clone(),
        ];
        let error = store.certify(key.clone(), &coefficients, &other).expect_err("conflict");
        assert!(matches!(
            error,
            SequentialCrossfitError::Cache(NuisanceCacheError::ConflictingFit)
        ));
        assert_eq!(store.computed(), 1);
        assert_eq!(std::ptr::from_ref(store.lookup(&key).expect("fit")), kept);
        let stored = store.lookup(&key).expect("fit").prediction(0).expect("row");
        assert_eq!(stored.to_bits(), 0.25_f64.to_bits());
    }

    #[test]
    fn a_different_source_does_not_lend_its_terminal_fit() {
        let donor = Marks::balanced().identity();
        let mut other = Marks::balanced();
        other.source = 9;
        let recipient = other.identity();
        let mut store = SequentialNuisanceStore::new();
        for second in [ActionLevel::Zero, ActionLevel::One] {
            let coefficients = vec![
                FoldCoefficients::try_new(0, &[100.0, 100.0]).expect("coef"),
                FoldCoefficients::try_new(1, &[100.0, 100.0]).expect("coef"),
            ];
            let blocks = vec![predictions(0, 0..8, 100.0), predictions(1, 8..16, 100.0)];
            store
                .certify(
                    donor.fit_key(SequentialRole::TerminalOutcome { second }),
                    &coefficients,
                    &blocks,
                )
                .expect("forged terminal");
        }
        let contrast = run(&recipient, &mut store, treated(), untreated());
        close(contrast.contrast, 3.0);
        assert_eq!(contrast.fits_computed, 6);
    }

    fn predictions(
        fold: u32,
        rows: std::ops::Range<u64>,
        value: f64,
    ) -> BlockedOutOfFoldPredictions {
        let predicted: Vec<PredictedRow> =
            rows.map(|row| PredictedRow::try_new(row, value).expect("prediction")).collect();
        BlockedOutOfFoldPredictions::try_new(fold, &predicted).expect("block")
    }
}
