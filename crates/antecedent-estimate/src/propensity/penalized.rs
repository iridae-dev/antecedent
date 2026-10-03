//! Explicit penalized binary-propensity configuration for cross-fitted AIPW.
//!
//! This is a *declared nuisance choice*, distinct from `GlmOptions::ridge_on_separation`
//! (a rescue the estimation paths refuse to keep). A ridge-logistic propensity is fit on each
//! cross-fit fold's **training rows only**; its penalty is chosen from a fixed grid by inner
//! cross-validated log loss on those same training rows (seeded and replayable), so the
//! evaluation fold never informs the penalty. The fits reuse `antecedent-learn`'s
//! [`RidgeLogisticLearner`] and cross-fit driver; there is no second ML stack.
//!
//! Penalty scale: the objective is `-loglik + (lambda / 2) * ||beta||^2` on a *sum* (not mean)
//! log likelihood over the fold's training rows, with the intercept unpenalized and each
//! non-intercept column standardized by the training rows' own mean and standard deviation
//! (a column with no variance is zeroed, so it carries no coefficient). The penalty grid is
//! therefore comparable across covariate units but not across training-set sizes.
//!
//! No interval is licensed for this route (see `docs/guides/penalized-aipw.md`): the configuration
//! travels with the score table through its provenance string, and
//! [`provenance_withholds_interval`] is the one test every consumer of a frozen table uses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::hash::{Hash, Hasher};

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    DesignView, LearnerFactory, LearnerProvenance, RidgeLogisticLearner, TargetView,
    cross_fit_with_folds,
};

use crate::error::EstimationError;
use crate::learn_nuisance::{crossfit_fold_plan, intercept_is_constant, learn_err};
use crate::splitmix::mix64;

/// Inner cross-validation folds used to choose the ridge penalty when none is declared.
pub const DEFAULT_RIDGE_INNER_FOLDS: usize = 5;

/// Default ridge penalty grid (sum-scale `lambda`, standardized covariates).
pub const DEFAULT_RIDGE_GRID: [f64; 6] = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3];

/// Largest declared penalty grid.
const MAX_RIDGE_GRID: usize = 64;

/// Largest declared number of inner cross-validation folds.
const MAX_RIDGE_INNER_FOLDS: usize = 20;

/// Marker the score table's provenance carries when its propensity was ridge-penalized.
const PENALIZED_PROVENANCE_TAG: &str = ";propensity=ridge_logistic";

/// Stream tag of the inner penalty-selection fold plan.
const INNER_PLAN_TAG: u64 = 0x1AE5_17D0_F01D_0001;

/// A refusal carrying its registered reason code and the namespaced detail callers switch on.
pub(crate) fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn invalid(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "penalized_propensity.invalid_penalty",
        message,
    )
}

/// Declared ridge-penalty tuning rule: a fixed grid and the inner folds that rank it.
///
/// The grid is stored sorted ascending and deduplicated, so equal declarations have one
/// canonical form. Fields are private: every value passed [`Self::new`], so each penalty is
/// finite and positive and equality is total.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeTuning {
    lambdas: Vec<f64>,
    inner_folds: usize,
}

impl Eq for RidgeTuning {}

impl Default for RidgeTuning {
    fn default() -> Self {
        Self { lambdas: DEFAULT_RIDGE_GRID.to_vec(), inner_folds: DEFAULT_RIDGE_INNER_FOLDS }
    }
}

impl RidgeTuning {
    /// Declare a penalty grid and the number of inner cross-validation folds.
    ///
    /// # Errors
    ///
    /// A refusal with the `invalid_argument` reason code for an empty or oversized grid, a
    /// penalty that is not finite and positive, or an inner fold count outside `2..=20`.
    pub fn new(lambdas: &[f64], inner_folds: usize) -> Result<Self, EstimationError> {
        if lambdas.is_empty() || lambdas.len() > MAX_RIDGE_GRID {
            return Err(invalid(&format!(
                "ridge penalty grid must hold 1 to {MAX_RIDGE_GRID} values, got {}",
                lambdas.len()
            )));
        }
        if let Some(bad) = lambdas.iter().find(|l| !l.is_finite() || **l <= 0.0) {
            return Err(invalid(&format!("ridge penalty must be finite and > 0, got {bad}")));
        }
        if !(2..=MAX_RIDGE_INNER_FOLDS).contains(&inner_folds) {
            return Err(invalid(&format!(
                "ridge penalty selection needs 2 to {MAX_RIDGE_INNER_FOLDS} inner folds, \
                 got {inner_folds}"
            )));
        }
        let mut sorted = lambdas.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup_by(|a, b| a.total_cmp(b).is_eq());
        Ok(Self { lambdas: sorted, inner_folds })
    }

    /// The penalty grid, ascending.
    #[must_use]
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    /// Inner cross-validation folds that rank the grid.
    #[must_use]
    pub const fn inner_folds(&self) -> usize {
        self.inner_folds
    }
}

/// Penalty on the binary propensity nuisance of a cross-fitted AIPW.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum PropensityPenalty {
    /// The unpenalized logistic MLE (the established route).
    #[default]
    None,
    /// Ridge logistic with a penalty chosen on each fold's training rows.
    RidgeLogistic(RidgeTuning),
    /// Lasso logistic. Closed: variable selection changes the inference contract, so the
    /// option is declarable and refused at execution (`selection_inference_not_licensed`).
    Lasso,
}

/// Declared GLM-to-ML nuisance fallback.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum NuisanceFallback {
    /// A failed nuisance fit is a refusal.
    #[default]
    None,
    /// Fall back to a flexible learner when the GLM fit fails. Closed: the destination route
    /// has no support or uncertainty license, so a failed fit is refused with the failure
    /// recorded (`nuisance_fallback_not_licensed`) and never silently replaced.
    Ml,
}

/// Canonical, fingerprintable configuration of the binary propensity nuisance.
///
/// `Eq` and `Hash` are consistent with [`Self::canonical_bytes`], the byte form a batch or
/// contract identity digests: two configurations share nuisance fits only if equal.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PropensityNuisance {
    penalty: PropensityPenalty,
    fallback: NuisanceFallback,
}

impl Hash for PropensityNuisance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_bytes().hash(state);
    }
}

impl PropensityNuisance {
    /// Ridge-logistic propensity with the declared tuning rule.
    #[must_use]
    pub fn ridge_logistic(tuning: RidgeTuning) -> Self {
        Self { penalty: PropensityPenalty::RidgeLogistic(tuning), fallback: NuisanceFallback::None }
    }

    /// Lasso propensity: a closed option, refused by [`Self::validate_for_execution`].
    #[must_use]
    pub fn lasso() -> Self {
        Self { penalty: PropensityPenalty::Lasso, fallback: NuisanceFallback::None }
    }

    /// This configuration with a declared fallback policy.
    #[must_use]
    pub fn with_fallback(self, fallback: NuisanceFallback) -> Self {
        Self { fallback, ..self }
    }

    /// Declared penalty.
    #[must_use]
    pub const fn penalty(&self) -> &PropensityPenalty {
        &self.penalty
    }

    /// Declared fallback policy.
    #[must_use]
    pub const fn fallback(&self) -> NuisanceFallback {
        self.fallback
    }

    /// Whether this is the established unpenalized, no-fallback configuration.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether a penalty (including a closed one) is declared.
    #[must_use]
    pub const fn is_penalized(&self) -> bool {
        !matches!(self.penalty, PropensityPenalty::None)
    }

    /// Refuse the closed options before any work is done.
    ///
    /// # Errors
    ///
    /// `selection_inference_not_licensed` for a lasso penalty.
    pub fn validate_for_execution(&self) -> Result<(), EstimationError> {
        if matches!(self.penalty, PropensityPenalty::Lasso) {
            return Err(refuse(
                antecedent_core::reason_code!("selection_inference_not_licensed"),
                "penalized_propensity.selection_closed",
                "a lasso propensity selects covariates from the data, so an ordinary \
                 influence-function interval is not valid after selection; the option is \
                 closed, use the ridge-logistic penalty or the unpenalized propensity",
            ));
        }
        Ok(())
    }

    /// Record a failed nuisance fit under a declared fallback: the fallback destination has no
    /// license, so the failure is the result and nothing is silently substituted. Any other
    /// declaration (or a non-fit failure) passes `error` through unchanged.
    #[must_use]
    pub fn record_failed_fit(&self, error: EstimationError) -> EstimationError {
        if self.fallback == NuisanceFallback::Ml && matches!(error, EstimationError::Stats(_)) {
            return refuse(
                antecedent_core::reason_code!("nuisance_fallback_not_licensed"),
                "penalized_propensity.fallback_closed",
                &format!(
                    "the primary nuisance fit failed ({error}); the declared ML fallback has no \
                     support or uncertainty license for this estimand, so the failed fit is \
                     reported and no fallback result is produced"
                ),
            );
        }
        error
    }

    /// Canonical byte form: a versioned tag, then the penalty and fallback as fixed-width
    /// fields (counts as little-endian `u64`, penalties as little-endian `f64` bit patterns).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"antecedent.propensity_nuisance.v1\0".to_vec();
        match &self.penalty {
            PropensityPenalty::None => bytes.push(0),
            PropensityPenalty::RidgeLogistic(tuning) => {
                bytes.push(1);
                bytes.extend_from_slice(&(tuning.inner_folds as u64).to_le_bytes());
                bytes.extend_from_slice(&(tuning.lambdas.len() as u64).to_le_bytes());
                for lambda in &tuning.lambdas {
                    bytes.extend_from_slice(&lambda.to_bits().to_le_bytes());
                }
            }
            PropensityPenalty::Lasso => bytes.push(2),
        }
        bytes.push(match self.fallback {
            NuisanceFallback::None => 0,
            NuisanceFallback::Ml => 1,
        });
        bytes
    }

    /// Human-readable canonical key, stable across runs (penalties as `f64` bit patterns).
    #[must_use]
    pub fn canonical_key(&self) -> String {
        let mut key = match &self.penalty {
            PropensityPenalty::None => "glm".to_string(),
            PropensityPenalty::RidgeLogistic(tuning) => {
                let grid: Vec<String> =
                    tuning.lambdas.iter().map(|l| format!("{:016x}", l.to_bits())).collect();
                format!("ridge_logistic.cv(k={},grid={})", tuning.inner_folds, grid.join(","))
            }
            PropensityPenalty::Lasso => "lasso".to_string(),
        };
        if self.fallback == NuisanceFallback::Ml {
            key.push_str(";fallback=ml");
        }
        key
    }

    /// The `nuisance_provenance` suffix a score table carries for this configuration and
    /// the penalties it selected per fold; empty for an unpenalized configuration.
    pub(crate) fn provenance_suffix(&self, selections: &[RidgeFoldSelection]) -> String {
        if !self.is_penalized() {
            return String::new();
        }
        let mut suffix = format!(";propensity={}", self.canonical_key());
        if !selections.is_empty() {
            let chosen: Vec<String> =
                selections.iter().map(|s| format!("{:016x}", s.lambda.to_bits())).collect();
            suffix.push_str(";selected_lambda=");
            suffix.push_str(&chosen.join(","));
        }
        suffix
    }
}

/// Whether a score table's provenance marks a ridge-penalized propensity or whole-cluster
/// (cluster-DML) cross-fitting.
///
/// Such a table is a point-and-score artifact: it can be retargeted and its scores
/// summarized, but no interval or joint covariance is published from it.
#[must_use]
pub fn provenance_withholds_interval(provenance: &str) -> bool {
    provenance.contains(PENALIZED_PROVENANCE_TAG)
        || crate::cluster_dml_aipw::provenance_marks_cluster_units(provenance)
}

/// Penalty chosen on one cross-fit fold's training rows.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RidgeFoldSelection {
    /// Outer fold the penalty was chosen for.
    pub(crate) fold: usize,
    /// Selected penalty.
    pub(crate) lambda: f64,
    /// Inner cross-validated mean log loss of the selected penalty.
    pub(crate) inner_logloss: f64,
    /// Fitted learner identity of the final training-fold model.
    pub(crate) provenance: LearnerProvenance,
}

/// One outer fold's training and evaluation rows for a penalized propensity fit.
pub(crate) struct RidgeFoldInput<'a> {
    /// Column-major `[1 | Z...]` training design.
    pub(crate) design_train: &'a [f64],
    /// Training rows.
    pub(crate) n_train: usize,
    /// Column-major `[1 | Z...]` evaluation design.
    pub(crate) design_valid: &'a [f64],
    /// Evaluation rows.
    pub(crate) n_valid: usize,
    /// Design width.
    pub(crate) ncols: usize,
    /// Training treatment, 0/1.
    pub(crate) t_train: &'a [f64],
    /// Unit (original row) identity of each training row.
    pub(crate) units: &'a [u32],
    /// Outer fold index.
    pub(crate) fold: usize,
    /// Seed of the fold plan the outer folds were drawn from.
    pub(crate) seed: u64,
}

/// Standardize every non-intercept column by the training rows' mean and standard
/// deviation, applying the training statistics to the evaluation rows.
fn standardize(input: &RidgeFoldInput<'_>) -> (Vec<f64>, Vec<f64>) {
    let (n_train, n_valid) = (input.n_train, input.n_valid);
    let mut train = input.design_train.to_vec();
    let mut valid = input.design_valid.to_vec();
    for c in 1..input.ncols {
        let col = &input.design_train[c * n_train..(c + 1) * n_train];
        let mean = col.iter().sum::<f64>() / n_train as f64;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n_train as f64;
        let sd = var.sqrt();
        let scale = if sd > 1e-12 * mean.abs().max(1.0) { 1.0 / sd } else { 0.0 };
        for v in &mut train[c * n_train..(c + 1) * n_train] {
            *v = (*v - mean) * scale;
        }
        for v in &mut valid[c * n_valid..(c + 1) * n_valid] {
            *v = (*v - mean) * scale;
        }
    }
    (train, valid)
}

/// Fit the ridge-logistic propensity of one outer fold and predict its evaluation rows.
///
/// The penalty is the grid value with the smallest inner cross-validated log loss over the
/// training rows (ties go to the larger penalty); a grid value whose inner fit fails is not a
/// candidate. The inner folds are a seeded, arm-stratified, unit-level plan, so the selection
/// is a function of the training rows, the seed, and the grid alone.
///
/// # Errors
///
/// A design without a leading intercept, an inner fold plan the training units cannot
/// support, cancellation (`cancelled_no_claim`), or no grid value with a usable fit (the
/// failure of the last candidate is recorded in the message).
pub(crate) fn fit_ridge_fold(
    tuning: &RidgeTuning,
    input: &RidgeFoldInput<'_>,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, RidgeFoldSelection), EstimationError> {
    if !intercept_is_constant(input.design_train, input.n_train) {
        return Err(EstimationError::data_msg(
            "ridge-logistic propensity requires the leading intercept column",
        ));
    }
    let (train, valid) = standardize(input);
    let train_view =
        DesignView::from_column_major(&train, input.n_train, input.ncols).map_err(learn_err)?;
    let valid_view =
        DesignView::from_column_major(&valid, input.n_valid, input.ncols).map_err(learn_err)?;
    let target = TargetView::new(input.t_train);

    let arms: Vec<u32> = input.t_train.iter().map(|&t| u32::from(t > 0.5)).collect();
    let plan = crossfit_fold_plan(
        &arms,
        input.units,
        tuning.inner_folds,
        mix64(input.seed ^ INNER_PLAN_TAG.wrapping_add(input.fold as u64)),
    )?;
    let inner: Vec<u16> = plan.iter().map(|&f| u16::try_from(f).unwrap_or(u16::MAX)).collect();

    let mut best: Option<(f64, f64, _)> = None;
    let mut failed = 0usize;
    let mut last_failure = String::new();
    for &lambda in tuning.lambdas() {
        if ctx.cancellation.is_cancelled() {
            return Err(refuse(
                antecedent_core::reason_code!("cancelled_no_claim"),
                "penalized_propensity.cancelled",
                "ridge-logistic penalty selection was cancelled; no estimate is reported",
            ));
        }
        let factory = RidgeLogisticLearner::new(lambda).map_err(learn_err)?;
        let inner_loss =
            cross_fit_with_folds(&factory, train_view, target, inner.clone(), ctx, None)
                .map_err(|error| error.to_string())
                .and_then(|oof| {
                    oof.validation
                        .logloss
                        .filter(|loss| loss.is_finite())
                        .ok_or_else(|| "non-finite inner log loss".to_string())
                });
        match inner_loss {
            // A penalty is a candidate only if its fit on all training rows is usable too: the
            // inner folds fit fewer rows, so a penalty can pass inside and still saturate here.
            Ok(loss) if best.as_ref().is_none_or(|(_, b, _)| loss <= *b) => {
                match factory.fit(train_view, target, None, ctx) {
                    Ok(model) => best = Some((lambda, loss, model)),
                    Err(error) => {
                        failed += 1;
                        last_failure = error.to_string();
                    }
                }
            }
            Ok(_) => {}
            Err(failure) => {
                failed += 1;
                last_failure = failure;
            }
        }
    }
    let Some((lambda, inner_logloss, model)) = best else {
        return Err(EstimationError::stats_msg(format!(
            "ridge-logistic propensity: none of the {failed} penalties in the grid produced a \
             usable inner cross-validated fit on fold {} (last failure: {last_failure})",
            input.fold
        )));
    };
    let mut predictions = vec![0.0; input.n_valid];
    model.predict(valid_view, &mut predictions, ctx).map_err(learn_err)?;
    // The learner reports the family; the chosen penalty is part of the fitted identity,
    // spelled like `LearnerSpec::name` (`logistic:<bits of lambda>`).
    let mut provenance = model.provenance();
    provenance.spec = format!("logistic:{}", lambda.to_bits());
    let selection = RidgeFoldSelection { fold: input.fold, lambda, inner_logloss, provenance };
    Ok((predictions, selection))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "canonical grids and bit patterns are compared exactly on purpose"
)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use antecedent_stats::{FaerBackend, GlmOptions, PropensityWorkspace, fit_propensity};

    use super::*;
    use crate::splitmix::splitmix64_unit;

    fn hash_of(nuisance: &PropensityNuisance) -> u64 {
        let mut hasher = DefaultHasher::new();
        nuisance.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn tuning_validates_and_canonicalizes_the_declared_grid() {
        let tuning = RidgeTuning::new(&[10.0, 0.5, 10.0, 2.0], 4).unwrap();
        assert_eq!(tuning.lambdas(), &[0.5, 2.0, 10.0]);
        assert_eq!(tuning.inner_folds(), 4);
        for bad in [
            RidgeTuning::new(&[], 5),
            RidgeTuning::new(&[0.0], 5),
            RidgeTuning::new(&[-1.0], 5),
            RidgeTuning::new(&[f64::NAN], 5),
            RidgeTuning::new(&[f64::INFINITY], 5),
            RidgeTuning::new(&[1.0], 1),
            RidgeTuning::new(&[1.0], 21),
            RidgeTuning::new(&[1.0; 65], 5),
        ] {
            let error = bad.unwrap_err();
            assert!(error.to_string().contains("invalid_argument"), "{error}");
        }
    }

    #[test]
    fn canonical_identity_separates_every_declared_choice() {
        let base = PropensityNuisance::ridge_logistic(RidgeTuning::new(&[1.0, 10.0], 5).unwrap());
        let same = PropensityNuisance::ridge_logistic(RidgeTuning::new(&[10.0, 1.0], 5).unwrap());
        assert_eq!(base, same);
        assert_eq!(hash_of(&base), hash_of(&same));
        assert_eq!(base.canonical_bytes(), same.canonical_bytes());
        let others = [
            PropensityNuisance::default(),
            PropensityNuisance::lasso(),
            PropensityNuisance::ridge_logistic(RidgeTuning::new(&[1.0, 10.0], 4).unwrap()),
            PropensityNuisance::ridge_logistic(RidgeTuning::new(&[1.0, 10.5], 5).unwrap()),
            PropensityNuisance::ridge_logistic(RidgeTuning::new(&[1.0], 5).unwrap()),
            base.clone().with_fallback(NuisanceFallback::Ml),
        ];
        for other in &others {
            assert_ne!(&base, other);
            assert_ne!(base.canonical_bytes(), other.canonical_bytes());
            assert_ne!(base.canonical_key(), other.canonical_key());
        }
        assert!(PropensityNuisance::default().is_default());
        assert_eq!(PropensityNuisance::default().canonical_key(), "glm");
        assert!(!base.is_default() && base.is_penalized());
    }

    #[test]
    fn lasso_is_closed_with_its_reason_code_and_ridge_is_not() {
        let error = PropensityNuisance::lasso().validate_for_execution().unwrap_err();
        assert!(error.to_string().contains("selection_inference_not_licensed"), "{error}");
        assert!(PropensityNuisance::default().validate_for_execution().is_ok());
        assert!(
            PropensityNuisance::ridge_logistic(RidgeTuning::default())
                .validate_for_execution()
                .is_ok()
        );
    }

    #[test]
    fn provenance_marks_only_ridge_tables_and_records_the_selected_penalties() {
        let ridge = PropensityNuisance::ridge_logistic(RidgeTuning::default());
        let selection = |fold, lambda| RidgeFoldSelection {
            fold,
            lambda,
            inner_logloss: 0.5,
            provenance: LearnerProvenance {
                spec: "logistic".into(),
                implementation: "faer_ridge".into(),
                version: "0".into(),
            },
        };
        let suffix = ridge.provenance_suffix(&[selection(0, 1.0), selection(1, 10.0)]);
        assert!(provenance_withholds_interval(&format!("aipw.crossfit.v1{suffix}")));
        assert!(suffix.contains(&format!(
            "{:016x},{:016x}",
            1.0_f64.to_bits(),
            10.0_f64.to_bits()
        )));
        assert_eq!(PropensityNuisance::default().provenance_suffix(&[]), "");
        let fallback_only = PropensityNuisance::default().with_fallback(NuisanceFallback::Ml);
        assert_eq!(fallback_only.provenance_suffix(&[]), "");
        assert!(!provenance_withholds_interval("aipw.crossfit.v1;batch.shared_design"));
    }

    /// Independent standardization: the training mean/sd are applied to the evaluation rows,
    /// a constant column is zeroed, and the intercept is untouched.
    #[test]
    fn standardization_uses_training_statistics_only() {
        let train = [1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 6.0, 8.0, 3.0, 3.0, 3.0, 3.0];
        let valid = [1.0, 1.0, 100.0, -3.0, 5.0, 7.0];
        let units = [0, 1, 2, 3];
        let t = [0.0, 1.0, 0.0, 1.0];
        let input = RidgeFoldInput {
            design_train: &train,
            n_train: 4,
            design_valid: &valid,
            n_valid: 2,
            ncols: 3,
            t_train: &t,
            units: &units,
            fold: 0,
            seed: 1,
        };
        let (zt, zv) = standardize(&input);
        let sd = 5.0_f64.sqrt();
        assert_eq!(&zt[..4], &[1.0; 4]);
        for (got, raw) in zt[4..8].iter().zip([2.0, 4.0, 6.0, 8.0]) {
            assert!((got - (raw - 5.0) / sd).abs() < 1e-12);
        }
        assert_eq!(&zt[8..], &[0.0; 4]);
        assert_eq!(&zv[..2], &[1.0, 1.0]);
        assert!((zv[2] - (100.0 - 5.0) / sd).abs() < 1e-12);
        assert!((zv[3] - (-3.0 - 5.0) / sd).abs() < 1e-12);
        assert_eq!(&zv[4..], &[0.0, 0.0]);
    }
    /// `[1 | a | b | a]` for 80 training and 20 evaluation rows, with a treatment that
    /// depends on `a` and `b`: the last column duplicates the second.
    struct Duplicated {
        train: Vec<f64>,
        valid: Vec<f64>,
        t: Vec<f64>,
        units: Vec<u32>,
    }

    fn duplicated() -> Duplicated {
        let (n, nv) = (80usize, 20usize);
        let mut state = 17u64;
        let mut cols = [vec![1.0; n + nv], vec![0.0; n + nv], vec![0.0; n + nv]];
        let mut t = vec![0.0; n + nv];
        for i in 0..n + nv {
            let a = splitmix64_unit(&mut state) - 0.5;
            let b = splitmix64_unit(&mut state) - 0.5;
            cols[1][i] = a;
            cols[2][i] = b;
            let p = 1.0 / (1.0 + (-(2.0 * a - b)).exp());
            t[i] = f64::from(u8::from(splitmix64_unit(&mut state) < p));
        }
        let design = |rows: std::ops::Range<usize>| -> Vec<f64> {
            [&cols[0], &cols[1], &cols[2], &cols[1]]
                .iter()
                .flat_map(|c| c[rows.clone()].iter().copied())
                .collect()
        };
        Duplicated {
            train: design(0..n),
            valid: design(n..n + nv),
            t: t[..n].to_vec(),
            units: (0..80).collect(),
        }
    }

    /// An exact duplicate column is a rank deficiency: the unpenalized logistic refuses it,
    /// the penalized fit does not (the standardized duplicates share one penalty), and the
    /// penalty it selects comes from the declared grid.
    #[test]
    fn a_duplicated_column_defeats_the_unpenalized_fit_but_not_the_ridge_fit() {
        let d = duplicated();
        let unpenalized = fit_propensity(
            &d.train,
            80,
            4,
            &d.t,
            &FaerBackend,
            &mut PropensityWorkspace::default(),
            &GlmOptions::default(),
        );
        assert!(unpenalized.is_err(), "rank-deficient design must refuse the plain logistic");
        let tuning = RidgeTuning::new(&[0.1, 1.0, 10.0], 4).unwrap();
        let input = RidgeFoldInput {
            design_train: &d.train,
            n_train: 80,
            design_valid: &d.valid,
            n_valid: 20,
            ncols: 4,
            t_train: &d.t,
            units: &d.units,
            fold: 0,
            seed: 5,
        };
        let (predicted, selection) =
            fit_ridge_fold(&tuning, &input, &ExecutionContext::for_tests(5)).unwrap();
        assert_eq!(predicted.len(), 20);
        assert!(predicted.iter().all(|p| *p > 0.0 && *p < 1.0));
        assert!(tuning.lambdas().contains(&selection.lambda));
        assert_eq!(selection.provenance.spec, format!("logistic:{}", selection.lambda.to_bits()));
        // The same inputs replay bit for bit.
        let (again, same) =
            fit_ridge_fold(&tuning, &input, &ExecutionContext::for_tests(5)).unwrap();
        assert_eq!(
            again.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
            predicted.iter().map(|p| p.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(same, selection);
    }

    #[test]
    fn a_cancelled_penalty_selection_is_a_stop_never_a_verdict() {
        let d = duplicated();
        let input = RidgeFoldInput {
            design_train: &d.train,
            n_train: 80,
            design_valid: &d.valid,
            n_valid: 20,
            ncols: 4,
            t_train: &d.t,
            units: &d.units,
            fold: 0,
            seed: 5,
        };
        let ctx = ExecutionContext::for_tests(5);
        ctx.cancellation.cancel();
        let error = fit_ridge_fold(&RidgeTuning::default(), &input, &ctx).unwrap_err();
        assert!(error.to_string().contains("cancelled_no_claim"), "{error}");
    }

    #[test]
    fn a_design_without_a_leading_intercept_is_refused() {
        let d = duplicated();
        let shifted: Vec<f64> = d.train.iter().map(|v| v + 3.0).collect();
        let input = RidgeFoldInput {
            design_train: &shifted,
            n_train: 80,
            design_valid: &d.valid,
            n_valid: 20,
            ncols: 4,
            t_train: &d.t,
            units: &d.units,
            fold: 0,
            seed: 5,
        };
        let error =
            fit_ridge_fold(&RidgeTuning::default(), &input, &ExecutionContext::for_tests(5))
                .unwrap_err();
        assert!(error.to_string().contains("intercept"), "{error}");
    }
}
