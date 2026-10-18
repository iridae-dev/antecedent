//! Cross-validated mechanism scores.

use std::borrow::Cow;

use antecedent_core::CausalRng;
use antecedent_data::TabularData;
use antecedent_kernels::{ViewError, shuffled_fold_assignment};
use antecedent_stats::{FaerBackend, LeastSquaresWorkspace};

use crate::batch::ParentBatch;
use crate::compile::{CompiledCausalModel, MechanismSlot, ParentGatherPlan};
use crate::error::ModelError;
use crate::mechanism::log_prob_column;

use super::{MechanismCandidate, MechanismFamily, ValidationSplits};

/// Spread, relative to the outcome's largest magnitude, above which
/// [`MechanismFamily::Constant`] is inadmissible.
///
/// `Constant` claims the variable is deterministic. A held-out likelihood cannot
/// score a point mass against a density, so its admissibility is decided on the
/// claim itself: any spread above numerical noise means the claim is false, and it
/// would silently strip every root of its distribution. The threshold is relative so
/// a variable measured in small units is not declared deterministic for its units.
pub(super) const CONSTANT_FAMILY_MAX_RELATIVE_SD: f64 = 1e-6;

/// Folds of the deterministic cross-validation that scores every family.
pub(crate) const MECHANISM_CV_FOLDS: usize = 5;

/// Held-out folds of the time-ordered scheme (one more block than this trains only).
pub(super) const TIME_ORDERED_FOLDS: usize = MECHANISM_CV_FOLDS;

/// Fewest rows per block a time-ordered scheme accepts: a block must hold enough
/// rows to train and to score a flexible family.
pub(super) const MIN_ROWS_PER_TIME_BLOCK: usize = 8;

/// Seed of the partition behind [`MECHANISM_CV_FOLDS`]. Fixed so the score of a
/// family is a function of the table alone; not a user-facing random stream.
pub(super) const CV_FOLD_SEED: u64 = 0x4D45_4348_4356_0001;

/// Column view a mechanism fit reads: the parent columns of one node and the
/// optional unit grouping. Fits and their cross-validation refits go through
/// this view, so a training fold is a row subset of the same object rather than
/// a second table.
pub(super) struct FitFrame<'d> {
    /// Parent columns, borrowed from the table or owned for a fold subset.
    pub(super) parents: Vec<Cow<'d, [f64]>>,
    /// Raw `UnitId` labels per row (the group structure is validated where used).
    pub(super) groups: Option<Vec<u32>>,
    /// Row count this frame covers.
    pub(super) n: usize,
    /// How held-out folds are cut from these rows. Training subsets selected from
    /// a frame keep the scheme, so a fit that cross-validates internally (the tree
    /// families' out-of-fold noise scale) splits the way the selection does.
    pub(super) splits: ValidationSplits,
}

impl<'d> FitFrame<'d> {
    /// Build a design frame for one child and its parents.
    pub(super) fn from_data(
        gather: &ParentGatherPlan,
        model: &CompiledCausalModel,
        data: &'d TabularData,
        n: usize,
    ) -> Result<Self, ModelError> {
        let parents = super::fit::gather_parent_cols(gather, model, data)?;
        if parents.iter().any(|c| c.len() < n) {
            return Err(ModelError::Shape {
                message: "parent column shorter than the outcome".into(),
            });
        }
        Ok(Self {
            parents,
            groups: super::fit::unit_id_groups(data, n),
            n,
            splits: ValidationSplits::Shuffled,
        })
    }

    /// Same frame with the declared validation-split scheme.
    pub(super) fn with_splits(mut self, splits: ValidationSplits) -> Self {
        self.splits = splits;
        self
    }

    /// Folds and seed the declared scheme plans for these rows; refuses a scheme
    /// the rows cannot support.
    pub(super) fn split_plan(&self) -> Result<SplitPlan, ModelError> {
        split_plan(self.n, self.splits, self.groups.as_deref())
    }

    /// Held-out folds of these rows under the declared scheme, optionally
    /// stratified (shuffled scheme only).
    pub(super) fn validation_folds(
        &self,
        strata: Option<&[u32]>,
    ) -> Result<Vec<FoldSplit>, ModelError> {
        validation_folds(self.n, self.splits, self.groups.as_deref(), strata)
    }

    /// Parent-column count.
    pub(super) fn n_parents(&self) -> usize {
        self.parents.len()
    }

    /// Owned frame over `rows` (in the given order).
    pub(super) fn select(&self, rows: &[usize]) -> FitFrame<'static> {
        FitFrame {
            parents: self
                .parents
                .iter()
                .map(|c| Cow::Owned(rows.iter().map(|&r| c[r]).collect::<Vec<f64>>()))
                .collect(),
            groups: self.groups.as_ref().map(|g| rows.iter().map(|&r| g[r]).collect()),
            n: rows.len(),
            splits: self.splits,
        }
    }

    /// Same rows, replaced parent columns (e.g. within-group demeaned).
    pub(super) fn with_parents(&self, parents: Vec<Vec<f64>>) -> FitFrame<'static> {
        FitFrame {
            parents: parents.into_iter().map(Cow::Owned).collect(),
            groups: self.groups.clone(),
            n: self.n,
            splits: self.splits,
        }
    }

    /// Column-major parent matrix `[parent * n + row]`.
    pub(super) fn column_major(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * self.n_parents());
        for c in &self.parents {
            out.extend_from_slice(&c[..self.n]);
        }
        out
    }
}

/// Whether this family models a categorical outcome.
pub(super) fn family_is_categorical(family: MechanismFamily) -> bool {
    family.is_categorical()
}

/// Dense stratum id per row for a categorical outcome (quantized value order).
#[allow(
    clippy::cast_possible_truncation,
    reason = "keys quantise outcomes to 1e-6 with a saturating cast that only merges values beyond 9e12 in magnitude, and a stratum id is a position among at most y.len() distinct keys, which fits u32"
)]
/// Dense codes of the finite outcome values, in first-seen order.
pub(super) fn category_strata(y: &[f64]) -> Vec<u32> {
    let key = |v: f64| (v * 1e6).round() as i64;
    let mut keys: Vec<i64> = y.iter().map(|&v| key(v)).collect();
    keys.sort_unstable();
    keys.dedup();
    y.iter().map(|&v| keys.binary_search(&key(v)).map_or(0, |i| i as u32)).collect()
}

/// Seeded fold of every row, shared by every family scored on the same node so
/// candidates are compared on identical training/held-out splits. Stratified on
/// the outcome for categorical families so each training fold keeps every level.
pub(super) fn cv_fold_plan(n: usize, strata: Option<&[u32]>) -> Result<Vec<usize>, ModelError> {
    let folds = MECHANISM_CV_FOLDS.min(n);
    if folds < 2 {
        return Err(ModelError::Unsupported {
            message: "not enough rows to cross-validate a mechanism family".into(),
        });
    }
    let mut rng = CausalRng::from_seed(CV_FOLD_SEED);
    shuffled_fold_assignment(&mut rng, n, folds, strata).map_err(fold_error)
}

fn fold_error(error: ViewError) -> ModelError {
    ModelError::Shape {
        message: match error {
            ViewError::InvalidShape { message } => message.to_owned(),
            other => other.to_string(),
        },
    }
}

/// One training / held-out row split, rows ascending.
pub(super) struct FoldSplit {
    pub(super) train: Vec<usize>,
    pub(super) test: Vec<usize>,
}

/// Folds and seed a validation scheme plans for a set of rows.
pub(super) struct SplitPlan {
    pub(super) folds: usize,
    pub(super) seed: Option<u64>,
}

/// Held-out folds of `n` rows under `splits`.
///
/// `Shuffled` is the seeded shuffled partition (stratified when `strata` is given);
/// the other schemes ignore `strata` because their folds are fixed by the rows'
/// dependence, not by the outcome's levels (a held-out level absent from the
/// training rows is simply not scored).
pub(super) fn validation_folds(
    n: usize,
    splits: ValidationSplits,
    groups: Option<&[u32]>,
    strata: Option<&[u32]>,
) -> Result<Vec<FoldSplit>, ModelError> {
    match splits {
        ValidationSplits::Shuffled => {
            let fold = cv_fold_plan(n, strata)?;
            let folds = fold.iter().copied().max().map_or(0, |m| m + 1);
            Ok((0..folds)
                .map(|k| {
                    let (train, test): (Vec<usize>, Vec<usize>) =
                        (0..n).partition(|&r| fold[r] != k);
                    FoldSplit { train, test }
                })
                .collect())
        }
        ValidationSplits::TimeOrdered { purge } => {
            let blocks = TIME_ORDERED_FOLDS + 1;
            if n < MIN_ROWS_PER_TIME_BLOCK * blocks {
                return Err(ModelError::InvalidSelection {
                    message: format!(
                        "time-ordered validation needs at least {} rows, got {n}",
                        MIN_ROWS_PER_TIME_BLOCK * blocks
                    ),
                });
            }
            let edge = |j: usize| j * n / blocks;
            Ok((1..blocks)
                .map(|k| FoldSplit {
                    train: (0..edge(k).saturating_sub(purge)).collect(),
                    test: (edge(k)..edge(k + 1)).collect(),
                })
                .collect())
        }
        ValidationSplits::Clustered => {
            let units = clustered_units(groups, n)?;
            let folds = MECHANISM_CV_FOLDS.min(units.len());
            let mut rng = CausalRng::from_seed(CV_FOLD_SEED);
            let unit_fold =
                shuffled_fold_assignment(&mut rng, units.len(), folds, None).map_err(fold_error)?;
            let groups = groups.unwrap_or_default();
            let row_fold: Vec<usize> =
                groups.iter().map(|g| units.binary_search(g).map_or(0, |u| unit_fold[u])).collect();
            Ok((0..folds)
                .map(|k| {
                    let (train, test): (Vec<usize>, Vec<usize>) =
                        (0..n).partition(|&r| row_fold[r] != k);
                    FoldSplit { train, test }
                })
                .collect())
        }
    }
}

/// Sorted distinct unit labels, refusing rows that have no usable unit structure.
fn clustered_units(groups: Option<&[u32]>, n: usize) -> Result<Vec<u32>, ModelError> {
    let Some(groups) = groups.filter(|g| g.len() == n && super::fit::group_structure_usable(g))
    else {
        return Err(ModelError::InvalidSelection {
            message: "clustered validation needs a UnitId column with at least two units and \
                      fewer units than rows"
                .into(),
        });
    };
    let mut units = groups.to_vec();
    units.sort_unstable();
    units.dedup();
    Ok(units)
}

/// Folds and seed `splits` plans for `n` rows, refusing an unsupported scheme.
pub(super) fn split_plan(
    n: usize,
    splits: ValidationSplits,
    groups: Option<&[u32]>,
) -> Result<SplitPlan, ModelError> {
    match splits {
        // Too few rows to cut is a per-family failure (a degenerate constant needs no
        // folds), not a refusal of the whole node.
        ValidationSplits::Shuffled => {
            Ok(SplitPlan { folds: MECHANISM_CV_FOLDS.min(n), seed: Some(CV_FOLD_SEED) })
        }
        ValidationSplits::TimeOrdered { .. } => {
            validation_folds(n, splits, groups, None)?;
            Ok(SplitPlan { folds: TIME_ORDERED_FOLDS, seed: None })
        }
        ValidationSplits::Clustered => {
            let units = clustered_units(groups, n)?;
            Ok(SplitPlan { folds: MECHANISM_CV_FOLDS.min(units.len()), seed: Some(CV_FOLD_SEED) })
        }
    }
}

/// Mean held-out predictive log-likelihood of `family`, refit on each training
/// fold and scored on the rows it did not see.
///
/// One criterion for every family: the log-likelihood is a density, so
/// rescaling the outcome shifts every family's score by the same `ln(scale)` and
/// cannot change which family wins. Held-out rows of a categorical outcome whose
/// level is absent from a training fold are not scored (a sampling zero, not
/// evidence against the family).
pub(super) fn oof_mean_loglik(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, ModelError> {
    let categorical = family_is_categorical(family);
    let strata = categorical.then(|| category_strata(y));
    let folds = frame.validation_folds(strata.as_deref())?;
    let p = frame.n_parents();
    let mut total = 0.0;
    let mut scored = 0usize;
    let mut last_fold_error: Option<ModelError> = None;
    for FoldSplit { train, test } in &folds {
        poll(cancelled)?;
        if train.len() < 2 || test.is_empty() {
            continue;
        }
        let y_train: Vec<f64> = train.iter().map(|&r| y[r]).collect();
        // A fold whose training rows cannot be fit (a quasi-separated multinomial, a
        // parent constant within the fold) is not scored rather than failing the family:
        // the mechanism itself is the full-data fit, which already succeeded. If no fold
        // can be scored the family is refused with the last fit error.
        let slot = match super::fit::fit_frame_family(
            family,
            &frame.select(train),
            &y_train,
            backend,
            ls_ws,
            None,
        ) {
            Ok(slot) => slot,
            Err(e) => {
                last_fold_error = Some(e);
                continue;
            }
        };
        let held_out = frame.select(test);
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
pub(super) fn oof_linear_residuals(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, ModelError> {
    let n = y.len();
    let fold = cv_fold_plan(n, None)?;
    let folds = fold.iter().copied().max().map_or(0, |m| m + 1);
    let mut residual = vec![0.0; n];
    for k in 0..folds {
        poll(cancelled)?;
        let (train, test): (Vec<usize>, Vec<usize>) = (0..n).partition(|&r| fold[r] != k);
        if train.len() < 2 || test.is_empty() {
            continue;
        }
        let y_train: Vec<f64> = train.iter().map(|&r| y[r]).collect();
        let slot = super::fit::fit_frame_family(
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
///
/// A basis expansion's own extra terms are *not* charged beyond what its
/// plain (non-expanded) counterpart already pays, for the same reason a
/// [`MechanismSlot::GaussianProcess`]'s kernel is not charged for its
/// effectively unbounded flexibility: both are scored out-of-fold, on rows
/// that never saw the expansion's fit, so the held-out log-likelihood already
/// prices however much of that flexibility is real signal versus noise.
/// Charging a further `k ln n / 2n` per expansion term double-counts that
/// cost — it stacks a training-sample complexity penalty on top of a
/// criterion whose entire purpose is to measure generalization directly —
/// and for a basis with many terms relative to `n` the stacked penalty can
/// exceed the genuine held-out log-likelihood gain of following real
/// curvature, so the expansion loses to a plainer family even when it
/// predicts held-out rows measurably better. `LinearBasis` and
/// `DiscreteBasis` are therefore charged exactly what `LinearGaussian` and
/// `Discrete` charge a fit over the same parents — the same handful of
/// nuisance parameters (an intercept and a noise scale, or one weight per
/// non-reference category) every family in the comparison carries regardless
/// of richness — so an expansion's basis size cannot tilt the comparison
/// either way; only the difference in held-out fit does.
pub(super) fn parameter_count(slot: &MechanismSlot, n_parents: usize) -> f64 {
    let p = n_parents;
    let count = match slot {
        MechanismSlot::LinearGaussian { .. }
        | MechanismSlot::HierarchicalLinear { .. }
        | MechanismSlot::Bvar { .. }
        | MechanismSlot::LinearBasis { .. }
        // An ensemble is charged what its linear counterpart is: its flexibility is
        // priced by the held-out likelihood it is scored on, not by a leaf count.
        | MechanismSlot::TreeAdditive { .. } => p + 2,
        MechanismSlot::Discrete { support, logit_coeffs, .. } => {
            (support.len().saturating_sub(1)) * if logit_coeffs.is_some() { 1 + p } else { 1 }
        }
        MechanismSlot::DiscreteBasis { support, .. }
        | MechanismSlot::TreeCategorical { support, .. } => {
            (support.len().saturating_sub(1)) * (1 + p)
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

/// Cooperative cancellation point of the polled fitting entry points.
pub(super) fn poll(cancelled: &dyn Fn() -> bool) -> Result<(), ModelError> {
    if cancelled() { Err(ModelError::Cancelled) } else { Ok(()) }
}

/// Out-of-fold score of one family on `frame`.
pub(super) fn score_family(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    cancelled: &dyn Fn() -> bool,
) -> Result<(MechanismCandidate, MechanismSlot), ModelError> {
    let fitted = super::fit::fit_frame_family(family, frame, y, backend, ls_ws, None)?;
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
    // (held-out log-likelihood, penalty actually charged); the score is their difference.
    let (held_out, charged) = match &fitted {
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
            let score = if sd > CONSTANT_FAMILY_MAX_RELATIVE_SD * max_abs {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            (score, 0.0)
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
                oof_linear_residuals(frame, y, backend, ls_ws, cancelled)?
            };
            let state = MechanismSlot::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            };
            let mut lp = vec![0.0; n];
            log_prob_column(&state, &residual, ParentBatch::empty(n), &mut lp)?;
            (lp.iter().sum::<f64>() / n.max(1) as f64, penalty)
        }
        _ => (oof_mean_loglik(family, frame, y, backend, ls_ws, cancelled)?, penalty),
    };
    Ok((
        MechanismCandidate {
            family,
            score: held_out - charged,
            held_out_log_likelihood: held_out,
            complexity_penalty: charged,
            fit_cost: 1.0 + p as f64,
            eval_cost: 1.0 + p as f64,
        },
        fitted,
    ))
}
