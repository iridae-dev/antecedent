//! Tree-ensemble mechanism fitters.
//!
//! Regression families fit `y = f(pa) + ε` with a boosted or random-forest `f` and
//! a homoscedastic Gaussian `ε` whose scale is the root-mean-square of the
//! ensemble's out-of-fold residuals: the training residuals of a tree ensemble are
//! optimistic by construction, so the noise scale is measured on rows the fit did
//! not see, split the way the selection is split. Classification families fit one
//! tree classifier per category (one for a binary outcome) and normalise.
//!
//! Providers are reached through `antecedent-learn` and exported to its
//! provider-independent tree map, so the fitted slot carries no foreign model.

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    DesignView, ForestSpec, GbtSpec, LearnError, LearnerSpec, PredictionTask, TargetView,
    resolve_for,
};

use crate::batch::ParentBatch;
use crate::compile::MechanismSlot;
use crate::error::ModelError;
use crate::tree::TreeEnsemble;

use super::{discrete_categories, discrete_support, sigma_floor};
use crate::registry::cv::FitFrame;

/// Fewest rows a tree family fits: below this an ensemble has too few rows per
/// leaf for its out-of-fold noise scale to mean anything.
const MIN_TREE_ROWS: usize = 40;

/// Seed of every ensemble fit. Fixed so a family's fit is a function of the table
/// and the declared splits alone, like the fold partition.
const TREE_FIT_SEED: u64 = 0x5452_4545_4D45_0001;

/// Boosted-tree hyperparameters of the mechanism families: shallow, moderately
/// shrunk trees, so the noise-scale estimate and the held-out score price an
/// ensemble that generalises rather than one tuned to its training rows.
const BOOSTED: GbtSpec = GbtSpec { trees: 100, depth: 3, learning_rate: 0.1 };

/// Forest hyperparameters of the regression family: leaves of at least five rows.
const FOREST_REGRESSION: ForestSpec =
    ForestSpec { extra_trees: false, n_trees: 100, min_samples_leaf: Some(5) };

/// Forest hyperparameters of the classification family: the probability task's
/// own leaf default keeps leaf means off 0 and 1.
const FOREST_CLASSIFICATION: ForestSpec =
    ForestSpec { extra_trees: false, n_trees: 100, min_samples_leaf: None };

/// Ensemble provider of a tree family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::registry) enum TreeLearner {
    /// Gradient-boosted trees.
    Boosted,
    /// Random forest.
    Forest,
}

impl TreeLearner {
    fn spec(self, task: PredictionTask) -> LearnerSpec {
        match (self, task) {
            (Self::Boosted, _) => LearnerSpec::GradientBoostedTrees(BOOSTED),
            (Self::Forest, PredictionTask::Regression) => {
                LearnerSpec::RandomForest(FOREST_REGRESSION)
            }
            (Self::Forest, PredictionTask::BinaryProbability) => {
                LearnerSpec::RandomForest(FOREST_CLASSIFICATION)
            }
        }
    }
}

fn learn_error(error: LearnError) -> ModelError {
    match error {
        LearnError::ProviderUnavailable { spec, .. } => ModelError::Unsupported {
            message: format!(
                "tree-ensemble provider `{spec}` is not compiled in (enable the `ml-gbdt` / \
                 `ml-forest` feature)"
            ),
        },
        other => ModelError::Unsupported { message: format!("tree ensemble fit: {other}") },
    }
}

/// Fit one ensemble on a column-major design of `n` rows and `p` columns.
fn fit_ensemble(
    learner: TreeLearner,
    task: PredictionTask,
    design: &[f64],
    n: usize,
    p: usize,
    target: &[f64],
) -> Result<TreeEnsemble, ModelError> {
    let factory = resolve_for(learner.spec(task), task).map_err(learn_error)?;
    let view = DesignView::from_column_major(design, n, p).map_err(learn_error)?;
    // One thread: the fit is a pure function of the table, never of a scheduler.
    let ctx = ExecutionContext::production(TREE_FIT_SEED, 1);
    let fitted = factory.fit(view, TargetView::new(target), None, &ctx).map_err(learn_error)?;
    let portable = fitted.portable().map_err(learn_error)?;
    TreeEnsemble::new(portable)
}

/// Refuse inputs a tree family cannot honour, before any fit.
fn check_inputs(frame: &FitFrame<'_>, y: &[f64]) -> Result<(), ModelError> {
    if frame.n_parents() == 0 {
        return Err(ModelError::Unsupported {
            message: "a tree-ensemble mechanism needs at least one parent".into(),
        });
    }
    if y.len() < MIN_TREE_ROWS {
        return Err(ModelError::Unsupported {
            message: format!(
                "a tree-ensemble mechanism needs at least {MIN_TREE_ROWS} rows, got {}",
                y.len()
            ),
        });
    }
    let finite = y.iter().all(|v| v.is_finite())
        && frame.parents.iter().all(|c| c[..frame.n].iter().all(|v| v.is_finite()));
    if !finite {
        return Err(ModelError::Unsupported {
            message: "a tree-ensemble mechanism needs finite outcome and parent values".into(),
        });
    }
    Ok(())
}

/// Additive-noise regression on a tree ensemble.
pub(in crate::registry) fn fit_tree_additive(
    learner: TreeLearner,
    frame: &FitFrame<'_>,
    y: &[f64],
) -> Result<MechanismSlot, ModelError> {
    check_inputs(frame, y)?;
    let p = frame.n_parents();
    let n = y.len();
    let design = frame.column_major();
    let ensemble = fit_ensemble(learner, PredictionTask::Regression, &design, n, p, y)?;

    // Out-of-fold residuals under the frame's declared split scheme.
    let mut sum_sq = 0.0;
    let mut scored = 0usize;
    for fold in frame.validation_folds(None)? {
        if fold.train.len() < MIN_TREE_ROWS / 2 || fold.test.is_empty() {
            continue;
        }
        let train = frame.select(&fold.train);
        let y_train: Vec<f64> = fold.train.iter().map(|&r| y[r]).collect();
        let fold_ensemble = fit_ensemble(
            learner,
            PredictionTask::Regression,
            &train.column_major(),
            fold.train.len(),
            p,
            &y_train,
        )?;
        let held_out = frame.select(&fold.test);
        let values = held_out.column_major();
        let mut mean = vec![0.0; fold.test.len()];
        fold_ensemble.predict_column(
            ParentBatch { n_rows: fold.test.len(), n_parents: p, values: &values },
            &mut mean,
        )?;
        for (&r, m) in fold.test.iter().zip(&mean) {
            sum_sq += (y[r] - m).powi(2);
            scored += 1;
        }
    }
    if scored == 0 {
        return Err(ModelError::Unsupported {
            message: "no out-of-fold residuals to scale the tree-ensemble noise".into(),
        });
    }
    let sigma = (sum_sq / scored as f64).sqrt().max(sigma_floor(y));
    Ok(MechanismSlot::TreeAdditive { ensemble, sigma })
}

/// Categorical mechanism with tree-classifier class probabilities.
pub(in crate::registry) fn fit_tree_categorical(
    learner: TreeLearner,
    frame: &FitFrame<'_>,
    y: &[f64],
) -> Result<MechanismSlot, ModelError> {
    check_inputs(frame, y)?;
    let (support, probs) = discrete_support(y, None)?;
    let k = support.len();
    if k < 2 {
        return Err(ModelError::Unsupported {
            message: "a tree classifier needs at least two outcome levels".into(),
        });
    }
    let category = discrete_categories(y, &support)?;
    let p = frame.n_parents();
    let n = y.len();
    let design = frame.column_major();
    // A binary outcome needs one classifier for the upper level; more levels use one
    // classifier per level against the rest.
    let levels: Vec<usize> = if k == 2 { vec![1] } else { (0..k).collect() };
    let mut classifiers = Vec::with_capacity(levels.len());
    for level in levels {
        let target: Vec<f64> = category
            .iter()
            .map(|&c| if usize::try_from(c).is_ok_and(|c| c == level) { 1.0 } else { 0.0 })
            .collect();
        classifiers.push(fit_ensemble(
            learner,
            PredictionTask::BinaryProbability,
            &design,
            n,
            p,
            &target,
        )?);
    }
    Ok(MechanismSlot::TreeCategorical {
        support: support.into(),
        probs: probs.into(),
        classifiers: classifiers.into(),
    })
}
