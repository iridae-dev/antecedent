//! Forust GBDT adapter. Hidden behind [`crate::LearnerSpec::GradientBoostedTrees`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster;
use forust_ml::objective::ObjectiveType;

use crate::dense::{gather_physical, materialize_dense_colmajor, require_binary_labels};
use crate::design::{DesignView, Layout, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::GbtSpec;

/// Most cells (rows times features) of the design one boosted fit admits.
///
/// Forust bins and copies the design, so the fit's working set grows with this product. The
/// cap bounds the input the uninterruptible fit is allowed to see; the host's memory budget
/// bounds it further through admission.
const MAX_GBT_DESIGN_CELLS: u64 = 1 << 31;

/// Bytes charged per design cell for the binned copy Forust builds.
const GBT_BINNED_BYTES_PER_CELL: u64 = 4;

/// Bytes charged per row for the gradient, hessian, prediction, weight, target and index
/// vectors a boosting round keeps.
const GBT_BYTES_PER_ROW: u64 = 64;

/// Bytes charged per tree node: split, children, weight and bookkeeping.
const GBT_BYTES_PER_NODE: u64 = 128;

/// Refuse a design above [`MAX_GBT_DESIGN_CELLS`] before anything is sized from it.
fn check_design_cells(rows: usize, cols: usize) -> Result<u64, LearnError> {
    let cells = u64::try_from(rows)
        .ok()
        .zip(u64::try_from(cols).ok())
        .and_then(|(r, c)| r.checked_mul(c))
        .ok_or(LearnError::ResourceLimit {
            message: "gradient-boosted trees design size overflows the cell count",
        })?;
    if cells > MAX_GBT_DESIGN_CELLS {
        return Err(LearnError::ResourceLimit {
            message: "gradient-boosted trees design above the cell limit",
        });
    }
    Ok(cells)
}

/// Bytes one boosted fit holds: the design copy when the view must be gathered, Forust's
/// binned copy, the per-row vectors and the trees, each bounded by the rows it can split.
fn gbt_working_bytes(
    cells: u64,
    rows: usize,
    copied_design: bool,
    spec: GbtSpec,
) -> Result<u64, LearnError> {
    let overflow =
        LearnError::ResourceLimit { message: "gradient-boosted trees working set overflows" };
    let rows = u64::try_from(rows).map_err(|_| overflow.clone())?;
    // `validate` bounded the depth, so the shift cannot overflow.
    let full_tree = (1_u64 << (spec.depth + 1)) - 1;
    let nodes_per_tree = full_tree.min(rows.saturating_mul(2).saturating_add(1));
    let design_copy = if copied_design { cells.checked_mul(8) } else { Some(0) };
    let binned = cells.checked_mul(GBT_BINNED_BYTES_PER_CELL);
    let vectors = rows.checked_mul(GBT_BYTES_PER_ROW);
    let trees = u64::from(spec.trees)
        .checked_mul(nodes_per_tree)
        .and_then(|nodes| nodes.checked_mul(GBT_BYTES_PER_NODE));
    [design_copy, binned, vectors, trees]
        .into_iter()
        .try_fold(0_u64, |sum, part| part.and_then(|bytes| sum.checked_add(bytes)))
        .ok_or(overflow)
}

/// Forust gradient-boosted trees.
///
/// # Cancellation
///
/// Forust offers no interrupt, so a started fit cannot be cancelled and runs to completion.
/// [`LearnerFactory::fit`] therefore refuses a spec above [`crate::MAX_GBT_TREES`] trees or
/// [`crate::MAX_GBT_DEPTH`] levels and a design above the cell limit, admits the working set it
/// estimates from rows, features, trees and depth against the context's memory budget before the
/// design is copied, and polls cancellation before it starts and after Forust returns.
#[derive(Clone, Copy, Debug)]
pub struct GbtLearner {
    spec: GbtSpec,
    task: PredictionTask,
}

impl GbtLearner {
    /// GBDT factory for a public spec and prediction task.
    #[must_use]
    pub const fn new(spec: GbtSpec, task: PredictionTask) -> Self {
        Self { spec, task }
    }
}

impl LearnerFactory for GbtLearner {
    fn task(&self) -> PredictionTask {
        self.task
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
            binary_probability: true,
            sample_weights: true,
            missing_values: true,
            deterministic_seed: true,
            parallelism_control: true,
            ..LearnerCapabilities::none()
        }
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        crate::spec::LearnerSpec::GradientBoostedTrees(self.spec).validate()?;
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        // Admitted before the design is copied; held until the fit returns. Forust cannot be
        // interrupted once `fit` starts below, so this and the input caps are its only bound.
        let cells = check_design_cells(x.nrows(), x.ncols())?;
        let copied_design =
            x.row_selection().is_some() || x.dense().layout() != Layout::ColumnMajor;
        let bytes = gbt_working_bytes(cells, x.nrows(), copied_design, self.spec)?;
        let _admission = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => LearnError::Cancelled,
            other => LearnError::Admission(other),
        })?;
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        if matches!(self.task, PredictionTask::BinaryProbability) {
            require_binary_labels(&gathered_y)?;
        }
        let matrix = Matrix::new(&design, nrows, ncols);
        // Forust exposes a global-pool boolean, not a bounded thread lease.
        // Outer folds own the ExecutionContext parallelism budget.
        let parallel = false;
        let mut seed_rng = ctx.rng.stream_for(StreamDomain::Learner, crate::streams::GBT);
        let objective = match self.task {
            PredictionTask::Regression => ObjectiveType::SquaredLoss,
            PredictionTask::BinaryProbability => ObjectiveType::LogLoss,
        };
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the booster backend takes its learning rate as f32, and a rate in (0, 1] loses only sub-ulp precision"
        )]
        let mut model = GradientBooster::default()
            .set_objective_type(objective)
            .set_iterations(self.spec.trees as usize)
            .set_max_depth(self.spec.depth as usize)
            .set_learning_rate(self.spec.learning_rate as f32)
            .set_parallel(parallel)
            .set_seed(seed_rng.next_u64());
        if let Some(w) = weights {
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, x.nrows())?;
            if gathered_w.iter().any(|v| !v.is_finite() || *v < 0.0)
                || !gathered_w.iter().any(|v| *v > 0.0)
                || !gathered_w.iter().sum::<f64>().is_finite()
            {
                return Err(LearnError::Shape {
                    message: "prior weights must be finite, non-negative and not all zero",
                });
            }
            model.fit(&matrix, &gathered_y, &gathered_w, None).map_err(|err| forust_err(&err))?;
        } else {
            model.fit_unweighted(&matrix, &gathered_y, None).map_err(|err| forust_err(&err))?;
        }
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        Ok(Box::new(GbtPredictor { model, task: self.task, columns: x.ncols() }))
    }
}

struct GbtPredictor {
    columns: usize,
    model: GradientBooster,
    task: PredictionTask,
}

impl FittedPredictor for GbtPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        if out.len() != x.nrows() {
            return Err(LearnError::Shape { message: "predict out length != logical rows" });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let matrix = Matrix::new(&design, nrows, ncols);
        // Forust exposes a global-pool boolean, not a bounded thread lease.
        // Outer folds own the ExecutionContext parallelism budget.
        let parallel = false;
        let preds = self.model.predict(&matrix, parallel);
        if preds.len() != out.len() {
            return Err(LearnError::Shape { message: "forust predict length mismatch" });
        }
        match self.task {
            PredictionTask::Regression => out.copy_from_slice(&preds),
            PredictionTask::BinaryProbability => {
                for (slot, raw) in out.iter_mut().zip(preds) {
                    *slot = crate::learner::clip_probability(antecedent_kernels::sigmoid(raw));
                }
            }
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        use crate::{PortablePredictor, PredictionMap, PredictionNode};
        if !self.model.missing.is_nan() {
            return Err(LearnError::Unsupported {
                message: "non-NaN missing sentinel has no portable codec",
            });
        }
        let trees = self
            .model
            .trees
            .iter()
            .map(|tree| {
                tree.nodes
                    .iter()
                    .map(|n| {
                        if n.is_leaf {
                            PredictionNode::Leaf { value: Some(f64::from(n.weight_value)) }
                        } else {
                            PredictionNode::Split {
                                feature: n.split_feature,
                                threshold: n.split_value,
                                inclusive: false,
                                left: n.left_child,
                                right: n.right_child,
                                missing: n.missing_node,
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let result = PortablePredictor {
            version: 1,
            columns: self.columns,
            provenance: self.provenance(),
            model: PredictionMap::Trees {
                trees,
                base: self.model.base_score,
                average: false,
                logistic: self.task == PredictionTask::BinaryProbability,
                probability: false,
            },
        };
        result.validate()?;
        Ok(result)
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "gradient_boosted_trees".into(),
            implementation: "forust-ml".into(),
            version: "0.5".into(),
        }
    }
}

fn forust_err(err: &forust_ml::errors::ForustError) -> LearnError {
    LearnError::Backend(format!("{err}"))
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    fn fixture() -> (Vec<f64>, Vec<f64>) {
        ((0..8_i32).map(f64::from).collect(), (0..8_i32).map(f64::from).collect())
    }

    #[test]
    fn design_cells_refuse_one_past_the_bound() {
        let bound = usize::try_from(MAX_GBT_DESIGN_CELLS).unwrap();
        assert_eq!(check_design_cells(bound, 1).unwrap(), MAX_GBT_DESIGN_CELLS);
        assert!(matches!(check_design_cells(bound + 1, 1), Err(LearnError::ResourceLimit { .. })));
        assert!(matches!(check_design_cells(usize::MAX, 2), Err(LearnError::ResourceLimit { .. })));
    }

    #[test]
    fn fit_refuses_specs_above_the_tree_and_depth_limits() {
        let (x, y) = fixture();
        let view = DesignView::from_column_major(&x, 8, 1).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let too_many = GbtSpec { trees: crate::MAX_GBT_TREES + 1, depth: 2, learning_rate: 0.1 };
        let too_deep = GbtSpec { trees: 2, depth: crate::MAX_GBT_DEPTH + 1, learning_rate: 0.1 };
        for spec in [too_many, too_deep] {
            let learner = GbtLearner::new(spec, PredictionTask::Regression);
            assert!(matches!(
                learner.fit(view, TargetView::new(&y), None, &ctx),
                Err(LearnError::ResourceLimit { .. })
            ));
        }
    }

    #[test]
    fn fit_is_admitted_against_the_memory_budget_and_polls_cancellation() {
        let (x, y) = fixture();
        let view = DesignView::from_column_major(&x, 8, 1).unwrap();
        let learner = GbtLearner::new(
            GbtSpec { trees: 2, depth: 2, learning_rate: 0.1 },
            PredictionTask::Regression,
        );
        let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(1));
        assert!(matches!(
            learner.fit(view, TargetView::new(&y), None, &limited),
            Err(LearnError::Admission(
                antecedent_core::execution::AdmissionError::ExceedsRemainder { .. }
            ))
        ));
        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        assert!(matches!(
            learner.fit(view, TargetView::new(&y), None, &cancelled),
            Err(LearnError::Cancelled)
        ));
        let roomy = ExecutionContext::for_tests(1).with_memory_limit(Some(1 << 20));
        assert!(learner.fit(view, TargetView::new(&y), None, &roomy).is_ok());
    }

    #[test]
    fn working_bytes_grow_with_trees_depth_and_rows() {
        let spec = GbtSpec { trees: 10, depth: 3, learning_rate: 0.1 };
        let base = gbt_working_bytes(1_000, 1_000, false, spec).unwrap();
        let more_trees = GbtSpec { trees: 20, ..spec };
        assert!(gbt_working_bytes(1_000, 1_000, false, more_trees).unwrap() > base);
        let deeper = GbtSpec { depth: 4, ..spec };
        assert!(gbt_working_bytes(1_000, 1_000, false, deeper).unwrap() > base);
        assert!(gbt_working_bytes(1_000, 1_000, true, spec).unwrap() > base);
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use crate::design::RowSelection;

    #[test]
    fn gbt_checks_only_logical_weights_before_fit() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 2.0, 3.0];
        let view = DesignView::from_column_major(&x, 4, 1).unwrap();
        let learner = GbtLearner::new(
            GbtSpec { trees: 2, depth: 2, learning_rate: 0.1 },
            PredictionTask::Regression,
        );
        let ctx = ExecutionContext::for_tests(1);
        for weights in [
            [-1.0, 1.0, 1.0, 1.0],
            [f64::NAN, 1.0, 1.0, 1.0],
            [f64::INFINITY, 1.0, 1.0, 1.0],
            [0.0; 4],
            [f64::MAX; 4],
        ] {
            assert!(matches!(
                learner.fit(view, TargetView::new(&y), Some(&weights), &ctx),
                Err(LearnError::Shape { .. })
            ));
        }
        let rows = [1, 2, 3];
        let selected = view.with_rows(RowSelection::new(&rows)).unwrap();
        assert!(
            learner
                .fit(selected, TargetView::new(&y), Some(&[f64::NAN, 1.0, 0.0, 1.0]), &ctx)
                .is_ok()
        );
    }
}
