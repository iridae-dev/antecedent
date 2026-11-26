//! Random-forest learners behind [`crate::LearnerSpec::RandomForest`].
//!
//! [`ForestLearner::fit`] grows each tree on its own derived stream under
//! [`ExecutionContext::map_indexed_with_spare_workers`] and records [`PER_TREE_FOREST`].
//! [`ForestLearner::fit_smartcore`] is the serial SmartCore 0.4 fit: one seed,
//! every tree on the calling worker.
//!
//! SmartCore's loop cannot be opened in front of that seed. Bootstrap draws
//! advance one generator, then each tree draws its splits from a fresh
//! generator reseeded with the same seed. The per-tree fitter is private to
//! the provider, and the number of split draws depends on the grown tree, so
//! those draws cannot be recorded and handed back through the public `fit`.
//! A one-tree forest on a derived seed would reseed the splits together with
//! the bootstrap.
//!
//! Cross-fit trees use the fold's worker and any spare slots the fold can lease
//! without waiting. A top-level fit leases one worker per tree up to the context's
//! thread budget.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution::{AdmissionError, MapExecutionError};
use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, derive_stream};
use smartcore::ensemble::extra_trees_regressor::{
    ExtraTreesRegressor, ExtraTreesRegressorParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::dense::{gather_physical, materialize_dense_colmajor, require_binary_labels};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::ForestSpec;
use crate::{PortablePredictor, PredictionMap, PredictionNode};

/// Provenance implementation of [`ForestLearner::fit`].
///
/// Receipts for that path name this grower. [`ForestLearner::fit_smartcore`]
/// still records `smartcore` / `0.4`.
pub const PER_TREE_FOREST: &str = "antecedent.per_tree_forest";

/// Random forest / extra-trees factory.
#[derive(Clone, Copy, Debug)]
pub struct ForestLearner {
    spec: ForestSpec,
    task: PredictionTask,
}

impl ForestLearner {
    /// Forest factory for a public spec and prediction task.
    #[must_use]
    pub const fn new(spec: ForestSpec, task: PredictionTask) -> Self {
        Self { spec, task }
    }

    /// Serial SmartCore fit on one seed. Every tree grows on the calling worker.
    ///
    /// # Errors
    ///
    /// Same shape, task, and provider failures as the historical `fit`.
    pub fn fit_smartcore(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let prepared = self.prepare(x, y, weights)?;
        let mut rng = ctx.rng.stream_for(StreamDomain::Learner, crate::streams::FOREST);
        let seed = rng.next_u64();
        let n_trees = self.spec.n_trees as usize;
        let leaf = self.spec.resolved_min_samples_leaf(self.task) as usize;
        let matrix = dense_column_major(prepared.design, prepared.nrows, prepared.ncols)?;
        let model = if self.spec.extra_trees {
            FittedForest::Extra(
                ExtraTreesRegressor::fit(
                    &matrix,
                    &prepared.y,
                    ExtraTreesRegressorParameters::default()
                        .with_seed(seed)
                        .with_n_trees(n_trees)
                        .with_min_samples_leaf(leaf),
                )
                .map_err(|e| LearnError::Backend(e.to_string()))?,
            )
        } else {
            FittedForest::Rf(
                RandomForestRegressor::fit(
                    &matrix,
                    &prepared.y,
                    RandomForestRegressorParameters::default()
                        .with_seed(seed)
                        .with_n_trees(n_trees)
                        .with_min_samples_leaf(leaf),
                )
                .map_err(|e| LearnError::Backend(e.to_string()))?,
            )
        };
        Ok(Box::new(FittedForestPredictor { model, task: self.task, columns: prepared.ncols }))
    }

    fn prepare<'a>(
        &self,
        x: DesignView<'a>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
    ) -> Result<Prepared<'a>, LearnError> {
        if weights.is_some() {
            return Err(LearnError::Unsupported {
                message: "random forest does not accept sample weights",
            });
        }
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        crate::spec::LearnerSpec::RandomForest(self.spec).validate()?;
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if nrows == 0 || ncols == 0 {
            return Err(LearnError::Shape { message: "forest design must be non-empty" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        if matches!(self.task, PredictionTask::BinaryProbability) {
            require_binary_labels(&gathered_y)?;
        }
        if gathered_y.iter().any(|value| !value.is_finite()) {
            return Err(LearnError::NonFinite { message: "forest target" });
        }
        Ok(Prepared { design, y: gathered_y, nrows, ncols })
    }

    fn fit_per_tree(
        &self,
        prepared: &Prepared<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let n_trees = self.spec.n_trees as usize;
        let min_leaf = self.spec.resolved_min_samples_leaf(self.task) as usize;
        let grow = Grow {
            design: &prepared.design,
            y: &prepared.y,
            nrows: prepared.nrows,
            ncols: prepared.ncols,
            min_leaf,
            random_splits: self.spec.extra_trees,
        };
        #[cfg(test)]
        let recorder = tests::recording();
        #[cfg(test)]
        let fit_thread = std::thread::current().id();
        let trees = ctx
            .map_indexed_with_spare_workers(
                n_trees,
                |index, inner| -> Result<Vec<PredictionNode>, LearnError> {
                    if inner.cancellation.is_cancelled() {
                        return Err(LearnError::Cancelled);
                    }
                    #[cfg(test)]
                    if let Some(recorder) = &recorder {
                        recorder.note_worker(fit_thread, inner);
                    }
                    let mut rng = inner.rng.stream_for(
                        StreamDomain::Learner,
                        derive_stream(crate::streams::FOREST, index as u64),
                    );
                    Ok(grow_tree(&grow, &mut rng))
                },
            )
            .map_err(map_forest_error)?;
        let model = assemble_forest(trees, prepared.ncols, self.task)?;
        Ok(Box::new(GrownForest { model }))
    }
}

impl LearnerFactory for ForestLearner {
    fn task(&self) -> PredictionTask {
        self.task
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
            binary_probability: true,
            deterministic_seed: true,
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
        let prepared = self.prepare(x, y, weights)?;
        self.fit_per_tree(&prepared, ctx)
    }
}

struct Prepared<'a> {
    design: std::borrow::Cow<'a, [f64]>,
    y: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

struct Grow<'a> {
    design: &'a [f64],
    y: &'a [f64],
    nrows: usize,
    ncols: usize,
    min_leaf: usize,
    /// Extra-trees: every row once, one random cut per chosen feature.
    /// A random forest bootstraps and searches the best cut on those features.
    random_splits: bool,
}

struct Split {
    feature: usize,
    threshold: f64,
}

fn map_forest_error(error: MapExecutionError<LearnError>) -> LearnError {
    match error {
        MapExecutionError::Work(error) => error,
        MapExecutionError::Admission(AdmissionError::Cancelled) => LearnError::Cancelled,
        MapExecutionError::Admission(error) => LearnError::Admission(error),
    }
}

fn assemble_forest(
    trees: Vec<Vec<PredictionNode>>,
    columns: usize,
    task: PredictionTask,
) -> Result<PortablePredictor, LearnError> {
    // A probability task clips the averaged leaf means into (0, 1) at prediction.
    // The task default leaf size keeps most leaves off 0 and 1; the clip covers
    // a leaf that is still a single class.
    let model = PortablePredictor {
        version: 1,
        columns,
        provenance: LearnerProvenance {
            spec: "random_forest".into(),
            implementation: PER_TREE_FOREST.into(),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        model: PredictionMap::Trees {
            trees,
            base: 0.0,
            average: true,
            logistic: false,
            probability: task == PredictionTask::BinaryProbability,
        },
    };
    model.validate()?;
    Ok(model)
}

fn grow_tree(data: &Grow<'_>, rng: &mut CausalRng) -> Vec<PredictionNode> {
    let sample = sample_rows(data.nrows, !data.random_splits, rng);
    let mut nodes = vec![leaf_node(mean_of(&sample, data.y))];
    let mut stack = vec![Work { index: 0, rows: sample }];
    while let Some(work) = stack.pop() {
        let Some(split) = find_split(data, &work.rows, rng) else {
            nodes[work.index] = leaf_node(mean_of(&work.rows, data.y));
            continue;
        };
        let (left_rows, right_rows) = partition_rows(data, &work.rows, &split);
        if left_rows.is_empty() || right_rows.is_empty() {
            nodes[work.index] = leaf_node(mean_of(&work.rows, data.y));
            continue;
        }
        let left = nodes.len();
        nodes.push(leaf_node(0.0));
        let right = nodes.len();
        nodes.push(leaf_node(0.0));
        nodes[work.index] = PredictionNode::Split {
            feature: split.feature,
            threshold: split.threshold,
            inclusive: true,
            left,
            right,
            missing: right,
        };
        stack.push(Work { index: right, rows: right_rows });
        stack.push(Work { index: left, rows: left_rows });
    }
    nodes
}

struct Work {
    index: usize,
    rows: Vec<usize>,
}

fn leaf_node(value: f64) -> PredictionNode {
    PredictionNode::Leaf { value: Some(value) }
}

fn mean_of(rows: &[usize], y: &[f64]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().map(|&row| y[row]).sum::<f64>() / rows.len() as f64
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the draw is reduced modulo the row count, and that count fits usize"
)]
fn sample_rows(nrows: usize, bootstrap: bool, rng: &mut CausalRng) -> Vec<usize> {
    if !bootstrap || nrows == 0 {
        return (0..nrows).collect();
    }
    (0..nrows).map(|_| (rng.next_u64() as usize) % nrows).collect()
}

fn find_split(data: &Grow<'_>, rows: &[usize], rng: &mut CausalRng) -> Option<Split> {
    if data.min_leaf == 0 || rows.len() < data.min_leaf.saturating_mul(2) || data.ncols == 0 {
        return None;
    }
    let features = choose_features(data.ncols, rng);
    let mut best: Option<(Split, f64)> = None;
    for feature in features {
        let Some((threshold, score)) = candidate_cut(data, rows, feature, rng) else {
            continue;
        };
        if best.as_ref().is_none_or(|(_, kept)| score > *kept) {
            best = Some((Split { feature, threshold }, score));
        }
    }
    best.map(|(split, _)| split)
}

fn candidate_cut(
    data: &Grow<'_>,
    rows: &[usize],
    feature: usize,
    rng: &mut CausalRng,
) -> Option<(f64, f64)> {
    if data.random_splits {
        random_cut(data, rows, feature, rng)
    } else {
        best_cut(data, rows, feature)
    }
}

#[allow(
    clippy::float_cmp,
    reason = "finite split values use the same numerical equality as the partition"
)]
fn best_cut(data: &Grow<'_>, rows: &[usize], feature: usize) -> Option<(f64, f64)> {
    let column = data.column(feature);
    let mut finite: Vec<usize> =
        rows.iter().copied().filter(|&row| column[row].is_finite()).collect();
    finite.sort_by(|&a, &b| column[a].total_cmp(&column[b]));
    let n = rows.len();
    let total: f64 = rows.iter().map(|&row| data.y[row]).sum();
    let parent = total * total / n as f64;
    let mut left_sum = 0.0;
    let mut left_n = 0usize;
    let mut best: Option<(f64, f64)> = None;
    let mut i = 0;
    while i < finite.len() {
        let value = column[finite[i]];
        let mut j = i;
        while j < finite.len() && column[finite[j]] == value {
            left_sum += data.y[finite[j]];
            left_n += 1;
            j += 1;
        }
        let right_n = n - left_n;
        if left_n >= data.min_leaf && right_n >= data.min_leaf {
            let score = side_score(left_n, left_sum, right_n, total - left_sum);
            if score > parent && best.as_ref().is_none_or(|(_, kept)| score > *kept) {
                let threshold = if j < finite.len() {
                    separating_threshold(value, column[finite[j]])
                } else {
                    value
                };
                best = Some((threshold, score));
            }
        }
        i = j;
    }
    best
}

fn random_cut(
    data: &Grow<'_>,
    rows: &[usize],
    feature: usize,
    rng: &mut CausalRng,
) -> Option<(f64, f64)> {
    let column = data.column(feature);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &row in rows {
        let value = column[row];
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min >= max {
        return None;
    }
    let threshold = min + (max - min) * rng.next_f64();
    let score = split_score(data, rows, column, threshold)?;
    Some((threshold, score))
}

fn split_score(data: &Grow<'_>, rows: &[usize], column: &[f64], threshold: f64) -> Option<f64> {
    let n = rows.len();
    if n == 0 || data.min_leaf == 0 {
        return None;
    }
    let mut total = 0.0;
    let mut left_sum = 0.0;
    let mut left_n = 0usize;
    for &row in rows {
        let response = data.y[row];
        total += response;
        let value = column[row];
        if value.is_finite() && value <= threshold {
            left_sum += response;
            left_n += 1;
        }
    }
    let right_n = n - left_n;
    if left_n < data.min_leaf || right_n < data.min_leaf {
        return None;
    }
    let score = side_score(left_n, left_sum, right_n, total - left_sum);
    let parent = total * total / n as f64;
    (score > parent).then_some(score)
}

fn side_score(left_n: usize, left_sum: f64, right_n: usize, right_sum: f64) -> f64 {
    left_n as f64 * (left_sum / left_n as f64).powi(2)
        + right_n as f64 * (right_sum / right_n as f64).powi(2)
}

const fn separating_threshold(left_max: f64, right_min: f64) -> f64 {
    let mid = left_max.midpoint(right_min);
    if mid >= right_min { left_max } else { mid }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the draw is reduced modulo the remaining feature count, which fits usize"
)]
fn choose_features(p: usize, rng: &mut CausalRng) -> Vec<usize> {
    let mut features: Vec<usize> = (0..p).collect();
    let mtry = mtry(p);
    if mtry >= p {
        return features;
    }
    for i in 0..mtry {
        let rest = p - i;
        let j = i + (rng.next_u64() as usize % rest);
        features.swap(i, j);
    }
    features.truncate(mtry);
    features
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "floor(sqrt(p)) of a feature count is a non-negative integer no larger than p"
)]
fn mtry(p: usize) -> usize {
    if p <= 1 {
        return p;
    }
    let drawn = (p as f64).sqrt().floor() as usize;
    drawn.clamp(1, p)
}

impl Grow<'_> {
    fn column(&self, feature: usize) -> &[f64] {
        let start = feature * self.nrows;
        &self.design[start..start + self.nrows]
    }
}

fn partition_rows(data: &Grow<'_>, rows: &[usize], split: &Split) -> (Vec<usize>, Vec<usize>) {
    let column = data.column(split.feature);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &row in rows {
        let value = column[row];
        if value.is_finite() && value <= split.threshold {
            left.push(row);
        } else {
            right.push(row);
        }
    }
    (left, right)
}

enum FittedForest {
    Rf(RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    Extra(ExtraTreesRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
}

struct FittedForestPredictor {
    columns: usize,
    model: FittedForest,
    task: PredictionTask,
}

impl FittedPredictor for FittedForestPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if out.len() != nrows {
            return Err(LearnError::Shape { message: "predict out length != logical rows" });
        }
        let matrix = dense_column_major(design, nrows, ncols)?;
        let preds = match &self.model {
            FittedForest::Rf(m) => {
                m.predict(&matrix).map_err(|e| LearnError::Backend(e.to_string()))?
            }
            FittedForest::Extra(m) => {
                m.predict(&matrix).map_err(|e| LearnError::Backend(e.to_string()))?
            }
        };
        if preds.len() != out.len() {
            return Err(LearnError::Shape { message: "forest predict length mismatch" });
        }
        match self.task {
            PredictionTask::Regression => out.copy_from_slice(&preds),
            // Leaf means of a 0/1 outcome estimate E[T|X] = P(T=1|X). This is not a log-loss
            // classifier. With fully grown trees (leaf size 1) out-of-fold means are often
            // exactly 0 or 1 and this clamp would become the operative propensity (weights up
            // to 1e9); the probability task therefore defaults to a leaf size of
            // `FOREST_PROBABILITY_MIN_LEAF`, which keeps leaf means off the boundary. The clamp
            // only guards what remains: floating error and single-class leaves.
            PredictionTask::BinaryProbability => {
                for (slot, p) in out.iter_mut().zip(preds) {
                    *slot = crate::learner::clip_probability(p);
                }
            }
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        #[derive(serde::Deserialize)]
        struct Forest {
            trees: Vec<Tree>,
        }
        #[derive(serde::Deserialize)]
        struct Tree {
            nodes: Vec<Node>,
        }
        #[derive(serde::Deserialize)]
        struct Node {
            output: f64,
            split_feature: usize,
            split_value: Option<f64>,
            true_child: Option<usize>,
            false_child: Option<usize>,
        }
        // Read the pinned provider's serde output only during export. Portable readers
        // never deserialize a foreign model or execute its unchecked tree traversal.
        let json = match &self.model {
            FittedForest::Rf(model) => serde_json::to_value(model),
            FittedForest::Extra(model) => serde_json::to_value(model),
        }
        .map_err(|e| LearnError::Backend(e.to_string()))?;
        let forest: Forest = serde_json::from_value(json["forest_regressor"].clone())
            .map_err(|e| LearnError::Backend(e.to_string()))?;
        let trees = forest
            .trees
            .into_iter()
            .map(|tree| {
                tree.nodes
                    .into_iter()
                    .map(|n| match (n.true_child, n.false_child, n.split_value) {
                        (Some(left), Some(right), Some(threshold)) => Ok(PredictionNode::Split {
                            feature: n.split_feature,
                            threshold,
                            inclusive: true,
                            left,
                            right,
                            missing: right,
                        }),
                        (None, None, _) => Ok(PredictionNode::Leaf { value: Some(n.output) }),
                        _ => Err(LearnError::Shape { message: "incomplete provider tree" }),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = PortablePredictor {
            version: 1,
            columns: self.columns,
            provenance: self.provenance(),
            model: PredictionMap::Trees {
                trees,
                base: 0.0,
                average: true,
                logistic: false,
                probability: self.task == PredictionTask::BinaryProbability,
            },
        };
        result.validate()?;
        Ok(result)
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "random_forest".into(),
            implementation: "smartcore".into(),
            version: "0.4".into(),
        }
    }
}

struct GrownForest {
    model: PortablePredictor,
}

impl FittedPredictor for GrownForest {
    fn portable(&self) -> Result<PortablePredictor, LearnError> {
        self.model.validate()?;
        Ok(self.model.clone())
    }

    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        self.model.predict(x, out, ctx)
    }

    fn provenance(&self) -> LearnerProvenance {
        self.model.provenance()
    }
}

/// The legacy SmartCore path requires an owned, mutable `Array2` type at fit and predict.
/// Move a gathered design into that matrix or copy a borrowed input for this path.
fn dense_column_major(
    colmajor: std::borrow::Cow<'_, [f64]>,
    nrows: usize,
    ncols: usize,
) -> Result<DenseMatrix<f64>, LearnError> {
    // `from_2d_vec` rebuilt this column-major buffer as rows and then copied it back.
    // `DenseMatrix::new(..., column_major: true)` keeps the same layout and values.
    let cells = nrows.saturating_mul(ncols);
    if nrows == 0 || ncols == 0 || colmajor.len() < cells {
        return Err(LearnError::Shape { message: "forest design must be non-empty" });
    }
    let values = match colmajor {
        std::borrow::Cow::Owned(mut owned) => {
            owned.truncate(cells);
            owned
        }
        std::borrow::Cow::Borrowed(borrowed) => borrowed[..cells].to_vec(),
    };
    DenseMatrix::new(nrows, ncols, values, true).map_err(|e| LearnError::Backend(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};
    use std::thread::ThreadId;

    use antecedent_core::ExecutionContext;

    use super::{ForestLearner, PER_TREE_FOREST};
    use crate::design::{DesignView, TargetView};
    use crate::error::LearnError;
    use crate::learner::{FittedPredictor, LearnerFactory, PredictionTask};
    use crate::spec::ForestSpec;

    #[derive(Clone, Default)]
    pub(super) struct Recording(Arc<Mutex<HashMap<ThreadId, HashSet<ThreadId>>>>);

    thread_local! {
        static RECORD: RefCell<Option<Recording>> = const { RefCell::new(None) };
    }

    pub(super) fn recording() -> Option<Recording> {
        RECORD.with(|record| record.borrow().clone())
    }

    impl Recording {
        fn with<T>(&self, f: impl FnOnce() -> T) -> T {
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    RECORD.set(None);
                }
            }
            RECORD.set(Some(self.clone()));
            let _reset = Reset;
            f()
        }

        pub(super) fn note_worker(&self, fit: ThreadId, ctx: &ExecutionContext) {
            assert!(ctx.admission().available().threads() < ctx.admission().thread_budget());
            self.0.lock().unwrap().entry(fit).or_default().insert(std::thread::current().id());
        }

        fn max_workers_per_fit(&self) -> usize {
            self.0.lock().unwrap().values().map(HashSet::len).max().unwrap_or(0)
        }
    }

    fn step_design(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = if i * 2 < n { 0.0 } else { 1.0 };
        }
        (x, y)
    }

    #[test]
    fn forest_fold_fit_keeps_a_packed_design_borrowed() {
        let matrix = [0.0, 1.0, 4.0, 9.0];
        let outcomes = [0.0, 2.0, 8.0, 18.0];
        let forest = ForestLearner::new(ForestSpec::default(), PredictionTask::Regression);
        let design = DesignView::from_column_major(&matrix, matrix.len(), 1).unwrap();
        let prepared = forest.prepare(design, TargetView::new(&outcomes), None).unwrap();
        assert_eq!(prepared.design.as_ptr(), matrix.as_ptr());
        forest.fit_per_tree(&prepared, &ExecutionContext::for_tests(4)).unwrap();
    }

    fn spec(n_trees: u32, leaf: u32, extra_trees: bool) -> ForestSpec {
        ForestSpec { extra_trees, n_trees, min_samples_leaf: Some(leaf) }
    }

    fn predict(
        fitted: &dyn FittedPredictor,
        x: &[f64],
        n: usize,
        ctx: &ExecutionContext,
    ) -> Vec<f64> {
        let view = DesignView::from_column_major(x, n, 2).unwrap();
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, ctx).unwrap();
        out
    }

    #[test]
    fn per_tree_fit_names_itself_and_smartcore_stays_available() {
        let n = 24usize;
        let (x, y) = step_design(n);
        let ctx = ExecutionContext::for_tests(4);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let learner = ForestLearner::new(spec(4, 1, false), PredictionTask::Regression);
        let Err(err) = learner.fit(view, TargetView::new(&y), Some(&y), &ctx) else {
            panic!("a weight vector on a forest fit is unsupported");
        };
        assert!(matches!(err, LearnError::Unsupported { .. }));

        let grown = learner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let legacy = learner.fit_smartcore(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(grown.provenance().spec, "random_forest");
        assert_eq!(grown.provenance().implementation, PER_TREE_FOREST);
        assert_eq!(grown.provenance().version, env!("CARGO_PKG_VERSION"));
        assert_eq!(legacy.provenance().implementation, "smartcore");
        assert_eq!(legacy.provenance().version, "0.4");
        let grown_y = predict(grown.as_ref(), &x, n, &ctx);
        let legacy_y = predict(legacy.as_ref(), &x, n, &ctx);
        assert!(grown_y.iter().all(|value| value.is_finite()));
        assert!(legacy_y.iter().all(|value| value.is_finite()));
        let again = learner.fit_smartcore(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(legacy_y, predict(again.as_ref(), &x, n, &ctx));
    }

    #[test]
    fn same_seed_matches_for_both_forest_kinds() {
        let n = 32usize;
        let (x, y) = step_design(n);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        for extra in [false, true] {
            let learner = ForestLearner::new(spec(5, 2, extra), PredictionTask::Regression);
            let first_ctx = ExecutionContext::for_tests(11);
            let second_ctx = ExecutionContext::for_tests(11);
            let first = learner.fit(view, TargetView::new(&y), None, &first_ctx).unwrap();
            let second = learner.fit(view, TargetView::new(&y), None, &second_ctx).unwrap();
            assert_eq!(
                predict(first.as_ref(), &x, n, &first_ctx),
                predict(second.as_ref(), &x, n, &second_ctx),
                "extra_trees={extra}"
            );
        }
    }

    #[test]
    fn signed_zeros_with_different_outcomes_stay_in_one_leaf() {
        let x = [-0.0, 0.0, -0.0, 0.0];
        let y = [0.0, 1.0, 0.0, 1.0];
        let ctx = ExecutionContext::for_tests(2);
        let view = DesignView::from_column_major(&x, x.len(), 1).unwrap();
        let fitted = ForestLearner::new(spec(4, 1, false), PredictionTask::Regression)
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        let mut predictions = [0.0; 4];
        fitted.predict(view, &mut predictions, &ctx).unwrap();
        assert!(predictions.iter().all(|value| value.is_finite()));
        assert_eq!(predictions.map(f64::to_bits), [predictions[0].to_bits(); 4]);
    }

    #[test]
    fn leaf_size_changes_a_step_fit() {
        let n = 32usize;
        // One column, so the single tried feature is the step. Two columns draw
        // floor(sqrt(p)) = 1 feature, and a one-tree fit can draw the constant one.
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| if i * 2 < n { 0.0 } else { 1.0 }).collect();
        let ctx = ExecutionContext::for_tests(2);
        let view = DesignView::from_column_major(&x, n, 1).unwrap();
        let deep = ForestLearner::new(spec(1, 1, false), PredictionTask::Regression)
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        let shallow = ForestLearner::new(
            spec(1, u32::try_from(n).unwrap(), false),
            PredictionTask::Regression,
        )
        .fit(view, TargetView::new(&y), None, &ctx)
        .unwrap();
        let mut deep_y = vec![0.0; n];
        let mut shallow_y = vec![0.0; n];
        deep.predict(view, &mut deep_y, &ctx).unwrap();
        shallow.predict(view, &mut shallow_y, &ctx).unwrap();
        assert!(
            shallow_y
                .windows(2)
                .all(|pair| pair[0].total_cmp(&pair[1]) == std::cmp::Ordering::Equal)
        );
        assert_ne!(deep_y, shallow_y);
    }

    #[test]
    fn probability_fit_stays_inside_the_open_unit_interval() {
        let n = 40usize;
        let (x, y) = step_design(n);
        let ctx = ExecutionContext::for_tests(6);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let learner = ForestLearner::new(
            ForestSpec { extra_trees: false, n_trees: 6, min_samples_leaf: None },
            PredictionTask::BinaryProbability,
        );
        let fitted = learner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let out = predict(fitted.as_ref(), &x, n, &ctx);
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
    }

    #[test]
    fn per_tree_fit_leases_one_worker_per_tree_without_raising_the_pool() {
        let n = 24usize;
        let n_trees = 4u32;
        let (x, y) = step_design(n);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let learner = ForestLearner::new(spec(n_trees, 1, false), PredictionTask::Regression);

        let serial = ExecutionContext::for_tests(3);
        let serial_budget = serial.admission().thread_budget();
        let serial_free = serial.admission().available().threads();
        let recording = Recording::default();
        let serial_fit =
            recording.with(|| learner.fit(view, TargetView::new(&y), None, &serial)).unwrap();
        assert_eq!(recording.max_workers_per_fit(), 1);
        assert_eq!(serial.admission().thread_budget(), serial_budget);
        assert_eq!(serial.admission().available().threads(), serial_free);

        let parallel = ExecutionContext::production(3, n_trees);
        let budget = parallel.admission().thread_budget();
        let free = parallel.admission().available().threads();
        assert_eq!(budget, n_trees);
        assert_eq!(free, n_trees);
        let recording = Recording::default();
        let parallel_fit =
            recording.with(|| learner.fit(view, TargetView::new(&y), None, &parallel)).unwrap();
        assert_eq!(recording.max_workers_per_fit(), usize::try_from(n_trees).unwrap());
        assert_eq!(parallel.admission().thread_budget(), budget);
        assert_eq!(parallel.admission().available().threads(), free);
        assert_eq!(
            predict(serial_fit.as_ref(), &x, n, &serial),
            predict(parallel_fit.as_ref(), &x, n, &parallel)
        );
    }

    #[test]
    fn cross_fitted_nuisance_forest_borrows_the_single_spare_worker() {
        struct RecordedForest {
            forest: ForestLearner,
            recording: Recording,
        }

        impl LearnerFactory for RecordedForest {
            fn task(&self) -> PredictionTask {
                self.forest.task()
            }

            fn capabilities(&self) -> crate::learner::LearnerCapabilities {
                self.forest.capabilities()
            }

            fn fit(
                &self,
                x: DesignView<'_>,
                y: TargetView<'_>,
                weights: Option<&[f64]>,
                ctx: &ExecutionContext,
            ) -> Result<Box<dyn FittedPredictor>, LearnError> {
                self.recording.with(|| self.forest.fit(x, y, weights, ctx))
            }
        }

        let n = 24usize;
        let (x, y) = step_design(n);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let forest = ForestLearner::new(spec(4, 1, false), PredictionTask::Regression);
        let learner = RecordedForest { forest, recording: Recording::default() };
        let parallel = ExecutionContext::production(3, 3);
        let serial = ExecutionContext::for_tests(3);

        let expected =
            crate::crossfit::cross_fit(&forest, view, TargetView::new(&y), 2, &serial, None)
                .unwrap();
        let actual =
            crate::crossfit::cross_fit(&learner, view, TargetView::new(&y), 2, &parallel, None)
                .unwrap();
        assert_eq!(actual.predictions, expected.predictions);
        assert_eq!(actual.fold_assignment, expected.fold_assignment);
        assert_eq!(learner.recording.max_workers_per_fit(), 2);
        assert_eq!(parallel.admission().available().threads(), 3);
    }
}
