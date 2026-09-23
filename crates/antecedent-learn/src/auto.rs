//! Restrained Auto: OOF loss between a parametric learner and optional GBT.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::first_col_is_exact_ones;

use crate::crossfit::{NuisanceDiagnostics, cross_fit};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{FittedPredictor, LearnerFactory, LearnerProvenance, PredictionTask};
use crate::spec::{ElasticNetSpec, GbtSpec, LearnerSpec, LogisticSpec, RidgeSpec, resolve_for};

const LARGE_N: usize = 5_000;
const SHALLOW_GBT: GbtSpec = GbtSpec { trees: 80, depth: 3, learning_rate: 0.1 };

/// Select a factory for `task` by restrained OOF comparison.
///
/// # Errors
///
/// Empty design, or every candidate fails to fit.
pub fn resolve_auto(
    task: PredictionTask,
    x: DesignView<'_>,
    y: TargetView<'_>,
    ctx: &ExecutionContext,
) -> Result<(Box<dyn LearnerFactory>, NuisanceDiagnostics), LearnError> {
    let sparse = x.is_sparse();
    let (values, n, p) = crate::dense::materialize_dense_colmajor(x)?;
    let target = if y.len() == x.physical_nrows() {
        crate::dense::gather_physical(y.values(), x, n)?
    } else if y.len() == n {
        y.values()[..n].to_vec()
    } else {
        return Err(LearnError::Shape {
            message: "auto selection needs a non-empty aligned design",
        });
    };
    let view = DesignView::from_column_major(values.as_ref(), n, p)?;
    let tree_view = if p > 1 && first_col_is_exact_ones(values.as_ref(), n) {
        Some(DesignView::from_column_major(&values.as_ref()[n..], n, p - 1)?)
    } else {
        None
    };
    let strips_intercept = tree_view.is_some();
    let (factory, diagnostics, winner) =
        select_auto(task, view, tree_view, TargetView::new(&target), ctx, sparse)?;
    let factory: Box<dyn LearnerFactory> = if strips_intercept && drops_constant_intercept(winner) {
        Box::new(DropInterceptFactory { inner: factory })
    } else {
        factory
    };
    Ok((factory, diagnostics))
}

/// Tree learners share the explicit-GBT contract: a constant intercept column is dropped.
const fn drops_constant_intercept(spec: LearnerSpec) -> bool {
    matches!(
        spec,
        LearnerSpec::GradientBoostedTrees(_)
            | LearnerSpec::RandomForest(_)
            | LearnerSpec::NeuralNet(_)
    )
}

fn select_auto(
    task: PredictionTask,
    x: DesignView<'_>,
    x_tree: Option<DesignView<'_>>,
    y: TargetView<'_>,
    ctx: &ExecutionContext,
    sparse: bool,
) -> Result<(Box<dyn LearnerFactory>, NuisanceDiagnostics, LearnerSpec), LearnError> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || y.len() != x.physical_nrows() {
        return Err(LearnError::Shape {
            message: "auto selection needs a non-empty aligned design",
        });
    }
    let parametric = match task {
        PredictionTask::Regression => LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 }),
        PredictionTask::BinaryProbability => LearnerSpec::Logistic(LogisticSpec::default()),
    };
    let gbdt = if n < LARGE_N {
        LearnerSpec::GradientBoostedTrees(SHALLOW_GBT)
    } else {
        LearnerSpec::GradientBoostedTrees(GbtSpec::default())
    };
    let candidates = auto_candidates(n, p, sparse, task, parametric, gbdt);
    let folds = n.clamp(2, 5);
    if candidates.len() == 1 {
        let factory = resolve_for(candidates[0], task)?;
        return Ok((
            factory,
            NuisanceDiagnostics {
                winner: Some(candidates[0].name()),
                ..NuisanceDiagnostics::default()
            },
            candidates[0],
        ));
    }
    let mut best: Option<(f64, LearnerSpec, Box<dyn LearnerFactory>, NuisanceDiagnostics)> = None;
    let mut challenger_loss = None;
    let mut non_finite: Vec<&'static str> = Vec::new();
    for spec in candidates {
        let factory = match resolve_for(spec, task) {
            Ok(f) => f,
            Err(LearnError::ProviderUnavailable { .. }) => continue,
            Err(e) => return Err(e),
        };
        // Match explicit GBT/forest design handling: trees never see a constant intercept.
        let design = if drops_constant_intercept(spec) { x_tree.unwrap_or(x) } else { x };
        let oof = cross_fit(factory.as_ref(), design, y, folds, ctx, None)?;
        let loss = match task {
            PredictionTask::Regression => oof.validation.rmse,
            PredictionTask::BinaryProbability => oof.validation.logloss,
        };
        // A candidate whose held-out loss is missing or non-finite (NaN predictions from a
        // diverged fit) is a failed candidate, not a contender: `loss < best` is false
        // against NaN, so admitting one first would let it win by never being replaced.
        let Some(loss) = admissible_loss(loss) else {
            non_finite.push(spec.name());
            continue;
        };
        match &best {
            None => best = Some((loss, spec, factory, oof.validation)),
            Some((best_loss, ..)) => {
                if loss < *best_loss {
                    challenger_loss = Some(*best_loss);
                    best = Some((loss, spec, factory, oof.validation));
                } else {
                    challenger_loss = Some(loss);
                }
            }
        }
    }
    if best.is_none() && !non_finite.is_empty() {
        return Err(LearnError::Backend(format!(
            "auto selection: every candidate produced a non-finite out-of-fold loss ({})",
            non_finite.join(", ")
        )));
    }
    let Some((_, spec, factory, mut validation)) = best else {
        return Err(LearnError::ProviderUnavailable {
            spec: "auto",
            required: required_for_auto(task),
        });
    };
    validation.winner = Some(spec.name());
    validation.challenger_loss = challenger_loss;
    Ok((factory, validation, spec))
}

/// The held-out loss of a candidate that may compete: present and finite.
fn admissible_loss(loss: Option<f64>) -> Option<f64> {
    loss.filter(|l| l.is_finite())
}

fn auto_candidates(
    n: usize,
    p: usize,
    sparse: bool,
    task: PredictionTask,
    parametric: LearnerSpec,
    gbdt: LearnerSpec,
) -> Vec<LearnerSpec> {
    let gbdt_on = cfg!(feature = "ml-gbdt") && !sparse;
    let mut candidates = Vec::new();
    if sparse || p > n {
        if matches!(task, PredictionTask::Regression) {
            candidates.push(LearnerSpec::ElasticNet(ElasticNetSpec::default()));
        }
        candidates.push(parametric);
    } else {
        candidates.push(parametric);
    }
    if gbdt_on {
        candidates.push(gbdt);
    }
    candidates.sort_by_key(|spec| spec.name());
    candidates.dedup();
    candidates
}

/// Capabilities advertised for Auto when no candidate resolved.
#[must_use]
pub fn required_for_auto(task: PredictionTask) -> crate::learner::LearnerCapabilities {
    let mut cap = crate::learner::LearnerCapabilities::none();
    match task {
        PredictionTask::Regression => cap.regression = true,
        PredictionTask::BinaryProbability => cap.binary_probability = true,
    }
    cap
}

/// Tree winner fitted without a constant intercept; predict and export retain that contract.
struct DropInterceptPredictor {
    inner: Box<dyn FittedPredictor>,
}

impl FittedPredictor for DropInterceptPredictor {
    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        let mut portable = self.inner.portable()?;
        portable.columns = portable
            .columns
            .checked_add(1)
            .ok_or(LearnError::Shape { message: "portable prediction columns overflow" })?;
        match &mut portable.model {
            crate::PredictionMap::Linear { coefficients, .. } => coefficients.insert(0, 0.0),
            crate::PredictionMap::Trees { trees, .. } => {
                for tree in trees {
                    for node in tree {
                        if let crate::PredictionNode::Split { feature, .. } = node {
                            *feature = feature.checked_add(1).ok_or(LearnError::Shape {
                                message: "portable tree feature index overflow",
                            })?;
                        }
                    }
                }
            }
        }
        portable.validate()?;
        Ok(portable)
    }

    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        let (values, n, p) = crate::dense::materialize_dense_colmajor(x)?;
        if p > 1 && first_col_is_exact_ones(values.as_ref(), n) {
            let stripped = DesignView::from_column_major(&values[n..], n, p - 1)?;
            self.inner.predict(stripped, out, ctx)
        } else {
            let view = DesignView::from_column_major(values.as_ref(), n, p)?;
            self.inner.predict(view, out, ctx)
        }
    }

    fn provenance(&self) -> LearnerProvenance {
        self.inner.provenance()
    }
}

/// A selected tree factory whose fit input includes a generated leading intercept.
///
/// `resolve_auto` exposes the selected factory publicly, so this wrapper must make its
/// fitting behavior agree with the intercept-free OOF comparison used to select it.
struct DropInterceptFactory {
    inner: Box<dyn LearnerFactory>,
}

impl LearnerFactory for DropInterceptFactory {
    fn task(&self) -> PredictionTask {
        self.inner.task()
    }

    fn capabilities(&self) -> crate::LearnerCapabilities {
        self.inner.capabilities()
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
        if weights.is_some_and(|w| w.len() != x.physical_nrows()) {
            return Err(LearnError::Shape { message: "weights length != physical rows" });
        }
        let (values, n, p) = crate::dense::materialize_dense_colmajor(x)?;
        if p <= 1 || !first_col_is_exact_ones(values.as_ref(), n) {
            return Err(LearnError::Shape {
                message: "selected tree requires a leading constant intercept column",
            });
        }
        let view = DesignView::from_column_major(&values[n..], n, p - 1)?;
        let target = crate::dense::gather_physical(y.values(), x, n)?;
        let weights = weights.map(|w| crate::dense::gather_physical(w, x, n)).transpose()?;
        let fitted = self.inner.fit(view, TargetView::new(&target), weights.as_deref(), ctx)?;
        Ok(Box::new(DropInterceptPredictor { inner: fitted }))
    }
}

/// Auto selection is deferred until fit so outer validation rows cannot select a model.
pub(crate) struct AutoLearner(pub PredictionTask);

impl LearnerFactory for AutoLearner {
    fn task(&self) -> PredictionTask {
        self.0
    }

    fn capabilities(&self) -> crate::LearnerCapabilities {
        required_for_auto(self.0)
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if weights.is_some() {
            return Err(LearnError::Unsupported { message: "Auto does not accept sample weights" });
        }
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let sparse = x.is_sparse();
        let (values, n, p) = crate::dense::materialize_dense_colmajor(x)?;
        let target = crate::dense::gather_physical(y.values(), x, n)?;
        let view = DesignView::from_column_major(values.as_ref(), n, p)?;
        let target = TargetView::new(&target);
        let drop_intercept = p > 1 && first_col_is_exact_ones(values.as_ref(), n);
        let tree_view = if drop_intercept {
            Some(DesignView::from_column_major(&values.as_ref()[n..], n, p - 1)?)
        } else {
            None
        };
        let (factory, _, winner) = select_auto(self.0, view, tree_view, target, ctx, sparse)?;
        if drops_constant_intercept(winner) && drop_intercept {
            let fit_view = DesignView::from_column_major(&values.as_ref()[n..], n, p - 1)?;
            let fitted = factory.fit(fit_view, target, None, ctx)?;
            Ok(Box::new(DropInterceptPredictor { inner: fitted }))
        } else {
            factory.fit(view, target, None, ctx)
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod audit_tests {
    use super::*;
    #[test]
    fn selection_cannot_read_rows_outside_training_view() {
        let n = 30;
        let values: Vec<f64> = (0..n).map(|_| 1.0).chain((0..n).map(|i| i as f64)).collect();
        let target: Vec<f64> = (0..n).map(|i| 2.0 + i as f64).collect();
        let rows: Vec<u32> = (0..20).collect();
        let view = DesignView::from_column_major(&values, n, 2).unwrap();
        let train = view.with_rows(crate::RowSelection::new(&rows)).unwrap();
        let ctx = ExecutionContext::for_tests(22);
        let factory = AutoLearner(PredictionTask::Regression);
        let a = factory.fit(train, TargetView::new(&target), None, &ctx).unwrap();
        let mut changed = target.clone();
        changed[20..].fill(1e9);
        let b = factory.fit(train, TargetView::new(&changed), None, &ctx).unwrap();
        let mut pa = vec![0.0; n];
        let mut pb = vec![0.0; n];
        a.predict(view, &mut pa, &ctx).unwrap();
        b.predict(view, &mut pb, &ctx).unwrap();
        assert_eq!(a.provenance(), b.provenance());
        assert_eq!(pa, pb);
    }

    /// A NaN loss compares false against everything, so a NaN first candidate would never
    /// be displaced; a missing diagnostic used to read as `+inf` and hide the same defect.
    #[test]
    fn missing_or_non_finite_losses_cannot_compete() {
        assert_eq!(admissible_loss(Some(0.25)), Some(0.25));
        assert_eq!(admissible_loss(Some(f64::NAN)), None);
        assert_eq!(admissible_loss(Some(f64::INFINITY)), None);
        assert_eq!(admissible_loss(None), None);
    }

    #[test]
    fn large_n_still_compares_a_parametric_challenger() {
        let parametric = LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 });
        let gbdt = LearnerSpec::GradientBoostedTrees(GbtSpec::default());
        let candidates =
            auto_candidates(5_000, 3, false, PredictionTask::Regression, parametric, gbdt);
        assert!(candidates.iter().any(|spec| matches!(spec, LearnerSpec::Ridge(_))));
        #[cfg(feature = "ml-gbdt")]
        assert!(candidates.iter().any(|spec| matches!(spec, LearnerSpec::GradientBoostedTrees(_))));
    }

    #[test]
    fn wide_or_sparse_designs_include_elastic_net() {
        let parametric = LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 });
        let gbdt = LearnerSpec::GradientBoostedTrees(SHALLOW_GBT);
        let wide = auto_candidates(20, 40, false, PredictionTask::Regression, parametric, gbdt);
        assert!(wide.iter().any(|spec| matches!(spec, LearnerSpec::ElasticNet(_))));
        let sparse = auto_candidates(30, 3, true, PredictionTask::Regression, parametric, gbdt);
        assert!(sparse.iter().any(|spec| matches!(spec, LearnerSpec::ElasticNet(_))));
        assert!(!sparse.iter().any(|spec| matches!(spec, LearnerSpec::GradientBoostedTrees(_))));
    }

    #[cfg(feature = "ml-gbdt")]
    #[test]
    fn auto_tree_winner_strips_the_same_intercept_as_explicit_gbt() {
        // XOR-like nonlinear outcome so restrained Auto prefers GBDT over ridge.
        let n = 80usize;
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = if i % 2 == 0 { 1.0 } else { -1.0 };
            let b = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
            x[i] = 1.0;
            x[n + i] = a;
            x[2 * n + i] = b;
            y[i] = a * b;
        }
        let view = DesignView::from_column_major(&x, n, 3).unwrap();
        let ctx = ExecutionContext::for_tests(9);
        let auto = AutoLearner(PredictionTask::Regression)
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        assert_eq!(auto.provenance().spec, "gradient_boosted_trees");
        let explicit =
            resolve_for(LearnerSpec::GradientBoostedTrees(SHALLOW_GBT), PredictionTask::Regression)
                .unwrap()
                .fit(
                    DesignView::from_column_major(&x[n..], n, 2).unwrap(),
                    TargetView::new(&y),
                    None,
                    &ctx,
                )
                .unwrap();
        let mut pa = vec![0.0; n];
        let mut pe = vec![0.0; n];
        auto.predict(view, &mut pa, &ctx).unwrap();
        explicit
            .predict(DesignView::from_column_major(&x[n..], n, 2).unwrap(), &mut pe, &ctx)
            .unwrap();
        let mae = (0..n).map(|i| (pa[i] - pe[i]).abs()).sum::<f64>() / n as f64;
        assert!(mae < 0.15, "auto GBT with intercept strip must match explicit GBT; mae={mae}");

        let portable = auto.portable().unwrap();
        assert_eq!(portable.columns, 3);
        let mut pp = vec![0.0; n];
        portable.predict(view, &mut pp, &ctx).unwrap();
        let portable_mae = (0..n).map(|i| (pa[i] - pp[i]).abs()).sum::<f64>() / n as f64;
        assert!(portable_mae < 1e-12, "portable Auto tree must retain its intercept schema");
    }

    #[cfg(feature = "ml-gbdt")]
    #[test]
    fn resolved_auto_tree_factory_keeps_its_intercept_contract() {
        let n = 80usize;
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = if i % 2 == 0 { 1.0 } else { -1.0 };
            let b = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
            x[i] = 1.0;
            x[n + i] = a;
            x[2 * n + i] = b;
            y[i] = a * b;
        }
        let view = DesignView::from_column_major(&x, n, 3).unwrap();
        let ctx = ExecutionContext::for_tests(10);
        let (factory, diagnostics) =
            resolve_auto(PredictionTask::Regression, view, TargetView::new(&y), &ctx).unwrap();
        assert_eq!(diagnostics.winner, Some("gradient_boosted_trees"));
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let mut actual = vec![0.0; n];
        fitted.predict(view, &mut actual, &ctx).unwrap();
        let expected =
            resolve_for(LearnerSpec::GradientBoostedTrees(SHALLOW_GBT), PredictionTask::Regression)
                .unwrap()
                .fit(
                    DesignView::from_column_major(&x[n..], n, 2).unwrap(),
                    TargetView::new(&y),
                    None,
                    &ctx,
                )
                .unwrap();
        let mut expected_predictions = vec![0.0; n];
        expected
            .predict(
                DesignView::from_column_major(&x[n..], n, 2).unwrap(),
                &mut expected_predictions,
                &ctx,
            )
            .unwrap();
        let mae =
            (0..n).map(|i| (actual[i] - expected_predictions[i]).abs()).sum::<f64>() / n as f64;
        assert!(mae < 0.15, "resolved Auto tree must fit the intercept-free design; mae={mae}");
    }
}
