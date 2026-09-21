//! Restrained Auto: OOF loss between a parametric learner and optional GBT.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::crossfit::{NuisanceDiagnostics, cross_fit};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{LearnerFactory, PredictionTask};
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
    select_auto(task, x, y, ctx, x.is_sparse())
}

fn select_auto(
    task: PredictionTask,
    x: DesignView<'_>,
    y: TargetView<'_>,
    ctx: &ExecutionContext,
    sparse: bool,
) -> Result<(Box<dyn LearnerFactory>, NuisanceDiagnostics), LearnError> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || y.len() != x.physical_nrows() {
        return Err(LearnError::Shape {
            message: "auto selection needs a non-empty aligned design",
        });
    }
    let parametric = match task {
        PredictionTask::Regression => LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 }),
        PredictionTask::BinaryProbability => LearnerSpec::Logistic(LogisticSpec {}),
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
        ));
    }
    let mut best: Option<(f64, LearnerSpec, Box<dyn LearnerFactory>, NuisanceDiagnostics)> = None;
    let mut challenger_loss = None;
    for spec in candidates {
        let factory = match resolve_for(spec, task) {
            Ok(f) => f,
            Err(LearnError::ProviderUnavailable { .. }) => continue,
            Err(e) => return Err(e),
        };
        let oof = cross_fit(factory.as_ref(), x, y, folds, ctx, None)?;
        let loss = match task {
            PredictionTask::Regression => oof.validation.rmse.unwrap_or(f64::INFINITY),
            PredictionTask::BinaryProbability => oof.validation.logloss.unwrap_or(f64::INFINITY),
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
    let Some((_, spec, factory, mut validation)) = best else {
        return Err(LearnError::ProviderUnavailable {
            spec: "auto",
            required: required_for_auto(task),
        });
    };
    validation.winner = Some(spec.name());
    validation.challenger_loss = challenger_loss;
    Ok((factory, validation))
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
    ) -> Result<Box<dyn crate::FittedPredictor>, LearnError> {
        if weights.is_some() {
            return Err(LearnError::Unsupported { message: "Auto does not accept sample weights" });
        }
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let sparse = x.is_sparse();
        let (values, n, p) = crate::dense::materialize_dense_colmajor(x)?;
        let target = crate::dense::gather_physical(y.values(), x, n)?;
        let view = DesignView::from_column_major(&values, n, p)?;
        let target = TargetView::new(&target);
        let (factory, _) = select_auto(self.0, view, target, ctx, sparse)?;
        factory.fit(view, target, None, ctx)
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
}
