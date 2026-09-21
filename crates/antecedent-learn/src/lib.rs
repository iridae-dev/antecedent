//! Prediction contract for Antecedent nuisance models (ADR 0023).
//!
//! This crate owns [`DesignView`], [`LearnerFactory`], and [`FittedPredictor`].
//! It does not identify effects or compute orthogonal scores.
//! `antecedent-estimate` must not name Forust, `SmartCore`, Burn, or other providers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod auto;
pub mod crossfit;
mod dense;
pub mod design;
pub mod elastic_net;
pub mod error;
#[cfg(feature = "ml-forest")]
pub mod forest;
#[cfg(feature = "ml-gbdt")]
pub mod gbt;
pub mod learner;
pub mod linear;
pub mod logistic;
#[cfg(feature = "ml-gpu")]
pub mod neural;
pub mod ridge;
pub mod spec;
pub mod transform;

pub use auto::resolve_auto;
pub use crossfit::{CrossFittedPrediction, NuisanceDiagnostics, assign_folds, cross_fit, diagnose};
pub use design::{
    DenseDesign, DesignStorage, DesignView, Layout, RowSelection, SparseDesignView, TargetView,
};
pub use elastic_net::ElasticNetLearner;
pub use error::LearnError;
pub use learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
pub use linear::LinearLearner;
pub use logistic::LogisticLearner;
pub use ridge::RidgeLearner;
pub use spec::{
    ElasticNetSpec, ForestSpec, GbtSpec, LearnerSpec, LinearSpec, LogisticSpec, NeuralSpec,
    RidgeSpec, require_task, resolve, resolve_for,
};
pub use transform::{FittedTransformer, Identity, Log1p, TransformerFactory};

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;
    use antecedent_core::ExecutionContext;

    fn line_design(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 3.0 + 4.0 * (i as f64);
        }
        (x, y)
    }

    #[test]
    fn ridge_default_is_a_positive_penalty() {
        assert_eq!(RidgeSpec::default().lambda, 1.0);
        assert_eq!(LearnerSpec::parse("ridge").unwrap(), LearnerSpec::Ridge(RidgeSpec::default()));
    }

    #[test]
    fn elastic_net_default_is_a_positive_mix() {
        assert_eq!(ElasticNetSpec::default(), ElasticNetSpec { lambda: 1.0, l1_ratio: 0.5 });
        assert_eq!(
            LearnerSpec::parse("elastic_net").unwrap(),
            LearnerSpec::ElasticNet(ElasticNetSpec::default())
        );
        let n = 20usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let fitted = resolve(LearnerSpec::ElasticNet(ElasticNetSpec::default()))
            .unwrap()
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        assert_eq!(fitted.provenance().spec, "elastic_net");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let ols = LinearLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let mut a = vec![0.0; n];
        ols.predict(view, &mut a, &ctx).unwrap();
        let err = (0..n).map(|i| (out[i] - a[i]).abs()).sum::<f64>() / n as f64;
        assert!(err > 0.0, "default elastic-net must not collapse to OLS");
    }

    #[test]
    fn logistic_refuses_non_binary_labels() {
        let x = [1.0, 1.0, 0.0, 1.0];
        let y = [0.2, 0.8];
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, 2, 2).unwrap();
        let err = match LogisticLearner.fit(view, TargetView::new(&y), None, &ctx) {
            Ok(_) => panic!("non-binary labels should fail"),
            Err(error) => error,
        };
        assert!(matches!(err, LearnError::Shape { message } if message.contains("0 or 1")));
    }

    #[test]
    fn linear_rejects_binary_probability() {
        let err = LinearLearner::for_task(PredictionTask::BinaryProbability).unwrap_err();
        assert!(matches!(
            err,
            LearnError::TaskMismatch { requested: PredictionTask::BinaryProbability, .. }
        ));
        assert!(!LinearLearner.capabilities().binary_probability);
        assert_eq!(LinearLearner.task(), PredictionTask::Regression);
    }

    #[test]
    fn resolve_linear_and_refuse_unbuilt_specs() {
        let factory = resolve(LearnerSpec::Linear(LinearSpec::default())).unwrap();
        assert_eq!(factory.task(), PredictionTask::Regression);
        let ridge = resolve(LearnerSpec::Ridge(RidgeSpec { lambda: 0.1 })).unwrap();
        assert_eq!(ridge.task(), PredictionTask::Regression);
        let logistic = resolve(LearnerSpec::Logistic(LogisticSpec::default())).unwrap();
        assert_eq!(logistic.task(), PredictionTask::BinaryProbability);
        match resolve(LearnerSpec::NeuralNet(NeuralSpec::default())) {
            #[cfg(feature = "ml-gpu")]
            Ok(factory) => assert_eq!(factory.task(), PredictionTask::Regression),
            #[cfg(not(feature = "ml-gpu"))]
            Ok(_) => panic!("neural_net should be unresolved"),
            #[cfg(feature = "ml-gpu")]
            Err(e) => panic!("neural_net should resolve: {e}"),
            #[cfg(not(feature = "ml-gpu"))]
            Err(e) => {
                assert!(matches!(e, LearnError::ProviderUnavailable { spec: "neural_net", .. }))
            }
        }
        match resolve(LearnerSpec::GradientBoostedTrees(GbtSpec::default())) {
            #[cfg(feature = "ml-gbdt")]
            Ok(factory) => assert_eq!(factory.task(), PredictionTask::Regression),
            #[cfg(not(feature = "ml-gbdt"))]
            Ok(_) => panic!("gbt should be unresolved"),
            #[cfg(feature = "ml-gbdt")]
            Err(e) => panic!("gbt should resolve: {e}"),
            #[cfg(not(feature = "ml-gbdt"))]
            Err(e) => assert!(matches!(
                e,
                LearnError::ProviderUnavailable { spec: "gradient_boosted_trees", .. }
            )),
        }
    }

    #[test]
    fn ridge_near_zero_matches_ols() {
        let n = 20usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let ols = LinearLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let ridge = RidgeLearner::new(RidgeSpec { lambda: 1e-10 })
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        ols.predict(view, &mut a, &ctx).unwrap();
        ridge.predict(view, &mut b, &ctx).unwrap();
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-6);
        }
        let err = RidgeLearner::for_task(PredictionTask::BinaryProbability, RidgeSpec::default())
            .unwrap_err();
        assert!(matches!(err, LearnError::TaskMismatch { .. }));
        assert!(
            RidgeLearner::new(RidgeSpec::default())
                .fit(view, TargetView::new(&y), Some(&[1.0; 20]), &ctx)
                .is_err()
        );
    }

    #[test]
    fn logistic_probabilities_are_clamped() {
        let n = 40usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = (i as f64 - 20.0) / 8.0;
            y[i] = match i {
                0..=11 => {
                    if i % 5 == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                12..=27 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
        }
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let fitted = LogisticLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().spec, "logistic");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        for p in &out {
            assert!(*p > 0.0 && *p < 1.0);
        }
        assert!(out[n - 1] > out[0]);
        assert!(LogisticLearner::for_task(PredictionTask::Regression).is_err());
    }

    #[test]
    fn ols_fit_predict_and_provenance() {
        let n = 20usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let fitted = LinearLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "faer");
        assert_eq!(fitted.provenance().spec, "linear");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        for i in 0..n {
            assert!((out[i] - y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn weighted_matches_unweighted_on_unit_weights() {
        let n = 15usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let w = vec![1.0; n];
        let unweighted = LinearLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let weighted = LinearLearner.fit(view, TargetView::new(&y), Some(&w), &ctx).unwrap();
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        unweighted.predict(view, &mut a, &ctx).unwrap();
        weighted.predict(view, &mut b, &ctx).unwrap();
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn index_only_folds_fit_without_caller_materializing_x_train() {
        let n = 10usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let even: Vec<u32> = (0..n as u32).step_by(2).collect();
        let odd: Vec<u32> = (1..n as u32).step_by(2).collect();
        let train = view.with_rows(RowSelection::new(&even)).unwrap();
        let test = view.with_rows(RowSelection::new(&odd)).unwrap();
        let fitted = LinearLearner.fit(train, TargetView::new(&y), None, &ctx).unwrap();
        let mut out = vec![0.0; test.nrows()];
        fitted.predict(test, &mut out, &ctx).unwrap();
        for (k, &i) in odd.iter().enumerate() {
            assert!((out[k] - y[i as usize]).abs() < 1e-8);
        }
        match train.storage() {
            DesignStorage::Dense(d) => assert_eq!(d.values().as_ptr(), x.as_ptr()),
            DesignStorage::SparseCsr(_) => panic!("expected dense"),
        }
    }

    #[test]
    fn identity_transform_is_stateless() {
        let x = [1.0, 1.0, 0.0, 1.0];
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, 2, 2).unwrap();
        let rows = [0_u32, 1];
        let fitted = Identity.fit(view, RowSelection::new(&rows), &ctx).unwrap();
        let (buf, nrows, ncols) = fitted.transform(view, &ctx).unwrap();
        assert_eq!((nrows, ncols), (2, 2));
        assert_eq!(buf, x);
    }

    #[cfg(feature = "ml-gbdt")]
    #[test]
    fn gbt_fits_line_and_probabilities() {
        let n = 30usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let spec = GbtSpec { trees: 20, depth: 3, learning_rate: 0.2 };
        let factory =
            resolve_for(LearnerSpec::GradientBoostedTrees(spec), PredictionTask::Regression)
                .unwrap();
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "forust-ml");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().any(|v| v.is_finite()));

        let mut t = vec![0.0; n];
        for i in 0..n {
            t[i] = if i >= 15 { 1.0 } else { 0.0 };
        }
        let clf =
            resolve_for(LearnerSpec::GradientBoostedTrees(spec), PredictionTask::BinaryProbability)
                .unwrap();
        let fitted = clf.fit(view, TargetView::new(&t), None, &ctx).unwrap();
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));

        let even: Vec<u32> = (0..n as u32).step_by(2).collect();
        let train = view.with_rows(RowSelection::new(&even)).unwrap();
        let folded = factory.fit(train, TargetView::new(&y), None, &ctx).unwrap();
        let mut fold_out = vec![0.0; train.nrows()];
        folded.predict(train, &mut fold_out, &ctx).unwrap();
        assert_eq!(fold_out.len(), even.len());
    }

    #[cfg(feature = "ml-gpu")]
    #[test]
    fn neural_net_fits_line_and_probabilities() {
        let n = 24usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let spec = NeuralSpec { hidden: 8, epochs: 60, learning_rate: 0.05 };
        let factory =
            resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::Regression).unwrap();
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "burn");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let mut t = vec![0.0; n];
        for i in 0..n {
            t[i] = if i >= n / 2 { 1.0 } else { 0.0 };
        }
        let clf =
            resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::BinaryProbability).unwrap();
        let fitted = clf.fit(view, TargetView::new(&t), None, &ctx).unwrap();
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
    }

    #[cfg(feature = "ml-forest")]
    #[test]
    fn forest_fits_line_and_probabilities() {
        let n = 40usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let factory = resolve_for(
            LearnerSpec::RandomForest(ForestSpec::default()),
            PredictionTask::Regression,
        )
        .unwrap();
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "smartcore");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().any(|v| v.is_finite()));
        let mut t = vec![0.0; n];
        for i in 0..n {
            t[i] = if i >= n / 2 { 1.0 } else { 0.0 };
        }
        let clf = resolve_for(
            LearnerSpec::RandomForest(ForestSpec::default()),
            PredictionTask::BinaryProbability,
        )
        .unwrap();
        let fitted = clf.fit(view, TargetView::new(&t), None, &ctx).unwrap();
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
    }

    #[test]
    fn auto_prefers_ridge_on_a_line() {
        let n = 40usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let (factory, diag) =
            resolve_auto(PredictionTask::Regression, view, TargetView::new(&y), &ctx).unwrap();
        assert_eq!(factory.task(), PredictionTask::Regression);
        assert_eq!(diag.winner, Some("ridge"));
    }

    #[cfg(feature = "ml-gbdt")]
    #[test]
    fn auto_prefers_gbt_on_nonlinear_outcome() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / 10.0;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = (t * 2.0).sin();
        }
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let (_factory, diag) =
            resolve_auto(PredictionTask::Regression, view, TargetView::new(&y), &ctx).unwrap();
        assert_eq!(diag.winner, Some("gradient_boosted_trees"));
    }
}
