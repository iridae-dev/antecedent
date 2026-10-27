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
pub mod bayesian_basis;
pub mod bayesian_basis_regression;
pub mod categorical;
mod coefficient;
pub mod crossfit;
pub mod nested_markov_bayesian;
pub use categorical::FiniteJoint;
mod dense;
pub mod design;
pub mod elastic_net;
pub mod error;
pub mod fit_counts;
mod fitted_effect;
pub use fitted_effect::{
    EffectSupport, FeatureRange, FittedEffect, FittedEffectError, SupportReport,
};
#[cfg(feature = "ml-forest")]
pub mod forest;
#[cfg(feature = "ml-gbdt")]
pub mod gbt;
pub mod lasso_logistic;
pub mod learner;
pub mod portable;
pub use portable::{PortablePredictor, PredictionMap, PredictionNode};
pub mod linear;
pub mod logistic;
#[cfg(feature = "ml-neural")]
pub mod neural;
pub mod posterior_prediction;
pub mod recipes;
pub mod ridge;
pub mod spec;
mod streams;
pub mod transform;
pub mod transported_effect;

#[cfg(test)]
pub(crate) use auto::resolve_auto;
pub use bayesian_basis::{
    BasisData, BasisTargetPopulation, BayesianBasisEffect, BayesianBasisGComputation,
    BayesianBasisSpec,
};
pub use bayesian_basis_regression::{
    BASIS_MAX_DEGREE, BasisDesign, BasisRegressionPosterior, InformationRank,
    KnownVarianceBasisRegression, KnownVarianceBasisSpec, PolynomialBasis, information_rank,
};
pub use crossfit::cross_fit_selected;
pub use crossfit::{
    CrossFittedPrediction, NuisanceDiagnostics, assign_folds, cross_fit_with_folds,
    cross_fit_with_folds_weighted, diagnose, try_diagnose,
};
pub use design::{DenseDesign, DesignView, Layout, RowSelection, TargetView};
pub use error::LearnError;
pub use lasso_logistic::LassoLogisticLearner;
pub use learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
#[cfg(test)]
pub(crate) use linear::LinearLearner;
#[cfg(test)]
pub(crate) use logistic::LogisticLearner;
pub use logistic::RidgeLogisticLearner;
pub use posterior_prediction::{
    PosteriorPrediction, PosteriorPredictionDiagnostics, PosteriorPredictionProvenance,
};
#[cfg(test)]
pub(crate) use ridge::RidgeLearner;
#[cfg(test)]
pub(crate) use spec::FOREST_PROBABILITY_MIN_LEAF;
pub use spec::{
    DevicePolicy, ElasticNetSpec, ForestSpec, GbtSpec, LearnerSpec, LinearSpec, LogisticSpec,
    NeuralSpec, RidgeSpec, resolve, resolve_for, resolve_propensity_for,
};
pub use transform::{FittedTransformer, Identity, TransformerFactory};

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "the tests build small synthetic designs, casting small indices and comparing floats copied without rounding"
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
        let err = match LogisticLearner::default().fit(view, TargetView::new(&y), None, &ctx) {
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
            #[cfg(feature = "ml-neural")]
            Ok(factory) => assert_eq!(factory.task(), PredictionTask::Regression),
            #[cfg(not(feature = "ml-neural"))]
            Ok(_) => panic!("neural_net should be unresolved"),
            #[cfg(feature = "ml-neural")]
            Err(e) => panic!("neural_net should resolve: {e}"),
            #[cfg(not(feature = "ml-neural"))]
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
        // Unit prior weights reproduce the unweighted ridge fit.
        let unweighted = RidgeLearner::new(RidgeSpec::default())
            .fit(view, TargetView::new(&y), None, &ctx)
            .unwrap();
        let weighted = RidgeLearner::new(RidgeSpec::default())
            .fit(view, TargetView::new(&y), Some(&[1.0; 20]), &ctx)
            .unwrap();
        unweighted.predict(view, &mut a, &ctx).unwrap();
        weighted.predict(view, &mut b, &ctx).unwrap();
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-12);
        }
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
        let fitted = LogisticLearner::default().fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().spec, "logistic");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        for p in &out {
            assert!(*p > 0.0 && *p < 1.0);
        }
        assert!(out[n - 1] > out[0]);
        assert!(LogisticLearner::for_task(PredictionTask::Regression).is_err());
    }

    /// A ridge spec asked for on a probability nuisance must fit a *penalized* logistic
    /// model, not silently drop its `lambda`. On perfectly separated data the unpenalized
    /// MLE diverges (and is refused), while the ridge fit exists and has a bounded slope
    /// (the penalized objective at the solution cannot exceed its value at zero,
    /// `20 ln 2`, so `λβ²/2 ≤ 20 ln 2` and `|β| ≤ sqrt(40 ln 2 / λ)` = 5.3 at λ = 1).
    #[test]
    fn ridge_spec_maps_to_a_penalized_logistic_model() {
        let n = 20usize;
        let mut x = vec![1.0; n];
        x.extend((0..n).map(|i| i as f64));
        let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i >= 10))).collect();
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let ctx = ExecutionContext::for_tests(1);

        let spec = LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 });
        let mapped = spec.for_task(PredictionTask::BinaryProbability);
        assert_eq!(mapped, LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 1.0 }));
        assert_ne!(mapped.identity(), LearnerSpec::Logistic(LogisticSpec::default()).identity());

        assert!(LogisticLearner::default().fit(view, TargetView::new(&y), None, &ctx).is_err());
        let factory = resolve_for(mapped, PredictionTask::BinaryProbability).unwrap();
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "faer_ridge");
        let portable = fitted.portable().unwrap();
        let PredictionMap::Linear { coefficients, .. } = portable.model else {
            panic!("logistic exports a linear map");
        };
        assert!(coefficients[1] > 0.0 && coefficients[1] < 5.4, "slope {}", coefficients[1]);
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out[0] < 0.5 && out[n - 1] > 0.5);

        // An elastic net with an L1 part has no probability counterpart: it is left as an
        // elastic net and resolving it for the probability task refuses.
        let net = LearnerSpec::ElasticNet(ElasticNetSpec { l1_ratio: 0.5, lambda: 1.0 });
        assert_eq!(net.for_task(PredictionTask::BinaryProbability), net);
        assert!(matches!(
            resolve_for(net, PredictionTask::BinaryProbability),
            Err(LearnError::TaskMismatch { .. })
        ));
    }

    /// The forest's size and leaf size are public hyperparameters that belong to its
    /// identity (two forests that differ in either are different estimators), and the
    /// probability task defaults to a leaf size that keeps leaf means off 0 and 1.
    #[test]
    fn forest_spec_exposes_trees_and_leaf_size_in_identity() {
        let base = ForestSpec::default();
        assert_eq!(base.n_trees, 64);
        assert_eq!(
            base.resolved_min_samples_leaf(PredictionTask::BinaryProbability),
            FOREST_PROBABILITY_MIN_LEAF
        );
        assert_eq!(base.resolved_min_samples_leaf(PredictionTask::Regression), 1);
        let explicit = ForestSpec { min_samples_leaf: Some(3), ..base };
        assert_eq!(explicit.resolved_min_samples_leaf(PredictionTask::BinaryProbability), 3);
        let ids: Vec<String> = [
            base,
            ForestSpec { n_trees: 128, ..base },
            explicit,
            ForestSpec { extra_trees: true, ..base },
        ]
        .into_iter()
        .map(|spec| LearnerSpec::RandomForest(spec).identity())
        .collect();
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert!(LearnerSpec::RandomForest(ForestSpec { n_trees: 0, ..base }).validate().is_err());
        assert!(
            LearnerSpec::RandomForest(ForestSpec { min_samples_leaf: Some(0), ..base })
                .validate()
                .is_err()
        );
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
        assert_eq!(fitted.provenance().version, learner::FAER_VERSION);
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
        let even: Vec<u64> = (0..n).step_by(2).map(|i| u64::try_from(i).expect("row")).collect();
        let odd: Vec<u64> = (1..n).step_by(2).map(|i| u64::try_from(i).expect("row")).collect();
        let train = view.with_rows(RowSelection::new(&even)).unwrap();
        let test = view.with_rows(RowSelection::new(&odd)).unwrap();
        let fitted = LinearLearner.fit(train, TargetView::new(&y), None, &ctx).unwrap();
        let mut out = vec![0.0; test.nrows()];
        fitted.predict(test, &mut out, &ctx).unwrap();
        for (k, &i) in odd.iter().enumerate() {
            assert!((out[k] - y[usize::try_from(i).expect("row")]).abs() < 1e-8);
        }
        assert_eq!(train.dense().values().as_ptr(), x.as_ptr());
    }

    #[test]
    fn identity_transform_is_stateless() {
        let x = [1.0, 1.0, 0.0, 1.0];
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, 2, 2).unwrap();
        let rows = [0_u64, 1];
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
        // The target is the line 3 + 4i (range 116): every prediction is finite, the fit
        // tracks the line to within a leaf width, and it spans most of the range.
        assert!(out.iter().all(|v| v.is_finite()));
        let mae = out.iter().zip(&y).map(|(p, t)| (p - t).abs()).sum::<f64>() / n as f64;
        assert!(mae < 8.0, "mean absolute error {mae}");
        assert!(out[n - 1] - out[0] > 90.0, "span {}", out[n - 1] - out[0]);

        let mut t = vec![0.0; n];
        for (i, value) in t.iter_mut().enumerate() {
            *value = if i >= 15 { 1.0 } else { 0.0 };
        }
        let clf =
            resolve_for(LearnerSpec::GradientBoostedTrees(spec), PredictionTask::BinaryProbability)
                .unwrap();
        let fitted = clf.fit(view, TargetView::new(&t), None, &ctx).unwrap();
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
        // Labels are 0 for i < 15 and 1 from 15 on: the classes are recovered.
        assert!(out[..10].iter().all(|p| *p < 0.5), "{:?}", &out[..10]);
        assert!(out[20..].iter().all(|p| *p > 0.5), "{:?}", &out[20..]);

        let even: Vec<u64> = (0..n).step_by(2).map(|i| u64::try_from(i).expect("row")).collect();
        let train = view.with_rows(RowSelection::new(&even)).unwrap();
        let folded = factory.fit(train, TargetView::new(&y), None, &ctx).unwrap();
        let mut fold_out = vec![0.0; train.nrows()];
        folded.predict(train, &mut fold_out, &ctx).unwrap();
        assert_eq!(fold_out.len(), even.len());
    }

    #[cfg(feature = "ml-neural")]
    #[test]
    fn neural_net_fits_line_and_probabilities() {
        let n = 24usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let spec =
            NeuralSpec { hidden: 8, epochs: 60, learning_rate: 0.05, ..NeuralSpec::default() };
        let factory =
            resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::Regression).unwrap();
        let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        assert_eq!(fitted.provenance().implementation, "burn");
        let mut out = vec![0.0; n];
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let mut t = vec![0.0; n];
        for (i, value) in t.iter_mut().enumerate() {
            *value = if i >= n / 2 { 1.0 } else { 0.0 };
        }
        let clf =
            resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::BinaryProbability).unwrap();
        let fitted = clf.fit(view, TargetView::new(&t), None, &ctx).unwrap();
        fitted.predict(view, &mut out, &ctx).unwrap();
        assert!(out.iter().all(|p| *p > 0.0 && *p < 1.0));
    }

    #[cfg(feature = "ml-neural")]
    #[test]
    fn neural_device_policy_labels_the_route_that_ran() {
        use crate::DevicePolicy;
        let n = 24usize;
        let (x, y) = line_design(n);
        let ctx = ExecutionContext::for_tests(1);
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let with = |device| NeuralSpec { hidden: 4, epochs: 5, device, ..NeuralSpec::default() };
        let fit = |device| {
            resolve_for(LearnerSpec::NeuralNet(with(device)), PredictionTask::Regression)
                .unwrap()
                .fit(view, TargetView::new(&y), None, &ctx)
        };
        // The default request keeps the plain label and the same weights.
        let cpu = fit(DevicePolicy::Cpu).unwrap();
        assert_eq!(cpu.provenance().implementation, "burn");
        // A small fit under `auto` stays on the host and says why. It never claims a device.
        let auto = fit(DevicePolicy::Auto).unwrap();
        let label = auto.provenance().implementation;
        assert!(label.contains("device=host-cpu") && label.contains("requested=auto"), "{label}");
        assert!(label.contains("fallback="), "{label}");
        assert!(!label.contains("wgpu"), "{label}");
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        cpu.predict(view, &mut a, &ctx).unwrap();
        auto.predict(view, &mut b, &ctx).unwrap();
        assert_eq!(a, b, "the fallback is the same seeded cpu fit");
        // `gpu` without a linked device runtime is refused with its reason.
        #[cfg(not(feature = "ml-neural-wgpu"))]
        {
            let Err(error) = fit(DevicePolicy::Gpu) else { panic!("gpu must be refused") };
            let text = error.to_string();
            assert!(text.contains("runtime_not_compiled"), "{text}");
            assert!(text.contains("device=gpu"), "{text}");
        }
        // Device selection is part of the spec identity; the default is not.
        assert_eq!(LearnerSpec::NeuralNet(with(DevicePolicy::Cpu)).identity(), {
            LearnerSpec::NeuralNet(NeuralSpec {
                device: DevicePolicy::Cpu,
                ..with(DevicePolicy::Cpu)
            })
            .identity()
        });
        assert_ne!(
            LearnerSpec::NeuralNet(with(DevicePolicy::Cpu)).identity(),
            LearnerSpec::NeuralNet(with(DevicePolicy::Gpu)).identity()
        );
        let json = serde_json::to_string(&LearnerSpec::NeuralNet(with(DevicePolicy::Cpu))).unwrap();
        assert!(!json.contains("device"), "{json}");
        let round: LearnerSpec =
            serde_json::from_str(r#"{"kind":"neural_net","device":"auto"}"#).unwrap();
        assert!(matches!(round, LearnerSpec::NeuralNet(s) if s.device == DevicePolicy::Auto));
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
        for (i, value) in t.iter_mut().enumerate() {
            *value = if i >= n / 2 { 1.0 } else { 0.0 };
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
