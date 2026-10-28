//! What the registry hands the tree providers, and what it does with their fits.
//!
//! The providers (forust-ml, smartcore) grow the trees; this repository chooses the
//! hyperparameters, the seed, the rows and the noise scale. Here an independent re-derivation of
//! those choices, written from the registry's documentation rather than its code, is compared with
//! the registry's own fit:
//! * the stored ensemble equals, exactly, a direct provider fit on all rows with the documented
//!   settings (boosted: 100 trees, depth 3, learning rate 0.1; forest: 100 trees, leaves of at
//!   least five rows for regression and the probability default for classification; the fixed
//!   fit seed and one thread), and differs from a fit with another leaf size, so the comparison
//!   can fail;
//! * the additive-noise scale is the root mean square of the out-of-fold residuals of the
//!   declared time-ordered folds (blocks of n/6 rows, each fold trained on the rows before its
//!   block, folds with fewer than twenty training rows skipped), floored at 1e-8 times the
//!   outcome's root mean square;
//! * a categorical fit stores one classifier for a binary outcome (the upper level) and one per
//!   level otherwise, each fit to the 0/1 indicator of its level.
//!
//! Needs the `ml-gbdt` and `ml-forest` features.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(all(feature = "ml-gbdt", feature = "ml-forest"))]

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_learn::{
    DesignView, ForestSpec, GbtSpec, LearnerSpec, PortablePredictor, PredictionTask, TargetView,
    resolve_for,
};
use antecedent_model::{
    CompiledCausalModel, MechanismFamily, MechanismRegistry, MechanismSlot, NodeFamilyConstraint,
    SelectionOptions, SelectionPolicy, ValidationSplits,
};

const SEED: u64 = 0x5452_4545_4D45_0001;

fn table(x: &[f64], y: &[f64], outcome: ValueType) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for (name, vt) in [("x", ValueType::Continuous), ("y", outcome)] {
        b.add_variable(
            name,
            vt,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let validity = ValidityBitmap::all_valid(x.len());
    let cols = [x, y]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.to_vec()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap())
}

fn direct(spec: LearnerSpec, task: PredictionTask, x: &[f64], y: &[f64]) -> PortablePredictor {
    let factory = resolve_for(spec, task).unwrap();
    let view = DesignView::from_column_major(x, x.len(), 1).unwrap();
    let ctx = ExecutionContext::production(SEED, 1);
    factory.fit(view, TargetView::new(y), None, &ctx).unwrap().portable().unwrap()
}

fn predict(p: &PortablePredictor, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    p.predict_trees_with(x.len(), &|row, _| x[row], &mut out).unwrap();
    out
}

fn fit(
    family: MechanismFamily,
    data: &TabularData,
    splits: ValidationSplits,
) -> antecedent_model::CompiledMechanismStore {
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let options = SelectionOptions {
        constraints: vec![
            (
                VariableId::from_raw(0),
                NodeFamilyConstraint {
                    allowed: Some(Arc::from([MechanismFamily::LinearGaussian])),
                    ..NodeFamilyConstraint::default()
                },
            ),
            (
                VariableId::from_raw(1),
                NodeFamilyConstraint { required: Some(family), ..NodeFamilyConstraint::default() },
            ),
        ],
        splits,
    };
    MechanismRegistry::standard()
        .assign_and_fit_with(&model, data, SelectionPolicy::BestScore, &options)
        .unwrap()
        .0
}

fn series() -> (Vec<f64>, Vec<f64>) {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| f64::from(i) / 8.0 + f64::from((i * 7) % 5) / 64.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| if *v < 3.5 { 1.0 } else { 3.0 } + f64::from((i * 13 % 11) as u32) / 16.0)
        .collect();
    (x, y)
}

#[test]
fn registry_tree_fit_wiring_reference_additive_fits_use_the_documented_settings_and_folds() {
    let (x, y) = series();
    let n = x.len();
    let data = table(&x, &y, ValueType::Continuous);
    let cases = [
        (
            MechanismFamily::TreeForestAdditive,
            LearnerSpec::RandomForest(ForestSpec {
                extra_trees: false,
                n_trees: 100,
                min_samples_leaf: Some(5),
            }),
        ),
        (
            MechanismFamily::TreeBoostedAdditive,
            LearnerSpec::GradientBoostedTrees(GbtSpec { trees: 100, depth: 3, learning_rate: 0.1 }),
        ),
    ];
    for (family, spec) in cases {
        let store = fit(family, &data, ValidationSplits::TimeOrdered { purge: 0 });
        let MechanismSlot::TreeAdditive { ensemble, sigma } = &store.slots[1] else {
            panic!("{}: an additive tree slot", family.id());
        };
        let want = direct(spec, PredictionTask::Regression, &x, &y);
        assert_eq!(ensemble.predictor(), &want, "{}: the ensemble on all rows", family.id());
        if let LearnerSpec::RandomForest(_) = spec {
            let other = LearnerSpec::RandomForest(ForestSpec {
                extra_trees: false,
                n_trees: 100,
                min_samples_leaf: Some(1),
            });
            assert_ne!(
                &direct(other, PredictionTask::Regression, &x, &y),
                ensemble.predictor(),
                "the leaf size must matter for the comparison to be able to fail"
            );
        }
        // Out-of-fold residuals of the declared folds: block j covers rows [j n / 6, (j+1) n / 6).
        let edge = |j: usize| j * n / 6;
        let (mut sum_sq, mut scored) = (0.0, 0usize);
        for k in 1..6 {
            let train: Vec<usize> = (0..edge(k)).collect();
            let test: Vec<usize> = (edge(k)..edge(k + 1)).collect();
            if train.len() < 20 {
                continue;
            }
            let xt: Vec<f64> = train.iter().map(|&r| x[r]).collect();
            let yt: Vec<f64> = train.iter().map(|&r| y[r]).collect();
            let fold = direct(spec, PredictionTask::Regression, &xt, &yt);
            let xs: Vec<f64> = test.iter().map(|&r| x[r]).collect();
            for (&r, m) in test.iter().zip(predict(&fold, &xs)) {
                sum_sq += (y[r] - m).powi(2);
                scored += 1;
            }
        }
        let rms_y = (y.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let want_sigma = (sum_sq / scored as f64).sqrt().max(1e-8 * rms_y);
        assert!(
            (sigma - want_sigma).abs() <= 1e-12 * want_sigma,
            "{}: noise scale {sigma} vs {want_sigma}",
            family.id()
        );
        assert_eq!(scored, 40, "four folds of ten rows");
    }
}

#[test]
fn registry_tree_fit_wiring_reference_classifiers_fit_the_indicator_of_each_level() {
    let (x, _) = series();
    let n = x.len();
    // Two levels, then three: the level follows the parent with a deterministic wobble.
    let two: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| f64::from(u8::from(*v + f64::from((i % 3) as u32) / 4.0 > 3.6)))
        .collect();
    let three: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let w = *v + f64::from((i % 4) as u32) / 4.0;
            if w < 2.4 {
                1.0
            } else if w < 5.0 {
                2.0
            } else {
                4.0
            }
        })
        .collect();
    for (family, learner) in [
        (
            MechanismFamily::TreeForestCategorical,
            LearnerSpec::RandomForest(ForestSpec {
                extra_trees: false,
                n_trees: 100,
                min_samples_leaf: None,
            }),
        ),
        (
            MechanismFamily::TreeBoostedCategorical,
            LearnerSpec::GradientBoostedTrees(GbtSpec { trees: 100, depth: 3, learning_rate: 0.1 }),
        ),
    ] {
        for y in [&two, &three] {
            let data = table(&x, y, ValueType::Binary);
            let store = fit(family, &data, ValidationSplits::Shuffled);
            let MechanismSlot::TreeCategorical { support, classifiers, .. } = &store.slots[1]
            else {
                panic!("{}: a categorical tree slot", family.id());
            };
            let mut levels = y.to_vec();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            assert_eq!(support.to_vec(), levels, "{}", family.id());
            let wanted_levels: Vec<usize> =
                if levels.len() == 2 { vec![1] } else { (0..levels.len()).collect() };
            assert_eq!(classifiers.len(), wanted_levels.len(), "{}", family.id());
            for (classifier, level) in classifiers.iter().zip(wanted_levels) {
                let target: Vec<f64> =
                    y.iter().map(|v| f64::from(u8::from(*v == levels[level]))).collect();
                let want = direct(learner, PredictionTask::BinaryProbability, &x, &target);
                assert_eq!(classifier.predictor(), &want, "{}: level {level}", family.id());
            }
            assert_eq!(x.len(), n);
        }
    }
}
