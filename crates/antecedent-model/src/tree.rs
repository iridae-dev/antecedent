//! Tree-ensemble conditional means for mechanism slots.
//!
//! A [`TreeEnsemble`] is a validated, provider-independent tree map
//! ([`antecedent_learn::PortablePredictor`]) over a node's parent columns. It
//! carries no foreign model: the fitted ensemble is exported once at fit time, so
//! evaluating, sampling, abducting and serializing a mechanism never depend on
//! which provider fitted it, or on the provider being compiled in.
//!
//! The ensemble supplies the *conditional mean* (additive regression slot) or a
//! *class score* (categorical slot). The sampling distribution, noise semantics
//! and log density live with the slot kernels in [`crate::mechanism`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_learn::{PortablePredictor, PredictionMap};

use crate::batch::ParentBatch;
use crate::error::ModelError;

/// A validated tree ensemble over a mechanism's parent columns.
///
/// Prediction is piecewise constant: outside the training support a parent value
/// falls into an extreme leaf, so the surface is flat there. A hard intervention
/// to a value beyond the observed parent range therefore reads the boundary
/// leaf, not an extrapolated trend.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeEnsemble {
    predictor: Arc<PortablePredictor>,
}

impl TreeEnsemble {
    /// Accept an exported ensemble.
    ///
    /// # Errors
    ///
    /// [`ModelError::Shape`] when the payload fails its structural validation, is
    /// not a tree map, or reads no parent column.
    pub fn new(predictor: PortablePredictor) -> Result<Self, ModelError> {
        predictor
            .validate()
            .map_err(|e| ModelError::Shape { message: format!("invalid tree ensemble: {e}") })?;
        if !matches!(predictor.model, PredictionMap::Trees { .. }) || predictor.columns == 0 {
            return Err(ModelError::Shape {
                message: "tree ensemble must be a tree map over at least one parent column".into(),
            });
        }
        Ok(Self { predictor: Arc::new(predictor) })
    }

    /// Parent columns the ensemble reads.
    #[must_use]
    pub fn n_parents(&self) -> usize {
        self.predictor.columns
    }

    /// The exported provider-independent payload.
    #[must_use]
    pub fn predictor(&self) -> &PortablePredictor {
        &self.predictor
    }

    /// Public learner name that fitted the ensemble (`gradient_boosted_trees`,
    /// `random_forest`), for provenance.
    #[must_use]
    pub fn learner(&self) -> &str {
        &self.predictor.provenance.spec
    }

    /// Predict every row of `parents` into `out[..parents.n_rows]`.
    ///
    /// # Errors
    ///
    /// Parent arity or buffer mismatch, or a prediction failure.
    pub fn predict_column(
        &self,
        parents: ParentBatch<'_>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        let n = parents.n_rows;
        if parents.n_parents != self.n_parents() {
            return Err(ModelError::Shape {
                message: "tree ensemble arity does not match the parent batch".into(),
            });
        }
        if parents.values.len() < n.saturating_mul(parents.n_parents) || out.len() < n {
            return Err(ModelError::Shape { message: "tree ensemble buffers too short".into() });
        }
        self.predictor
            .predict_trees_with(n, &|row, feature| parents.values[feature * n + row], out)
            .map_err(|e| ModelError::Numerical { message: format!("tree ensemble: {e}") })
    }
}

/// Class probabilities of a tree-classifier categorical mechanism at every row,
/// row-major `[row * k + category]`.
///
/// A binary support has one classifier scoring `P(support[1])`. A support of
/// `k > 2` levels has one one-vs-rest classifier per level, and the `k` scores
/// are renormalised to a distribution, so each row's probabilities are positive
/// (the providers clip scores away from 0 and 1) and sum to one.
///
/// # Errors
///
/// Classifier count or arity inconsistent with the support, or a prediction
/// failure.
pub(crate) fn class_probabilities(
    support_len: usize,
    classifiers: &[TreeEnsemble],
    parents: ParentBatch<'_>,
    out: &mut [f64],
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    let k = support_len;
    let expected = if k == 2 { 1 } else { k };
    if k < 2 || classifiers.len() != expected {
        return Err(ModelError::Shape {
            message: "tree classifier count does not match the support".into(),
        });
    }
    if out.len() < n.saturating_mul(k) {
        return Err(ModelError::Shape { message: "class probability buffer too short".into() });
    }
    let mut score = vec![0.0; n];
    for (c, classifier) in classifiers.iter().enumerate() {
        classifier.predict_column(parents, &mut score)?;
        for (r, &q) in score.iter().enumerate() {
            out[r * k + c] = q;
        }
    }
    for r in 0..n {
        let row = &mut out[r * k..(r + 1) * k];
        if k == 2 {
            let q = row[0];
            row[0] = 1.0 - q;
            row[1] = q;
        } else {
            let total: f64 = row.iter().sum();
            for p in row.iter_mut() {
                *p /= total;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::CausalRng;
    use antecedent_learn::{LearnerProvenance, PredictionNode};

    use super::*;
    use crate::batch::MechanismWorkspace;
    use crate::compile::MechanismSlot;
    use crate::mechanism::{
        NoiseInferenceMode, evaluate_column, infer_noise_column, infer_noise_column_rng,
        log_prob_column, sample_noise_column,
    };

    /// One stump on parent 0: leaf `low` at `x <= threshold`, `high` above.
    fn stump(threshold: f64, low: f64, high: f64, probability: bool) -> TreeEnsemble {
        let predictor = PortablePredictor {
            version: 1,
            columns: 1,
            provenance: LearnerProvenance {
                spec: "random_forest".into(),
                implementation: "test".into(),
                version: "1".into(),
            },
            model: PredictionMap::Trees {
                trees: vec![vec![
                    PredictionNode::Split {
                        feature: 0,
                        threshold,
                        inclusive: true,
                        left: 1,
                        right: 2,
                        missing: 2,
                    },
                    PredictionNode::Leaf { value: Some(low) },
                    PredictionNode::Leaf { value: Some(high) },
                ]],
                base: 0.0,
                average: true,
                logistic: false,
                probability,
            },
        };
        TreeEnsemble::new(predictor).unwrap()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn batch(values: &[f64]) -> ParentBatch<'_> {
        ParentBatch { n_rows: values.len(), n_parents: 1, values }
    }

    #[test]
    fn additive_tree_inverts_noise_exactly_and_replays_the_observation() {
        let slot =
            MechanismSlot::TreeAdditive { ensemble: stump(0.0, -1.0, 3.0, false), sigma: 0.5 };
        let parents = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let observed = [-1.4, -0.7, -1.1, 3.6, 2.2];
        let mut noise = [0.0; 5];
        infer_noise_column(&slot, &observed, batch(&parents), &mut noise).unwrap();
        // The conditional mean is -1 for x <= 0 and 3 above; the noise is the residual.
        for (n, expected) in noise.iter().zip([-0.4, 0.3, -0.1, 0.6, -0.8]) {
            assert!((n - expected).abs() < 1e-12, "{n} vs {expected}");
        }
        let mut replay = [0.0; 5];
        evaluate_column(
            &slot,
            batch(&parents),
            &noise,
            &mut replay,
            &mut MechanismWorkspace::default(),
        )
        .unwrap();
        assert_eq!(bits(&replay), bits(&observed));
        let mode = infer_noise_column_rng(
            &slot,
            &observed,
            batch(&parents),
            &mut noise,
            &mut CausalRng::from_seed(1),
        )
        .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Invertible);
    }

    #[test]
    fn additive_tree_density_is_gaussian_around_the_ensemble_mean() {
        let slot =
            MechanismSlot::TreeAdditive { ensemble: stump(0.0, -1.0, 3.0, false), sigma: 0.5 };
        let parents = [-1.0, 1.0];
        let observed = [-1.0, 3.5];
        let mut lp = [0.0; 2];
        log_prob_column(&slot, &observed, batch(&parents), &mut lp).unwrap();
        let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5_f64.ln();
        assert!((lp[0] - log_norm).abs() < 1e-12);
        assert!((lp[1] - (log_norm - 0.5)).abs() < 1e-12);
        let mut noise = vec![0.0; 20_000];
        sample_noise_column(&slot, noise.len(), &mut CausalRng::from_seed(3), &mut noise).unwrap();
        let sd = (noise.iter().map(|e| e * e).sum::<f64>() / noise.len() as f64).sqrt();
        assert!((sd - 0.5).abs() < 0.02, "sampled noise sd {sd}");
    }

    #[test]
    fn binary_tree_classifier_is_a_normalised_categorical_with_posterior_abduction() {
        // P(support[1]) = 0.2 at x <= 0 and 0.9 above.
        let slot = MechanismSlot::TreeCategorical {
            support: Arc::from([0.0, 1.0]),
            probs: Arc::from([0.5, 0.5]),
            classifiers: Arc::from([stump(0.0, 0.2, 0.9, true)]),
        };
        let parents = [-1.0, 1.0];
        let mut total = [0.0; 2];
        for (k, value) in [0.0, 1.0].into_iter().enumerate() {
            let mut lp = [0.0; 2];
            log_prob_column(&slot, &[value, value], batch(&parents), &mut lp).unwrap();
            for (t, l) in total.iter_mut().zip(lp) {
                *t += l.exp();
            }
            let expected = [[0.8, 0.1], [0.2, 0.9]][k];
            for (l, e) in lp.iter().zip(expected) {
                assert!((l.exp() - e).abs() < 1e-12);
            }
        }
        assert!(total.iter().all(|t| (t - 1.0).abs() < 1e-12));

        let observed = [1.0, 0.0];
        let mut noise = [0.0; 2];
        let mode = infer_noise_column_rng(
            &slot,
            &observed,
            batch(&parents),
            &mut noise,
            &mut CausalRng::from_seed(7),
        )
        .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Posterior);
        // The abduced noise lies in the observed category's CDF bin: replaying it returns
        // the observed values, and the bin edges are the model probabilities.
        assert!(
            noise[0] > 0.8 && noise[0] < 1.0,
            "bin of category 1 at x<=0 is (0.8, 1): {}",
            noise[0]
        );
        assert!(
            noise[1] > 0.0 && noise[1] < 0.1,
            "bin of category 0 at x>0 is (0, 0.1): {}",
            noise[1]
        );
        let mut replay = [0.0; 2];
        evaluate_column(
            &slot,
            batch(&parents),
            &noise,
            &mut replay,
            &mut MechanismWorkspace::default(),
        )
        .unwrap();
        assert_eq!(bits(&replay), bits(&observed));
        assert!(infer_noise_column(&slot, &observed, batch(&parents), &mut noise).is_err());
    }

    #[test]
    fn one_vs_rest_classifiers_are_renormalised_per_row() {
        let slot = MechanismSlot::TreeCategorical {
            support: Arc::from([0.0, 1.0, 2.0]),
            probs: Arc::from([0.3, 0.3, 0.4]),
            classifiers: Arc::from([
                stump(0.0, 0.6, 0.1, true),
                stump(0.0, 0.2, 0.2, true),
                stump(0.0, 0.2, 0.7, true),
            ]),
        };
        let parents = [-1.0, 1.0];
        let mut row = [0.0; 6];
        class_probabilities(3, &[], batch(&parents), &mut row).unwrap_err();
        let MechanismSlot::TreeCategorical { classifiers, .. } = &slot else { unreachable!() };
        class_probabilities(3, classifiers, batch(&parents), &mut row).unwrap();
        for (got, want) in row.iter().zip([0.6, 0.2, 0.2, 0.1, 0.2, 0.7]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert!((row[3] + row[4] + row[5] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn arity_and_payload_are_validated() {
        let ensemble = stump(0.0, 0.0, 1.0, false);
        let mut out = [0.0; 2];
        let wrong = ParentBatch { n_rows: 2, n_parents: 2, values: &[0.0; 4] };
        assert!(ensemble.predict_column(wrong, &mut out).is_err());
        let mut linear = ensemble.predictor().clone();
        linear.model = PredictionMap::Linear { coefficients: vec![1.0], logistic: false };
        assert!(TreeEnsemble::new(linear).is_err());
        let mut cyclic = ensemble.predictor().clone();
        if let PredictionMap::Trees { trees, .. } = &mut cyclic.model {
            trees[0][0] = PredictionNode::Split {
                feature: 0,
                threshold: 0.0,
                inclusive: true,
                left: 0,
                right: 0,
                missing: 0,
            };
        }
        assert!(TreeEnsemble::new(cyclic).is_err());
    }
}
