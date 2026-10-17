//! Provider-independent, validated prediction state. No executable payloads.

use crate::{DesignView, FittedPredictor, LearnError, LearnerProvenance};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

/// One finite decision-tree node. Child indices must point forward.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PredictionNode {
    /// A missing honest leaf contributes no vote.
    Leaf {
        /// Predicted value.
        value: Option<f64>,
    },
    /// Binary split with an explicit missing-value branch.
    Split {
        /// Design column.
        feature: usize,
        /// Split threshold.
        threshold: f64,
        /// Include equality in the left branch.
        inclusive: bool,
        /// Left child index.
        left: usize,
        /// Right child index.
        right: usize,
        /// Missing child index.
        missing: usize,
    },
}

/// Versioned numerical prediction map. Schema and causal identity belong to its owner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortablePredictor {
    /// Codec version.
    pub version: u32,
    /// Exact design width, including any intercept.
    pub columns: usize,
    /// Fit implementation, retained for provenance rather than dispatch.
    pub provenance: LearnerProvenance,
    /// Numerical map.
    pub model: PredictionMap,
}

/// Supported provider-independent numerical maps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PredictionMap {
    /// Linear predictor, optionally mapped through a logistic link.
    Linear {
        /// Coefficients aligned with the design.
        coefficients: Vec<f64>,
        /// Apply sigmoid and the provider's probability clipping.
        logistic: bool,
    },
    /// Ordered ensemble with sum or valid-vote mean aggregation.
    Trees {
        /// Flattened trees.
        trees: Vec<Vec<PredictionNode>>,
        /// Initial prediction.
        base: f64,
        /// Average nonmissing votes instead of summing.
        average: bool,
        /// Apply a sigmoid to the aggregate.
        logistic: bool,
        /// Clip an already probabilistic aggregate.
        probability: bool,
    },
}

impl PortablePredictor {
    /// Validate all payload structure before prediction or acceptance.
    /// # Errors
    /// Unsupported version, invalid numbers, shapes, or cyclic/backward tree links.
    pub fn validate(&self) -> Result<(), LearnError> {
        let invalid = || LearnError::Shape { message: "invalid portable predictor" };
        if self.version != 1 {
            return Err(invalid());
        }
        match &self.model {
            PredictionMap::Linear { coefficients, .. } => {
                if coefficients.len() != self.columns || coefficients.iter().any(|x| !x.is_finite())
                {
                    return Err(invalid());
                }
            }
            PredictionMap::Trees { trees, base, .. } => {
                if trees.is_empty() || !base.is_finite() {
                    return Err(invalid());
                }
                for tree in trees {
                    if tree.is_empty() {
                        return Err(invalid());
                    }
                    for (i, node) in tree.iter().enumerate() {
                        match node {
                            PredictionNode::Leaf { value } => {
                                if value.is_some_and(|v| !v.is_finite()) {
                                    return Err(invalid());
                                }
                            }
                            PredictionNode::Split {
                                feature,
                                threshold,
                                left,
                                right,
                                missing,
                                ..
                            } => {
                                if *feature >= self.columns
                                    || !threshold.is_finite()
                                    || [left, right, missing]
                                        .iter()
                                        .any(|c| **c <= i || **c >= tree.len())
                                {
                                    return Err(invalid());
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl PortablePredictor {
    /// Predict a tree-ensemble map for `n_rows` rows read through `value(row, feature)`.
    ///
    /// For callers that hold their design in a layout [`DesignView`] does not cover
    /// (a mechanism's gathered parent columns). The payload is not re-validated per
    /// call: [`Self::validate`] it once on acceptance, and make sure `value` answers
    /// every feature index below the payload's `columns` for every row below `n_rows`.
    ///
    /// # Errors
    /// A linear map, a short `out`, a tree link out of range, no supporting leaf at
    /// a row, or a non-finite prediction.
    pub fn predict_trees_with(
        &self,
        n_rows: usize,
        value: &dyn Fn(usize, usize) -> f64,
        out: &mut [f64],
    ) -> Result<(), LearnError> {
        let PredictionMap::Trees { trees, base, average, logistic, probability } = &self.model
        else {
            return Err(LearnError::Unsupported { message: "not a tree-ensemble prediction map" });
        };
        if out.len() < n_rows {
            return Err(LearnError::Shape { message: "predict out shorter than rows" });
        }
        for (row, slot) in out.iter_mut().enumerate().take(n_rows) {
            let aggregation = TreeAggregation {
                base: *base,
                average: *average,
                logistic: *logistic,
                probability: *probability,
            };
            let predicted =
                tree_ensemble_value(trees, aggregation, &mut |feature| Ok(value(row, feature)))?;
            if !predicted.is_finite() {
                return Err(LearnError::Shape { message: "nonfinite prediction" });
            }
            *slot = predicted;
        }
        Ok(())
    }
}

/// How an ensemble's per-tree leaf values combine into one prediction.
#[derive(Clone, Copy)]
struct TreeAggregation {
    base: f64,
    average: bool,
    logistic: bool,
    probability: bool,
}

/// One row's prediction from an ensemble; `feature_value(f)` reads the row's column `f`.
fn tree_ensemble_value(
    trees: &[Vec<PredictionNode>],
    aggregation: TreeAggregation,
    feature_value: &mut dyn FnMut(usize) -> Result<f64, LearnError>,
) -> Result<f64, LearnError> {
    let mut sum = 0.0;
    let mut votes = 0usize;
    for tree in trees {
        let mut index = 0;
        loop {
            let Some(node) = tree.get(index) else {
                return Err(LearnError::Shape {
                    message: "prediction tree child index out of range",
                });
            };
            match *node {
                PredictionNode::Leaf { value } => {
                    if let Some(v) = value {
                        sum += v;
                        votes += 1;
                    }
                    break;
                }
                PredictionNode::Split { feature, threshold, inclusive, left, right, missing } => {
                    let v = feature_value(feature)?;
                    #[allow(
                        clippy::float_cmp,
                        reason = "an inclusive split sends a value equal to the stored threshold left, so exact equality is the contract"
                    )]
                    let goes_left = !v.is_nan() && (v < threshold || (inclusive && v == threshold));
                    index = if v.is_nan() {
                        missing
                    } else if goes_left {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }
    if votes == 0 {
        return Err(LearnError::Unsupported {
            message: "no supported prediction at requested row",
        });
    }
    let mut value = aggregation.base + if aggregation.average { sum / votes as f64 } else { sum };
    if aggregation.logistic {
        value = sigmoid(value);
    } else if aggregation.probability {
        value = crate::learner::clip_probability(value);
    }
    Ok(value)
}

impl FittedPredictor for PortablePredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        self.validate()?;
        if x.ncols() != self.columns || out.len() != x.nrows() {
            return Err(LearnError::Shape {
                message: "prediction schema does not match fitted design",
            });
        }
        match &self.model {
            PredictionMap::Linear { coefficients, logistic } => {
                crate::dense::predict_linear(coefficients, x, out)?;
                if *logistic {
                    for value in out.iter_mut() {
                        *value = sigmoid(*value);
                    }
                }
            }
            PredictionMap::Trees { trees, base, average, logistic, probability } => {
                for (row, value) in out.iter_mut().enumerate() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(LearnError::Unsupported { message: "prediction cancelled" });
                    }
                    *value = tree_ensemble_value(
                        trees,
                        TreeAggregation {
                            base: *base,
                            average: *average,
                            logistic: *logistic,
                            probability: *probability,
                        },
                        &mut |feature| x.get(row, feature),
                    )?;
                }
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Unsupported { message: "prediction cancelled" });
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(LearnError::Shape { message: "nonfinite prediction" });
        }
        Ok(())
    }
    fn provenance(&self) -> LearnerProvenance {
        self.provenance.clone()
    }
    fn portable(&self) -> Result<PortablePredictor, LearnError> {
        self.validate()?;
        Ok(self.clone())
    }
}

/// The stable logistic with the provider's explicit probability clip.
fn sigmoid(value: f64) -> f64 {
    crate::learner::clip_probability(antecedent_kernels::sigmoid(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LearnerSpec, PredictionTask, TargetView, resolve_for};

    #[test]
    fn portable_cpu_predictions_match_fitted_providers() {
        let n = 64;
        let mut x = vec![1.0; n * 2];
        let y: Vec<_> = (0..n)
            .map(|i| {
                let v = i as f64 / 10.0;
                x[n + i] = v;
                2.0 + v.sin()
            })
            .collect();
        let view = DesignView::from_column_major(&x, n, 2).unwrap();
        let ctx = ExecutionContext::for_tests(18);
        for name in ["linear", "ridge", "elastic_net", "gradient_boosted_trees", "random_forest"] {
            let spec = LearnerSpec::parse(name).unwrap();
            let factory = match resolve_for(spec, PredictionTask::Regression) {
                Ok(f) => f,
                Err(LearnError::ProviderUnavailable { .. }) => continue,
                Err(e) => panic!("{name}: {e}"),
            };
            let fitted = factory.fit(view, TargetView::new(&y), None, &ctx).unwrap();
            let portable = fitted.portable().unwrap();
            let bytes = serde_json::to_vec(&portable).unwrap();
            let loaded: PortablePredictor = serde_json::from_slice(&bytes).unwrap();
            let mut before = vec![0.0; n];
            let mut after = vec![0.0; n];
            fitted.predict(view, &mut before, &ctx).unwrap();
            loaded.predict(view, &mut after, &ctx).unwrap();
            for (a, b) in before.iter().zip(&after) {
                assert!((a - b).abs() < 1e-12, "{name}: {a} != {b}");
            }
        }
    }

    #[test]
    fn malformed_links_and_versions_are_rejected() {
        let mut model = PortablePredictor {
            version: 1,
            columns: 1,
            provenance: LearnerProvenance {
                spec: "test".into(),
                implementation: "test".into(),
                version: "1".into(),
            },
            model: PredictionMap::Trees {
                trees: vec![vec![PredictionNode::Split {
                    feature: 0,
                    threshold: 0.0,
                    inclusive: true,
                    left: 0,
                    right: 0,
                    missing: 0,
                }]],
                base: 0.0,
                average: true,
                logistic: false,
                probability: false,
            },
        };
        assert!(model.validate().is_err());
        model.model = PredictionMap::Linear { coefficients: vec![1.0], logistic: false };
        assert!(model.validate().is_ok());
        model.version = 2;
        assert!(model.validate().is_err());
    }
}
