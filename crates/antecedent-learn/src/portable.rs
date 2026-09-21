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

impl FittedPredictor for PortablePredictor {
    #[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
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
                    let mut sum = 0.0;
                    let mut votes = 0usize;
                    for tree in trees {
                        let mut index = 0;
                        loop {
                            let Some(node) = tree.get(index) else {
                                return Err(LearnError::Shape {
                                    message: "prediction tree node index out of range",
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
                                PredictionNode::Split {
                                    feature,
                                    threshold,
                                    inclusive,
                                    left,
                                    right,
                                    missing,
                                } => {
                                    let v = x.get(row, feature)?;
                                    index = if v.is_nan() {
                                        missing
                                    } else if v < threshold || (inclusive && v == threshold) {
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
                    #[allow(clippy::cast_precision_loss)]
                    {
                        *value = base + if *average { sum / votes as f64 } else { sum };
                    }
                    if *logistic {
                        *value = sigmoid(*value);
                    } else if *probability {
                        *value = value.clamp(1e-9, 1.0 - 1e-9);
                    }
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

fn sigmoid(value: f64) -> f64 {
    (1.0 / (1.0 + (-value).exp())).clamp(1e-9, 1.0 - 1e-9)
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
                #[allow(clippy::cast_precision_loss)]
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
