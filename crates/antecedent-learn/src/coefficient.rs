//! Fitted linear map shared by the coefficient learners.
//!
//! The fit stays in the caller. This type only predicts and records provenance, so two
//! learners with the same coefficients emit the same predictions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::dense::predict_linear;
use crate::design::DesignView;
use crate::error::LearnError;
use crate::learner::{self, FittedPredictor, LearnerProvenance};

/// Coefficients, link, and provenance of one linear fit.
pub(crate) struct CoefficientPredictor {
    coefficients: Vec<f64>,
    logistic: bool,
    spec: &'static str,
    implementation: &'static str,
    version: &'static str,
}

impl CoefficientPredictor {
    /// Predict from `coefficients`. `logistic` applies the clipped sigmoid after the linear map.
    pub(crate) fn new(
        coefficients: Vec<f64>,
        logistic: bool,
        spec: &'static str,
        implementation: &'static str,
        version: &'static str,
    ) -> Self {
        Self { coefficients, logistic, spec, implementation, version }
    }
}

impl FittedPredictor for CoefficientPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.coefficients, x, out)?;
        if self.logistic {
            for slot in out.iter_mut() {
                *slot = learner::clip_probability(antecedent_kernels::sigmoid(*slot));
            }
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Ok(crate::PortablePredictor {
            version: 1,
            columns: self.coefficients.len(),
            provenance: self.provenance(),
            model: crate::PredictionMap::Linear {
                coefficients: self.coefficients.clone(),
                logistic: self.logistic,
            },
        })
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: self.spec.into(),
            implementation: self.implementation.into(),
            version: self.version.into(),
        }
    }
}
