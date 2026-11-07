//! Learner factory / fitted predictor contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::design::{DesignView, TargetView};
use crate::error::LearnError;

/// What a nuisance model must emit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictionTask {
    /// `E[Y | X]` — real-valued prediction.
    Regression,
    /// `P(T = 1 | X)` — a calibrated probability in `[0, 1]`.
    BinaryProbability,
}

/// Capabilities the planner checks before fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent switch of the record; they do not form a state machine"
)]
pub struct LearnerCapabilities {
    /// Real-valued regression.
    pub regression: bool,
    /// Calibrated binary probabilities.
    pub binary_probability: bool,
    /// Sample weights at fit time.
    pub sample_weights: bool,
    /// Native missing-value handling.
    pub missing_values: bool,
    /// Native categorical columns (no one-hot).
    pub categorical_native: bool,
    /// Deterministic given `ExecutionContext` seed.
    pub deterministic_seed: bool,
    /// Honors `ExecutionContext` thread budget.
    pub parallelism_control: bool,
}

impl LearnerCapabilities {
    /// No capabilities.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            regression: false,
            binary_probability: false,
            sample_weights: false,
            missing_values: false,
            categorical_native: false,
            deterministic_seed: false,
            parallelism_control: false,
        }
    }
}

/// Hidden implementation identity recorded on artifacts.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerProvenance {
    /// Public spec name (`"linear"`, `"gradient_boosted_trees"`, …).
    pub spec: String,
    /// Hidden provider (`"faer"`, `"forust-ml"`, …).
    pub implementation: String,
    /// Provider crate / pin version.
    pub version: String,
}

/// Unfitted learner. `antecedent-estimate` depends on this trait only.
pub trait LearnerFactory: Send + Sync {
    /// Prediction task this factory implements.
    fn task(&self) -> PredictionTask;

    /// Planner-visible capabilities.
    fn capabilities(&self) -> LearnerCapabilities;

    /// Fit on a (possibly row-selected) design.
    ///
    /// `y` and `weights` are aligned with the design's **physical** rows.
    ///
    /// # Errors
    ///
    /// Shape, task, or backend failure.
    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError>;
}

/// Fitted predictor. Writes one value per logical design row into `out`.
pub trait FittedPredictor: Send + Sync {
    /// Export a provider-independent numerical map, when supported.
    /// # Errors
    /// A provider without a codec explicitly refuses portable export.
    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Err(LearnError::Unsupported { message: "portable export unavailable for this predictor" })
    }

    /// Predict into `out` (`out.len()` must equal `x.nrows()`).
    ///
    /// # Errors
    ///
    /// Shape or backend failure.
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError>;

    /// Implementation provenance.
    fn provenance(&self) -> LearnerProvenance;
}

/// Copies only selected rows before invoking a factory or fitted predictor.
///
/// `DesignView` normally retains access to its complete backing matrix. Adapters that
/// invoke external code should use this wrapper so neither the training nor prediction
/// callback can inspect rows outside its selection through the view or target buffers.
pub struct IsolatedFactory<'a> {
    inner: &'a dyn LearnerFactory,
}

impl<'a> IsolatedFactory<'a> {
    /// Wrap a factory with row isolation.
    #[must_use]
    pub const fn new(inner: &'a dyn LearnerFactory) -> Self {
        Self { inner }
    }
}

fn isolated_bytes(x: DesignView<'_>, vectors: usize) -> Result<u64, LearnError> {
    x.ncols()
        .checked_add(vectors)
        .and_then(|width| width.checked_mul(x.nrows()))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(LearnError::Shape { message: "isolated learner materialization size overflow" })
}

fn isolated_design(x: DesignView<'_>) -> Result<Vec<f64>, LearnError> {
    let mut values = Vec::with_capacity(
        x.nrows()
            .checked_mul(x.ncols())
            .ok_or(LearnError::Shape { message: "isolated learner design size overflow" })?,
    );
    for column in 0..x.ncols() {
        for row in 0..x.nrows() {
            values.push(x.get(row, column)?);
        }
    }
    Ok(values)
}

impl LearnerFactory for IsolatedFactory<'_> {
    fn task(&self) -> PredictionTask {
        self.inner.task()
    }
    fn capabilities(&self) -> LearnerCapabilities {
        self.inner.capabilities()
    }
    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if y.len() != x.physical_nrows() || weights.is_some_and(|w| w.len() != x.physical_nrows()) {
            return Err(LearnError::Shape {
                message: "isolated learner input lengths do not match",
            });
        }
        let _hold = ctx
            .acquire_admission(0, isolated_bytes(x, 1 + usize::from(weights.is_some()))?)
            .map_err(LearnError::Admission)?;
        let matrix = isolated_design(x)?;
        let mut target = Vec::with_capacity(x.nrows());
        let mut selected_weights = weights.map(|_| Vec::with_capacity(x.nrows()));
        for row in 0..x.nrows() {
            let index = x.physical_index(row)?;
            target.push(y.values()[index]);
            if let (Some(all), Some(selected)) = (weights, &mut selected_weights) {
                selected.push(all[index]);
            }
        }
        let view = DesignView::from_column_major(&matrix, x.nrows(), x.ncols())?;
        let fitted =
            self.inner.fit(view, TargetView::new(&target), selected_weights.as_deref(), ctx)?;
        Ok(Box::new(IsolatedPredictor(fitted)))
    }
}

struct IsolatedPredictor(Box<dyn FittedPredictor>);
impl FittedPredictor for IsolatedPredictor {
    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        self.0.portable()
    }
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        let _hold =
            ctx.acquire_admission(0, isolated_bytes(x, 0)?).map_err(LearnError::Admission)?;
        let matrix = isolated_design(x)?;
        self.0.predict(DesignView::from_column_major(&matrix, x.nrows(), x.ncols())?, out, ctx)
    }
    fn provenance(&self) -> LearnerProvenance {
        self.0.provenance()
    }
}

/// faer provenance already reported by the linear, ridge, and logistic learners.
///
/// Provenance only. Coefficients do not depend on this string.
pub(crate) const FAER_VERSION: &str = "0.24";

/// Coordinate descent owned by this crate. Not a faer release.
pub(crate) const COORDINATE_DESCENT_VERSION: &str = "1";

/// Shared clip for binary probabilities so a logit never sees 0 or 1.
pub(crate) const PROB_CLIP: f64 = 1e-9;

/// Clamp `p` into `[PROB_CLIP, 1 - PROB_CLIP]`.
#[must_use]
pub(crate) fn clip_probability(p: f64) -> f64 {
    p.clamp(PROB_CLIP, 1.0 - PROB_CLIP)
}

#[cfg(test)]
mod isolation_tests {
    use super::*;
    use crate::RowSelection;

    struct CheckingFactory;
    impl LearnerFactory for CheckingFactory {
        fn task(&self) -> PredictionTask {
            PredictionTask::Regression
        }
        fn capabilities(&self) -> LearnerCapabilities {
            LearnerCapabilities {
                regression: true,
                sample_weights: true,
                ..LearnerCapabilities::none()
            }
        }
        fn fit(
            &self,
            x: DesignView<'_>,
            y: TargetView<'_>,
            weights: Option<&[f64]>,
            _: &ExecutionContext,
        ) -> Result<Box<dyn FittedPredictor>, LearnError> {
            assert!(x.row_selection().is_none());
            assert_eq!(x.dense().values(), &[10.0, 30.0]);
            assert_eq!(y.values(), &[1.0, 3.0]);
            assert_eq!(weights.unwrap(), &[2.0, 4.0]);
            Ok(Box::new(CheckingPredictor))
        }
    }
    struct CheckingPredictor;
    impl FittedPredictor for CheckingPredictor {
        fn predict(
            &self,
            x: DesignView<'_>,
            out: &mut [f64],
            _: &ExecutionContext,
        ) -> Result<(), LearnError> {
            assert!(x.row_selection().is_none());
            assert_eq!(x.dense().values(), &[20.0]);
            out.fill(1.0);
            Ok(())
        }
        fn provenance(&self) -> LearnerProvenance {
            LearnerProvenance {
                spec: "test".into(),
                implementation: "test".into(),
                version: "1".into(),
            }
        }
    }
    #[test]
    fn isolated_factory_exposes_only_selected_rows_and_weights() {
        let x = DesignView::from_column_major(&[10.0, 20.0, 30.0], 3, 1).unwrap();
        let factory = IsolatedFactory::new(&CheckingFactory);
        let ctx = ExecutionContext::for_tests(1);
        let model = factory
            .fit(
                x.with_rows(RowSelection::new(&[0, 2])).unwrap(),
                TargetView::new(&[1.0, 2.0, 3.0]),
                Some(&[2.0, 3.0, 4.0]),
                &ctx,
            )
            .unwrap();
        model.predict(x.with_rows(RowSelection::new(&[1])).unwrap(), &mut [0.0], &ctx).unwrap();
    }
}
