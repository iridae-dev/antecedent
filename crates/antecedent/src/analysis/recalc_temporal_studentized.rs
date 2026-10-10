//! Exact balanced-unit variance authority for the original checked two-step adapter.
//! The IID unit mean bootstrap-t is equal-tailed, not a generic finite-sample guarantee.
use super::{CheckedUnitDraw, EstimationError, ExecutionContext, UnitHistories};
use antecedent_estimate::temporal_dependent_interval::{
    TemporalEstimator, balanced_linear_unit_scores,
};
pub(super) struct CheckedStudentizedUnitDraw<'a>(pub(super) CheckedUnitDraw<'a>);
impl TemporalEstimator for CheckedStudentizedUnitDraw<'_> {
    fn label(&self) -> &'static str {
        self.0.label()
    }
    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        self.0.estimate(units)
    }
    fn balanced_unit_scores(
        &self,
        units: &[&UnitHistories],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        balanced_linear_unit_scores(self, units, ctx)
    }
}
