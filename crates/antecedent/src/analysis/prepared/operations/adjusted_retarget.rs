//! Same-row target standardization over successfully retained checked fits.
use super::{CheckedGlmAdjustmentOperation, CheckedLinearOperation, FrozenGlmFit, lock_frozen_fit};
use crate::analysis::prepared::sealed::{AdjustedRetargetOutput, AdjustedReuse};
use crate::error::CausalError;
use antecedent_core::{ExecutionContext, TargetPopulation};
use antecedent_estimate::LinearFitKind;
use std::sync::Arc;

fn unavailable() -> CausalError {
    crate::unsupported_reason!(
        "score_table_unavailable",
        "retarget requires a successful retained fit on the frozen rows: estimate an AllObserved mean with zero bootstrap, OLS or an unpenalized GLM, and no source sampling weights; otherwise use a prepared score table"
    )
}

impl CheckedLinearOperation {
    pub(in crate::analysis::prepared) fn adjusted_reuse_descriptor(
        &self,
    ) -> Option<AdjustedReuse<'_>> {
        (self.sealed_for_direct_execution()
            && matches!(self.fitter.fit_kind, LinearFitKind::Ols)
            && self.fitter.inference.bootstrap_replicates == 0
            && self.fitter.sampling_weights.is_none()
            && matches!(self.query.target_population, TargetPopulation::AllObserved)
            && self.query.effect_modifiers.is_empty())
        .then(|| AdjustedReuse {
            model: "additive_linear_mean_v1",
            covariance: self.fitter.inference.se_kind.as_str(),
            design: &self.preparation.problem().design,
            adjustment: &self.cache.estimand.adjustment_set,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
        })
    }
    pub(in crate::analysis::prepared) fn publish_adjusted_fit_success(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "retain_adjusted_fit" });
        }
        if let Some(fit) = lock_frozen_fit(&self.frozen_fit, ctx)?
            .as_mut()
            .filter(|f| f.kernel == ctx.kernel_policy)
        {
            fit.retargetable = true;
        }
        Ok(())
    }
    pub(in crate::analysis::prepared) fn has_adjusted_fit(&self) -> bool {
        self.adjusted_reuse_descriptor().is_some()
            && self
                .frozen_fit
                .try_lock()
                .is_ok_and(|fit| fit.as_ref().is_some_and(|f| f.retargetable))
    }
    pub(in crate::analysis::prepared) fn retarget_fitted(
        &self,
        _weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<AdjustedRetargetOutput, CausalError> {
        if self.adjusted_reuse_descriptor().is_none() {
            return Err(unavailable());
        }
        self.check_checked_target()?;
        let cache = lock_frozen_fit(&self.frozen_fit, ctx)?;
        let fit = cache
            .as_ref()
            .filter(|f| f.retargetable && f.kernel == ctx.kernel_policy)
            .ok_or_else(unavailable)?;
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
            1,
        );
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::ModelFit,
            0,
        );
        // This additive linear model's active-control contrast is constant over rows.
        // Preserve the original estimator's covariance, including its HC/cluster declaration.
        let retention = if fit.retention.belongs_to(&ctx.admission()) {
            fit.retention.observe_current();
            Arc::clone(&fit.retention)
        } else {
            let geometry = crate::analysis::compiled_linear::resources::LinearResources::prepared(
                &self.preparation,
                &self.fitter,
                ctx,
            )?;
            Arc::new(
                ctx.acquire_admission(
                    0,
                    geometry.output
                        - crate::analysis::compiled_linear::resources::RESULT_ENVELOPE_BYTES,
                )
                .map_err(|e| CausalError::Resource { message: e.to_string() })?,
            )
        };
        Ok(AdjustedRetargetOutput { estimate: fit.estimate.clone(), retained: vec![retention] })
    }
}

impl CheckedGlmAdjustmentOperation {
    pub(in crate::analysis::prepared) fn adjusted_reuse_descriptor(
        &self,
    ) -> Option<AdjustedReuse<'_>> {
        (self.sealed_for_direct_execution()
            && self.fitter.inference.bootstrap_replicates == 0
            && self.fitter.sampling_weights.is_none()
            && matches!(self.query.target_population, TargetPopulation::AllObserved)
            && self.query.effect_modifiers.is_empty()
            && self.preparation.family == self.fitter.family
            && self.preparation.adjustment_set == self.cache.estimand.adjustment_set)
            .then(|| AdjustedReuse {
                model: "glm_response_mean_v1",
                covariance: self.fitter.inference.se_kind.as_str(),
                design: &self.preparation.design,
                adjustment: &self.cache.estimand.adjustment_set,
                treatment: self.query.treatment,
                outcome: self.query.outcome,
            })
    }
    pub(in crate::analysis::prepared) fn has_adjusted_fit(&self) -> bool {
        self.adjusted_reuse_descriptor().is_some()
            && self.frozen_fit.try_lock().is_ok_and(|fit| fit.is_some())
    }
    pub(in crate::analysis::prepared) fn retarget_fitted(
        &self,
        weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<AdjustedRetargetOutput, CausalError> {
        if self.adjusted_reuse_descriptor().is_none() {
            return Err(unavailable());
        }
        let cache = lock_frozen_fit(&self.frozen_fit, ctx)?;
        let fit =
            cache.as_ref().filter(|f| f.kernel == ctx.kernel_policy).ok_or_else(unavailable)?;
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
            1,
        );
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::ModelFit,
            0,
        );
        let _import = if fit.retention.belongs_to(&ctx.admission()) {
            fit.retention.observe_current();
            None
        } else {
            Some(
                ctx.acquire_admission(0, self.retained_fit_bytes()?)
                    .map_err(|e| CausalError::Resource { message: e.to_string() })?,
            )
        };
        if fit.point.n_rows() != self.preparation.design.nrows {
            return Err(unavailable());
        }
        let (value, se) = fit.point.retarget(weights).map_err(CausalError::from)?;
        let mut estimate = fit.estimate.clone();
        estimate.ate = value;
        estimate.uncertainty.analytic.se = se;
        Ok(AdjustedRetargetOutput { estimate, retained: vec![] })
    }
    pub(in crate::analysis::prepared) fn retained_fit_bytes(&self) -> Result<u64, CausalError> {
        let n = self.preparation.design.ncols;
        n.checked_mul(n)
            .and_then(|cells| cells.checked_add(n))
            .and_then(|cells| cells.checked_add(self.preparation.design.matrix.len()))
            .and_then(|cells| cells.checked_mul(size_of::<f64>()))
            .and_then(|bytes| bytes.checked_add(size_of::<FrozenGlmFit>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| CausalError::Resource {
                message: "retained GLM coefficient law size overflow".into(),
            })
    }
    pub(in crate::analysis::prepared) fn publish_fitted_point(
        &self,
        point: antecedent_estimate::RetainedGlmPoint,
        estimate: &antecedent_estimate::EffectEstimate,
        retention: Arc<antecedent_core::execution::AdmissionLease>,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "retain_adjusted_fit" });
        }
        *lock_frozen_fit(&self.frozen_fit, ctx)? = Some(FrozenGlmFit {
            point,
            estimate: estimate.clone(),
            kernel: ctx.kernel_policy,
            retention,
        });
        Ok(())
    }
}
