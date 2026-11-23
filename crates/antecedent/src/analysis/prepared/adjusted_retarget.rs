//! Shared prepared retargeting of fitted mean models, kept distinct from DR scores.
use super::PreparedStudy;
use super::retarget::{admit_retarget_bytes, weight_output_bytes};
use crate::error::CausalError;
use antecedent_core::{ContractIdentities, ExecutionContext, VariableId};
use antecedent_estimate::DirectedAncestry;
use antecedent_io::{AdjustedFitReuseIdentityWire, PayloadDigestWire};
use std::sync::Arc;

impl PreparedStudy {
    pub(crate) fn adjusted_fit_reuse_wire(
        &self,
        identities: &ContractIdentities,
    ) -> Option<AdjustedFitReuseIdentityWire> {
        let fitted = self.execution().operation().adjusted_reuse()?;
        let rows: Vec<u64> = if fitted.design.row_selection.is_empty() {
            (0..u64::try_from(fitted.design.nrows).ok()?).collect()
        } else {
            fitted
                .design
                .row_selection
                .iter()
                .map(|r| u64::try_from(*r))
                .collect::<Result<_, _>>()
                .ok()?
        };
        Some(AdjustedFitReuseIdentityWire {
            format: 1,
            model: fitted.model.into(),
            covariance: fitted.covariance.into(),
            data_snapshot: *identities.data_snapshot.as_bytes(),
            identification: *identities.identification.as_bytes(),
            inference_binding: *identities.inference_binding.as_bytes(),
            row_index: PayloadDigestWire::u64s("adjusted_fit.row_index", &rows),
            design: PayloadDigestWire::f64s("adjusted_fit.design", &fitted.design.matrix),
            n_columns: u64::try_from(fitted.design.ncols).ok()?,
            adjustment_set: fitted.adjustment.iter().map(|v| v.raw()).collect(),
            treatment: fitted.treatment.raw(),
            outcome: fitted.outcome.raw(),
        })
    }

    pub(super) fn retarget_adjusted_model(
        &self,
        weights: &[f64],
        depends_on: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<crate::StudyResult, CausalError> {
        self.transform_capability(antecedent_contract::TransformIntent::Retarget)?;
        let operation = self.execution().operation();
        let fitted = operation.adjusted_reuse().ok_or(CausalError::Unsupported {
            message: "retarget requires frozen scores or a compatible retained adjusted mean model",
        })?;
        if weights.len() != fitted.design.nrows
            || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !weights.iter().any(|w| *w > 0.0)
        {
            return Err(CausalError::from(antecedent_estimate::EstimationError::data_msg(
                "adjusted retarget weights must be finite, non-negative, have positive mass and align with the frozen complete-case rows",
            )));
        }
        if depends_on.is_empty() && antecedent_estimate::changes_target(weights) {
            let refusal = antecedent_estimate::RetargetRefusal::UndeclaredNonconstantWeights;
            return Err(
                CausalError::Unsupported { message: refusal.as_str() }.diagnosed(refusal.fields())
            );
        }
        let graph: Option<&dyn DirectedAncestry> = self
            .analysis
            .graph
            .as_dag()
            .map(|g| g as _)
            .or_else(|| self.analysis.graph.as_admg().map(|g| g as _));
        antecedent_estimate::check_target_dependencies(
            depends_on,
            fitted.treatment,
            &[],
            fitted.adjustment,
            graph,
        )
        .map_err(CausalError::from)?;
        let scratch = fitted
            .design
            .nrows
            .checked_add(fitted.design.ncols)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| CausalError::Resource {
                message: "adjusted target scratch size overflow".into(),
            })?;
        let _scratch = admit_retarget_bytes(ctx, scratch)?;
        let output_retention = admit_retarget_bytes(
            ctx,
            u64::try_from(size_of::<antecedent_estimate::EffectEstimate>()).map_err(|_| {
                CausalError::Resource { message: "adjusted output size overflow".into() }
            })?,
        )?;
        let weight_retention = antecedent_estimate::changes_target(weights)
            .then(|| {
                admit_retarget_bytes(ctx, weight_output_bytes(weights.len(), depends_on.len())?)
            })
            .transpose()?;
        let output =
            operation.retarget_adjusted(weights, ctx)?.ok_or(CausalError::Unsupported {
                message: "the sealed operation retained no adjusted fit",
            })?;
        let diagnostics = vec![antecedent_core::Diagnostic::new(
            "retarget.adjusted_standardization",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "standardized fitted mean contrast; covariance retains the original estimator's construction, conditional on the declared fixed target law; this is not doubly robust estimation",
        )];
        let mut result =
            self.retarget_effect_to_result(output.estimate, diagnostics, weights, depends_on)?;
        result.retained_resources.extend(output.retained);
        result.retained_resources.push(Arc::new(output_retention));
        if let Some(lease) = weight_retention {
            result.retained_resources.push(Arc::new(lease));
        }
        Ok(result)
    }
}
