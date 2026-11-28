//! Re-lower a listed estimator around the identification already owned by a prepared study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;

use crate::analysis::builder::DataInput;
use crate::analysis::contract::matrix_cell;
use crate::analysis::lowering::program_is_bound_to;
use crate::analysis::prepared::IdentificationCache;
use crate::planner::lowered::{LoweredProgram, RetainedIdentification};
use crate::support::CellStatus::Supported;
use crate::{CausalError, EstimatorId, PreparedStudy};

impl PreparedStudy {
    /// Prepare a branch that executes `estimator` on this study's retained identification.
    ///
    /// The new estimator uses its default configuration. The query, structure, identifier,
    /// inference mode, validation, and rows are inherited. Execute using the ordinary prepared
    /// estimate entry points.
    ///
    /// # Errors
    /// Refuses an estimator that is unlisted or invalid for the study's cell, a route without a
    /// reusable single-structure proof, or a swap that needs a different identification claim.
    /// Also returns compilation and admission failures for the new estimation plan.
    pub fn swap_estimator(
        &self,
        estimator: EstimatorId,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let source = self.study();
        let cell = matrix_cell(source).ok_or_else(|| {
            crate::unsupported_reason!(
                "route_not_supported",
                "estimator swap requires a support cell"
            )
        })?;
        if crate::support::classify_estimator(cell, estimator) != Supported {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "the requested estimator is not listed for the prepared study's support cell"
            ));
        }
        if source.support_status() != Some(Supported)
            || source.declared.learner_factories.is_some()
            || (estimator == EstimatorId::CellAipw
                && (source.declared.score_dependence.is_some()
                    || matches!(&source.data, DataInput::Tabular(table) if table.storage().weights().is_some())))
        {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "estimator swap requires an admitted source and an estimator with supported inherited settings"
            ));
        }
        crate::strategy_table::estimator_targets_cell(cell, estimator)?;
        crate::strategy_table::estimator_targets(
            &source.query,
            &source.settings.inference,
            None,
            estimator,
        )?;

        let proof = self
            .plan()
            .lowered()
            .filter(|program| program_is_bound_to(program, source))
            .and_then(LoweredProgram::retained_identification)
            .filter(|proof| proof.key.is_some())
            .ok_or_else(no_cached_identification)?;
        require_compatible_proof(proof, estimator)?;
        let previous = self.contract()?.identities;
        if previous.identification_product.is_none() {
            return Err(no_cached_identification());
        }
        if self.plan().logical.record.estimator.as_deref() == Some(estimator.as_str()) {
            return Ok(self.clone());
        }

        let mut study = source.clone();
        study.declared.estimator = Some(estimator);
        let spec = crate::estimator_spec::EstimatorSpec::from(estimator);
        let identity = crate::analysis::declared_learners::spec_identity(&spec, None);
        study.declared.estimator_spec = Some(Arc::new(spec));
        Arc::make_mut(&mut study.config).estimator_spec_identity = Some(identity);

        let branch = study.prepare_with_swapped_estimator(proof, ctx)?;
        let binding = &branch.plan().logical.record;
        if binding.estimator.as_deref() != Some(estimator.as_str())
            || binding.identifier != self.plan().logical.record.identifier
        {
            return Err(incompatible_proof());
        }
        let next = branch.contract()?.identities;
        if previous.identification != next.identification
            || previous.identification_product != next.identification_product
            || previous.program == next.program
        {
            return Err(incompatible_proof());
        }
        Ok(branch)
    }
}

fn require_compatible_proof(
    proof: &RetainedIdentification,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    match proof.caches() {
        [IdentificationCache::Static(cache)] => {
            let method = if cache.estimand.is_adjustment_shaped() {
                Some(antecedent_expr::EstimandMethod::BackdoorAdjustment)
            } else {
                cache.estimand.method_kind().ok()
            };
            if !method.is_some_and(|method| {
                crate::strategy_table::estimand_compatible_with_estimator(method, &estimator)
            }) {
                return Err(incompatible_proof());
            }
            let (claim, estimand) =
                crate::strategy_table::select_claim(cache.identification.clone(), estimator)?;
            if estimand != cache.estimand
                || crate::analysis::identification_seal::seal_static(&claim, &estimand)?
                    != proof.static_seal().ok_or_else(no_cached_identification)?
            {
                return Err(incompatible_proof());
            }
        }
        [IdentificationCache::Temporal(cache)] if estimator.claim_restriction().is_none() => {
            for entry in cache.by_horizon.iter() {
                let estimand =
                    crate::strategy_table::select_estimand(&entry.identification, estimator)?;
                if estimand != entry.estimand
                    || !estimand.method_kind().is_ok_and(|method| {
                        crate::strategy_table::estimand_compatible_with_estimator(
                            method, &estimator,
                        )
                    })
                {
                    return Err(incompatible_proof());
                }
            }
        }
        _ => return Err(no_cached_identification()),
    }
    Ok(())
}

fn no_cached_identification() -> CausalError {
    crate::unsupported_reason!(
        "route_not_supported",
        "estimator swap requires an identification product retained for this study"
    )
}

fn incompatible_proof() -> CausalError {
    crate::unsupported_reason!(
        "strategy_incompatible",
        "the requested estimator cannot consume the same retained identification claim"
    )
}
