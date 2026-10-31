//! Prepared-lifecycle entry of the retained summary refresh.
//!
//! The estimator a prepared study froze decides which summary its result rests on
//! ([`RefreshFamily::of_estimator`]). The study starts that summary over rows of its source,
//! applies each later change to it (append, window, evict, weight, mask, replace) and checks the
//! complete result against a fresh reconstruction, and writes it to disk under the operation's
//! admission budget. Nothing here re-identifies or recompiles the study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::PreparedStudy;
use crate::error::CausalError;
use crate::retained_refresh::{
    RefreshFamily, RefreshMutation, RefreshObservation, RetainedRefresh, SpilledRefresh,
};
use antecedent_core::ExecutionContext;
use antecedent_state::{ConjugateOlsPrior, RefreshOutcome, SpillRegistry};
use std::sync::Arc;

impl PreparedStudy {
    /// The summary family the frozen estimator keeps.
    ///
    /// # Errors
    /// The estimator's result is not a function of a least-squares or scatter summary
    /// (matching, forests, cross-fitted learners, discrete plug-ins); the next attempt is
    /// [`Self::refresh`] on the new rows.
    pub fn retained_refresh_family(&self) -> Result<RefreshFamily, CausalError> {
        let estimator = self.plan.binding().estimator.named();
        estimator.and_then(RefreshFamily::of_estimator).ok_or_else(|| CausalError::Compile {
            message: format!(
                "the estimator of this prepared study ({}) keeps no refreshable least-squares \
                 or scatter summary; re-estimate on the new rows with refresh",
                estimator.map_or("unnamed", |id| id.as_str())
            ),
        })
    }

    /// Refuse a handle of another family, or one started by another prepared study.
    fn check_handle(
        &self,
        family: RefreshFamily,
        bound_to: Option<antecedent_core::SemanticDigest>,
    ) -> Result<(), CausalError> {
        if let Some(bound) = bound_to {
            if self.plan.lowered_program_identity() != Some(bound) {
                return Err(CausalError::Compile {
                    message: "this retained refresh was started by another prepared study; \
                              start a new retained refresh from this study"
                        .into(),
                });
            }
        }
        let expected = self.retained_refresh_family()?;
        if expected == family {
            Ok(())
        } else {
            Err(CausalError::Compile {
                message: format!(
                    "this prepared study keeps a {} summary, not the {} summary of the handle; \
                     start a new retained refresh from this study",
                    expected.as_str(),
                    family.as_str()
                ),
            })
        }
    }

    /// Start the summary this study's estimator keeps over `rows` of `source`, `width`
    /// columns wide. A conjugate study also names its prior and the label of the first block.
    ///
    /// # Errors
    /// The refusals of [`Self::retained_refresh_family`] and [`RetainedRefresh::start`].
    pub fn start_retained_refresh(
        &self,
        source: &str,
        width: usize,
        rows: &[RefreshObservation],
        conjugate: Option<(&ConjugateOlsPrior, &str)>,
    ) -> Result<RetainedRefresh, CausalError> {
        let family = self.retained_refresh_family()?;
        Ok(RetainedRefresh::start(family, source, width, rows, conjugate)?
            .bind(self.plan.lowered_program_identity()))
    }

    /// Apply one change to a summary started by [`Self::start_retained_refresh`] and compare
    /// the complete result with a fresh reconstruction from the retained rows. An append
    /// updates in place; every other change recomputes from the rows. `refresh` is unchanged
    /// when the change or the comparison is refused.
    ///
    /// # Errors
    /// A handle of another family or another prepared study, and the refusals of [`RetainedRefresh::apply_verified`].
    pub fn apply_retained_refresh(
        &self,
        refresh: &mut RetainedRefresh,
        mutation: &RefreshMutation,
    ) -> Result<RefreshOutcome, CausalError> {
        self.check_handle(refresh.family(), refresh.bound_to())?;
        refresh.apply_verified(mutation)
    }

    /// Write the summary to disk as this study's disk-backed result, under `ctx`'s admission
    /// budget and `registry`'s byte budget.
    ///
    /// # Errors
    /// A handle of another family or another prepared study, and the refusals of [`RetainedRefresh::spill`].
    pub fn spill_retained_refresh(
        &self,
        refresh: &RetainedRefresh,
        ctx: &ExecutionContext,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpilledRefresh, CausalError> {
        self.check_handle(refresh.family(), refresh.bound_to())?;
        refresh.spill(ctx, registry, key)
    }

    /// Read a spilled summary back under `ctx`'s admission budget.
    ///
    /// # Errors
    /// A handle of another family or another prepared study, and the refusals of [`SpilledRefresh::reload`].
    pub fn reload_retained_refresh(
        &self,
        spilled: &SpilledRefresh,
        ctx: &ExecutionContext,
    ) -> Result<RetainedRefresh, CausalError> {
        self.check_handle(spilled.family(), spilled.bound_to())?;
        spilled.reload(ctx)
    }
}
