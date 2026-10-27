//! Click controls for one estimate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::PreparedStudy;
use antecedent_core::ExecutionContext;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{AteAnalysisResult, ate_result_from_analysis, py_execution_context_ext};

/// Execution controls for one estimate click.
///
/// Cancellation and progress ride on the execution context. A stage sink is
/// set on a private clone of the prepared study for this click only, so it is
/// never retained by the handle, a snapshot, or an exported claim.
pub(super) struct ClickControls {
    cancel: Option<antecedent_core::CancellationToken>,
    progress: Option<Arc<dyn antecedent_core::ProgressSink>>,
    stage: Option<Arc<dyn antecedent::StageResultSink>>,
}

impl ClickControls {
    pub(super) fn parse(
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<&Bound<'_, PyAny>>,
        on_stage: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            cancel: cancel.map(|token| token.inner),
            progress: crate::callbacks::progress_sink_from_py(on_progress)?,
            stage: crate::callbacks::stage_sink_from_py(on_stage)?,
        })
    }

    pub(super) fn ctx(&self, seed: u64, threads: Option<u32>) -> antecedent_core::ExecutionContext {
        py_execution_context_ext(
            seed,
            crate::resolve_user_threads(threads),
            self.cancel.clone(),
            self.progress.clone(),
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        )
    }

    /// The study this click executes: the handle itself, or a stage-streaming clone.
    pub(super) fn study(&self, inner: &Arc<PreparedStudy>) -> PyResult<Arc<PreparedStudy>> {
        match &self.stage {
            None => Ok(Arc::clone(inner)),
            Some(sink) => {
                refuse_unstaged(inner)?;
                let mut staged = (**inner).clone();
                staged.set_stage_sink(Some(crate::interrupt::bind_stage_sink(Arc::clone(sink))));
                Ok(Arc::new(staged))
            }
        }
    }

    /// Owned copy for a click that replaces retained data.
    pub(super) fn owned_study(&self, inner: &Arc<PreparedStudy>) -> PyResult<PreparedStudy> {
        let mut owned = (**inner).clone();
        if let Some(sink) = &self.stage {
            refuse_unstaged(inner)?;
            owned.set_stage_sink(Some(crate::interrupt::bind_stage_sink(Arc::clone(sink))));
        }
        Ok(owned)
    }
}

pub(super) fn map_ate(
    names: &[String],
    result: &antecedent::StudyResult,
    _prepared: &PreparedStudy,
    _ctx: &ExecutionContext,
) -> PyResult<AteAnalysisResult> {
    ate_result_from_analysis(names, result, false)
}

pub(super) fn refuse_unstaged(inner: &PreparedStudy) -> PyResult<()> {
    if inner.streams_stages() {
        return Ok(());
    }
    Err(crate::refusal(
        antecedent_core::reason_code!("stage_stream_unavailable"),
        "on_stage streams identify, estimate_point, \
         uncertainty and validate for one identification and one scalar effect whose point is \
         fitted before its uncertainty; this route fits its point and uncertainty together or \
         mixes several identifications, so it has no such stages (use on_progress)",
    ))
}

/// Strip a click-only stage sink before a study is retained.
pub(super) fn unstaged(mut study: PreparedStudy) -> PreparedStudy {
    study.set_stage_sink(None);
    study
}
