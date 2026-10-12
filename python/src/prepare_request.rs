//! Shared prologue for [`PreparedAnalysis`](super::PyPreparedAnalysis) prepare routes.
//!
//! Each `prepare_*` method keeps its Python name and signature. The repeated
//! work — parse `options`, load columns or a frame, leave the `GIL`, then build
//! and `prepare` — lives here once.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::{TabularData, TimeSeriesData};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::prepared_options::PrepareOptions;

/// Inputs every prepare route shares after the Python arguments are parsed.
pub(super) struct PrepareRequest {
    /// Column names, in schema order. Retained for the handle.
    pub(super) names: Vec<String>,
    /// Parsed `options` mapping.
    pub(super) opts: PrepareOptions,
    /// Prepare-time seed.
    pub(super) seed: u64,
    /// Prepare-time thread budget; `None` is the resolver's default.
    pub(super) threads: Option<u32>,
    /// Arrow bytes borrowed by [`Self::load_columns_recording`].
    borrowed_bytes: Option<u64>,
}

impl PrepareRequest {
    /// Parse `options` and retain the prepare-time execution controls.
    pub(super) fn parse(
        names: Vec<String>,
        options: Option<&Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            names,
            opts: PrepareOptions::parse(options)?,
            seed,
            threads,
            borrowed_bytes: None,
        })
    }

    /// Refuse a transferred prior on a route that cannot hydrate one.
    pub(super) fn refuse_prior(&self, route: &str) -> PyResult<()> {
        self.opts.refuse_prior_transfer(route)
    }

    /// Refuse a non-default target population on a route that cannot carry one.
    pub(super) fn refuse_population(&self, route: &str) -> PyResult<()> {
        self.opts.refuse_population(route)
    }

    /// Refuse prior transfer and a non-default population, same route wording.
    pub(super) fn refuse_design(&self, route: &str) -> PyResult<()> {
        self.refuse_prior(route)?;
        self.refuse_population(route)
    }

    /// Load tabular columns. Borrowed Arrow bytes are not retained.
    pub(super) fn load_columns(
        &self,
        _py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<TabularData> {
        Ok(crate::tabular_from_py_columns(self.names.clone(), columns)?.0)
    }

    /// Load tabular columns and remember borrowed Arrow bytes.
    ///
    /// Only the static average-effect `prepare` records them; every other route
    /// drops the count, and a later bound estimate clears `bytes_borrowed`.
    pub(super) fn load_columns_recording(
        &mut self,
        _py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<TabularData> {
        let (data, borrowed) = crate::tabular_from_py_columns(self.names.clone(), columns)?;
        self.borrowed_bytes = borrowed;
        Ok(data)
    }

    /// Load series columns, or a panel / multi-environment / event frame.
    pub(super) fn load_frame(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
        frame: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<super::FrameInput> {
        super::FrameInput::ingest(py, &self.names, columns, frame)
    }

    /// Run `build` off the `GIL` and return its value.
    pub(super) fn off_gil<T, F>(self, py: Python<'_>, build: F) -> PyResult<T>
    where
        T: Send,
        F: FnOnce(Self) -> PyResult<T> + Send,
    {
        crate::detach_catch(py, move || build(self))
    }

    /// `apply_inference`, build, prepare, and wrap the handle.
    pub(super) fn finish(
        mut self,
        builder: antecedent::StudyBuilder,
        series: bool,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let builder = self.opts.apply_inference(builder)?;
        self.compile(builder, series)
    }

    /// Write the caller's budget onto `builder`, then [`Self::finish`].
    pub(super) fn finish_applied(
        self,
        builder: antecedent::StudyBuilder,
        series: bool,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let builder = self.opts.apply(builder);
        self.finish(builder, series)
    }

    /// Build and prepare on a fresh context. Inference is already on `builder`.
    pub(super) fn compile(
        self,
        builder: antecedent::StudyBuilder,
        series: bool,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let ctx = self.opts.ctx(self.seed, self.threads);
        self.compile_with(builder, series, &ctx)
    }

    /// Build and prepare on `ctx`, the same context an earlier discovery step used.
    pub(super) fn compile_with(
        self,
        builder: antecedent::StudyBuilder,
        series: bool,
        ctx: &antecedent_core::ExecutionContext,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let analysis = builder.build().map_err(crate::py_err)?;
        let prepared = analysis.prepare(ctx).map_err(crate::py_err)?;
        let mut out = super::finished_prepared(prepared, self.names, series);
        out.borrowed_bytes = self.borrowed_bytes;
        Ok(out)
    }
}

/// One series, or one event stream aligned onto a grid.
///
/// A `DBN` graph-posterior mixture is licensed only on those two inputs.
pub(super) fn one_series(input: super::FrameInput) -> PyResult<TimeSeriesData> {
    match input.materialize()? {
        super::FrameData::Series(series) => Ok(series),
        super::FrameData::Events(events, align) => events
            .align_to_grid(align)
            .map_err(|e| crate::py_msg(format!("event align_to_grid: {e}"))),
        _ => Err(crate::refusal(
            antecedent_core::reason_code!("data_modality_not_licensed"),
            "a DBN graph-posterior mixture is \
             licensed on one series or one event stream",
        )),
    }
}
