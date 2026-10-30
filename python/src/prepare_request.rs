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

    /// Refuse declared sampling weights on a route that reads none of them.
    fn refuse_sampling_weights(&self) -> PyResult<()> {
        if self.opts.sampling_weights.is_none() {
            return Ok(());
        }
        Err(crate::estimator_config::coded_refusal(
            antecedent_estimate::sampling_weights::unsupported_route(
                "this route reads no sampling weights; declare them on an estimator \
                 (weights=) or on a randomized trial or balanced two-period panel DiD",
            ),
        ))
    }

    /// Load tabular columns. Borrowed Arrow bytes are not retained.
    pub(super) fn load_columns(
        &self,
        _py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<TabularData> {
        self.refuse_sampling_weights()?;
        Ok(crate::tabular_from_py_columns(self.names.clone(), columns)?.0)
    }

    /// Attach the declared sampling weights to `data` as its observation weights.
    ///
    /// The weights are validated first, so an invalid one is refused naming its row.
    fn attach_sampling_weights(&self, data: TabularData) -> PyResult<TabularData> {
        let Some(weights) = &self.opts.sampling_weights else {
            return Ok(data);
        };
        antecedent_estimate::SamplingWeights::new(weights.to_vec())
            .map_err(crate::estimator_config::coded_refusal)?;
        data.with_weights(std::sync::Arc::clone(weights)).map_err(|error| {
            crate::estimator_config::coded_refusal(antecedent_estimate::sampling_weights::invalid(
                &error.to_string(),
            ))
        })
    }

    /// Load tabular columns and attach the declared sampling weights as the table's
    /// observation weights, for a route that estimates in the weighted population.
    pub(super) fn load_weighted_columns(
        &self,
        _py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<TabularData> {
        self.attach_sampling_weights(crate::tabular_from_py_columns(self.names.clone(), columns)?.0)
    }

    /// Load a single series and attach the declared sampling weights to it; a panel,
    /// multi-environment or event frame has no weighted construction.
    pub(super) fn load_weighted_series(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
        frame: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<super::FrameInput> {
        let input = super::FrameInput::ingest(py, &self.names, columns, frame)?;
        if self.opts.sampling_weights.is_none() {
            return Ok(input);
        }
        match input {
            super::FrameInput::Series(data) => {
                Ok(super::FrameInput::Series(self.attach_sampling_weights(data)?))
            }
            _ => Err(crate::estimator_config::coded_refusal(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "sampling weights are read by a single series; panel, multi-environment and \
                     event frames have no sampling-weighted construction",
                ),
            )),
        }
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
        self.refuse_sampling_weights()?;
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
        self.refuse_sampling_weights()?;
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
        self,
        builder: antecedent::StudyBuilder,
        series: bool,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let ctx = self.opts.ctx(self.seed, self.threads);
        self.finish_with(builder, series, &ctx)
    }

    /// Apply scientific declarations and compile on the caller's existing context.
    pub(super) fn finish_with(
        mut self,
        builder: antecedent::StudyBuilder,
        series: bool,
        ctx: &antecedent_core::ExecutionContext,
    ) -> PyResult<super::PyPreparedAnalysis> {
        let builder = self.opts.apply_inference(builder)?;
        let builder = self.opts.apply_selection(builder);
        self.compile_with(builder, series, ctx)
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
        let builder = match &self.opts.data_diagnostics {
            Some(policy) => builder.data_diagnostics(policy.clone()),
            None => builder,
        };
        let analysis = builder.build().map_err(crate::py_err)?;
        let prepared = match analysis.prepare_deferred(ctx) {
            Ok(prepared) => prepared,
            // A refused study the caller asked to have described keeps its refusal and, beside
            // it, Rust's descriptive fallback of that same study.
            Err(refusal) => {
                return Err(match &self.opts.fallback {
                    Some(declaration) => crate::fallback_stage_api::describe_refusal(
                        &analysis,
                        declaration,
                        &self.names,
                        self.seed,
                        crate::resolve_user_threads(self.threads),
                        ctx,
                        refusal,
                    ),
                    None => crate::py_err(refusal),
                });
            }
        };
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
        super::FrameData::Events(events, align, _retention) => events
            .align_to_grid(align)
            .map_err(|e| crate::py_msg(format!("event align_to_grid: {e}"))),
        _ => Err(crate::refusal(
            antecedent_core::reason_code!("data_modality_not_licensed"),
            "a DBN graph-posterior mixture is \
             licensed on one series or one event stream",
        )),
    }
}
