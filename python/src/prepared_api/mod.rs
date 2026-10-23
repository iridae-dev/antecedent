//! Compile-once / re-estimate-many [`PreparedStudy`] Python OO surface.
//!
//! The Python class is [`PyPreparedAnalysis`] (`PreparedAnalysis`). Method names
//! registered with PyO3 are the Rust function names in the sibling modules.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::PreparedStudy;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_data::TabularData;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{
    AteAnalysisResult, ate_result_from_analysis, detach_catch, py_err, py_execution_context_ext,
    suite_from_refute,
};

mod contract;
mod controls;
mod execute;
mod frame;
mod inspect;
mod posterior;
mod prepare_design;
mod prepare_effect;
mod prepare_posterior;
mod prepare_query;
mod prepare_response;
mod prepare_temporal_design;
mod prepare_unlicensed_design;
mod routing;
mod treatment_score;

#[path = "../prepare_request.rs"]
mod prepare_request;

pub(crate) use contract::{
    parse_transform_intent, response_from_retained, response_from_study, transform_report_map,
};
use controls::{ClickControls, map_ate, unstaged};
use frame::{FrameData, FrameInput};
pub(crate) use posterior::contract_to_map;
use posterior::{
    SeriesGraphQuery, dbn_posterior, exact_dag_posterior, finish_series_graph_posterior,
    finish_static_graph_posterior, insert_reuse_identities, require_prepared_names,
    take_supplied_posterior,
};
use prepare_request::{PrepareRequest, one_series};
use routing::{
    StaticClassGraph, apply_estimator, bind_temporal_class, conditional_query, configured_spec,
    extract_temporal_class, intervention_response_query, parse_estimator, parse_identifier,
    parse_response_options, response_query, static_ate_query, static_kind_query,
    temporal_effect_query, temporal_mediation_query, with_graph, with_response_options,
};

/// Durable prepare-once / estimate-many handle for static ATE on a supplied DAG.
#[pyclass(name = "PreparedAnalysis")]
pub struct PyPreparedAnalysis {
    /// Arc so per-click estimate/refute detach with a refcount bump, not a
    /// deep `PreparedStudy` clone; `refresh` clones-on-write to swap data.
    pub(crate) inner: Arc<PreparedStudy>,
    pub(in crate::prepared_api) names: Vec<String>,
    /// Last estimate result retained for second-click refute.
    pub(in crate::prepared_api) last: Option<Arc<antecedent::StudyResult>>,
    pub(in crate::prepared_api) last_study: Option<Arc<PreparedStudy>>,
    pub(in crate::prepared_api) last_seed: u64,
    pub(in crate::prepared_api) last_threads: u32,
    pub(in crate::prepared_api) borrowed_bytes: Option<u64>,
    /// When true, estimate/refresh clicks use series data (`estimate_series`).
    pub(in crate::prepared_api) series: bool,
}

impl PyPreparedAnalysis {
    pub(crate) fn from_study(prepared: PreparedStudy, names: Vec<String>) -> Self {
        Self {
            inner: Arc::new(prepared),
            names,
            last: None,
            last_study: None,
            last_seed: 1,
            last_threads: 1,
            borrowed_bytes: None,
            series: false,
        }
    }

    /// Execute one click on new data, optionally replacing the retained data.
    ///
    /// The executed study (with the click's data bound) becomes the snapshot
    /// source for exports and second-click refutes; a refresh also replaces the
    /// handle's retained study. A click-only stage sink is stripped first.
    pub(in crate::prepared_api) fn click<T: Send>(
        &mut self,
        py: Python<'_>,
        data: FrameInput,
        refresh: bool,
        seed: u64,
        threads: Option<u32>,
        controls: ClickControls,
        map: fn(
            &[String],
            &antecedent::StudyResult,
            &PreparedStudy,
            &ExecutionContext,
        ) -> PyResult<T>,
    ) -> PyResult<T> {
        let mut study = controls.owned_study(&self.inner)?;
        let out_names = self.names.clone();
        let series = self.series;
        let (mapped, result, bound) = detach_catch(py, move || {
            let ctx = controls.ctx(seed, threads);
            let result = match data {
                // Both estimate and refresh bind the click's data to the executed
                // copy, so its snapshot re-executes on exactly that data.
                FrameInput::Series(data) if !series => study.refresh(data, &ctx).map_err(py_err)?,
                frame => frame.materialize()?.bind_and_run(&mut study, &ctx)?,
            };
            let mapped = map(&out_names, &result, &study, &ctx)?;
            Ok((mapped, result, Arc::new(unstaged(study))))
        })?;
        if refresh {
            self.inner = Arc::clone(&bound);
            self.borrowed_bytes = None;
        }
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Execute the retained data.
    pub(in crate::prepared_api) fn click_bound<T: Send>(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        controls: ClickControls,
        map: fn(
            &[String],
            &antecedent::StudyResult,
            &PreparedStudy,
            &ExecutionContext,
        ) -> PyResult<T>,
    ) -> PyResult<T> {
        let borrowed_bytes = self.borrowed_bytes;
        let study = controls.study(&self.inner)?;
        let names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = controls.ctx(seed, threads);
            let mut result = study.estimate_retained(&ctx).map_err(py_err)?;
            result.performance.bytes_borrowed = borrowed_bytes;
            let mapped = map(&names, &result, &study, &ctx)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(Arc::clone(&self.inner));
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// A design-defined population cannot be reweighted.
    ///
    /// A transport study's target population is the selection diagram's
    /// target rows under known selection probabilities, and an interference
    /// contrast is a finite-population design estimand; neither has a score
    /// table whose rows a weight vector could redeclare.
    pub(in crate::prepared_api) fn refuse_design_retarget(&self) -> PyResult<()> {
        let route = match self.inner.query() {
            CausalQuery::Transport(_) => "a TransportQuery study",
            CausalQuery::Interference(_) => "an InterferenceQuery study",
            CausalQuery::AnomalyAttribution(_) => "an AnomalyAttribution study",
            CausalQuery::ChangeAttribution(_) => "a ChangeAttribution study",
            _ => return Ok(()),
        };
        Err(crate::refusal(
            antecedent_core::reason_code!("population_not_estimable"),
            format!(
                "{route} estimates the population its design defines; a row-weight retarget has \
                 no score table to reweight and would redeclare the design's own target"
            ),
        ))
    }

    pub(in crate::prepared_api) fn finish_ate_refute(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        let prior = self
            .last
            .clone()
            .ok_or_else(|| crate::value_err("call estimate/refresh before refute"))?;
        let refute_suite = suite_from_refute(Some(&suite))?;
        let cancel_token = cancel.map(|c| c.inner);
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                cancel_token,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refute(&prior, &data, refute_suite, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, &result, false)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }
}

fn finished_prepared(
    prepared: PreparedStudy,
    names: Vec<String>,
    series: bool,
) -> PyPreparedAnalysis {
    PyPreparedAnalysis {
        inner: Arc::new(prepared),
        names,
        last: None,
        last_study: None,
        last_seed: 1,
        last_threads: 1,
        borrowed_bytes: None,
        series,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedAnalysis>()?;
    Ok(())
}
