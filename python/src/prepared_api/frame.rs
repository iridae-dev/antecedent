//! Panel, multi-environment, and event frames ingested under the GIL.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{PreparedStudy, Study};
use antecedent_core::CausalSchema;
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::{py_err, py_msg, series_from_tabular, tabular_from_py_columns};

/// Data partitions of a panel, multi-environment, or event frame, ingested under the GIL.
pub(super) enum FrameInput {
    Series(TabularData),
    Panel { units: Vec<TabularData>, unit_ids: Vec<u32> },
    MultiEnv(Vec<TabularData>),
    Events { data: TabularData, event_times_ns: Vec<i64>, align_interval_ns: u64 },
}

/// Materialized frame data, built without the GIL.
pub(super) enum FrameData {
    Series(TimeSeriesData),
    Panel(antecedent_data::PanelData),
    MultiEnv(antecedent_data::MultiEnvironmentData),
    Events(antecedent_data::EventData, u64),
}

impl FrameInput {
    pub(super) fn ingest(
        _py: Python<'_>,
        names: &[String],
        columns: Vec<Bound<'_, PyAny>>,
        frame: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let Some(frame) = frame else {
            return Ok(Self::Series(tabular_from_py_columns(names.to_vec(), columns)?.0));
        };
        let kind: String = frame
            .get_item("kind")?
            .ok_or_else(|| crate::value_err("frame requires kind"))?
            .extract()?;
        let partitions = |key: &str| -> PyResult<Vec<TabularData>> {
            let parts: Vec<Vec<Bound<'_, PyAny>>> = frame
                .get_item(key)?
                .ok_or_else(|| crate::value_err(format!("{kind} frame requires {key}")))?
                .extract()?;
            parts
                .into_iter()
                .map(|cols| Ok(tabular_from_py_columns(names.to_vec(), cols)?.0))
                .collect()
        };
        match kind.as_str() {
            "panel" => {
                let units = partitions("unit_columns")?;
                let unit_ids: Vec<u32> = frame
                    .get_item("unit_ids")?
                    .ok_or_else(|| crate::value_err("panel frame requires unit_ids"))?
                    .extract()?;
                if units.is_empty() || units.len() != unit_ids.len() {
                    return Err(crate::value_err(
                        "panel frame needs one unit id per non-empty unit",
                    ));
                }
                Ok(Self::Panel { units, unit_ids })
            }
            "multi_env" => {
                let envs = partitions("env_columns")?;
                if envs.is_empty() {
                    return Err(crate::value_err("multi_env frame needs ≥1 environment"));
                }
                Ok(Self::MultiEnv(envs))
            }
            "events" => {
                let event_times_ns: Vec<i64> = frame
                    .get_item("event_times_ns")?
                    .ok_or_else(|| crate::value_err("events frame requires event_times_ns"))?
                    .extract()?;
                let align_interval_ns: u64 = frame
                    .get_item("align_interval_ns")?
                    .ok_or_else(|| crate::value_err("events frame requires align_interval_ns"))?
                    .extract()?;
                let (data, _) = tabular_from_py_columns(names.to_vec(), columns)?;
                Ok(Self::Events { data, event_times_ns, align_interval_ns })
            }
            other => Err(crate::value_err(format!(
                "unknown frame kind {other:?}; use panel|multi_env|events"
            ))),
        }
    }

    pub(super) fn materialize(self) -> PyResult<FrameData> {
        match self {
            Self::Series(data) => Ok(FrameData::Series(series_from_tabular(data)?)),
            Self::Panel { units, unit_ids } => {
                let units = units
                    .into_iter()
                    .zip(unit_ids)
                    .map(|(data, unit_id)| {
                        Ok(antecedent_data::PanelUnit {
                            unit_id,
                            series: series_from_tabular(data)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(FrameData::Panel(
                    antecedent_data::PanelData::try_new(Arc::from(units)).map_err(py_err)?,
                ))
            }
            Self::MultiEnv(envs) => {
                let series =
                    envs.into_iter().map(series_from_tabular).collect::<PyResult<Vec<_>>>()?;
                Ok(FrameData::MultiEnv(
                    antecedent_data::MultiEnvironmentData::try_new(Arc::from(series))
                        .map_err(py_err)?,
                ))
            }
            Self::Events { data, event_times_ns, align_interval_ns } => {
                let events = antecedent_data::EventData::try_new(
                    data.storage().clone(),
                    Arc::from(event_times_ns),
                )
                .map_err(py_err)?;
                Ok(FrameData::Events(events, align_interval_ns))
            }
        }
    }
}

impl FrameData {
    pub(super) fn schema(&self) -> &CausalSchema {
        match self {
            Self::Series(data) => data.schema(),
            Self::Panel(data) => data.schema(),
            Self::MultiEnv(data) => data.schema(),
            Self::Events(data, _) => data.schema(),
        }
    }

    pub(super) fn builder(self) -> PyResult<antecedent::StudyBuilder> {
        Ok(match self {
            Self::Series(data) => Study::series(data),
            Self::Panel(data) => Study::panel(data),
            Self::MultiEnv(data) => Study::series_multi(data),
            Self::Events(data, align) => Study::events(&data, align).map_err(py_err)?,
        })
    }

    /// Bind this frame to `study` (same modality, schema and regularity) and execute.
    pub(super) fn bind_and_run(
        self,
        study: &mut PreparedStudy,
        ctx: &antecedent_core::ExecutionContext,
    ) -> PyResult<antecedent::StudyResult> {
        let result = match self {
            Self::Series(data) => study.refresh_series(data, ctx),
            Self::Panel(data) => study.refresh_panel(data, ctx),
            Self::MultiEnv(data) => study.refresh_multi_env(data, ctx),
            Self::Events(data, align) => {
                let aligned = data
                    .align_to_grid(align)
                    .map_err(|e| py_msg(format!("event align_to_grid: {e}")))?;
                study.refresh_series(aligned, ctx)
            }
        };
        result.map_err(py_err)
    }
}
