//! Shared graph / dummy-CI construction for PyO3 bindings (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::discovery::{SpaceDummyCiMode, TimeDummyCiMode};
use antecedent_core::{CausalSchema, Lag, VariableId};
use antecedent_data::{
    SamplingRegularity, TableView, TabularData, TimeDummyEncoding, TimeIndex, TimeSeriesData,
};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalDataError, py_err};

/// Build a static [`Dag`] from named directed edges against a schema.
///
/// Unknown names map to [`CausalDataError`] (schema lookup at the Python boundary).
pub(crate) fn dag_from_named_edges(
    schema: &CausalSchema,
    edges: &[(String, String)],
) -> PyResult<Dag> {
    let n_vars =
        u32::try_from(schema.len()).map_err(|_| PyValueError::new_err("too many variables"))?;
    let mut dag = Dag::with_variables(n_vars);
    for (from, to) in edges {
        let from_id = schema_var_id(schema, from)?;
        let to_id = schema_var_id(schema, to)?;
        dag.insert_directed(
            DenseNodeId::from_raw(from_id.raw()),
            DenseNodeId::from_raw(to_id.raw()),
        )
        .map_err(py_err)?;
    }
    Ok(dag)
}

/// Resolve a variable name via schema, with a typed data-error message.
pub(crate) fn schema_var_id(schema: &CausalSchema, name: &str) -> PyResult<VariableId> {
    schema
        .id_of(name)
        .map_err(|e| CausalDataError::new_err(format!("unknown variable {name}: {e}")))
}

/// Build a [`TemporalDag`] from lagged named edges using a name→id resolver.
pub(crate) fn temporal_dag_from_lagged_edges<F>(
    mut resolve: F,
    edges: &[(String, u32, String, u32)],
) -> PyResult<TemporalDag>
where
    F: FnMut(&str) -> PyResult<VariableId>,
{
    let mut g = TemporalDag::empty();
    for (src, slag, tgt, tlag) in edges {
        let s = ensure_lagged(&mut g, resolve(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
        let t = ensure_lagged(&mut g, resolve(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
        g.insert_directed(s, t).map_err(py_err)?;
    }
    Ok(g)
}

/// Build a [`TemporalDag`] from lagged edges resolved against a schema.
pub(crate) fn temporal_dag_from_schema_edges(
    schema: &CausalSchema,
    edges: &[(String, u32, String, u32)],
) -> PyResult<TemporalDag> {
    temporal_dag_from_lagged_edges(|nm| schema_var_id(schema, nm), edges)
}

/// Regular unit-interval series view over tabular storage (shared discover/analyze path).
pub(crate) fn series_from_tabular(tabular: TabularData) -> PyResult<TimeSeriesData> {
    let n = tabular.row_count();
    TimeSeriesData::try_new(
        tabular.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .map_err(py_err)
}

pub(crate) fn parse_space_dummy_ci(s: &str) -> PyResult<SpaceDummyCiMode> {
    match s {
        "scalar" | "scalar_one_hot" | "one_hot" => Ok(SpaceDummyCiMode::ScalarOneHot),
        "multivariate" | "multivariate_block" | "block" => Ok(SpaceDummyCiMode::MultivariateBlock),
        other => Err(PyValueError::new_err(format!(
            "space_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
        ))),
    }
}

pub(crate) fn parse_time_dummy_ci(s: &str) -> PyResult<TimeDummyCiMode> {
    match s {
        "scalar" | "scalar_one_hot" | "one_hot" => Ok(TimeDummyCiMode::ScalarOneHot),
        "multivariate" | "multivariate_block" | "block" => Ok(TimeDummyCiMode::MultivariateBlock),
        other => Err(PyValueError::new_err(format!(
            "time_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
        ))),
    }
}

pub(crate) fn parse_time_dummy_encoding(s: &str) -> PyResult<TimeDummyEncoding> {
    match s {
        "integer" | "integer_index" | "index" => Ok(TimeDummyEncoding::IntegerIndex),
        "one_hot" | "onehot" | "oh" => Ok(TimeDummyEncoding::OneHot),
        other => Err(PyValueError::new_err(format!(
            "time_dummy_encoding must be 'integer' or 'one_hot', got '{other}'"
        ))),
    }
}

/// Parse space / time-encoding / time CI mode strings used by multi-env discovery.
pub(crate) fn parse_dummy_ci_modes(
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
) -> PyResult<(SpaceDummyCiMode, TimeDummyEncoding, TimeDummyCiMode)> {
    Ok((
        parse_space_dummy_ci(space_dummy_ci)?,
        parse_time_dummy_encoding(time_dummy_encoding)?,
        parse_time_dummy_ci(time_dummy_ci)?,
    ))
}

pub(crate) fn space_dummy_ci_from_bool(multivariate: bool) -> SpaceDummyCiMode {
    if multivariate { SpaceDummyCiMode::MultivariateBlock } else { SpaceDummyCiMode::ScalarOneHot }
}

pub(crate) fn time_dummy_ci_from_bool(multivariate: bool) -> TimeDummyCiMode {
    if multivariate { TimeDummyCiMode::MultivariateBlock } else { TimeDummyCiMode::ScalarOneHot }
}
