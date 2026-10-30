//! Native bridge for fit-free data diagnostics.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::prepared_api::PyPreparedAnalysis;
use antecedent::{DataDiagnosticContext, DataDiagnostics};
use pyo3::prelude::*;

pub(crate) fn policy_setting(text: &str) -> PyResult<Option<DataDiagnostics>> {
    let policy: Option<DataDiagnostics> =
        serde_json::from_str(text).map_err(|e| crate::value_err(e.to_string()))?;
    if let Some(policy) = &policy {
        policy.validate().map_err(|e| crate::value_err(e.to_string()))?;
    }
    Ok(policy)
}

#[pymethods]
impl PyPreparedAnalysis {
    fn data_diagnostics_summary(&self) -> Option<(usize, usize, usize)> {
        self.inner.data_diagnostics().map(summary)
    }
    fn data_diagnostics_json(&self) -> PyResult<Option<String>> {
        self.inner
            .data_diagnostics()
            .map(|r| serde_json::to_string(r).map_err(|e| crate::value_err(e.to_string())))
            .transpose()
    }

    #[pyo3(signature = (policy_json, *, seed=1, threads=None))]
    fn diagnose_data_json(
        &self,
        py: Python<'_>,
        policy_json: String,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let policy = policy_setting(&policy_json)?.unwrap_or_default();
        let inner = std::sync::Arc::clone(&self.inner);
        crate::detach_catch(py, move || {
            let ctx = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
            let report = inner.diagnose_data(&policy, &ctx).map_err(crate::py_err)?;
            serde_json::to_string(&report).map_err(|e| crate::value_err(e.to_string()))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (names, columns, policy_json, context_json, *, seed=1, threads=None, column_types_json=None))]
fn diagnose_data_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    policy_json: String,
    context_json: String,
    seed: u64,
    threads: Option<u32>,
    column_types_json: Option<String>,
) -> PyResult<String> {
    let policy = policy_setting(&policy_json)?.unwrap_or_default();
    let context: DataDiagnosticContext =
        serde_json::from_str(&context_json).map_err(|e| crate::value_err(e.to_string()))?;
    let column_types = column_types_json
        .map(|text| serde_json::from_str(&text).map_err(|e| crate::value_err(e.to_string())))
        .transpose()?
        .unwrap_or_default();
    let data = typed_table(names, columns, column_types)?;
    crate::detach_catch(py, move || {
        let ctx = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
        let report =
            antecedent::diagnose_data(&data, &policy, &context, &ctx).map_err(crate::py_err)?;
        serde_json::to_string(&report).map_err(|e| crate::value_err(e.to_string()))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(data_diagnostic_registry_json, module)?)?;
    module.add_function(wrap_pyfunction!(diagnose_data_json, module)?)
}

#[pymethods]
impl crate::ate_api::PyPreparedBatch {
    fn data_diagnostics_summary(&self) -> Vec<Option<(usize, usize, usize)>> {
        self.inner.plans().iter().map(|p| p.data_diagnostics().map(summary)).collect()
    }
    fn data_diagnostics_json(&self) -> PyResult<Vec<Option<String>>> {
        self.inner
            .plans()
            .iter()
            .map(|p| {
                p.data_diagnostics()
                    .map(|r| serde_json::to_string(r).map_err(|e| crate::value_err(e.to_string())))
                    .transpose()
            })
            .collect()
    }

    #[pyo3(signature = (policy_json, *, seed=1, threads=None))]
    fn diagnose_data_json(
        &self,
        py: Python<'_>,
        policy_json: String,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Vec<String>> {
        let policy = policy_setting(&policy_json)?.unwrap_or_default();
        let inner = std::sync::Arc::clone(&self.inner);
        crate::detach_catch(py, move || {
            let ctx = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
            inner
                .diagnose_data(&policy, &ctx)
                .map_err(crate::py_err)?
                .iter()
                .map(|report| {
                    serde_json::to_string(report).map_err(|e| crate::value_err(e.to_string()))
                })
                .collect()
        })
    }
}

// Standalone diagnostics preserve integer and boolean equality. Analysis ingestion has its
// own deliberate f64 restriction, which must not erase evidence before profiling.
fn typed_table(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    column_types: std::collections::BTreeMap<String, DiagnosticStorage>,
) -> PyResult<antecedent_data::TabularData> {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        BooleanColumn, Float64Column, Int64Column, OwnedColumn, OwnedColumnarStorage, TabularData,
        ValidityBitmap,
    };
    use std::sync::Arc;
    if names.len() != columns.len() || names.is_empty() {
        return Err(crate::value_err("diagnostics require named columns of equal length"));
    }
    let mut schema = CausalSchemaBuilder::new();
    let mut owned = Vec::new();
    for (index, (name, object)) in names.iter().zip(&columns).enumerate() {
        let id = VariableId::from_raw(
            u32::try_from(index).map_err(|_| crate::value_err("too many columns"))?,
        );
        let (column, value_type) = match crate::NumericColumn::from_object(object)? {
            crate::NumericColumn::I64(array) if column_types.contains_key(name) => {
                let storage = &column_types[name];
                let values = array.as_array().iter().copied().collect::<Vec<_>>();
                if storage.kind == "timestamp" {
                    let validity = validity_from(&values, |v| *v != i64::MIN)?;
                    (
                        OwnedColumn::Timestamp(
                            antecedent_data::TimestampColumn::new(
                                id,
                                Arc::<[i64]>::from(values),
                                validity,
                            )
                            .map_err(crate::py_err)?,
                        ),
                        ValueType::Unspecified,
                    )
                } else if storage.kind == "categorical" {
                    use antecedent_data::{
                        CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
                    };
                    let validity = validity_from(&values, |v| *v >= 0)?;
                    let domain_id = antecedent_core::CategoryDomainId::from_raw(id.raw());
                    let domain = Arc::new(CategoryDomain {
                        id: domain_id,
                        levels: storage
                            .levels
                            .iter()
                            .map(|label| CategoryLevel { label: Arc::from(label.as_str()) })
                            .collect::<Vec<_>>()
                            .into(),
                        ordered: false,
                        reference: None,
                        unknown_policy: UnknownCategoryPolicy::Fail,
                    });
                    let codes = values
                        .iter()
                        .map(|v| {
                            u32::try_from((*v).max(0))
                                .map(CategoryCode::from_raw)
                                .map_err(|_| crate::value_err("category code exceeds u32"))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    (
                        OwnedColumn::Categorical(
                            antecedent_data::CategoricalColumn::try_new(
                                id, codes, validity, domain,
                            )
                            .map_err(crate::py_err)?,
                        ),
                        ValueType::Categorical,
                    )
                } else {
                    return Err(crate::value_err("unknown diagnostic column storage"));
                }
            }
            crate::NumericColumn::I64(array) => {
                let values: Arc<[i64]> =
                    array.as_array().iter().copied().collect::<Vec<_>>().into();
                let validity = ValidityBitmap::all_valid(values.len());
                (
                    OwnedColumn::Int64(
                        Int64Column::new(id, values, validity).map_err(crate::py_err)?,
                    ),
                    ValueType::Unspecified,
                )
            }
            crate::NumericColumn::Bool(array) => {
                let values: Arc<[u8]> =
                    array.as_array().iter().map(|v| u8::from(*v)).collect::<Vec<_>>().into();
                let validity = ValidityBitmap::all_valid(values.len());
                (
                    OwnedColumn::Boolean(
                        BooleanColumn::new(id, values, validity).map_err(crate::py_err)?,
                    ),
                    ValueType::Binary,
                )
            }
            numeric => {
                let (values, _) = numeric.owned(name)?;
                let validity = ValidityBitmap::all_valid(values.len());
                (
                    OwnedColumn::Float64(
                        Float64Column::new(id, values, validity).map_err(crate::py_err)?,
                    ),
                    ValueType::Continuous,
                )
            }
        };
        schema
            .add_variable(
                Arc::<str>::from(name.as_str()),
                value_type,
                SmallRoleSet::empty(),
                None,
                column_types
                    .get(name)
                    .filter(|kind| kind.kind == "categorical")
                    .map(|_| antecedent_core::CategoryDomainId::from_raw(id.raw())),
                MeasurementSpec::default(),
            )
            .map_err(crate::py_err)?;
        owned.push(column);
    }
    let storage =
        OwnedColumnarStorage::try_new(schema.build().map_err(crate::py_err)?, owned, None, None)
            .map_err(crate::py_err)?;
    Ok(TabularData::new(storage))
}

fn summary(report: &antecedent::DataDiagnosticsReport) -> (usize, usize, usize) {
    (
        report.findings.iter().filter(|f| f.severity == "review").count(),
        report.findings.iter().filter(|f| f.severity == "blocking").count(),
        report.checks.iter().filter(|c| c.status == "not_evaluated").count(),
    )
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticStorage {
    kind: String,
    levels: Vec<String>,
}

fn validity_from(
    values: &[i64],
    observed: impl Fn(&i64) -> bool,
) -> PyResult<antecedent_data::ValidityBitmap> {
    let mut bytes = vec![0_u8; values.len().div_ceil(8)];
    for (index, value) in values.iter().enumerate() {
        if observed(value) {
            bytes[index / 8] |= 1 << (index % 8);
        }
    }
    antecedent_data::ValidityBitmap::from_bytes(bytes, values.len()).map_err(crate::py_err)
}

#[pyfunction]
fn data_diagnostic_registry_json() -> PyResult<String> {
    serde_json::to_string(antecedent_validate::data_diagnostics::DATA_DIAGNOSTIC_CHECKS)
        .map_err(|e| crate::value_err(e.to_string()))
}
