//! Bulk numeric transport inputs bypass JSON; Rust owns each detached snapshot.
use crate::ingest::snapshot_array;
use antecedent::estimate::{SmoothedDoseInput, TrialAipwInput};
use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyDict};

fn required<'py>(data: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(name)?.ok_or_else(|| crate::value_err(format!("missing transport column {name}")))
}
fn floats(data: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<f64>> {
    let item = required(data, name)?;
    let array: PyReadonlyArray1<'_, f64> = item.extract()?;
    Ok(snapshot_array(&array))
}
/// A float column that may be absent or `None`.
fn optional_floats(data: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<Vec<f64>>> {
    match data.get_item(name)? {
        Some(item) if !item.is_none() => {
            let array: PyReadonlyArray1<'_, f64> = item.extract()?;
            Ok(Some(snapshot_array(&array)))
        }
        _ => Ok(None),
    }
}
/// A cluster-id column that may be absent or `None`.
fn optional_ids(data: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<Vec<u32>>> {
    match data.get_item(name)? {
        Some(item) if !item.is_none() => {
            let array: PyReadonlyArray1<'_, u32> = item.extract()?;
            Ok(Some(snapshot_array(&array)))
        }
        _ => Ok(None),
    }
}
/// A lag that may be absent or `None`.
fn optional_lag(data: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<u32>> {
    match data.get_item(name)? {
        Some(item) if !item.is_none() => Ok(Some(item.extract()?)),
        _ => Ok(None),
    }
}
fn booleans(data: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<bool>> {
    let item = required(data, name)?;
    let array: PyReadonlyArray1<'_, bool> = item.extract()?;
    Ok(snapshot_array(&array))
}
fn covariates(data: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<f64>>> {
    let item = required(data, "covariates")?;
    let arrays: Vec<PyReadonlyArray1<'_, f64>> = item.extract()?;
    Ok(arrays.iter().map(snapshot_array).collect())
}

pub(crate) fn trial(
    data: &Bound<'_, PyAny>,
    names: &[String],
    continuous: bool,
) -> PyResult<TrialAipwInput> {
    let payload = data.call_method1("_buffers", (names.to_vec(),))?;
    let data = payload.cast::<PyDict>()?;
    let sampling: String = required(data, "sampling")?.extract()?;
    if continuous {
        antecedent::declared::IidSampling::parse_learned_continuous(&sampling)
            .map_err(crate::transport_common::error)?;
    }
    Ok(TrialAipwInput {
        features: required(data, "features")?.extract()?,
        covariates: covariates(data)?,
        outcome: floats(data, "outcome")?,
        treatment: booleans(data, "treatment")?,
        source: booleans(data, "source")?,
        randomization: floats(data, "randomization")?,
        sampling: serde_json::from_value(serde_json::Value::String(sampling))
            .map_err(|e| crate::value_err(e.to_string()))?,
    })
}
pub(crate) fn smoothed(data: &Bound<'_, PyAny>, names: &[String]) -> PyResult<SmoothedDoseInput> {
    let payload = data.call_method1("_buffers", (names.to_vec(),))?;
    let data = payload.cast::<PyDict>()?;
    let sampling: String = required(data, "sampling")?.extract()?;
    antecedent::declared::IidSampling::parse_smoothed_dose(&sampling)
        .map_err(crate::transport_common::error)?;
    Ok(SmoothedDoseInput {
        features: required(data, "features")?.extract()?,
        covariates: covariates(data)?,
        outcome: floats(data, "outcome")?,
        dose: floats(data, "dose")?,
        source: booleans(data, "source")?,
        dose_density: floats(data, "dose_density")?,
        sampling: serde_json::from_value(serde_json::Value::String(sampling))
            .map_err(|e| crate::value_err(e.to_string()))?,
        sampling_weights: optional_floats(data, "sampling_weights")?,
        cluster: optional_ids(data, "cluster")?,
        serial_lag: optional_lag(data, "serial_lag")?,
    })
}
