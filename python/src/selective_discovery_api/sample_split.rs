//! Discovery and effect analysis over a declared row split.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Discovery followed by analysis on a row split.
// ---------------------------------------------------------------------------

fn rule_of(text: &str) -> PyResult<SplitRule> {
    let parsed = parse_json(text)?;
    let map = parsed.as_object().ok_or_else(|| input_err("a split rule is a JSON object"))?;
    match get_text(map, "kind")?.as_str() {
        "explicit" => {
            let rows = field(map, "discovery_rows")
                .and_then(Json::as_array)
                .ok_or_else(|| input_err("field `discovery_rows` must be a list"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| input_err("discovery rows are row indices"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(SplitRule::Explicit { discovery_rows: rows })
        }
        "seeded_permutation" => Ok(SplitRule::SeededPermutation {
            seed: get_u64(map, "seed")?,
            discovery_fraction: field(map, "discovery_fraction")
                .and_then(Json::as_f64)
                .ok_or_else(|| input_err("field `discovery_fraction` must be a number"))?,
        }),
        other => Err(input_err(format!("unknown split rule {other:?}"))),
    }
}

fn rule_value(rule: &SplitRule) -> Json {
    match rule {
        SplitRule::Explicit { discovery_rows } => {
            let mut rows = discovery_rows.clone();
            rows.sort_unstable();
            json!({ "kind": "explicit", "discovery_rows": rows })
        }
        SplitRule::SeededPermutation { seed, discovery_fraction } => json!({
            "kind": "seeded_permutation",
            "seed": seed,
            "discovery_fraction": discovery_fraction,
        }),
    }
}

fn binding_json(binding: &AnalysisBinding) -> String {
    json!({
        "receipt_identity": hex(&binding.receipt_identity),
        "structure_identity": hex(&binding.structure_identity),
        "result_identity": hex(&binding.result_identity),
        "analysis_data_identity": hex(&binding.analysis_data_identity),
        "identity": hex(&binding.identity),
        "conditional_on_selected_graph": binding.conditional_on_selected_graph(),
        "whole_method_calibration": binding.whole_method_calibration(),
    })
    .to_string()
}

fn columns_of(py: Python<'_>, data: &TabularData) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    data.schema()
        .variables()
        .iter()
        .map(|spec| {
            let values = data.float64_values(spec.id).map_err(|e| input_err(e.to_string()))?;
            Ok(PyArray1::from_vec(py, values).unbind())
        })
        .collect()
}

/// Rebuild `data` with the observation weights and the analysis mask the caller supplied
/// (`mask[i]` true keeps row `i`). Either may be absent; a length that differs from the row
/// count, or weights that are not a finite non-negative measure, are input errors.
fn table_with_rows(
    data: TabularData,
    weights: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
) -> PyResult<TabularData> {
    if weights.is_none() && mask.is_none() {
        return Ok(data);
    }
    let n = data.row_count();
    let storage = data.storage();
    let mask = mask
        .map(|keep| {
            if keep.len() != n {
                return Err(input_err(format!("mask has {} entries for {n} rows", keep.len())));
            }
            let mut bytes = vec![0u8; n.div_ceil(8)];
            for (i, kept) in keep.iter().enumerate() {
                if *kept {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            ValidityBitmap::from_bytes(bytes, n).map_err(|e| input_err(e.to_string()))
        })
        .transpose()?;
    let weights = weights
        .map(|w| {
            if w.len() == n {
                Ok(std::sync::Arc::<[f64]>::from(w))
            } else {
                Err(input_err(format!("weights has {} entries for {n} rows", w.len())))
            }
        })
        .transpose()?;
    OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        storage.columns().to_vec(),
        mask.or_else(|| storage.analysis_mask().cloned()),
        weights.or_else(|| storage.weights().map(std::sync::Arc::from)),
    )
    .map(TabularData::new)
    .map_err(|e| input_err(e.to_string()))
}

/// A row split, the discovery run on its discovery part and the receipt tying them together.
#[pyclass(name = "SampleSplitDiscoveryHandle", skip_from_py_object)]
pub(crate) struct SampleSplitDiscoveryHandle {
    inner: SplitDiscovery,
    names: Vec<String>,
}

#[pymethods]
impl SampleSplitDiscoveryHandle {
    /// The split receipt as JSON.
    fn receipt_json(&self) -> String {
        let receipt = self.inner.receipt();
        json!({
            "rule": rule_value(&receipt.rule),
            "n_rows": receipt.n_rows,
            "masked_out_rows": receipt.masked_out_rows,
            "incomplete_rows": receipt.incomplete_rows,
            "n_discovery": receipt.n_discovery,
            "n_analysis": receipt.n_analysis,
            "discovery_rows_identity": hex(&receipt.discovery_rows_identity),
            "analysis_rows_identity": hex(&receipt.analysis_rows_identity),
            "mask_identity": receipt.mask_identity.as_ref().map(hex),
            "weights": receipt.weights.as_ref().map(|w| json!({
                "source": hex(&w.source),
                "discovery": hex(&w.discovery),
                "analysis": hex(&w.analysis),
                "discovery_effective_sample_size": w.discovery_effective_sample_size,
                "analysis_effective_sample_size": w.analysis_effective_sample_size,
            })),
            "source_data_identity": hex(&receipt.source_data.digest),
            "discovery_data_identity": hex(&receipt.discovery_data.digest),
            "analysis_data_identity": hex(&receipt.analysis_data.digest),
            "identity": hex(&receipt.identity),
            "conditional_on_selected_graph": true,
            "whole_method_calibration": WHOLE_METHOD_CALIBRATION,
        })
        .to_string()
    }

    /// The original row index of each discovery row, in the part's row order.
    fn discovery_row_ids(&self) -> Vec<usize> {
        self.inner.discovery_row_ids().to_vec()
    }

    /// The original row index of each analysis row, in the part's row order.
    fn analysis_row_ids(&self) -> Vec<usize> {
        self.inner.analysis_row_ids().to_vec()
    }

    /// The analysis rows' weights, unnormalized; `None` for an unweighted source.
    fn analysis_weights(&self) -> Option<Vec<f64>> {
        self.inner.analysis_weights().map(<[f64]>::to_vec)
    }

    /// The discovery outcome as JSON.
    fn outcome_json(&self) -> String {
        outcome_json(self.inner.outcome())
    }

    /// The discovery part's columns, in `names` order.
    fn discovery_columns(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        columns_of(py, self.inner.discovery_rows())
    }

    /// The analysis part's columns, in `names` order.
    fn analysis_columns(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        columns_of(py, self.inner.analysis_rows())
    }

    /// The two parts as evidence-ledger source declarations `(discovery, analysis)`, keyed
    /// by `snapshot_id` and row index; or a refusal when the receipt does not verify.
    fn evidence_sources(&self, snapshot_id: &[u8]) -> Reply {
        match split_evidence_sources(self.inner.receipt(), snapshot_id) {
            Ok((discovery, analysis)) => Ok((
                Some(
                    json!({
                        "discovery": declaration_json(&discovery),
                        "analysis": declaration_json(&analysis),
                    })
                    .to_string(),
                ),
                None,
            )),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok((None, Some(refusal))),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Re-derive the partition from the rule; a refusal when the receipt does not hold.
    fn verify(&self) -> Option<String> {
        self.inner.verify().err().and_then(|e| error_json(&e))
    }

    /// Replay the split on the supplied source table (columns, weights, mask) and require it to
    /// be the table the receipt describes; a refusal when it is not.
    fn verify_source(
        &self,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        weights: Option<Vec<f64>>,
        mask: Option<Vec<bool>>,
    ) -> PyResult<Option<String>> {
        if names != self.names {
            return Err(input_err(
                "the source table must carry the discovery table's columns in the same order",
            ));
        }
        let batch = crate::numeric_columns_to_batch(&names, &columns)?;
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        let data = table_with_rows(data, weights, mask)?;
        match self.inner.verify_against_source(&data) {
            Ok(()) => Ok(None),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok(Some(refusal)),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Check that the supplied columns (and weights) are exactly the analysis part; returns the
    /// binding JSON or a refusal.
    fn check_analysis_table(
        &self,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        if names != self.names {
            return Err(input_err(
                "the analysis table must carry the discovery table's columns in the same order",
            ));
        }
        let batch = crate::numeric_columns_to_batch(&names, &columns)?;
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        // A split of a masked source gives each part the all-included mask of rows that passed
        // it, so the table to check carries that mask too.
        let mask = self.inner.receipt().mask_identity.map(|_| vec![true; data.row_count()]);
        let data = table_with_rows(data, weights, mask)?;
        match self.inner.check_analysis_table(&data) {
            Ok(binding) => Ok((Some(binding_json(&binding)), None)),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok((None, Some(refusal))),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Refusal of an interval for an analysis that reused the rows discovery saw.
    fn request_unsplit_interval(&self) -> String {
        match self.inner.request_unsplit_interval() {
            Err(error) => error_json(&error).unwrap_or_default(),
            Ok(never) => match never {},
        }
    }
}

type Split = (Option<SampleSplitDiscoveryHandle>, Option<String>, Option<String>);

/// Split the rows by the declared rule and discover on the discovery part.
///
/// Returns `(handle, discovery_report_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (contract_json, names, columns, rule_json, *, weights=None, mask=None, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: contract, data, weights, mask, rule")]
fn sample_split_discover(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    rule_json: &str,
    weights: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Split> {
    let parsed = parse_json(contract_json)?;
    let map = contract_map(&parsed, "static")?;
    let seed = get_u64(map, "seed")?;
    let contract = static_contract(map, names.len())?;
    let rule = rule_of(rule_json)?;
    let batch = crate::numeric_columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result: Result<SplitDiscovery, SelectiveDiscoveryError> =
        crate::detach_catch(py, move || {
            let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
            let data = table_with_rows(data, weights, mask)?;
            Ok(discover_on_split(&contract, &data, rule, &ctx))
        })?;
    match result {
        Ok(split) => {
            let report = report_json(split.report());
            Ok((Some(SampleSplitDiscoveryHandle { inner: split, names }), Some(report), None))
        }
        Err(error) => match error_json(&error) {
            Some(refusal) => Ok((None, None, Some(refusal))),
            None => Err(discovery_failure(&error)),
        },
    }
}

/// The refusal an io-level split-effect failure carries, as the structured JSON the Python
/// layer raises; any other io failure is an error, not a refusal.
fn effect_refusal(error: antecedent::io::IoError) -> PyResult<String> {
    match error {
        antecedent::io::IoError::Refused { code, message } => {
            let detail = message.split_once(": ").map_or("refused", |(d, _)| d);
            Ok(refusal_json(code, STAGE, detail, &message, &message))
        }
        other => Err(crate::transport_common::error(other)),
    }
}

fn basis_json(basis: &antecedent::schema::CalibrationBasis) -> Json {
    json!({
        "query": &*basis.query,
        "graph_class": &*basis.graph_class,
        "structure": &*basis.structure,
        "modality": &*basis.modality,
        "inference": &*basis.inference,
        "estimator": &*basis.estimator,
        "interval_method": &*basis.interval_method,
        "se_kind": &*basis.se_kind,
        "dependence": &*basis.dependence,
        "functional": &*basis.functional,
        "identification": &*basis.identification,
        "row_count": basis.row_count,
    })
}

fn effect_json(estimate: &SplitEffect) -> String {
    let (lower, upper) = estimate.interval();
    json!({
        "effect": estimate.effect(),
        "lower": lower,
        "upper": upper,
        "level": estimate.level(),
        "standard_error": estimate.standard_error(),
        "analysis_rows": estimate.analysis_rows(),
        "discovery_rows": estimate.discovery_rows(),
        "adjustment_set": estimate.adjustment_set(),
        "receipt_identity": hex(estimate.receipt_identity()),
        "structure_identity": hex(estimate.structure_identity()),
        "basis": basis_json(&estimate.keyed_basis()),
    })
    .to_string()
}

fn parent_set_effect_json(estimate: &ParentSetEffect) -> String {
    let reported = estimate.reported();
    let unresolved: Vec<Json> = estimate
        .unresolved_edges()
        .iter()
        .map(|edge| json!({ "a": edge.a, "b": edge.b, "reason": edge.reason }))
        .collect();
    json!({
        "scalar": reported.name,
        "effect": reported.point,
        "lower": reported.lower,
        "upper": reported.upper,
        "level": estimate.level(),
        "standard_error": reported.standard_error,
        "coefficient": estimate.coefficient(),
        "transformation": estimate.transformation().name(),
        "analysis_rows": estimate.analysis_rows(),
        "discovery_rows": estimate.discovery_rows(),
        "treatment_parents": estimate.treatment_parents(),
        "treatment_children": estimate.treatment_children(),
        "unresolved_edges": unresolved,
        "edge_count": estimate.edge_count(),
        "acceptance": estimate.acceptance(),
        "receipt_identity": hex(estimate.receipt_identity()),
        "structure_identity": hex(estimate.structure_identity()),
        "basis": basis_json(&estimate.keyed_basis()),
    })
    .to_string()
}

type Effect = (Option<String>, Option<Vec<u8>>, Option<String>);

/// The linear effect of a treatment on an outcome after PC discovery on one half of a seeded
/// row split and parent adjustment on the other half.
///
/// Returns `(estimate_json, source_bytes, refusal)`.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, split_seed, seed, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: data, roles and seeds")]
fn sample_split_effect(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    treatment: u32,
    outcome: u32,
    split_seed: u64,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Effect> {
    let batch = crate::numeric_columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || {
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        Ok(estimate_effect_on_split(
            &data,
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
            split_seed,
            &ctx,
        ))
    })?;
    match result {
        Ok(estimate) => {
            Ok((Some(effect_json(&estimate)), Some(estimate.source_artifact().to_vec()), None))
        }
        Err(error) => Ok((None, None, Some(effect_refusal(error)?))),
    }
}

type Measured = (Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>);

/// Replay a split-effect source and attest its interval from the coverage records.
///
/// Returns `(measured, refusal)`; a refusal is `not_supported` when no current attesting
/// record covers the construction at this analysis row count.
#[pyfunction]
#[pyo3(signature = (source, seed, *, threads=None, cancel=None))]
fn sample_split_effect_measure(
    py: Python<'_>,
    source: &[u8],
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Measured> {
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || Ok(measure_source(source, &ctx)))?;
    match result {
        Ok(measured) => Ok((
            Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
            None,
        )),
        Err(error) => Ok((None, Some(effect_refusal(error)?))),
    }
}

/// The effect of a treatment on an outcome after PC discovery on one half of a seeded row split,
/// accepted on the treatment's parent set, for a declared linear or log-linear study.
///
/// The transformation is parsed before any data are read. Returns
/// `(estimate_json, source_bytes, refusal)`.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, split_seed, seed, transformation, max_degree, log_shift=None, knowledge=None, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: data, roles, declarations and seeds")]
fn sample_split_parent_set_effect(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    treatment: u32,
    outcome: u32,
    split_seed: u64,
    seed: u64,
    transformation: &str,
    max_degree: u32,
    log_shift: Option<f64>,
    knowledge: Option<&str>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Effect> {
    crate::bridge::check_count(
        "max_degree",
        u64::from(max_degree),
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge: Option<Knowledge> = knowledge
        .map(|text| {
            serde_json::from_str(text)
                .map_err(|e| input_err(format!("knowledge is not a valid declaration: {e}")))
        })
        .transpose()?;
    let transformation = match Transformation::parse(transformation) {
        Ok(declared) => declared,
        Err(error) => return Ok((None, None, Some(effect_refusal(error)?))),
    };
    let options = ParentSetOptions { transformation, max_degree, log_shift };
    let batch = crate::numeric_columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || {
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        let (treatment, outcome) = (VariableId::from_raw(treatment), VariableId::from_raw(outcome));
        Ok(match knowledge {
            None => {
                estimate_parent_set_effect(&data, treatment, outcome, options, split_seed, &ctx)
            }
            declared => estimate_parent_set_effect_with_knowledge(
                &data, treatment, outcome, options, declared, split_seed, &ctx,
            ),
        })
    })?;
    match result {
        Ok(estimate) => Ok((
            Some(parent_set_effect_json(&estimate)),
            Some(estimate.source_artifact().to_vec()),
            None,
        )),
        Err(error) => Ok((None, None, Some(effect_refusal(error)?))),
    }
}

/// Replay a parent-set split-effect source and attest its interval from the coverage records.
///
/// Returns `(measured, refusal)`; a refusal is `not_supported` when no current attesting
/// record covers the construction at this variable count, degree and analysis row count.
#[pyfunction]
#[pyo3(signature = (source, seed, *, threads=None, cancel=None))]
fn sample_split_parent_set_effect_measure(
    py: Python<'_>,
    source: &[u8],
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Measured> {
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || Ok(measure_parent_set_source(source, &ctx)))?;
    match result {
        Ok(measured) => Ok((
            Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
            None,
        )),
        Err(error) => Ok((None, Some(effect_refusal(error)?))),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SampleSplitDiscoveryHandle>()?;
    module.add_function(wrap_pyfunction!(sample_split_discover, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_effect, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_effect_measure, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_parent_set_effect, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_parent_set_effect_measure, module)?)?;
    Ok(())
}
