//! One actual held native response and two executing original attested mean callbacks.
use crate::recalc_api::{context, invalid, plan_refusal_json, refusal_json};
use crate::recalc_bounds::{sequence_item, sequence_len};
use antecedent::analysis::recalc_composite::{CompositeError, CompositeRequest, CompositeSession};
use antecedent::analysis::recalc_external::ExternalMeanProvider;
use antecedent_core::recalc::Branch;
use antecedent_design::composition_boundary::{SupportPolicy, UnsupportedActionPolicy};
use antecedent_design::decision_artifact::contract_from_json;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::BTreeMap;
fn field<'py>(dict: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(name)?
        .ok_or_else(|| invalid("composite_recalc.invalid_request", format!("missing{name}")))
}
fn bounded_text(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<String> {
    let value = field(dict, name)?;
    let value = value.cast::<PyString>()?;
    if value.len()? > 1024 * 1024 || value.to_str()?.len() > 1024 * 1024 {
        return Err(invalid("composite_recalc.invalid_request", "text exceeds byte bound"));
    }
    Ok(value.to_str()?.to_owned())
}
fn bounded_names(value: &Bound<'_, PyAny>, limit: usize) -> PyResult<Vec<String>> {
    let len = sequence_len(value, "composite_recalc.invalid_request")?;
    if len > limit {
        return Err(invalid("composite_recalc.invalid_request", "too many columns"));
    }
    let mut names = Vec::with_capacity(len);
    for index in 0..len {
        let name = sequence_item(value, index)?;
        let name = name.cast::<PyString>()?;
        if name.len()? > 256 || name.to_str()?.len() > 256 {
            return Err(invalid("composite_recalc.invalid_request", "column name exceeds bound"));
        }
        names.push(name.to_str()?.to_owned());
    }
    Ok(names)
}
fn native_request(
    payload: &Bound<'_, PyDict>,
) -> PyResult<antecedent::analysis::recalc_static::StaticResponseRequest> {
    let names = field(payload, "names")?;
    if sequence_len(&names, "composite_recalc.invalid_request")? > 12 {
        return Err(invalid("composite_recalc.invalid_request", "too many native columns"));
    }
    let columns = field(payload, "columns")?;
    if sequence_len(&columns, "composite_recalc.invalid_request")? > 12 {
        return Err(invalid("composite_recalc.invalid_request", "too many native columns"));
    }
    let names = bounded_names(&names, 12)?;
    let columns: Vec<PyReadonlyArray1<'_, f64>> = columns.extract()?;
    let graph = field(payload, "graph")?.extract::<PyRef<'_, crate::graphs::Admg>>()?;
    crate::recalc_static_api::response_request(
        names,
        &columns,
        &graph,
        &bounded_text(payload, "specification")?,
    )
}
fn request(payload: &Bound<'_, PyDict>) -> PyResult<CompositeRequest> {
    let callbacks = field(payload, "externals")?;
    if sequence_len(&callbacks, "composite_recalc.invalid_request")? != 2 {
        return Err(invalid("composite_recalc.invalid_request", "exactly two callbacks required"));
    }
    let mut externals = Vec::new();
    for i in 0..2 {
        let item = sequence_item(&callbacks, i)?;
        let item = item.cast::<PyDict>()?;
        let names = field(item, "names")?;
        let columns = field(item, "columns")?;
        if sequence_len(&names, "composite_recalc.invalid_request")? > 64
            || sequence_len(&columns, "composite_recalc.invalid_request")? > 64
        {
            return Err(invalid("composite_recalc.invalid_request", "too many callback columns"));
        }
        let names = bounded_names(&names, 64)?;
        let columns: Vec<PyReadonlyArray1<'_, f64>> = columns.extract()?;
        externals.push(crate::recalc_external_api::request(
            names,
            &columns,
            &bounded_text(item, "specification")?,
        )?);
    }
    let binding = bounded_text(payload, "program")?;
    let contract = bounded_text(payload, "contract")?;
    if contract.len() > 1024 * 1024 {
        return Err(invalid("composite_recalc.invalid_request", "contract exceeds byte bound"));
    }
    let decision_contract = contract_from_json(&contract).map_err(crate::py_msg)?;
    let policy = match bounded_text(payload, "policy")?.as_str() {
        "compare_supported" => UnsupportedActionPolicy::CompareSupported,
        "require_all" => UnsupportedActionPolicy::RequireAllActions,
        _ => return Err(invalid("composite_recalc.invalid_request", "unknown support policy")),
    };
    Ok(CompositeRequest {
        native: native_request(payload)?,
        native_program: crate::program_claims_api::parse_binding(&binding)?,
        externals,
        decision_contract,
        input_order: bounded_names(&field(payload, "input_order")?, 3)?,
        support_policy: SupportPolicy {
            unsupported: policy,
            weakest_support: antecedent_core::SupportStatus::Supported,
        },
    })
}
fn error(error: &CompositeError) -> String {
    match error {
        CompositeError::Native(cause) => crate::recalc_static_api::static_error(cause),
        CompositeError::External { branch, cause } => {
            crate::recalc_external_api::error(cause, *branch)
        }
        CompositeError::Binding(cause) => serde_json::to_string(
            &antecedent_io::external_binding_wire::RefusalWire::from((**cause).clone()),
        )
        .expect("refusal serialization"),
        CompositeError::Refused(plan) => plan_refusal_json(plan),
        _ => {
            let detail = match error {
                CompositeError::Request(detail) => *detail,
                _ => "composite_recalc.receipt_invalid",
            };
            refusal_json(
                error
                    .request_reason_code()
                    .unwrap_or(antecedent_core::reason_code!("invalid_argument")),
                "decision",
                detail,
                None,
                None,
                &format!("{error:?}"),
                None,
            )
        }
    }
}
fn conditional_payload(
    result: Result<
        antecedent::analysis::conditional_study_ranking::ConditionalStudyRanking,
        antecedent::analysis::conditional_study_ranking::ConditionalRankingError,
    >,
) -> PyResult<crate::recalc_api::RunPayload> {
    match result {
        Ok(result) => {
            let bytes = result.to_bytes().map_err(|e| crate::py_msg(format!("{e:?}")))?;
            let report = String::from_utf8(bytes.clone()).map_err(crate::py_msg)?;
            Ok((Some(report), Some(bytes), None))
        }
        Err(cause) => Ok((None, None, Some(serde_json::json!({"code":cause.code,"detail":cause.detail,"stage":"conditional_study_ranking","message":format!("{cause:?}")}).to_string()))),
    }
}
#[pyclass(name = "CompositeSessionHandle", skip_from_py_object)]
struct PyCompositeSession {
    inner: CompositeSession,
}
#[pymethods]
impl PyCompositeSession {
    #[new]
    fn new(
        native: PyRef<'_, crate::recalc_static_api::PyStaticSession>,
        payload: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let native_request = native_request(payload)?;
        CompositeSession::from_native(native.clone_native(), &native_request, &context(1, None))
            .map(|inner| Self { inner })
            .map_err(|cause| {
                invalid("composite_recalc.native_state_mismatch", format!("{cause:?}"))
            })
    }
    fn plan(&self, payload: &Bound<'_, PyDict>) -> PyResult<(Option<String>, Option<String>)> {
        let request = request(payload)?;
        match self.inner.plan(&request) {
            Ok(plan) => Ok((
                Some(
                    serde_json::to_string(&antecedent_io::recalc_receipt_artifact::plan_to_wire(
                        &plan,
                    ))
                    .map_err(crate::py_msg)?,
                ),
                None,
            )),
            Err(cause) => Ok((None, Some(error(&cause)))),
        }
    }
    #[pyo3(signature=(payload,providers,*,seed=1,threads=None,cancel=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        payload: &Bound<'_, PyDict>,
        providers: &Bound<'_, PyDict>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyRef<'_, crate::PyCancellationToken>>,
    ) -> PyResult<crate::recalc_api::RunPayload> {
        let request = request(payload)?;
        if providers.len() > 2 {
            return Err(invalid("composite_recalc.invalid_request", "too many providers"));
        }
        let mut actual = Vec::new();
        for (branch, provider) in providers.iter() {
            let branch: u8 = branch.extract()?;
            let branch = Branch::new(branch).ok_or_else(|| {
                invalid("composite_recalc.invalid_request", "branch exceeds bound")
            })?;
            let provider =
                provider.extract::<PyRef<'_, crate::recalc_external_api::PyCallbackProvider>>()?;
            actual.push((branch, provider.clone_actual_provider(py)?));
        }
        let mut ctx = context(seed, threads);
        if let Some(cancel) = cancel {
            ctx.cancellation = cancel.inner.clone();
        }
        let outcome = py.detach(|| {
            let mut providers = actual
                .iter_mut()
                .map(|(branch, provider)| (*branch, provider as &mut dyn ExternalMeanProvider))
                .collect::<BTreeMap<_, _>>();
            self.inner.execute(&request, &mut providers, &ctx)
        });
        match outcome {
            Err(cause) => Ok((None, None, Some(error(&cause)))),
            Ok(out) => crate::recalc_family_api::payload(
                &out.previous,
                &out.requested,
                &out.capabilities,
                &out.plan,
                &out.receipt,
                serde_json::json!({"decision":crate::composition_api::supported_json(&out.decision)}),
            ),
        }
    }
    fn rank_conditional_studies(
        &self,
        policy_json: &str,
    ) -> PyResult<crate::recalc_api::RunPayload> {
        use antecedent::analysis::conditional_study_ranking::{
            ConditionalStudyPolicy, ConditionalStudyRanking,
        };
        if policy_json.len() > 256 * 1024 {
            return Err(invalid(
                "conditional_study_ranking.invalid_policy",
                "policy exceeds byte bound",
            ));
        }
        let policy: ConditionalStudyPolicy = serde_json::from_str(policy_json).map_err(|_| {
            invalid("conditional_study_ranking.invalid_policy", "invalid policy schema")
        })?;
        conditional_payload(ConditionalStudyRanking::execute(&self.inner, policy))
    }
    #[pyo3(signature=(artifact,expected_identity=None))]
    fn consume_conditional_studies(
        &self,
        artifact: &Bound<'_, pyo3::types::PyBytes>,
        expected_identity: Option<&str>,
    ) -> PyResult<crate::recalc_api::RunPayload> {
        use antecedent::analysis::conditional_study_ranking::ConditionalStudyRanking;
        conditional_payload(ConditionalStudyRanking::consume(
            artifact.as_bytes(),
            &self.inner,
            expected_identity,
        ))
    }
    fn native_response(&self) -> PyResult<Option<crate::response_api::ResponseAnalysisResult>> {
        let native = self.inner.native();
        let (Some(result), Some(prepared), Some(ctx)) =
            (native.result(), native.prepared(), native.producing_context())
        else {
            return Ok(None);
        };
        let names =
            prepared.schema().variables().iter().map(|v| v.name.to_string()).collect::<Vec<_>>();
        crate::prepared_api::response_from_retained(&names, result, prepared, ctx).map(Some)
    }
    fn export_native(&self) -> PyResult<Vec<u8>> {
        self.inner.native().export_result(&context(1, None)).map_err(|cause| {
            invalid("composite_recalc.native_state_mismatch", format!("{cause:?}"))
        })
    }
    fn export_callback(&self, branch: u8) -> PyResult<Vec<u8>> {
        let branch = Branch::new(branch)
            .ok_or_else(|| invalid("composite_recalc.invalid_request", "branch exceeds bound"))?;
        self.inner
            .callback(branch)
            .ok_or_else(|| invalid("external_recalc.output_unavailable", "no issued callback"))?
            .export_output()
            .map_err(|cause| invalid("external_recalc.artifact_invalid", cause.to_string()))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompositeSession>()
}
