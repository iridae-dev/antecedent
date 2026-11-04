//! Native-issued ordinary Gaussian Bayesian selective adapter.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::NumericColumn;
use crate::recalc_api::{RunPayload, bounded_context, invalid, parse_json};
use crate::recalc_bounds::check_columns;
use crate::recalc_family_api::{caps, payload, resume, run_error};
use antecedent::BayesianConfig;
use antecedent::analysis::recalc_bayesian::{
    BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
    execute_bayesian_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent::io::recalc_receipt_artifact::{CapabilitiesWire, identities_to_wire, plan_to_wire};
use antecedent::prelude::{ExecutionContext, VariableId};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    model: String,
    lower_probability: f64,
    upper_probability: f64,
    threshold: f64,
    benefit_per_unit: f64,
    cost: f64,
}
fn inference(dict: &Bound<'_, PyDict>, features: usize) -> PyResult<BayesianConfig> {
    for (key, _) in dict.iter() {
        let key = key.extract::<String>()?;
        if ![
            "mode",
            "n_draws",
            "prior_scale",
            "prior_artifact",
            "prior_mapping",
            "composed_prior",
            "likelihood",
        ]
        .contains(&key.as_str())
        {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_INFERENCE,
                "unknown inference field",
            ));
        }
    }
    let get = |key| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
    };
    if let Some(bytes) = get("prior_artifact")? {
        if bytes.cast::<PyBytes>()?.as_bytes().len() > crate::bridge::MAX_DECLARED_ARTIFACT_BYTES {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::LIMITS_EXCEEDED,
                "prior artifact exceeds byte bound",
            ));
        }
    }
    let n = get("n_draws")?.map(|value| value.extract::<usize>()).transpose()?.unwrap_or(256);
    if n == 0 || n > 100_000 || n.checked_mul(features).is_none_or(|cells| cells > 1_000_000) {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::LIMITS_EXCEEDED,
            "posterior rows/features exceed bounded draw matrix",
        ));
    }
    let mode = get("mode")?
        .ok_or_else(|| {
            invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_INFERENCE,
                "missing Bayesian mode",
            )
        })?
        .extract::<String>()?;
    let scale =
        get("prior_scale")?.map(|value| value.extract::<f64>()).transpose()?.unwrap_or(10.0);
    let config = match mode.as_str() {
        "bayesian.conjugate" | "conjugate" => BayesianConfig::conjugate(),
        "bayesian.laplace" | "bayesian" | "laplace" => BayesianConfig::laplace(),
        "bayesian.hmc" | "hmc" => BayesianConfig::hmc(),
        _ => {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_INFERENCE,
                "unknown Bayesian backend",
            ));
        }
    }
    .n_draws(n)
    .prior_scale(scale);
    let spec = crate::prepared_options::InferenceSpec::parse(dict)?;
    let config = config.likelihood(
        spec.likelihood.unwrap_or(antecedent::inference::BayesLikelihood::GaussianIdentity),
    );
    if let Some(composed) = spec.composed_prior {
        crate::prior_bank::apply_owned_composed_prior(config, composed)
    } else if let Some(bytes) = spec.prior_artifact {
        Ok(config.prior_from_artifact(bytes, spec.prior_mapping))
    } else {
        Ok(config)
    }
}
fn request(
    names: Vec<String>,
    columns: &[NumericColumn<'_>],
    specification: &str,
    config: &Bound<'_, PyDict>,
    ctx: &ExecutionContext,
) -> PyResult<BayesianRequest> {
    check_columns(columns, ctx)?;
    if names.len() != columns.len() {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_DATA,
            "column names/data dimensions differ",
        ));
    }
    if names.len() > 64 {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::LIMITS_EXCEEDED,
            "Bayesian feature count exceeds64",
        ));
    }
    for (i, name) in names.iter().enumerate() {
        if name.is_empty() || names[..i].contains(name) {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_DATA,
                "column names must be non-empty and unique",
            ));
        }
    }
    let wire: RequestWire = parse_json(specification, "Bayesian request")?;
    let locate = |name: &str| -> PyResult<VariableId> {
        let Some(i) = names.iter().position(|n| n == name) else {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_QUERY,
                "unknown role variable",
            ));
        };
        Ok(VariableId::from_raw(crate::fit_u32(i)?))
    };
    let treatment = locate(&wire.treatment)?;
    let outcome = locate(&wire.outcome)?;
    let edges = wire
        .edges
        .iter()
        .map(|(a, b)| Ok((locate(a)?.raw(), locate(b)?.raw())))
        .collect::<PyResult<Vec<_>>>()?;
    let model = match wire.model.as_str() {
        "gaussian" => BayesianModel::Gaussian,
        "quadratic_basis" => BayesianModel::QuadraticBasis,
        _ => {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::BAYESIAN_INVALID_QUERY,
                "unknown checked Bayesian model",
            ));
        }
    };
    let inference = inference(config, names.len() + 1)?;
    let columns = crate::named_f64_columns(names, columns)?;
    Ok(BayesianRequest {
        columns: columns.into(),
        edges,
        treatment,
        outcome,
        model,
        inference,
        summary: PosteriorSummarySpec {
            lower_probability: wire.lower_probability,
            upper_probability: wire.upper_probability,
            threshold: wire.threshold,
        },
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}
#[pyclass(name = "BayesianSessionHandle")]
struct PyBayesianSession {
    inner: BayesianSession,
}
#[pymethods]
impl PyBayesianSession {
    #[new]
    fn new() -> Self {
        Self { inner: BayesianSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let (ids, ctx) = resume(previous_json, resume_json)?;
        Ok(Self { inner: BayesianSession::resume(ids, ctx) })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(crate::py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&caps(self.inner.boundary())))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(names,columns,specification,inference,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        specification: &str,
        inference: Bound<'_, PyDict>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = request(names, &columns, specification, &inference, &ctx)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &ctx)))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(names,columns,specification,inference,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        specification: &str,
        inference: Bound<'_, PyDict>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<RunPayload> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = request(names, &columns, specification, &inference, &ctx)?;
        drop(columns);
        drop(inference);
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let session = &mut self.inner;
        let ran = crate::detach_catch(py, move || {
            let ran = execute_bayesian_with_receipt(session, &request, &ctx);
            Ok(ran)
        })?;
        match ran {
            Ok(out) => payload(
                &previous,
                self.inner.identities(),
                &capabilities,
                &out.plan,
                &out.receipt,
                serde_json::json!({
            "law":{"mean":out.law.mean,"standard_deviation":out.law.standard_deviation,"lower_quantile":out.law.lower_quantile,"upper_quantile":out.law.upper_quantile,"probability_below":out.law.probability_below,"draws":out.law.draws},
            "decision":{"net_benefit":out.decision.net_benefit,"treat":out.decision.treat}}),
            ),
            Err(error) => Ok((
                None,
                None,
                Some(run_error(
                    &error,
                    antecedent::reason::reason_details::recalc::BAYESIAN_FIT_REFUSED,
                )),
            )),
        }
    }
    fn export_prior_source(&self) -> (Option<Vec<u8>>, Option<String>) {
        match self.inner.export_prior_source() {
            Ok(bytes) => (Some(bytes), None),
            Err(error) => (
                None,
                Some(run_error(
                    &error,
                    antecedent::reason::reason_details::recalc::BAYESIAN_FIT_REFUSED,
                )),
            ),
        }
    }
    fn effect_draws(&self) -> Option<Vec<f64>> {
        let posterior = self.inner.posterior()?;
        posterior.draws.column(posterior.effect_column()?).ok().map(<[f64]>::to_vec)
    }
    #[pyo3(signature=(*,seed=1,threads=None,cancel=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = crate::recalc_api::bounded_context(seed, threads, None, cancel);
        crate::detach_catch(py, || match self.inner.export_result(&ctx) {
            Ok(bytes) => Ok((Some(bytes), None)),
            Err(error) => Ok((
                None,
                Some(run_error(
                    &error,
                    antecedent::reason::reason_details::recalc::BAYESIAN_FIT_REFUSED,
                )),
            )),
        })
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBayesianSession>()?;
    crate::recalc_joint_bayesian_api::register(module)?;
    Ok(())
}
