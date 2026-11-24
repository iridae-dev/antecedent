//! Python bridge for the whole-chain bootstrap union of an assumption envelope.
//!
//! Python declares the original chain as the propagation-chain record wire and the resampling
//! design: raw sampling units under one scheme (independent rows, clusters, stratified
//! clusters, moving blocks, stratified moving blocks or panel blocks), or reruns declared by
//! the caller. The compiler program, every replicate schedule and recompiled chain, the
//! endpoint union, the portable artifact and every refusal are Rust's. The union is a
//! bootstrap candidate: its coverage is `not_measured` and nothing here earns it.

use antecedent::analysis::propagation_chain::ChainRefusal;
use antecedent::analysis::sampling_assumption_union::{
    ClusterSamplingAssumptionUnionRequest, CompiledClusterSamplingAssumptionUnion,
    CompiledIidSamplingAssumptionUnion, CompiledMovingBlockSamplingAssumptionUnion,
    CompiledPanelBlockSamplingAssumptionUnion, CompiledSamplingAssumptionUnion,
    CompiledStratifiedClusterSamplingAssumptionUnion,
    CompiledStratifiedMovingBlockSamplingAssumptionUnion, IidSamplingAssumptionUnionRequest,
    MovingBlockSamplingAssumptionUnionRequest, PanelBlockSamplingAssumptionUnionRequest,
    SamplingAssumptionUnionError, SamplingAssumptionUnionRequest, SamplingAssumptionUnionResult,
    StratifiedClusterSamplingAssumptionUnionRequest,
    StratifiedMovingBlockSamplingAssumptionUnionRequest, compile_cluster_sampling_assumption_union,
    compile_iid_sampling_assumption_union, compile_moving_block_sampling_assumption_union,
    compile_panel_block_sampling_assumption_union, compile_sampling_assumption_union,
    compile_stratified_cluster_sampling_assumption_union,
    compile_stratified_moving_block_sampling_assumption_union,
    consume_cluster_sampling_assumption_union, consume_iid_sampling_assumption_union,
    consume_moving_block_sampling_assumption_union, consume_panel_block_sampling_assumption_union,
    consume_sampling_assumption_union, consume_stratified_cluster_sampling_assumption_union,
    consume_stratified_moving_block_sampling_assumption_union,
};
use antecedent::planner::lowered::LoweredProgram;
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::PyCancellationToken;
use crate::bridge::{MAX_DATA_DECLARATION_BYTES, check_declared_size};
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

const STAGE: &str = "sampling_assumption_union";

/// The resampling scheme of one union, with the artifact feature its compiler seals.
#[derive(Clone, Copy)]
enum Scheme {
    Declared,
    Iid,
    Cluster,
    StratifiedCluster,
    MovingBlock,
    StratifiedMovingBlock,
    PanelBlock,
}

impl Scheme {
    const ALL: [Self; 7] = [
        Self::Declared,
        Self::Iid,
        Self::Cluster,
        Self::StratifiedCluster,
        Self::MovingBlock,
        Self::StratifiedMovingBlock,
        Self::PanelBlock,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Iid => "iid",
            Self::Cluster => "cluster",
            Self::StratifiedCluster => "stratified_cluster",
            Self::MovingBlock => "moving_block",
            Self::StratifiedMovingBlock => "stratified_moving_block",
            Self::PanelBlock => "panel_block",
        }
    }

    const fn feature(self) -> &'static str {
        match self {
            Self::Declared => "sampling_assumption_union_v1",
            Self::Iid => "iid_sampling_assumption_union_v1",
            Self::Cluster => "cluster_sampling_assumption_union_v1",
            Self::StratifiedCluster => "stratified_cluster_sampling_assumption_union_v1",
            Self::MovingBlock => "moving_block_sampling_assumption_union_v1",
            Self::StratifiedMovingBlock => "stratified_moving_block_sampling_assumption_union_v1",
            Self::PanelBlock => "panel_block_sampling_assumption_union_v1",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scheme| scheme.name() == name)
    }

    fn from_feature(feature: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scheme| scheme.feature() == feature)
    }
}

/// One compiled union, whichever scheme produced it.
enum Compiled {
    Declared(Box<CompiledSamplingAssumptionUnion>),
    Iid(Box<CompiledIidSamplingAssumptionUnion>),
    Cluster(Box<CompiledClusterSamplingAssumptionUnion>),
    StratifiedCluster(Box<CompiledStratifiedClusterSamplingAssumptionUnion>),
    MovingBlock(Box<CompiledMovingBlockSamplingAssumptionUnion>),
    StratifiedMovingBlock(Box<CompiledStratifiedMovingBlockSamplingAssumptionUnion>),
    PanelBlock(Box<CompiledPanelBlockSamplingAssumptionUnion>),
}

/// Apply one expression to the compiled program, whichever scheme it is.
macro_rules! dispatch {
    ($compiled:expr, $plan:ident => $body:expr) => {
        match $compiled {
            Compiled::Declared($plan) => $body,
            Compiled::Iid($plan) => $body,
            Compiled::Cluster($plan) => $body,
            Compiled::StratifiedCluster($plan) => $body,
            Compiled::MovingBlock($plan) => $body,
            Compiled::StratifiedMovingBlock($plan) => $body,
            Compiled::PanelBlock($plan) => $body,
        }
    };
}

impl Compiled {
    fn lowered(&self) -> &LoweredProgram {
        dispatch!(self, plan => plan.lowered())
    }

    fn plan_view(&self) -> antecedent::planner::PlanView {
        dispatch!(self, plan => plan.plan_view())
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        dispatch!(self, plan => plan.execute(ctx))
    }

    fn export(
        &self,
        result: &SamplingAssumptionUnionResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        dispatch!(self, plan => plan.export(result, ctx))
    }
}

fn request<T: DeserializeOwned>(text: &str) -> PyResult<T> {
    check_declared_size(text, MAX_DATA_DECLARATION_BYTES, "sampling assumption union")?;
    serde_json::from_str(text).map_err(|error| {
        crate::value_err(format!("invalid sampling assumption union declaration: {error}"))
    })
}

fn compile(
    scheme: Scheme,
    text: &str,
    ctx: &ExecutionContext,
) -> PyResult<Result<Compiled, SamplingAssumptionUnionError>> {
    Ok(match scheme {
        Scheme::Declared => {
            let request: SamplingAssumptionUnionRequest = request(text)?;
            compile_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::Declared(Box::new(plan)))
        }
        Scheme::Iid => {
            let request: IidSamplingAssumptionUnionRequest = request(text)?;
            compile_iid_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::Iid(Box::new(plan)))
        }
        Scheme::Cluster => {
            let request: ClusterSamplingAssumptionUnionRequest = request(text)?;
            compile_cluster_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::Cluster(Box::new(plan)))
        }
        Scheme::StratifiedCluster => {
            let request: StratifiedClusterSamplingAssumptionUnionRequest = request(text)?;
            compile_stratified_cluster_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::StratifiedCluster(Box::new(plan)))
        }
        Scheme::MovingBlock => {
            let request: MovingBlockSamplingAssumptionUnionRequest = request(text)?;
            compile_moving_block_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::MovingBlock(Box::new(plan)))
        }
        Scheme::StratifiedMovingBlock => {
            let request: StratifiedMovingBlockSamplingAssumptionUnionRequest = request(text)?;
            compile_stratified_moving_block_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::StratifiedMovingBlock(Box::new(plan)))
        }
        Scheme::PanelBlock => {
            let request: PanelBlockSamplingAssumptionUnionRequest = request(text)?;
            compile_panel_block_sampling_assumption_union(request, ctx)
                .map(|plan| Compiled::PanelBlock(Box::new(plan)))
        }
    })
}

fn consume(
    scheme: Scheme,
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    match scheme {
        Scheme::Declared => consume_sampling_assumption_union(bytes, expected, ctx),
        Scheme::Iid => consume_iid_sampling_assumption_union(bytes, expected, ctx),
        Scheme::Cluster => consume_cluster_sampling_assumption_union(bytes, expected, ctx),
        Scheme::StratifiedCluster => {
            consume_stratified_cluster_sampling_assumption_union(bytes, expected, ctx)
        }
        Scheme::MovingBlock => consume_moving_block_sampling_assumption_union(bytes, expected, ctx),
        Scheme::StratifiedMovingBlock => {
            consume_stratified_moving_block_sampling_assumption_union(bytes, expected, ctx)
        }
        Scheme::PanelBlock => consume_panel_block_sampling_assumption_union(bytes, expected, ctx),
    }
}

/// A refused chain, with its registered code, namespaced detail and next attempt.
pub(crate) fn chain_refusal_json(refusal: &ChainRefusal, stage: &str) -> String {
    json!({
        "code": refusal.code(),
        "stage": stage,
        "detail": refusal.detail(),
        "message": refusal.to_string(),
        "offending": refusal.stage(),
        "next_attempt": refusal.next_attempt().name(),
        "remedy": refusal.next_attempt().sentence(),
    })
    .to_string()
}

/// A refused declaration, or on replay a refused binding of the bytes to the retained
/// identity, seed or result.
fn invalid_json(message: &str, replay: bool) -> String {
    let (code, detail) = if replay {
        (
            antecedent::reason::reason_code!("external_binding_mismatch"),
            antecedent::reason::reason_details::sampling_assumption_union::REPLAY_MISMATCH,
        )
    } else {
        (
            antecedent::reason::reason_code!("invalid_argument"),
            antecedent::reason::reason_details::sampling_assumption_union::INVALID_DECLARATION,
        )
    };
    json!({
        "code": code,
        "stage": STAGE,
        "detail": detail,
        "message": message,
        "offending": null,
        "next_attempt": null,
        "remedy": null,
    })
    .to_string()
}

/// The refusal JSON of a union error; an unreadable artifact raises its serialization error and
/// a cancelled run raises the cancellation, which is no refusal of the declaration.
fn refusal_json(error: SamplingAssumptionUnionError, replay: bool) -> PyResult<String> {
    match error {
        SamplingAssumptionUnionError::Chain(refusal)
            if refusal.code() == antecedent::reason::reason_code!("cancelled_no_claim") =>
        {
            Err(cancelled())
        }
        SamplingAssumptionUnionError::Chain(refusal) => Ok(chain_refusal_json(&refusal, STAGE)),
        SamplingAssumptionUnionError::Invalid(message) => Ok(invalid_json(message, replay)),
        SamplingAssumptionUnionError::Io(error) => Err(crate::py_err(error)),
    }
}

fn result_json(scheme: Scheme, result: &SamplingAssumptionUnionResult) -> String {
    json!({
        "scheme": scheme.name(),
        "feature": scheme.feature(),
        "original_lower": result.original_lower,
        "original_upper": result.original_upper,
        "union_lower": result.union_lower,
        "union_upper": result.union_upper,
        "level": result.level,
        "calibration": result.calibration,
        "chain_identities": result.chain_identities,
        "program_identity": result.lowered_program_identity.to_hex(),
    })
    .to_string()
}

fn cancelled() -> PyErr {
    crate::CausalCancelledError::new_err("sampling assumption union cancelled")
}

/// A compiled and executed union, retained so the result it reproduces can be sealed.
#[pyclass(name = "SamplingAssumptionUnionHandle", skip_from_py_object)]
pub(crate) struct PySamplingAssumptionUnion {
    plan: Compiled,
    result: SamplingAssumptionUnionResult,
    scheme: Scheme,
    seed: u64,
}

#[pymethods]
impl PySamplingAssumptionUnion {
    /// The executed result as JSON: the plug-in and union endpoints, the level, the
    /// calibration standing, every chain identity and the program identity.
    #[getter]
    fn result_json(&self) -> String {
        result_json(self.scheme, &self.result)
    }

    /// The compiled route the program was lowered for.
    #[getter]
    fn route(&self) -> &'static str {
        self.plan.lowered().route()
    }

    /// The support decision of the unit the program's route runs under, as JSON; `null` while
    /// no unit is bound to the route.
    #[getter]
    fn unit_support_json(&self) -> String {
        crate::errors::unit_support_value(&self.plan.plan_view()).to_string()
    }

    /// Seal the declaration and the result after re-executing it under the bound seed.
    /// Returns `(bytes, refusal)`.
    #[pyo3(signature = (cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        cancel: Option<PyCancellationToken>,
    ) -> PyResult<Refused<Py<PyBytes>>> {
        let ctx = execution_context(self.seed, None, cancel);
        let exported = crate::detach_catch(py, || Ok(self.plan.export(&self.result, &ctx)))?;
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        match exported {
            Ok(bytes) => Ok((Some(PyBytes::new(py, &bytes).unbind()), None)),
            Err(error) => Ok((None, Some(refusal_json(error, false)?))),
        }
    }
}

/// Compile the declared union under `scheme`, execute it under `seed` and keep the program.
///
/// `request_json` is the scheme's request: the original chain draft and the raw unit
/// identities, groups, block length, replicate count and level, or the declared reruns.
/// Returns `(handle, refusal)`.
#[pyfunction]
#[pyo3(signature = (scheme, request_json, seed=1, cancel=None))]
fn sampling_assumption_union_compile(
    py: Python<'_>,
    scheme: &str,
    request_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<PySamplingAssumptionUnion>> {
    let scheme = Scheme::from_name(scheme)
        .ok_or_else(|| crate::value_err(format!("unknown resampling scheme `{scheme}`")))?;
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || {
        Ok(match compile(scheme, request_json, &ctx)? {
            Ok(plan) => plan.execute(&ctx).map(|result| (plan, result)),
            Err(error) => Err(error),
        })
    })?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    match outcome {
        Ok((plan, result)) => {
            Ok((Some(PySamplingAssumptionUnion { plan, result, scheme, seed }), None))
        }
        Err(error) => Ok((None, Some(refusal_json(error, false)?))),
    }
}

/// The feature an exported union declares; it selects the consumer, which checks it again.
#[derive(Deserialize)]
struct FeatureProbe {
    required_feature: String,
}

/// Independently consume exported union bytes under the retained program identity and the
/// seed they were produced with: every chain is recompiled and rerun and the result is
/// compared. Returns `(result_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, seed=1, cancel=None))]
fn sampling_assumption_union_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let expected = crate::digest::digest(expected_identity, "expected_identity")?;
    let probe: FeatureProbe = antecedent::io::from_cbor(artifact).map_err(crate::py_err)?;
    let Some(scheme) = Scheme::from_feature(&probe.required_feature) else {
        return Ok((
            None,
            Some(invalid_json("the artifact is not a sampling assumption union", true)),
        ));
    };
    let ctx = execution_context(seed, None, cancel);
    let outcome = crate::detach_catch(py, || Ok(consume(scheme, artifact, expected, &ctx)))?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    match outcome {
        Ok(result) => Ok((Some(result_json(scheme, &result)), None)),
        Err(error) => Ok((None, Some(refusal_json(error, true)?))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySamplingAssumptionUnion>()?;
    m.add_function(wrap_pyfunction!(sampling_assumption_union_compile, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_assumption_union_consume, m)?)?;
    Ok(())
}
