//! Python bridge for decisions on a compiled graph-imputation mixture.
//!
//! A decision reads the mixture at the source (the binary average effect, or the paired
//! treatment-world means), or transports it to an independent target frame: binary strata, a
//! Gaussian kernel density ratio over mixed covariates, or a raw frame missing at random. A
//! target can be resampled by rows or by whole clusters, and treatment worlds can be individual
//! predictive outcomes (Bernoulli for a binary outcome, a declared Gaussian residual law for a
//! continuous one). A transported fit and the chain its decision read can also be bundled, so a
//! consumer refits the fit and replays the chain.
//!
//! Python declares the contract, the target frame, the resampling policy and the residual law;
//! the producer is recompiled from the declaration its prepared handle retains. The declaration
//! chooses the route: the contract's coordinates choose the effect or the treatment-world family,
//! the producer's graph support chooses the confounded pretreatment route, and the target's
//! columns choose binary, kernel or missing-at-random transport. Every check, fit, decision,
//! artifact and refusal is Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as its own typed exception. No interval, coverage or decision reliability is
//! reported: every result says `not_measured`.

use antecedent::analysis::ChainDecisionError;
use antecedent::analysis::ChainDecisionResult;
use antecedent::analysis::composition::{
    BundleLimits, ChainBundleRecomputeError, CompiledGraphImputationTransportNodeBundle,
    GraphImputationTransportBundleError, GraphImputationTransportNodeBundleReceipt,
    compile_graph_imputation_transport_node_bundle_recomputation,
    consume_portable_graph_imputation_transport_node_bundle_recomputation,
    graph_imputation_transport_bundle_consumer,
};
use antecedent::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSupport};
use antecedent::analysis::graph_imputation_effect_decision::{
    CompiledGraphImputationEffectDecision, CompiledGraphImputationWorldDecision,
    GaussianPredictivePolicy, GraphImputationEffectDecisionError,
    GraphImputationEffectDecisionResult, GraphImputationWorldDecisionResult,
    compile_graph_imputation_bernoulli_predictive_world_decision,
    compile_graph_imputation_effect_decision,
    compile_graph_imputation_gaussian_target_predictive_world_decision,
    compile_graph_imputation_predictive_cluster_bootstrap_decision,
    compile_graph_imputation_transport_world_bootstrap_decision,
    compile_graph_imputation_transport_world_decision, compile_graph_imputation_world_decision,
    consume_graph_imputation_effect_decision, consume_graph_imputation_world_decision,
};
use antecedent::analysis::graph_imputation_kernel_transport_decision::{
    CompiledGraphImputationKernelTransportDecision, TargetFrameBootstrap as KernelFrameBootstrap,
    compile_graph_imputation_confounded_kernel_transport_decision,
    compile_graph_imputation_kernel_transport_cluster_bootstrap_decision,
    compile_graph_imputation_kernel_transport_decision,
    compile_graph_imputation_kernel_transport_target_bootstrap_decision,
    consume_graph_imputation_kernel_transport_decision,
};
use antecedent::analysis::graph_imputation_mar_transport_decision::{
    CompiledMarTransportDecision, MarClusterBootstrap, MarTransportError, MarTransportResult,
    compile_graph_imputation_mar_transport_decision,
    compile_graph_imputation_mar_transport_with_cluster,
    consume_graph_imputation_mar_transport_decision,
};
use antecedent::analysis::graph_imputation_transport_decision::{
    CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError,
    GraphImputationTransportDecisionResult, TargetClusterBootstrap, TargetFrameBootstrap,
    compile_graph_imputation_confounded_transport_decision,
    compile_graph_imputation_confounded_transport_target_bootstrap_decision,
    compile_graph_imputation_transport_cluster_bootstrap_decision,
    compile_graph_imputation_transport_decision,
    compile_graph_imputation_transport_target_bootstrap_decision,
    consume_graph_imputation_transport_decision,
};
use antecedent::analysis::inverse_query::DecisionContract;
use antecedent::design::decision_artifact::contract_from_json_refusal;
use antecedent::io::IoError;
use antecedent::prelude::ExecutionContext;
use antecedent::reason::reason_code;
use antecedent::reason::reason_details::{composition_bundle, graph_imputation};
use antecedent::scientific::{ScientificQuantity, SemanticDigest};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size, external_refusal_json};
use crate::composition_bundle_api::{PyBundleBuilder, PyCompositionBundle};
use crate::graph_selection_api::{
    GraphImputationDeclaration, PyCompiledGraphImputation, producer_refusal,
};

const STAGE: &str = "graph_imputation_decision";

/// The functional of an average-effect coordinate; any other coordinate is a treatment world.
const EFFECT_FUNCTIONAL: &str = "average_effect";
/// The functional of an individual predictive treatment-world coordinate.
const PREDICTIVE_FUNCTIONAL: &str = "outcome";

type Refused<T> = (Option<T>, Option<String>);

// ------------------------------------------------------------------------------- refusals

fn refusal(code: &str, detail: &str, message: &str) -> String {
    RefusalJson::new(code, STAGE, detail, message).json()
}

fn invalid(detail: &str, message: impl std::fmt::Display) -> String {
    refusal(reason_code!("invalid_argument"), detail, &message.to_string())
}

fn io_refusal(error: &IoError) -> String {
    let code = error.reason_code().unwrap_or(reason_code!("invalid_argument"));
    let detail = if code == reason_code!("cancelled_no_claim") {
        graph_imputation::CANCELLED
    } else if code == reason_code!("execution_resource_limit") {
        graph_imputation::RESOURCE_LIMIT
    } else {
        graph_imputation::EXECUTION_REFUSED
    };
    refusal(code, detail, &error.to_string())
}

fn chain_refusal(error: &ChainDecisionError) -> String {
    match error {
        ChainDecisionError::Chain(chain) => {
            RefusalJson::new(chain.code(), STAGE, chain.detail(), &chain.to_string()).json()
        }
        ChainDecisionError::Contract(contract) => external_refusal_json(&contract.to_refusal()),
        ChainDecisionError::Io(io) => io_refusal(io),
        ChainDecisionError::Evaluation(_) | ChainDecisionError::Coordinate(_) => refusal(
            reason_code!("decision_contract_unsatisfied"),
            graph_imputation::EXECUTION_REFUSED,
            &error.to_string(),
        ),
    }
}

fn effect_refusal(error: &GraphImputationEffectDecisionError, detail: &str) -> String {
    match error {
        GraphImputationEffectDecisionError::Invalid(message) => invalid(detail, message),
        GraphImputationEffectDecisionError::Producer(producer) => producer_refusal(producer),
        GraphImputationEffectDecisionError::Contract(contract) => {
            external_refusal_json(&contract.to_refusal())
        }
        GraphImputationEffectDecisionError::Chain(chain) => chain_refusal(chain),
        GraphImputationEffectDecisionError::Io(io) => io_refusal(io),
    }
}

fn transport_refusal(error: &GraphImputationTransportDecisionError, detail: &str) -> String {
    match error {
        GraphImputationTransportDecisionError::Invalid(message) => invalid(detail, message),
        GraphImputationTransportDecisionError::Producer(producer) => producer_refusal(producer),
        GraphImputationTransportDecisionError::Contract(contract) => {
            external_refusal_json(&contract.to_refusal())
        }
        GraphImputationTransportDecisionError::Chain(chain) => chain_refusal(chain),
        GraphImputationTransportDecisionError::Io(io) => io_refusal(io),
    }
}

fn mar_refusal(error: &MarTransportError, detail: &str) -> String {
    match error {
        MarTransportError::Invalid(message) => invalid(detail, message),
        MarTransportError::Producer(producer) => producer_refusal(producer),
        MarTransportError::Contract(contract) => external_refusal_json(&contract.to_refusal()),
        MarTransportError::Chain(chain) => chain_refusal(chain),
        MarTransportError::Io(io) => io_refusal(io),
    }
}

fn recompute_refusal(error: &ChainBundleRecomputeError) -> String {
    match error {
        ChainBundleRecomputeError::Bundle(bundle) => {
            crate::composition_bundle_api::error_refusal(bundle)
        }
        ChainBundleRecomputeError::Chain(chain) => {
            RefusalJson::new(chain.code(), STAGE, chain.detail(), &chain.to_string()).json()
        }
        ChainBundleRecomputeError::Io(io) => io_refusal(io),
        ChainBundleRecomputeError::Invalid(message) => {
            invalid(composition_bundle::INVALID_CONTAINER, message)
        }
    }
}

fn transport_bundle_refusal(error: &GraphImputationTransportBundleError) -> String {
    match error {
        GraphImputationTransportBundleError::Bundle(inner) => recompute_refusal(inner),
        GraphImputationTransportBundleError::Invalid(message) => {
            invalid(composition_bundle::INVALID_CONTAINER, message)
        }
        GraphImputationTransportBundleError::NotSupported(refused) => {
            RefusalJson::new(refused.code(), STAGE, refused.detail, &refused.to_string()).json()
        }
    }
}

// ---------------------------------------------------------------------------- declarations

/// A target frame as Python declares it: columns and selected mechanisms by source column name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetIn {
    columns: Vec<(String, Vec<Option<f64>>)>,
    selection: Vec<String>,
    source_population: String,
    target_population: String,
    target_sample: String,
    #[serde(default)]
    source_units: Option<Vec<String>>,
    #[serde(default)]
    target_units: Option<Vec<String>>,
    #[serde(default)]
    clusters: Option<Vec<String>>,
    #[serde(default)]
    bandwidths: Vec<(String, f64)>,
    #[serde(default)]
    min_kernel_overlap: f64,
    #[serde(default)]
    missing_at_random: Option<MissingAtRandomIn>,
}

/// The declared missing-at-random design of a raw target frame.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MissingAtRandomIn {
    observation_mechanism: String,
    sampling_design: String,
    observation_probabilities: Vec<f64>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    #[serde(default)]
    always_observed: Vec<String>,
}

/// Target resampling: whole clusters when the target declares them, rows otherwise.
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapIn {
    replicates: u32,
    seed: u64,
}

/// The declared Gaussian residual law of two continuous predictive treatment worlds.
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidualIn {
    variance: f64,
    cross_world_correlation: f64,
    seed: u64,
}

/// The native target declaration, kept as its wire value until the route's compiler reads it.
enum Frame {
    Binary(Json),
    Kernel(Json),
    MissingAtRandom(Json),
}

struct DeclaredTarget {
    frame: Frame,
    clusters: Option<Vec<String>>,
}

fn parse<T: DeserializeOwned>(text: &str, limit: usize, what: &str) -> PyResult<T> {
    check_declared_size(text, limit, what)?;
    serde_json::from_str(text).map_err(|error| crate::value_err(format!("invalid {what}: {error}")))
}

/// Read a native target declaration into the type the route's compiler takes.
fn from_json<T: DeserializeOwned>(value: Json) -> Result<T, String> {
    serde_json::from_value(value)
        .map_err(|error| invalid(graph_imputation::COMPILED, format!("invalid target: {error}")))
}

/// Resolve the declared target against the producer's columns and source sample identity.
#[allow(clippy::too_many_lines, reason = "one declaration read in order in one place")]
fn target_of(
    input: TargetIn,
    declaration: &GraphImputationDeclaration,
    source_sample: &str,
    confounded: bool,
) -> Result<DeclaredTarget, String> {
    let detail = graph_imputation::COMPILED;
    let index = |name: &str| -> Result<u32, String> {
        declaration
            .names
            .iter()
            .position(|column| column == name)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| {
                invalid(detail, format!("`{name}` is not a column of the graph-imputation source"))
            })
    };
    let mut columns = input
        .columns
        .into_iter()
        .map(|(name, values)| Ok((index(name.as_str())?, values)))
        .collect::<Result<Vec<_>, String>>()?;
    columns.sort_by_key(|(id, _)| *id);
    let mut selection =
        input.selection.iter().map(|name| index(name.as_str())).collect::<Result<Vec<_>, _>>()?;
    selection.sort_unstable();
    let mut bandwidths = input
        .bandwidths
        .into_iter()
        .map(|(name, bandwidth)| Ok((index(name.as_str())?, bandwidth)))
        .collect::<Result<Vec<_>, String>>()?;
    bandwidths.sort_by_key(|(id, _)| *id);
    if bandwidths.is_empty() && input.min_kernel_overlap != 0.0 {
        return Err(invalid(detail, "a kernel overlap floor needs kernel bandwidths"));
    }
    let rows = columns.first().map_or(0, |(_, values)| values.len());
    let source_units = input
        .source_units
        .unwrap_or_else(|| (0..declaration.rows).map(|row| format!("source-{row}")).collect());
    let target_units = input
        .target_units
        .unwrap_or_else(|| (0..rows).map(|row| format!("target-{row}")).collect());
    if let Some(design) = input.missing_at_random {
        let mut always = design
            .always_observed
            .iter()
            .map(|name| index(name.as_str()))
            .collect::<Result<Vec<_>, _>>()?;
        always.sort_unstable();
        let clusters = input.clusters.unwrap_or_else(|| target_units.clone());
        let weights = design.weights.unwrap_or_else(|| vec![1.0; rows]);
        let frame = json!({
            "observation_mechanism_id": design.observation_mechanism,
            "sampling_design_id": design.sampling_design,
            "frame": {
                "columns": columns,
                "row_ids": target_units,
                "cluster_ids": clusters,
                "weights": weights,
                "observation_probabilities": design.observation_probabilities,
                "always_observed": always,
            },
            "selection_targets": selection,
            "source_population": input.source_population,
            "target_population": input.target_population,
            "source_sample": source_sample,
            "target_sample": input.target_sample,
            "source_unit_ids": source_units,
            "bandwidths": bandwidths,
            "min_kernel_overlap": input.min_kernel_overlap,
            "kernel": !bandwidths.is_empty(),
            "confounded": confounded,
        });
        return Ok(DeclaredTarget { frame: Frame::MissingAtRandom(frame), clusters: None });
    }
    if columns.iter().any(|(_, values)| values.iter().any(Option::is_none)) {
        return Err(invalid(
            detail,
            "a target column has a missing cell; declare the target missing at random",
        ));
    }
    let columns: Vec<(u32, Vec<f64>)> = columns
        .into_iter()
        .map(|(id, values)| (id, values.into_iter().flatten().collect()))
        .collect();
    let mut frame = json!({
        "columns": columns,
        "selection_targets": selection,
        "source_population": input.source_population,
        "target_population": input.target_population,
        "source_sample": source_sample,
        "target_sample": input.target_sample,
        "source_unit_ids": source_units,
        "target_unit_ids": target_units,
    });
    let frame = if bandwidths.is_empty() {
        Frame::Binary(frame)
    } else {
        if let Json::Object(map) = &mut frame {
            map.insert("bandwidths".into(), json!(bandwidths));
            map.insert("min_kernel_overlap".into(), json!(input.min_kernel_overlap));
        }
        Frame::Kernel(frame)
    };
    Ok(DeclaredTarget { frame, clusters: input.clusters })
}

/// The distinct coordinates the contract's actions read, in the order they first appear.
fn coordinates_of(contract: &DecisionContract) -> Vec<ScientificQuantity> {
    let mut coordinates: Vec<ScientificQuantity> = Vec::new();
    for input in contract.actions.iter().flat_map(|action| &action.inputs) {
        if !coordinates.contains(input) {
            coordinates.push(input.clone());
        }
    }
    coordinates
}

/// The two treatment-world coordinates, control before active.
fn world_pair(mut coordinates: Vec<ScientificQuantity>) -> Result<[ScientificQuantity; 2], String> {
    coordinates.sort_by(|left, right| left.regime_id.cmp(&right.regime_id));
    <[ScientificQuantity; 2]>::try_from(coordinates).map_err(|_| {
        invalid(
            graph_imputation::COMPILED,
            "a treatment-world decision reads exactly one coordinate for each of two treatment \
             worlds",
        )
    })
}

// --------------------------------------------------------------------------------- routes

#[derive(Clone, Copy)]
enum Route {
    Effect,
    World,
    Transport,
    Kernel,
    MissingAtRandom,
}

impl Route {
    const fn label(self) -> &'static str {
        match self {
            Self::Effect => "effect",
            Self::World => "world",
            Self::Transport => "transport",
            Self::Kernel => "kernel_transport",
            Self::MissingAtRandom => "mar_transport",
        }
    }
}

enum Compiled {
    Effect(Box<CompiledGraphImputationEffectDecision>),
    World(Box<CompiledGraphImputationWorldDecision>),
    Transport(Box<CompiledGraphImputationTransportDecision>),
    Kernel(Box<CompiledGraphImputationKernelTransportDecision>),
    MissingAtRandom(Box<CompiledMarTransportDecision>),
}

enum Executed {
    Effect(Box<GraphImputationEffectDecisionResult>),
    World(Box<GraphImputationWorldDecisionResult>),
    Transport(Box<GraphImputationTransportDecisionResult>),
    MissingAtRandom(Box<MarTransportResult>),
}

fn hex(bytes: [u8; 32]) -> String {
    SemanticDigest::from_bytes(bytes).to_hex()
}

fn decision_json(decision: &ChainDecisionResult) -> Json {
    use antecedent::design::decision_eval::Verdict;
    let evaluation = &decision.decision;
    let (verdict, chosen): (&str, Vec<&str>) = match &evaluation.verdict {
        Verdict::UniquelyOptimal(action) => ("uniquely_optimal", vec![action.as_str()]),
        Verdict::Indistinguishable(actions) => {
            ("indistinguishable", actions.iter().map(String::as_str).collect())
        }
        Verdict::NoAdmissibleAction => ("no_admissible_action", Vec::new()),
    };
    let actions: Vec<Json> = evaluation
        .actions
        .iter()
        .map(|action| {
            json!({
                "id": action.id,
                "admissible": action.admissible,
                "exclusions": action.exclusions.iter().map(|exclusion| json!({
                    "constraint_id": exclusion.constraint_id,
                    "probability": exclusion.probability,
                    "required": exclusion.required,
                })).collect::<Vec<_>>(),
                "expected_utility": action.expected_utility,
                "value": action.value,
                "standard_error": action.standard_error,
                "expected_regret": action.expected_regret,
                "max_regret": action.max_regret,
            })
        })
        .collect();
    json!({
        "contract_identity": evaluation.contract_identity,
        "criterion_label": evaluation.criterion_label,
        "verdict": verdict,
        "chosen": chosen,
        "evpi": evaluation.evpi,
        "n_draws": evaluation.n_draws,
        "effective_draws": evaluation.effective_draws,
        "assumptions": evaluation.assumptions,
        "actions": actions,
        "chain_identity": decision.chain.identity(),
        "chain_program": decision.lowered_program_identity.to_hex(),
    })
}

/// The JSON of a target-standardized point result (transport, kernel or missing at random).
macro_rules! point_result_json {
    ($route:expr, $result:expr) => {{
        let result = $result;
        json!({
            "route": $route.label(),
            "program": result.lowered_program_identity.to_hex(),
            "calibration": result.calibration,
            "receipt": hex(result.producer.receipt),
            "mean": result.producer.mean,
            "diagnostic_sd": result.producer.diagnostic_sd,
            "draws": result.producer.draws,
            "worlds": Vec::<[f64; 2]>::new(),
            "decision": decision_json(&result.decision),
        })
    }};
}

impl Executed {
    fn decision(&self) -> &ChainDecisionResult {
        match self {
            Self::Effect(result) => &result.decision,
            Self::World(result) => &result.decision,
            Self::Transport(result) => &result.decision,
            Self::MissingAtRandom(result) => &result.decision,
        }
    }

    fn json(&self, route: Route) -> String {
        let value = match self {
            Self::Effect(result) => json!({
                "route": route.label(),
                "program": result.lowered_program_identity.to_hex(),
                "calibration": result.calibration,
                "receipt": hex(result.producer.receipt),
                "mean": result.producer.mean,
                "diagnostic_sd": result.producer.diagnostic_sd,
                "draws": result.producer.effect_draws(),
                "worlds": Vec::<[f64; 2]>::new(),
                "decision": decision_json(&result.decision),
            }),
            Self::World(result) => json!({
                "route": route.label(),
                "program": result.lowered_program_identity.to_hex(),
                "calibration": result.calibration,
                "receipt": hex(result.producer.receipt),
                "mean": Json::Null,
                "diagnostic_sd": Json::Null,
                "draws": Vec::<f64>::new(),
                "worlds": result.producer.worlds,
                "decision": decision_json(&result.decision),
            }),
            Self::Transport(result) => point_result_json!(route, result),
            Self::MissingAtRandom(result) => point_result_json!(route, result),
        };
        value.to_string()
    }
}

impl Compiled {
    const fn route(&self) -> Route {
        match self {
            Self::Effect(_) => Route::Effect,
            Self::World(_) => Route::World,
            Self::Transport(_) => Route::Transport,
            Self::Kernel(_) => Route::Kernel,
            Self::MissingAtRandom(_) => Route::MissingAtRandom,
        }
    }

    fn program(&self) -> SemanticDigest {
        match self {
            Self::Effect(plan) => plan.lowered().identity(),
            Self::World(plan) => plan.lowered().identity(),
            Self::Transport(plan) => plan.lowered().identity(),
            Self::Kernel(plan) => plan.lowered().identity(),
            Self::MissingAtRandom(plan) => plan.lowered().identity(),
        }
    }

    fn plan_view(&self) -> antecedent::planner::PlanView {
        match self {
            Self::Effect(plan) => plan.plan_view(),
            Self::World(plan) => plan.plan_view(),
            Self::Transport(plan) => plan.plan_view(),
            Self::Kernel(plan) => plan.plan_view(),
            Self::MissingAtRandom(plan) => plan.plan_view(),
        }
    }

    fn execute(&self, ctx: &ExecutionContext) -> Result<Executed, String> {
        let detail = graph_imputation::EXECUTION_REFUSED;
        match self {
            Self::Effect(plan) => plan
                .execute(ctx)
                .map(|result| Executed::Effect(Box::new(result)))
                .map_err(|error| effect_refusal(&error, detail)),
            Self::World(plan) => plan
                .execute(ctx)
                .map(|result| Executed::World(Box::new(result)))
                .map_err(|error| effect_refusal(&error, detail)),
            Self::Transport(plan) => plan
                .execute(ctx)
                .map(|result| Executed::Transport(Box::new(result)))
                .map_err(|error| transport_refusal(&error, detail)),
            Self::Kernel(plan) => plan
                .execute(ctx)
                .map(|result| Executed::Transport(Box::new(result)))
                .map_err(|error| transport_refusal(&error, detail)),
            Self::MissingAtRandom(plan) => plan
                .execute(ctx)
                .map(|result| Executed::MissingAtRandom(Box::new(result)))
                .map_err(|error| mar_refusal(&error, detail)),
        }
    }

    fn export(
        &self,
        executed: &Executed,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, String> {
        let detail = graph_imputation::EXECUTION_REFUSED;
        match (self, executed) {
            (Self::Effect(plan), Executed::Effect(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| effect_refusal(&error, detail)),
            (Self::World(plan), Executed::World(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| effect_refusal(&error, detail)),
            (Self::Transport(plan), Executed::Transport(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| transport_refusal(&error, detail)),
            (Self::Kernel(plan), Executed::Transport(result)) => plan
                .export(result, artifact_id, ctx)
                .map_err(|error| transport_refusal(&error, detail)),
            (Self::MissingAtRandom(plan), Executed::MissingAtRandom(result)) => {
                plan.export(result, artifact_id, ctx).map_err(|error| mar_refusal(&error, detail))
            }
            _ => Err(invalid(detail, "the executed result belongs to another route")),
        }
    }
}

/// The average-effect family: at the source, or transported to a declared target frame.
#[allow(clippy::too_many_lines, reason = "one route table read in order in one place")]
fn effect_route(
    producer: CompiledGraphImputation,
    contract: DecisionContract,
    quantity: ScientificQuantity,
    target: Option<DeclaredTarget>,
    bootstrap: Option<BootstrapIn>,
    residual: Option<ResidualIn>,
    confounded: bool,
) -> Result<Compiled, String> {
    let detail = graph_imputation::COMPILED;
    if residual.is_some() {
        return Err(invalid(
            detail,
            "a Gaussian residual law belongs to predictive treatment worlds, not an average effect",
        ));
    }
    let Some(DeclaredTarget { frame, clusters }) = target else {
        if bootstrap.is_some() {
            return Err(invalid(detail, "resampling needs a target frame"));
        }
        return compile_graph_imputation_effect_decision(producer, quantity, contract)
            .map(|plan| Compiled::Effect(Box::new(plan)))
            .map_err(|error| effect_refusal(&error, detail));
    };
    let rows = |policy: BootstrapIn| TargetFrameBootstrap {
        replicates: policy.replicates,
        seed: policy.seed,
    };
    let kernel_rows = |policy: BootstrapIn| KernelFrameBootstrap {
        replicates: policy.replicates,
        seed: policy.seed,
    };
    let groups = |policy: BootstrapIn, cluster_ids: Vec<String>| TargetClusterBootstrap {
        replicates: policy.replicates,
        seed: policy.seed,
        cluster_ids,
    };
    match frame {
        Frame::Binary(value) => {
            let plan = match (bootstrap, clusters) {
                (None, _) if confounded => compile_graph_imputation_confounded_transport_decision(
                    producer,
                    from_json(value)?,
                    quantity,
                    contract,
                ),
                (None, _) => compile_graph_imputation_transport_decision(
                    producer,
                    from_json(value)?,
                    quantity,
                    contract,
                ),
                (Some(policy), Some(cluster_ids)) => {
                    compile_graph_imputation_transport_cluster_bootstrap_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        groups(policy, cluster_ids),
                        confounded,
                    )
                }
                (Some(policy), None) if confounded => {
                    compile_graph_imputation_confounded_transport_target_bootstrap_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        rows(policy),
                    )
                }
                (Some(policy), None) => {
                    compile_graph_imputation_transport_target_bootstrap_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        rows(policy),
                    )
                }
            };
            plan.map(|plan| Compiled::Transport(Box::new(plan)))
                .map_err(|error| transport_refusal(&error, detail))
        }
        Frame::Kernel(value) => {
            let plan = match (bootstrap, clusters) {
                (None, _) if confounded => {
                    compile_graph_imputation_confounded_kernel_transport_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        None,
                    )
                }
                (None, _) => compile_graph_imputation_kernel_transport_decision(
                    producer,
                    from_json(value)?,
                    quantity,
                    contract,
                ),
                (Some(policy), Some(cluster_ids)) => {
                    compile_graph_imputation_kernel_transport_cluster_bootstrap_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        groups(policy, cluster_ids),
                        confounded,
                    )
                }
                (Some(policy), None) if confounded => {
                    compile_graph_imputation_confounded_kernel_transport_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        Some(kernel_rows(policy)),
                    )
                }
                (Some(policy), None) => {
                    compile_graph_imputation_kernel_transport_target_bootstrap_decision(
                        producer,
                        from_json(value)?,
                        quantity,
                        contract,
                        kernel_rows(policy),
                    )
                }
            };
            plan.map(|plan| Compiled::Kernel(Box::new(plan)))
                .map_err(|error| transport_refusal(&error, detail))
        }
        Frame::MissingAtRandom(value) => {
            let plan = match bootstrap {
                None => compile_graph_imputation_mar_transport_decision(
                    producer,
                    from_json(value)?,
                    quantity,
                    contract,
                ),
                Some(policy) => compile_graph_imputation_mar_transport_with_cluster(
                    producer,
                    from_json(value)?,
                    quantity,
                    contract,
                    Some(MarClusterBootstrap { replicates: policy.replicates, seed: policy.seed }),
                ),
            };
            plan.map(|plan| Compiled::MissingAtRandom(Box::new(plan)))
                .map_err(|error| mar_refusal(&error, detail))
        }
    }
}

/// The paired treatment-world family: source means, target means, or target predictive worlds.
#[allow(clippy::too_many_lines, reason = "one route table read in order in one place")]
fn world_route(
    producer: CompiledGraphImputation,
    contract: DecisionContract,
    quantities: [ScientificQuantity; 2],
    target: Option<DeclaredTarget>,
    bootstrap: Option<BootstrapIn>,
    residual: Option<ResidualIn>,
) -> Result<Compiled, String> {
    let detail = graph_imputation::COMPILED;
    let predictive =
        quantities.iter().all(|quantity| quantity.functional_id == PREDICTIVE_FUNCTIONAL);
    let Some(DeclaredTarget { frame, clusters }) = target else {
        if bootstrap.is_some() || residual.is_some() {
            return Err(invalid(detail, "resampling and a residual law need a target frame"));
        }
        return compile_graph_imputation_world_decision(producer, quantities, contract)
            .map(|plan| Compiled::World(Box::new(plan)))
            .map_err(|error| effect_refusal(&error, detail));
    };
    let Frame::Binary(value) = frame else {
        return Err(invalid(detail, "treatment-world decisions take a binary target frame"));
    };
    let plan = if predictive {
        if bootstrap.is_some() && clusters.is_none() {
            return Err(invalid(
                detail,
                "predictive treatment worlds resample whole target clusters; declare the \
                 target's clusters",
            ));
        }
        let fixed = match residual {
            None => compile_graph_imputation_bernoulli_predictive_world_decision(
                producer,
                from_json(value)?,
                quantities,
                contract,
            ),
            Some(law) => compile_graph_imputation_gaussian_target_predictive_world_decision(
                producer,
                from_json(value)?,
                quantities,
                contract,
                GaussianPredictivePolicy {
                    residual_variance: law.variance,
                    cross_world_correlation: law.cross_world_correlation,
                    seed: law.seed,
                },
            ),
        };
        match (fixed, bootstrap, clusters) {
            (Ok(fixed), Some(policy), Some(cluster_ids)) => {
                compile_graph_imputation_predictive_cluster_bootstrap_decision(
                    fixed,
                    TargetClusterBootstrap {
                        replicates: policy.replicates,
                        seed: policy.seed,
                        cluster_ids,
                    },
                )
            }
            (fixed, _, _) => fixed,
        }
    } else {
        if residual.is_some() {
            return Err(invalid(
                detail,
                "a Gaussian residual law belongs to predictive treatment worlds, not world means",
            ));
        }
        match (bootstrap, clusters) {
            (None, _) => compile_graph_imputation_transport_world_decision(
                producer,
                from_json(value)?,
                quantities,
                contract,
            ),
            (Some(policy), None) => compile_graph_imputation_transport_world_bootstrap_decision(
                producer,
                from_json(value)?,
                quantities,
                contract,
                TargetFrameBootstrap { replicates: policy.replicates, seed: policy.seed },
            ),
            (Some(_), Some(_)) => {
                return Err(invalid(
                    detail,
                    "treatment-world means resample target rows; whole-cluster resampling \
                     belongs to predictive worlds",
                ));
            }
        }
    };
    plan.map(|plan| Compiled::World(Box::new(plan))).map_err(|error| effect_refusal(&error, detail))
}

fn compile(
    declaration: &GraphImputationDeclaration,
    contract: DecisionContract,
    target: Option<TargetIn>,
    bootstrap: Option<BootstrapIn>,
    residual: Option<ResidualIn>,
    ctx: &ExecutionContext,
) -> Result<Compiled, String> {
    let coordinates = coordinates_of(&contract);
    if coordinates.is_empty() {
        return Err(invalid(
            graph_imputation::COMPILED,
            "the contract's actions read no coordinate",
        ));
    }
    let producer = declaration.compile(ctx).map_err(|error| producer_refusal(&error))?;
    let confounded = declaration.spec.support == GraphImputationSupport::PreTreatmentCovariates;
    let source_sample = producer.source_identity().to_hex();
    let target = target
        .map(|target| target_of(target, declaration, &source_sample, confounded))
        .transpose()?;
    if coordinates.iter().all(|quantity| quantity.functional_id == EFFECT_FUNCTIONAL) {
        // Every action reads the one effect coordinate; the compiler refuses any other.
        let Some(quantity) = coordinates.into_iter().next() else {
            return Err(invalid(graph_imputation::COMPILED, "the contract reads no coordinate"));
        };
        effect_route(producer, contract, quantity, target, bootstrap, residual, confounded)
    } else {
        world_route(producer, contract, world_pair(coordinates)?, target, bootstrap, residual)
    }
}

/// The support decision of the unit a compiled program runs under, as JSON; `None` while no
/// unit is bound to its route. Reading it changes no program identity.
pub(crate) fn graph_unit_support_json(view: &antecedent::planner::PlanView) -> Option<String> {
    view.unit_support().map(|unit| {
        json!({
            "unit": unit.unit(),
            "status": unit.status().as_str(),
            "supported": unit.supported(),
            "missing_evidence": unit.missing_evidence(),
        })
        .to_string()
    })
}

// -------------------------------------------------------------------------- Python handles

/// A compiled decision on the graph-imputation mixture and its last execution.
#[pyclass(name = "GraphImputationDecisionHandle", skip_from_py_object)]
struct PyGraphImputationDecision {
    compiled: Compiled,
    executed: Option<Executed>,
    seed: u64,
}

#[pymethods]
impl PyGraphImputationDecision {
    /// The compiled program identity (lowercase hex) a consumer retains.
    fn program(&self) -> String {
        self.compiled.program().to_hex()
    }

    /// The route family: `effect`, `world`, `transport`, `kernel_transport` or `mar_transport`.
    fn route(&self) -> &'static str {
        self.compiled.route().label()
    }

    /// The support decision of the unit the compiled program runs under, as JSON; `None` while
    /// no unit is bound to its route.
    fn unit_support(&self) -> Option<String> {
        graph_unit_support_json(&self.compiled.plan_view())
    }

    /// Refit the graph and imputation schedule and evaluate the decision: `(json, refusal)`.
    fn execute(&mut self, py: Python<'_>) -> PyResult<Refused<String>> {
        let ctx = crate::execution_api::context(self.seed);
        let compiled = &self.compiled;
        let executed = crate::detach_catch(py, move || Ok(compiled.execute(&ctx)))?;
        Ok(match executed {
            Ok(executed) => {
                let text = executed.json(self.compiled.route());
                self.executed = Some(executed);
                (Some(text), None)
            }
            Err(refused) => (None, Some(refused)),
        })
    }

    /// Replay the last execution and seal it as portable bytes: `(bytes, refusal)`.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> PyResult<Refused<Py<PyBytes>>> {
        let Some(executed) = &self.executed else {
            return Ok((
                None,
                Some(invalid(graph_imputation::EXECUTION_REFUSED, "execute before exporting")),
            ));
        };
        let ctx = crate::execution_api::context(self.seed);
        let compiled = &self.compiled;
        let bytes =
            crate::detach_catch(py, move || Ok(compiled.export(executed, artifact_id, &ctx)))?;
        Ok(match bytes {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(refused) => (None, Some(refused)),
        })
    }

    /// The propagation-chain artifact the last execution decided on: `(bytes, refusal)`.
    fn chain_artifact(&self, py: Python<'_>) -> Refused<Py<PyBytes>> {
        let Some(executed) = &self.executed else {
            return (
                None,
                Some(invalid(graph_imputation::EXECUTION_REFUSED, "execute before exporting")),
            );
        };
        match executed.decision().chain.export() {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(chain) => (
                None,
                Some(
                    RefusalJson::new(chain.code(), STAGE, chain.detail(), &chain.to_string())
                        .json(),
                ),
            ),
        }
    }
}

/// Compile a decision on a prepared graph-imputation handle: `(handle, refusal)`.
#[pyfunction]
#[pyo3(signature = (producer, contract_json, target_json=None, bootstrap_json=None, residual_json=None, seed=1))]
#[allow(clippy::needless_pass_by_value)]
fn graph_imputation_decision_compile(
    py: Python<'_>,
    producer: PyRef<'_, PyCompiledGraphImputation>,
    contract_json: &str,
    target_json: Option<&str>,
    bootstrap_json: Option<&str>,
    residual_json: Option<&str>,
    seed: u64,
) -> PyResult<Refused<PyGraphImputationDecision>> {
    check_declared_size(
        contract_json,
        crate::bridge::MAX_CONFIG_DECLARATION_BYTES,
        "decision contract",
    )?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refused) => return Ok((None, Some(external_refusal_json(&refused)))),
    };
    let target = target_json
        .map(|text| {
            parse::<TargetIn>(text, crate::bridge::MAX_DATA_DECLARATION_BYTES, "transport target")
        })
        .transpose()?;
    let bootstrap = bootstrap_json
        .map(|text| {
            parse::<BootstrapIn>(text, crate::bridge::MAX_CONFIG_DECLARATION_BYTES, "bootstrap")
        })
        .transpose()?;
    let residual = residual_json
        .map(|text| {
            parse::<ResidualIn>(text, crate::bridge::MAX_CONFIG_DECLARATION_BYTES, "residual law")
        })
        .transpose()?;
    let declaration = producer.declaration.clone();
    let ctx = crate::execution_api::context(seed);
    let compiled = crate::detach_catch(py, move || {
        Ok(compile(&declaration, contract, target, bootstrap, residual, &ctx))
    })?;
    Ok(match compiled {
        Ok(compiled) => (Some(PyGraphImputationDecision { compiled, executed: None, seed }), None),
        Err(refused) => (None, Some(refused)),
    })
}

/// The leading feature fields of a decision artifact; every other field is ignored here and
/// checked by the route's own consumer.
#[derive(Deserialize)]
struct FeatureProbe {
    #[serde(default)]
    required_feature: Option<String>,
    #[serde(default)]
    feature: Option<String>,
}

fn route_of(data: &[u8]) -> Result<Route, String> {
    let probe: FeatureProbe =
        antecedent::io::from_cbor(data).map_err(|error| io_refusal(&error))?;
    match (probe.required_feature.as_deref(), probe.feature.as_deref()) {
        (Some(feature), _) if feature.starts_with("graph_imputation_effect_decision") => {
            Ok(Route::Effect)
        }
        (_, Some(feature)) if feature.starts_with("graph_imputation_mar_transport") => {
            Ok(Route::MissingAtRandom)
        }
        (_, Some(_)) => Ok(Route::World),
        (Some(feature), _) if feature.contains("kernel_transport") => Ok(Route::Kernel),
        (Some(_), _) => Ok(Route::Transport),
        (None, None) => Err(invalid(
            graph_imputation::EXECUTION_REFUSED,
            "the bytes are not a graph-imputation decision artifact",
        )),
    }
}

fn consume(data: &[u8], program: SemanticDigest, ctx: &ExecutionContext) -> Result<String, String> {
    let route = route_of(data)?;
    let detail = graph_imputation::EXECUTION_REFUSED;
    let executed = match route {
        Route::Effect => consume_graph_imputation_effect_decision(data, program, ctx)
            .map(|result| Executed::Effect(Box::new(result)))
            .map_err(|error| effect_refusal(&error, detail)),
        Route::World => consume_graph_imputation_world_decision(data, program, ctx)
            .map(|result| Executed::World(Box::new(result)))
            .map_err(|error| effect_refusal(&error, detail)),
        Route::Transport => consume_graph_imputation_transport_decision(data, program, ctx)
            .map(|result| Executed::Transport(Box::new(result)))
            .map_err(|error| transport_refusal(&error, detail)),
        Route::Kernel => consume_graph_imputation_kernel_transport_decision(data, program, ctx)
            .map(|result| Executed::Transport(Box::new(result)))
            .map_err(|error| transport_refusal(&error, detail)),
        Route::MissingAtRandom => {
            consume_graph_imputation_mar_transport_decision(data, program, ctx)
                .map(|result| Executed::MissingAtRandom(Box::new(result)))
                .map_err(|error| mar_refusal(&error, detail))
        }
    }?;
    Ok(executed.json(route))
}

/// Independently replay a decision artifact under the retained program identity:
/// `(json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, program, seed=1))]
fn graph_imputation_decision_consume(
    py: Python<'_>,
    data: &[u8],
    program: &str,
    seed: u64,
) -> PyResult<Refused<String>> {
    let program = crate::digest::digest(program, "program")?;
    let ctx = crate::execution_api::context(seed);
    let consumed = crate::detach_catch(py, move || Ok(consume(data, program, &ctx)))?;
    Ok(match consumed {
        Ok(text) => (Some(text), None),
        Err(refused) => (None, Some(refused)),
    })
}

// --------------------------------------------------------------------------------- bundles

fn receipt_json(receipt: &GraphImputationTransportNodeBundleReceipt) -> String {
    let dependencies: Vec<Json> = receipt
        .bundle
        .dependencies
        .iter()
        .map(|dependency| {
            json!({
                "node_id": dependency.node_id,
                "artifact_identity": dependency.artifact_identity,
                "chain_digest": dependency.chain_digest,
            })
        })
        .collect();
    json!({
        "bundle_identity": receipt.bundle.bundle_identity,
        "terminal_node": receipt.bundle.terminal_node,
        "terminal_identity": receipt.bundle.terminal_identity,
        "chain_program": hex(receipt.bundle.lowered_program_identity),
        "dependencies": dependencies,
        "producer_program": hex(receipt.producer_program_identity),
        "program": hex(receipt.lowered_program_identity),
    })
    .to_string()
}

/// Embed a transported graph-imputation fit as a bundle node; the fit is replayed from its
/// artifact before it is added. Returns the refusal, if any.
#[pyfunction]
#[pyo3(signature = (builder, node_id, data, program, seed=1))]
#[allow(clippy::needless_pass_by_value)]
fn graph_imputation_transport_bundle_add_fit(
    py: Python<'_>,
    mut builder: PyRefMut<'_, PyBundleBuilder>,
    node_id: &str,
    data: &[u8],
    program: &str,
    seed: u64,
) -> PyResult<Option<String>> {
    let program = crate::digest::digest(program, "program")?;
    let ctx = crate::execution_api::context(seed);
    let inner = &mut builder.inner;
    crate::detach_catch(py, move || {
        Ok(inner
            .add_graph_imputation_transport_fit(node_id, data, program, &ctx)
            .err()
            .map(|error| transport_bundle_refusal(&error)))
    })
}

/// A compiled bundle recomputation: a transported fit and the chain terminal it feeds.
#[pyclass(name = "GraphImputationTransportBundleHandle", skip_from_py_object)]
struct PyGraphImputationTransportBundle {
    inner: CompiledGraphImputationTransportNodeBundle,
    receipt: Option<GraphImputationTransportNodeBundleReceipt>,
    seed: u64,
}

#[pymethods]
impl PyGraphImputationTransportBundle {
    /// The compiled program identity (lowercase hex) a consumer retains.
    fn program(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Refit the embedded fit and replay its chain terminal: `(receipt json, refusal)`.
    fn execute(&mut self, py: Python<'_>) -> PyResult<Refused<String>> {
        let ctx = crate::execution_api::context(self.seed);
        let inner = &self.inner;
        let executed = crate::detach_catch(py, move || {
            Ok(inner.execute(&ctx).map_err(|error| transport_bundle_refusal(&error)))
        })?;
        Ok(match executed {
            Ok(receipt) => {
                let text = receipt_json(&receipt);
                self.receipt = Some(receipt);
                (Some(text), None)
            }
            Err(refused) => (None, Some(refused)),
        })
    }

    /// Replay the last receipt and seal the bundle with it: `(bytes, refusal)`.
    fn export(&self, py: Python<'_>) -> PyResult<Refused<Py<PyBytes>>> {
        let Some(receipt) = &self.receipt else {
            return Ok((
                None,
                Some(invalid(composition_bundle::INVALID_CONTAINER, "execute before exporting")),
            ));
        };
        let ctx = crate::execution_api::context(self.seed);
        let inner = &self.inner;
        let bytes = crate::detach_catch(py, move || {
            Ok(inner
                .export_portable(receipt, &ctx)
                .map_err(|error| transport_bundle_refusal(&error)))
        })?;
        Ok(match bytes {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(refused) => (None, Some(refused)),
        })
    }
}

/// Compile the bundle recomputation of a transported fit node and its direct chain terminal:
/// `(handle, refusal)`.
#[pyfunction]
#[pyo3(signature = (bundle, fit, terminal, seed=1))]
#[allow(clippy::needless_pass_by_value)]
fn graph_imputation_transport_bundle_compile(
    py: Python<'_>,
    bundle: PyRef<'_, PyCompositionBundle>,
    fit: &str,
    terminal: &str,
    seed: u64,
) -> PyResult<Refused<PyGraphImputationTransportBundle>> {
    let bundle = bundle.bundle.clone();
    let ctx = crate::execution_api::context(seed);
    let compiled = crate::detach_catch(py, move || {
        Ok(compile_graph_imputation_transport_node_bundle_recomputation(
            bundle, fit, terminal, &ctx,
        )
        .map_err(|error| transport_bundle_refusal(&error)))
    })?;
    Ok(match compiled {
        Ok(inner) => (Some(PyGraphImputationTransportBundle { inner, receipt: None, seed }), None),
        Err(refused) => (None, Some(refused)),
    })
}

/// Recompile and replay a portable bundle under the retained bundle and program identities:
/// `(receipt json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, bundle_identity, program, seed=1))]
fn graph_imputation_transport_bundle_consume(
    py: Python<'_>,
    data: &[u8],
    bundle_identity: &str,
    program: &str,
    seed: u64,
) -> PyResult<Refused<String>> {
    let program = crate::digest::digest(program, "program")?;
    let ctx = crate::execution_api::context(seed);
    let consumed = crate::detach_catch(py, move || {
        Ok(consume_portable_graph_imputation_transport_node_bundle_recomputation(
            data,
            bundle_identity,
            program,
            &ctx,
        )
        .map(|receipt| receipt_json(&receipt))
        .map_err(|error| transport_bundle_refusal(&error)))
    })?;
    Ok(match consumed {
        Ok(text) => (Some(text), None),
        Err(refused) => (None, Some(refused)),
    })
}

/// Consume a composition bundle, refitting every embedded graph-imputation transport fit:
/// `(consumed json, refusal)`.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, supplied_json=None, seed=1))]
fn graph_imputation_fit_bundle_consume(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    supplied_json: Option<&str>,
    seed: u64,
) -> PyResult<Refused<String>> {
    let supplied = crate::composition_bundle_api::supplied_from(supplied_json)?;
    let ctx = crate::execution_api::context(seed);
    let consumed = crate::detach_catch(py, move || {
        Ok(graph_imputation_transport_bundle_consumer(&ctx)
            .consume(data, &BundleLimits::default(), expected_identity, &supplied)
            .map(|consumed| crate::composition_bundle_api::consumed_json(&consumed))
            .map_err(|error| crate::composition_bundle_api::error_refusal(&error)))
    })?;
    Ok(match consumed {
        Ok(text) => (Some(text), None),
        Err(refused) => (None, Some(refused)),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGraphImputationDecision>()?;
    module.add_class::<PyGraphImputationTransportBundle>()?;
    module.add_function(wrap_pyfunction!(graph_imputation_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(graph_imputation_decision_consume, module)?)?;
    module.add_function(wrap_pyfunction!(graph_imputation_transport_bundle_add_fit, module)?)?;
    module.add_function(wrap_pyfunction!(graph_imputation_transport_bundle_compile, module)?)?;
    module.add_function(wrap_pyfunction!(graph_imputation_transport_bundle_consume, module)?)?;
    module.add_function(wrap_pyfunction!(graph_imputation_fit_bundle_consume, module)?)?;
    Ok(())
}
