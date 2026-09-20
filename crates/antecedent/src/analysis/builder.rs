//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Builder types.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, PopulationRegistry, ResponseQuery,
    TemporalEffectQuery, VariableId,
};
use antecedent_data::{
    DiscoveryEstimationSplit, EventData, MultiEnvironmentData, NetworkData, PanelData, TableView,
    TabularData, TimeSeriesData,
};
use antecedent_discovery::GraphPosterior;
use antecedent_estimate::{ContinuousResponseOptions, OverlapPolicy};
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalCpdag, TemporalDag, TemporalPag,
};
use antecedent_validate::CustomEffectValidator;

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{EstimatorId, IdentifierId};

use super::execute::Study;
use super::latency::{ComputeBudget, LatencyMode, ResolvedLatencyBudget, refuse_non_report_hmc};

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Cheap interactive validators: overlap + E-value only.
    Cheap,
    /// Placebo + random common cause (linear backdoor only).
    PlaceboAndRcc,
    /// Full validation suite (applicable validators only; others NotApplicable).
    Full,
}

impl RefuteSuite {
    /// Wire id recorded on the logical plan (`None` when validation is skipped).
    #[must_use]
    pub const fn validation_suite_id(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Cheap => Some("overlap+evalue"),
            Self::PlaceboAndRcc => Some("placebo+rcc"),
            Self::Full => Some("validation.full"),
        }
    }

    /// Label for second-click refute diagnostics (`none` when the suite is skipped).
    #[must_use]
    pub const fn diagnostic_label(self) -> &'static str {
        match self.validation_suite_id() {
            Some(id) => id,
            None => "none",
        }
    }
}

/// Trial membership and known probabilities for a licensed [`CausalQuery::Transport`] cell.
#[derive(Clone, Debug)]
pub struct TransportTrialSpec {
    /// Source-trial membership (`true` = source experiment row).
    pub trial: VariableId,
    /// P(S=1 | X) on every row, strictly inside (0, 1).
    pub selection_probability: VariableId,
    /// P(A=1 | X, S=1) on trial rows; unread on target rows.
    pub treatment_probability: VariableId,
}

/// Fixed network and realized assignment for a licensed [`CausalQuery::Interference`] cell.
#[derive(Clone, Debug)]
pub struct InterferenceSpec {
    /// Unit table plus incoming exposure edges.
    pub network: NetworkData,
    /// Realized binary assignment in unit-row order.
    pub assignment: Arc<[bool]>,
}

impl InterferenceSpec {
    /// The same fixed network and realized assignment over `units`.
    ///
    /// The network's edges and the assignment are the design; the unit table is
    /// the data. A study binds its network to the unit table it executes, so a
    /// refresh or an estimate click on new outcomes executes on those outcomes
    /// and its data snapshot names them.
    ///
    /// # Errors
    ///
    /// The edges or the assignment do not fit `units`.
    pub(crate) fn bound_to(&self, units: &TabularData) -> Result<Self, CausalError> {
        if self.network.units().storage().content_digest() == units.storage().content_digest() {
            return Ok(self.clone());
        }
        if self.assignment.len() != units.row_count() {
            return Err(CausalError::Compile {
                message: "interference network, assignment, and unit table row counts must match"
                    .into(),
            });
        }
        Ok(Self {
            network: NetworkData::try_new(units.clone(), self.network.edges().to_vec())?,
            assignment: Arc::clone(&self.assignment),
        })
    }
}

/// Refuse a transport construction outside the licensed cell.
///
/// The licensed `TransportQuery` cell transports a mean `ResponseCurve` on the
/// complete, all-observed, static response by binary trial-to-target IPW. A
/// derivative, a non-mean outcome functional, an embedded population, an
/// observation mechanism or a temporal attachment would be labelled with the
/// executed binary contrast it is not, so it is refused.
fn refuse_unlicensed_transport(query: &antecedent_core::TransportQuery) -> Result<(), CausalError> {
    let response = &query.response;
    let licensed =
        matches!(response.functional, antecedent_core::ResponseFunctional::MeanCurve { .. })
            && response.outcome_functional.is_mean()
            && matches!(response.target_population, antecedent_core::TargetPopulation::AllObserved)
            && matches!(response.observation, antecedent_core::ObservationSpec::Complete)
            && response.observation_assumptions.is_empty()
            && response.temporal.is_none();
    if licensed {
        Ok(())
    } else {
        Err(crate::support_reason!(
            "construction_not_licensed",
            "TransportQuery is licensed for a mean ResponseCurve on the complete, all-observed \
             static response, transported by binary trial-to-target IPW"
        ))
    }
}

/// Refuse an interference design outside the licensed cell (NeighborCount
/// exposure under Bernoulli assignment).
fn refuse_unlicensed_interference(
    query: &antecedent_core::InterferenceQuery,
) -> Result<(), CausalError> {
    let licensed = matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
        && matches!(query.exposure, antecedent_core::ExposureMapping::NeighborCount);
    if licensed {
        Ok(())
    } else {
        Err(crate::support_reason!(
            "construction_not_licensed",
            "InterferenceQuery is licensed for NeighborCount exposure under Bernoulli assignment"
        ))
    }
}

/// Refusal for panel response on a non-temporal or static graph.
pub(crate) const PANEL_RESPONSE_CLASS_REFUSAL: &str = concat!(
    "panel ResponseCurve / InterventionResponse is licensed on a supplied ",
    "TemporalDag, TemporalCpdag, or TemporalPag",
);

/// Refusal for a panel class multi-step Sustained run under a discovery split.
pub(crate) const PANEL_CLASS_SEQUENTIAL_SPLIT_REFUSAL: &str =
    "class-aware multi-step sustained requires no discovery-estimation split";

/// Refusal for a transferred or informative prior on panel class multi-step Sustained.
pub(crate) const PANEL_CLASS_SEQUENTIAL_PRIOR_REFUSAL: &str =
    "multi-step Sequence transfer stays refused on incomplete classes";

/// Single owner for panel data-route licenses. Build, prepare, compile, and
/// execute consult this; they do not restate the same refusals.
///
/// Every panel route requires one sampling regularity across units: a horizon of
/// `h` steps must mean the same duration in every unit.
pub(crate) fn refuse_unlicensed_panel_route(
    query: &CausalQuery,
    class: GraphClass,
    inference: &InferenceMode,
    panel: &PanelData,
    split: Option<&DiscoveryEstimationSplit>,
) -> Result<(), CausalError> {
    panel_shared_regularity(panel)?;
    match query {
        CausalQuery::Response(query) => refuse_unlicensed_panel_response(query, class),
        CausalQuery::TemporalEffect(query) => {
            refuse_unlicensed_panel_effect(query, class, inference, split)
        }
        _ => Ok(()),
    }
}

/// The one time-index regularity every panel unit shares.
///
/// # Errors
///
/// When the panel is empty or its units disagree on regularity.
pub(crate) fn panel_shared_regularity(
    panel: &PanelData,
) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first = &panel
        .unit(0)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?
        .series
        .time_index()
        .regularity;
    if panel.units().iter().any(|unit| &unit.series.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "panel analysis requires every panel unit to share one time-index \
                      regularity: a horizon step would otherwise mean a different duration in \
                      different units; align the units first"
                .into(),
        });
    }
    Ok(first.clone())
}

fn refuse_unlicensed_panel_response(
    query: &ResponseQuery,
    class: GraphClass,
) -> Result<(), CausalError> {
    if !query.is_temporal()
        || !matches!(
            class,
            GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
        )
    {
        return Err(CausalError::Unsupported { message: PANEL_RESPONSE_CLASS_REFUSAL });
    }
    Ok(())
}

fn refuse_unlicensed_panel_effect(
    query: &TemporalEffectQuery,
    class: GraphClass,
    inference: &InferenceMode,
    split: Option<&DiscoveryEstimationSplit>,
) -> Result<(), CausalError> {
    if !class.is_incomplete_temporal() || !query.is_multi_step_sustained() {
        return Ok(());
    }
    // Panel class multi-step Sustained fits every unit's full series; a split's
    // discovery rows would be reused for estimation, and the per-unit sequential
    // g-computation has no mapping for a transferred coefficient prior.
    if split.is_some() {
        return Err(CausalError::Unsupported { message: PANEL_CLASS_SEQUENTIAL_SPLIT_REFUSAL });
    }
    if let InferenceMode::Bayesian(cfg) = inference {
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            return Err(CausalError::Unsupported { message: PANEL_CLASS_SEQUENTIAL_PRIOR_REFUSAL });
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) enum DataInput {
    Tabular(TabularData),
    Temporal(TimeSeriesData),
    /// Event data aligned onto a regular duration grid (stored as series).
    Event(TimeSeriesData),
    /// Multi-environment series.
    MultiEnv(MultiEnvironmentData),
    /// Multi-unit panel (stacked cluster-HAC estimate).
    Panel(PanelData),
}

/// Running-variable configuration for the `rd.sharp` estimator; required when `rd.sharp` is
/// selected as the estimator (see [`StudyBuilder::rd_config`]).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RdConfig {
    /// Running (assignment) variable.
    pub running_variable: VariableId,
    /// Discontinuity cutoff.
    pub cutoff: f64,
    /// Symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is retained).
    pub bandwidth: f64,
    /// Analytic SE kind for the jump coefficient. `Hc1` (default) and `Hc0`/`Hc2`/`Hc3`
    /// use the heteroskedasticity-robust residual sandwich; `Homoskedastic` is an
    /// explicit opt-in that assumes a constant outcome variance inside the window.
    /// Label-based kinds are refused at execute.
    pub se_kind: antecedent_estimate::AnalyticSeKind,
}

impl RdConfig {
    /// Construct an RD design configuration (HC1 analytic SE).
    #[must_use]
    pub const fn new(running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        Self {
            running_variable,
            cutoff,
            bandwidth,
            se_kind: antecedent_estimate::AnalyticSeKind::Hc1,
        }
    }

    /// Select the analytic SE kind (e.g. [`antecedent_estimate::AnalyticSeKind::Homoskedastic`]
    /// to opt into the classical constant-variance formula).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: antecedent_estimate::AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }
}

/// Placeholder structure for [`Study::graph`][super::execute::Study] when a graph
/// posterior drives the analysis instead.
///
/// This carries only the shape (variable count / modality) needed for logical-plan
/// bookkeeping (row counts, data classification, …). It is never consulted for
/// identification — identification runs per-graph, against the real posterior atoms,
/// inside `execute()`. `n_vars` comes from the supplied [`GraphPosterior`], not from
/// re-inspecting `data`, so it always matches the ensemble the caller discovered.
fn stub_accepted_graph_for(
    data: &DataInput,
    n_vars: usize,
    atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
) -> Result<AcceptedGraph, CausalError> {
    match data {
        DataInput::Tabular(_) => {
            let n = u32::try_from(n_vars).map_err(|_| CausalError::Compile {
                message: "too many variables for graph-posterior stub graph".into(),
            })?;
            match atom_kind {
                antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                    Ok(AcceptedGraph::dag(Dag::with_variables(n)))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                    AcceptedGraph::cpdag(Cpdag::with_variables(n))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    Ok(AcceptedGraph::pag(Pag::with_variables(n)))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Admg => {
                    let mut admg = Admg::with_variables(n);
                    if n >= 2 {
                        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                    }
                    Ok(AcceptedGraph::admg(admg))
                }
                _ => Err(CausalError::Unsupported {
                    message: "graph-posterior stub supports only static Dag, Cpdag, Pag, or Admg atoms",
                }),
            }
        }
        DataInput::Temporal(_) | DataInput::Event(_) => match atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                Ok(AcceptedGraph::temporal_dag(TemporalDag::empty()))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                AcceptedGraph::temporal_cpdag(TemporalCpdag::empty())
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                Ok(AcceptedGraph::temporal_pag(TemporalPag::empty()))
            }
            _ => Err(CausalError::Unsupported {
                message: "graph-posterior stub supports only TemporalDag, TemporalCpdag, or \
                          TemporalPag atoms on temporal/event data",
            }),
        },
        DataInput::MultiEnv(_) | DataInput::Panel(_) => Err(CausalError::Unsupported {
            message: "graph-posterior analysis supports tabular or temporal/event data only",
        }),
    }
}

/// Marker: the study structure came from [`StudyBuilder::graph`].
#[derive(Clone, Copy, Debug)]
struct CallerGraph;

/// A caller graph and a tier background both describe the study structure.
fn tiered_graph_conflict() -> CausalError {
    CausalError::Conflict {
        what: "graph",
        detail: "both .graph(..) and .tiered_background(..) were set; the tier background \
                 materializes its own closure ADMG / PAG, so supply exactly one structure input",
    }
}

/// Borrow the schema backing `data`, regardless of modality.
fn data_schema(data: &DataInput) -> &CausalSchema {
    match data {
        DataInput::Tabular(d) => d.schema(),
        DataInput::Temporal(d) | DataInput::Event(d) => d.schema(),
        DataInput::MultiEnv(d) => d.schema(),
        DataInput::Panel(d) => d.schema(),
    }
}

/// Node count of `graph`, for classes where a node is one variable.
///
/// `None` for the temporal classes: their nodes are (variable, lag) pairs, so
/// `node_count()` is a multiple of the variable count rather than equal to it, and
/// there is no accessor here that recovers the lag depth honestly. Static classes
/// (`Dag`, `Admg`, `Cpdag`, `Pag`) are positional — node `i` *is* variable `i` — so
/// their node count is directly comparable to a schema's variable count.
fn static_node_count(graph: &AcceptedGraph) -> Option<usize> {
    match graph.class() {
        GraphClass::Dag => graph.as_dag().map(Dag::node_count),
        GraphClass::Admg => graph.as_admg().map(Admg::node_count),
        GraphClass::Cpdag => graph.as_cpdag().map(Cpdag::node_count),
        GraphClass::Pag => graph.as_pag().map(Pag::node_count),
        GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag => None,
    }
}

/// Refuse a `graph` whose node indices cannot possibly describe `schema`.
///
/// Static graph nodes are positional (`DenseNodeId(i)` is `VariableId(i)`) with no
/// stored record of which schema those indices meant — a structure built against one
/// schema is silently meaningless against another with the same shape. Two
/// independent checks guard against that:
///
/// - **Shape** (always, static classes only): the graph's node count must equal the
///   number of variables in `schema`. Temporal classes are exempt (see
///   [`static_node_count`]).
/// - **Names** (only when the graph was bound via [`AcceptedGraph::with_schema`]):
///   the bound variable names must match `schema`'s names, in order. This check
///   applies to every class, temporal included, because the bound name list is
///   variable-level, not node-level — comparing it needs no lag arithmetic.
///
/// # Errors
///
/// [`CausalError::SchemaMismatch`] on either disagreement.
fn validate_schema_binding(
    graph: &AcceptedGraph,
    schema: &CausalSchema,
) -> Result<(), CausalError> {
    if let Some(node_count) = static_node_count(graph) {
        let n_vars = schema.len();
        if node_count != n_vars {
            return Err(CausalError::SchemaMismatch {
                detail: format!(
                    "graph has {node_count} nodes but data has {n_vars} variables; the \
                     structure does not describe this table"
                ),
            });
        }
    }

    if let Some(names) = graph.variable_names() {
        let data_vars = schema.variables();
        if names.len() != data_vars.len() {
            return Err(CausalError::SchemaMismatch {
                detail: format!(
                    "graph is bound to {} variables but data has {} variables; the structure \
                     does not describe this table",
                    names.len(),
                    data_vars.len()
                ),
            });
        }
        for (i, (bound_name, var)) in names.iter().zip(data_vars.iter()).enumerate() {
            if bound_name.as_ref() != var.name.as_ref() {
                return Err(CausalError::SchemaMismatch {
                    detail: format!(
                        "graph is bound to variable {i} `{bound_name}` but data has `{}` at \
                         that position; the structure was built against a different schema",
                        var.name
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Builder for static or temporal analysis.
#[derive(Clone)]
pub struct StudyBuilder {
    data: DataInput,
    graph: Option<AcceptedGraph>,
    /// Alternative to [`Self::graph`]: a posterior over structures rather than one
    /// accepted structure. Mutually exclusive with `graph` (checked at [`Self::build`]).
    graph_posterior: Option<GraphPosterior>,
    /// Caller-supplied mass over incomplete-temporal class members.
    class_prior: Option<crate::ClassPrior>,
    /// Optional cap on TemporalCpdag / TemporalPag completion search.
    max_completions: Option<usize>,
    /// Set by [`Self::graph`]: `Accepted` vs `Explicit`. Posterior overrides at build.
    structure_source: Option<crate::support::StructureSource>,
    query: Option<CausalQuery>,
    refute: RefuteSuite,
    /// Whether [`Self::refute`] was set explicitly (wins over latency mode).
    refute_explicit: bool,
    bootstrap_replicates: u32,
    /// Whether [`Self::bootstrap_replicates`] was set explicitly.
    bootstrap_explicit: bool,
    split: Option<DiscoveryEstimationSplit>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    /// Caller-configured estimator (superset of [`Self::estimator`]); `Some` only when
    /// [`StudyBuilder::estimator`] was called with a configured estimator rather
    /// than a bare [`EstimatorId`].
    estimator_spec: Option<EstimatorSpec>,
    /// Numerical/inference options for response-family estimators.
    response_options: Option<ContinuousResponseOptions>,
    observation_options: antecedent_estimate::ObservationEstimatorOptions,
    observation_delayed_entry: Option<antecedent_core::VariableId>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
    /// Whether Bayesian `n_draws` were set via [`ComputeBudget`] (mode draw map skipped).
    n_draws_explicit: bool,
    /// Optional override for propensity / AIPW overlap (clip/trim). `None` keeps estimator defaults.
    overlap_policy: Option<OverlapPolicy>,
    /// Optional bindings for named predicates / custom target distributions.
    population_registry: Option<PopulationRegistry>,
    /// Custom slow-path validators appended after the built-in refute suite.
    custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
    /// Optional tier-rule background (fast-path generalized adjustment).
    tiered: Option<antecedent_graph::TieredBackground>,
    /// Set when [`Self::graph`] is called. A caller graph and a tier background
    /// are two sources of truth for one structure; build refuses the pair.
    caller_graph: Option<CallerGraph>,
    /// Refused at build: coarsened continuous coordinate is not a point CDE.
    continuous_cell: Option<(antecedent_core::VariableId, std::sync::Arc<[f64]>)>,
    /// Optional latency tier (maps to known-equivalent budgets unless overridden).
    latency_mode: Option<LatencyMode>,
    /// Optional field-level compute budget overrides.
    compute_budget: ComputeBudget,
    /// Optional progressive stage-result sink (Identify → Point → Uncertainty → Validate).
    stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
    /// Selection-node targets for a transport query (empty = Direct).
    selection_targets: Option<Arc<[VariableId]>>,
    /// Trial columns for a transport query.
    transport_trial: Option<TransportTrialSpec>,
    /// Network + assignment for an interference query.
    interference: Option<InterferenceSpec>,
}

impl std::fmt::Debug for StudyBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StudyBuilder")
            .field("continuous_cell", &self.continuous_cell)
            .field("data", &"<data>")
            .field("graph", &self.graph)
            .field("tiered", &self.tiered)
            .field("caller_graph", &self.caller_graph)
            .field("graph_posterior", &self.graph_posterior)
            .field("class_prior", &self.class_prior)
            .field("max_completions", &self.max_completions)
            .field("structure_source", &self.structure_source)
            .field("query", &self.query.as_ref().map(|_| "<query>"))
            .field("refute", &self.refute)
            .field("refute_explicit", &self.refute_explicit)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("bootstrap_explicit", &self.bootstrap_explicit)
            .field("split", &self.split)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("estimator_spec", &self.estimator_spec)
            .field("response_options", &self.response_options)
            .field("observation_options", &self.observation_options)
            .field("observation_delayed_entry", &self.observation_delayed_entry)
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("n_draws_explicit", &self.n_draws_explicit)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("custom_validators", &self.custom_validators.len())
            .field("latency_mode", &self.latency_mode)
            .field("compute_budget", &self.compute_budget)
            .field("stage_sink_is_some", &self.stage_sink.is_some())
            .field("selection_targets", &self.selection_targets)
            .field("transport_trial", &self.transport_trial)
            .field("interference", &self.interference)
            .finish()
    }
}

/// Whether an omitted replicate count is a real resampling budget for this route.
///
/// Static response surfaces carry analytic or influence-function uncertainty,
/// Bayesian responses carry posterior intervals, and counterfactual unit
/// effects have no sampling uncertainty. Reporting [`StudyBuilder::OMITTED_BOOTSTRAP`]
/// replicates there would describe a budget that never runs, so the omitted
/// count is zero. An explicit count is never rewritten here.
/// Refuse an estimator that does not implement the requested inference mode.
///
/// Bayesian-only estimators never run under Frequentist inference, and the
/// Frequentist-only estimators below never run under Bayesian inference. The
/// remaining identifier-native estimators (`functional.*`, `mediation.linear`,
/// `gcm.fit`, the temporal and derivative estimators) apply the requested
/// mode at execution, and the query-specific pairings are checked where the
/// query's expected estimator is resolved. Without this refusal a Frequentist
/// estimator would run under a Bayesian request and be bound to the Bayesian
/// license coordinate and inference binding.
fn refuse_estimator_inference_mismatch(
    query: &CausalQuery,
    estimator: EstimatorId,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    const _: () = assert!(
        crate::error::is_runtime_refusal_code("estimator_inference_mismatch"),
        "`estimator_inference_mismatch` is not a runtime_refusal code"
    );
    // The refusal removes a wrong label, not a capability: the estimator stays
    // available under the inference mode it implements, and the message names
    // both ways forward.
    macro_rules! frequentist {
        ($name:literal) => {
            Err(CausalError::Unsupported {
                message: concat!(
                    "reason",
                    "=estimator_inference_mismatch: estimator ",
                    $name,
                    " is Frequentist, so it does not run under inference=Bayesian (it would \
                     report a sampling interval under a Bayesian label). Omit estimator= to use \
                     the Bayesian estimator, or use inference=Frequentist to run ",
                    $name
                ),
            })
        };
    }
    macro_rules! bayesian {
        ($name:literal) => {
            Err(CausalError::Unsupported {
                message: concat!(
                    "reason",
                    "=estimator_inference_mismatch: estimator ",
                    $name,
                    " is Bayesian, so it does not run under inference=Frequentist. Omit \
                     estimator= to use the Frequentist estimator, or use inference=Bayesian to \
                     run ",
                    $name
                ),
            })
        };
    }
    match inference {
        InferenceMode::Frequentist => match estimator {
            EstimatorId::BayesianGcomp => bayesian!("bayesian.gcomp"),
            EstimatorId::BayesianConditional => bayesian!("conditional.bayesian"),
            EstimatorId::BayesianTemporalGcomp => bayesian!("bayesian.temporal.gcomp"),
            EstimatorId::TemporalResponseBayesian => bayesian!("response.temporal.bayesian"),
            EstimatorId::ResponseBayesian => bayesian!("response.bayesian"),
            EstimatorId::BayesianTemporalMediation => bayesian!("temporal.mediation.bayesian"),
            _ => Ok(()),
        },
        InferenceMode::Bayesian(_) => {
            let average = matches!(query, CausalQuery::AverageEffect(_));
            match estimator {
                EstimatorId::LinearAdjustmentAte if average => {
                    frequentist!("linear.adjustment.ate")
                }
                EstimatorId::PropensityWeighting if average => frequentist!("propensity.weighting"),
                EstimatorId::PropensityMatching if average => frequentist!("propensity.matching"),
                EstimatorId::PropensityStratification if average => {
                    frequentist!("propensity.stratification")
                }
                EstimatorId::DistanceMatching if average => frequentist!("distance.matching"),
                EstimatorId::Aipw if average => frequentist!("aipw"),
                EstimatorId::Dml if average => frequentist!("dml"),
                EstimatorId::DrLearner if average => frequentist!("dr.learner"),
                EstimatorId::CausalForest if average => frequentist!("causal.forest"),
                EstimatorId::GlmAdjustment if average => frequentist!("glm.adjustment"),
                EstimatorId::FrontDoorTwoStage if average => frequentist!("frontdoor.two_stage"),
                EstimatorId::IvWald if average => frequentist!("iv.wald"),
                EstimatorId::Iv2Sls if average => frequentist!("iv.2sls"),
                EstimatorId::RdSharp if average => frequentist!("rd.sharp"),
                EstimatorId::ConditionalLinearAdjustment => {
                    frequentist!("conditional.linear.adjustment")
                }
                EstimatorId::CellAipw => frequentist!("cell.aipw"),
                EstimatorId::TransportTrialIpw => frequentist!("transport.trial_ipw"),
                EstimatorId::InterferenceHtHajek => frequentist!("interference.ht_hajek"),
                _ => Ok(()),
            }
        }
    }
}

fn omitted_bootstrap_resamples(query: &CausalQuery, inference: &InferenceMode) -> bool {
    match query {
        CausalQuery::Response(q) => {
            q.is_temporal() && matches!(inference, InferenceMode::Frequentist)
        }
        CausalQuery::Counterfactual(_) => false,
        _ => true,
    }
}

/// Whether an executor can estimate `query` in its declared target population.
///
/// The one owner of which query kinds take a population other than
/// [`antecedent_core::TargetPopulation::AllObserved`] at build time: only an
/// [`CausalQuery::AverageEffect`], whose estimators then accept or refuse the
/// specific target (ATT/ATC, predicate, custom distribution) themselves. Every
/// other population-scoped kind (temporal effects, mediation, interventional
/// distributions, path-specific effects, responses and derivatives, conditional
/// effects) has no estimator for another target, so a declared population is
/// refused here rather than dropped by a route that ignores it. A row-weight
/// target is produced by a retarget of frozen scores, never estimated from a
/// build.
fn population_estimable(query: &CausalQuery) -> bool {
    match query.target_population() {
        None | Some(antecedent_core::TargetPopulation::AllObserved) => true,
        Some(_) => matches!(query, CausalQuery::AverageEffect(_)),
    }
}

/// Refuse a non-Gaussian Bayesian likelihood where no executor fits it.
///
/// The one owner of which routes honour [`crate::BayesianConfig::likelihood`].
/// Bayesian g-computation of a tabular [`CausalQuery::AverageEffect`] mean on
/// one explicit or accepted [`GraphClass::Dag`] fits the declared Bernoulli or
/// Poisson GLM and averages the inverse link over the rows; that construction
/// has repeated-sampling coverage records. Every other Bayesian route
/// (conditional effects, mediation, temporal effects and responses, continuous
/// responses, panels, ADMG front door, class envelopes, tiered backgrounds and
/// graph-posterior mixtures) is either a Gaussian identity-link model or has no
/// coverage measurement under another link, the conjugate backend is Gaussian by
/// construction, and a transferred prior is mapped on the identity-link
/// coefficient scale. Those combinations refuse here rather than silently fit
/// a Gaussian model or report an unmeasured construction.
fn refuse_unsupported_likelihood(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    if cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity {
        return Ok(());
    }
    if cfg.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
        return Err(crate::unsupported_reason!(
            "likelihood_not_supported",
            "the conjugate backend fits a Gaussian identity-link model only; use the laplace or \
             hmc backend for a Bernoulli or Poisson likelihood"
        ));
    }
    if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
        return Err(crate::unsupported_reason!(
            "likelihood_not_supported",
            "a transferred prior is mapped onto identity-link coefficients; a Bernoulli or \
             Poisson likelihood fits under the isotropic prior_scale only"
        ));
    }
    let licensed = matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && matches!(
            query,
            CausalQuery::AverageEffect(q)
                if matches!(q.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        );
    if licensed {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "likelihood_not_supported",
        "a Bernoulli or Poisson likelihood is fitted by Bayesian g-computation of a tabular \
         AverageEffect mean on one Dag only; this route fits a Gaussian identity-link model or \
         mixes structures, so it refuses the likelihood rather than fit a different model than \
         the one declared"
    ))
}

/// Whether the executors for `query` run caller custom validators.
///
/// A custom validator refutes one scalar average effect. Function-valued
/// responses, distributions, path-specific and mediation contrasts,
/// counterfactual unit effects, and attribution have no such refutation
/// problem, panel class completions are mixed without a per-completion scalar
/// refuter, and a Bayesian graph-posterior response level is not refuted.
/// Supplying validators there is refused rather than skipped.
fn custom_validators_apply(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    graph_posterior: bool,
    inference: &InferenceMode,
) -> bool {
    match query {
        CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_) => true,
        CausalQuery::TemporalEffect(_) => {
            !(matches!(data, DataInput::Panel(_)) && class.is_incomplete_temporal())
        }
        CausalQuery::Response(q) => {
            !q.is_temporal()
                && matches!(class, GraphClass::Dag | GraphClass::Admg)
                && matches!(
                    q.functional,
                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                )
                && !(graph_posterior && matches!(inference, InferenceMode::Bayesian(_)))
        }
        _ => false,
    }
}

impl StudyBuilder {
    /// Replicate count used when the caller omits [`Self::bootstrap_replicates`]
    /// on a route that resamples. Language bindings read this value instead of
    /// keeping their own copy.
    ///
    /// The omitted tier *is* the Standard tier, so this is the Standard tier's
    /// replicate count by construction rather than a second copy of `199`.
    pub const OMITTED_BOOTSTRAP: u32 = super::latency::STANDARD_BOOTSTRAP;
    /// Validation suite used when the caller omits [`Self::refute`]. A cell
    /// that does not license this suite is downgraded at build.
    pub const OMITTED_REFUTE: RefuteSuite = RefuteSuite::PlaceboAndRcc;

    fn from_data(data: DataInput) -> Self {
        Self {
            data,
            graph: None,
            graph_posterior: None,
            class_prior: None,
            max_completions: None,
            structure_source: None,
            query: None,
            refute: Self::OMITTED_REFUTE,
            refute_explicit: false,
            bootstrap_replicates: Self::OMITTED_BOOTSTRAP,
            bootstrap_explicit: false,
            split: None,
            identifier: None,
            estimator: None,
            estimator_spec: None,
            response_options: None,
            observation_options: antecedent_estimate::ObservationEstimatorOptions::default(),
            observation_delayed_entry: None,
            rd: None,
            inference: InferenceMode::Frequentist,
            n_draws_explicit: false,
            overlap_policy: None,
            population_registry: None,
            custom_validators: Vec::new(),
            tiered: None,
            caller_graph: None,
            continuous_cell: None,
            latency_mode: None,
            compute_budget: ComputeBudget::new(),
            stage_sink: None,
            selection_targets: None,
            transport_trial: None,
            interference: None,
        }
    }

    /// Capability report without granting a license.
    ///
    /// Missing graph or query is a binding report with no neighbors. Present
    /// coordinates classify through [`Self::inspect`] and do not identify.
    ///
    /// # Errors
    ///
    /// Schema mismatch or canonical-encoding failures after graph and query are
    /// supplied.
    pub fn capability(&self) -> Result<super::OperationReport, CausalError> {
        if self.graph.is_none() && self.graph_posterior.is_none() {
            return Ok(super::OperationReport::binding_missing(
                super::OperationKind::Inspect,
                "graph",
            ));
        }
        if self.query.is_none() {
            return Ok(super::OperationReport::binding_missing(
                super::OperationKind::Inspect,
                "query",
            ));
        }
        Ok(self.clone().inspect()?.capability())
    }

    /// Cheap structural inspection without requiring a licensed cell.
    ///
    /// Closed, n/a, and refused coordinates still produce a contract: support
    /// status is recorded and identification stays unavailable. This does not
    /// authorize [`Study::prepare`] or execution — those still go through
    /// [`Self::build`].
    ///
    /// # Errors
    ///
    /// Missing graph / query, schema mismatch, or canonical-encoding failures.
    pub fn inspect(self) -> Result<super::CausalContract, CausalError> {
        self.finish(true)?.inspect()
    }

    /// Supply the causal structure.
    ///
    /// An [`AcceptedGraph`] is matrix-axis `accepted`. A bare
    /// [`antecedent_graph::Dag`], [`antecedent_graph::Admg`],
    /// [`antecedent_graph::Pag`], [`antecedent_graph::Cpdag`], or
    /// [`antecedent_graph::TemporalDag`] is `explicit`. A
    /// [`antecedent_graph::TemporalCpdag`] is not [`crate::IntoGraphInput`]
    /// until the class-preserving temporal identifier lands — use the fallible
    /// [`AcceptedGraph::temporal_cpdag`] first. [`antecedent_graph::TemporalPag`]
    /// is the same: use [`AcceptedGraph::temporal_pag`].
    ///
    /// Mutually exclusive with [`Self::tiered_background`], which materializes
    /// its own structure: setting both, in either order, is refused
    /// ([`CausalError::Conflict`]).
    #[must_use]
    pub fn graph(mut self, structure: impl crate::IntoGraphInput) -> Self {
        let (graph, source) = structure.into_graph_input();
        self.graph = Some(graph);
        self.structure_source = Some(source);
        self.caller_graph = Some(CallerGraph);
        self
    }

    /// Declare a tier order over existing ADMG / PAG semantics.
    ///
    /// CoDetermined certifies the tier-closure set in `O(p)` as
    /// `generalized.adjustment`. Unknown keeps two canonical sets as an
    /// envelope. Materializes the background as the study graph.
    ///
    /// Mutually exclusive with [`Self::graph`]: the background *is* the study
    /// structure, so a caller graph set before or after it is refused rather
    /// than silently ignored by the tier route.
    ///
    /// # Errors
    ///
    /// Missing tabular schema, invalid tier geometry, or
    /// [`CausalError::Conflict`] when [`Self::graph`] was already called.
    pub fn tiered_background(
        mut self,
        background: antecedent_graph::TieredBackground,
    ) -> Result<Self, CausalError> {
        if self.caller_graph.is_some() {
            return Err(tiered_graph_conflict());
        }
        let schema = match &self.data {
            DataInput::Tabular(data) => data.schema().clone(),
            DataInput::Temporal(data) | DataInput::Event(data) => data.schema().clone(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "TieredBackground requires tabular or series data",
                });
            }
        };
        let graph = match background.within_tier {
            antecedent_graph::WithinTier::CoDetermined => {
                crate::AcceptedGraph::from(background.to_admg(&schema)?)
            }
            antecedent_graph::WithinTier::Unknown => {
                crate::AcceptedGraph::from(background.to_pag(&schema)?)
            }
        };
        self.graph = Some(graph);
        self.structure_source = Some(crate::support::StructureSource::Explicit);
        self.tiered = Some(background);
        Ok(self)
    }

    /// Supply a posterior over graph structures instead of a single accepted graph.
    ///
    /// The effect is estimated per graph and combined into an envelope; unidentified
    /// posterior mass is retained on the result rather than being renormalised away.
    /// Mutually exclusive with [`StudyBuilder::graph`] — setting both is refused at
    /// [`Self::build`] time ([`CausalError::Conflict`]). Requires
    /// [`crate::inference::InferenceMode::Bayesian`] (checked when the analysis is
    /// compiled, not here): a graph posterior is a mixture over structures, and only
    /// Bayesian inference can combine per-graph effect draws into an envelope.
    #[must_use]
    pub fn graph_posterior(mut self, posterior: GraphPosterior) -> Self {
        self.graph_posterior = Some(posterior);
        self
    }

    /// Supply mass over incomplete-temporal class members.
    ///
    /// This is not a graph posterior and not completion enumeration. Mutually
    /// exclusive with [`Self::graph_posterior`]. A probability-weighted Bayesian
    /// mixture requires this; without it the result is an identified set.
    #[must_use]
    pub fn class_prior(mut self, prior: crate::ClassPrior) -> Self {
        self.class_prior = Some(prior);
        self
    }

    /// Cap TemporalCpdag / TemporalPag completion search.
    ///
    /// A cap below the true class size cannot confer class-wide point
    /// identification (`full_mass_scope`, `truncated_atoms`).
    #[must_use]
    pub fn max_completions(mut self, n: usize) -> Self {
        self.max_completions = Some(n);
        self
    }

    /// Average-effect query (static). Prefer [`Self::query`] with any [`CausalQuery`]-convertible type.
    #[must_use]
    pub fn average_effect(self, query: AverageEffectQuery) -> Self {
        self.query(query)
    }

    /// Set the causal query. Accepts [`CausalQuery`] or types that convert into it
    /// (e.g. [`AverageEffectQuery`], [`TemporalEffectQuery`]).
    #[must_use]
    pub fn query(mut self, query: impl Into<CausalQuery>) -> Self {
        let q = query.into();
        if self.estimator_spec.is_none()
            && matches!(self.estimator, Some(EstimatorId::BayesianGcomp))
        {
            match &q {
                CausalQuery::AverageEffect(_) => {}
                CausalQuery::TemporalEffect(_) => {
                    self.estimator = Some(EstimatorId::TemporalLinearAdjustment);
                }
                _ => {
                    // Remainder cells keep their identifier-native estimators.
                    self.estimator = None;
                }
            }
        }
        self.query = Some(q);
        self
    }

    /// Temporal effect query (alias of [`Self::query`]).
    #[must_use]
    pub fn temporal_query(self, query: TemporalEffectQuery) -> Self {
        self.query(query)
    }

    /// Discovery / estimation temporal-gap split.
    #[must_use]
    pub fn split(mut self, split: DiscoveryEstimationSplit) -> Self {
        self.split = Some(split);
        self
    }

    /// Configure refutation suite (static path).
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
        self.refute_explicit = true;
        self
    }

    /// Bootstrap replicates for the primary estimate.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self.bootstrap_explicit = true;
        self
    }

    /// Latency tier (`Interactive` / `Standard` / `Report`).
    ///
    /// Maps to known-equivalent bootstrap / refute / draw budgets. Explicit
    /// [`Self::bootstrap_replicates`], [`Self::refute`], and [`Self::compute_budget`]
    /// field overrides always win.
    #[must_use]
    pub fn latency_mode(mut self, mode: LatencyMode) -> Self {
        self.latency_mode = Some(mode);
        self
    }

    /// Field-level compute budget overrides (applied after latency mode mapping).
    ///
    /// [`ComputeBudget::wall_ms`] is advisory only: it is recorded on the
    /// resolved budget and is not a hard stop. Bootstrap, draw, and refute
    /// overrides *are* applied. Cancellation is a separate token, not this field.
    #[must_use]
    pub fn compute_budget(mut self, budget: ComputeBudget) -> Self {
        if budget.bootstrap.is_some() {
            self.bootstrap_explicit = true;
        }
        if budget.validators.is_some() {
            self.refute_explicit = true;
        }
        if budget.n_draws.is_some() {
            self.n_draws_explicit = true;
        }
        self.compute_budget = budget;
        self
    }

    /// Select the identification strategy for the static ATE path.
    ///
    /// Defaults to [`IdentifierId::BackdoorAdjustment`] when unset. Wire strings such as
    /// `"backdoor.adjustment"` parse via `identifier.parse::<IdentifierId>()` (see its
    /// [`std::str::FromStr`] impl). `compile` refuses any identifier/estimator pair outside
    /// the allowlist. Ignored on the temporal path (which always uses
    /// [`IdentifierId::TemporalBackdoorUnfolded`]).
    #[must_use]
    pub fn identifier(mut self, id: IdentifierId) -> Self {
        self.identifier = Some(id);
        self
    }

    /// Select the estimator for the static ATE path.
    ///
    /// Defaults to [`EstimatorId::LinearAdjustmentAte`] when unset. Wire strings such as
    /// `"linear.adjustment.ate"` parse via `estimator.parse::<EstimatorId>()` (see its
    /// [`std::str::FromStr`] impl). `compile` refuses any identifier/estimator pair outside
    /// the allowlist. Ignored on the temporal path (which always uses
    /// [`EstimatorId::TemporalLinearAdjustment`]).
    ///
    /// Request a coarsened continuous coordinate for cell-AIPW.
    ///
    /// Refused at [`Self::build`]: coarsening D is not a controlled direct
    /// effect at a point (`do(D=d0)`). Point CDE is unlicensed.
    #[must_use]
    pub fn continuous_cell(
        mut self,
        variable: antecedent_core::VariableId,
        grid: impl Into<std::sync::Arc<[f64]>>,
    ) -> Self {
        self.continuous_cell = Some((variable, grid.into()));
        self
    }

    /// Accepts either a bare [`EstimatorId`] (study fills bootstrap / overlap defaults, exactly
    /// as before) or a fully caller-configured estimator (e.g.
    /// `LinearAdjustmentAte::new().with_se_kind(..)`), via `impl Into<`[`EstimatorSpec`]`>`.
    /// Combining a configured estimator with an explicit [`Self::bootstrap_replicates`] or
    /// [`Self::overlap_policy`] is refused at [`Self::build`] time
    /// ([`CausalError::Conflict`]) rather than silently picking a winner.
    #[must_use]
    pub fn estimator(mut self, spec: impl Into<EstimatorSpec>) -> Self {
        let spec = spec.into();
        self.estimator = Some(spec.id());
        self.estimator_spec = Some(spec);
        self
    }

    /// Configure numerical and fixed-grid uncertainty options for response queries.
    ///
    /// A simultaneous band requires both `simultaneous_replicates` and an explicit
    /// bandwidth; the estimator refuses an implicit undersmoothing rule.
    #[must_use]
    pub fn response_options(mut self, options: ContinuousResponseOptions) -> Self {
        self.response_options = Some(options);
        self
    }

    /// Explicit numerical options for the observation correction on response curves.
    #[must_use]
    pub fn observation_options(
        mut self,
        options: antecedent_estimate::ObservationEstimatorOptions,
    ) -> Self {
        self.observation_options = options;
        self
    }

    /// Delayed-entry column for marginal right-censoring IPCW.
    #[must_use]
    pub fn observation_delayed_entry(mut self, variable: antecedent_core::VariableId) -> Self {
        self.observation_delayed_entry = Some(variable);
        self
    }

    /// Selection-node targets for [`CausalQuery::Transport`] (empty = Direct formula).
    #[must_use]
    pub fn selection_targets(mut self, targets: impl Into<Arc<[VariableId]>>) -> Self {
        self.selection_targets = Some(targets.into());
        self
    }

    /// Trial membership and known probabilities for [`CausalQuery::Transport`].
    #[must_use]
    pub fn transport_trial(mut self, spec: TransportTrialSpec) -> Self {
        self.transport_trial = Some(spec);
        self
    }

    /// Fixed network and realized assignment for [`CausalQuery::Interference`].
    #[must_use]
    pub fn interference(mut self, spec: InterferenceSpec) -> Self {
        self.interference = Some(spec);
        self
    }

    /// Configure frequentist vs Bayesian inference.
    ///
    /// For static backdoor ATE, [`InferenceMode::Bayesian`] selects estimator
    /// [`EstimatorId::BayesianGcomp`]. Other staged queries keep the identifier-native
    /// estimator (`functional.effect`, `functional.distribution`, `mediation.linear`,
    /// `gcm.fit`, Riesz/point derivatives); Bayesian mode is applied at execute time.
    /// Temporal queries keep [`EstimatorId::TemporalLinearAdjustment`].
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_)) && self.estimator_spec.is_none() {
            match &self.query {
                None | Some(CausalQuery::AverageEffect(_)) => {
                    self.estimator = Some(EstimatorId::BayesianGcomp);
                }
                Some(CausalQuery::TemporalEffect(_)) => {}
                Some(_) => {
                    if matches!(self.estimator, Some(EstimatorId::BayesianGcomp)) {
                        self.estimator = None;
                    }
                }
            }
        }
        if matches!(mode, InferenceMode::Frequentist)
            && self.estimator_spec.is_none()
            && self.estimator == Some(EstimatorId::BayesianGcomp)
        {
            self.estimator = None;
        }
        self.inference = mode;
        self
    }

    /// Overlap / positivity policy for propensity and AIPW estimators.
    ///
    /// When unset, those estimators keep their built-in defaults (clip = 0.01, no trim).
    /// Ignored by estimators that require [`OverlapPolicy::ExplicitOverride`] (linear, GLM, IV,
    /// front-door, RD).
    #[must_use]
    pub fn overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.overlap_policy = Some(policy);
        self
    }

    /// Bindings for named predicates and custom target-distribution weights.
    #[must_use]
    pub fn population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.population_registry = Some(registry);
        self
    }

    /// Append custom effect validators ( slow path).
    #[must_use]
    pub fn custom_validators(mut self, validators: Vec<Arc<dyn CustomEffectValidator>>) -> Self {
        self.custom_validators = validators;
        self
    }

    /// Configure the running variable / cutoff / bandwidth required by the `rd.sharp`
    /// estimator. `compile` refuses `rd.sharp` without this.
    #[must_use]
    pub fn rd_config(mut self, running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        self.rd = Some(RdConfig::new(running_variable, cutoff, bandwidth));
        self
    }

    /// Full `rd.sharp` design, including the analytic SE kind
    /// ([`RdConfig::with_se_kind`]). Replaces any earlier [`Self::rd_config`].
    #[must_use]
    pub fn rd_design(mut self, config: RdConfig) -> Self {
        self.rd = Some(config);
        self
    }

    /// Stream intermediate stage payloads (identify → point → uncertainty → validate).
    ///
    /// Final [`super::execute::Study::run`] still returns the complete result.
    #[must_use]
    pub fn stage_sink(mut self, sink: Arc<dyn super::stage::StageResultSink>) -> Self {
        self.stage_sink = Some(sink);
        self
    }

    /// Build the analysis object.
    ///
    /// # Errors
    ///
    /// Missing graph / query, Interactive+HMC, [`CausalError::Conflict`] when both
    /// [`Self::graph`] and [`Self::graph_posterior`] were set (or when a configured
    /// [`Self::estimator`] and an explicit [`Self::bootstrap_replicates`] /
    /// [`Self::overlap_policy`] disagree about who owns that setting).
    /// [`CausalError::SchemaMismatch`] when a directly-supplied [`Self::graph`]'s node
    /// count does not match the data's variable count, or — when the graph was bound
    /// via [`AcceptedGraph::with_schema`] — its bound variable names do not match the
    /// data's, in order. Not checked for [`Self::graph_posterior`], which carries a
    /// placeholder graph.
    pub fn build(self) -> Result<Study, CausalError> {
        self.finish(false)
    }

    fn finish(self, inspect_only: bool) -> Result<Study, CausalError> {
        if let Some(spec) = &self.estimator_spec {
            if spec.is_configured() {
                if self.bootstrap_explicit {
                    return Err(CausalError::Conflict {
                        what: "bootstrap_replicates",
                        detail: "set on both the builder and the configured estimator; set it \
                                 in one place (prefer the estimator)",
                    });
                }
                if self.overlap_policy.is_some() {
                    return Err(CausalError::Conflict {
                        what: "overlap_policy",
                        detail: "set on both the builder and the configured estimator; set it \
                                 in one place (prefer the estimator)",
                    });
                }
            }
        }
        if self.caller_graph.is_some() && self.tiered.is_some() {
            return Err(tiered_graph_conflict());
        }
        let data = self.data;
        if self.class_prior.is_some() && self.graph_posterior.is_some() {
            return Err(CausalError::Conflict {
                what: "class_prior",
                detail: "class_prior and graph_posterior are distinct structural-mass contracts; \
                         supply exactly one",
            });
        }
        let (graph, graph_posterior) = match (self.graph, self.graph_posterior) {
            (Some(_), Some(_)) => {
                return Err(CausalError::Conflict {
                    what: "graph",
                    detail: "both .graph(..) and .graph_posterior(..) were set; supply exactly \
                             one causal-structure input",
                });
            }
            (Some(g), None) => {
                validate_schema_binding(&g, data_schema(&data))?;
                (g, None)
            }
            (None, Some(gp)) => {
                let stub = stub_accepted_graph_for(&data, gp.n_vars, gp.atom_kind)?;
                (stub, Some(gp))
            }
            (None, None) => return Err(CausalError::Missing { field: "graph" }),
        };
        let mut refute = self.refute;
        let mut bootstrap_replicates = self.bootstrap_replicates;
        let mut inference = self.inference;
        let latency_mode = self.latency_mode;

        if let Some(mode) = latency_mode {
            refuse_non_report_hmc(mode, &inference)?;
            let resolved =
                ResolvedLatencyBudget::from_mode(mode).with_overrides(self.compute_budget);
            if !self.bootstrap_explicit {
                bootstrap_replicates = resolved.bootstrap;
            } else if let Some(b) = self.compute_budget.bootstrap {
                bootstrap_replicates = b;
            }
            if !self.refute_explicit {
                refute = resolved.refute;
            } else if let Some(v) = self.compute_budget.validators {
                refute = v;
            }
            inference = match inference {
                InferenceMode::Bayesian(cfg) => {
                    let draws = if self.n_draws_explicit {
                        self.compute_budget.n_draws.unwrap_or(cfg.n_draws)
                    } else {
                        resolved.n_draws
                    };
                    InferenceMode::Bayesian(cfg.n_draws(draws))
                }
                InferenceMode::Frequentist => InferenceMode::Frequentist,
            };
        } else if self.compute_budget.bootstrap.is_some()
            || self.compute_budget.validators.is_some()
            || self.compute_budget.n_draws.is_some()
        {
            if let Some(b) = self.compute_budget.bootstrap {
                bootstrap_replicates = b;
            }
            if let Some(v) = self.compute_budget.validators {
                refute = v;
            }
            if let Some(n) = self.compute_budget.n_draws {
                inference = match inference {
                    InferenceMode::Bayesian(cfg) => InferenceMode::Bayesian(cfg.n_draws(n)),
                    InferenceMode::Frequentist => InferenceMode::Frequentist,
                };
            }
        }

        let query = self.query.ok_or(CausalError::Missing { field: "query" })?;
        refuse_unsupported_likelihood(
            &query,
            &data,
            graph.class(),
            graph_posterior.is_none() && self.tiered.is_none(),
            &inference,
        )?;
        if !population_estimable(&query) {
            return Err(crate::unsupported_reason!(
                "population_not_estimable",
                "only an AverageEffect estimates a target population other than AllObserved; \
                 this query kind has no weighting or subpopulation estimator for another target. \
                 Declare the AllObserved population, or prepare an AllObserved AIPW or cell-AIPW \
                 study and retarget its frozen scores"
            ));
        }
        if let Some(configured) =
            self.estimator_spec.as_ref().and_then(EstimatorSpec::bootstrap_replicates)
        {
            // A configured estimator owns its replicate count (an explicit
            // builder count beside it is refused above). The study reports and
            // executes that same count instead of its own omitted default.
            bootstrap_replicates = configured;
        }
        if !self.bootstrap_explicit && !omitted_bootstrap_resamples(&query, &inference) {
            bootstrap_replicates = 0;
        }
        if !self.custom_validators.is_empty()
            && !custom_validators_apply(
                &query,
                &data,
                graph.class(),
                graph_posterior.is_some(),
                &inference,
            )
        {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "custom validators refute one scalar average effect; this query and data route \
                 has no scalar refutation problem for them to run on"
            ));
        }
        let structure = if graph_posterior.is_some() {
            crate::support::StructureSource::GraphPosterior
        } else {
            self.structure_source.unwrap_or(crate::support::StructureSource::Explicit)
        };
        // Graph-posterior stubs are empty bookkeeping graphs. Collapse based on
        // stub edges (an empty Admg looks like a DAG) would relabel licensed
        // Admg posterior cells. Trust the atom-kind stub class instead.
        let graph_class = if graph_posterior.is_some() {
            graph.class()
        } else {
            crate::support::effective_graph_class(&graph, &query)
        };
        let matrix_class = if graph_posterior.is_some() {
            graph.class().as_str()
        } else {
            crate::support::matrix_graph_class(&graph, &query, self.tiered.as_ref())
        };
        let selected = self.estimator_spec.as_ref().map(crate::estimator_spec::EstimatorSpec::id);
        let functional = match &query {
            CausalQuery::AverageEffect(q) => Some(&q.outcome_functional),
            CausalQuery::ConditionalEffect(q) => Some(&q.inner.outcome_functional),
            CausalQuery::Response(q) => Some(&q.outcome_functional),
            _ => None,
        };
        if functional.is_some_and(|f| !matches!(f, antecedent_core::OutcomeFunctional::Mean)) {
            let transforms_outcome = matches!(inference, InferenceMode::Frequentist)
                && graph_posterior.is_none()
                && match &query {
                    CausalQuery::AverageEffect(_) => {
                        graph_class == GraphClass::Dag || self.tiered.is_some()
                    }
                    CausalQuery::ConditionalEffect(_) => {
                        matches!(graph_class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag)
                    }
                    CausalQuery::Response(q) => q.temporal.is_none(),
                    _ => false,
                };
            if !transforms_outcome {
                return Err(CausalError::Unsupported {
                    message: "this inference/graph/query path does not implement the requested outcome functional",
                });
            }
        }
        if functional.and_then(antecedent_core::OutcomeFunctional::quantile_level).is_some() {
            let supported = matches!(inference, InferenceMode::Frequentist)
                && graph_posterior.is_none()
                && match &query {
                    CausalQuery::AverageEffect(q) => {
                        selected == Some(EstimatorId::Aipw)
                            && q.target_population == antecedent_core::TargetPopulation::AllObserved
                            && (graph_class == GraphClass::Dag
                                || self.tiered.as_ref().is_some_and(|b| {
                                    b.within_tier == antecedent_graph::WithinTier::CoDetermined
                                }))
                    }
                    CausalQuery::ConditionalEffect(q) => {
                        q.inner.target_population == antecedent_core::TargetPopulation::AllObserved
                            && q.inner.effect_modifiers.len() == 1
                            && matches!(
                                graph_class,
                                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
                            )
                    }
                    CausalQuery::Response(q) => {
                        selected == Some(EstimatorId::CellAipw)
                            && q.target_population == antecedent_core::TargetPopulation::AllObserved
                            && q.temporal.is_none()
                    }
                    _ => false,
                };
            if !supported {
                return Err(CausalError::Unsupported {
                    message: "quantiles require Frequentist AllObserved AIPW AverageEffect, binary ConditionalEffect with one modifier, or cell-AIPW joint response; use prepare + retarget for score-table target weights",
                });
            }
            if self.refute != crate::RefuteSuite::None {
                return Err(CausalError::Unsupported {
                    message: "quantile functionals currently require refute=none; mean-effect refuters do not validate a quantile",
                });
            }
        }

        if self.tiered.is_some() && self.identifier.is_some() {
            return Err(CausalError::Unsupported {
                message: "TieredBackground selects its own identifier; omit identifier",
            });
        }
        if self.continuous_cell.is_some() {
            return Err(CausalError::Unsupported {
                message: antecedent_estimate::POINT_CDE_UNLICENSED,
            });
        }
        let codetermined_joint = self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
            && matches!(
                &query,
                CausalQuery::Response(q)
                    if q.temporal.is_none()
                        && matches!(
                            &q.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse {
                                interventions,
                                ..
                            } if interventions.len() >= 2
                        )
            );
        let mut identification_cache = None;
        let cell_aipw =
            selected == Some(EstimatorId::CellAipw) || (selected.is_none() && codetermined_joint);
        if cell_aipw {
            if let Some(background) = &self.tiered {
                let CausalQuery::Response(response) = &query else {
                    return Err(CausalError::Unsupported {
                        message: "cell-AIPW on a tiered background requires joint InterventionResponse",
                    });
                };
                if matches!(inference, InferenceMode::Bayesian(_)) {
                    return Err(CausalError::Unsupported {
                        message: "CoDetermined joint cells are Frequentist cell.aipw",
                    });
                }
                if selected.is_some_and(|id| id != EstimatorId::CellAipw) {
                    return Err(CausalError::Unsupported {
                        message: "CoDetermined joint cells require estimator cell.aipw",
                    });
                }
                let identification = match graph.as_admg() {
                    Some(admg) => {
                        antecedent_identify::identify_tiered_joint_on(background, admg, response)?
                    }
                    None => antecedent_identify::identify_tiered_joint(
                        background,
                        data_schema(&data),
                        response,
                    )?,
                };
                if !matches!(
                    identification.status,
                    antecedent_core::IdentificationStatus::NonparametricallyIdentified
                        | antecedent_core::IdentificationStatus::PartiallyIdentified
                ) || identification.estimands.is_empty()
                {
                    return Err(CausalError::Unsupported {
                        message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                    });
                }
                let estimand = identification.estimands[0].clone();
                identification_cache =
                    Some(Arc::new(super::prepared::CachedStaticIdentification {
                        identification,
                        estimand,
                    }));
            } else if graph_class != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "cell-AIPW requires a static DAG with one certified common adjustment set",
                });
            }
        }
        if matches!(functional, Some(antecedent_core::OutcomeFunctional::ExceedanceGrid(_))) {
            let score_grid = matches!(inference, InferenceMode::Frequentist)
                && graph_posterior.is_none()
                && match &query {
                    CausalQuery::AverageEffect(q) => {
                        selected == Some(EstimatorId::Aipw)
                            && matches!(
                                q.target_population,
                                antecedent_core::TargetPopulation::AllObserved
                            )
                            && (graph_class == GraphClass::Dag
                                || self.tiered.as_ref().is_some_and(|b| {
                                    b.within_tier == antecedent_graph::WithinTier::CoDetermined
                                }))
                    }
                    CausalQuery::Response(q) => {
                        selected == Some(EstimatorId::CellAipw)
                            && q.temporal.is_none()
                            && (graph_class == GraphClass::Dag
                                || self.tiered.as_ref().is_some_and(|b| {
                                    b.within_tier == antecedent_graph::WithinTier::CoDetermined
                                }))
                    }
                    CausalQuery::ConditionalEffect(_) => {
                        matches!(graph_class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag)
                    }
                    _ => false,
                };
            if !score_grid {
                return Err(CausalError::Unsupported {
                    message: "full exceedance grids require explicit iid AIPW, cell-AIPW joint response, or Frequentist ConditionalEffect on Dag/Cpdag/Pag",
                });
            }
        }
        if let Some(estimator) = self.estimator {
            refuse_estimator_inference_mismatch(&query, estimator, &inference)?;
        }
        if let Some(spec) = &self.estimator_spec {
            let bayesian = matches!(inference, InferenceMode::Bayesian(_));
            let expected = match &query {
                CausalQuery::ConditionalEffect(_) => Some(if bayesian {
                    EstimatorId::BayesianConditional
                } else {
                    EstimatorId::ConditionalLinearAdjustment
                }),
                CausalQuery::Response(q) => Some(if q.temporal.is_some() {
                    if bayesian {
                        EstimatorId::TemporalResponseBayesian
                    } else {
                        EstimatorId::TemporalResponseGcomp
                    }
                } else if bayesian
                    && !matches!(
                        &q.functional,
                        antecedent_core::ResponseFunctional::AverageDerivative { .. }
                            | antecedent_core::ResponseFunctional::PointDerivative { .. }
                            | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                            | antecedent_core::ResponseFunctional::Jacobian { .. }
                    )
                {
                    EstimatorId::ResponseBayesian
                } else {
                    EstimatorId::default_for_response(&q.functional)
                }),
                CausalQuery::Mediation(_) if graph_class == GraphClass::TemporalDag => {
                    Some(if bayesian {
                        EstimatorId::BayesianTemporalMediation
                    } else {
                        EstimatorId::TemporalMediation
                    })
                }
                _ => None,
            };
            if let Some(expected) = expected {
                let cell_aipw_ok = spec.id() == EstimatorId::CellAipw
                    && matches!(
                        &query,
                        CausalQuery::Response(q)
                            if q.temporal.is_none()
                                && matches!(
                                    q.functional,
                                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                                )
                    );
                let admg_functional_ok = spec.id() == EstimatorId::FunctionalEffect
                    && graph_class == GraphClass::Admg
                    && matches!(
                        &query,
                        CausalQuery::Response(q)
                            if q.temporal.is_none()
                                && matches!(
                                    q.functional,
                                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                                        | antecedent_core::ResponseFunctional::MeanCurve { .. }
                                )
                    );
                if spec.id() != expected && !cell_aipw_ok && !admg_functional_ok {
                    return Err(crate::compile_reason!(
                        "strategy_incompatible",
                        "query and inference require estimator {}; got {}",
                        expected.as_str(),
                        spec.id().as_str()
                    ));
                }
            }
        }

        let mut refute_default_downgrade: Option<RefuteSuite> = None;
        if !self.refute_explicit {
            let requested = crate::support::support_cell_named(
                &query,
                matrix_class,
                structure,
                &inference,
                refute,
            );
            let without_validation = crate::support::support_cell_named(
                &query,
                matrix_class,
                structure,
                &inference,
                RefuteSuite::None,
            );
            if requested.is_some_and(|cell| {
                matches!(
                    crate::support::classify(cell),
                    crate::support::CellStatus::NotApplicable { .. }
                        | crate::support::CellStatus::Refused
                )
            }) && without_validation.is_some_and(|cell| {
                matches!(
                    crate::support::classify(cell),
                    crate::support::CellStatus::Licensed
                        | crate::support::CellStatus::Allowlisted { .. }
                )
            }) {
                refute_default_downgrade = Some(refute);
                refute = RefuteSuite::None;
            }
        }
        if self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::Unknown)
        {
            match &query {
                CausalQuery::ConditionalEffect(_) => {
                    return Err(CausalError::Unsupported {
                        message: "Unknown-tier ConditionalEffect is not licensed; the \
                                  two-scenario envelope is AverageEffect only",
                    });
                }
                CausalQuery::Response(q) if q.functional.treatment_ids().len() <= 1 => {
                    return Err(CausalError::Unsupported {
                        message: "Unknown-tier single-treatment Response is not licensed; the \
                                  two-scenario envelope is AverageEffect only",
                    });
                }
                _ => {}
            }
        }
        if let DataInput::Panel(panel) = &data {
            refuse_unlicensed_panel_route(
                &query,
                graph.class(),
                &inference,
                panel,
                self.split.as_ref(),
            )?;
        }
        if self.class_prior.is_some() && matches!(inference, crate::InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "class_prior is a structural probability over class members and \
                          requires Bayesian inference; Frequentist incomplete-class Pulse \
                          keeps enumeration-weighted ATE without treating those weights as \
                          probabilities",
            });
        }
        if self.class_prior.is_some() && !graph.class().is_incomplete_temporal() {
            return Err(CausalError::Unsupported {
                message: "class_prior requires an incomplete temporal graph class",
            });
        }
        let support_status = if let Some(cell) =
            crate::support::support_cell_named(&query, matrix_class, structure, &inference, refute)
        {
            if inspect_only {
                Some(crate::support::classify(cell))
            } else {
                Some(crate::support::refuse_if_not_applicable(cell)?)
            }
        } else {
            None
        };

        let (selection_diagram, transport_trial, interference) = match &query {
            CausalQuery::Transport(transport) => {
                if !inspect_only {
                    refuse_unlicensed_transport(transport)?;
                }
                if self.interference.is_some() {
                    return Err(CausalError::Unsupported {
                        message: "interference network is not used by TransportQuery",
                    });
                }
                let admg = graph.as_admg().ok_or(CausalError::Unsupported {
                    message: "TransportQuery requires a supplied Admg",
                })?;
                let targets = self.selection_targets.clone().unwrap_or_default();
                let diagram = antecedent_graph::SelectionDiagram::try_new(admg.clone(), targets)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let trial = self.transport_trial.clone().ok_or(CausalError::Unsupported {
                    message: "TransportQuery requires StudyBuilder::transport_trial",
                })?;
                (Some(diagram), Some(trial), None)
            }
            CausalQuery::Interference(design) => {
                if !inspect_only {
                    refuse_unlicensed_interference(design)?;
                }
                if self.transport_trial.is_some() || self.selection_targets.is_some() {
                    return Err(CausalError::Unsupported {
                        message: "transport trial columns are not used by InterferenceQuery",
                    });
                }
                let spec = self.interference.clone().ok_or(CausalError::Unsupported {
                    message: "InterferenceQuery requires StudyBuilder::interference",
                })?;
                let DataInput::Tabular(units) = &data else {
                    return Err(CausalError::Unsupported {
                        message: "InterferenceQuery requires tabular unit data",
                    });
                };
                if spec.network.units().row_count() != units.row_count()
                    || spec.assignment.len() != units.row_count()
                {
                    return Err(CausalError::Compile {
                        message: "interference network, assignment, and unit table row counts \
                                  must match"
                            .into(),
                    });
                }
                (None, None, Some(spec.bound_to(units)?))
            }
            _ if self.transport_trial.is_some()
                || self.selection_targets.is_some()
                || self.interference.is_some() =>
            {
                return Err(CausalError::Unsupported {
                    message: "selection_targets / transport_trial / interference are only valid \
                              for TransportQuery or InterferenceQuery",
                });
            }
            _ => (None, None, None),
        };

        Ok(Study {
            data,
            graph,
            graph_posterior,
            class_prior: self.class_prior,
            max_completions: self.max_completions,
            structure_source: structure,
            support_status,
            query,
            refute,
            refute_default_downgrade,
            bootstrap_replicates,
            split: self.split,
            identifier: self.identifier,
            estimator: self.estimator,
            estimator_spec_identity: self
                .estimator_spec
                .as_ref()
                .map(super::contract_identity::estimator_spec_identity),
            estimator_spec: self.estimator_spec,
            response_options: self.response_options,
            observation_options: self.observation_options,
            observation_delayed_entry: self.observation_delayed_entry,
            rd: self.rd,
            inference,
            overlap_policy: self.overlap_policy,
            population_registry: self.population_registry,
            custom_validators: self.custom_validators,
            latency_mode,
            stage_sink: self.stage_sink,
            identification_cache,
            mediation_adjustment_cache: None,
            pag_identification_cache: None,
            cpdag_identification_cache: None,
            temporal_identification_cache: None,
            temporal_class_identification_cache: None,
            graph_posterior_identification_cache: None,
            dbn_posterior_identification_cache: None,
            temporal_class_posterior_identification_cache: None,
            tiered: self.tiered,
            continuous_cell: self.continuous_cell,
            shared_batch_design: None,
            selection_diagram,
            transport_trial,
            interference,
            transport_identification_cache: None,
        })
    }
}

impl Study {
    /// Matrix structure-source axis recorded at [`StudyBuilder::build`].
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.structure_source
    }

    /// Evidence contract recorded at [`StudyBuilder::build`]: licensed, or
    /// allowlisted-and-unlicensed. `None` when the query is not on the public
    /// matrix axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.support_status
    }

    /// Start a builder over tabular data.
    #[must_use]
    pub fn tabular(data: TabularData) -> StudyBuilder {
        StudyBuilder::from_data(DataInput::Tabular(data))
    }

    /// Start a builder over temporal series data.
    #[must_use]
    pub fn series(data: TimeSeriesData) -> StudyBuilder {
        StudyBuilder::from_data(DataInput::Temporal(data))
    }

    /// Start a builder over multi-environment series (context-aware temporal analysis).
    #[must_use]
    pub fn series_multi(data: MultiEnvironmentData) -> StudyBuilder {
        StudyBuilder::from_data(DataInput::MultiEnv(data))
    }

    /// Start a builder over multi-unit panel data (stacked cluster-HAC estimate).
    #[must_use]
    pub fn panel(data: PanelData) -> StudyBuilder {
        StudyBuilder::from_data(DataInput::Panel(data))
    }

    /// Start a builder over irregular event data, aligned onto a regular duration grid
    /// (§5.4) immediately — eagerly, at the call site, rather than deferred to `build()`.
    /// Integer-lag algorithms then run on the aligned series; raw event indices are never
    /// treated as lags.
    ///
    /// # Errors
    ///
    /// Event alignment failure (`align_interval_ns` incompatible with the event stream).
    pub fn events(data: &EventData, align_interval_ns: u64) -> Result<StudyBuilder, CausalError> {
        let aligned = data
            .align_to_grid(align_interval_ns)
            .map_err(|e| CausalError::Compile { message: format!("event align_to_grid: {e}") })?;
        Ok(StudyBuilder::from_data(DataInput::Event(aligned)))
    }
}

#[cfg(test)]
mod estimator_spec_conflict_tests {
    use antecedent_estimate::LinearAdjustmentAte;

    use super::*;

    /// Minimal valid tabular data. The conflict check in [`StudyBuilder::build`] runs
    /// before the graph / query presence checks, so these tests never need a real
    /// graph or query — only a builder that exists at all, which now requires data.
    fn toy_data() -> TabularData {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};

        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let n = 4usize;
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(vec![0.0, 1.0, 0.0, 1.0]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(vec![1.0, 3.0, 1.1, 2.9]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    fn toy_builder() -> StudyBuilder {
        Study::tabular(toy_data())
    }

    #[test]
    fn configured_estimator_plus_explicit_bootstrap_replicates_conflicts() {
        let result = toy_builder()
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(500))
            .bootstrap_replicates(100)
            .build();
        match result {
            Err(CausalError::Conflict { what, .. }) => assert_eq!(what, "bootstrap_replicates"),
            other => panic!("expected CausalError::Conflict, got {other:?}"),
        }
    }

    #[test]
    fn configured_estimator_plus_explicit_overlap_policy_conflicts() {
        let result = toy_builder()
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(500))
            .overlap_policy(OverlapPolicy::ExplicitOverride)
            .build();
        match result {
            Err(CausalError::Conflict { what, .. }) => assert_eq!(what, "overlap_policy"),
            other => panic!("expected CausalError::Conflict, got {other:?}"),
        }
    }

    #[test]
    fn configured_estimator_alone_does_not_conflict() {
        let result = toy_builder()
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(500))
            .build();
        assert!(!matches!(result, Err(CausalError::Conflict { .. })));
    }

    #[test]
    fn explicit_bootstrap_replicates_alone_does_not_conflict() {
        let result = toy_builder().bootstrap_replicates(100).build();
        assert!(!matches!(result, Err(CausalError::Conflict { .. })));
    }
}
