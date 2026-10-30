//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Builder types.

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

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
use crate::error::CausalError;
use crate::estimator_spec::EstimatorSpec;
use crate::inference::InferenceMode;
use crate::strategy_table::{EstimatorId, IdentifierId};

use super::execute::Study;
use super::latency::{ComputeBudget, LatencyMode};

mod finish;
mod settings;

use settings::{Declarations, ExecutionSettings, ExposureInputs, InferenceSettings};

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Cheap interactive validators: overlap + E-value only.
    Cheap,
    /// Placebo + random common cause (linear backdoor only).
    PlaceboAndRcc,
    /// Full validation suite (applicable validators only; others `NotApplicable`).
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
pub(super) fn refuse_unlicensed_transport(
    query: &antecedent_core::TransportQuery,
) -> Result<(), CausalError> {
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

/// Refuse interference designs without a checked execution path.
pub(super) fn refuse_unlicensed_interference(
    query: &antecedent_core::InterferenceQuery,
) -> Result<(), CausalError> {
    let licensed =
        (matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
            && matches!(query.exposure, antecedent_core::ExposureMapping::NeighborCount))
            || (matches!(
                query.assignment,
                antecedent_core::AssignmentDesign::ClusterRandomization { .. }
            ) && matches!(query.exposure, antecedent_core::ExposureMapping::NeighborFraction))
            || (matches!(
                query.assignment,
                antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
            ) && matches!(
                query.exposure,
                antecedent_core::ExposureMapping::NeighborCount
                    | antecedent_core::ExposureMapping::NeighborFraction
                    | antecedent_core::ExposureMapping::WeightedNeighborExposure
            ))
            || (matches!(
                query.assignment,
                antecedent_core::AssignmentDesign::ObservedExposure { .. }
            ) && matches!(
                query.exposure,
                antecedent_core::ExposureMapping::NeighborCount
                    | antecedent_core::ExposureMapping::NeighborFraction
                    | antecedent_core::ExposureMapping::WeightedNeighborExposure
            ));
    if licensed {
        Ok(())
    } else {
        Err(crate::support_reason!(
            "construction_not_licensed",
            "InterferenceQuery supports Bernoulli/NeighborCount, cluster total effects, exact two-stage saturation, or supplied-propensity observational exposure with built-in neighbor mappings"
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
    let first = &panel.unit(0).map_err(CausalError::rejected)?.series.time_index().regularity;
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

#[derive(Clone)]
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

impl std::fmt::Debug for DataInput {
    /// Shape only: a derived impl would print every column value.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, unit, count) = match self {
            Self::Tabular(data) => ("Tabular", "rows", data.row_count()),
            Self::Temporal(data) => ("Temporal", "rows", data.row_count()),
            Self::Event(data) => ("Event", "rows", data.row_count()),
            Self::MultiEnv(data) => ("MultiEnv", "environments", data.env_count()),
            Self::Panel(data) => ("Panel", "rows", data.total_rows()),
        };
        f.debug_struct(kind).field(unit, &count).finish_non_exhaustive()
    }
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
pub(super) fn stub_accepted_graph_for(
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
                            .map_err(CausalError::rejected)?;
                    }
                    Ok(AcceptedGraph::admg(admg))
                }
                antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    let mut admg = Admg::with_variables(n);
                    if n >= 2 {
                        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                            .map_err(CausalError::rejected)?;
                    }
                    let temporal =
                        antecedent_graph::TemporalAdmg::from_static_contemporaneous(&admg)
                            .map_err(CausalError::rejected)?;
                    Ok(AcceptedGraph::temporal_admg(temporal))
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
            // Lagged temporal-ADMG atoms: the posterior carries the templates; the stub only
            // names their class.
            antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                Ok(AcceptedGraph::temporal_admg(antecedent_graph::TemporalAdmg::empty()))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Admg => Err(CausalError::Unsupported {
                message: "Admg graph posteriors require tabular data; a series posterior over \
                          latent-confounded templates is a lagged TemporalAdmg posterior",
            }),
            _ => Err(CausalError::Unsupported {
                message: "graph-posterior stub supports only TemporalDag, TemporalCpdag, \
                          TemporalPag, or lagged TemporalAdmg atoms on temporal/event data",
            }),
        },
        DataInput::MultiEnv(_) | DataInput::Panel(_) => Err(CausalError::Unsupported {
            message: "graph-posterior analysis supports tabular or temporal/event data only",
        }),
    }
}

/// The empty structure of the class a discovery study answers with.
///
/// A discovery study supplies no graph: this edgeless graph of the returned
/// class (CPDAG and PAG stay classes) states the accepted graph class its
/// contract reports. It is never consulted for identification; the discovered
/// structure reaches a downstream study only through its review gate.
///
/// # Errors
///
/// A schema wider than a node id holds.
pub(super) fn discovery_stub_graph(
    data: &DataInput,
    class: antecedent_core::DiscoveredClass,
) -> Result<AcceptedGraph, CausalError> {
    use antecedent_core::DiscoveredClass as C;
    let width = u32::try_from(data_schema(data).len()).map_err(|_| CausalError::Compile {
        message: "too many variables for a discovery stub graph".into(),
    })?;
    match class {
        C::Dag | C::DagPosterior | C::DagEdgeMarginals => {
            Ok(AcceptedGraph::dag(Dag::with_variables(width)))
        }
        // A class posterior is over CPDAGs; a stability table has no class of its own and is
        // given the class of the search it re-runs by `discovery_stub_class` before this call.
        C::Cpdag | C::CpdagPosterior | C::StructureStability => {
            AcceptedGraph::cpdag(Cpdag::with_variables(width))
        }
        C::Pag | C::PagPosterior => Ok(AcceptedGraph::pag(Pag::with_variables(width))),
        C::TemporalDag | C::TemporalDagPosterior => {
            Ok(AcceptedGraph::temporal_dag(TemporalDag::empty()))
        }
        C::TemporalCpdag | C::RegimeTemporalCpdags => {
            AcceptedGraph::temporal_cpdag(TemporalCpdag::empty())
        }
        C::TemporalPag => Ok(AcceptedGraph::temporal_pag(TemporalPag::empty())),
    }
}

/// The class the stub graph of a discovery study states: the class the search returns, or,
/// for a stability search, the class of the search it re-runs.
pub(super) fn discovery_stub_class(
    method: &antecedent_core::DiscoveryMethod,
) -> antecedent_core::DiscoveredClass {
    match method {
        antecedent_core::DiscoveryMethod::BootstrapStability { base, .. } => base.output(),
        // A class posterior over lagged templates is answered on the temporal side.
        antecedent_core::DiscoveryMethod::ClassPosterior { base } if base.is_temporal() => {
            base.output()
        }
        other => other.output(),
    }
}

/// Marker: the study structure came from [`StudyBuilder::graph`].
#[derive(Clone, Copy, Debug)]
struct CallerGraph;

/// A setting named on both the builder and a configured estimator: the same conflict for
/// every setting, carrying the setting as the diagnosed subject.
pub(super) fn estimator_setting_conflict(what: &'static str) -> CausalError {
    const DETAIL: &str = "set on both the builder and the configured estimator; set it \
                          in one place (prefer the estimator)";
    CausalError::Conflict { what, detail: DETAIL }.diagnosed(antecedent_estimate::RefusalFields {
        stage: Some("study_build".to_string()),
        subject: Some(what.to_string()),
        reason: Some("set_on_builder_and_configured_estimator".to_string()),
        remedy: Some("set it in one place, preferably on the configured estimator".to_string()),
        ..antecedent_estimate::RefusalFields::default()
    })
}

/// A caller graph and a tier background both describe the study structure.
pub(super) fn tiered_graph_conflict() -> CausalError {
    CausalError::Conflict {
        what: "graph",
        detail: "both .graph(..) and .tiered_background(..) were set; the tier background \
                 materializes its own closure ADMG / PAG, so supply exactly one structure input",
    }
}

/// Borrow the schema backing `data`, regardless of modality.
pub(super) fn data_schema(data: &DataInput) -> &CausalSchema {
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
    match graph.graph_ref() {
        GraphRef::Dag(g) => Some(g.node_count()),
        GraphRef::Admg(g) => Some(g.node_count()),
        GraphRef::Cpdag(g) => Some(g.node_count()),
        GraphRef::Pag(g) => Some(g.node_count()),
        GraphRef::TemporalDag(_)
        | GraphRef::TemporalCpdag(_)
        | GraphRef::TemporalPag(_)
        | GraphRef::TemporalAdmg(_)
        | GraphRef::RandomizedTrial => None,
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
pub(super) fn validate_schema_binding(
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
#[derive(Clone, Debug)]
pub struct StudyBuilder {
    data_diagnostics: Option<crate::DataDiagnostics>,
    data: DataInput,
    graph: Option<AcceptedGraph>,
    /// Alternative to [`Self::graph`]: a posterior over structures rather than one
    /// accepted structure. Mutually exclusive with `graph` (checked at [`Self::build`]).
    graph_posterior: Option<GraphPosterior>,
    query: Option<CausalQuery>,
    /// Deferred typed refusal from the most recent discovery declaration.
    discovery_query_error: Option<antecedent_core::QueryError>,
    /// A requested bootstrap interval, set by [`Self::bootstrap_interval`].
    bootstrap_interval: Option<antecedent_estimate::IntervalRequest>,
    /// Set when [`Self::graph`] is called. A caller graph and a tier background
    /// are two sources of truth for one structure; build refuses the pair.
    caller_graph: Option<CallerGraph>,
    /// Structure, procedure and population declarations.
    declared: Declarations,
    /// Transport trial and interference network inputs.
    exposure: ExposureInputs,
    /// Inference mode, validation suite and bootstrap budget.
    settings: InferenceSettings,
    /// Latency tier, compute budget and progressive result sink.
    execution: ExecutionSettings,
    /// Executable search of a structure-discovery query.
    discovery: Option<Arc<super::structure_route::DiscoverySpec>>,
    /// Mechanism fit of an attribution or simulation query.
    mechanisms: Option<Arc<super::structure_route::MechanismProgram>>,
    /// Provenance of a graph posterior accepted from a compiled structure search.
    posterior_provenance: Option<Arc<crate::accepted::StructureProvenance>>,
}

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
pub(super) fn refuse_estimator_inference_mismatch(
    query: &CausalQuery,
    estimator: EstimatorId,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    const _: () = assert!(
        crate::error::is_runtime_refusal_code("estimator_inference_mismatch"),
        "`estimator_inference_mismatch` is not a runtime_refusal code"
    );
    match inference {
        InferenceMode::Frequentist => refuse_bayesian_estimator_under_frequentist(estimator),
        InferenceMode::Bayesian(_) => refuse_frequentist_estimator_under_bayesian(query, estimator),
    }
}

/// Refuses a Bayesian-only estimator selected under Frequentist inference.
fn refuse_bayesian_estimator_under_frequentist(estimator: EstimatorId) -> Result<(), CausalError> {
    match estimator {
        EstimatorId::BayesianGcomp => bayesian!("bayesian.gcomp"),
        EstimatorId::BayesianBasisGcomp => bayesian!("bayesian.basis.gcomp"),
        EstimatorId::BayesianIvJointLinear => bayesian!("iv.bayesian_joint_linear"),
        EstimatorId::BayesianRdLocalLinear => bayesian!("rd.bayesian_local_linear"),
        EstimatorId::BayesianRobustAte => bayesian!("bayesian.robust_ate"),
        EstimatorId::BayesianPanelDid => bayesian!("bayesian.panel_did"),
        EstimatorId::BayesianSyntheticControl => bayesian!("bayesian.synthetic_control"),
        EstimatorId::BayesianLocalRatio => bayesian!("bayesian.local_ratio"),
        EstimatorId::BayesianPiecewiseSurvival => bayesian!("bayesian.piecewise_survival"),
        EstimatorId::BayesianRandomizedQuantile => bayesian!("bayesian.randomized_quantile"),
        EstimatorId::BayesianDosePolicy => bayesian!("bayesian.continuous_dose_policy"),
        EstimatorId::BayesianRandomizedFinitePopulation => {
            bayesian!("bayesian.randomized_finite_population")
        }
        EstimatorId::BayesianRandomizedSuperpopulation => {
            bayesian!("bayesian.randomized_superpopulation")
        }
        EstimatorId::BayesianSwitchback => bayesian!("bayesian.switchback_effect"),
        EstimatorId::BayesianComplierEffect => bayesian!("bayesian.complier_effect"),
        EstimatorId::BayesianBinaryPolicyValue => bayesian!("bayesian.binary_policy_value"),
        EstimatorId::BayesianSequentialRegimeValue => {
            bayesian!("bayesian.sequential_regime_value")
        }
        EstimatorId::BayesianSequentialCrossfitContrast => {
            bayesian!("bayesian.sequential_crossfit_contrast")
        }
        EstimatorId::BayesianPeriodHistoryOutcomeModel => {
            bayesian!("bayesian.period_history_outcome_model")
        }
        EstimatorId::BayesianTemporalDesignExcursion => {
            bayesian!("bayesian.temporal_design.excursion")
        }
        EstimatorId::BayesianIvBounds => bayesian!("bayesian.iv_bounds.endpoint_bootstrap"),
        EstimatorId::BayesianTemporalRegimeSurvival => {
            bayesian!("bayesian.temporal_design.regime_survival")
        }
        EstimatorId::TransportTrialBayesianBootstrap => {
            bayesian!("transport.trial_bayesian_bootstrap")
        }
        EstimatorId::BayesianConditional => bayesian!("conditional.bayesian"),
        EstimatorId::BayesianTemporalGcomp => bayesian!("bayesian.temporal.gcomp"),
        EstimatorId::BayesianTemporalWhiteInnovation => {
            bayesian!("bayesian.temporal.white_innovation_linear")
        }
        EstimatorId::BayesianTemporalGlmLinkContrast => {
            bayesian!("bayesian.temporal.glm_link_contrast")
        }
        EstimatorId::BayesianTemporalLaggedInstrument => {
            bayesian!("bayesian.temporal.lagged_instrument")
        }
        EstimatorId::TemporalResponseBayesian => bayesian!("response.temporal.bayesian"),
        EstimatorId::ResponseBayesian => bayesian!("response.bayesian"),
        EstimatorId::BayesianTemporalMediation => bayesian!("temporal.mediation.bayesian"),
        _ => Ok(()),
    }
}

/// Refuses a Frequentist-only estimator selected under Bayesian inference.
fn refuse_frequentist_estimator_under_bayesian(
    query: &CausalQuery,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let average = matches!(query, CausalQuery::AverageEffect(_));
    match estimator {
        EstimatorId::LinearAdjustmentAte if average => {
            frequentist!("linear.adjustment.ate")
        }
        // Propensity draws are ancillary for the weighting effect, so a Bayesian
        // weighting estimator has no posterior to report: the typed boundary names
        // the missing outcome law and the posteriors that answer the same effect.
        EstimatorId::PropensityWeighting if average => Err(CausalError::Estimate(
            antecedent_estimate::penalized_propensity_posterior::weighting_effect_posterior_refusal(
            ),
        )),
        // A matching rule has no likelihood, so the mismatch is the typed boundary of a
        // matching posterior (premise and next attempt), not a mislabelled interval.
        EstimatorId::PropensityMatching | EstimatorId::DistanceMatching if average => {
            Err(super::matching_posterior_refusal::matching_posterior_refusal(
                super::matching_posterior_refusal::MatchingSetting::Static,
            ))
        }
        EstimatorId::PeriodHistoryMatching => {
            Err(super::matching_posterior_refusal::matching_posterior_refusal(
                super::matching_posterior_refusal::MatchingSetting::WithinPeriodHistory,
            ))
        }
        EstimatorId::PropensityStratification if average => {
            frequentist!("propensity.stratification")
        }
        EstimatorId::Aipw if average => frequentist!("aipw"),
        EstimatorId::Dml if average => frequentist!("dml"),
        EstimatorId::DrLearner if average => frequentist!("dr.learner"),
        EstimatorId::CausalForest if average => frequentist!("causal.forest"),
        EstimatorId::GlmAdjustment if average => frequentist!("glm.adjustment"),
        EstimatorId::FrontDoorTwoStage if average => {
            frequentist!("frontdoor.linear_two_stage")
        }
        EstimatorId::FrontDoorFunctional if average => frequentist!("frontdoor.functional"),
        EstimatorId::IvWald if average => frequentist!("iv.wald"),
        EstimatorId::Iv2Sls if average => frequentist!("iv.2sls"),
        EstimatorId::RdSharp if average => frequentist!("rd.sharp"),
        EstimatorId::ConditionalLinearAdjustment => {
            frequentist!("conditional.linear.adjustment")
        }
        EstimatorId::CellAipw => frequentist!("cell.aipw"),
        EstimatorId::TemporalGlmLinkContrast => frequentist!("temporal.glm_link_contrast"),
        EstimatorId::IvBoundsSharp => frequentist!("iv_bounds.balke_pearl"),
        EstimatorId::TemporalDesignExcursion => frequentist!("temporal_design.excursion"),
        EstimatorId::TemporalRegimeSurvival => frequentist!("temporal_design.regime_survival"),
        EstimatorId::TemporalLaggedInstrument => frequentist!("temporal.lagged_instrument"),
        EstimatorId::RandomizedSurvivalProductLimit => {
            frequentist!("randomized.survival_product_limit")
        }
        EstimatorId::ContinuousDoseKernel => {
            frequentist!("policy.triangular_kernel_inverse_density")
        }
        EstimatorId::TransportTrialIpw => frequentist!("transport.trial_ipw"),
        EstimatorId::InterferenceHtHajek => frequentist!("interference.ht_hajek"),
        EstimatorId::InterferenceClusterNeyman => {
            frequentist!("interference.cluster_neyman")
        }
        EstimatorId::InterferenceSaturationExact => {
            frequentist!("interference.saturation_exact")
        }
        EstimatorId::InterferenceObservationalIpw => {
            frequentist!("interference.observational_ipw")
        }
        EstimatorId::RandomizedHt
        | EstimatorId::RandomizedNeyman
        | EstimatorId::RandomizedSwitchbackHt
            if matches!(query, CausalQuery::RandomizedEffect(_)) =>
        {
            frequentist!("the design-based randomized estimator")
        }
        _ => Ok(()),
    }
}

pub(super) fn omitted_bootstrap_resamples(query: &CausalQuery, inference: &InferenceMode) -> bool {
    match query {
        CausalQuery::Response(q) => {
            q.is_temporal() && matches!(inference, InferenceMode::Frequentist)
        }
        // A Frequentist counterfactual or attribution interval is a whole-procedure
        // mechanism-refit bootstrap that runs only when a replicate count is requested.
        CausalQuery::Counterfactual(_)
        | CausalQuery::NestedCounterfactual(_)
        | CausalQuery::AnomalyAttribution(_)
        | CausalQuery::ChangeAttribution(_) => false,
        CausalQuery::Attribute(_) | CausalQuery::Simulate(_) => {
            !super::execute::attribute_bootstrap_supported(query)
        }
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
pub(super) fn population_estimable(query: &CausalQuery) -> bool {
    match query.target_population() {
        None | Some(antecedent_core::TargetPopulation::AllObserved) => true,
        Some(_) => matches!(query, CausalQuery::AverageEffect(_)),
    }
}

/// Refuse [`crate::PosteriorDrawControls`] where they would not be enforced or are not met.
///
/// The controls bound the draws behind a Bayesian g-computation interval of a tabular
/// average effect on one fixed `Dag`; any other route refuses them rather than ignore them.
/// A draw count below the backend's supported minimum is refused naming it, never rewritten.
pub(super) fn refuse_unsupported_draw_controls(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    controls: Option<crate::PosteriorDrawControls>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    if controls.is_none() {
        return Ok(());
    }
    let cfg = match inference {
        InferenceMode::Bayesian(cfg) => Some(cfg),
        InferenceMode::Frequentist => None,
    };
    let enforced = cfg.is_some()
        && matches!(query, CausalQuery::AverageEffect(_))
        && matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && matches!(estimator, None | Some(EstimatorId::BayesianGcomp));
    if !enforced {
        return Err(crate::unsupported_reason!(
            "invalid_argument",
            "posterior draw controls bound the draws behind a Bayesian g-computation interval of \
             a tabular AverageEffect on one fixed Dag; this route does not enforce them, so it \
             refuses them rather than publish an interval they do not bound"
        ));
    }
    let Some(cfg) = cfg else {
        return Ok(());
    };
    if cfg.n_draws >= antecedent_estimate::supported_min_draws(cfg.backend) {
        return Ok(());
    }
    Err(if cfg.backend == antecedent_estimate::BayesianBackendKind::Hmc {
        crate::unsupported_reason!(
            "posterior_draws_below_supported",
            "the supported minimum is 3000 posterior draws for the hmc backend, the floor its \
             chain diagnostics need; request at least 3000 draws"
        )
    } else {
        crate::unsupported_reason!(
            "posterior_draws_below_supported",
            "the supported minimum is 2000 posterior draws; request at least 2000 draws"
        )
    })
}

/// Refuse retained predictive replicates where no executor builds them or a bound is zero.
///
/// Only a tabular `AverageEffect` on one fixed `Dag` under Bayesian g-computation retains
/// replicates; every other route refuses the declaration rather than return a result that
/// silently lacks them.
pub(super) fn refuse_unsupported_predictive_replicates(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    let Some(bounds) = cfg.predictive_replicates else {
        return Ok(());
    };
    if bounds.n_sims == 0 || bounds.max_rows == 0 {
        return Err(crate::unsupported_reason!(
            "invalid_argument",
            "predictive replicates need at least one simulation (n_sims) and a positive row \
             bound (max_rows)"
        ));
    }
    let supported = matches!(query, CausalQuery::AverageEffect(_))
        && matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && matches!(estimator, None | Some(EstimatorId::BayesianGcomp));
    if supported {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "retained predictive replicates are built by Bayesian g-computation of a tabular \
         AverageEffect on one fixed Dag; this route does not build them, so it refuses the \
         declaration rather than return a result without them"
    ))
}

/// Refuse a declared white-innovation premise where no executor reads it.
///
/// The untempered posterior is the full-likelihood posterior of one lag-aligned Gaussian
/// regression, so it exists for the temporal pulse, the one-step sustained effect and the
/// response curve at the single horizon of one step ahead, on one fixed `TemporalDag` under the
/// isotropic Gaussian prior only. Every other route would ignore the
/// premise, so it refuses it instead of publish a posterior the premise did not shape.
pub(super) fn refuse_unsupported_white_innovation(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    if cfg.white_innovation.is_none() {
        return Ok(());
    }
    // A response curve reads the premise at the single horizon of one step ahead.
    let single_horizon_response = matches!(
        query,
        CausalQuery::Response(q)
            if q.temporal.as_ref().is_some_and(|spec| &*spec.horizons == [1_u32].as_slice())
    );
    let single_window = single_horizon_response
        || matches!(
            query,
            CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
        );
    let supported = single_window
        && matches!(data, DataInput::Temporal(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
        && !cfg.has_transfer()
        && cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity;
    if supported {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "a declared white-innovation premise licenses the untempered Gaussian posterior of a \
         single-step temporal pulse, one-step sustained effect or one-step single-horizon \
         response curve on one fixed TemporalDag under the isotropic Gaussian prior; this route \
         does not read it, so it refuses the premise \
         rather than publish a posterior it did not shape. Next attempt: drop the declaration to \
         use the tempered posterior, or compose per-mechanism posteriors for a longer horizon"
    ))
}

/// Refuse a selected lagged-instrument estimator where its construction does not apply.
///
/// The instrument is certified on the unfolded window graph of a single-step pulse or one-step
/// sustained effect over one series, so it needs the `temporal.iv.unfolded` identifier, one fixed
/// `TemporalDag` and series data; every other route would not read the selection.
pub(super) fn refuse_unsupported_temporal_instrument(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let selected = matches!(
        estimator,
        Some(EstimatorId::TemporalLaggedInstrument | EstimatorId::BayesianTemporalLaggedInstrument)
    );
    if !selected && identifier != Some(IdentifierId::TemporalIvUnfolded) {
        return Ok(());
    }
    let single_window = matches!(
        query,
        CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
    );
    if selected
        && identifier == Some(IdentifierId::TemporalIvUnfolded)
        && single_window
        && matches!(data, DataInput::Temporal(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
    {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "the lagged-instrument estimators and the temporal.iv.unfolded identifier go together and \
         identify a single-step pulse or one-step sustained effect over one series on one fixed \
         TemporalDag; this route does not read the selection. Next attempt: select the temporal \
         linear adjustment, or declare the instrument's graph and select both the identifier and a \
         lagged-instrument estimator"
    ))
}

/// Refuse a selected temporal GLM link contrast where its construction does not apply.
///
/// The contrast is a mean-scale difference of a Bernoulli temporal model whose rows must be
/// independent subjects, so it exists for a single-step pulse or one-step sustained effect on one
/// fixed `TemporalDag` over a panel. One dependent series has no independent-trajectory
/// information matrix; every other route would not read the selection.
pub(super) fn refuse_unsupported_temporal_link(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if !matches!(
        estimator,
        Some(EstimatorId::TemporalGlmLinkContrast | EstimatorId::BayesianTemporalGlmLinkContrast)
    ) {
        return Ok(());
    }
    if matches!(data, DataInput::Temporal(_)) {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a single dependent series has no independent-trajectory information matrix, so the \
             temporal GLM link contrast is not fitted as if its rows were iid. Next attempt: \
             supply a panel of independent subjects (one trajectory per subject), or use the \
             temporal linear adjustment, whose uncertainty accounts for serial dependence"
        ));
    }
    let single_window = matches!(
        query,
        CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
    );
    if single_window
        && matches!(data, DataInput::Panel(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
    {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "the temporal GLM link contrast is the mean-scale contrast of a single-step pulse or \
         one-step sustained effect on one fixed TemporalDag over a panel of independent \
         subjects; this route does not read the selection. Next attempt: select the temporal \
         linear adjustment or its Bayesian g-computation for other temporal effects"
    ))
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
pub(super) fn refuse_unsupported_likelihood(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    // The temporal link contrast reads the Bernoulli likelihood itself and checks it at fit.
    if estimator == Some(EstimatorId::BayesianTemporalGlmLinkContrast) {
        return Ok(());
    }
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
    let supported_query = match query {
        CausalQuery::AverageEffect(q) => {
            matches!(q.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        }
        CausalQuery::ConditionalEffect(q) => {
            matches!(q.inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        }
        CausalQuery::Response(q) => matches!(
            q.functional,
            antecedent_core::ResponseFunctional::MeanCurve { .. }
                | antecedent_core::ResponseFunctional::InterventionResponse { .. }
        ),
        _ => false,
    };
    let licensed = matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && supported_query;
    if licensed {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "likelihood_not_supported",
        "a Bernoulli or Poisson likelihood is fitted by Bayesian g-computation of a tabular \
         AverageEffect or ConditionalEffect mean, or a static response level, on one Dag only; this route fits a Gaussian identity-link model or \
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
pub(super) fn custom_validators_apply(
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
            query: None,
            discovery_query_error: None,
            bootstrap_interval: None,
            caller_graph: None,
            declared: Declarations::default(),
            exposure: ExposureInputs::default(),
            settings: InferenceSettings::omitted(Self::OMITTED_REFUTE, Self::OMITTED_BOOTSTRAP),
            execution: ExecutionSettings::default(),
            discovery: None,
            mechanisms: None,
            posterior_provenance: None,
            data_diagnostics: Some(crate::DataDiagnostics::default()),
        }
    }

    /// Search for causal structure over `variables` with `spec`.
    ///
    /// Sets the study's query to the [`CausalQuery::Discover`] query `spec`
    /// states and retains the executable search. A discovery study takes no
    /// graph: the discovered structure (a DAG, a CPDAG or PAG that stays a
    /// class, a temporal class, one class per regime, or a structure
    /// posterior) is the answer. It reaches a downstream study only through
    /// [`crate::accept_discovered`].
    #[must_use]
    pub fn discovery(
        mut self,
        spec: super::structure_route::DiscoverySpec,
        variables: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        match spec.query(variables) {
            Ok(query) => {
                self.query = Some(CausalQuery::Discover(query));
                self.discovery_query_error = None;
            }
            Err(error) => {
                self.query = None;
                self.discovery_query_error = Some(error);
            }
        }
        self.discovery = Some(Arc::new(spec));
        self
    }

    /// Fit the mechanisms of a [`CausalQuery::Attribute`] or
    /// [`CausalQuery::Simulate`] query with `program`. Without one the route
    /// fits the standard registry by best held-out score.
    #[must_use]
    pub fn mechanisms(mut self, program: super::structure_route::MechanismProgram) -> Self {
        self.mechanisms = Some(Arc::new(program));
        self
    }

    /// Supply a structure posterior accepted from a compiled discovery as the
    /// study's graph posterior.
    ///
    /// Behaves as [`Self::graph_posterior`] with the posterior's atoms and
    /// weights, and the study's contract and program identity carry the
    /// search's [`crate::StructureProvenance`]. The identification digest does
    /// not: the same posterior supplied by hand identifies the same way.
    #[must_use]
    pub fn accepted_graph_posterior(
        mut self,
        accepted: super::structure_route::AcceptedPosterior,
    ) -> Self {
        let (posterior, provenance) = accepted.into_parts();
        self.graph_posterior = Some(posterior);
        self.posterior_provenance = Some(provenance);
        self
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

    /// Identify the query against the structure, reading no rows.
    ///
    /// Build the study over a table with a schema and no observations to identify before
    /// any data exist (the staged identify-then-estimate workflow): identification is a
    /// function of structure and query, so the answer is the one a study over the full table
    /// derives. The study is assembled as [`Self::inspect`] assembles it, so a coordinate the
    /// support matrix closes still identifies (its estimation stays refused at
    /// [`Self::build`]). Returns the point result of a DAG or ADMG, or the envelope over the
    /// completions of a CPDAG, PAG or temporal class.
    ///
    /// # Errors
    ///
    /// Missing graph or query, a schema mismatch, a graph-posterior study, an unsupported
    /// graph-class/query pair, or identification failure.
    pub fn identify_structure(self) -> Result<crate::Identification, CausalError> {
        self.finish(true)?.identify_structure()
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
        self.declared.structure_source = Some(source);
        self.caller_graph = Some(CallerGraph);
        self
    }

    /// Declare a tier order over existing ADMG / PAG semantics.
    ///
    /// `CoDetermined` certifies the tier-closure set in `O(p)` as
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
        self.declared.structure_source = Some(crate::support::StructureSource::Explicit);
        self.declared.tiered = Some(background);
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
        self.declared.class_prior = Some(prior);
        self
    }

    /// Cap `TemporalCpdag` / `TemporalPag` completion search.
    ///
    /// A cap below the true class size cannot confer class-wide point
    /// identification (`full_mass_scope`, `truncated_atoms`).
    #[must_use]
    pub fn max_completions(mut self, n: usize) -> Self {
        self.declared.max_completions = Some(n);
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
        if self.declared.estimator_spec.is_none()
            && matches!(
                self.declared.estimator,
                Some(EstimatorId::BayesianGcomp | EstimatorId::BayesianBasisGcomp)
            )
        {
            // Only an average effect keeps the Bayesian g-computation
            // placeholder; a conditional effect resolves its own Bayesian
            // default, matching the query-then-inference call order.
            match &q {
                CausalQuery::AverageEffect(_) => {}
                CausalQuery::TemporalEffect(_) => {
                    self.declared.estimator = Some(EstimatorId::TemporalLinearAdjustment);
                }
                _ => {
                    // Remainder cells keep their identifier-native estimators.
                    self.declared.estimator = None;
                }
            }
        }
        self.query = Some(q);
        self.discovery_query_error = None;
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
        self.declared.split = Some(split);
        self
    }

    /// Configure refutation suite (static path).
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.settings.refute = suite;
        self.settings.refute_explicit = true;
        self
    }

    /// Bootstrap replicates for the primary estimate.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.settings.bootstrap_replicates = n;
        self.settings.bootstrap_explicit = true;
        self
    }

    /// Request a bootstrap interval produced by the complete resampling procedure: waves of
    /// replicates that stop once the request's Monte Carlo tolerances on the standard error and
    /// on the interval half-width are met, capped by [`Self::bootstrap_replicates`], which must
    /// be set explicitly as that cap. The result's diagnostics carry the interval, every wave,
    /// the Monte Carlo errors and the stop reason
    /// ([`antecedent_estimate::EstimateDiagnostics::resampling`]).
    ///
    /// The selected estimator must be linear adjustment, propensity weighting, propensity
    /// stratification, AIPW or GLM adjustment; a bare [`EstimatorId`] is built with the study's
    /// settings, a configured estimator takes the request. Anything else is refused at
    /// [`Self::build`] ([`CausalError::Conflict`]). The coverage of the stopping procedure is a
    /// measured property of the request, not asserted: the estimate's calibration key carries
    /// the procedure so a fixed-count calibration never binds to it.
    #[must_use]
    pub fn bootstrap_interval(mut self, request: antecedent_estimate::IntervalRequest) -> Self {
        self.bootstrap_interval = Some(request);
        self
    }

    /// Retain the joint replicates of a temporal response surface, so horizons can be composed
    /// from realizations coupled across them (see
    /// [`crate::analysis::horizon_composition`]).
    ///
    /// Under Frequentist inference the replicates are the joint circular-block bootstrap
    /// replicates of the whole dose-by-horizon surface and need `bootstrap_replicates`; under
    /// Bayesian inference they are the draws of the block-weight posterior over the surface
    /// (one Exp(1) weight per block of consecutive times shared by every horizon), which
    /// replaces the per-horizon conjugate fits, whose draws are independent marginals. The
    /// option changes the contract's inference binding, and a Bayesian run under it is
    /// unlicensed.
    #[must_use]
    pub fn retain_joint_horizons(mut self, on: bool) -> Self {
        self.settings.retain_joint_horizons = on;
        self
    }

    /// Latency tier (`Interactive` / `Standard` / `Report`).
    ///
    /// Maps to known-equivalent bootstrap / refute / draw budgets. Explicit
    /// [`Self::bootstrap_replicates`], [`Self::refute`], and [`Self::compute_budget`]
    /// field overrides always win.
    #[must_use]
    pub fn latency_mode(mut self, mode: LatencyMode) -> Self {
        self.execution.latency_mode = Some(mode);
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
            self.settings.bootstrap_explicit = true;
        }
        if budget.validators.is_some() {
            self.settings.refute_explicit = true;
        }
        if budget.n_draws.is_some() {
            self.settings.n_draws_explicit = true;
        }
        self.execution.compute_budget = budget;
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
        self.declared.identifier = Some(id);
        self
    }

    /// Bound the Bayesian posterior draws behind the published interval.
    ///
    /// See [`crate::PosteriorDrawControls`]: the draw count must reach the backend's
    /// supported minimum, and the interval is refused when its endpoint Monte Carlo
    /// error is too large. Only a tabular `AverageEffect` on a fixed `Dag` under
    /// Bayesian g-computation enforces them; any other route refuses them at build.
    #[must_use]
    pub fn draw_controls(mut self, controls: crate::PosteriorDrawControls) -> Self {
        self.declared.draw_controls = Some(controls);
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
        self.declared.continuous_cell = Some((variable, grid.into()));
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
        self.declared.estimator = Some(spec.id());
        self.declared.estimator_spec = Some(spec);
        self
    }

    /// Configure numerical and fixed-grid uncertainty options for response queries.
    ///
    /// A simultaneous band requires both `simultaneous_replicates` and an explicit
    /// bandwidth; the estimator refuses an implicit undersmoothing rule.
    #[must_use]
    pub fn response_options(mut self, options: ContinuousResponseOptions) -> Self {
        self.declared.response_options = Some(options);
        self
    }

    /// Explicit numerical options for the observation correction on response curves.
    #[must_use]
    pub fn observation_options(
        mut self,
        options: antecedent_estimate::ObservationEstimatorOptions,
    ) -> Self {
        self.declared.observation_options = options;
        self
    }

    /// Delayed-entry column for marginal right-censoring IPCW.
    #[must_use]
    pub fn observation_delayed_entry(mut self, variable: antecedent_core::VariableId) -> Self {
        self.declared.observation_delayed_entry = Some(variable);
        self
    }

    /// Selection-node targets for [`CausalQuery::Transport`] (empty = Direct formula).
    #[must_use]
    pub fn selection_targets(mut self, targets: impl Into<Arc<[VariableId]>>) -> Self {
        self.exposure.selection_targets = Some(targets.into());
        self
    }

    /// Trial membership and known probabilities for [`CausalQuery::Transport`].
    #[must_use]
    pub fn transport_trial(mut self, spec: TransportTrialSpec) -> Self {
        self.exposure.transport_trial = Some(spec);
        self
    }

    /// Fixed network and realized assignment for [`CausalQuery::Interference`].
    #[must_use]
    pub fn interference(mut self, spec: InterferenceSpec) -> Self {
        self.exposure.interference = Some(spec);
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
        if matches!(mode, InferenceMode::Bayesian(_))
            && self.declared.estimator_spec.is_none()
            && !matches!(self.declared.estimator, Some(EstimatorId::BayesianBasisGcomp))
        {
            match &self.query {
                None | Some(CausalQuery::AverageEffect(_)) => {
                    self.declared.estimator = Some(EstimatorId::BayesianGcomp);
                }
                Some(CausalQuery::TemporalEffect(_)) => {}
                Some(_) => {
                    if matches!(self.declared.estimator, Some(EstimatorId::BayesianGcomp)) {
                        self.declared.estimator = None;
                    }
                }
            }
        }
        if matches!(mode, InferenceMode::Frequentist)
            && self.declared.estimator_spec.is_none()
            && matches!(
                self.declared.estimator,
                Some(EstimatorId::BayesianGcomp | EstimatorId::BayesianBasisGcomp)
            )
        {
            self.declared.estimator = None;
        }
        self.settings.inference = mode;
        self
    }

    /// Declare serially independent innovations on the Bayesian inference already set.
    ///
    /// Equivalent to [`crate::BayesianConfig::white_innovation`]; it leaves a frequentist
    /// study unchanged, and a route that does not read the premise refuses it at build.
    #[must_use]
    pub fn bayesian_white_innovation(
        mut self,
        gate: antecedent_estimate::white_innovation_posterior::InnovationWhitenessGate,
    ) -> Self {
        if let InferenceMode::Bayesian(cfg) = &mut self.settings.inference {
            cfg.white_innovation = Some(gate);
        }
        self
    }

    /// Retain the posterior predictive replicates of the Bayesian fit, bounded by `n_sims`
    /// draws and `max_rows` design rows.
    ///
    /// The study builds one predictive sample law per design row at fit time, admitted
    /// against the Execution memory pool and returned on the result for predictive
    /// assessment ([`crate::StudyResult::predictive_replicates`]). It leaves a frequentist
    /// study unchanged; a route that does not retain them, a zero bound, or a design with
    /// more than `max_rows` rows is refused rather than silently unretained.
    #[must_use]
    pub fn bayesian_predictive_replicates(mut self, n_sims: u32, max_rows: u32) -> Self {
        if let InferenceMode::Bayesian(cfg) = &mut self.settings.inference {
            cfg.predictive_replicates =
                Some(crate::inference::PredictiveReplicateBounds { n_sims, max_rows });
        }
        self
    }

    /// Declare the likelihood and prior of a randomized-design Bayesian estimator.
    ///
    /// Equivalent to [`crate::BayesianConfig::design_model`]; it leaves a frequentist study
    /// unchanged, and a route that does not read the declaration ignores nothing silently: the
    /// estimators that need it refuse its absence.
    #[must_use]
    pub fn bayesian_design_model(mut self, model: crate::DesignPosteriorModel) -> Self {
        if let InferenceMode::Bayesian(cfg) = &mut self.settings.inference {
            cfg.design_model = Some(model);
        }
        self
    }

    /// Overlap / positivity policy for propensity and AIPW estimators.
    ///
    /// When unset, those estimators keep their built-in defaults (clip = 0.01, no trim).
    /// Ignored by estimators that require [`OverlapPolicy::ExplicitOverride`] (linear, GLM, IV,
    /// front-door, RD).
    #[must_use]
    pub fn overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.declared.overlap_policy = Some(policy);
        self
    }

    /// Bind participant labels to the rows of the data. The binding enters the data-snapshot
    /// identity only when set; it labels rows and changes no estimate.
    #[must_use]
    pub fn participants(mut self, binding: antecedent_io::ParticipantBindingWire) -> Self {
        self.declared.participants = Some(binding);
        self
    }

    /// Declare how missing values are handled.
    ///
    /// Without a declaration, average-effect routes keep their complete-case selection and
    /// report the rows it dropped, while randomized, panel and temporal routes refuse any
    /// missing value. A declaration enters the data-snapshot identity only when a route
    /// would behave differently for it.
    #[must_use]
    pub fn missing(mut self, declaration: antecedent_data::MissingDeclaration) -> Self {
        self.declared.missing = Some(declaration);
        self
    }

    /// Declare how the rows were sampled, once for the whole study.
    ///
    /// Independent subject trajectories refuse a row-level bootstrap, a clustered panel is
    /// resampled and split by whole cluster, and a dependent time series is resampled in
    /// contiguous blocks and split with an embargo. Every estimator of the study that
    /// bootstraps or splits is held to the declared design, and a request the design does not
    /// allow is refused with a typed sampling-design error. The declaration enters the
    /// data-snapshot identity. Without it each estimator takes the design its grouping implies.
    #[must_use]
    pub fn sampling_design(mut self, design: antecedent_core::DeclaredSamplingDesign) -> Self {
        self.declared.sampling_design = Some(design);
        self
    }

    /// Declare the cluster labels or the serial kernel that the variance of a score-table
    /// route must read.
    ///
    /// The declaration pairs with [`Self::sampling_design`]: cluster labels (one per row of the
    /// table) go with a clustered panel and a Newey-West lag with a dependent time series, rows
    /// taken in table order as unit-spaced time. Only the joint-cell response reads it, as the
    /// cluster-robust or Newey-West variance of its influence values; any other route refuses it
    /// instead of ignoring it. It enters the data-snapshot identity.
    #[must_use]
    pub fn score_dependence(mut self, dependence: antecedent_estimate::ScoreDependence) -> Self {
        self.declared.score_dependence = Some(dependence);
        self
    }

    /// Bindings for named predicates and custom target-distribution weights.
    #[must_use]
    pub fn population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.declared.population_registry = Some(registry);
        self
    }

    /// Append custom effect validators ( slow path).
    #[must_use]
    pub fn custom_validators(mut self, validators: Vec<Arc<dyn CustomEffectValidator>>) -> Self {
        self.declared.custom_validators = validators;
        self
    }

    /// Configure the running variable / cutoff / bandwidth required by the `rd.sharp`
    /// estimator. `compile` refuses `rd.sharp` without this.
    #[must_use]
    pub fn rd_config(mut self, running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        self.declared.rd = Some(RdConfig::new(running_variable, cutoff, bandwidth));
        self
    }

    /// Full `rd.sharp` design, including the analytic SE kind
    /// ([`RdConfig::with_se_kind`]). Replaces any earlier [`Self::rd_config`].
    #[must_use]
    pub fn rd_design(mut self, config: RdConfig) -> Self {
        self.declared.rd = Some(config);
        self
    }

    /// Stream intermediate stage payloads (identify → point → uncertainty → validate).
    ///
    /// Final [`super::execute::Study::run`] still returns the complete result.
    #[must_use]
    pub fn stage_sink(mut self, sink: Arc<dyn super::stage::StageResultSink>) -> Self {
        self.execution.stage_sink = Some(sink);
        self
    }

    /// Configure automatic fit-free diagnostics; `None` disables collection.
    #[must_use]
    pub fn data_diagnostics(mut self, policy: Option<crate::DataDiagnostics>) -> Self {
        self.data_diagnostics = policy;
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

    /// Hand a requested bootstrap interval to the selected estimator. An id-only estimator is
    /// built with the study's settings first, so those settings are then the estimator's own and
    /// the builder-level replicate and overlap settings no longer conflict with it.
    fn apply_bootstrap_interval(&mut self) -> Result<(), CausalError> {
        let Some(request) = self.bootstrap_interval.take() else {
            return Ok(());
        };
        let Some(spec) = self.declared.estimator_spec.take() else {
            return Err(CausalError::Conflict {
                what: "bootstrap_interval",
                detail: "name the estimator that produces the interval (estimator(..)); the \
                         default estimator is chosen from the graph and query at compile time",
            });
        };
        let was_default = !spec.is_configured();
        if was_default && !self.settings.bootstrap_explicit {
            return Err(CausalError::Conflict {
                what: "bootstrap_interval",
                detail: "set bootstrap_replicates(..) explicitly: it is the replicate cap of the \
                         requested interval",
            });
        }
        let defaults = crate::estimator_spec::BuildDefaults {
            bootstrap_replicates: self.settings.bootstrap_replicates,
            overlap_policy: self.declared.overlap_policy,
            population_registry: self.declared.population_registry.as_ref(),
            sampling_design: self.declared.sampling_design,
        };
        self.declared.estimator_spec = Some(spec.with_bootstrap_interval(request, defaults)?);
        if was_default {
            self.settings.bootstrap_explicit = false;
            self.declared.overlap_policy = None;
        }
        Ok(())
    }

    fn finish(mut self, inspect_only: bool) -> Result<Study, CausalError> {
        if let Some(error) = self.discovery_query_error.take() {
            return Err(CausalError::from(error));
        }
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if let Some(policy) =
                cfg.external_compose.as_ref().and_then(|external| external.conflict_policy)
            {
                policy.validate().map_err(CausalError::from)?;
            }
        }
        self.apply_bootstrap_interval()?;
        if let (Some(CausalQuery::SyntheticControl(query)), InferenceMode::Frequentist) =
            (self.query.as_mut(), &self.settings.inference)
        {
            // A declared serial or clustered design on the single treated unit is honoured by
            // the interval built for it, not ignored.
            query.honor_declared_dependence(self.declared.sampling_design);
        }
        let discovery = self.discovery.take();
        let mechanisms = self.mechanisms.take();
        let posterior_provenance = self.posterior_provenance.take();
        let mut study = finish::assemble(finish::FinishInput {
            inspect_only,
            data: self.data,
            graph: self.graph,
            graph_posterior: self.graph_posterior,
            query: self.query,
            caller_graph: self.caller_graph.is_some(),
            declared: self.declared,
            exposure: self.exposure,
            settings: self.settings,
            execution: self.execution,
        })?;
        super::structure_route::attach_specifications(
            &mut study,
            discovery,
            mechanisms,
            inspect_only,
        )?;
        if let Some(provenance) = posterior_provenance {
            if study.graph_posterior.is_some() {
                Arc::make_mut(&mut study.config).posterior_provenance = Some(provenance);
            }
        }
        Arc::make_mut(&mut study.config).data_diagnostics = self.data_diagnostics;
        Ok(study)
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
        match result.map_err(CausalError::into_peeled) {
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
        match result.map_err(CausalError::into_peeled) {
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

    fn conflict_of(result: Result<Study, CausalError>) -> &'static str {
        match result.map_err(CausalError::into_peeled) {
            Err(CausalError::Conflict { what, .. }) => what,
            other => panic!("expected CausalError::Conflict, got {other:?}"),
        }
    }

    #[test]
    fn a_requested_interval_needs_a_named_estimator_and_an_explicit_replicate_cap() {
        use antecedent_estimate::IntervalRequest;
        let request = IntervalRequest::default();
        let unnamed = toy_builder().bootstrap_replicates(100).bootstrap_interval(request).build();
        assert_eq!(conflict_of(unnamed), "bootstrap_interval");
        let uncapped = toy_builder()
            .estimator(crate::strategy_table::EstimatorId::LinearAdjustmentAte)
            .bootstrap_interval(request)
            .build();
        assert_eq!(conflict_of(uncapped), "bootstrap_interval");
        // An estimator whose bootstrap is not the wave procedure refuses the request.
        let matching = toy_builder()
            .estimator(crate::strategy_table::EstimatorId::PropensityMatching)
            .bootstrap_replicates(100)
            .bootstrap_interval(request)
            .build();
        assert_eq!(conflict_of(matching), "bootstrap_interval");
        // A configured estimator takes the request; the builder's own replicate setting would
        // still conflict with it.
        let configured = toy_builder()
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(300))
            .bootstrap_interval(request)
            .build();
        assert!(!matches!(configured, Err(CausalError::Conflict { .. })));
        let doubled = toy_builder()
            .estimator(LinearAdjustmentAte::new().with_bootstrap_interval(request))
            .bootstrap_interval(IntervalRequest { level: 0.9, ..request })
            .build();
        assert_eq!(conflict_of(doubled), "bootstrap_interval");
    }

    #[test]
    fn a_requested_interval_on_a_named_estimator_becomes_that_estimators_own_setting() {
        use antecedent_estimate::IntervalRequest;
        let request = IntervalRequest { level: 0.9, ..IntervalRequest::default() };
        let spec = EstimatorSpec::Default(crate::strategy_table::EstimatorId::Aipw)
            .with_bootstrap_interval(
                request,
                crate::estimator_spec::BuildDefaults {
                    bootstrap_replicates: 150,
                    overlap_policy: None,
                    population_registry: None,
                    sampling_design: None,
                },
            )
            .unwrap();
        assert_eq!(spec.bootstrap_interval(), Some(request));
        assert_eq!(spec.bootstrap_replicates(), Some(150));
        assert_eq!(spec.bootstrap_interval_calibration_label(), Some("bootstrap=adaptive_waves"));
        // The request is part of the estimator's identity.
        let plain = EstimatorSpec::from(antecedent_estimate::AipwAte::new());
        assert_ne!(
            crate::analysis::contract_identity::estimator_spec_identity(&spec),
            crate::analysis::contract_identity::estimator_spec_identity(&plain)
        );
    }
}
