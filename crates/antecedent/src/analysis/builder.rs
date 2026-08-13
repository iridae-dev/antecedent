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
    AverageEffectQuery, CausalQuery, CausalSchema, PopulationRegistry, TemporalEffectQuery,
    VariableId,
};
use antecedent_data::{
    DiscoveryEstimationSplit, EventData, MultiEnvironmentData, PanelData, TableView, TabularData,
    TimeSeriesData,
};
use antecedent_discovery::GraphPosterior;
use antecedent_estimate::{ContinuousResponseOptions, OverlapPolicy};
use antecedent_graph::{Admg, Cpdag, Dag, Pag, TemporalDag};
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
}

impl RdConfig {
    /// Construct an RD design configuration.
    #[must_use]
    pub const fn new(running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        Self { running_variable, cutoff, bandwidth }
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
fn stub_accepted_graph_for(data: &DataInput, n_vars: usize) -> Result<AcceptedGraph, CausalError> {
    match data {
        DataInput::Tabular(_) => {
            let n = u32::try_from(n_vars).map_err(|_| CausalError::Compile {
                message: "too many variables for graph-posterior stub graph".into(),
            })?;
            Ok(AcceptedGraph::dag(Dag::with_variables(n)))
        }
        DataInput::Temporal(_) | DataInput::Event(_) => {
            Ok(AcceptedGraph::temporal_dag(TemporalDag::empty()))
        }
        DataInput::MultiEnv(_) | DataInput::Panel(_) => Err(CausalError::Unsupported {
            message: "graph-posterior analysis supports tabular or temporal/event data only",
        }),
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
    /// Optional latency tier (maps to known-equivalent budgets unless overridden).
    latency_mode: Option<LatencyMode>,
    /// Optional field-level compute budget overrides.
    compute_budget: ComputeBudget,
    /// Optional progressive stage-result sink (Identify → Point → Uncertainty → Validate).
    stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
}

impl std::fmt::Debug for StudyBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StudyBuilder")
            .field("data", &"<data>")
            .field("graph", &self.graph)
            .field("graph_posterior", &self.graph_posterior)
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
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("n_draws_explicit", &self.n_draws_explicit)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("custom_validators", &self.custom_validators.len())
            .field("latency_mode", &self.latency_mode)
            .field("compute_budget", &self.compute_budget)
            .field("stage_sink_is_some", &self.stage_sink.is_some())
            .finish()
    }
}

impl StudyBuilder {
    fn from_data(data: DataInput) -> Self {
        Self {
            data,
            graph: None,
            graph_posterior: None,
            query: None,
            refute: RefuteSuite::PlaceboAndRcc,
            refute_explicit: false,
            bootstrap_replicates: 50,
            bootstrap_explicit: false,
            split: None,
            identifier: None,
            estimator: None,
            estimator_spec: None,
            response_options: None,
            rd: None,
            inference: InferenceMode::Frequentist,
            n_draws_explicit: false,
            overlap_policy: None,
            population_registry: None,
            custom_validators: Vec::new(),
            latency_mode: None,
            compute_budget: ComputeBudget::new(),
            stage_sink: None,
        }
    }

    /// Supply the causal structure. Accepts an [`AcceptedGraph`] directly, or any type
    /// with `impl Into<AcceptedGraph>` — [`antecedent_graph::Dag`], [`antecedent_graph::Admg`],
    /// [`antecedent_graph::Pag`], [`antecedent_graph::TemporalDag`], and
    /// [`antecedent_graph::TemporalPag`] all convert infallibly. A [`antecedent_graph::Cpdag`]
    /// or [`antecedent_graph::TemporalCpdag`] is not `Into<AcceptedGraph>` (they can carry
    /// unresolved marks) — build one via the fallible [`AcceptedGraph::cpdag`] /
    /// [`AcceptedGraph::temporal_cpdag`] first.
    #[must_use]
    pub fn graph(mut self, structure: impl Into<AcceptedGraph>) -> Self {
        self.graph = Some(structure.into());
        self
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
        if matches!(q, CausalQuery::TemporalEffect(_)) {
            // Bayesian inference must not force the static BayesianGcomp estimator on temporal.
            if matches!(self.estimator, Some(EstimatorId::BayesianGcomp)) {
                self.estimator = Some(EstimatorId::TemporalLinearAdjustment);
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

    /// Configure frequentist vs Bayesian inference.
    ///
    /// For static ATE, [`InferenceMode::Bayesian`] selects estimator [`EstimatorId::BayesianGcomp`].
    /// Temporal queries keep [`EstimatorId::TemporalLinearAdjustment`]; Bayesian mode is applied
    /// at execute time on the lag-aligned design.
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_))
            && !matches!(self.query, Some(CausalQuery::TemporalEffect(_)))
        {
            self.estimator = Some(EstimatorId::BayesianGcomp);
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
        self.rd = Some(RdConfig { running_variable, cutoff, bandwidth });
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
        let data = self.data;
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
                let stub = stub_accepted_graph_for(&data, gp.n_vars)?;
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

        Ok(Study {
            data,
            graph,
            graph_posterior,
            query: self.query.ok_or(CausalError::Missing { field: "query" })?,
            refute,
            bootstrap_replicates,
            split: self.split,
            identifier: self.identifier,
            estimator: self.estimator,
            estimator_spec: self.estimator_spec,
            response_options: self.response_options,
            rd: self.rd,
            inference,
            overlap_policy: self.overlap_policy,
            population_registry: self.population_registry,
            custom_validators: self.custom_validators,
            latency_mode,
            stage_sink: self.stage_sink,
        })
    }
}

impl Study {
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
