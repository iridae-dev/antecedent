// SPDX-License-Identifier: MIT OR Apache-2.0
// Free functions supporting Study execute paths.

use antecedent_core::StreamDomain;

pub(crate) fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), CausalError> {
    match query {
        CausalQuery::Counterfactual(q) => {
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "counterfactual missing outcome".into(),
            })?;
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).unwrap_or(outcome);
            Ok((treatment, outcome))
        }
        CausalQuery::AnomalyAttribution(q) => {
            let outcome = first_target(&q.targets, "anomaly attribution")?;
            Ok((outcome, outcome))
        }
        CausalQuery::ChangeAttribution(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::MechanismChange(q) => {
            let outcome = first_target(&q.targets, "mechanism change")?;
            Ok((outcome, outcome))
        }
        CausalQuery::UnitChange(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::Attribute(q) => {
            let outcome = q.outcome()?;
            Ok((outcome, outcome))
        }
        CausalQuery::Simulate(q) => {
            // An unintervened sample names no variable; the record names the
            // model's first one, as a plain observational draw reads them all.
            let (treatment, outcome) = q.endpoints();
            let first = VariableId::from_raw(0);
            Ok((treatment.or(outcome).unwrap_or(first), outcome.or(treatment).unwrap_or(first)))
        }
        _ => Err(CausalError::Compile { message: "gcm_query_vars: unsupported query".into() }),
    }
}

/// Whether a graph-posterior atom is a CPDAG or PAG equivalence class (as opposed to a single
/// DAG or ADMG), which routes through the class-envelope executors.
pub(crate) const fn is_class_atom(kind: antecedent_discovery::GraphPosteriorAtomKind) -> bool {
    matches!(
        kind,
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag
    )
}

/// The first declared target of a GCM query. An empty target list is refused rather than read
/// as variable 0, which would silently pick the first schema variable.
fn first_target(
    targets: &[VariableId],
    query_kind: &'static str,
) -> Result<VariableId, CausalError> {
    targets.first().copied().ok_or_else(|| CausalError::Compile {
        message: [query_kind, " query declares no target variable"].concat(),
    })
}

/// Named execute / compile cell. Graph-completion arms still specialize on class.
#[derive(Clone, Copy)]
pub(crate) enum AnalysisRoute {
    Response,
    TemporalResponse,
    StaticAte,
    Distribution,
    PathSpecific,
    Conditional,
    TemporalMediation,
    StaticMediation,
    Counterfactual,
    Anomaly,
    ChangeAttribution,
    MechanismChange,
    UnitChange,
    TemporalEffect,
    TemporalCrossWorld,
    PanelTemporalEffect,
    PanelTemporalResponse,
    MultiEnvTemporalEffect,
    Transport,
    Interference,
    RandomizedEffect,
    PolicyValue,
    ContinuousDoseResponse,
    PanelDid,
    SyntheticControl,
    LocalPolynomialRatio,
    Survival,
    LongitudinalRegime,
    SequentialContrast,
    HistoryMatch,
    Discover,
    Attribute,
    Simulate,
    TemporalDesign,
    IvBounds,
}

#[derive(Clone, Copy)]
pub(super) enum DataModality {
    Tabular,
    TemporalOrEvent,
    Panel,
    MultiEnv,
}

pub(super) fn data_modality(data: &DataInput) -> DataModality {
    match data {
        DataInput::Tabular(_) => DataModality::Tabular,
        DataInput::Temporal(_) | DataInput::Event(_) => DataModality::TemporalOrEvent,
        DataInput::Panel(_) => DataModality::Panel,
        DataInput::MultiEnv(_) => DataModality::MultiEnv,
    }
}

pub(super) fn classify_analysis_route(
    data: &DataInput,
    query: &CausalQuery,
) -> Option<AnalysisRoute> {
    classify_route(data_modality(data), query)
}

#[allow(
    clippy::match_same_arms,
    reason = "one arm per (modality, query kind) pair so the routing table reads pair by pair"
)]
pub(super) fn classify_route(modality: DataModality, query: &CausalQuery) -> Option<AnalysisRoute> {
    Some(match (modality, query) {
        (DataModality::Tabular, CausalQuery::Response(q)) if q.is_temporal() => return None,
        (DataModality::Tabular, CausalQuery::Response(_)) => AnalysisRoute::Response,
        (DataModality::TemporalOrEvent, CausalQuery::Response(q)) if q.is_temporal() => {
            AnalysisRoute::TemporalResponse
        }
        (DataModality::Tabular, CausalQuery::AverageEffect(_)) => AnalysisRoute::StaticAte,
        (DataModality::Tabular, CausalQuery::Distribution(_)) => AnalysisRoute::Distribution,
        (DataModality::TemporalOrEvent, CausalQuery::Distribution(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::Tabular, CausalQuery::PathSpecific(_)) => AnalysisRoute::PathSpecific,
        (DataModality::TemporalOrEvent, CausalQuery::PathSpecific(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::TemporalOrEvent, CausalQuery::NestedCounterfactual(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::TemporalOrEvent, CausalQuery::Counterfactual(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::Tabular, CausalQuery::ConditionalEffect(_)) => AnalysisRoute::Conditional,
        (DataModality::TemporalOrEvent, CausalQuery::Mediation(_)) => {
            AnalysisRoute::TemporalMediation
        }
        (DataModality::Tabular, CausalQuery::Mediation(_)) => AnalysisRoute::StaticMediation,
        (DataModality::Tabular, CausalQuery::NestedCounterfactual(_)) => {
            AnalysisRoute::StaticMediation
        }
        (DataModality::Tabular, CausalQuery::Counterfactual(_)) => AnalysisRoute::Counterfactual,
        (DataModality::Tabular, CausalQuery::AnomalyAttribution(_)) => AnalysisRoute::Anomaly,
        (DataModality::Tabular, CausalQuery::ChangeAttribution(_)) => {
            AnalysisRoute::ChangeAttribution
        }
        (DataModality::TemporalOrEvent, CausalQuery::AnomalyAttribution(_)) => {
            AnalysisRoute::Anomaly
        }
        (DataModality::TemporalOrEvent, CausalQuery::ChangeAttribution(_)) => {
            AnalysisRoute::ChangeAttribution
        }
        (DataModality::Tabular | DataModality::TemporalOrEvent, CausalQuery::MechanismChange(_)) => {
            AnalysisRoute::MechanismChange
        }
        (DataModality::Tabular | DataModality::TemporalOrEvent, CausalQuery::UnitChange(_)) => {
            AnalysisRoute::UnitChange
        }
        (DataModality::TemporalOrEvent, CausalQuery::TemporalEffect(_)) => {
            AnalysisRoute::TemporalEffect
        }
        (DataModality::Panel, CausalQuery::TemporalEffect(_)) => AnalysisRoute::PanelTemporalEffect,
        (DataModality::Panel, CausalQuery::Response(q)) if q.is_temporal() => {
            AnalysisRoute::PanelTemporalResponse
        }
        (DataModality::MultiEnv, CausalQuery::TemporalEffect(_)) => {
            AnalysisRoute::MultiEnvTemporalEffect
        }
        (DataModality::Tabular, CausalQuery::Transport(_)) => AnalysisRoute::Transport,
        (DataModality::Tabular, CausalQuery::Interference(_)) => AnalysisRoute::Interference,
        (DataModality::Tabular, CausalQuery::RandomizedEffect(_)) => {
            AnalysisRoute::RandomizedEffect
        }
        (DataModality::Tabular, CausalQuery::PolicyValue(_)) => AnalysisRoute::PolicyValue,
        (DataModality::Tabular, CausalQuery::ContinuousDoseResponse(_)) => {
            AnalysisRoute::ContinuousDoseResponse
        }
        (DataModality::Tabular, CausalQuery::PanelDid(_)) => AnalysisRoute::PanelDid,
        (DataModality::Tabular, CausalQuery::SyntheticControl(_)) => {
            AnalysisRoute::SyntheticControl
        }
        (DataModality::Tabular, CausalQuery::LocalPolynomialRatio(_)) => {
            AnalysisRoute::LocalPolynomialRatio
        }
        (DataModality::Tabular, CausalQuery::Survival(_)) => AnalysisRoute::Survival,
        (DataModality::Tabular, CausalQuery::LongitudinalRegime(_)) => {
            AnalysisRoute::LongitudinalRegime
        }
        (DataModality::Tabular, CausalQuery::HistoryMatch(_)) => AnalysisRoute::HistoryMatch,
        (DataModality::Tabular, CausalQuery::SequentialContrast(_)) => {
            AnalysisRoute::SequentialContrast
        }
        // Every modality: the method decides which data it searches, and
        // admission refuses a method on data it does not run on.
        (_, CausalQuery::Discover(_)) => AnalysisRoute::Discover,
        (DataModality::Tabular | DataModality::TemporalOrEvent, CausalQuery::Attribute(_)) => {
            AnalysisRoute::Attribute
        }
        (DataModality::Tabular | DataModality::TemporalOrEvent, CausalQuery::Simulate(_)) => {
            AnalysisRoute::Simulate
        }
        (DataModality::Tabular, CausalQuery::TemporalDesign(_)) => AnalysisRoute::TemporalDesign,
        (DataModality::Tabular, CausalQuery::IvBounds(_)) => AnalysisRoute::IvBounds,
        _ => return None,
    })
}

pub(super) enum GcmSlot {
    Counterfactual(crate::gcm::IteResult),
    Anomaly(Vec<antecedent_attribution::AnomalyScores>),
    Change(antecedent_attribution::ChangeAttributionResult),
    Mechanism(Vec<antecedent_attribution::MechanismChangeDetection>),
    Unit(antecedent_attribution::UnitChangeResult),
    Attribution(Box<crate::analysis::structure_route::AttributionPayload>),
    Simulation(Box<crate::analysis::structure_route::SimulationPayload>),
}

impl GcmSlot {
    fn into_payload(self) -> crate::result::ResultPayload {
        use crate::result::ResultPayload;
        match self {
            Self::Counterfactual(value) => ResultPayload::Counterfactual(value),
            Self::Anomaly(value) => ResultPayload::Anomaly(value),
            Self::Change(value) => ResultPayload::ChangeAttribution(value),
            Self::Mechanism(value) => ResultPayload::MechanismChange(value),
            Self::Unit(value) => ResultPayload::UnitChange(value),
            Self::Attribution(value) => ResultPayload::Attribution(value),
            Self::Simulation(value) => ResultPayload::Simulation(value),
        }
    }
}

pub(super) fn provenance_ids(
    artifact: impl Into<Arc<str>>,
    op: impl Into<Arc<str>>,
) -> (Arc<str>, Arc<str>) {
    (artifact.into(), op.into())
}

pub(super) fn is_gcm_route(route: AnalysisRoute) -> bool {
    matches!(
        route,
        AnalysisRoute::Counterfactual
            | AnalysisRoute::Anomaly
            | AnalysisRoute::ChangeAttribution
            | AnalysisRoute::MechanismChange
            | AnalysisRoute::UnitChange
    )
}

pub(super) fn identify_cached_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "exec.identify.cached",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "identification reused from the prepare-time cache".to_string(),
    )
}

/// The identifier and estimator a route executed with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Strategy {
    pub(super) identifier: IdentifierId,
    pub(super) estimator: EstimatorId,
}

impl Strategy {
    pub(crate) const fn new(identifier: IdentifierId, estimator: EstimatorId) -> Self {
        Self { identifier, estimator }
    }
}

/// The execution frame every finish of one execute path shares.
#[derive(Clone, Copy)]
pub(super) struct ExecuteRun<'a> {
    pub(super) physical: &'a PhysicalExecutionPlan,
    pub(super) ctx: &'a ExecutionContext,
    pub(super) started: Instant,
    /// Whether identification was served from the prepare-time cache.
    pub(super) identify_cached: bool,
}

/// The identified class envelope a CPDAG/PAG execute path finishes from.
pub(super) struct ClassEnvelopeRun<'a, G> {
    pub(super) run: ExecuteRun<'a>,
    pub(super) envelope: &'a IdentificationEnvelope<G>,
    pub(super) envelope_diagnostic: Diagnostic,
}

/// An identification result together with the estimand selected from it.
pub(super) struct IdentifiedClaim {
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
}

pub(super) struct IdentifiedExecuteFinish<'a> {
    pub physical: &'a PhysicalExecutionPlan,
    pub identification: IdentificationResult,
    pub estimand: IdentifiedEstimand,
    pub estimate: EffectEstimate,
    pub identifier_id: IdentifierId,
    pub estimator_id: EstimatorId,
    pub treatment: VariableId,
    pub outcome: VariableId,
    pub identify_cached: bool,
    pub extra_diagnostics: Vec<Diagnostic>,
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    pub distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    pub mediation: Option<antecedent_estimate::TemporalMediationEstimate>,
    pub wall_time_ns: u64,
    pub bootstrap_replicates_ok: Option<u32>,
    pub cancelled: bool,
    pub early_stopped: bool,
    pub extras: IdentifiedExecuteExtras,
}

impl<'a> IdentifiedExecuteFinish<'a> {
    /// A finish for a freshly computed, unrefuted point result; the setters
    /// below override the slots a route fills.
    pub(super) fn new(
        physical: &'a PhysicalExecutionPlan,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        estimate: EffectEstimate,
        strategy: Strategy,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Self {
        Self {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: strategy.identifier,
            estimator_id: strategy.estimator,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: 0,
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }
    }

    pub(super) fn identify_cached(mut self, cached: bool) -> Self {
        self.identify_cached = cached;
        self
    }

    pub(super) fn extra_diagnostics(mut self, diagnostics: Vec<Diagnostic>) -> Self {
        self.extra_diagnostics = diagnostics;
        self
    }

    pub(super) fn refutations(
        mut self,
        refutations: Vec<antecedent_validate::RefutationReport>,
    ) -> Self {
        self.refutations = refutations;
        self
    }

    pub(super) fn distribution(
        mut self,
        distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    ) -> Self {
        self.distribution = distribution;
        self
    }

    pub(super) fn mediation(
        mut self,
        mediation: Option<antecedent_estimate::TemporalMediationEstimate>,
    ) -> Self {
        self.mediation = mediation;
        self
    }

    pub(super) fn wall_time_ns(mut self, wall_time_ns: u64) -> Self {
        self.wall_time_ns = wall_time_ns;
        self
    }

    pub(super) fn bootstrap_replicates_ok(mut self, replicates_ok: Option<u32>) -> Self {
        self.bootstrap_replicates_ok = replicates_ok;
        self
    }

    pub(super) fn cancelled(mut self, cancelled: bool) -> Self {
        self.cancelled = cancelled;
        self
    }

    pub(super) fn early_stopped(mut self, early_stopped: bool) -> Self {
        self.early_stopped = early_stopped;
        self
    }

    pub(super) fn extras(mut self, extras: IdentifiedExecuteExtras) -> Self {
        self.extras = extras;
        self
    }
}

