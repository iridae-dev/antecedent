//! Finite, explicitly supplied graph/selection scenarios for one transport question.
//!
//! This is not equivalence-class transport. Each scenario is a fixed selection
//! ADMG over the same named variables, decided independently by the licensed
//! classical catalog route: a bounded catalog-aware search, and, only when that
//! search certifies nothing, the classical identifier for an independently
//! verified s-hedge. A scenario's evidence binding is its own; nothing certified
//! for one scenario is reused for another. Every scenario declares the same
//! named coordinate schema (domains, cardinalities, units). One search budget
//! bounds the whole set. Declared weights are carried beside the structural
//! set and never renormalized over the scenarios that survive.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    EvidenceCatalog, ExecutionContext, NodeRef, SearchBudget, SearchLimits, SearchReceipt,
    SearchStop, Value, VariableDomain, VariableId, reason_code,
};
use antecedent_graph::SelectionDiagram;

use super::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    ClassicalTransportResult, IdentificationError, SHedgeRecord, SearchCharge, SharedSearch,
    identify_catalog_transport_metered, identify_classical_transport_metered,
};

/// Scenarios in one set.
pub const SCENARIO_MAX_COUNT: usize = 64;
/// Observed variables in each scenario graph.
pub const SCENARIO_MAX_OBSERVED: usize = 12;
/// Tolerance on declared weights summing to at most one.
pub const SCENARIO_WEIGHT_TOLERANCE: f64 = 1e-12;

/// Sum of masses independent of their order: the values are sorted, then added
/// with Neumaier compensation, so any renaming or reordering of scenarios gives
/// the same bits. Every declared-mass total of a scenario set uses this sum.
#[must_use]
#[doc(hidden)]
pub fn mass_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = sum + value;
        compensation +=
            if sum.abs() >= value.abs() { (sum - next) + value } else { (value - next) + sum };
        sum = next;
    }
    sum + compensation
}

/// Mass left over once `identified` mass is accounted for: one minus it, zero
/// when it is within [`SCENARIO_WEIGHT_TOLERANCE`] of one (the tolerance
/// `try_new` accepts weights under), never negative.
#[must_use]
#[doc(hidden)]
pub fn unaccounted_after(identified: f64) -> f64 {
    let rest = 1.0 - identified;
    if rest <= SCENARIO_WEIGHT_TOLERANCE { 0.0 } else { rest }
}

/// Detail of a scenario that a budget stop or cancellation left unevaluated.
/// The stop itself (`search.operations`, `search.depth`, `search.memory` or
/// `search.cancelled`) is carried beside it.
pub const SCENARIO_UNEVALUATED_DETAIL: &str = "scenarios.unevaluated_budget";
/// Registered runtime-refusal code of an unevaluated scenario: the same
/// `transport_budget_cancel` every 2.2 transport search reports for a budget
/// or cancellation stop, never a verdict about the scenario.
pub const SCENARIO_UNEVALUATED_CODE: &str = reason_code!("transport_budget_cancel");

/// A refused scenario set: a registered top-level reason code and a stable
/// `scenarios.*` detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct ScenarioSetRefusal {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `scenarios.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl ScenarioSetRefusal {
    fn invalid(detail: &'static str, message: impl Into<String>) -> Self {
        Self { code: reason_code!("invalid_argument"), detail, message: message.into() }
    }

    fn bound(detail: &'static str, message: impl Into<String>) -> Self {
        Self { code: reason_code!("route_not_supported"), detail, message: message.into() }
    }

    /// A scenario, law or request that disagrees with the shared coordinate schema.
    #[must_use]
    pub fn coordinate_mismatch(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("schema_mismatch"),
            detail: "scenarios.coordinate_mismatch",
            message: message.into(),
        }
    }
}

impl From<ScenarioSetRefusal> for IdentificationError {
    fn from(refusal: ScenarioSetRefusal) -> Self {
        Self::invalid_input(refusal.to_string())
    }
}

/// One named coordinate of a scenario's declared schema: the shared variable,
/// its name, value domain (with cardinality for a categorical domain) and unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenarioCoordinate {
    /// Variable id in the scenario graph.
    pub variable: VariableId,
    /// Declared variable name.
    pub name: Arc<str>,
    /// Declared value domain.
    pub domain: VariableDomain,
    /// Declared physical unit, if any.
    pub unit: Option<Arc<str>>,
}

impl ScenarioCoordinate {
    /// Whether `value` lies in the declared domain. An unspecified domain makes
    /// no claim about which numbers occur, but every domain is numeric, so a
    /// non-finite or non-numeric value belongs to no domain and is refused.
    #[must_use]
    #[allow(clippy::float_cmp)] // Domain membership is exact, not approximate equality.
    pub fn accepts(&self, value: &Value) -> bool {
        let value = value.as_f64();
        match self.domain {
            // An unspecified domain claims nothing about which finite numbers
            // occur, but a non-finite or non-numeric value is in no domain.
            VariableDomain::Unspecified | VariableDomain::Continuous => {
                value.is_some_and(f64::is_finite)
            }
            VariableDomain::Binary => value.is_some_and(|v| v == 0.0 || v == 1.0),
            VariableDomain::Count => value.is_some_and(|v| v >= 0.0 && v.fract() == 0.0),
            VariableDomain::Categorical { cardinality } => {
                value.is_some_and(|v| v >= 0.0 && v < f64::from(cardinality) && v.fract() == 0.0)
            }
        }
    }

    /// Smallest and largest value of a finite declared domain; `None` for a
    /// continuous, count or unspecified domain.
    #[must_use]
    pub fn support_limits(&self) -> Option<(f64, f64)> {
        match self.domain {
            VariableDomain::Binary => Some((0.0, 1.0)),
            VariableDomain::Categorical { cardinality } if cardinality > 0 => {
                Some((0.0, f64::from(cardinality - 1)))
            }
            _ => None,
        }
    }
}

/// One supplied scenario: a named fixed graph, the mechanisms that may differ
/// between source and target under it, and its declared coordinate schema.
#[derive(Clone, Debug)]
pub struct TransportScenario {
    /// Unique scenario name.
    pub name: Arc<str>,
    /// Causal graph over the shared variables, with this scenario's selections.
    pub diagram: SelectionDiagram,
    /// Declared weight, when the set is weighted.
    pub weight: Option<f64>,
    /// Declared coordinates: every graph variable exactly once.
    pub coordinates: Arc<[ScenarioCoordinate]>,
}

/// A validated scenario set in canonical (name) order with one shared
/// coordinate schema.
#[derive(Clone, Debug)]
pub struct TransportScenarioSet {
    scenarios: Arc<[TransportScenario]>,
    schema: Arc<[ScenarioCoordinate]>,
    weighted: bool,
}

/// A scenario's declared coordinates in variable order, checked against its graph.
fn scenario_schema(
    scenario: &TransportScenario,
) -> Result<Vec<ScenarioCoordinate>, ScenarioSetRefusal> {
    let mut schema = scenario.coordinates.to_vec();
    schema.sort_by_key(|c| c.variable);
    let nodes =
        scenario.diagram.causal_graph().nodes().iter().copied().collect::<BTreeSet<NodeRef>>();
    let declared = schema.iter().map(|c| NodeRef::Static(c.variable)).collect::<BTreeSet<_>>();
    let names = schema.iter().map(|c| c.name.clone()).collect::<BTreeSet<_>>();
    if declared != nodes
        || declared.len() != schema.len()
        || names.len() != schema.len()
        || schema.iter().any(|c| c.name.trim().is_empty())
        || schema.iter().any(|c| matches!(c.domain, VariableDomain::Categorical { cardinality: 0 }))
    {
        return Err(ScenarioSetRefusal::coordinate_mismatch(format!(
            "scenario {} must declare every graph variable exactly once with a unique \
             non-empty name and a valid domain",
            scenario.name
        )));
    }
    Ok(schema)
}

/// Name the first field on which two schemas disagree.
fn schema_difference(a: &[ScenarioCoordinate], b: &[ScenarioCoordinate]) -> String {
    if a.len() != b.len() {
        return "different variables".into();
    }
    for (x, y) in a.iter().zip(b) {
        let field = if x.variable != y.variable {
            "variables"
        } else if x.name != y.name {
            "name"
        } else if x.domain != y.domain {
            "domain or cardinality"
        } else if x.unit != y.unit {
            "unit"
        } else {
            continue;
        };
        return format!("{field} of {}", x.name);
    }
    "nothing".into()
}

impl TransportScenarioSet {
    /// Validate and canonicalize a scenario set.
    ///
    /// Every scenario shares the same observed variables and declares the same
    /// coordinate schema (names, domains and cardinalities, units), no two
    /// scenarios coincide in graph and selections, and weights are declared for
    /// all scenarios or none: finite, non-negative and summing to at most one.
    /// The undeclared remainder is kept as residual mass.
    ///
    /// # Errors
    /// A [`ScenarioSetRefusal`]: `schema_mismatch` / `scenarios.coordinate_mismatch`
    /// for disagreeing coordinates, `route_not_supported` for an exceeded bound,
    /// and `invalid_argument` otherwise.
    pub fn try_new(scenarios: Vec<TransportScenario>) -> Result<Self, ScenarioSetRefusal> {
        use ScenarioSetRefusal as R;
        if scenarios.is_empty() {
            return Err(R::invalid("scenarios.empty", "a scenario set needs a scenario"));
        }
        if scenarios.len() > SCENARIO_MAX_COUNT {
            return Err(R::bound(
                "scenarios.count",
                format!("{} scenarios exceed the bound of {SCENARIO_MAX_COUNT}", scenarios.len()),
            ));
        }
        let mut scenarios = scenarios;
        scenarios.sort_by(|a, b| a.name.cmp(&b.name));
        let first = scenarios[0].diagram.causal_graph();
        if first.node_count() > SCENARIO_MAX_OBSERVED {
            return Err(R::bound(
                "scenarios.observed_count",
                format!("more than {SCENARIO_MAX_OBSERVED} observed variables"),
            ));
        }
        // A schema declares static variables only, so a graph with any other
        // node kind fails the schema check below.
        let variables = first.nodes().iter().copied().collect::<BTreeSet<_>>();
        let schema = scenario_schema(&scenarios[0])?;
        let mut names = BTreeSet::new();
        let mut signatures = BTreeSet::new();
        for scenario in &scenarios {
            if scenario.name.trim().is_empty() || !names.insert(scenario.name.clone()) {
                return Err(R::invalid(
                    "scenarios.duplicate_or_empty_name",
                    "scenario names must be unique and non-empty",
                ));
            }
            let nodes = scenario.diagram.causal_graph().nodes();
            if nodes.len() != variables.len() || nodes.iter().any(|node| !variables.contains(node))
            {
                return Err(R::coordinate_mismatch(format!(
                    "scenario {} has different variables",
                    scenario.name
                )));
            }
            let own = scenario_schema(scenario)?;
            if own != schema {
                return Err(R::coordinate_mismatch(format!(
                    "scenarios {} and {} disagree on the {}",
                    scenarios[0].name,
                    scenario.name,
                    schema_difference(&schema, &own)
                )));
            }
            if !signatures.insert(super::graph_signature(&scenario.diagram)) {
                return Err(R::invalid(
                    "scenarios.duplicate_scenario",
                    format!("scenario {} repeats another's graph and selections", scenario.name),
                ));
            }
        }
        let weighted = scenarios[0].weight.is_some();
        let bad_weights = || {
            R::invalid(
                "scenarios.invalid_weights",
                "weights are declared for all scenarios or none, finite, non-negative and \
                 summing to at most one",
            )
        };
        if scenarios.iter().any(|s| s.weight.is_some() != weighted) {
            return Err(bad_weights());
        }
        if weighted {
            let weights = scenarios.iter().filter_map(|s| s.weight).collect::<Vec<_>>();
            let total = mass_sum(weights.iter().copied());
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                || total > 1.0 + SCENARIO_WEIGHT_TOLERANCE
            {
                return Err(bad_weights());
            }
        }
        Ok(Self { scenarios: scenarios.into(), schema: schema.into(), weighted })
    }

    /// Scenarios in canonical order.
    #[must_use]
    pub fn scenarios(&self) -> &[TransportScenario] {
        &self.scenarios
    }

    /// The shared coordinate schema, in variable order.
    #[must_use]
    pub fn schema(&self) -> &[ScenarioCoordinate] {
        &self.schema
    }

    /// The shared coordinate of `variable`.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId) -> Option<&ScenarioCoordinate> {
        self.schema.iter().find(|c| c.variable == variable)
    }

    /// Check that `value` is a declared value of `variable`: numeric, finite
    /// and inside the declared domain (an unspecified domain still refuses a
    /// non-finite or non-numeric value). No unit is checked; values carry none.
    ///
    /// # Errors
    /// `schema_mismatch` / `scenarios.coordinate_mismatch` for an undeclared
    /// variable or a value outside its declared domain.
    pub fn check_value(
        &self,
        variable: VariableId,
        value: &Value,
        context: &str,
    ) -> Result<(), ScenarioSetRefusal> {
        match self.coordinate(variable) {
            None => Err(ScenarioSetRefusal::coordinate_mismatch(format!(
                "{context} names variable {} outside the shared schema",
                variable.raw()
            ))),
            Some(c) if !c.accepts(value) => Err(ScenarioSetRefusal::coordinate_mismatch(format!(
                "{context} gives {} the value {value:?} outside its declared {:?} domain",
                c.name, c.domain
            ))),
            Some(_) => Ok(()),
        }
    }

    /// Check the shared catalog against the set. Its environment coordinates
    /// agree with the shared schema: a declared domain equals the schema's
    /// unless either is unspecified (an unspecified domain makes no claim, so it
    /// cannot disagree), and a declared unit equals the schema's when both
    /// declare one. Units are therefore compared only between schema
    /// declarations and catalog environments; laws, samples and the request
    /// carry no units, so a value's unit is never checked, only its numeric
    /// membership in the declared domain
    /// ([`Self::check_value`]). Selection targets belong to each scenario, so no
    /// environment of the shared catalog declares any; each scenario's decision
    /// binds the catalog with that scenario's selections on the source
    /// environment.
    ///
    /// # Errors
    /// `schema_mismatch` / `scenarios.coordinate_mismatch` for a disagreeing
    /// coordinate; `invalid_argument` / `scenarios.catalog_selections` for an
    /// environment declaring selection targets.
    pub fn check_catalog(&self, catalog: &EvidenceCatalog) -> Result<(), ScenarioSetRefusal> {
        for environment in catalog.environments.iter() {
            if !environment.selection_targets.is_empty() {
                return Err(ScenarioSetRefusal::invalid(
                    "scenarios.catalog_selections",
                    format!(
                        "catalog environment {} declares selection targets; selections belong \
                         to each scenario",
                        environment.identity
                    ),
                ));
            }
            for declared in environment.variables.iter() {
                let Some(shared) = self.coordinate(declared.variable) else {
                    return Err(ScenarioSetRefusal::coordinate_mismatch(format!(
                        "catalog environment {} declares variable {} outside the shared schema",
                        environment.identity,
                        declared.variable.raw()
                    )));
                };
                let domain = matches!(declared.domain, VariableDomain::Unspecified)
                    || matches!(shared.domain, VariableDomain::Unspecified)
                    || declared.domain == shared.domain;
                let unit = match (&declared.unit, &shared.unit) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                };
                if !domain || !unit {
                    return Err(ScenarioSetRefusal::coordinate_mismatch(format!(
                        "catalog environment {} declares {} with a different {}",
                        environment.identity,
                        shared.name,
                        if domain { "unit" } else { "domain or cardinality" }
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check that the question's outcomes and treatments are shared coordinates.
    ///
    /// # Errors
    /// `schema_mismatch` / `scenarios.coordinate_mismatch`.
    pub fn check_query(&self, query: &ClassicalTransportQuery) -> Result<(), ScenarioSetRefusal> {
        match query
            .outcomes
            .iter()
            .chain(query.treatments.iter())
            .find(|v| self.coordinate(**v).is_none())
        {
            Some(v) => Err(ScenarioSetRefusal::coordinate_mismatch(format!(
                "the question names variable {} outside the shared schema",
                v.raw()
            ))),
            None => Ok(()),
        }
    }

    /// Whether weights were declared.
    #[must_use]
    pub const fn weighted(&self) -> bool {
        self.weighted
    }

    /// Declared mass not assigned to any scenario; `None` for an unweighted set.
    #[must_use]
    pub fn residual_mass(&self) -> Option<f64> {
        self.weighted
            .then(|| unaccounted_after(mass_sum(self.scenarios.iter().filter_map(|s| s.weight))))
    }

    /// The shared observed variables.
    #[must_use]
    pub fn variables(&self) -> Vec<VariableId> {
        self.schema.iter().map(|c| c.variable).collect()
    }

    /// Live bytes one entered scenario holds: its graph with quadratic
    /// coordinate storage, the same per-state shape the sID engine charges.
    fn scenario_bytes(&self) -> u64 {
        let n = self.schema.len();
        let per = n.saturating_mul(n).saturating_mul(64).saturating_add(512);
        u64::try_from(per).unwrap_or(u64::MAX)
    }
}

/// How one scenario was decided. Only [`Self::StructurallyUnidentified`] is an
/// impossibility claim, and only for that scenario.
#[derive(Clone, Debug)]
pub enum ScenarioOutcome {
    /// A checked derivation whose leaves bind to this scenario's own evidence.
    Identified(Box<BoundTransportFunctional>),
    /// An independently verified s-hedge: the question is not transportable
    /// under this scenario even with every source experiment.
    StructurallyUnidentified(Box<SHedgeRecord>),
    /// A derivation exists but the catalog lacks a factor it needs.
    MissingEvidence {
        /// Unmet obligations, per strategy.
        obligations: Arc<[Arc<str>]>,
    },
    /// The bounded search certified nothing; not an impossibility claim.
    NotCertified {
        /// Scope notes, per strategy.
        obligations: Arc<[Arc<str>]>,
    },
    /// Not decided: a search bound or cancellation stopped it. Detail
    /// [`SCENARIO_UNEVALUATED_DETAIL`].
    Unevaluated {
        /// The bound that stopped it.
        stop: SearchStop,
    },
}

impl ScenarioOutcome {
    /// The registered runtime-refusal code of an outcome that is a budget or
    /// cancellation stop ([`SCENARIO_UNEVALUATED_CODE`]); `None` for a decided
    /// outcome.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Unevaluated { .. } => Some(SCENARIO_UNEVALUATED_CODE),
            _ => None,
        }
    }

    /// Stable status name.
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self {
            Self::Identified(_) => "identified",
            Self::StructurallyUnidentified(_) => "structurally_unidentified",
            Self::MissingEvidence { .. } => "missing_evidence",
            Self::NotCertified { .. } => "not_certified",
            Self::Unevaluated { .. } => "unevaluated",
        }
    }
}

/// One decided scenario.
#[derive(Clone, Debug)]
pub struct ScenarioDecision {
    /// The scenario as supplied.
    pub scenario: TransportScenario,
    /// Its outcome.
    pub outcome: ScenarioOutcome,
}

/// The limits a scenario set was decided under, recorded so a consumer replays
/// under exactly the producer's limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScenarioDecisionLimits {
    /// The one shared budget: every scenario entered (one operation at depth
    /// one) and every step of every scenario's search and verification replay.
    pub budget: SearchLimits,
    /// The context's hard memory limit during the decision.
    pub memory_limit_bytes: Option<u64>,
}

/// Every scenario's decision, in canonical order, plus the shared budget's
/// receipt when a budget stop or cancellation left scenarios unevaluated.
#[derive(Clone, Debug)]
pub struct ScenarioSetDecision {
    /// The validated set.
    pub set: TransportScenarioSet,
    /// One decision per scenario, failed as well as successful.
    pub decisions: Vec<ScenarioDecision>,
    /// Present when the shared budget or cancellation stopped the set.
    pub receipt: Option<SearchReceipt>,
    /// The limits in force.
    pub limits: ScenarioDecisionLimits,
}

/// Decide every scenario independently against the same question and catalog.
///
/// One [`SearchBudget`] bounds the whole set. Each scenario entered is charged
/// one operation at depth one with its own graph bytes on top of the memory the
/// scenarios already decided keep holding, and its catalog-aware search, the
/// classical s-hedge check (the witness construction and its independent
/// verification, one operation each), every pretreatment-subset separation test
/// and every verification replay charge the same budget at their own recursion
/// depth and live bytes. Memory is cumulative: a decided scenario keeps holding
/// its engine's peak live state (an upper bound on what its derivation
/// retains), which every later charge sits on top of. The operation, depth and
/// memory limits and cancellation therefore bound the set as a whole: when any
/// charge stops, the scenario being decided and every later one are recorded
/// [`ScenarioOutcome::Unevaluated`] with one cumulative receipt, never dropped.
/// The receipt's `explored` lists the scenarios fully decided, in order, and
/// `unevaluated` the scenario being decided when the stop came and every later
/// one: the two are disjoint. Progress is reported to the context's sink after
/// each scenario.
///
/// # Errors
/// An invalid query or catalog, or a catalog or question disagreeing with the
/// shared schema ([`TransportScenarioSet::check_catalog`],
/// [`TransportScenarioSet::check_query`]).
pub fn decide_transport_scenarios(
    set: &TransportScenarioSet,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<ScenarioSetDecision, IdentificationError> {
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    set.check_catalog(catalog)?;
    set.check_query(query)?;
    let total = set.scenarios().len();
    let remaining = |from: usize| -> Vec<String> {
        set.scenarios()[from..].iter().map(|s| s.name.to_string()).collect()
    };
    let mut decisions = Vec::with_capacity(total);
    let mut receipt = None;
    let mut search = match SearchBudget::new(budget, ctx) {
        Ok(budget) => Some(SharedSearch::new(budget)),
        Err(mut stopped) => {
            stopped.unevaluated = remaining(0);
            receipt = Some(stopped);
            None
        }
    };
    let mut explored = Vec::new();
    let (entry_bytes, mut retained) = (set.scenario_bytes(), 0_u64);
    for (index, scenario) in set.scenarios().iter().enumerate() {
        if let Some(active) = search.as_mut() {
            active.begin(retained);
            if let Err(stop) = active.charge(1, entry_bytes) {
                receipt = Some(active.receipt(stop, explored.clone(), remaining(index)));
                search = None;
            }
        }
        let outcome = if let Some(active) = search.as_mut() {
            let bound = scenario_catalog(catalog, scenario, query);
            match decide_one(&scenario.diagram, query, &bound, active, ctx) {
                Ok(outcome) => {
                    explored.push(scenario.name.to_string());
                    retained = retained.saturating_add(active.peak_bytes());
                    outcome
                }
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = active.stop_of(&error);
                    receipt = Some(active.receipt(stop, explored.clone(), remaining(index)));
                    search = None;
                    ScenarioOutcome::Unevaluated { stop }
                }
                Err(error) => return Err(error),
            }
        } else {
            let stop = receipt.as_ref().map_or(SearchStop::Operations, |r| r.stop);
            ScenarioOutcome::Unevaluated { stop }
        };
        decisions.push(ScenarioDecision { scenario: scenario.clone(), outcome });
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)] // At most 64 scenarios.
            progress.report(decisions.len() as f64 / total as f64, "transport scenarios");
        }
    }
    Ok(ScenarioSetDecision {
        set: set.clone(),
        decisions,
        receipt,
        limits: ScenarioDecisionLimits { budget, memory_limit_bytes: ctx.memory.hard_limit_bytes },
    })
}

/// The shared catalog as one scenario declares it: the source environment,
/// when the catalog has one, carries that scenario's selection targets.
fn scenario_catalog(
    catalog: &EvidenceCatalog,
    scenario: &TransportScenario,
    query: &ClassicalTransportQuery,
) -> EvidenceCatalog {
    let mut bound = catalog.clone();
    if catalog.environments.iter().any(|e| e.identity == query.source) {
        bound.environments = catalog
            .environments
            .iter()
            .map(|e| {
                let mut e = e.clone();
                if e.identity == query.source {
                    e.selection_targets = Arc::from(scenario.diagram.selection_targets());
                }
                e
            })
            .collect();
    }
    bound
}

/// Decide one scenario on the shared budget. A budget or cancellation error is
/// returned for the caller to record against the whole set.
fn decide_one(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<ScenarioOutcome, IdentificationError> {
    Ok(match identify_catalog_transport_metered(diagram, query, catalog, search.meter(), ctx)? {
        CatalogTransportResult::Identified(bound) => ScenarioOutcome::Identified(bound),
        CatalogTransportResult::MissingEvidence { obligations, .. } => {
            ScenarioOutcome::MissingEvidence { obligations }
        }
        CatalogTransportResult::NotCertified { obligations, .. } => {
            match identify_classical_transport_metered(diagram, query, search.meter(), ctx)? {
                ClassicalTransportResult::ProvenNonTransportable(hedge) => {
                    ScenarioOutcome::StructurallyUnidentified(Box::new(hedge.to_record()))
                }
                _ => ScenarioOutcome::NotCertified { obligations },
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::Engine;
    use super::*;
    use antecedent_graph::{Admg, DenseNodeId};

    fn coordinates(nodes: u32) -> Arc<[ScenarioCoordinate]> {
        (0..nodes)
            .map(|i| ScenarioCoordinate {
                variable: VariableId::from_raw(i),
                name: Arc::from(format!("v{i}")),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect()
    }

    fn scenario(
        name: &str,
        nodes: u32,
        selections: &[u32],
        weight: Option<f64>,
    ) -> TransportScenario {
        let mut graph = Admg::with_variables(nodes);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let selections = selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        TransportScenario {
            name: Arc::from(name),
            diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections))
                .unwrap(),
            weight,
            coordinates: coordinates(nodes),
        }
    }

    fn code(
        result: Result<TransportScenarioSet, ScenarioSetRefusal>,
    ) -> (&'static str, &'static str) {
        match result {
            Err(refusal) => (refusal.code, refusal.detail),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    fn edited(edit: impl Fn(&mut ScenarioCoordinate)) -> TransportScenario {
        let mut b = scenario("b", 2, &[1], None);
        let mut coordinates = b.coordinates.to_vec();
        edit(&mut coordinates[1]);
        b.coordinates = coordinates.into();
        b
    }

    #[test]
    fn scenario_sets_are_canonical_and_validated() {
        let set = TransportScenarioSet::try_new(vec![
            scenario("b", 2, &[1], None),
            scenario("a", 2, &[], None),
        ])
        .unwrap();
        assert_eq!(set.scenarios().iter().map(|s| &*s.name).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(set.residual_mass(), None);
        assert_eq!(set.schema().len(), 2);
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![])),
            ("invalid_argument", "scenarios.empty")
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], None),
                scenario("a", 2, &[1], None)
            ])),
            ("invalid_argument", "scenarios.duplicate_or_empty_name")
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[1], None),
                scenario("b", 2, &[1], None)
            ])),
            ("invalid_argument", "scenarios.duplicate_scenario")
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], None),
                scenario("b", 3, &[], None)
            ])),
            ("schema_mismatch", "scenarios.coordinate_mismatch")
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], Some(-0.1)),
                scenario("b", 2, &[1], Some(0.2))
            ])),
            ("invalid_argument", "scenarios.invalid_weights")
        );
        let weighted = TransportScenarioSet::try_new(vec![
            scenario("a", 2, &[], Some(0.25)),
            scenario("b", 2, &[1], Some(0.5)),
        ])
        .unwrap();
        assert!((weighted.residual_mass().unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn the_frozen_scenario_bound_admits_64_and_refuses_65() {
        // Six nodes give 64 distinct selection sets, one per scenario.
        let set = |count: usize| {
            TransportScenarioSet::try_new(
                (0..count)
                    .map(|i| {
                        let selections =
                            (0..6u32).filter(|b| i & (1 << b) != 0).collect::<Vec<_>>();
                        scenario(&format!("s{i:02}"), 6, &selections, None)
                    })
                    .collect(),
            )
        };
        assert_eq!(SCENARIO_MAX_COUNT, 64);
        assert_eq!(set(SCENARIO_MAX_COUNT).unwrap().scenarios().len(), 64);
        assert_eq!(code(set(SCENARIO_MAX_COUNT + 1)), ("route_not_supported", "scenarios.count"));
    }

    #[test]
    fn scenarios_must_share_names_domains_cardinalities_and_units() {
        let a = || scenario("a", 2, &[], None);
        let mismatch = ("schema_mismatch", "scenarios.coordinate_mismatch");
        let name = edited(|c| c.name = Arc::from("renamed"));
        assert_eq!(code(TransportScenarioSet::try_new(vec![a(), name])), mismatch);
        let domain = edited(|c| c.domain = VariableDomain::Continuous);
        assert_eq!(code(TransportScenarioSet::try_new(vec![a(), domain])), mismatch);
        let cardinality = edited(|c| c.domain = VariableDomain::Categorical { cardinality: 3 });
        let four = edited(|c| c.domain = VariableDomain::Categorical { cardinality: 4 });
        let mut three = a();
        three.coordinates = cardinality.coordinates.clone();
        assert_eq!(code(TransportScenarioSet::try_new(vec![three, four])), mismatch);
        let unit = edited(|c| c.unit = Some(Arc::from("mg")));
        assert_eq!(code(TransportScenarioSet::try_new(vec![a(), unit])), mismatch);
        // A schema that does not cover the graph exactly is refused on its own.
        let mut partial = a();
        partial.coordinates = partial.coordinates[..1].to_vec().into();
        assert_eq!(code(TransportScenarioSet::try_new(vec![partial])), mismatch);
        // Values and variables are checked against the shared schema.
        let set = TransportScenarioSet::try_new(vec![a()]).unwrap();
        assert!(set.check_value(VariableId::from_raw(1), &Value::f64(1.0), "law").is_ok());
        let outside = set.check_value(VariableId::from_raw(1), &Value::f64(2.0), "law");
        assert_eq!(outside.unwrap_err().detail, "scenarios.coordinate_mismatch");
        assert!(set.check_value(VariableId::from_raw(7), &Value::f64(0.0), "law").is_err());
    }

    #[test]
    fn an_unspecified_domain_refuses_values_that_belong_to_no_domain() {
        let mut open = scenario("a", 2, &[], None);
        let mut coordinates = open.coordinates.to_vec();
        coordinates[1].domain = VariableDomain::Unspecified;
        open.coordinates = coordinates.into();
        let set = TransportScenarioSet::try_new(vec![open]).unwrap();
        let y = VariableId::from_raw(1);
        // Any finite number may occur under an unspecified domain ...
        assert!(set.check_value(y, &Value::f64(-7.5), "law").is_ok());
        assert!(set.check_value(y, &Value::Int64(3), "law").is_ok());
        // ... but a non-finite or non-numeric value belongs to no domain.
        for value in [
            Value::f64(f64::NAN),
            Value::f64(f64::INFINITY),
            Value::f64(f64::NEG_INFINITY),
            Value::Label(Arc::from("high")),
        ] {
            let refused = set.check_value(y, &value, "law").unwrap_err();
            assert_eq!(
                (refused.code, refused.detail),
                ("schema_mismatch", "scenarios.coordinate_mismatch"),
                "{value:?}"
            );
        }
    }

    #[test]
    fn masses_sum_independently_of_order_and_snap_within_the_tolerance() {
        // 0.7 + 0.2 + 0.1 is 0.9999999999999999 left to right and 1.0 right to left.
        let orders = [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.7, 0.1], [0.1, 0.7, 0.2]];
        let sums = orders.map(mass_sum);
        assert!(sums.iter().all(|s| s.to_bits() == sums[0].to_bits()), "{sums:?}");
        assert!(unaccounted_after(sums[0]).abs() == 0.0);
        assert!(unaccounted_after(0.999_999_999_999_9).abs() == 0.0);
        assert!((unaccounted_after(0.9) - 0.1).abs() < 1e-15);
        assert!(unaccounted_after(1.5).abs() == 0.0);
        // Declared weights are accepted and leave no residual in either order.
        for weights in [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]] {
            let set = TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], Some(weights[0])),
                scenario("b", 2, &[1], Some(weights[1])),
                scenario("c", 2, &[0], Some(weights[2])),
            ])
            .unwrap();
            assert_eq!(set.residual_mass(), Some(0.0), "{weights:?}");
        }
    }

    /// The obstruction check of the classical route charges the shared budget:
    /// the witness construction and its independent verification cost one
    /// operation each, so a cancellation raised on either is observed.
    #[test]
    fn the_obstruction_check_is_charged_to_the_shared_budget() {
        // z -> x, z -> y, x -> y, x <-> y with selection on y: an s-hedge.
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram =
            SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(2)]))
                .unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(2)]),
            treatments: Arc::from([VariableId::from_raw(1)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let run = |cancel_after: Option<usize>| {
            let ctx = ExecutionContext::for_tests(1);
            let budget =
                SearchBudget::new(SearchLimits { operations: 100_000, depth: 256 }, &ctx).unwrap();
            let mut search = SharedSearch::new(budget);
            search.cancel_after = cancel_after.map(|n| (n, ctx.cancellation.clone()));
            let result =
                identify_classical_transport_metered(&diagram, &query, search.meter(), &ctx);
            let consumed = search
                .receipt(SearchStop::Operations, Vec::new(), Vec::new())
                .operations_consumed
                .unwrap();
            (
                result.map(|r| matches!(r, ClassicalTransportResult::ProvenNonTransportable(_))),
                consumed,
            )
        };
        let (proven, total) = run(None);
        assert!(proven.unwrap());
        // The search alone (no obstruction check) charges two operations fewer:
        // the witness construction and its independent verification.
        let searched = {
            let ctx = ExecutionContext::for_tests(1);
            let budget =
                SearchBudget::new(SearchLimits { operations: 100_000, depth: 256 }, &ctx).unwrap();
            let mut search = SharedSearch::new(budget);
            let mut engine = Engine::new_metered(&diagram, &query, search.meter(), &ctx).unwrap();
            let state = engine.initial().unwrap();
            let mut found = engine.solve(state.clone(), false, 0).unwrap();
            if found.is_none() {
                found = engine.solve(state, true, 0).unwrap();
            }
            assert!(found.is_none(), "the s-hedge graph has no derivation");
            drop(engine);
            search.receipt(SearchStop::Operations, Vec::new(), Vec::new()).operations_consumed
        };
        assert_eq!(Some(total), searched.map(|s| s + 2));
        // Cancelled before the last charge, no charge remains to observe it unless
        // the witness construction and verification are themselves charged.
        for after in [total - 2, total - 1] {
            assert!(
                matches!(run(Some(after)).0, Err(IdentificationError::Cancelled)),
                "cancel after {after} of {total} operations"
            );
        }
        assert!(run(Some(total)).0.unwrap());
    }
}
