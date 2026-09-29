//! Finite, explicitly supplied graph/selection scenarios for one transport question.
//!
//! This is not equivalence-class transport. Each scenario is a fixed selection
//! ADMG over the same named variables, decided independently by the licensed
//! classical catalog route: a bounded catalog-aware search, and, only when that
//! search certifies nothing, the classical identifier for an independently
//! verified s-hedge. A scenario's evidence binding is its own; nothing certified
//! for one scenario is reused for another. Declared weights are carried beside
//! the structural set and never renormalized over the scenarios that survive.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    EvidenceCatalog, ExecutionContext, NodeRef, SearchBudget, SearchLimits, SearchReceipt,
    VariableId,
};
use antecedent_graph::SelectionDiagram;

use super::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    ClassicalTransportResult, IdentificationError, SHedgeRecord, SidLimits,
    identify_catalog_transport, identify_classical_transport,
};

/// Scenarios in one set.
pub const SCENARIO_MAX_COUNT: usize = 64;
/// Observed variables in each scenario graph.
pub const SCENARIO_MAX_OBSERVED: usize = 12;
/// Tolerance on declared weights summing to at most one.
pub const SCENARIO_WEIGHT_TOLERANCE: f64 = 1e-12;

/// One supplied scenario: a named fixed graph and the mechanisms that may differ
/// between source and target under it.
#[derive(Clone, Debug)]
pub struct TransportScenario {
    /// Unique scenario name.
    pub name: Arc<str>,
    /// Causal graph over the shared variables, with this scenario's selections.
    pub diagram: SelectionDiagram,
    /// Declared weight, when the set is weighted.
    pub weight: Option<f64>,
}

/// A validated scenario set in canonical (name) order.
#[derive(Clone, Debug)]
pub struct TransportScenarioSet {
    scenarios: Arc<[TransportScenario]>,
    weighted: bool,
}

impl TransportScenarioSet {
    /// Validate and canonicalize a scenario set.
    ///
    /// Every scenario shares the same observed variables (the same node set), no
    /// two scenarios coincide in graph and selections, and weights are declared
    /// for all scenarios or none: finite, non-negative and summing to at most one.
    /// The undeclared remainder is kept as residual mass.
    ///
    /// # Errors
    /// [`IdentificationError::InvalidInput`] with a `scenarios.*` code, or
    /// [`IdentificationError::UnsupportedInput`] for an exceeded bound.
    pub fn try_new(scenarios: Vec<TransportScenario>) -> Result<Self, IdentificationError> {
        let invalid = |code: &'static str| IdentificationError::invalid_input(code);
        if scenarios.is_empty() {
            return Err(invalid("scenarios.empty"));
        }
        if scenarios.len() > SCENARIO_MAX_COUNT {
            return Err(IdentificationError::UnsupportedInput { code: "scenarios.count" });
        }
        let mut scenarios = scenarios;
        scenarios.sort_by(|a, b| a.name.cmp(&b.name));
        let first = scenarios[0].diagram.causal_graph();
        if first.node_count() > SCENARIO_MAX_OBSERVED {
            return Err(IdentificationError::UnsupportedInput { code: "scenarios.observed_count" });
        }
        let variables = first.nodes().iter().copied().collect::<BTreeSet<_>>();
        if variables.iter().any(|node| !matches!(node, NodeRef::Static(_))) {
            return Err(invalid("scenarios.non_static_graph"));
        }
        let mut names = BTreeSet::new();
        let mut signatures = BTreeSet::new();
        for scenario in &scenarios {
            if scenario.name.trim().is_empty() || !names.insert(scenario.name.clone()) {
                return Err(invalid("scenarios.duplicate_or_empty_name"));
            }
            let nodes = scenario.diagram.causal_graph().nodes();
            if nodes.len() != variables.len() || nodes.iter().any(|node| !variables.contains(node)) {
                return Err(invalid("scenarios.coordinate_mismatch"));
            }
            if !signatures.insert(super::graph_signature(&scenario.diagram)) {
                return Err(invalid("scenarios.duplicate_scenario"));
            }
        }
        let weighted = scenarios[0].weight.is_some();
        if scenarios.iter().any(|s| s.weight.is_some() != weighted) {
            return Err(invalid("scenarios.invalid_weights"));
        }
        if weighted {
            let weights = scenarios.iter().filter_map(|s| s.weight).collect::<Vec<_>>();
            let total: f64 = weights.iter().sum();
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                || total > 1.0 + SCENARIO_WEIGHT_TOLERANCE
            {
                return Err(invalid("scenarios.invalid_weights"));
            }
        }
        Ok(Self { scenarios: scenarios.into(), weighted })
    }

    /// Scenarios in canonical order.
    #[must_use]
    pub fn scenarios(&self) -> &[TransportScenario] {
        &self.scenarios
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
            .then(|| (1.0 - self.scenarios.iter().filter_map(|s| s.weight).sum::<f64>()).max(0.0))
    }

    /// The shared observed variables.
    #[must_use]
    pub fn variables(&self) -> Vec<VariableId> {
        self.scenarios[0]
            .diagram
            .causal_graph()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                NodeRef::Static(v) => Some(*v),
                _ => None,
            })
            .collect()
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
    /// Not decided: the scenario budget, this scenario's search limits, or
    /// cancellation stopped it.
    Unevaluated {
        /// `scenarios.scenario_budget`, `scenarios.search_budget` or `scenarios.cancelled`.
        reason: &'static str,
    },
}

impl ScenarioOutcome {
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

/// Every scenario's decision, in canonical order, plus the scenario-budget
/// receipt when a budget or cancellation left scenarios unevaluated.
#[derive(Clone, Debug)]
pub struct ScenarioSetDecision {
    /// The validated set.
    pub set: TransportScenarioSet,
    /// One decision per scenario, failed as well as successful.
    pub decisions: Vec<ScenarioDecision>,
    /// Present when scenarios were left unevaluated by the scenario budget.
    pub receipt: Option<SearchReceipt>,
}

/// Decide every scenario independently against the same question and catalog.
///
/// `budget.operations` bounds how many scenarios are decided, in canonical
/// order; the rest are recorded unevaluated with a receipt, never dropped.
/// `limits` bounds each scenario's own search; exhausting it leaves that
/// scenario unevaluated and the others proceed. Cancellation leaves every
/// remaining scenario unevaluated.
///
/// # Errors
/// An invalid query or catalog, or a query naming variables outside the shared set.
pub fn decide_transport_scenarios(
    set: &TransportScenarioSet,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SidLimits,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<ScenarioSetDecision, IdentificationError> {
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    let variables = set.variables();
    if query.outcomes.iter().chain(query.treatments.iter()).any(|v| !variables.contains(v)) {
        return Err(IdentificationError::invalid_input("scenarios.query_outside_coordinates"));
    }
    let mut decisions = Vec::with_capacity(set.scenarios().len());
    let mut receipt = None;
    let mut meter =
        match SearchBudget::new(SearchLimits { depth: budget.depth.max(1), ..budget }, ctx) {
            Ok(meter) => Some(meter),
            Err(stopped) => {
                receipt = Some(stopped);
                None
            }
        };
    let mut explored = Vec::new();
    for scenario in set.scenarios() {
        let stop = match meter.as_mut() {
            None => Some(()),
            Some(meter) => match meter.charge(0, 0) {
                Ok(()) => None,
                Err(stop) => {
                    let unevaluated = set
                        .scenarios()
                        .iter()
                        .skip(decisions.len())
                        .map(|s| s.name.to_string())
                        .collect();
                    receipt = Some(meter.receipt(stop, explored.clone(), unevaluated));
                    Some(())
                }
            },
        };
        if stop.is_some() {
            meter = None;
            let reason = match receipt.as_ref().map(|r| r.stop) {
                Some(antecedent_core::SearchStop::Cancelled) => "scenarios.cancelled",
                _ => "scenarios.scenario_budget",
            };
            decisions.push(ScenarioDecision {
                scenario: scenario.clone(),
                outcome: ScenarioOutcome::Unevaluated { reason },
            });
            continue;
        }
        explored.push(scenario.name.to_string());
        let outcome = decide_one(&scenario.diagram, query, catalog, limits, ctx)?;
        decisions.push(ScenarioDecision { scenario: scenario.clone(), outcome });
    }
    if let Some(receipt) = receipt.as_mut() {
        receipt.unevaluated = decisions
            .iter()
            .filter(|d| matches!(d.outcome, ScenarioOutcome::Unevaluated { .. }))
            .map(|d| d.scenario.name.to_string())
            .collect();
    }
    Ok(ScenarioSetDecision { set: set.clone(), decisions, receipt })
}

fn decide_one(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<ScenarioOutcome, IdentificationError> {
    let unevaluated = |error: &IdentificationError| ScenarioOutcome::Unevaluated {
        reason: if matches!(error, IdentificationError::Cancelled) {
            "scenarios.cancelled"
        } else {
            "scenarios.search_budget"
        },
    };
    let searched = match identify_catalog_transport(diagram, query, catalog, limits, ctx) {
        Ok(result) => result,
        Err(error) if error.is_budget_or_cancel() => return Ok(unevaluated(&error)),
        Err(error) => return Err(error),
    };
    Ok(match searched {
        CatalogTransportResult::Identified(bound) => ScenarioOutcome::Identified(bound),
        CatalogTransportResult::MissingEvidence { obligations, .. } => {
            ScenarioOutcome::MissingEvidence { obligations }
        }
        CatalogTransportResult::NotCertified { obligations, .. } => {
            match identify_classical_transport(diagram, query, limits, ctx) {
                Ok(ClassicalTransportResult::ProvenNonTransportable(hedge)) => {
                    ScenarioOutcome::StructurallyUnidentified(Box::new(hedge.to_record()))
                }
                Ok(_) => ScenarioOutcome::NotCertified { obligations },
                Err(error) if error.is_budget_or_cancel() => unevaluated(&error),
                Err(error) => return Err(error),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::{Admg, DenseNodeId};

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
        }
    }

    fn code(result: Result<TransportScenarioSet, IdentificationError>) -> String {
        match result {
            Err(IdentificationError::InvalidInput { message }) => message,
            Err(IdentificationError::UnsupportedInput { code }) => code.to_owned(),
            other => panic!("expected a refusal, got {other:?}"),
        }
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
        assert_eq!(code(TransportScenarioSet::try_new(vec![])), "scenarios.empty");
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], None),
                scenario("a", 2, &[1], None)
            ])),
            "scenarios.duplicate_or_empty_name"
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[1], None),
                scenario("b", 2, &[1], None)
            ])),
            "scenarios.duplicate_scenario"
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], None),
                scenario("b", 3, &[], None)
            ])),
            "scenarios.coordinate_mismatch"
        );
        assert_eq!(
            code(TransportScenarioSet::try_new(vec![
                scenario("a", 2, &[], Some(-0.1)),
                scenario("b", 2, &[1], Some(0.2))
            ])),
            "scenarios.invalid_weights"
        );
        let weighted = TransportScenarioSet::try_new(vec![
            scenario("a", 2, &[], Some(0.25)),
            scenario("b", 2, &[1], Some(0.5)),
        ])
        .unwrap();
        assert!((weighted.residual_mass().unwrap() - 0.25).abs() < 1e-12);
        let many = (0..=SCENARIO_MAX_COUNT)
            .map(|i| scenario(&format!("s{i}"), 2, &[], None))
            .collect::<Vec<_>>();
        assert_eq!(code(TransportScenarioSet::try_new(many)), "scenarios.count");
    }
}
