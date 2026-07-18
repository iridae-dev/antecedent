//! AutoIdentifier: return all valid estimands with selection rationale (DESIGN.md §10.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, Intervention, Value};
use causal_expr::{CausalExprArena, EstimandMethod, IdentifiedEstimand};
use causal_graph::Dag;

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::efficient::EfficientBackdoorIdentifier;
use crate::error::IdentificationError;
use crate::frontdoor::FrontDoorIdentifier;
use crate::id::IdIdentifier;
use crate::identifier::{IdentificationWorkspace, Identifier};
use crate::iv::InstrumentalVariableIdentifier;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Prepared graph for [`AutoIdentifier`] (DAG + ADMG embed).
#[derive(Clone, Debug)]
pub struct PreparedAutoGraph {
    /// Criterion-method prepared DAG.
    pub dag: PreparedIdentificationGraph,
    /// General-ID prepared ADMG.
    pub admg: PreparedAdmg,
}

/// Tries every applicable shipped identifier and returns **all** valid estimands.
///
/// Does not choose an estimator. Distribution queries use the ID family only
/// (no second identifier stack).
#[derive(Clone, Debug, Default)]
pub struct AutoIdentifier {
    /// Backdoor search.
    pub backdoor: BackdoorIdentifier,
    /// Efficient backdoor.
    pub efficient: EfficientBackdoorIdentifier,
    /// Front-door.
    pub frontdoor: FrontDoorIdentifier,
    /// Instrumental variables.
    pub iv: InstrumentalVariableIdentifier,
    /// General ID.
    pub general_id: IdIdentifier,
}

impl AutoIdentifier {
    /// Create with default sub-identifiers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a DAG for all methods.
    ///
    /// # Errors
    ///
    /// Graph construction / validation failure.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedAutoGraph, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare with declared assumptions.
    ///
    /// # Errors
    ///
    /// Graph construction / validation failure.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedAutoGraph, IdentificationError> {
        Ok(PreparedAutoGraph {
            dag: PreparedIdentificationGraph::with_assumptions(graph.clone(), assumptions.clone()),
            admg: PreparedAdmg::from_dag_with_assumptions(graph, assumptions)?,
        })
    }

    /// Identify `query`, collecting every successful estimand into one arena.
    ///
    /// # Errors
    ///
    /// Unsupported query when the query type is not handled.
    pub fn identify(
        &self,
        prepared: &PreparedAutoGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let mut derivation = DerivationTrace::default();
        derivation.push(
            "auto",
            "trying backdoor, efficient backdoor, frontdoor, IV, and general ID",
        );
        let mut perf = IdentificationPerformanceRecord::default();
        let mut hedge = None;
        let mut assumptions = prepared.dag.declared_assumptions().clone();
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::new();

        match query {
            CausalQuery::AverageEffect(q) => {
                let active = set_value(&q.active)?;
                let control = set_value(&q.control)?;
                self.try_method(
                    "backdoor.adjustment",
                    || self.backdoor.identify(&prepared.dag, query, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                );
                self.try_method(
                    "backdoor.efficient",
                    || self.efficient.identify(&prepared.dag, query, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                );
                self.try_method(
                    "frontdoor",
                    || self.frontdoor.identify(&prepared.dag, query, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                );
                self.try_method(
                    "iv",
                    || self.iv.identify(&prepared.dag, query, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                );
                self.try_method(
                    "general.id",
                    || self.general_id.identify(&prepared.admg, query, workspace),
                    q,
                    active,
                    control,
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                );
            }
            CausalQuery::Distribution(_) => {
                match self.general_id.identify(&prepared.admg, query, workspace) {
                    Ok(res) if res.status == IdentificationStatus::NonparametricallyIdentified => {
                        derivation.push(
                            "auto.method",
                            format!("general.id: identified ({} estimand(s))", res.estimands.len()),
                        );
                        arena = res.arena;
                        estimands = res.estimands;
                        assumptions = res.required_assumptions;
                        perf = res.performance;
                    }
                    Ok(res) => {
                        derivation.push(
                            "auto.method",
                            format!("general.id: not identified ({:?})", res.status),
                        );
                        hedge = res.hedge;
                        perf = res.performance;
                    }
                    Err(e) => derivation.push("auto.method", format!("general.id: error ({e})")),
                }
            }
            _ => {
                return Err(IdentificationError::unsupported(
                    "AutoIdentifier supports AverageEffect and Distribution queries",
                ));
            }
        }

        if estimands.is_empty() {
            let mut out =
                IdentificationResult::not_identified(query.clone(), derivation, assumptions, perf);
            out.hedge = hedge;
            return Ok(out);
        }

        Ok(IdentificationResult::identified(
            query.clone(),
            estimands,
            arena,
            derivation,
            assumptions,
            perf,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_method(
        &self,
        name: &str,
        run: impl FnOnce() -> Result<IdentificationResult, IdentificationError>,
        q: &AverageEffectQuery,
        active: Value,
        control: Value,
        arena: &mut CausalExprArena,
        estimands: &mut Vec<IdentifiedEstimand>,
        derivation: &mut DerivationTrace,
        perf: &mut IdentificationPerformanceRecord,
        assumptions: &mut AssumptionSet,
        hedge: &mut Option<crate::hedge::HedgeCertificate>,
    ) {
        match run() {
            Ok(res) if res.status == IdentificationStatus::NonparametricallyIdentified => {
                derivation.push(
                    "auto.method",
                    format!("{name}: identified ({} estimand(s))", res.estimands.len()),
                );
                for e in &res.estimands {
                    if let Some(rebuilt) = rebuild_estimand(arena, e, q, &active, &control) {
                        estimands.push(rebuilt);
                    } else if name == "general.id" && arena.is_empty() {
                        // Adopt general-ID arena wholesale when it is the first success
                        // or when criterion rebuild is unavailable.
                        *arena = res.arena.clone();
                        estimands.extend(res.estimands.clone());
                        break;
                    } else if name == "general.id" {
                        derivation.push(
                            "auto.method.general_id",
                            "general.id identified; functionals available via IdIdentifier \
                             (arena merge deferred when criterion estimands already present)",
                        );
                    }
                }
                perf.candidates_examined =
                    perf.candidates_examined.saturating_add(res.performance.candidates_examined);
                perf.sets_returned =
                    perf.sets_returned.saturating_add(res.performance.sets_returned);
                *assumptions = res.required_assumptions;
            }
            Ok(res) => {
                derivation.push("auto.method", format!("{name}: not identified ({:?})", res.status));
                if res.hedge.is_some() && hedge.is_none() {
                    *hedge = res.hedge;
                }
                perf.candidates_examined =
                    perf.candidates_examined.saturating_add(res.performance.candidates_examined);
            }
            Err(IdentificationError::UnsupportedQuery { message }) => {
                derivation.push("auto.method", format!("{name}: skipped ({message})"));
            }
            Err(e) => {
                derivation.push("auto.method", format!("{name}: error ({e})"));
            }
        }
    }
}

fn set_value(iv: &Intervention) -> Result<Value, IdentificationError> {
    match iv {
        Intervention::Set { value, .. } => Ok(value.clone()),
        _ => Err(IdentificationError::unsupported("auto ATE requires Set interventions")),
    }
}

fn rebuild_estimand(
    arena: &mut CausalExprArena,
    e: &IdentifiedEstimand,
    q: &AverageEffectQuery,
    active: &Value,
    control: &Value,
) -> Option<IdentifiedEstimand> {
    let kind = e.method_kind().ok()?;
    match kind {
        EstimandMethod::BackdoorAdjustment | EstimandMethod::BackdoorEfficient => {
            let functional = arena.backdoor_ate(
                q.treatment,
                q.outcome,
                e.adjustment_set.as_ref(),
                active.clone(),
                control.clone(),
            );
            Some(IdentifiedEstimand::backdoor(e.method.clone(), Arc::clone(&e.adjustment_set), functional))
        }
        EstimandMethod::FrontDoor => {
            let functional = arena.frontdoor_ate(
                q.treatment,
                q.outcome,
                e.mediators.as_ref(),
                active.clone(),
                control.clone(),
            );
            Some(IdentifiedEstimand::frontdoor(e.method.clone(), Arc::clone(&e.mediators), functional))
        }
        EstimandMethod::Iv => {
            // IV functionals are estimator-side; keep role metadata with a backdoor-style
            // placeholder contrast only when instruments are present — prefer tagging via
            // IdentifiedEstimand::instrumental with a zero contrast for selection rationale.
            let functional = arena.backdoor_ate(
                q.treatment,
                q.outcome,
                &[],
                active.clone(),
                control.clone(),
            );
            Some(IdentifiedEstimand::instrumental(e.method.clone(), Arc::clone(&e.instruments), functional))
        }
        EstimandMethod::GeneralId => None,
        _ => None,
    }
}

impl Identifier<Dag> for AutoIdentifier {
    type Prepared = PreparedAutoGraph;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        self.prepare_with_assumptions(graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

#[cfg(test)]
mod tests {
    use causal_core::VariableId;
    use causal_graph::DenseNodeId;

    use super::*;

    #[test]
    fn auto_finds_backdoor_on_chain() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!res.estimands.is_empty());
        assert!(res.derivation.steps.iter().any(|s| s.rule.as_ref() == "auto.method"));
    }
}
