//! `AutoIdentifier`: return all valid estimands with selection rationale.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::unused_self
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    Intervention, Value,
};
use antecedent_expr::{CausalExprArena, EstimandMethod, IdentifiedEstimand};
use antecedent_graph::Dag;

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::efficient::EfficientBackdoorIdentifier;
use crate::error::IdentificationError;
use crate::frontdoor::FrontDoorIdentifier;
use crate::id::IdIdentifier;
use crate::idc::IdcIdentifier;
use crate::identifier::{IdentificationWorkspace, Identifier};
use crate::iv::InstrumentalVariableIdentifier;
use crate::path_specific::PathSpecificIdentifier;
use crate::prepared::PreparedAdmg;
use crate::rd::{SharpRdConfig, SharpRdIdentifier};
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
/// Does not choose an estimator. Distribution queries use the ID/IDC family only
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
    /// Conditional interventional distributions (IDC).
    pub idc: IdcIdentifier,
    /// Path-restricted natural effects.
    pub path_specific: PathSpecificIdentifier,
    /// Optional sharp RD design config. When set, Auto attempts [`SharpRdIdentifier`].
    pub rd: Option<SharpRdConfig>,
}

impl AutoIdentifier {
    /// Create with default sub-identifiers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach sharp RD design parameters for Auto identification.
    #[must_use]
    pub fn with_rd(mut self, config: SharpRdConfig) -> Self {
        self.rd = Some(config);
        self
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
        derivation
            .push("auto", "trying backdoor, efficient backdoor, frontdoor, IV, RD, and general ID");
        let mut perf = IdentificationPerformanceRecord::default();
        let mut hedge = None;
        let mut assumptions = prepared.dag.declared_assumptions().clone();
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::new();
        let mut diagnostics = Vec::new();

        match query {
            CausalQuery::AverageEffect(q) => {
                let (active_do, control_do, bernoulli_scale) =
                    crate::intervention_support::normalize_ate_pair(&q.active, &q.control)?;
                let active = set_value(&active_do)?;
                let control = set_value(&control_do)?;
                if let Some(scale) = bernoulli_scale {
                    diagnostics.push(Diagnostic::new(
                        "auto.stochastic.bernoulli_scale",
                        DiagnosticKind::Execution,
                        DiagnosticSeverity::Info,
                        format!(
                            "Bernoulli / binary mixture ATE: identified hard do(1)−do(0); \
                             stochastic effect scale = {scale} (multiply hard ATE by scale)"
                        ),
                    ));
                    derivation.push(
                        "auto.stochastic",
                        format!("bernoulli mixture scale={scale} on hard unit contrast"),
                    );
                }
                // Rebuild query with normalized Sets so sub-identifiers see hard interventions.
                let q_norm = AverageEffectQuery::new(
                    q.treatment,
                    q.outcome,
                    Arc::clone(&q.effect_modifiers),
                    control_do,
                    active_do,
                    q.target_population.clone(),
                );
                let query_norm = CausalQuery::AverageEffect(q_norm.clone());
                let q = &q_norm;
                self.try_method(
                    "backdoor.adjustment",
                    || self.backdoor.identify(&prepared.dag, &query_norm, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                    &mut diagnostics,
                );
                self.try_method(
                    "backdoor.efficient",
                    || self.efficient.identify(&prepared.dag, &query_norm, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                    &mut diagnostics,
                );
                self.try_method(
                    "frontdoor",
                    || self.frontdoor.identify(&prepared.dag, &query_norm, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                    &mut diagnostics,
                );
                self.try_method(
                    "iv",
                    || self.iv.identify(&prepared.dag, &query_norm, workspace),
                    q,
                    active.clone(),
                    control.clone(),
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                    &mut diagnostics,
                );
                if let Some(cfg) = &self.rd {
                    self.try_method(
                        "rd.sharp",
                        || SharpRdIdentifier::new(*cfg).identify(query_norm.clone()),
                        q,
                        active.clone(),
                        control.clone(),
                        &mut arena,
                        &mut estimands,
                        &mut derivation,
                        &mut perf,
                        &mut assumptions,
                        &mut hedge,
                        &mut diagnostics,
                    );
                } else {
                    diagnostics.push(Diagnostic::new(
                        "auto.rd.missing_config",
                        DiagnosticKind::Execution,
                        DiagnosticSeverity::Info,
                        "sharp RD skipped: no running-variable / cutoff / bandwidth config on AutoIdentifier",
                    ));
                    derivation.push("auto.method", "rd.sharp: not applicable (missing RD config)");
                }
                self.try_method(
                    "general.id",
                    || self.general_id.identify(&prepared.admg, &query_norm, workspace),
                    q,
                    active,
                    control,
                    &mut arena,
                    &mut estimands,
                    &mut derivation,
                    &mut perf,
                    &mut assumptions,
                    &mut hedge,
                    &mut diagnostics,
                );
            }
            CausalQuery::Distribution(q) => {
                let method = if q.conditioning.is_empty() { "general.id" } else { "general.idc" };
                let run = if q.conditioning.is_empty() {
                    self.general_id.identify(&prepared.admg, query, workspace)
                } else {
                    self.idc.identify(&prepared.admg, query, workspace)
                };
                match run {
                    Ok(res) if res.status == IdentificationStatus::NonparametricallyIdentified => {
                        derivation.push(
                            "auto.method",
                            format!("{method}: identified ({} estimand(s))", res.estimands.len()),
                        );
                        arena = res.arena;
                        estimands = res.estimands;
                        assumptions = res.required_assumptions;
                        perf = res.performance;
                        diagnostics.extend(res.diagnostics);
                    }
                    Ok(res) => {
                        derivation.push(
                            "auto.method",
                            format!("{method}: not identified ({:?})", res.status),
                        );
                        diagnostics.push(Diagnostic::new(
                            format!("auto.{method}.not_identified"),
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            format!("{method} did not identify the query ({:?})", res.status),
                        ));
                        hedge = res.hedge;
                        perf = res.performance;
                        diagnostics.extend(res.diagnostics);
                    }
                    Err(IdentificationError::UnsupportedQuery { message }) => {
                        diagnostics.push(Diagnostic::new(
                            format!("auto.{method}.unsupported"),
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            format!("{method}: unsupported ({message})"),
                        ));
                        derivation
                            .push("auto.method", format!("{method}: unsupported ({message})"));
                    }
                    Err(e) => {
                        diagnostics.push(Diagnostic::new(
                            format!("auto.{method}.error"),
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            format!("{method}: error ({e})"),
                        ));
                        derivation.push("auto.method", format!("{method}: error ({e})"));
                    }
                }
            }
            CausalQuery::PathSpecific(_) => {
                match self.path_specific.identify(&prepared.admg, query, workspace) {
                    Ok(res) if res.status == IdentificationStatus::NonparametricallyIdentified => {
                        derivation.push(
                            "auto.method",
                            format!(
                                "path_specific.natural: identified ({} estimand(s))",
                                res.estimands.len()
                            ),
                        );
                        arena = res.arena;
                        estimands = res.estimands;
                        assumptions = res.required_assumptions;
                        perf = res.performance;
                        diagnostics.extend(res.diagnostics);
                    }
                    Ok(res) => {
                        derivation.push(
                            "auto.method",
                            format!("path_specific.natural: not identified ({:?})", res.status),
                        );
                        diagnostics.push(Diagnostic::new(
                            "auto.path_specific.not_identified",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            format!(
                                "path_specific.natural did not identify the query ({:?})",
                                res.status
                            ),
                        ));
                        hedge = res.hedge;
                        perf = res.performance;
                        diagnostics.extend(res.diagnostics);
                    }
                    Err(IdentificationError::UnsupportedQuery { message }) => {
                        diagnostics.push(Diagnostic::new(
                            "auto.path_specific.unsupported",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            format!("path_specific.natural: unsupported ({message})"),
                        ));
                        derivation.push(
                            "auto.method",
                            format!("path_specific.natural: unsupported ({message})"),
                        );
                    }
                    Err(e) => {
                        diagnostics.push(Diagnostic::new(
                            "auto.path_specific.error",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            format!("path_specific.natural: error ({e})"),
                        ));
                        derivation
                            .push("auto.method", format!("path_specific.natural: error ({e})"));
                    }
                }
            }
            _ => {
                return Err(IdentificationError::unsupported(
                    "AutoIdentifier supports AverageEffect, Distribution, and PathSpecific queries",
                ));
            }
        }

        if estimands.is_empty() {
            let mut out =
                IdentificationResult::not_identified(query.clone(), derivation, assumptions, perf);
            out.hedge = hedge;
            out.diagnostics = diagnostics;
            return Ok(out);
        }

        let mut out = IdentificationResult::identified(
            query.clone(),
            estimands,
            arena,
            derivation,
            assumptions,
            perf,
        );
        out.diagnostics = diagnostics;
        Ok(out)
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
        diagnostics: &mut Vec<Diagnostic>,
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
                diagnostics.extend(res.diagnostics);
            }
            Ok(res) => {
                derivation
                    .push("auto.method", format!("{name}: not identified ({:?})", res.status));
                diagnostics.push(Diagnostic::new(
                    format!("auto.{name}.not_identified"),
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!("{name} did not identify the query ({:?})", res.status),
                ));
                if res.hedge.is_some() && hedge.is_none() {
                    *hedge = res.hedge;
                }
                perf.candidates_examined =
                    perf.candidates_examined.saturating_add(res.performance.candidates_examined);
                diagnostics.extend(res.diagnostics);
            }
            Err(IdentificationError::UnsupportedQuery { message }) => {
                diagnostics.push(Diagnostic::new(
                    format!("auto.{name}.unsupported"),
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Warning,
                    format!("{name}: unsupported ({message})"),
                ));
                derivation.push("auto.method", format!("{name}: unsupported ({message})"));
            }
            Err(e) => {
                diagnostics.push(Diagnostic::new(
                    format!("auto.{name}.error"),
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Warning,
                    format!("{name}: error ({e})"),
                ));
                derivation.push("auto.method", format!("{name}: error ({e})"));
            }
        }
    }
}

fn set_value(intervention: &Intervention) -> Result<Value, IdentificationError> {
    crate::intervention_support::require_set_value(intervention, "auto ATE")
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
            Some(IdentifiedEstimand::backdoor(
                e.method.clone(),
                Arc::clone(&e.adjustment_set),
                functional,
            ))
        }
        EstimandMethod::FrontDoor => {
            let functional = arena.frontdoor_ate(
                q.treatment,
                q.outcome,
                e.mediators.as_ref(),
                active.clone(),
                control.clone(),
            );
            Some(IdentifiedEstimand::frontdoor(
                e.method.clone(),
                Arc::clone(&e.mediators),
                functional,
            ))
        }
        EstimandMethod::Iv => {
            let functional =
                arena.iv_wald(q.treatment, q.outcome, e.instruments.as_ref(), active, control);
            Some(IdentifiedEstimand::instrumental(
                e.method.clone(),
                Arc::clone(&e.instruments),
                functional,
            ))
        }
        EstimandMethod::RdSharp => {
            let functional =
                arena.backdoor_ate(q.treatment, q.outcome, &[], active.clone(), control.clone());
            Some(IdentifiedEstimand::rd_sharp(
                functional,
                e.rd_design.unwrap_or(antecedent_expr::RdDesignParams::new(q.treatment, 0.0, 1.0)),
            ))
        }
        EstimandMethod::GeneralId => None,
        _ => None,
    }
}

impl crate::identifier::sealed::Sealed for AutoIdentifier {}

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
    use antecedent_core::{Intervention, MechanismOverride, VariableId};
    use antecedent_expr::EstimandMethod;
    use antecedent_graph::DenseNodeId;

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
        assert!(res.diagnostics.iter().any(|d| d.code.as_ref() == "auto.rd.missing_config"));
    }

    #[test]
    fn auto_distribution_uses_idc_when_conditioned() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                VariableId::from_raw(2),
                [antecedent_core::Intervention::set(
                    VariableId::from_raw(1),
                    antecedent_core::Value::f64(1.0),
                )],
            )
            .with_conditioning([VariableId::from_raw(0)]),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.derivation.steps.iter().any(|s| {
            s.detail.as_ref().contains("general.idc") || s.rule.as_ref().contains("idc")
        }));
    }

    #[test]
    fn auto_accepts_soft_constant_as_set() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            Arc::from([]),
            Intervention::set(VariableId::from_raw(0), Value::f64(0.0)),
            Intervention::soft(VariableId::from_raw(0), MechanismOverride::constant(1.0)),
            antecedent_core::TargetPopulation::AllObserved,
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!res.estimands.is_empty());
    }

    #[test]
    fn auto_rejects_soft_linear_gaussian() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            Arc::from([]),
            Intervention::set(VariableId::from_raw(0), Value::f64(0.0)),
            Intervention::soft(
                VariableId::from_raw(0),
                MechanismOverride::named("linear_gaussian", vec![1.0, 0.5]),
            ),
            antecedent_core::TargetPopulation::AllObserved,
        ));
        let mut ws = IdentificationWorkspace::default();
        let err = auto.identify(&prep, &q, &mut ws).unwrap_err();
        match err {
            IdentificationError::UnsupportedQuery { message } => {
                assert!(message.contains("Soft"), "{message}");
            }
            other => panic!("expected UnsupportedQuery, got {other:?}"),
        }
    }

    #[test]
    fn auto_iv_uses_wald_functional_with_instruments() {
        // Z -> T -> Y, U -> T, U -> Y
        let mut dag = Dag::with_variables(4);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // Z->T
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap(); // T->Y
        dag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap(); // U->T
        dag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap(); // U->Y
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let iv = res
            .estimands
            .iter()
            .find(|e| e.method_kind().ok() == Some(EstimandMethod::Iv))
            .expect("IV estimand");
        assert!(!iv.instruments.is_empty());
        assert_eq!(iv.instruments[0], VariableId::from_raw(0));
        let _ = res.arena.node(iv.functional);
    }

    #[test]
    fn auto_with_rd_config_identifies_sharp_rd() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let auto = AutoIdentifier::new().with_rd(SharpRdConfig {
            running_variable: VariableId::from_raw(2),
            cutoff: 0.0,
            bandwidth: 1.0,
        });
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.derivation.steps.iter().any(|s| s.detail.as_ref().contains("rd.sharp")));
        assert!(!res.diagnostics.iter().any(|d| d.code.as_ref() == "auto.rd.missing_config"));
    }
}
