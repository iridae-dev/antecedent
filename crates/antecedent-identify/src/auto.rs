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
    Intervention, TargetPopulation, Value,
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
use crate::response::ResponseIdentifier;
use crate::result::{
    DerivationTrace, EstimandClaim, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus,
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
    /// Optional sharp RD design. Auto uses it only for a query whose target population
    /// is the design's cutoff ([`TargetPopulation::LocalAtCutoff`]); it never infers one.
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
        let mut claims: Vec<EstimandClaim> = Vec::new();
        let mut diagnostics = Vec::new();

        match query {
            CausalQuery::AverageEffect(q) => {
                let (active_do, control_do, bernoulli_scale) =
                    crate::intervention_support::normalize_ate_pair(&q.active, &q.control)?;
                let active = set_value(&active_do)?;
                let control = set_value(&control_do)?;
                if let Some(scale) = bernoulli_scale {
                    // `scale = w_a - w_c` is exactly 1 only when the pair already *is*
                    // the hard do(1)/do(0) contrast (e.g. a degenerate Bernoulli(1)
                    // matched with Set(0)): the functional every strategy below builds
                    // is then already the requested estimand. For any other weight the
                    // true stochastic contrast is `scale * (hard ATE)`, and nothing in
                    // the expression algebra can multiply a functional by a free-standing
                    // scalar, so shipping the hard functional under an "identified" status
                    // would silently return the wrong number. Refuse instead of guessing.
                    if (scale - 1.0).abs() > 1e-9 {
                        diagnostics.push(Diagnostic::new(
                            "auto.stochastic.bernoulli_scale_unsupported",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            format!(
                                "Bernoulli / binary mixture ATE: the true effect is \
                                 scale * E[Y|do(1)] − E[Y|do(0)] with scale = {scale}, but no \
                                 returned functional can carry that scalar factor; refusing \
                                 rather than returning the unscaled hard contrast as if it \
                                 were the requested stochastic effect"
                            ),
                        ));
                        derivation.push(
                            "auto.stochastic",
                            format!(
                                "bernoulli mixture scale={scale}: not applied to any functional, refusing"
                            ),
                        );
                        let mut out = IdentificationResult::not_identified(
                            query.clone(),
                            derivation,
                            assumptions,
                            perf,
                        );
                        out.diagnostics = diagnostics;
                        return Ok(out);
                    }
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
                // A sharp RD design speaks for units at its cutoff; the graph strategies
                // speak for the population the query names. They answer different
                // questions, so which ones run is decided by the population asked for and
                // their estimands are never listed side by side.
                if matches!(q.target_population, TargetPopulation::LocalAtCutoff { .. }) {
                    if let Some(cfg) = &self.rd {
                        self.try_method(
                            "rd.sharp",
                            || {
                                SharpRdIdentifier::new(*cfg)
                                    .identify_on(prepared.dag.dag(), query_norm.clone())
                            },
                            q,
                            active,
                            control,
                            &mut arena,
                            &mut estimands,
                            &mut derivation,
                            &mut perf,
                            &assumptions,
                            &mut claims,
                            &mut hedge,
                            &mut diagnostics,
                        );
                    } else {
                        diagnostics.push(Diagnostic::new(
                            "auto.rd.missing_config",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Warning,
                            "the effect at a running-variable cutoff needs a sharp RD design \
                             (running variable, cutoff, bandwidth); none was supplied and none \
                             is inferred",
                        ));
                        derivation
                            .push("auto.method", "rd.sharp: not applicable (missing RD config)");
                    }
                } else {
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
                        &assumptions,
                        &mut claims,
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
                        &assumptions,
                        &mut claims,
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
                        &assumptions,
                        &mut claims,
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
                        &assumptions,
                        &mut claims,
                        &mut hedge,
                        &mut diagnostics,
                    );
                    if self.rd.is_some() {
                        diagnostics.push(Diagnostic::new(
                            "auto.rd.local_estimand_not_requested",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            "sharp RD not used: the design identifies the effect for units at \
                             its cutoff, not the requested population effect; ask for \
                             TargetPopulation::LocalAtCutoff to use it",
                        ));
                        derivation.push(
                            "auto.method",
                            "rd.sharp: not applicable (the query does not target the cutoff)",
                        );
                    } else {
                        diagnostics.push(Diagnostic::new(
                            "auto.rd.missing_config",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Info,
                            "sharp RD skipped: no running-variable / cutoff / bandwidth config on AutoIdentifier",
                        ));
                        derivation
                            .push("auto.method", "rd.sharp: not applicable (missing RD config)");
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
                        &assumptions,
                        &mut claims,
                        &mut hedge,
                        &mut diagnostics,
                    );
                }
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
                        diagnostics.push(not_identified_diagnostic(method, method, &res));
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
                        diagnostics.push(not_identified_diagnostic(
                            "path_specific",
                            "path_specific.natural",
                            &res,
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
            CausalQuery::Response(_) => {
                let id = ResponseIdentifier { backdoor: self.backdoor.clone() };
                match id.identify(&prepared.dag, query, workspace) {
                    Ok(res) if res.status == IdentificationStatus::NonparametricallyIdentified => {
                        derivation.push(
                            "auto.method",
                            format!(
                                "response.backdoor: identified ({} pair(s))",
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
                            format!("response.backdoor: not identified ({:?})", res.status),
                        );
                        diagnostics.extend(res.diagnostics);
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
            _ => {
                return Err(IdentificationError::unsupported(
                    "AutoIdentifier supports AverageEffect, Distribution, PathSpecific, and Response queries",
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

        // The listing is nonparametric when any alternative is; each estimand's own status
        // and assumptions are its claim. The listing-level set is the union over the
        // alternatives, so it never understates what a listed estimand relies on.
        let all_parametric = !claims.is_empty()
            && claims
                .iter()
                .all(|c| c.status == IdentificationStatus::IdentifiedUnderParametricRestrictions);
        let status = if all_parametric {
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        } else {
            IdentificationStatus::NonparametricallyIdentified
        };
        for claim in &claims {
            assumptions.extend_unique(&claim.required_assumptions.entries);
        }
        let mut out = IdentificationResult::from_parts(
            status,
            query.clone(),
            estimands,
            arena,
            derivation,
            assumptions,
            Vec::new(),
            perf,
            None,
        )
        .with_estimand_claims(claims);
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
        assumptions_declared: &AssumptionSet,
        claims: &mut Vec<EstimandClaim>,
        hedge: &mut Option<crate::hedge::HedgeCertificate>,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        match run() {
            Ok(res)
                if matches!(
                    res.status,
                    IdentificationStatus::NonparametricallyIdentified
                        | IdentificationStatus::IdentifiedUnderParametricRestrictions
                ) =>
            {
                derivation.push(
                    "auto.method",
                    format!("{name}: identified ({} estimand(s))", res.estimands.len()),
                );
                // Each listed estimand keeps the claim of the strategy that produced it:
                // that strategy's status and assumptions plus the caller-declared ones.
                // A later strategy that also succeeds must not rewrite it.
                let declared = assumptions_declared.clone();
                let claim_of = |index: usize| {
                    let own = res.claim(index).expect("index is within res.estimands");
                    let mut required = declared.clone();
                    required.extend_unique(&own.required_assumptions.entries);
                    EstimandClaim { status: own.status, required_assumptions: required }
                };
                for (index, e) in res.estimands.iter().enumerate() {
                    if let Some(rebuilt) = rebuild_estimand(arena, e, q, &active, &control) {
                        estimands.push(rebuilt);
                        claims.push(claim_of(index));
                    } else if name == "general.id" && arena.is_empty() {
                        *arena = res.arena.clone();
                        estimands.extend(res.estimands.clone());
                        claims.extend((0..res.estimands.len()).map(&claim_of));
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
                diagnostics.extend(res.diagnostics);
            }
            Ok(res) => {
                derivation
                    .push("auto.method", format!("{name}: not identified ({:?})", res.status));
                diagnostics.push(not_identified_diagnostic(name, name, &res));
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

/// Whether a strategy stopped at a search budget before it could decide.
///
/// Strategies mark that exit with an `identify.<method>.search_bounded` diagnostic (or a
/// completion / history cap); their `NotIdentified` is then undecided rather than refuted.
fn search_bounded(res: &IdentificationResult) -> bool {
    crate::envelope::search_truncated(res)
        || res.diagnostics.iter().any(|d| d.code.as_ref().ends_with(".search_bounded"))
}

/// Auto's record of a strategy that returned `NotIdentified`.
///
/// A completed search is a scientific negative for that strategy. A search that ran out
/// of budget is not: it is reported as an execution warning so that nothing downstream
/// reads an unfinished search as a proof of non-identification.
fn not_identified_diagnostic(key: &str, method: &str, res: &IdentificationResult) -> Diagnostic {
    if search_bounded(res) {
        Diagnostic::new(
            format!("auto.{key}.search_bounded"),
            DiagnosticKind::Execution,
            DiagnosticSeverity::Warning,
            format!(
                "{method} stopped at its search budget before deciding; identifiability by \
                 this strategy is undecided, not refuted"
            ),
        )
    } else {
        Diagnostic::new(
            format!("auto.{key}.not_identified"),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("{method} did not identify the query ({:?})", res.status),
        )
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
            // A Wald ratio conditions on one instrument; a set that is not exactly one cannot
            // be rebuilt as an IV functional.
            let functional = arena
                .iv_wald(q.treatment, q.outcome, e.instruments.as_ref(), active, control)
                .ok()?;
            Some(IdentifiedEstimand::instrumental(
                e.method.clone(),
                Arc::clone(&e.instruments),
                functional,
            ))
        }
        EstimandMethod::RdSharp => {
            // The design is the caller's; an estimand without one cannot be rebuilt.
            let design = e.rd_design?;
            let functional = arena.rd_sharp_local_effect(
                q.treatment,
                q.outcome,
                design.running_variable,
                design.cutoff,
                active.clone(),
                control.clone(),
            );
            Some(IdentifiedEstimand::rd_sharp(functional, design))
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
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/auto_envelopes/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["expected_method_family"].as_str(), Some("backdoor"));
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

    fn iv_dag() -> Dag {
        // Z -> T -> Y, U -> T, U -> Y
        let mut dag = Dag::with_variables(4);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // Z->T
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap(); // T->Y
        dag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap(); // U->T
        dag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap(); // U->Y
        dag
    }

    fn has_exclusion(set: &AssumptionSet, instrument: u32) -> bool {
        set.entries.iter().any(|r| {
            r.assumption
                == antecedent_core::Assumption::ExclusionRestriction {
                    instrument: VariableId::from_raw(instrument),
                }
        })
    }

    #[test]
    fn auto_iv_claim_keeps_exclusion_restriction_after_general_id_succeeds() {
        // General ID runs last and also identifies this DAG (every node observed), returning
        // only caller-declared assumptions. The IV estimand must still carry its own record.
        let dag = iv_dag();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert!(
            res.derivation
                .steps
                .iter()
                .any(|s| s.detail.as_ref().starts_with("general.id: identified"))
        );
        assert_eq!(res.estimand_claims.len(), res.estimands.len());

        let iv_index = res
            .estimands
            .iter()
            .position(|e| e.method_kind().ok() == Some(EstimandMethod::Iv))
            .expect("IV estimand");
        let chosen = res.narrowed_to(iv_index).unwrap();
        assert_eq!(chosen.estimands.len(), 1);
        assert_eq!(chosen.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        assert!(has_exclusion(&chosen.required_assumptions, 0));
        assert!(
            chosen
                .required_assumptions
                .entries
                .iter()
                .any(|r| r.assumption == antecedent_core::Assumption::CausalMarkov)
        );
        assert!(chosen.required_assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "iv.relevance"
        )));
        assert!(chosen.required_assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == "iv.constant_linear_effect_or_monotonicity"
        )));

        // A back-door estimand in the same result does not inherit the IV records, and the
        // IV claim does not inherit a nonparametric banner from it.
        let bd_index = res
            .estimands
            .iter()
            .position(|e| e.method_kind().ok() == Some(EstimandMethod::BackdoorAdjustment))
            .expect("backdoor estimand");
        let bd = res.narrowed_to(bd_index).unwrap();
        assert_eq!(bd.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!has_exclusion(&bd.required_assumptions, 0));

        // The un-narrowed result lists every alternative, so its set covers all of them.
        assert!(has_exclusion(&res.required_assumptions, 0));
    }

    #[test]
    fn auto_listing_assumptions_cover_every_alternative() {
        let dag = iv_dag();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert!(has_exclusion(&res.required_assumptions, 0));
        assert!(
            res.required_assumptions
                .entries
                .iter()
                .any(|r| r.assumption == antecedent_core::Assumption::CausalMarkov)
        );
    }

    #[test]
    fn auto_claims_keep_caller_declared_assumptions() {
        let dag = iv_dag();
        let auto = AutoIdentifier::new();
        let mut declared = AssumptionSet::new();
        declared.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Consistency,
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let prep = auto.prepare_with_assumptions(&dag, declared).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        for index in 0..res.estimands.len() {
            let chosen = res.narrowed_to(index).unwrap();
            assert!(
                chosen
                    .required_assumptions
                    .entries
                    .iter()
                    .any(|r| r.assumption == antecedent_core::Assumption::Consistency),
                "estimand {index} lost the declared assumption"
            );
        }
    }

    #[test]
    fn auto_rejects_treatment_descendant_instrument() {
        // U → T → Y, U → Y, T → Z: Z is not a valid IV.
        let mut dag = Dag::with_variables(4);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // U→T
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // U→Y
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // T→Y
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap(); // T→Z
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert!(!res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(3)]));
    }

    #[test]
    fn auto_reports_a_path_budget_exit_as_inconclusive() {
        // 0=t, four fully connected layers of three, 13=y: 81 paths against a budget of 64.
        let mut dag = Dag::with_variables(14);
        let layer = |k: u32| (1 + 3 * k)..(4 + 3 * k);
        for v in layer(0) {
            dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(v)).unwrap();
        }
        for k in 0..3 {
            for u in layer(k) {
                for v in layer(k + 1) {
                    dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v))
                        .unwrap();
                }
            }
        }
        for u in layer(3) {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(13)).unwrap();
        }
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::PathSpecific(
            antecedent_core::PathSpecificEffectQuery::binary(
                VariableId::from_raw(0),
                VariableId::from_raw(13),
            )
            .with_path_nodes([VariableId::from_raw(1)]),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(
            !res.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Scientific),
            "a budget exit is not a scientific negative: {:?}",
            res.diagnostics
        );
        let bounded = res
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "auto.path_specific.search_bounded")
            .expect("auto bounded-search diagnostic");
        assert_eq!(bounded.kind, DiagnosticKind::Execution);
        assert_eq!(bounded.severity, DiagnosticSeverity::Warning);
    }

    /// R -> T -> Y with R -> Y: a sharp design on R (variables: 0 = T, 1 = Y, 2 = R).
    fn sharp_design_dag() -> Dag {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag
    }

    fn rd_config() -> SharpRdConfig {
        SharpRdConfig::new(VariableId::from_raw(2), 0.5, 1.0)
    }

    fn is_rd(e: &IdentifiedEstimand) -> bool {
        e.method_kind().ok() == Some(EstimandMethod::RdSharp)
    }

    #[test]
    fn rebuild_never_invents_an_rd_design() {
        // An RD-tagged estimand that carries no design has no running variable or cutoff
        // to rebuild from; substituting one would be a fabricated design.
        let bare = IdentifiedEstimand::backdoor(
            "rd.sharp",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut arena = CausalExprArena::new();
        assert!(
            rebuild_estimand(&mut arena, &bare, &q, &Value::f64(1.0), &Value::f64(0.0)).is_none()
        );
    }

    #[test]
    fn auto_does_not_offer_the_cutoff_effect_as_a_population_effect() {
        let dag = sharp_design_dag();
        let auto = AutoIdentifier::new().with_rd(rd_config());
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert!(!res.estimands.iter().any(is_rd), "RD answers a different question");
        let note = res
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "auto.rd.local_estimand_not_requested")
            .expect("explains why the design was not used");
        assert_eq!(note.kind, DiagnosticKind::Scientific);
        assert!(!res.diagnostics.iter().any(|d| d.code.as_ref() == "auto.rd.missing_config"));
    }

    #[test]
    fn auto_identifies_the_cutoff_effect_only_when_asked_for_it() {
        let dag = sharp_design_dag();
        let auto = AutoIdentifier::new().with_rd(rd_config());
        let prep = auto.prepare(&dag).unwrap();
        let local = rd_config().target_population();
        let q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(local.clone()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        // Graph strategies identify a population effect, not this one, so RD stands alone.
        assert_eq!(res.estimands.len(), 1);
        assert!(is_rd(&res.estimands[0]));
        assert_eq!(res.estimands[0].rd_design.unwrap().cutoff.to_bits(), 0.5f64.to_bits());
        assert_eq!(res.average_effect().unwrap().target_population, local);
        let ids: Vec<&str> = res
            .required_assumptions
            .entries
            .iter()
            .filter_map(|r| match &r.assumption {
                antecedent_core::Assumption::Custom { id, .. } => Some(id.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["rd.continuity", "rd.no_manipulation", "rd.sharp_assignment"]);
    }

    #[test]
    fn auto_without_a_design_does_not_identify_a_cutoff_effect() {
        let dag = sharp_design_dag();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(rd_config().target_population()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
        assert!(res.diagnostics.iter().any(|d| d.code.as_ref() == "auto.rd.missing_config"));
    }

    #[test]
    fn auto_refuses_a_design_the_graph_contradicts() {
        // T has a second cause, so assignment is not a function of R alone.
        let mut dag = Dag::with_variables(4);
        for (u, v) in [(2, 0), (3, 0), (0, 1), (3, 1)] {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        let auto = AutoIdentifier::new().with_rd(rd_config());
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(rd_config().target_population()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(
            res.diagnostics.iter().any(|d| d.code.as_ref() == "identify.rd.graph_incompatible")
        );
    }

    /// Exact law of a binary `T -> Y` SCM with no confounding: `P(T=1) = p_t`,
    /// `P(Y=1 | T=t) = f(t)`. Used to pin the true stochastic-intervention
    /// estimand against the functional Auto actually returns.
    struct BinaryTyLaw {
        p_t: f64,
        f: [f64; 2],
    }

    impl BinaryTyLaw {
        fn bit(assignment: &antecedent_expr::Assignment, var: VariableId) -> f64 {
            assignment.get(var).and_then(Value::as_f64).expect("bound")
        }
    }

    #[allow(
        clippy::float_cmp,
        clippy::map_unwrap_or,
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::precedence
    )]
    impl antecedent_expr::DistributionProvider for BinaryTyLaw {
        fn probability(
            &self,
            spec: &antecedent_expr::FactorSpec<'_>,
            assignment: &antecedent_expr::Assignment,
            _ctx: &antecedent_expr::EvalContext,
        ) -> Result<f64, antecedent_expr::EvalError> {
            let t = VariableId::from_raw(0);
            let y = VariableId::from_raw(1);
            // The empty-adjustment marginal `P(\emptyset)` used by backdoor's
            // Z-marginal factor when the adjustment set is empty.
            if spec.variables.is_empty() {
                return Ok(1.0);
            }
            // Every leaf here is either the marginal of T or the conditional of Y
            // given T (the only two factors a backdoor/general-ID functional on
            // this graph can ask for).
            if spec.variables.contains(&t) {
                let tv = Self::bit(assignment, t);
                return Ok(if tv == 1.0 { self.p_t } else { 1.0 - self.p_t });
            }
            if spec.variables.contains(&y) {
                let tv = spec
                    .conditioned_on
                    .iter()
                    .copied()
                    .chain(spec.intervention.iter().map(|a| a.variable))
                    .find(|&v| v == t)
                    .map(|_| Self::bit(assignment, t))
                    .unwrap_or(self.p_t);
                let yv = Self::bit(assignment, y);
                let p1 = self.f[usize::from(tv == 1.0)];
                return Ok(if yv == 1.0 { p1 } else { 1.0 - p1 });
            }
            Err(antecedent_expr::EvalError::MissingBinding(t))
        }

        fn support(
            &self,
            vars: &[VariableId],
            _ctx: &antecedent_expr::EvalContext,
        ) -> Result<Arc<[Arc<[Value]>]>, antecedent_expr::EvalError> {
            Ok((0..1usize << vars.len())
                .map(|row| {
                    (0..vars.len())
                        .map(|i| Value::f64(f64::from((row >> i & 1) as u8)))
                        .collect::<Arc<[_]>>()
                })
                .collect())
        }

        fn outcome(
            &self,
            var: VariableId,
            assignment: &antecedent_expr::Assignment,
            _ctx: &antecedent_expr::EvalContext,
        ) -> Result<f64, antecedent_expr::EvalError> {
            assignment
                .get(var)
                .and_then(Value::as_f64)
                .ok_or(antecedent_expr::EvalError::MissingBinding(var))
        }

        fn n_draws(&self) -> Option<usize> {
            None
        }
    }

    #[test]
    fn auto_bernoulli_mixture_ate_matches_the_true_stochastic_contrast() {
        // T -> Y, no confounding: Y = T exactly (f(0)=0, f(1)=1), so the true
        // hard contrast E[Y|do(1)] - E[Y|do(0)] is 1. A genuine Bernoulli
        // mixture active=Bernoulli(0.7) vs control=Bernoulli(0.2) has true
        // contrast (0.7-0.2)*1 = 0.5 (E[Y|do(Bernoulli(p))] = p, linear in p),
        // not the unscaled hard contrast 1.0.
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            Arc::from([]),
            Intervention::stochastic(
                VariableId::from_raw(0),
                antecedent_core::StochasticPolicy::Bernoulli { p: 0.2 },
            ),
            Intervention::stochastic(
                VariableId::from_raw(0),
                antecedent_core::StochasticPolicy::Bernoulli { p: 0.7 },
            ),
            antecedent_core::TargetPopulation::AllObserved,
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();

        let law = BinaryTyLaw { p_t: 0.5, f: [0.0, 1.0] };
        let ctx = antecedent_expr::EvalContext::default();
        let true_contrast = 0.5;
        let tol = 1e-9;

        // The identification must not silently ship the unscaled hard
        // contrast as if it were the stochastic estimand: whatever the
        // status, no returned estimand may evaluate to the wrong number.
        for estimand in &res.estimands {
            let plan = res.arena.compile(estimand.functional).unwrap();
            let value = plan.evaluate(&res.arena, &law, &ctx).unwrap();
            assert!(
                (value - true_contrast).abs() < tol,
                "estimand {} evaluated to {value}, true stochastic contrast is {true_contrast}",
                estimand.method
            );
        }
        // The crate cannot yet scale a functional by the mixture weight, so it
        // must refuse rather than mislabel the unscaled hard contrast as the
        // requested stochastic effect.
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "auto.stochastic.bernoulli_scale_unsupported"),
            "{:?}",
            res.diagnostics
        );
    }

    #[test]
    fn auto_degenerate_bernoulli_pair_still_identifies_the_hard_contrast() {
        // active=Bernoulli(1), control=Bernoulli(0) is exactly do(1) vs do(0)
        // in disguise (scale = 1 - 0 = 1): the existing hard-contrast path
        // must keep working without the new refusal firing.
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let auto = AutoIdentifier::new();
        let prep = auto.prepare(&dag).unwrap();
        let q = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            Arc::from([]),
            Intervention::stochastic(
                VariableId::from_raw(0),
                antecedent_core::StochasticPolicy::Bernoulli { p: 0.0 },
            ),
            Intervention::stochastic(
                VariableId::from_raw(0),
                antecedent_core::StochasticPolicy::Bernoulli { p: 1.0 },
            ),
            antecedent_core::TargetPopulation::AllObserved,
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = auto.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!res.estimands.is_empty());
        assert!(
            res.diagnostics.iter().any(|d| d.code.as_ref() == "auto.stochastic.bernoulli_scale")
        );

        let law = BinaryTyLaw { p_t: 0.5, f: [0.0, 1.0] };
        let ctx = antecedent_expr::EvalContext::default();
        for estimand in &res.estimands {
            let plan = res.arena.compile(estimand.functional).unwrap();
            let value = plan.evaluate(&res.arena, &law, &ctx).unwrap();
            assert!(
                (value - 1.0).abs() < 1e-9,
                "estimand {} evaluated to {value}",
                estimand.method
            );
        }
    }
}
