//! Sharp regression-discontinuity identification.
//!
//! A sharp design assigns treatment by a threshold on a running variable,
//! `T = 1{R ≥ c}`. Under continuity of the potential-outcome regressions
//! `E[Y(t) | R = r]` at `c`, the design identifies exactly one effect: the average
//! effect for units at the cutoff,
//!
//! `τ_c = lim_{r↓c} E[Y | R = r] − lim_{r↑c} E[Y | R = r]`
//!
//! (Hahn, Todd & van der Klaauw 2001). It does not identify the population average
//! effect, nor the effect at any other value of `R`: away from the cutoff one arm is
//! never observed. The result therefore always carries
//! [`TargetPopulation::LocalAtCutoff`] as its target, whatever population the caller
//! asked for, and says so in a diagnostic when it had to relabel.
//!
//! The design is supplied by the caller. It is never inferred, and the graph — when one
//! is given — must agree with it: the running variable is the treatment's only parent.
//! What is left to declare is recorded as assumptions: continuity at the cutoff, no
//! manipulation of the running variable around it, and sharp assignment (which the
//! estimator verifies row by row against the treatment column).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::manual_let_else, clippy::needless_pass_by_value)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, TargetPopulation, VariableId,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::Dag;

use crate::backdoor::var_to_dense;
use crate::error::IdentificationError;
use crate::result::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

/// Assumption id: potential-outcome regressions are continuous in `R` at the cutoff.
pub const RD_CONTINUITY_ID: &str = "rd.continuity";
/// Assumption id: units cannot sort themselves across the cutoff.
pub const RD_NO_MANIPULATION_ID: &str = "rd.no_manipulation";
/// Assumption id: treatment is the deterministic threshold rule `T = 1{R ≥ c}`.
pub const RD_SHARP_ASSIGNMENT_ID: &str = "rd.sharp_assignment";
/// Diagnostic code: the identified estimand is the effect at the cutoff.
pub const RD_LOCAL_ESTIMAND_DIAGNOSTIC_CODE: &str = "identify.rd.local_estimand";
/// Diagnostic code: the graph contradicts the declared design.
pub const RD_GRAPH_INCOMPATIBLE_DIAGNOSTIC_CODE: &str = "identify.rd.graph_incompatible";
/// Diagnostic code: the requested population is not the one the design speaks for.
pub const RD_POPULATION_DIAGNOSTIC_CODE: &str = "identify.rd.population_not_identified";

/// Configuration for sharp RD identification.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SharpRdConfig {
    /// Running variable.
    pub running_variable: VariableId,
    /// Cutoff threshold.
    pub cutoff: f64,
    /// Bandwidth around the cutoff.
    pub bandwidth: f64,
}

impl SharpRdConfig {
    /// Construct an RD identification config.
    #[must_use]
    pub const fn new(running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        Self { running_variable, cutoff, bandwidth }
    }

    /// The population this design identifies an effect for.
    #[must_use]
    pub fn target_population(&self) -> TargetPopulation {
        TargetPopulation::local_at_cutoff(self.running_variable, self.cutoff)
    }
}

/// Identifier for sharp regression discontinuity designs.
///
/// Identifies the average effect for units at the cutoff
/// ([`TargetPopulation::LocalAtCutoff`]) under explicit design assumptions. It never
/// claims a population average effect.
#[derive(Clone, Debug)]
pub struct SharpRdIdentifier {
    /// Design configuration.
    pub config: SharpRdConfig,
}

impl SharpRdIdentifier {
    /// Construct.
    #[must_use]
    pub const fn new(config: SharpRdConfig) -> Self {
        Self { config }
    }

    /// Identify the effect at the cutoff from the declared design alone.
    ///
    /// Use [`Self::identify_on`] when a causal graph is available: it additionally
    /// requires the graph to agree with the design.
    ///
    /// # Errors
    ///
    /// Query is not an average-effect query, or the design config is invalid.
    pub fn identify(
        &self,
        query: CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        self.identify_inner(None, &query)
    }

    /// Identify the effect at the cutoff, requiring `graph` to agree with the design.
    ///
    /// A sharp design says `T` is a deterministic function of `R` alone, so in the graph
    /// `R` must be the treatment's only parent. A graph that omits `R → T`, or gives `T`
    /// another cause, describes a different assignment mechanism; the result is then
    /// `NotIdentified` with an `identify.rd.graph_incompatible` diagnostic.
    ///
    /// # Errors
    ///
    /// Query is not an average-effect query, the design config is invalid, or a variable
    /// is not in the graph.
    pub fn identify_on(
        &self,
        graph: &Dag,
        query: CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        self.identify_inner(Some(graph), &query)
    }

    fn identify_inner(
        &self,
        graph: Option<&Dag>,
        query: &CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "sharp RD identifier requires an average-effect query",
            });
        };
        let ate = ate.clone();
        let cfg = self.config;
        let (active, control) = validate_design(&cfg, &ate)?;

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "rd.sharp",
            format!(
                "sharp RD at cutoff={} bandwidth={} on running variable {:?}",
                cfg.cutoff, cfg.bandwidth, cfg.running_variable
            ),
        );
        // The design speaks for units at the cutoff and for no one else.
        let local = cfg.target_population();
        let relabelled = match &ate.target_population {
            TargetPopulation::AllObserved => true,
            population if *population == local => false,
            other => {
                return Ok(refused(
                    query,
                    derivation,
                    RD_POPULATION_DIAGNOSTIC_CODE,
                    format!(
                        "a sharp RD design identifies the effect for units at the cutoff only; \
                         the requested target population {other:?} is not identified by it"
                    ),
                ));
            }
        };

        if let Some(graph) = graph {
            if let Some(problem) = graph_conflict(graph, &ate, cfg.running_variable)? {
                derivation.push("rd.sharp.graph", problem.clone());
                return Ok(refused(
                    query,
                    derivation,
                    RD_GRAPH_INCOMPATIBLE_DIAGNOSTIC_CODE,
                    problem,
                ));
            }
            derivation.push(
                "rd.sharp.graph",
                "the running variable is the treatment's only parent, as a threshold rule requires",
            );
        }

        let mut arena = CausalExprArena::new();
        let functional = arena.rd_sharp_local_effect(
            ate.treatment,
            ate.outcome,
            cfg.running_variable,
            cfg.cutoff,
            active,
            control,
        );
        let estimand = IdentifiedEstimand::rd_sharp(
            functional,
            antecedent_expr::RdDesignParams::new(cfg.running_variable, cfg.cutoff, cfg.bandwidth),
        );
        derivation.push(
            "rd.sharp.estimand",
            "effect for units at the cutoff: difference of the one-sided limits of E[Y | R = r]",
        );

        let identified_query = CausalQuery::AverageEffect(ate.with_target_population(local));
        let mut out = IdentificationResult::identified(
            identified_query,
            vec![estimand],
            arena,
            derivation,
            design_assumptions(),
            IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 1 },
        );
        out.diagnostics.push(if relabelled {
            Diagnostic::new(
                RD_LOCAL_ESTIMAND_DIAGNOSTIC_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the population average effect is not identified by a sharp RD design; the \
                     identified estimand is the average effect for units at {:?} = {} and says \
                     nothing about units away from the cutoff",
                    cfg.running_variable, cfg.cutoff
                ),
            )
        } else {
            Diagnostic::new(
                RD_LOCAL_ESTIMAND_DIAGNOSTIC_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "identified estimand: average effect for units at {:?} = {}; a sharp RD \
                     design does not identify the population average effect or the effect away \
                     from the cutoff",
                    cfg.running_variable, cfg.cutoff
                ),
            )
        });
        Ok(out)
    }
}

/// Check the design parameters against the query; return the normalized `(active, control)`.
fn validate_design(
    cfg: &SharpRdConfig,
    ate: &AverageEffectQuery,
) -> Result<(antecedent_core::Value, antecedent_core::Value), IdentificationError> {
    if !cfg.bandwidth.is_finite() || cfg.bandwidth <= 0.0 {
        return Err(IdentificationError::UnsupportedQuery {
            message: "sharp RD bandwidth must be finite and positive",
        });
    }
    if !cfg.cutoff.is_finite() {
        return Err(IdentificationError::UnsupportedQuery {
            message: "sharp RD cutoff must be finite",
        });
    }
    if cfg.running_variable == ate.treatment || cfg.running_variable == ate.outcome {
        return Err(IdentificationError::UnsupportedQuery {
            message: "sharp RD running variable must differ from the treatment and the outcome",
        });
    }
    match (
        crate::intervention_support::normalize_to_set(&ate.active),
        crate::intervention_support::normalize_to_set(&ate.control),
    ) {
        (
            Ok(antecedent_core::Intervention::Set { value: active, .. }),
            Ok(antecedent_core::Intervention::Set { value: control, .. }),
        ) => Ok((active, control)),
        _ => Err(IdentificationError::UnsupportedQuery {
            message: "sharp RD requires Set (or Soft(constant)/Shift) interventions",
        }),
    }
}

/// `NotIdentified` for a completed, scientific reason (`code`, `message`).
fn refused(
    query: &CausalQuery,
    derivation: DerivationTrace,
    code: &'static str,
    message: String,
) -> IdentificationResult {
    let mut out = IdentificationResult::not_identified(
        query.clone(),
        derivation,
        AssumptionSet::new(),
        IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 0 },
    );
    out.diagnostics.push(Diagnostic::new(
        code,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        message,
    ));
    out
}

/// Why `graph` cannot host a sharp design on `running`, if it cannot.
fn graph_conflict(
    graph: &Dag,
    ate: &AverageEffectQuery,
    running: VariableId,
) -> Result<Option<String>, IdentificationError> {
    let t = var_to_dense(ate.treatment, graph)?;
    let r = var_to_dense(running, graph)?;
    let parents = graph.parents(t);
    if !parents.contains(&r) {
        return Ok(Some(format!(
            "the graph has no edge from the running variable {running:?} into the treatment \
             {:?}, so it does not describe a threshold assignment on that variable",
            ate.treatment
        )));
    }
    if parents.len() > 1 {
        return Ok(Some(format!(
            "the treatment {:?} has {} parents in the graph; under sharp assignment it is a \
             deterministic function of the running variable {running:?} alone (a treatment with \
             other causes is a fuzzy design)",
            ate.treatment,
            parents.len()
        )));
    }
    Ok(None)
}

fn design_assumptions() -> AssumptionSet {
    let record = |id: &str, description: &str, status: AssumptionStatus| AssumptionRecord {
        assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("rd.sharp") },
        scope: AssumptionScope::Identification,
        status,
    };
    let mut assumptions = AssumptionSet::new();
    assumptions.push(record(
        RD_CONTINUITY_ID,
        "the potential-outcome regressions E[Y(1) | R = r] and E[Y(0) | R = r] are continuous \
         in the running variable at the cutoff",
        AssumptionStatus::Untestable,
    ));
    assumptions.push(record(
        RD_NO_MANIPULATION_ID,
        "units cannot precisely manipulate the running variable to sort across the cutoff",
        AssumptionStatus::Declared,
    ));
    assumptions.push(record(
        RD_SHARP_ASSIGNMENT_ID,
        "treatment is the deterministic threshold rule T = 1{R >= c}; the estimator checks it \
         against the treatment column",
        AssumptionStatus::Declared,
    ));
    assumptions
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_expr::ExprNode;
    use antecedent_graph::DenseNodeId;

    use super::*;
    use crate::result::IdentificationStatus;

    const T: u32 = 0;
    const Y: u32 = 1;
    const R: u32 = 2;

    fn identifier() -> SharpRdIdentifier {
        SharpRdIdentifier::new(SharpRdConfig::new(VariableId::from_raw(R), 0.5, 1.0))
    }

    fn ate() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y))
    }

    fn dag(edges: &[(u32, u32)], n: u32) -> Dag {
        let mut g = Dag::with_variables(n);
        for &(u, v) in edges {
            g.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        g
    }

    fn assumption_ids(result: &IdentificationResult) -> Vec<&str> {
        result
            .required_assumptions
            .entries
            .iter()
            .filter_map(|r| match &r.assumption {
                Assumption::Custom { id, .. } => Some(id.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn population_ate_request_is_answered_with_the_cutoff_effect_and_says_so() {
        let result = identifier().identify(CausalQuery::AverageEffect(ate())).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        let identified = result.average_effect().unwrap();
        assert_eq!(
            identified.target_population,
            TargetPopulation::local_at_cutoff(VariableId::from_raw(R), 0.5)
        );
        assert_ne!(identified.target_population, TargetPopulation::AllObserved);
        let note = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == RD_LOCAL_ESTIMAND_DIAGNOSTIC_CODE)
            .expect("local-estimand diagnostic");
        assert_eq!(note.kind, DiagnosticKind::Scientific);
        assert_eq!(note.severity, DiagnosticSeverity::Warning);
        assert_eq!(
            assumption_ids(&result),
            vec![RD_CONTINUITY_ID, RD_NO_MANIPULATION_ID, RD_SHARP_ASSIGNMENT_ID]
        );
    }

    #[test]
    fn explicit_cutoff_population_is_identified_without_a_relabel_warning() {
        let query = ate().with_target_population(identifier().config.target_population());
        let result = identifier().identify(CausalQuery::AverageEffect(query.clone())).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.average_effect().unwrap().target_population, query.target_population);
        assert!(result.diagnostics.iter().all(|d| d.severity != DiagnosticSeverity::Warning));
    }

    #[test]
    fn other_populations_are_not_identified() {
        let treated = ate().with_target_population(TargetPopulation::Treated);
        let other_cutoff = ate().with_target_population(TargetPopulation::local_at_cutoff(
            VariableId::from_raw(R),
            0.25,
        ));
        for query in [treated, other_cutoff] {
            let result = identifier().identify(CausalQuery::AverageEffect(query)).unwrap();
            assert_eq!(result.status, IdentificationStatus::NotIdentified);
            assert!(result.estimands.is_empty());
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == RD_POPULATION_DIAGNOSTIC_CODE)
            );
        }
    }

    #[test]
    fn functional_is_the_one_sided_limit_contrast_at_the_cutoff() {
        let result = identifier().identify(CausalQuery::AverageEffect(ate())).unwrap();
        let mut naive_arena = result.arena.clone();
        let naive = naive_arena.backdoor_ate(
            VariableId::from_raw(T),
            VariableId::from_raw(Y),
            &[],
            antecedent_core::Value::f64(1.0),
            antecedent_core::Value::f64(0.0),
        );
        let functional = result.estimands[0].functional;
        assert_ne!(functional, naive, "the RD functional is not the unadjusted contrast");
        let ExprNode::Contrast { left, right, .. } = result.arena.node(functional).clone() else {
            panic!("RD functional must be a contrast");
        };
        for side in [left, right] {
            let ExprNode::Expectation { distribution, .. } = result.arena.node(side).clone() else {
                panic!("each side is a conditional mean");
            };
            let ExprNode::Distribution { conditioned_on, intervention, .. } =
                result.arena.node(distribution).clone()
            else {
                panic!("each side is one observational factor");
            };
            assert!(result.arena.var_set(conditioned_on).contains(&VariableId::from_raw(R)));
            assert!(result.arena.intervention_assignments(intervention).iter().any(|b| {
                b.variable == VariableId::from_raw(R) && b.value.as_f64() == Some(0.5)
            }));
        }
        assert_eq!(result.estimands[0].rd_design.unwrap().cutoff.to_bits(), 0.5f64.to_bits());
    }

    #[test]
    fn graph_must_make_the_running_variable_the_only_cause_of_treatment() {
        let q = || CausalQuery::AverageEffect(ate());
        // R -> T -> Y with R -> Y: the textbook sharp design.
        let sharp = dag(&[(R, T), (T, Y), (R, Y)], 3);
        let ok = identifier().identify_on(&sharp, q()).unwrap();
        assert_eq!(ok.status, IdentificationStatus::NonparametricallyIdentified);

        // No R -> T edge: the graph does not describe a threshold rule on R.
        let missing = dag(&[(T, Y), (R, Y)], 3);
        // A second cause of T: assignment is not a function of R alone (fuzzy).
        let fuzzy = dag(&[(R, T), (3, T), (T, Y), (3, Y)], 4);
        for graph in [missing, fuzzy] {
            let res = identifier().identify_on(&graph, q()).unwrap();
            assert_eq!(res.status, IdentificationStatus::NotIdentified);
            assert!(res.estimands.is_empty());
            let d = res
                .diagnostics
                .iter()
                .find(|d| d.code.as_ref() == RD_GRAPH_INCOMPATIBLE_DIAGNOSTIC_CODE)
                .expect("graph-incompatible diagnostic");
            assert_eq!(d.kind, DiagnosticKind::Scientific);
        }
    }

    #[test]
    fn running_variable_cannot_be_the_treatment_or_the_outcome() {
        for running in [T, Y] {
            let id =
                SharpRdIdentifier::new(SharpRdConfig::new(VariableId::from_raw(running), 0.0, 1.0));
            assert!(matches!(
                id.identify(CausalQuery::AverageEffect(ate())),
                Err(IdentificationError::UnsupportedQuery { .. })
            ));
        }
    }
}
