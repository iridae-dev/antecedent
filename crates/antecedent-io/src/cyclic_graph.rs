// SPDX-License-Identifier: MIT OR Apache-2.0
//! Serializable cyclic graph reasoning requests, with theorem-scoped answers.
//! Graphical identification/discovery never inherits a numerical model-fit claim.

use crate::cyclic_system_artifact::CyclicArtifactError;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{CyclicMixedGraph, DenseNodeId, Endpoint};
use antecedent_identify::cyclic_id::{CyclicIdPremises, cyclic_backdoor_valid, identify_cyclic};
use serde::{Deserialize, Serialize};

/// Dense cyclic graph serialization; reciprocal and bidirected edges are distinct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicGraphSpec {
    /// Number of dense observed variables, at most 32.
    pub nodes: u32,
    /// Original directed mechanism edges, including reciprocal pairs.
    pub directed: Vec<(u32, u32)>,
    /// Latent-confounding edges, each unordered pair listed once.
    pub bidirected: Vec<(u32, u32)>,
}
fn invalid(e: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(e.to_string())
}
impl CyclicGraphSpec {
    /// Build a validated graph, refusing unknown/self/duplicate edges.
    pub fn graph(&self) -> Result<CyclicMixedGraph, CyclicArtifactError> {
        if self.nodes == 0
            || self.nodes > 32
            || self.directed.len() > 1024
            || self.bidirected.len() > 512
        {
            return Err(invalid("cyclic graph bounds"));
        }
        let mut graph = CyclicMixedGraph::with_variables(self.nodes);
        for &(a, b) in &self.directed {
            graph
                .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(invalid)?;
        }
        for &(a, b) in &self.bidirected {
            graph
                .insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(invalid)?;
        }
        Ok(graph)
    }
}

/// Retained graph reasoning/discovery program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicGraphRequest {
    /// One theorem-scoped observation/action rewrite with intervention indicators.
    Calculus {
        /// Original cyclic mixed graph.
        graph: CyclicGraphSpec,
        /// Theorem 7.2 rule: observation, exchange or action.
        rule: antecedent_identify::cyclic_calculus::CyclicCalculusRule,
        /// Variables whose observation or action is rewritten.
        x: Vec<usize>,
        /// Outcome variables.
        y: Vec<usize>,
        /// Observed conditioning variables.
        z: Vec<usize>,
        /// Baseline hard interventions, represented as fixed input nodes.
        w: Vec<usize>,
        /// Explicit compatible ioSCM premise.
        io_scm: bool,
        /// Standard measurable spaces admit the regular conditionals used by the theorem.
        standard_measurable_spaces: bool,
        /// Maximum pairwise sigma checks, distinct from mathematical failure.
        max_checks: usize,
    },
    /// Executable cyclic backdoor adjustment, Corollary 8.4(4).
    Adjust {
        /// Original cyclic mixed graph.
        graph: CyclicGraphSpec,
        /// Outcome variables.
        outcomes: Vec<usize>,
        /// Hard intervention coordinates.
        treatments: Vec<usize>,
        /// Declared adjustment set.
        adjustment: Vec<usize>,
        /// Integral or finite sum marginalization.
        continuous: bool,
        /// Explicit compatible ioSCM premise.
        io_scm: bool,
        /// Standard measurable spaces for conditional probabilities.
        standard_measurable_spaces: bool,
    },
    /// Equation-variable causal ordering and determinism-aware B-separation.
    Bipartite {
        /// Number of dense variables (including exogenous inputs).
        variables: usize,
        /// Independent exogenous input coordinates for the Markov interpretation.
        exogenous: Vec<usize>,
        /// Variable incidence of each named-equation position.
        equations: Vec<Vec<usize>>,
        /// Optional `(equation,variable)` hard replacement before graphical reasoning.
        replacement: Option<(usize, usize)>,
        /// First query endpoint.
        x: usize,
        /// Second query endpoint.
        y: usize,
        /// Conditioning variables; deterministic closure is computed internally.
        conditioning: Vec<usize>,
    },
    /// Domain invariance from a selector input attached to changed mechanisms.
    DomainInvariance {
        /// Variables before adding the domain-selector input.
        variables: usize,
        /// Independent exogenous inputs of the original model.
        exogenous: Vec<usize>,
        /// Equation incidence before attaching the selector.
        equations: Vec<Vec<usize>>,
        /// Equations allowed to vary across domains.
        changed_equations: Vec<usize>,
        /// Outcome variable coordinates.
        outcomes: Vec<usize>,
        /// Variables conditioned upon in the requested invariance.
        conditioning: Vec<usize>,
    },
    /// Generalized observation-only ioSCM ID.
    Identify {
        /// Original cyclic mixed graph.
        graph: CyclicGraphSpec,
        /// Outcome coordinates.
        outcomes: Vec<usize>,
        /// Intervention coordinates.
        treatments: Vec<usize>,
        /// Integral rather than sum marginalization.
        continuous: bool,
        /// Recursive step budget.
        max_steps: usize,
        /// Explicit ioSCM premise.
        io_scm: bool,
        /// Explicit product-SCC-density premise.
        scc_product_density: bool,
    },
    /// Sufficient sigma-backdoor criterion (not a completeness test).
    Backdoor {
        /// Original mixed graph.
        graph: CyclicGraphSpec,
        /// Outcomes.
        outcomes: Vec<usize>,
        /// Hard intervention targets.
        treatments: Vec<usize>,
        /// Proposed adjustment set.
        adjustment: Vec<usize>,
        /// Explicit ioSCM premise.
        io_scm: bool,
        /// Explicit product-SCC-density premise.
        scc_product_density: bool,
    },
    /// FCI under cyclic interpretation, either sampled CI or an explicit reference oracle.
    Discover {
        /// Semantic CI/discovery seed, independent of consumer context.
        #[serde(default)]
        seed: u64,
        /// Observed columns in dense variable order.
        columns: Vec<Vec<f64>>,
        /// Optional exact sigma-CI reference graph; not statistical discovery evidence.
        oracle: Option<CyclicGraphSpec>,
        /// Existing CI factory id, ignored only for an explicit exact oracle.
        ci: String,
        /// CI alpha in (0,1).
        alpha: f64,
        /// Conditioning-set search bound.
        max_cond_size: usize,
        /// Possible-D-Sep expansion bound.
        pds_max_nodes: usize,
        /// Explicit simple-SCM premise.
        simple_scm: bool,
        /// Explicit sigma-faithfulness premise.
        sigma_faithful: bool,
        /// Independent equilibrium replicates; no dependent-series substitution.
        independent_equilibrium_replicates: bool,
    },
}

/// Typed graph answer, preserving identification/search and inducing-path meanings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicGraphReport {
    /// Checked calculus rewrite; remaining intervention factors are not called identified.
    Calculus {
        /// `verified_rule`, `not_licensed_by_criterion` or `budget_exhausted`.
        status: String,
        /// Whether the executable rewrite has no remaining intervention factors.
        fully_observational: bool,
        /// Checked source/executable functional mapping only after full verification.
        program: Option<crate::FunctionalProgramWire>,
        /// Human-readable executable expression.
        expression: Option<String>,
        /// Actual bounded sigma checks performed.
        sigma_checks: usize,
        /// Original theorem conditions and graph transformation.
        derivation: Vec<String>,
    },
    /// Graphical B-separation and its explicit unique-solution/independence premises.
    Bipartite {
        /// Graphical separation result.
        separated: bool,
        /// Variables determined by conditioning.
        determined: Vec<usize>,
        /// Variable causal-ordering clusters.
        variable_clusters: Vec<usize>,
        /// Equation causal-ordering clusters.
        equation_clusters: Vec<usize>,
        /// Scientific scope of the Markov interpretation.
        assumptions: String,
    },
    /// Domain-selector separation, conditional on stated BGCM semantics.
    DomainInvariance {
        /// Every outcome B-separated from the domain selector given conditioning.
        invariant: bool,
        /// Required scientific premises, not inferred from equation incidence.
        assumptions: String,
    },
    /// A retained functional or explicit unsuccessful identification attempt.
    Identify {
        /// `identified`, `not_identified_by_procedure`, or `budget_exhausted`.
        status: String,
        /// Readable rendering of the checked functional IR, when identified.
        expression: Option<String>,
        /// Checked functional IR for typed independent consumers.
        program: Option<crate::FunctionalProgramWire>,
        /// Theorem scope and recursive derivation.
        derivation: Vec<String>,
    },
    /// Sufficient-criterion result; false is not a non-identifiability proof.
    Backdoor {
        /// The proposed set passed both sigma-separation checks.
        valid: bool,
    },
    /// PAG equivalence-class evidence, not a recovered original graph/effect magnitude.
    Discover {
        /// Theorem-scoped ancestral and direct-edge conclusions; unproved relations omitted.
        conclusions: Vec<(u32, u32, String, String)>,
        /// `(node_a,node_b,mark_a,mark_b)` in dense order.
        edges: Vec<(u32, u32, String, String)>,
        /// Existing engine diagnostics, including search truncation where reported.
        diagnostics: Vec<String>,
        /// Cyclic PAG semantic boundary.
        interpretation: String,
        /// `sigma_oracle` or the sampled CI name.
        ci_source: String,
    },
}
fn mark(e: Endpoint) -> String {
    match e {
        Endpoint::Tail => "tail",
        Endpoint::Arrow => "arrow",
        Endpoint::Circle => "circle",
        Endpoint::Conflict => "conflict",
    }
    .into()
}
impl CyclicGraphRequest {
    /// Execute through existing identification/discovery engines.
    #[allow(
        clippy::too_many_lines,
        reason = "typed scientific family dispatch retains distinct operator inputs and answer representations"
    )]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicGraphReport, CyclicArtifactError> {
        match self {
            Self::Calculus {
                graph,
                rule,
                x,
                y,
                z,
                w,
                io_scm,
                standard_measurable_spaces,
                max_checks,
            } => {
                use antecedent_identify::cyclic_calculus::{
                    CyclicCalculusStatus, apply_cyclic_calculus,
                };
                let answer = apply_cyclic_calculus(
                    &graph.graph()?,
                    *rule,
                    x,
                    y,
                    z,
                    w,
                    *io_scm,
                    *standard_measurable_spaces,
                    *max_checks,
                    ctx,
                )
                .map_err(invalid)?;
                let status = match answer.status {
                    CyclicCalculusStatus::Verified => "verified_rule",
                    CyclicCalculusStatus::NotLicensedByCriterion => "not_licensed_by_criterion",
                    CyclicCalculusStatus::BudgetExhausted => "budget_exhausted",
                }
                .into();
                let expression =
                    answer.program.as_ref().map(|p| p.arena().pretty(p.mapping().executable));
                let program = answer
                    .program
                    .as_ref()
                    .map(crate::functional_program_to_wire)
                    .transpose()
                    .map_err(invalid)?;
                Ok(CyclicGraphReport::Calculus {
                    status,
                    fully_observational: w.is_empty(),
                    program,
                    expression,
                    sigma_checks: answer.sigma_checks,
                    derivation: answer.derivation,
                })
            }
            Self::Adjust {
                graph,
                outcomes,
                treatments,
                adjustment,
                continuous,
                io_scm,
                standard_measurable_spaces,
            } => {
                if !standard_measurable_spaces {
                    return Err(invalid(
                        "adjustment requires the standard-measurable-space premise",
                    ));
                }
                let program = antecedent_identify::cyclic_calculus::identify_cyclic_adjustment(
                    &graph.graph()?,
                    outcomes,
                    treatments,
                    adjustment,
                    *continuous,
                    CyclicIdPremises { io_scm: *io_scm, scc_product_density: false },
                )
                .map_err(invalid)?;
                let status =
                    if program.is_some() { "identified" } else { "not_licensed_by_adjustment" }
                        .into();
                let expression = program.as_ref().map(|p| p.arena().pretty(p.mapping().executable));
                let program = program
                    .as_ref()
                    .map(crate::functional_program_to_wire)
                    .transpose()
                    .map_err(invalid)?;
                Ok(CyclicGraphReport::Identify{status,program,expression,derivation:vec!["Forre-Mooij 2020 Corollary 8.4(4); ioSCM, standard measurable spaces, common support. Product-SCC density is not required for adjustment.".into()]})
            }

            Self::Bipartite {
                variables,
                exogenous,
                equations,
                replacement,
                x,
                y,
                conditioning,
            } => {
                admit_bipartite_size(*variables, equations, "bipartite")?;
                let mut graph = antecedent_graph::bipartite::BipartiteCausalGraph::new(
                    *variables,
                    exogenous,
                    equations.clone(),
                )
                .map_err(invalid)?;
                if let Some((equation, variable)) = replacement {
                    graph = graph.replace_equation(*equation, *variable).map_err(invalid)?;
                }
                let result = graph.b_separates(*x, *y, conditioning).map_err(invalid)?;
                Ok(CyclicGraphReport::Bipartite {
                    separated: result.separated,
                    determined: result.determined,
                    variable_clusters: graph.variable_clusters().to_vec(),
                    equation_clusters: graph.equation_clusters().to_vec(),
                    assumptions: result.assumptions.into(),
                })
            }
            Self::DomainInvariance {
                variables,
                exogenous,
                equations,
                changed_equations,
                outcomes,
                conditioning,
            } => {
                // The selector input is one more variable than the request names.
                admit_bipartite_size(variables.saturating_add(1), equations, "domain invariance")?;
                if outcomes.is_empty()
                    || outcomes.iter().chain(conditioning).any(|&v| v >= *variables)
                    || changed_equations.iter().any(|&e| e >= equations.len())
                {
                    return Err(invalid("domain invariance coordinates"));
                }
                let selector = *variables;
                let mut incidence = equations.clone();
                for &equation in changed_equations {
                    if !incidence[equation].contains(&selector) {
                        incidence[equation].push(selector);
                    }
                }
                let mut external = exogenous.clone();
                external.push(selector);
                let graph = antecedent_graph::bipartite::BipartiteCausalGraph::new(
                    variables + 1,
                    &external,
                    incidence,
                )
                .map_err(invalid)?;
                let mut invariant = true;
                for &outcome in outcomes {
                    invariant &= graph
                        .b_separates(outcome, selector, conditioning)
                        .map_err(invalid)?
                        .separated;
                }
                Ok(CyclicGraphReport::DomainInvariance {invariant,assumptions:"BGCM domain-invariance recipe with a nonrandom selector input; every domain must have compatible measurable unique cluster solutions; original exogenous inputs independent; only named equations change".into()})
            }
            Self::Identify {
                graph,
                outcomes,
                treatments,
                continuous,
                max_steps,
                io_scm,
                scc_product_density,
            } => {
                let graph = graph.graph()?;
                let answer = identify_cyclic(
                    &graph,
                    outcomes,
                    treatments,
                    CyclicIdPremises { io_scm: *io_scm, scc_product_density: *scc_product_density },
                    *continuous,
                    *max_steps,
                    ctx,
                )
                .map_err(invalid)?;
                let status = match answer.status {
                    antecedent_identify::cyclic_id::CyclicIdStatus::Identified => "identified",
                    antecedent_identify::cyclic_id::CyclicIdStatus::NotIdentifiedByProcedure => {
                        "not_identified_by_procedure"
                    }
                    antecedent_identify::cyclic_id::CyclicIdStatus::BudgetExhausted => {
                        "budget_exhausted"
                    }
                }
                .into();
                let expression =
                    answer.program.as_ref().map(|p| p.arena().pretty(p.mapping().executable));
                let program = answer
                    .program
                    .as_ref()
                    .map(crate::functional_program_to_wire)
                    .transpose()
                    .map_err(invalid)?;
                Ok(CyclicGraphReport::Identify {
                    program,
                    status,
                    expression,
                    derivation: answer.derivation,
                })
            }
            Self::Backdoor {
                graph,
                outcomes,
                treatments,
                adjustment,
                io_scm,
                scc_product_density,
            } => {
                let graph = graph.graph()?;
                let valid = cyclic_backdoor_valid(
                    &graph,
                    outcomes,
                    treatments,
                    adjustment,
                    CyclicIdPremises { io_scm: *io_scm, scc_product_density: *scc_product_density },
                )
                .map_err(invalid)?;
                Ok(CyclicGraphReport::Backdoor { valid })
            }
            Self::Discover {
                seed,
                columns,
                oracle,
                ci,
                alpha,
                max_cond_size,
                pds_max_nodes,
                simple_scm,
                sigma_faithful,
                independent_equilibrium_replicates,
            } => {
                use antecedent_discovery::{
                    cyclic_fci::{CyclicFciPremises, SigmaCiOracle, discover_cyclic_fci},
                    engine::DiscoveryWorkspace,
                    fci::Fci,
                };
                let n = columns.len();
                if n == 0
                    || n > 32
                    || !alpha.is_finite()
                    || *alpha <= 0.0
                    || *alpha >= 1.0
                    || *max_cond_size > n
                    || *pds_max_nodes > 100_000
                    || ci.len() > 64
                    || columns.iter().any(|c| {
                        c.len() < 4 || c.len() > 10_000 || c.iter().any(|v| !v.is_finite())
                    })
                {
                    return Err(invalid("cyclic discovery input limits"));
                }
                let data = antecedent_data::TabularData::from_f64_columns(
                    columns.iter().enumerate().map(|(i, c)| (format!("v{i}"), c.as_slice())),
                )
                .map_err(invalid)?;
                let mut engine = Fci::new();
                engine.constraints.alpha = *alpha;
                engine.constraints.max_cond_size = *max_cond_size;
                engine.pds_max_nodes = *pds_max_nodes;
                let ci_source = if let Some(graph) = oracle {
                    if graph.nodes as usize != n {
                        return Err(invalid("sigma oracle/column binding size"));
                    }
                    engine = engine.with_ci(std::sync::Arc::new(
                        SigmaCiOracle::new(
                            graph.graph()?,
                            (0..graph.nodes).map(DenseNodeId::from_raw).collect(),
                        )
                        .map_err(invalid)?,
                    ));
                    "sigma_oracle".into()
                } else {
                    engine = engine.with_ci(antecedent_stats::ci_from_name(ci).map_err(invalid)?);
                    ci.clone()
                };
                let variables: Vec<_> =
                    (0..u32::try_from(n).map_err(invalid)?).map(VariableId::from_raw).collect();
                let mut context = ctx.clone();
                context.rng = antecedent_core::RngFactory::from_seed(*seed);
                let answer = discover_cyclic_fci(
                    &engine,
                    CyclicFciPremises {
                        simple_scm: *simple_scm,
                        sigma_faithful: *sigma_faithful,
                        independent_equilibrium_replicates: *independent_equilibrium_replicates,
                    },
                    &data,
                    &variables,
                    &mut DiscoveryWorkspace::default(),
                    &context,
                )
                .map_err(invalid)?;
                let pag = &answer.discovery.evidence.graph;
                let mut edges = Vec::new();
                for i in 0..n {
                    let a = DenseNodeId::from_raw(u32::try_from(i).map_err(invalid)?);
                    for (b, at_a, at_b) in pag.neighbors(a) {
                        if b.raw() > a.raw() {
                            edges.push((a.raw(), b.raw(), mark(at_a), mark(at_b)));
                        }
                    }
                }
                Ok(CyclicGraphReport::Discover {
                    conclusions: antecedent_discovery::cyclic_fci::cyclic_pag_conclusions(pag, ctx)
                        .map_err(invalid)?
                        .into_iter()
                        .map(|c| (c.source, c.target, c.relation, c.criterion))
                        .collect(),
                    edges,
                    diagnostics: answer
                        .discovery
                        .diagnostics
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect(),
                    interpretation: answer.interpretation.into(),
                    ci_source,
                })
            }
        }
    }
}

/// Admitted variables of an equation-variable (bipartite) graph, selector included.
const BIPARTITE_MAX_VARIABLES: usize = 16_384;
/// Admitted equation-variable incidences of an equation-variable (bipartite) graph.
const BIPARTITE_MAX_INCIDENCES: usize = 4_000_000;

/// Admission by size, not a fixed variable count: the causal-ordering matching allocates per
/// variable and the acyclification is polynomial in the incidence, quadratic only inside one
/// cluster. `variables` counts every node of the graph that will be built.
fn admit_bipartite_size(
    variables: usize,
    equations: &[Vec<usize>],
    what: &str,
) -> Result<(), CyclicArtifactError> {
    let incidences = equations.iter().try_fold(0_usize, |sum, e| sum.checked_add(e.len()));
    if variables > BIPARTITE_MAX_VARIABLES
        || equations.len() > BIPARTITE_MAX_VARIABLES
        || incidences.is_none_or(|n| n > BIPARTITE_MAX_INCIDENCES)
    {
        return Err(invalid(format!("{what} size exceeds the admitted variables or incidences")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bipartite(variables: usize, equations: Vec<Vec<usize>>) -> CyclicGraphRequest {
        CyclicGraphRequest::Bipartite {
            variables,
            exogenous: Vec::new(),
            equations,
            replacement: None,
            x: 0,
            y: 1,
            conditioning: Vec::new(),
        }
    }

    #[test]
    fn bipartite_request_is_admitted_by_size_before_any_allocation() {
        let ctx = ExecutionContext::for_tests(1);
        // `variables` sizes the matching tables; a few bytes of request must not ask for them.
        for variables in [BIPARTITE_MAX_VARIABLES + 1, u32::MAX as usize] {
            let error = bipartite(variables, Vec::new()).execute(&ctx).unwrap_err();
            assert!(error.to_string().contains("bipartite size exceeds"), "{error}");
        }
        let wide = vec![(0..BIPARTITE_MAX_VARIABLES).collect::<Vec<_>>(); 256];
        let error = bipartite(BIPARTITE_MAX_VARIABLES, wide).execute(&ctx).unwrap_err();
        assert!(error.to_string().contains("bipartite size exceeds"), "{error}");
    }

    #[test]
    fn small_bipartite_request_still_executes() {
        let ctx = ExecutionContext::for_tests(1);
        let report = bipartite(2, vec![vec![0], vec![0, 1]]).execute(&ctx).unwrap();
        assert!(matches!(report, CyclicGraphReport::Bipartite { .. }));
    }
}
