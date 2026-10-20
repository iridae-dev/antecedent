// SPDX-License-Identifier: MIT OR Apache-2.0
//! Forre–Mooij (2020) Theorem 7.2: all three cyclic do-calculus rules.
//! Fixed input nodes are explicitly conditioned; intervention indicators are
//! added after baseline graph surgery. A failed sufficient test is never a
//! non-identifiability proof, and exhausted checks are a separate outcome.
use crate::cyclic_id::{CyclicIdError, CyclicIdPremises, cyclic_backdoor_valid};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_expr::{
    CausalExprArena, DomainRef, FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_graph::{CyclicMixedGraph, DenseNodeId, SigmaSeparationWorkspace, sigma_separates};
use std::collections::BTreeSet;
/// The exact theorem-scoped rewrite requested by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CyclicCalculusRule {
    /// P(Y|X,Z,do(W)) = P(Y|Z,do(W)).
    Observation,
    /// P(Y|Z,do(X),do(W)) = P(Y|X,Z,do(W)).
    Exchange,
    /// P(Y|Z,do(X),do(W)) = P(Y|Z,do(W)).
    Action,
}
/// Outcome of the bounded sufficient-criterion check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CyclicCalculusStatus {
    /// Every required sigma-separation test verified the rewrite.
    Verified,
    /// A sigma-connection prevents using this rule's criterion.
    NotLicensedByCriterion,
    /// The pairwise-check budget ended before verification.
    BudgetExhausted,
}
/// Retained graph-checked rewrite with distinct source/executable IR roots.
pub struct CyclicCalculusApplication {
    /// No exhausted search is reported as a mathematical negative result.
    pub status: CyclicCalculusStatus,
    /// Present only after the rule's full sufficient criterion passes.
    pub program: Option<FunctionalProgram>,
    /// Actual pairwise sigma checks performed.
    pub sigma_checks: usize,
    /// Original theorem scope, fixed inputs and graph transformation.
    pub derivation: Vec<String>,
}
fn invalid(s: impl std::fmt::Display) -> CyclicIdError {
    CyclicIdError::Invalid(s.to_string())
}
fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).expect("bounded graph"))
}
fn var(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).expect("bounded graph"))
}
/// Check a cyclic do-calculus rule on disjoint observed X,Y,Z,W, with the
/// compatible ioSCM and standard-measurable-space assumptions explicitly declared.
/// Resulting conditional-distribution equalities apply on their common support.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the rule's four disjoint variable roles and bounded scientific premises are explicit"
)]
pub fn apply_cyclic_calculus(
    graph: &CyclicMixedGraph,
    rule: CyclicCalculusRule,
    x: &[usize],
    y: &[usize],
    z: &[usize],
    w: &[usize],
    io_scm: bool,
    standard_measurable_spaces: bool,
    max_checks: usize,
    ctx: &ExecutionContext,
) -> Result<CyclicCalculusApplication, CyclicIdError> {
    let n = graph.node_count();
    if n == 0
        || n > 32
        || y.is_empty()
        || !io_scm
        || !standard_measurable_spaces
        || max_checks > 100_000
    {
        return Err(invalid("cyclic calculus graph/query/premises"));
    }
    let mut seen = BTreeSet::new();
    for group in [x, y, z, w] {
        for &v in group {
            if v >= n || !seen.insert(v) {
                return Err(invalid("cyclic calculus requires valid disjoint X,Y,Z,W"));
            }
        }
    }
    let indicators = if rule == CyclicCalculusRule::Observation { 0 } else { x.len() };
    let mut extended =
        CyclicMixedGraph::with_variables(u32::try_from(n + indicators).map_err(invalid)?);
    for (a, b) in graph.directed_edge_pairs_sorted() {
        if !w.contains(&(b as usize)) {
            extended
                .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(invalid)?;
        }
    }
    for (a, b) in graph.bidirected_edge_pairs_sorted() {
        if !w.contains(&(a as usize)) && !w.contains(&(b as usize)) {
            extended
                .insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(invalid)?;
        }
    }
    let sources = if rule == CyclicCalculusRule::Observation {
        x.to_vec()
    } else {
        for (k, &v) in x.iter().enumerate() {
            extended.insert_directed(id(n + k), id(v)).map_err(invalid)?;
        }
        (0..x.len()).map(|k| n + k).collect()
    };
    let mut conditioned = z.to_vec();
    conditioned.extend_from_slice(w);
    if rule == CyclicCalculusRule::Exchange {
        conditioned.extend_from_slice(x);
    }
    conditioned.sort_unstable();
    let _lease = ctx
        .acquire_admission(1, ((n + indicators).pow(2) * 256 + 65536) as u64)
        .map_err(|e| CyclicIdError::Resource(e.to_string()))?;
    let mut workspace = SigmaSeparationWorkspace::default();
    let mut checks = 0;
    let derivation=vec![format!("Forre-Mooij 2020 Theorem 7.2, rule {rule:?}; compatible ioSCM and standard measurable spaces declared"),format!("Baseline do(W={w:?}) cuts incoming directed and incident bidirected edges; fixed input coordinates conditioned explicitly"),format!("Check Y={y:?} sigma-separated from {sources:?} given {conditioned:?}; intervention indicators are added after surgery"),"Conditional-distribution equality is on common support; no non-identifiability/completeness or positivity certificate is inferred".into()];
    for &a in y {
        for &b in &sources {
            if ctx.cancellation.is_cancelled() {
                return Err(CyclicIdError::Resource("cancelled".into()));
            }
            if checks >= max_checks {
                return Ok(CyclicCalculusApplication {
                    status: CyclicCalculusStatus::BudgetExhausted,
                    program: None,
                    sigma_checks: checks,
                    derivation,
                });
            }
            checks += 1;
            if !sigma_separates(
                &extended,
                id(a),
                id(b),
                &conditioned.iter().copied().map(id).collect::<Vec<_>>(),
                &mut workspace,
            )
            .map_err(invalid)?
            .status
            .is_separated()
            {
                return Ok(CyclicCalculusApplication {
                    status: CyclicCalculusStatus::NotLicensedByCriterion,
                    program: None,
                    sigma_checks: checks,
                    derivation,
                });
            }
        }
    }
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set(y.iter().copied().map(var));
    let mut source_condition = z.to_vec();
    if rule == CyclicCalculusRule::Observation {
        source_condition.extend_from_slice(x);
    }
    let source_condition = arena.intern_var_set(source_condition.into_iter().map(var));
    let mut source_actions = w.to_vec();
    if rule != CyclicCalculusRule::Observation {
        source_actions.extend_from_slice(x);
    }
    let source_actions = arena.intern_intervention_set(source_actions.into_iter().map(var));
    let source = arena.intern_distribution(
        outcomes,
        source_condition,
        source_actions,
        if w.is_empty() && rule == CyclicCalculusRule::Observation {
            DomainRef::Observational
        } else {
            DomainRef::Interventional
        },
    );
    let mut target_condition = z.to_vec();
    if rule == CyclicCalculusRule::Exchange {
        target_condition.extend_from_slice(x);
    }
    let target_condition = arena.intern_var_set(target_condition.into_iter().map(var));
    let actions = arena.intern_intervention_set(w.iter().copied().map(var));
    let executable = arena.intern_distribution(
        outcomes,
        target_condition,
        actions,
        if w.is_empty() { DomainRef::Observational } else { DomainRef::Interventional },
    );
    let schema = ProgramSchema::new(
        (0..n).map(|i| (var(i), ProgramVariable { name: format!("v{i}").into() })),
    );
    let program =
        FunctionalProgram::new(arena, schema, source, executable, ProgramLimits::default())
            .map_err(invalid)?;
    Ok(CyclicCalculusApplication {
        status: CyclicCalculusStatus::Verified,
        program: Some(program),
        sigma_checks: checks,
        derivation,
    })
}
/// Corollary 8.4(4) adjustment as an executable existing-IR functional, rather
/// than only a boolean criterion. No density/completeness hypothesis is added.
pub fn identify_cyclic_adjustment(
    graph: &CyclicMixedGraph,
    outcomes: &[usize],
    treatments: &[usize],
    adjustment: &[usize],
    continuous: bool,
    premises: CyclicIdPremises,
) -> Result<Option<FunctionalProgram>, CyclicIdError> {
    if !cyclic_backdoor_valid(graph, outcomes, treatments, adjustment, premises)? {
        return Ok(None);
    }
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set(outcomes.iter().copied().map(var));
    let x = arena.intern_intervention_set(treatments.iter().copied().map(var));
    let empty = arena.empty_var_set();
    let none = arena.empty_intervention_set();
    let source = arena.intern_distribution(y, empty, x, DomainRef::Interventional);
    let z = arena.intern_var_set(adjustment.iter().copied().map(var));
    let condition = arena.intern_var_set(treatments.iter().chain(adjustment).copied().map(var));
    let conditional = arena.intern_distribution(y, condition, none, DomainRef::Observational);
    let marginal = arena.intern_distribution(z, empty, none, DomainRef::Observational);
    let factors = arena.intern_list([conditional, marginal]);
    let body = arena.intern(antecedent_expr::ExprNode::Product(factors));
    let executable = if adjustment.is_empty() {
        conditional
    } else {
        arena.intern(if continuous {
            antecedent_expr::ExprNode::IntegralOut { variables: z, expr: body }
        } else {
            antecedent_expr::ExprNode::SumOut { variables: z, expr: body }
        })
    };
    let schema = ProgramSchema::new(
        (0..graph.node_count()).map(|i| (var(i), ProgramVariable { name: format!("v{i}").into() })),
    );
    Ok(Some(
        FunctionalProgram::new(arena, schema, source, executable, ProgramLimits::default())
            .map_err(invalid)?,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calculus_keeps_feedback_indicators_and_budget_distinct_from_negative_proofs() {
        let mut graph = CyclicMixedGraph::with_variables(3);
        graph.insert_directed(id(0), id(2)).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let rule = |g: &CyclicMixedGraph, kind, budget| {
            apply_cyclic_calculus(g, kind, &[0], &[2], &[], &[], true, true, budget, &ctx).unwrap()
        };
        assert_eq!(
            rule(&graph, CyclicCalculusRule::Exchange, 10).status,
            CyclicCalculusStatus::Verified
        );
        assert_eq!(
            rule(&graph, CyclicCalculusRule::Action, 10).status,
            CyclicCalculusStatus::NotLicensedByCriterion
        );
        assert_eq!(
            rule(&graph, CyclicCalculusRule::Exchange, 0).status,
            CyclicCalculusStatus::BudgetExhausted
        );
        graph.insert_directed(id(2), id(0)).unwrap();
        assert_eq!(
            rule(&graph, CyclicCalculusRule::Exchange, 10).status,
            CyclicCalculusStatus::NotLicensedByCriterion
        );
        let disconnected = CyclicMixedGraph::with_variables(3);
        assert_eq!(
            rule(&disconnected, CyclicCalculusRule::Action, 10).status,
            CyclicCalculusStatus::Verified
        );
        let program = identify_cyclic_adjustment(
            &disconnected,
            &[2],
            &[0],
            &[],
            false,
            CyclicIdPremises { io_scm: true, scc_product_density: false },
        )
        .unwrap()
        .unwrap();
        assert_ne!(program.mapping().source, program.mapping().executable);
    }
}

#[cfg(test)]
mod gaussian_reference_tests {
    use super::*;
    // Independent dense elimination and Gaussian conditioning; no sigma or IR
    // implementation is used to construct these probabilistic reference laws.
    fn inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut work = vec![vec![0.; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                work[i][j] = a[i][j];
            }
            work[i][n + i] = 1.;
        }
        for k in 0..n {
            let pivot = work[k][k];
            assert!(pivot.abs() > 1e-8);
            for j in 0..2 * n {
                work[k][j] /= pivot;
            }
            for i in 0..n {
                if i != k {
                    let factor = work[i][k];
                    for j in 0..2 * n {
                        work[i][j] -= factor * work[k][j];
                    }
                }
            }
        }
        work.iter().map(|r| r[n..].to_vec()).collect()
    }
    fn conditional(
        w: &[Vec<f64>],
        q: &[Vec<f64>],
        clamps: &[(usize, f64)],
        observed: &[(usize, f64)],
        y: usize,
    ) -> (f64, f64) {
        let free: Vec<_> = (0..3).filter(|i| !clamps.iter().any(|(j, _)| j == i)).collect();
        let a: Vec<Vec<_>> = free
            .iter()
            .map(|&i| free.iter().map(|&j| f64::from(i == j) - w[i][j]).collect())
            .collect();
        let inv = inverse(&a);
        let forcing: Vec<_> =
            free.iter().map(|&i| clamps.iter().map(|&(j, v)| w[i][j] * v).sum::<f64>()).collect();
        let mean: Vec<_> =
            inv.iter().map(|r| r.iter().zip(&forcing).map(|(a, b)| a * b).sum::<f64>()).collect();
        let mut covariance = vec![vec![0.0; free.len()]; free.len()];
        for i in 0..free.len() {
            for j in 0..free.len() {
                for k in 0..free.len() {
                    for l in 0..free.len() {
                        covariance[i][j] += inv[i][k] * q[free[k]][free[l]] * inv[j][l];
                    }
                }
            }
        }
        let yi = free.iter().position(|&i| i == y).unwrap();
        if observed.is_empty() {
            return (mean[yi], covariance[yi][yi]);
        }
        let oi: Vec<_> =
            observed.iter().map(|(i, _)| free.iter().position(|j| j == i).unwrap()).collect();
        let conditional_cov: Vec<Vec<_>> =
            oi.iter().map(|&i| oi.iter().map(|&j| covariance[i][j]).collect()).collect();
        let precision = inverse(&conditional_cov);
        let weights: Vec<_> = (0..oi.len())
            .map(|i| (0..oi.len()).map(|j| covariance[yi][oi[j]] * precision[j][i]).sum::<f64>())
            .collect();
        let m = mean[yi]
            + weights
                .iter()
                .enumerate()
                .map(|(i, b)| b * (observed[i].1 - mean[oi[i]]))
                .sum::<f64>();
        let v = covariance[yi][yi]
            - weights.iter().enumerate().map(|(i, b)| b * covariance[yi][oi[i]]).sum::<f64>();
        (m, v)
    }
    #[test]
    fn every_three_node_mixed_graph_verified_rule_matches_independent_gaussian_truth() {
        let ctx = ExecutionContext::for_tests(17);
        let mut verified = [0; 3];
        for mask in 0u16..512 {
            let mut graph = CyclicMixedGraph::with_variables(3);
            let mut weights = vec![vec![0.; 3]; 3];
            let mut q = vec![vec![0.; 3]; 3];
            for (i, row) in q.iter_mut().enumerate() {
                row[i] = 1.;
            }
            let mut bit = 0;
            for a in 0..3 {
                for (b, row) in weights.iter_mut().enumerate() {
                    if a != b {
                        if mask & (1 << bit) != 0 {
                            graph.insert_directed(id(a), id(b)).unwrap();
                            row[a] = 0.11 + 0.02 * (a + b) as f64;
                        }
                        bit += 1;
                    }
                }
            }
            for a in 0..3 {
                for b in a + 1..3 {
                    if mask & (1 << bit) != 0 {
                        graph.insert_bidirected(id(a), id(b)).unwrap();
                        q[a][a] += 0.1;
                        q[b][b] += 0.1;
                        q[a][b] += 0.1;
                        q[b][a] += 0.1;
                    }
                    bit += 1;
                }
            }
            for x in 0..3 {
                for y in 0..3 {
                    if x == y {
                        continue;
                    }
                    let remaining = 3 - x - y;
                    for mode in 0..3 {
                        let z = if mode == 1 { vec![remaining] } else { vec![] };
                        let fixed = if mode == 2 { vec![remaining] } else { vec![] };
                        let observed: Vec<_> = z.iter().map(|&i| (i, 0.3)).collect();
                        let clamps: Vec<_> = fixed.iter().map(|&i| (i, -0.2)).collect();
                        for (r, rule) in [
                            CyclicCalculusRule::Observation,
                            CyclicCalculusRule::Exchange,
                            CyclicCalculusRule::Action,
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            if apply_cyclic_calculus(
                                &graph,
                                rule,
                                &[x],
                                &[y],
                                &z,
                                &fixed,
                                true,
                                true,
                                100,
                                &ctx,
                            )
                            .unwrap()
                            .status
                                != CyclicCalculusStatus::Verified
                            {
                                continue;
                            }
                            verified[r] += 1;
                            let mut with_x = observed.clone();
                            with_x.push((x, 0.7));
                            let mut do_x = clamps.clone();
                            do_x.push((x, 0.7));
                            let (left, right) = match rule {
                                CyclicCalculusRule::Observation => (
                                    conditional(&weights, &q, &clamps, &with_x, y),
                                    conditional(&weights, &q, &clamps, &observed, y),
                                ),
                                CyclicCalculusRule::Exchange => (
                                    conditional(&weights, &q, &do_x, &observed, y),
                                    conditional(&weights, &q, &clamps, &with_x, y),
                                ),
                                CyclicCalculusRule::Action => (
                                    conditional(&weights, &q, &do_x, &observed, y),
                                    conditional(&weights, &q, &clamps, &observed, y),
                                ),
                            };
                            assert!(
                                (left.0 - right.0).abs() < 1e-11
                                    && (left.1 - right.1).abs() < 1e-11,
                                "mask {mask}, {rule:?}, x{x} y{y} mode{mode}: {left:?} {right:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(verified.iter().all(|&n| n > 100));
    }
}
