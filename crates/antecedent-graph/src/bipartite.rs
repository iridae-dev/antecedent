// SPDX-License-Identifier: MIT OR Apache-2.0
//! Equation-variable causal ordering and B-separation (Mooij 2026,
//! arXiv:2608.19831v1, Definitions 3/5/13/33 and Lemma 42).
//! Matching/SCC structure does not prove numerical solvability. The Markov
//! interpretation requires independent exogenous inputs and measurable unique
//! cluster solutions. B-separation is computed as D-separation in the paper's
//! acyclification, accounting for functional determination of endogenous nodes.

use crate::{CyclicDigraph, DSeparationWorkspace, Dag, DenseNodeId, GraphError};

/// Bounded incidence graph with explicitly distinguished exogenous variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BipartiteCausalGraph {
    variables: usize,
    exogenous: Vec<bool>,
    incidence: Vec<Vec<usize>>,
    cluster_of: Vec<usize>,
    equation_cluster: Vec<usize>,
    acyclification: Dag,
}

/// Graphical B-separation, distinct from a model-solvability certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BSeparation {
    /// True iff D-separated in the equation graph's acyclification.
    pub separated: bool,
    /// Variable coordinates determined by the conditioning variables.
    pub determined: Vec<usize>,
    /// Required interpretation, not inferred from incidence alone.
    pub assumptions: &'static str,
}
fn invalid(message: &'static str) -> GraphError {
    GraphError::InvalidEndpoints { message }
}
fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).expect("bounded bipartite order"))
}

fn augment(
    equation: usize,
    incidence: &[Vec<usize>],
    exogenous: &[bool],
    seen: &mut [bool],
    matched: &mut [Option<usize>],
) -> bool {
    for &variable in &incidence[equation] {
        if exogenous[variable] || seen[variable] {
            continue;
        }
        seen[variable] = true;
        if matched[variable].is_none_or(|other| augment(other, incidence, exogenous, seen, matched))
        {
            matched[variable] = Some(equation);
            return true;
        }
    }
    false
}

impl BipartiteCausalGraph {
    /// Construct a graph with a perfect endogenous-variable/equation matching.
    /// This implementation uses bounded augmenting paths and the existing SCC
    /// engine; the resulting clusters do not depend on the chosen matching.
    pub fn new(
        variables: usize,
        exogenous: &[usize],
        equations: Vec<Vec<usize>>,
    ) -> Result<Self, GraphError> {
        if variables == 0 || variables > 128 {
            return Err(GraphError::NodeBoundExceeded { nodes: variables, bound: 128 });
        }
        let mut external = vec![false; variables];
        for &v in exogenous {
            if v >= variables || external[v] {
                return Err(invalid("exogenous indices must be unique and in range"));
            }
            external[v] = true;
        }
        if equations.len() != variables - exogenous.len() {
            return Err(invalid("square endogenous equation system required"));
        }
        let mut incidence = equations;
        for neighbors in &mut incidence {
            neighbors.sort_unstable();
            if neighbors.iter().any(|&v| v >= variables)
                || neighbors.windows(2).any(|w| w[0] == w[1])
            {
                return Err(invalid("invalid or duplicate equation incidence"));
            }
        }
        let mut matched = vec![None; variables];
        for equation in 0..incidence.len() {
            if !augment(equation, &incidence, &external, &mut vec![false; variables], &mut matched)
            {
                return Err(invalid("no perfect endogenous equation-variable matching"));
            }
        }
        let mut solved_variable = vec![0; incidence.len()];
        for (variable, equation) in matched.iter().enumerate() {
            if let Some(e) = equation {
                solved_variable[*e] = variable;
            }
        }
        let mut directed = CyclicDigraph::with_variables(
            u32::try_from(variables).map_err(|_| GraphError::TooManyNodes)?,
        );
        for (equation, neighbors) in incidence.iter().enumerate() {
            for &v in neighbors {
                if v != solved_variable[equation] {
                    directed.insert_directed(id(v), id(solved_variable[equation]))?;
                }
            }
        }
        let condensation = directed.condensation();
        let cluster_of: Vec<_> = (0..variables)
            .map(|v| condensation.component_index(id(v)).expect("known node"))
            .collect();
        let equation_cluster: Vec<_> = solved_variable.iter().map(|&v| cluster_of[v]).collect();
        let mut edges = std::collections::BTreeSet::new();
        for (e, neighbors) in incidence.iter().enumerate() {
            for &input in neighbors {
                if cluster_of[input] != equation_cluster[e] {
                    for (v, &cluster) in cluster_of.iter().enumerate() {
                        if cluster == equation_cluster[e] {
                            edges.insert((input, v));
                        }
                    }
                }
            }
        }
        let mut acyclification =
            Dag::with_variables(u32::try_from(variables).map_err(|_| GraphError::TooManyNodes)?);
        for (a, b) in edges {
            acyclification.insert_directed(id(a), id(b))?;
        }
        Ok(Self {
            variables,
            exogenous: external,
            incidence,
            cluster_of,
            equation_cluster,
            acyclification,
        })
    }

    /// Dense cluster index for every variable; equation clusters are separate.
    #[must_use]
    pub fn variable_clusters(&self) -> &[usize] {
        &self.cluster_of
    }
    /// Dense cluster index for each named-equation position.
    #[must_use]
    pub fn equation_clusters(&self) -> &[usize] {
        &self.equation_cluster
    }
    /// Equation-aware acyclification, not an original mechanism graph.
    #[must_use]
    pub fn acyclification(&self) -> &Dag {
        &self.acyclification
    }

    /// Replace the incidence of equation `equation` by a hard assignment to
    /// `variable`. This may change matching, clusters, and causal ordering.
    pub fn replace_equation(&self, equation: usize, variable: usize) -> Result<Self, GraphError> {
        if equation >= self.incidence.len() || variable >= self.variables {
            return Err(invalid("equation intervention indices"));
        }
        let mut incidence = self.incidence.clone();
        incidence[equation] = vec![variable];
        let external: Vec<_> =
            self.exogenous.iter().enumerate().filter_map(|(i, &v)| v.then_some(i)).collect();
        Self::new(self.variables, &external, incidence)
    }

    /// B-separation via the paper's D-separation equivalence. An endogenous
    /// variable is deterministic given all external parents of its whole cluster.
    pub fn b_separates(
        &self,
        x: usize,
        y: usize,
        conditioning: &[usize],
    ) -> Result<BSeparation, GraphError> {
        if x >= self.variables
            || y >= self.variables
            || conditioning.iter().any(|&v| v >= self.variables)
        {
            return Err(invalid("B-separation endpoints/conditioning"));
        }
        let mut determined = vec![false; self.variables];
        for &v in conditioning {
            determined[v] = true;
        }
        loop {
            let mut changed = false;
            for v in 0..self.variables {
                if !self.exogenous[v]
                    && !determined[v]
                    && self.acyclification.parents(id(v)).iter().all(|p| determined[p.as_usize()])
                {
                    determined[v] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let closure: Vec<_> = (0..self.variables).filter(|&v| determined[v]).collect();
        let separated = if determined[x] || determined[y] {
            true
        } else if x == y {
            false
        } else {
            self.acyclification.is_d_separated(
                id(x),
                id(y),
                &closure.iter().copied().map(id).collect::<Vec<_>>(),
                &mut DSeparationWorkspace::default(),
            )?
        };
        Ok(BSeparation {
            separated,
            determined: closure,
            assumptions: "graphical B-separation; conditional-independence interpretation requires independent exogenous variables and measurable unique solutions for every endogenous cluster; matching alone establishes neither",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bipartite_bathtub_order_and_deterministic_closure() {
        // O,P,D,I,K,g; f1(O,I), f2(O,P,K), f3(P,D,g).
        let graph = BipartiteCausalGraph::new(
            6,
            &[3, 4, 5],
            vec![vec![0, 3], vec![0, 1, 4], vec![1, 2, 5]],
        )
        .unwrap();
        assert!(graph.acyclification().parents(id(0)).contains(&id(3)));
        assert!(graph.acyclification().parents(id(1)).contains(&id(0)));
        let independent = graph.b_separates(3, 4, &[]).unwrap();
        assert!(independent.separated);
        let determined = graph.b_separates(2, 4, &[3, 4, 5]).unwrap();
        assert!(determined.separated);
        assert!(determined.determined.contains(&2));
        assert!(!graph.b_separates(0, 1, &[]).unwrap().separated);
    }
    #[test]
    fn bipartite_feedback_clusters_and_equation_permutations() {
        // Endogenous supply/demand variables 0,1; independent exogenous 2,3.
        let a = BipartiteCausalGraph::new(4, &[2, 3], vec![vec![0, 1, 2], vec![0, 1, 3]]).unwrap();
        let b = BipartiteCausalGraph::new(4, &[2, 3], vec![vec![0, 1, 3], vec![0, 1, 2]]).unwrap();
        assert_eq!(a.variable_clusters(), b.variable_clusters());
        assert_eq!(a.variable_clusters()[0], a.variable_clusters()[1]);
        assert!(a.b_separates(0, 1, &[2, 3]).unwrap().separated);
        assert!(!a.b_separates(0, 1, &[2]).unwrap().separated);
        assert!(a.b_separates(2, 3, &[]).unwrap().separated);
        assert_ne!(
            a.replace_equation(0, 0).unwrap().acyclification(),
            a.replace_equation(1, 0).unwrap().acyclification()
        );
    }
}

#[cfg(test)]
mod independent_linear_truth_tests {
    use super::*;
    #[test]
    fn every_small_bipartite_separation_is_sound_for_independent_linear_equation_truths() {
        // Enumerate all incidences of two equations over two endogenous and
        // three independent unit-Gaussian inputs. Numerical truth is an
        // independent closed-form 2x2 solve, not the acyclification engine.
        let mut checked = 0;
        for mask in 0..(1usize << 10) {
            let equations: Vec<Vec<usize>> = (0..2)
                .map(|e| (0..5).filter(|&v| mask & (1 << (e * 5 + v)) != 0).collect())
                .collect();
            let Ok(graph) = BipartiteCausalGraph::new(5, &[2, 3, 4], equations.clone()) else {
                continue;
            };
            let matching = if equations[0].contains(&0) && equations[1].contains(&1) {
                [0, 1]
            } else {
                [1, 0]
            };
            let mut a = [[0.0; 2]; 2];
            let mut b = [[0.0; 3]; 2];
            for e in 0..2 {
                let row = matching[e];
                a[row][row] = 1.0;
                for &variable in &equations[e] {
                    if variable < 2 && variable != row {
                        a[row][variable] = 0.1;
                    }
                    if variable >= 2 {
                        b[row][variable - 2] = 0.15 * (e + 1) as f64 * (variable - 1) as f64;
                    }
                }
            }
            let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            let mut response = [[0.0; 3]; 5];
            for u in 0..3 {
                response[0][u] = (-a[1][1] * b[0][u] + a[0][1] * b[1][u]) / determinant;
                response[1][u] = (a[1][0] * b[0][u] - a[0][0] * b[1][u]) / determinant;
                response[2 + u][u] = 1.0;
            }
            let covariance =
                |x: usize, y: usize| (0..3).map(|u| response[x][u] * response[y][u]).sum::<f64>();
            for x in 0..5 {
                for y in x + 1..5 {
                    let conditions: Vec<Vec<usize>> = std::iter::once(vec![])
                        .chain((0..5).filter(|&z| z != x && z != y).map(|z| vec![z]))
                        .collect();
                    for condition in conditions {
                        let separation = graph.b_separates(x, y, &condition).unwrap();
                        if !separation.separated {
                            continue;
                        }
                        let mut truth = covariance(x, y);
                        if let Some(&z) = condition.first() {
                            let variance = covariance(z, z);
                            if variance > 1e-12 {
                                truth -= covariance(x, z) * covariance(y, z) / variance;
                            }
                        }
                        assert!(
                            truth.abs() < 1e-10,
                            "incidence={equations:?}, {x} vs {y} given {condition:?}: conditional covariance {truth}"
                        );
                        checked += 1;
                    }
                }
            }
            let reversed =
                BipartiteCausalGraph::new(5, &[2, 3, 4], equations.into_iter().rev().collect())
                    .unwrap();
            assert_eq!(graph.variable_clusters(), reversed.variable_clusters());
        }
        assert!(checked > 1000);
    }
}
