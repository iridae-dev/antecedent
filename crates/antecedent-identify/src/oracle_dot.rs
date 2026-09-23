//! Parse the frozen `graph_dot` strings recorded by the pinned external
//! identify() baseline (see `parity/baselines/`).
//!
//! `Name[latent]` nodes are projected out by [`antecedent_graph::latent_project`], so a
//! latent with observed parents (`X -> L -> Y`) yields the directed edge `X -> Y` and a chain
//! of latents (`U1 -> U2 -> {A, B}`) yields `A <-> B`. This crate cannot depend on
//! `antecedent-io` (cycle), so the dialect is parsed here rather than via `admg_from_dot`.
//!
//! Compiled for tests and under the `test-util` feature only: every error is a panic, which
//! suits frozen fixtures and not a library API.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_core::VariableId;
use antecedent_graph::{Admg, Dag, DenseNodeId, latent_project};

/// Observed-node intern table for a parsed oracle DOT string.
#[derive(Debug)]
pub struct OracleGraph {
    names: Vec<String>,
}

impl OracleGraph {
    /// Observed names in intern order (first-seen among non-latent nodes).
    #[must_use]
    pub fn observed(&self) -> &[String] {
        &self.names
    }

    /// Look up an observed node by name.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not an observed node in the parsed DOT.
    #[must_use]
    pub fn id(&self, name: &str) -> VariableId {
        let idx = self.names.iter().position(|n| n == name).unwrap_or_else(|| {
            panic!("oracle DOT has no observed node `{name}`; names={:?}", self.names)
        });
        VariableId::from_raw(u32::try_from(idx).expect("node index fits u32"))
    }

    #[cfg(test)]
    fn dense(&self, name: &str) -> DenseNodeId {
        DenseNodeId::from_raw(self.id(name).raw())
    }
}

struct Parsed {
    observed: Vec<String>,
    directed: Vec<(String, String)>,
    latents: HashSet<String>,
}

fn parse(dot: &str) -> Parsed {
    let start = dot.find('{').expect("oracle DOT must contain '{'");
    let end = dot.rfind('}').expect("oracle DOT must contain '}'");
    let body = &dot[start + 1..end];

    let mut latents = HashSet::new();
    for stmt in body.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        if let Some((name, attrs)) = node_attr(stmt) {
            if attrs.split([',', ' ']).any(|tok| tok.trim() == "latent") {
                latents.insert(name);
            }
        }
    }

    let mut observed = Vec::new();
    let mut directed = Vec::new();
    let mut interned = HashSet::new();
    let intern = |name: &str, observed: &mut Vec<String>, interned: &mut HashSet<String>| {
        if latents.contains(name) || !interned.insert(name.to_string()) {
            return;
        }
        observed.push(name.to_string());
    };

    for stmt in body.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        if let Some((from, to)) = stmt.split_once("->") {
            let from = from.trim().to_string();
            let to = to.trim().to_string();
            intern(&from, &mut observed, &mut interned);
            intern(&to, &mut observed, &mut interned);
            directed.push((from, to));
            continue;
        }
        if let Some((name, attrs)) = node_attr(stmt) {
            if !attrs.split([',', ' ']).any(|tok| tok.trim() == "latent") {
                intern(&name, &mut observed, &mut interned);
            }
        }
    }

    Parsed { observed, directed, latents }
}

fn node_attr(stmt: &str) -> Option<(String, String)> {
    let start = stmt.find('[')?;
    let end = stmt.rfind(']')?;
    if end <= start {
        return None;
    }
    let name = stmt[..start].trim();
    if name.is_empty() || name.contains("->") {
        return None;
    }
    Some((name.to_string(), stmt[start + 1..end].to_string()))
}

fn dense(names: &[String], name: &str) -> DenseNodeId {
    let idx = names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("name `{name}` is not an observed oracle node; names={names:?}"));
    DenseNodeId::from_raw(u32::try_from(idx).expect("node index fits u32"))
}

/// Parse a latent-free oracle DOT into a DAG.
///
/// # Panics
///
/// Panics if the DOT is malformed or contains latent nodes.
#[must_use]
pub fn dag_from_oracle_dot(dot: &str) -> (Dag, OracleGraph) {
    let parsed = parse(dot);
    assert!(
        parsed.latents.is_empty(),
        "DAG oracle DOT must not contain latent nodes: {:?}",
        parsed.latents
    );
    let n = u32::try_from(parsed.observed.len()).expect("node count fits u32");
    let mut g = Dag::with_variables(n);
    for (from, to) in &parsed.directed {
        g.insert_directed(dense(&parsed.observed, from), dense(&parsed.observed, to)).unwrap();
    }
    (g, OracleGraph { names: parsed.observed })
}

/// Parse an oracle DOT into an ADMG, projecting `Name[latent]` nodes.
///
/// Observed nodes keep their intern order as dense and variable ids; latents are appended
/// after them for the projection only.
///
/// # Panics
///
/// Panics if the DOT is malformed.
#[must_use]
pub fn admg_from_oracle_dot(dot: &str) -> (Admg, OracleGraph) {
    let parsed = parse(dot);
    let mut latents: Vec<&String> = parsed.latents.iter().collect();
    latents.sort_unstable();
    let mut all = parsed.observed.clone();
    all.extend(latents.into_iter().cloned());
    let n = u32::try_from(all.len()).expect("node count fits u32");
    let mut full = Dag::with_variables(n);
    for (from, to) in &parsed.directed {
        full.insert_directed(dense(&all, from), dense(&all, to)).unwrap();
    }
    let observed: Vec<DenseNodeId> = (0..parsed.observed.len())
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node index fits u32")))
        .collect();
    let g = latent_project(&full, &observed).expect("oracle DOT latent projection");
    (g, OracleGraph { names: parsed.observed })
}

#[cfg(test)]
mod tests {
    use super::{admg_from_oracle_dot, dag_from_oracle_dot};

    #[test]
    fn backdoor_dot_is_z_confounded_dag() {
        let (g, names) = dag_from_oracle_dot("digraph { z -> t; z -> y; t -> y; }");
        assert_eq!(names.observed(), ["z", "t", "y"]);
        assert_eq!(g.node_count(), 3);
        assert_eq!(names.id("z").raw(), 0);
        assert_eq!(names.id("t").raw(), 1);
        assert_eq!(names.id("y").raw(), 2);
        let z = names.dense("z");
        let t = names.dense("t");
        let y = names.dense("y");
        assert!(g.children(z).contains(&t));
        assert!(g.children(z).contains(&y));
        assert!(g.children(t).contains(&y));
    }

    #[test]
    fn hedge_dot_projects_latent_to_bow_arc() {
        let (g, names) = admg_from_oracle_dot("digraph { t -> y; U[latent]; U -> t; U -> y; }");
        assert_eq!(names.observed(), ["t", "y"]);
        assert_eq!(g.node_count(), 2);
        let t = names.dense("t");
        let y = names.dense("y");
        assert!(g.children(t).contains(&y));
        assert!(g.bidirected_neighbors(t).contains(&y));
        assert!(g.bidirected_neighbors(y).contains(&t));
    }

    #[test]
    fn latent_with_observed_parent_carries_the_directed_path() {
        // X -> L -> Y with L latent: the projection has the directed edge X -> Y.
        let (g, names) = admg_from_oracle_dot("digraph { x -> L; L -> y; L[latent]; }");
        assert_eq!(names.observed(), ["x", "y"]);
        let x = names.dense("x");
        let y = names.dense("y");
        assert!(g.children(x).contains(&y));
        assert!(g.bidirected_neighbors(x).is_empty());
    }

    #[test]
    fn latent_chain_confounds_its_observed_descendants() {
        // U1 -> U2 -> {a, b}, both latent: a and b share a latent ancestor, so a <-> b,
        // and the unrelated observed c stays unconfounded.
        let (g, names) = admg_from_oracle_dot(
            "digraph { U1[latent]; U2[latent]; U1 -> U2; U2 -> a; U2 -> b; c -> a; }",
        );
        assert_eq!(names.observed(), ["a", "b", "c"]);
        let a = names.dense("a");
        let b = names.dense("b");
        let c = names.dense("c");
        assert!(g.bidirected_neighbors(a).contains(&b));
        assert!(g.bidirected_neighbors(b).contains(&a));
        assert!(g.bidirected_neighbors(c).is_empty());
        assert!(g.children(c).contains(&a));
    }

    #[test]
    fn frontdoor_dot_projects_t_y_bidirected() {
        let (g, names) =
            admg_from_oracle_dot("digraph { t -> m; m -> y; U[latent]; U -> t; U -> y; }");
        assert_eq!(names.observed(), ["t", "m", "y"]);
        assert_eq!(g.node_count(), 3);
        let t = names.dense("t");
        let m = names.dense("m");
        let y = names.dense("y");
        assert!(g.children(t).contains(&m));
        assert!(g.children(m).contains(&y));
        assert!(!g.children(t).contains(&y));
        assert!(g.bidirected_neighbors(t).contains(&y));
        assert!(!g.bidirected_neighbors(t).contains(&m));
        assert!(!g.bidirected_neighbors(m).contains(&y));
    }
}
