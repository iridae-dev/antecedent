//! Parse the frozen `graph_dot` strings recorded by the pinned external
//! identify() baseline (see `parity/baselines/`).
//!
//! `Name[latent]` nodes are projected out: directed edges among observed nodes
//! are kept, and each latent's observed children are joined by bidirected
//! edges. This crate cannot depend on `antecedent-io` (cycle), so the dialect
//! is parsed here rather than via `admg_from_dot`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};

use antecedent_core::VariableId;
use antecedent_graph::{Admg, Dag, DenseNodeId};

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
/// # Panics
///
/// Panics if the DOT is malformed.
#[must_use]
pub fn admg_from_oracle_dot(dot: &str) -> (Admg, OracleGraph) {
    let parsed = parse(dot);
    let n = u32::try_from(parsed.observed.len()).expect("node count fits u32");
    let mut g = Admg::with_variables(n);
    for (from, to) in &parsed.directed {
        if parsed.latents.contains(from) || parsed.latents.contains(to) {
            continue;
        }
        g.insert_directed(dense(&parsed.observed, from), dense(&parsed.observed, to)).unwrap();
    }

    let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
    for (from, to) in &parsed.directed {
        if parsed.latents.contains(from) && !parsed.latents.contains(to) {
            children.entry(from.as_str()).or_default().push(to.as_str());
        }
    }
    for obs_children in children.values_mut() {
        obs_children.sort_unstable();
        obs_children.dedup();
        for i in 0..obs_children.len() {
            for j in (i + 1)..obs_children.len() {
                g.insert_bidirected(
                    dense(&parsed.observed, obs_children[i]),
                    dense(&parsed.observed, obs_children[j]),
                )
                .unwrap();
            }
        }
    }
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
