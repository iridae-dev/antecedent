//! Conditional independencies a plan's graph implies, and their tests on a snapshot.
//!
//! A graph over measured columns asserts that some pairs of columns are independent once a
//! set of other columns is held fixed. Those statements can be tested on a table with no model
//! fitted. A baseline records the test of each on the plan's data; an assessment repeats the
//! tests on new data, and an independence that held and now fails is evidence that the graph
//! the plan was identified under no longer describes the data.
//!
//! One statement is listed per pair of non-adjacent columns the graph separates. The
//! conditioning set starts from the other ancestors of the pair, which separates the pair
//! whenever any set of measured columns does, and is then reduced until no column can be
//! dropped. A pair no set separates (two columns joined only through unmeasured causes in a
//! way no measured column blocks) contributes nothing, and a graph that joins every pair lists
//! nothing at all: such a plan carries no test of its own structure.
//!
//! A CPDAG or a PAG stands for a class of graphs. A statement is listed for one only when it
//! holds in every member examined (at most [`MAX_CLASS_MEMBERS`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use antecedent_core::ExecutionContext;
use antecedent_graph::completion::CompletionSampler;
use antecedent_graph::{
    Admg, BitSet, Cpdag, CpdagCompletionSampler, DSeparationWorkspace, Dag, DenseNodeId, Endpoint,
    GraphWorkspace, NodeRef, Pag,
};
use antecedent_io_core::staleness_artifact::{GraphImplicationsWire, ImpliedIndependenceWire};
use antecedent_stats::ci::{
    CiBatchRequest, CiPreparationPlan, CiQuery, CiWorkspace, ConditionalIndependence,
    ConfidenceMethod, SignificanceMethod, ci_from_name,
};

use super::policy::StalenessPolicy;
use super::profile::ReadTable;
use super::{StalenessError, StalenessPlan};

/// Largest graph whose implications are listed.
pub const MAX_IMPLICATION_NODES: usize = 64;
/// Most members of a CPDAG or PAG class a statement is checked against.
pub const MAX_CLASS_MEMBERS: usize = 64;

/// One conditional independence a graph implies: `x` and `y` given `given`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImpliedIndependence {
    /// First column.
    pub x: String,
    /// Second column.
    pub y: String,
    /// Columns held fixed.
    pub given: Vec<String>,
}

/// The conditional independencies a plan's graph implies among its measured columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphImplications {
    /// One statement per pair of columns the graph separates; empty when it separates none.
    Listed(Vec<ImpliedIndependence>),
    /// The graph could not be read for implications, and why.
    Unavailable(String),
}

/// A separation in node positions: `(x, y, given)` with `x < y`.
type Separation = (usize, usize, Vec<usize>);

fn node(index: usize) -> DenseNodeId {
    DenseNodeId::from_index(index)
}

fn adjacent(graph: &Admg, a: DenseNodeId, b: DenseNodeId) -> bool {
    graph.parents(a).contains(&b)
        || graph.parents(b).contains(&a)
        || graph.bidirected_neighbors(a).contains(&b)
}

fn separated(
    graph: &Admg,
    x: DenseNodeId,
    y: DenseNodeId,
    given: &[DenseNodeId],
    workspace: &mut DSeparationWorkspace,
) -> bool {
    graph.is_m_separated(x, y, given, workspace).unwrap_or(false)
}

/// Whether `graph` separates the pair of `separation` given its conditioning set.
fn holds(graph: &Admg, separation: &Separation, workspace: &mut DSeparationWorkspace) -> bool {
    let given: Vec<DenseNodeId> = separation.2.iter().map(|&k| node(k)).collect();
    separated(graph, node(separation.0), node(separation.1), &given, workspace)
}

/// One separation per non-adjacent pair `graph` separates, with a conditioning set from which
/// no column can be dropped.
fn separations(graph: &Admg) -> Vec<Separation> {
    let n = graph.node_count();
    let mut out = Vec::new();
    let mut ancestors = BitSet::with_len(n);
    let mut closure = GraphWorkspace::default();
    let mut workspace = DSeparationWorkspace::default();
    for a in 0..n {
        for b in a + 1..n {
            let (x, y) = (node(a), node(b));
            if adjacent(graph, x, y) {
                continue;
            }
            ancestors.clear();
            graph.ancestors_of(&[x, y], &mut ancestors, &mut closure);
            let mut given: Vec<DenseNodeId> =
                ancestors.iter().filter(|v| *v != x && *v != y).collect();
            if !separated(graph, x, y, &given, &mut workspace) {
                continue;
            }
            let mut k = 0;
            while k < given.len() {
                let dropped = given.remove(k);
                if !separated(graph, x, y, &given, &mut workspace) {
                    given.insert(k, dropped);
                    k += 1;
                }
            }
            out.push((a, b, given.into_iter().map(DenseNodeId::as_usize).collect()));
        }
    }
    out
}

/// The separations of the first member that every other member also has.
fn shared_separations(members: &[Admg]) -> Vec<Separation> {
    let Some((first, rest)) = members.split_first() else {
        return Vec::new();
    };
    let mut workspace = DSeparationWorkspace::default();
    let mut kept = separations(first);
    kept.retain(|separation| rest.iter().all(|member| holds(member, separation, &mut workspace)));
    kept
}

/// A circle-free PAG (a maximal ancestral graph without selection) as an ADMG over the same
/// node positions; `None` when a mark is neither a tail nor an arrowhead.
fn mag_admg(mag: &Pag) -> Option<Admg> {
    let n = mag.node_count();
    let mut admg = Admg::with_variables(u32::try_from(n).ok()?);
    for i in 0..n {
        let a = node(i);
        for (b, at_a, at_b) in mag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => admg.insert_directed(a, b).ok()?,
                (Endpoint::Arrow, Endpoint::Tail) => admg.insert_directed(b, a).ok()?,
                (Endpoint::Arrow, Endpoint::Arrow) => admg.insert_bidirected(a, b).ok()?,
                _ => return None,
            }
        }
    }
    Some(admg)
}

fn too_large(n: usize) -> Option<GraphImplications> {
    (n > MAX_IMPLICATION_NODES).then(|| {
        GraphImplications::Unavailable(format!(
            "the graph has {n} nodes; implications are listed for at most {MAX_IMPLICATION_NODES}"
        ))
    })
}

fn named(
    nodes: &[NodeRef],
    name: &dyn Fn(&NodeRef) -> Option<String>,
    separations: Vec<Separation>,
) -> GraphImplications {
    let Some(names) = nodes.iter().map(name).collect::<Option<Vec<String>>>() else {
        return GraphImplications::Unavailable(
            "a node of the graph is not a column of the plan's data".into(),
        );
    };
    GraphImplications::Listed(
        separations
            .into_iter()
            .map(|(x, y, given)| ImpliedIndependence {
                x: names[x].clone(),
                y: names[y].clone(),
                given: given.into_iter().map(|k| names[k].clone()).collect(),
            })
            .collect(),
    )
}

impl GraphImplications {
    /// The implications of a DAG; `name` gives the column of a node.
    #[must_use]
    pub fn from_dag(dag: &Dag, name: &dyn Fn(&NodeRef) -> Option<String>) -> Self {
        if let Some(refused) = too_large(dag.node_count()) {
            return refused;
        }
        match Admg::from_dag(dag) {
            Ok(admg) => named(dag.nodes(), name, separations(&admg)),
            Err(error) => Self::Unavailable(format!("the graph could not be read: {error}")),
        }
    }

    /// The implications of an ADMG (m-separation); `name` gives the column of a node.
    #[must_use]
    pub fn from_admg(admg: &Admg, name: &dyn Fn(&NodeRef) -> Option<String>) -> Self {
        too_large(admg.node_count()).unwrap_or_else(|| named(admg.nodes(), name, separations(admg)))
    }

    /// The implications every examined DAG of a CPDAG's class shares.
    #[must_use]
    pub fn from_cpdag(cpdag: &Cpdag, name: &dyn Fn(&NodeRef) -> Option<String>) -> Self {
        if let Some(refused) = too_large(cpdag.node_count()) {
            return refused;
        }
        let Ok(sampler) = CpdagCompletionSampler::new(cpdag.clone(), MAX_CLASS_MEMBERS) else {
            return Self::Unavailable("the CPDAG holds conflict marks".into());
        };
        let members: Vec<Admg> =
            sampler.filter_map(|member| Admg::from_dag(&member.graph).ok()).collect();
        if members.is_empty() {
            return Self::Unavailable(
                "no DAG completes the CPDAG within the completion search budget".into(),
            );
        }
        named(cpdag.nodes(), name, shared_separations(&members))
    }

    /// The implications every examined maximal ancestral graph of a PAG's class shares.
    #[must_use]
    pub fn from_pag(pag: &Pag, name: &dyn Fn(&NodeRef) -> Option<String>) -> Self {
        if let Some(refused) = too_large(pag.node_count()) {
            return refused;
        }
        let Ok(sampler) = CompletionSampler::new(pag.clone(), MAX_CLASS_MEMBERS) else {
            return Self::Unavailable("the PAG has no ancestral completion".into());
        };
        let members: Vec<Admg> = sampler.filter_map(|member| mag_admg(&member.graph)).collect();
        if members.is_empty() {
            return Self::Unavailable(
                "no ancestral graph completes the PAG within the completion search budget".into(),
            );
        }
        named(pag.nodes(), name, shared_separations(&members))
    }
}

/// The policy's conditional-independence test, when it gives an analytic frequentist p-value.
pub(crate) fn resolve_test(
    policy: &StalenessPolicy,
) -> Result<std::sync::Arc<dyn ConditionalIndependence + Send + Sync>, StalenessError> {
    let test = ci_from_name(&policy.implication_test).map_err(|_| {
        StalenessError::Policy("implication_test names no conditional-independence test")
    })?;
    if !test.p_value_is_frequentist() {
        return Err(StalenessError::Policy(
            "implication_test must report a frequentist p-value, not a posterior probability",
        ));
    }
    if test.min_attainable_p(SignificanceMethod::Analytic) != 0.0 {
        return Err(StalenessError::Policy(
            "implication_test must have an analytic p-value; permutation-null tests are not run \
             by the staleness diagnostic",
        ));
    }
    Ok(test)
}

/// The position of numeric column `name` among `names`, adding it when it is new; `None` when
/// the table has no such numeric column.
fn locate<'n>(table: &ReadTable<'_>, names: &mut Vec<&'n str>, name: &'n str) -> Option<usize> {
    table.get(name).filter(|read| read.is_numeric())?;
    Some(names.iter().position(|known| *known == name).unwrap_or_else(|| {
        names.push(name);
        names.len() - 1
    }))
}

/// Statistic and p-value of each statement on `table`, in order; `None` where a column is
/// absent or not numeric, too few complete rows remain, or the test cannot be formed.
pub(crate) fn test_statements(
    statements: &[(&str, &str, Vec<&str>)],
    table: &ReadTable<'_>,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<(f64, f64)>>, StalenessError> {
    let test = resolve_test(policy)?;
    let mut names: Vec<&str> = Vec::new();
    let mut located: Vec<Option<(usize, usize, Vec<usize>)>> = Vec::with_capacity(statements.len());
    for (x, y, given) in statements {
        let positions = (|| {
            let x = locate(table, &mut names, x)?;
            let y = locate(table, &mut names, y)?;
            let given = given
                .iter()
                .map(|name| locate(table, &mut names, name))
                .collect::<Option<Vec<usize>>>()?;
            Some((x, y, given))
        })();
        located.push(positions);
    }
    if names.is_empty() {
        return Ok(vec![None; statements.len()]);
    }
    let reads: Vec<_> = names.iter().filter_map(|name| table.get(name)).collect();
    let complete: Vec<usize> =
        (0..table.rows).filter(|&i| reads.iter().all(|read| read.value(i).is_finite())).collect();
    let _lease;
    let storage: Vec<Cow<'_, [f64]>> = if complete.len() == table.rows {
        reads
            .iter()
            .map(|read| match read {
                super::profile::ColumnRead::Numeric(values) => Cow::Borrowed(values.as_ref()),
                _ => Cow::Owned(Vec::new()),
            })
            .collect()
    } else {
        let bytes = (complete.len() as u64)
            .saturating_mul(reads.len() as u64)
            .saturating_mul(size_of::<f64>() as u64);
        _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => StalenessError::Cancelled,
            other => StalenessError::Admission { bytes, reason: other.to_string() },
        })?;
        reads
            .iter()
            .map(|read| Cow::Owned(complete.iter().map(|&i| read.value(i)).collect()))
            .collect()
    };
    let columns: Vec<&[f64]> = storage.iter().map(AsRef::as_ref).collect();
    let plan = CiPreparationPlan {
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::None,
    };
    let Ok(prepared) = test.prepare(&columns, &plan, ctx) else {
        return Ok(vec![None; statements.len()]);
    };
    let mut workspace = CiWorkspace::default();
    let mut out = Vec::with_capacity(statements.len());
    for positions in located {
        super::check_cancelled(ctx)?;
        let Some((x, y, given)) = positions else {
            out.push(None);
            continue;
        };
        if complete.len() < given.len() + 4 {
            out.push(None);
            continue;
        }
        let queries = [CiQuery { x, y, z_start: 0, z_len: given.len() }];
        let request = CiBatchRequest {
            columns: &columns,
            queries: &queries,
            z_flat: &given,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let result = test
            .test_batch(&prepared, &request, &mut workspace, ctx)
            .ok()
            .and_then(|batch| batch.results.first().map(|r| (r.statistic, r.p_value)))
            .filter(|(statistic, p_value)| statistic.is_finite() && p_value.is_finite());
        out.push(result);
    }
    Ok(out)
}

fn statements_of(items: &[ImpliedIndependenceWire]) -> Vec<(&str, &str, Vec<&str>)> {
    items
        .iter()
        .map(|item| {
            (item.x.as_str(), item.y.as_str(), item.given.iter().map(String::as_str).collect())
        })
        .collect()
}

/// The record a baseline keeps of the plan's implications, tested on the plan's data.
pub(crate) fn record(
    plan: &StalenessPlan,
    table: &ReadTable<'_>,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<Option<GraphImplicationsWire>, StalenessError> {
    let test = policy.implication_test.clone();
    let listed = match &plan.implications {
        None => return Ok(None),
        Some(GraphImplications::Unavailable(reason)) => {
            return Ok(Some(GraphImplicationsWire {
                test,
                unavailable: Some(reason.clone()),
                items: Vec::new(),
            }));
        }
        Some(GraphImplications::Listed(listed)) => listed,
    };
    if listed.len() > policy.max_implications {
        return Ok(Some(GraphImplicationsWire {
            test,
            unavailable: Some(format!(
                "the graph implies {} conditional independencies, more than max_implications ({})",
                listed.len(),
                policy.max_implications
            )),
            items: Vec::new(),
        }));
    }
    let statements: Vec<(&str, &str, Vec<&str>)> = listed
        .iter()
        .map(|item| {
            (item.x.as_str(), item.y.as_str(), item.given.iter().map(String::as_str).collect())
        })
        .collect();
    let tested = test_statements(&statements, table, policy, ctx)?;
    let items = listed
        .iter()
        .zip(tested)
        .map(|(item, result)| ImpliedIndependenceWire {
            x: item.x.clone(),
            y: item.y.clone(),
            given: item.given.clone(),
            statistic: result.map(|r| r.0),
            p_value: result.map(|r| r.1),
        })
        .collect();
    Ok(Some(GraphImplicationsWire { test, unavailable: None, items }))
}

/// The recorded statements tested again on `table`.
pub(crate) fn retest(
    recorded: &GraphImplicationsWire,
    table: &ReadTable<'_>,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<(f64, f64)>>, StalenessError> {
    test_statements(&statements_of(&recorded.items), table, policy, ctx)
}

/// `x _||_ y | a, b`, the subject of a finding about one statement.
pub(crate) fn subject(item: &ImpliedIndependenceWire) -> String {
    if item.given.is_empty() {
        format!("{} _||_ {}", item.x, item.y)
    } else {
        format!("{} _||_ {} | {}", item.x, item.y, item.given.join(", "))
    }
}

#[cfg(test)]
#[path = "implications_tests.rs"]
mod scenario_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn index_name(node: &NodeRef) -> Option<String> {
        match node {
            NodeRef::Static(variable) => Some(format!("v{}", variable.raw())),
            _ => None,
        }
    }

    fn listed(implications: GraphImplications) -> Vec<(String, String, Vec<String>)> {
        match implications {
            GraphImplications::Listed(items) => {
                items.into_iter().map(|i| (i.x, i.y, i.given)).collect()
            }
            GraphImplications::Unavailable(reason) => panic!("unavailable: {reason}"),
        }
    }

    fn statement(x: &str, y: &str, given: &[&str]) -> (String, String, Vec<String>) {
        (x.into(), y.into(), given.iter().map(|g| (*g).to_owned()).collect())
    }

    fn dag(n: u32, edges: &[(usize, usize)]) -> Dag {
        let mut dag = Dag::with_variables(n);
        for &(from, to) in edges {
            dag.insert_directed(node(from), node(to)).unwrap();
        }
        dag
    }

    #[test]
    fn a_chain_separates_its_ends_given_the_middle_and_a_collider_given_nothing() {
        // 0 -> 1 -> 2: the ends are independent given the middle.
        let chain = GraphImplications::from_dag(&dag(3, &[(0, 1), (1, 2)]), &index_name);
        assert_eq!(listed(chain), vec![statement("v0", "v2", &["v1"])]);
        // 0 -> 2 <- 1: the causes are independent given nothing; the collider is not held fixed.
        let collider = GraphImplications::from_dag(&dag(3, &[(0, 2), (1, 2)]), &index_name);
        assert_eq!(listed(collider), vec![statement("v0", "v1", &[])]);
    }

    #[test]
    fn a_graph_that_joins_every_pair_lists_nothing() {
        let complete = GraphImplications::from_dag(&dag(3, &[(0, 1), (0, 2), (1, 2)]), &index_name);
        assert!(listed(complete).is_empty());
    }

    #[test]
    fn an_unmeasured_common_cause_removes_the_statements_it_breaks() {
        // 0 -> 1 -> 2 with 0 <-> 2: the ends are adjacent, so nothing is implied.
        let mut bow = Admg::from_dag(&dag(3, &[(0, 1), (1, 2)])).unwrap();
        bow.insert_bidirected(node(0), node(2)).unwrap();
        assert!(listed(GraphImplications::from_admg(&bow, &index_name)).is_empty());
        // 0 -> 1, 2 -> 3, 1 <-> 3: 0 and 2 are independent given nothing, and 0 and 3 too,
        // but 0 and 3 become dependent once 1 is held fixed, so 1 is never in the set.
        let mut mixed = Admg::from_dag(&dag(4, &[(0, 1), (2, 3)])).unwrap();
        mixed.insert_bidirected(node(1), node(3)).unwrap();
        assert_eq!(
            listed(GraphImplications::from_admg(&mixed, &index_name)),
            vec![
                statement("v0", "v2", &[]),
                statement("v0", "v3", &[]),
                statement("v1", "v2", &[]),
            ]
        );
    }

    #[test]
    fn every_listed_statement_is_a_separation_of_the_graph_with_no_column_to_spare() {
        let graph = dag(6, &[(0, 2), (1, 2), (2, 3), (1, 4), (3, 5), (4, 5)]);
        let admg = Admg::from_dag(&graph).unwrap();
        let mut workspace = DSeparationWorkspace::default();
        let found = separations(&admg);
        assert!(!found.is_empty());
        for separation in &found {
            assert!(holds(&admg, separation, &mut workspace), "{separation:?}");
            for skip in 0..separation.2.len() {
                let mut fewer = separation.clone();
                fewer.2.remove(skip);
                assert!(!holds(&admg, &fewer, &mut workspace), "{separation:?} minus {skip}");
            }
        }
        // Every non-adjacent pair of a DAG is separated by some set, so each is listed once.
        let non_adjacent = (0..6)
            .flat_map(|a| (a + 1..6).map(move |b| (a, b)))
            .filter(|&(a, b)| !adjacent(&admg, node(a), node(b)))
            .count();
        assert_eq!(found.len(), non_adjacent);
    }

    #[test]
    fn a_class_lists_only_what_all_of_its_members_share() {
        // The chain 0 - 1 - 2 as a CPDAG: every member separates the ends given the middle.
        let cpdag = Cpdag::from_dag(&dag(3, &[(0, 1), (1, 2)]));
        assert_eq!(
            listed(GraphImplications::from_cpdag(&cpdag, &index_name)),
            vec![statement("v0", "v2", &["v1"])]
        );
        // The same class as a PAG with circle marks at both ends of both edges.
        let mut pag = Pag::with_variables(3);
        pag.insert_circle_circle(node(0), node(1)).unwrap();
        pag.insert_circle_circle(node(1), node(2)).unwrap();
        assert_eq!(
            listed(GraphImplications::from_pag(&pag, &index_name)),
            vec![statement("v0", "v2", &["v1"])]
        );
    }

    #[test]
    fn a_graph_over_the_node_limit_is_unavailable_with_the_reason() {
        let big = Dag::with_variables(u32::try_from(MAX_IMPLICATION_NODES + 1).unwrap());
        assert!(matches!(
            GraphImplications::from_dag(&big, &index_name),
            GraphImplications::Unavailable(reason) if reason.contains("at most 64")
        ));
    }
}
