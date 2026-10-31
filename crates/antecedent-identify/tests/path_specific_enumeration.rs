//! Path-specific identification restated from the recanting-witness definition and swept over
//! every DAG of up to five nodes.
//!
//! The reference lists every directed path from the treatment to the outcome, decides which are
//! selected by the query's own wording (a path-restricted query selects the paths that visit
//! every named node; a direct contrast selects the paths that visit no mediator; a mediated
//! contrast the paths that visit at least one; a total contrast all of them), and applies the
//! Avin, Shpitser and Pearl definition: a recanting witness is a node `W` other than the
//! treatment reached by a prefix `q` with one continuation to the outcome that makes a selected
//! path and another that makes an unselected one. A DAG query is identified exactly when some
//! no witness exists (an empty selection is zero by structure, hence identified). The product
//! decides the same question from reachability of the treatment's children and shares no code with the enumeration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery, VariableId,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_identify::{IdentificationStatus, IdentificationWorkspace, PathSpecificIdentifier};

fn v(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

struct Graph {
    n: usize,
    child: Vec<Vec<bool>>,
}

impl Graph {
    fn from_mask(n: usize, mask: usize) -> Self {
        let mut child = vec![vec![false; n]; n];
        let mut bit = 0;
        for a in 0..n {
            for b in a + 1..n {
                child[a][b] = mask >> bit & 1 == 1;
                bit += 1;
            }
        }
        Self { n, child }
    }
    fn dag(&self) -> Dag {
        let mut g = Dag::with_variables(u32::try_from(self.n).unwrap());
        for a in 0..self.n {
            for b in 0..self.n {
                if self.child[a][b] {
                    g.insert_directed(
                        DenseNodeId::from_raw(a as u32),
                        DenseNodeId::from_raw(b as u32),
                    )
                    .unwrap();
                }
            }
        }
        g
    }
    /// Every directed path from `from` to `to`, as node lists.
    fn paths(&self, from: usize, to: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut path = vec![from];
        self.walk(&mut path, to, &mut out);
        out
    }
    fn walk(&self, path: &mut Vec<usize>, to: usize, out: &mut Vec<Vec<usize>>) {
        let last = *path.last().unwrap();
        if last == to {
            out.push(path.clone());
            return;
        }
        for next in 0..self.n {
            if self.child[last][next] {
                path.push(next);
                self.walk(path, to, out);
                path.pop();
            }
        }
    }
}

/// Identified exactly when no prefix is shared by a selected and an unselected path. With no path
/// selected the effect is structurally zero, which is identified.
fn reference(paths: &[Vec<usize>], selected: impl Fn(&[usize]) -> bool) -> bool {
    let chosen: Vec<&Vec<usize>> = paths.iter().filter(|p| selected(p)).collect();
    if chosen.is_empty() {
        return true;
    }
    for p in &chosen {
        for q in paths.iter().filter(|q| !selected(q)) {
            // A witness W sits at index k >= 1 of both, with the same prefix up to W.
            let shared = p.iter().zip(q.iter()).take_while(|(a, b)| a == b).count();
            if shared >= 2 {
                return false;
            }
        }
    }
    true
}

fn subsets_of(others: &[usize]) -> Vec<Vec<usize>> {
    (1..1usize << others.len())
        .map(|m| (0..others.len()).filter(|&i| m >> i & 1 == 1).map(|i| others[i]).collect())
        .collect()
}

fn status_ok(status: IdentificationStatus) -> bool {
    match status {
        IdentificationStatus::NonparametricallyIdentified => true,
        IdentificationStatus::NotIdentified => false,
        other => panic!("unexpected {other:?}"),
    }
}

fn sweep(n: usize) -> (usize, usize) {
    let identifier = PathSpecificIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let (mut identified, mut refused) = (0, 0);
    for mask in 0..1usize << (n * (n - 1) / 2) {
        let g = Graph::from_mask(n, mask);
        let prepared = identifier.prepare_dag(&g.dag()).unwrap();
        for t in 0..n {
            for y in t + 1..n {
                let paths = g.paths(t, y);
                if paths.is_empty() {
                    continue;
                }
                let others: Vec<usize> = (0..n).filter(|&i| i != t && i != y).collect();
                let mut run = |query: CausalQuery, expected: bool, context: String| {
                    let got =
                        status_ok(identifier.identify(&prepared, &query, &mut ws).unwrap().status);
                    assert_eq!(got, expected, "{context}");
                    if got {
                        identified += 1;
                    } else {
                        refused += 1;
                    }
                };
                // Unrestricted query: every path selected.
                run(
                    CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(t), v(y))),
                    reference(&paths, |_| true),
                    format!("n={n} mask={mask:b} T={t} Y={y} all"),
                );
                for nodes in subsets_of(&others) {
                    let required = nodes.clone();
                    run(
                        CausalQuery::PathSpecific(
                            PathSpecificEffectQuery::binary(v(t), v(y))
                                .with_path_nodes(nodes.iter().map(|&i| v(i)).collect::<Vec<_>>()),
                        ),
                        reference(&paths, |p| required.iter().all(|r| p.contains(r))),
                        format!("n={n} mask={mask:b} T={t} Y={y} through all of {nodes:?}"),
                    );
                    for (contrast, name) in [
                        (MediationContrast::Total, "total"),
                        (MediationContrast::NaturalDirect, "direct"),
                        (MediationContrast::NaturalIndirect, "indirect"),
                    ] {
                        let m = nodes.clone();
                        run(
                            CausalQuery::Mediation(MediationQuery::binary(
                                v(t),
                                v(y),
                                nodes.iter().map(|&i| v(i)).collect::<Vec<_>>(),
                                contrast,
                            )),
                            reference(&paths, |p| match contrast {
                                MediationContrast::Total => true,
                                MediationContrast::NaturalDirect => {
                                    !m.iter().any(|x| p.contains(x))
                                }
                                _ => m.iter().any(|x| p.contains(x)),
                            }),
                            format!("n={n} mask={mask:b} T={t} Y={y} {name} via {nodes:?}"),
                        );
                    }
                }
            }
        }
    }
    (identified, refused)
}

#[test]
fn identification_matches_the_recanting_witness_definition_on_every_small_dag() {
    for n in 3..=5 {
        let (identified, refused) = sweep(n);
        println!("n={n} identified={identified} refused={refused}");
        // A witness needs a node besides the treatment, the outcome and its own continuation,
        // so three nodes admit none and refuse nothing.
        assert!(identified > 30, "n={n}: {identified} {refused}");
        assert!(n < 4 || refused > 10, "n={n}: {identified} {refused}");
    }
}

#[test]
fn the_reference_finds_the_textbook_witness() {
    // T -> M -> Y and T -> Y with M selected: the direct edge starts an unselected path, but
    // it shares no prefix with the mediated one, so there is no witness.
    let paths = vec![vec![0, 1, 2], vec![0, 2]];
    assert!(reference(&paths, |p| p.contains(&1)));
    // T -> A -> W -> Y and T -> A -> Y with W required: the shared first hop is a witness.
    let paths = vec![vec![0, 1, 2, 3], vec![0, 1, 3]];
    assert!(!reference(&paths, |p| p.contains(&2)));
    // Nothing selected is zero by structure, which is identified.
    assert!(reference(&paths, |p| p.contains(&9)));
}

#[test]
fn path_specific_with_no_matching_path_is_identified_zero() {
    use antecedent_core::DiagnosticKind;
    use antecedent_graph::Admg;
    // X -> M -> Y with a bystander Z on no X -> Y path.
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let identifier = PathSpecificIdentifier::new();
    let prepared = identifier.prepare(&graph).unwrap();
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes(vec![v(3)]),
    );
    let result =
        identifier.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(result.estimands.len(), 1);
    let antecedent_expr::ExprNode::Contrast { left, right, .. } =
        result.arena.node(result.estimands[0].functional)
    else {
        panic!("a structurally zero effect is a contrast of identical arms");
    };
    assert_eq!(left, right);
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref()
            == antecedent_identify::path_specific::PATH_SPECIFIC_STRUCTURAL_ZERO_DIAGNOSTIC_CODE
            && d.kind == DiagnosticKind::Scientific
    }));
}
