//! The chain-observation recovery route under a Dirichlet posterior mean, against the
//! enumerated truth of the generating SCM.
//!
//! The pattern cells are the exact preimage of the SCM's observed pattern law under the update
//! (`w = u (1 + 9 c) - c`, so the Dirichlet mean with concentration `c` is exactly `u`); the
//! route (mean, then the ordered chain formula) must return the enumerated target law for each
//! head and for a dependent and an independent tail, and the supplied cells must differ visibly
//! from `u`. The SCM is written out independently of the library.

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_posterior_mean,
};
use antecedent_estimate::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryPlan, ChainRecoveryQuery,
    decide_chain_recovery,
};

const IDS: [u32; 6] = [0, 1, 2, 3, 4, 5];

/// Enumerated SCM: returns `(pattern law, truth)` with axes in declaration order.
fn scm(head: usize, dependent: bool) -> (ChainPatternLaw, [[f64; 2]; 2]) {
    let tail = 1 - head;
    let mut both = [[0.0; 2]; 2];
    let mut only = [[0.0; 2]; 2]; // only[axis][x]: that axis observed, the other missing
    let mut neither = 0.0;
    let mut truth = [[0.0; 2]; 2];
    for cell in 0..16usize {
        let x = [(cell >> 3) & 1, (cell >> 2) & 1];
        let r = [(cell >> 1) & 1, cell & 1];
        let p_x1 = if x[0] == 1 { 0.37 } else { 0.63 };
        let p_x2_one = if x[0] == 1 { 0.7 } else { 0.2 };
        let p_x2 = if x[1] == 1 { p_x2_one } else { 1.0 - p_x2_one };
        let p_head = if r[head] == 1 { 0.65 } else { 0.35 };
        let tail_one = if dependent {
            [[0.4, 0.55], [0.3, 0.8]][r[head]][x[head]]
        } else {
            [0.3, 0.8][r[head]]
        };
        let p_tail = if r[tail] == 1 { tail_one } else { 1.0 - tail_one };
        let mass = p_x1 * p_x2 * p_head * p_tail;
        truth[x[0]][x[1]] += mass;
        match (r[0], r[1]) {
            (1, 1) => both[x[0]][x[1]] += mass,
            (1, 0) => only[0][x[0]] += mass,
            (0, 1) => only[1][x[1]] += mass,
            _ => neither += mass,
        }
    }
    let law = ChainPatternLaw::new(both, only[0], only[1], neither).unwrap();
    (law, truth)
}

fn build(order: &[u32], edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in order {
        let id = graph.add_node(NodeRef::Static(VariableId::from_raw(*raw))).unwrap();
        dense.insert(*raw, id);
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    graph
}

fn edges(ids: [u32; 6], head: usize, dependent: bool) -> Vec<(u32, u32)> {
    let [x1, x2, r1, r2, p1, p2] = ids;
    let mut out = vec![(x1, x2), (x1, p1), (r1, p1), (x2, p2), (r2, p2)];
    if head == 0 {
        out.push((r1, r2));
        if dependent {
            out.push((x1, r2));
        }
    } else {
        out.push((r2, r1));
        if dependent {
            out.push((x2, r1));
        }
    }
    out
}

fn query(ids: [u32; 6]) -> ChainRecoveryQuery {
    let v = VariableId::from_raw;
    ChainRecoveryQuery {
        first: ChainPartial { variable: v(ids[0]), response: v(ids[2]), proxy: v(ids[4]) },
        second: ChainPartial { variable: v(ids[1]), response: v(ids[3]), proxy: v(ids[5]) },
    }
}

fn plan(ids: [u32; 6], order: &[u32], head: usize, dependent: bool) -> ChainRecoveryPlan {
    let ctx = ExecutionContext::for_tests(1);
    match decide_chain_recovery(&build(order, &edges(ids, head, dependent)), &query(ids), &ctx)
        .unwrap()
    {
        ChainRecoveryDecision::Recovered(plan) => *plan,
        ChainRecoveryDecision::NonRecoverable(_) => panic!("expected a recovery plan"),
    }
}

fn assert_close(law: &ChainRecoveredLaw, truth: &[[f64; 2]; 2]) {
    for (x1, row) in truth.iter().enumerate() {
        for (x2, expected) in row.iter().enumerate() {
            let got = law.cells()[x1][x2];
            assert!((got - expected).abs() < 1e-12, "cell ({x1},{x2}): {got} vs {expected}");
        }
    }
}

fn pattern_cells(law: &ChainPatternLaw) -> Vec<f64> {
    law.both()
        .iter()
        .flatten()
        .copied()
        .chain(law.only_first().iter().copied())
        .chain(law.only_second().iter().copied())
        .chain(std::iter::once(law.neither()))
        .collect()
}

fn pattern_from(cells: &[f64]) -> ChainPatternLaw {
    ChainPatternLaw::new(
        [[cells[0], cells[1]], [cells[2], cells[3]]],
        [cells[4], cells[5]],
        [cells[6], cells[7]],
        cells[8],
    )
    .unwrap()
}

fn preimage(u: &[f64], c: f64) -> Vec<f64> {
    let k = u.len() as f64;
    u.iter()
        .map(|&p| {
            let w = p * (1.0 + k * c) - c;
            assert!(w >= 0.0, "cell {p} is below the update floor");
            w
        })
        .collect()
}

#[test]
fn chain_recovery_of_the_dirichlet_updated_pattern_law_equals_the_enumerated_truth() {
    let c = 1e-3;
    for head in 0..2 {
        for dependent in [true, false] {
            let (observed, truth) = scm(head, dependent);
            let plan = plan(IDS, &IDS, head, dependent);
            let u = pattern_cells(&observed);
            let counts = preimage(&u, c);
            let moved = counts.iter().zip(&u).map(|(w, p)| (w - p).abs()).fold(0.0, f64::max);
            assert!(moved > 1e-4, "the update must be nontrivial: {moved}");
            let mean = dirichlet_posterior_mean(
                &counts,
                PosteriorRequest::IndependentNormal { scale: c },
                DirichletNamespace::ChainRecovery,
            )
            .unwrap();
            for (m, p) in mean.iter().zip(&u) {
                assert!((m - p).abs() < 1e-12, "{m} vs {p}");
            }
            let recovered = evaluate_chain_recovery(&plan, &pattern_from(&mean)).unwrap();
            assert_close(&recovered, &truth);
        }
    }
}
