//! Cycle refusal and feed-forward ordering of `check_composition` against a definition-level
//! oracle (`oracle_data/interface_cyclic_composition_reference.py`): every digraph on three
//! components with self-connections (512) and on four components without (4096) is decided
//! cyclic or acyclic by transitive closure, and an acyclic graph's order is the
//! lexicographically smallest topological permutation found by scanning all permutations.
//! A refused graph must name exactly one cycle that is a real closed walk over declared
//! connections, simple, and spelled from its smallest component.

use std::collections::BTreeSet;

use antecedent_contract::interface_contract::{
    ComponentConnection, ConsumedPort, ContractOperation, InterfaceContract, ProducedPort,
    SupportDomain, check_composition,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};

const REFERENCE: &str = include_str!("oracle_data/interface_cyclic_composition_reference.txt");
const NAMES: [&str; 4] = ["a", "b", "c", "d"];

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "adults".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// One component that produces `out` and consumes one port per possible producer.
fn component(index: usize, vertices: usize) -> InterfaceContract {
    InterfaceContract {
        component: NAMES[index].into(),
        produces: vec![ProducedPort::new(
            "out",
            quantity(),
            "estimand:ate",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )],
        consumes: (0..vertices)
            .map(|from| {
                ConsumedPort::new(
                    format!("in_{}", NAMES[from]),
                    quantity(),
                    "estimand:ate",
                    SupportDomain::Real,
                    [ContractOperation::Mean],
                )
            })
            .collect(),
    }
}

fn run(vertices: usize, self_loops: bool, lines: &mut std::str::Lines<'_>, expected_edges: usize) {
    let header = lines.next().expect("header");
    assert!(header.starts_with('#'), "{header}");
    let listing: Vec<(usize, usize)> = (0..vertices)
        .flat_map(|i| (0..vertices).map(move |j| (i, j)))
        .filter(|(i, j)| self_loops || i != j)
        .collect();
    assert_eq!(listing.len(), expected_edges);
    let components: Vec<InterfaceContract> =
        (0..vertices).map(|index| component(index, vertices)).collect();
    let mut cyclic = 0;
    for mask in 0..(1usize << listing.len()) {
        let line = lines.next().expect("one line per graph");
        let (stated, expected) = line.split_once(' ').expect("mask and answer");
        assert_eq!(stated.parse::<usize>().expect("mask"), mask);
        let edges: Vec<(usize, usize)> = listing
            .iter()
            .enumerate()
            .filter(|(k, _)| mask >> k & 1 == 1)
            .map(|(_, e)| *e)
            .collect();
        let connections: Vec<ComponentConnection> = edges
            .iter()
            .map(|(i, j)| {
                ComponentConnection::new(NAMES[*i], "out", NAMES[*j], format!("in_{}", NAMES[*i]))
            })
            .collect();
        let result = check_composition(&components, &connections);
        if expected == "-" {
            cyclic += 1;
            let refusals = result.expect_err(&format!("mask {mask} is cyclic"));
            assert_eq!(refusals.len(), 1, "mask {mask}: a cycle is reported alone");
            assert_eq!(refusals[0].detail, "interface_contract.cyclic_composition");
            let path = refusals[0].offending.clone().expect("cycle path");
            let nodes: Vec<&str> = path.split(" -> ").collect();
            assert!(nodes.len() >= 2 && nodes.first() == nodes.last(), "mask {mask}: {path}");
            let body = &nodes[..nodes.len() - 1];
            let distinct: BTreeSet<&&str> = body.iter().collect();
            assert_eq!(distinct.len(), body.len(), "mask {mask}: simple cycle {path}");
            assert_eq!(Some(&nodes[0]), body.iter().min(), "mask {mask}: starts at smallest");
            for pair in nodes.windows(2) {
                let from = NAMES.iter().position(|n| *n == pair[0]).expect("name");
                let to = NAMES.iter().position(|n| *n == pair[1]).expect("name");
                assert!(edges.contains(&(from, to)), "mask {mask}: {path} uses a missing edge");
            }
        } else {
            let composed = result.unwrap_or_else(|e| panic!("mask {mask} is acyclic: {e:?}"));
            assert_eq!(composed.order.concat(), expected, "mask {mask}");
            assert_eq!(composed.connections.len(), edges.len());
        }
    }
    assert!(cyclic > 0);
}

#[test]
fn every_small_digraph_is_ordered_or_refused_exactly_as_the_definition_says() {
    let mut lines = REFERENCE.lines();
    run(3, true, &mut lines, 9);
    run(4, false, &mut lines, 12);
    assert!(lines.next().is_none());
}
