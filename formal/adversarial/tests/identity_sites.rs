//! Identity sites of `formal/identity_sites.toml`, checked on the production encoders.
//!
//! Exact-law content identity (`ExactDiscreteLaw::content_identity`, scheme v2): the
//! intervention world is a set, every other field is significant, and the framing is
//! injective, including where the v1 `Debug` descriptor collided. Static graph identity
//! (`dag_identity`): independent of the dense order nodes were added in, and two DAGs share it
//! only when their variables and edges agree. Canonical values (`Value` on `CanonicalWriter`):
//! two values share a digest only when they are bit-equal, so `-0.0` and `0.0` differ and NaN
//! payloads are kept.
//!
//! The expected outcome of every case is the stated identity rule, computed here on the
//! generated input, never an earlier output of the code under test.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalEncode, CanonicalWriter, NodeRef, RegimeId, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io_core::identity::dag_identity;
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

// ---------------------------------------------------------------------------
// Exact-law content identity.
// ---------------------------------------------------------------------------

/// The generated fields of a law over one binary axis (variable 0).
#[derive(Clone, Debug)]
struct LawFields {
    population: String,
    regime: u32,
    /// `(variable, level)` with distinct variables, in the order listed.
    interventions: Vec<(u32, i64)>,
    snapshot: String,
    first_cell: f64,
}

const CELLS: [f64; 5] = [0.0, -0.0, 0.25, 0.5, 0.75];

fn law_fields() -> impl Strategy<Value = LawFields> {
    (
        "[a-z:]{1,5}",
        0u32..3,
        prop::collection::btree_map(5u32..8, 0i64..3, 0..3),
        "[a-z:]{1,5}",
        prop::sample::select(CELLS.to_vec()),
    )
        .prop_flat_map(|(population, regime, world, snapshot, first_cell)| {
            let listed: Vec<(u32, i64)> = world.into_iter().collect();
            (
                Just(population),
                Just(regime),
                Just(listed).prop_shuffle(),
                Just(snapshot),
                Just(first_cell),
            )
        })
        .prop_map(|(population, regime, interventions, snapshot, first_cell)| LawFields {
            population,
            regime,
            interventions,
            snapshot,
            first_cell,
        })
}

fn law(fields: &LawFields) -> ExactDiscreteLaw {
    let axis = DiscreteAxis {
        variable: VariableId::from_raw(0),
        values: Arc::from([Value::Int64(0), Value::Int64(1)]),
    };
    let world: Vec<InterventionAssignment> = fields
        .interventions
        .iter()
        .map(|&(variable, level)| {
            InterventionAssignment::concrete(VariableId::from_raw(variable), Value::Int64(level))
        })
        .collect();
    let second = 1.0 - fields.first_cell;
    ExactDiscreteLaw::try_new(
        fields.population.as_str(),
        RegimeId::from_raw(fields.regime),
        world,
        [axis],
        [fields.first_cell, second],
        fields.snapshot.as_str(),
        LawTolerance::default(),
    )
    .expect("a valid generated law")
}

type LawKey = (String, u32, Vec<(u32, i64)>, String, u64);

/// The law as the identity rule reads it: the intervention world as a set, cells by bits.
fn law_key(fields: &LawFields) -> LawKey {
    let mut world = fields.interventions.clone();
    world.sort_unstable();
    (
        fields.population.clone(),
        fields.regime,
        world,
        fields.snapshot.clone(),
        fields.first_cell.to_bits(),
    )
}

// ---------------------------------------------------------------------------
// Static graph identity.
// ---------------------------------------------------------------------------

/// A labelled DAG: distinct variables per slot, and edges `slot i -> slot j` for `i < j`.
#[derive(Clone, Debug)]
struct LabelledDag {
    labels: Vec<u32>,
    edges: Vec<(usize, usize)>,
}

fn labelled_dag() -> impl Strategy<Value = LabelledDag> {
    prop::collection::btree_set(0u32..8, 1..6)
        .prop_flat_map(|set| {
            let labels: Vec<u32> = set.into_iter().collect();
            let n = labels.len();
            let pairs: Vec<(usize, usize)> =
                (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
            let count = pairs.len();
            (Just(labels).prop_shuffle(), Just(pairs), prop::collection::vec(any::<bool>(), count))
        })
        .prop_map(|(labels, pairs, keep)| LabelledDag {
            labels,
            edges: pairs
                .into_iter()
                .zip(keep)
                .filter(|(_, keep)| *keep)
                .map(|(pair, _)| pair)
                .collect(),
        })
}

/// Build the DAG adding its slots in `order` (a permutation of the slots).
fn build(graph: &LabelledDag, order: &[usize]) -> Dag {
    let mut dag = Dag::empty();
    let mut dense = vec![DenseNodeId::from_raw(0); graph.labels.len()];
    for (position, &slot) in order.iter().enumerate() {
        dag.add_node(NodeRef::Static(VariableId::from_raw(graph.labels[slot]))).unwrap();
        dense[slot] = DenseNodeId::from_raw(u32::try_from(position).unwrap());
    }
    for &(from, to) in &graph.edges {
        dag.insert_directed(dense[from], dense[to]).unwrap();
    }
    dag
}

/// The DAG as the identity rule reads it: its variables and its variable-pair edges.
fn graph_key(graph: &LabelledDag) -> (Vec<u32>, Vec<(u32, u32)>) {
    let mut variables = graph.labels.clone();
    variables.sort_unstable();
    let mut edges: Vec<(u32, u32)> =
        graph.edges.iter().map(|&(from, to)| (graph.labels[from], graph.labels[to])).collect();
    edges.sort_unstable();
    (variables, edges)
}

// ---------------------------------------------------------------------------
// Canonical values.
// ---------------------------------------------------------------------------

fn value() -> impl Strategy<Value = Value> {
    prop_oneof![
        prop::sample::select(vec![0.0, -0.0, 1.0, f64::NAN, -f64::NAN, f64::INFINITY])
            .prop_map(Value::Float64),
        any::<u64>().prop_map(|bits| Value::Float64(f64::from_bits(bits))),
        (-2i64..3).prop_map(Value::Int64),
        any::<bool>().prop_map(Value::Bool),
        (0u32..3).prop_map(Value::Category),
        "[ab]{0,2}".prop_map(|label| Value::Label(Arc::from(label.as_str()))),
    ]
}

fn value_digest(value: &Value) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.adversarial.value");
    value.encode(&mut writer);
    writer.finish()
}

proptest! {
    #![proptest_config(config())]

    /// Listing the intervention world in another order keeps the law identity.
    #[test]
    fn law_identity_intervention_order_keeps_identity(fields in law_fields()) {
        let mut reordered = fields.clone();
        reordered.interventions.reverse();
        prop_assert_eq!(law(&fields).content_identity(), law(&reordered).content_identity());
    }

    /// Two laws share the v2 identity exactly when the rule says they are the same law: every
    /// field equal, the intervention world compared as a set and cells by bit pattern. The
    /// population and snapshot alphabet contains the v1 separator.
    #[test]
    fn law_identity_is_injective_on_law_fields(left in law_fields(), right in law_fields()) {
        let same = law_key(&left) == law_key(&right);
        prop_assert_eq!(law(&left).content_identity() == law(&right).content_identity(), same);
        prop_assert!(law(&left).accepts_content_identity(&law(&left).content_identity_v1()));
        // Each field alone is significant: a law that differs from `left` only in its snapshot
        // or only in its population has another identity.
        let mut snapshot = left.clone();
        snapshot.snapshot.push('z');
        prop_assert_ne!(law(&left).content_identity(), law(&snapshot).content_identity());
        let mut population = left.clone();
        population.population.push('z');
        prop_assert_ne!(law(&left).content_identity(), law(&population).content_identity());
    }

    /// Where the v1 descriptor joins raw text across a field boundary, two different laws share
    /// the v1 identity; the v2 identity keeps them apart.
    #[test]
    fn law_identity_separator_collision_is_distinct(head in "[a-z]{1,3}", tail in "[a-z]{1,3}") {
        let axes = format!(
            "{:?}",
            [DiscreteAxis {
                variable: VariableId::from_raw(0),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            }]
        );
        let regime = |raw: u32| format!("{:?}", RegimeId::from_raw(raw));
        let first = LawFields {
            population: head.clone(),
            regime: 0,
            interventions: vec![],
            snapshot: format!("x:{}:[]:{axes}:{tail}", regime(1)),
            first_cell: 0.25,
        };
        let second = LawFields {
            population: format!("{head}:{}:[]:{axes}:x", regime(0)),
            regime: 1,
            interventions: vec![],
            snapshot: tail,
            first_cell: 0.25,
        };
        prop_assert_ne!(law_key(&first), law_key(&second));
        prop_assert_eq!(law(&first).content_identity_v1(), law(&second).content_identity_v1());
        prop_assert_ne!(law(&first).content_identity(), law(&second).content_identity());
    }

    /// The DAG identity does not depend on the dense order its nodes were added in.
    #[test]
    fn graph_identity_dense_order_keeps_identity(
        (graph, order) in labelled_dag().prop_flat_map(|graph| {
            let slots: Vec<usize> = (0..graph.labels.len()).collect();
            (Just(graph), Just(slots).prop_shuffle())
        })
    ) {
        let natural: Vec<usize> = (0..graph.labels.len()).collect();
        let first = dag_identity(&build(&graph, &natural)).unwrap();
        let second = dag_identity(&build(&graph, &order)).unwrap();
        prop_assert!(first.is_canonical());
        prop_assert_eq!(first, second);
    }

    /// Two DAGs share an identity wire exactly when they have the same variables and edges.
    #[test]
    fn graph_identity_binds_variables_and_edges(left in labelled_dag(), right in labelled_dag()) {
        let natural = |graph: &LabelledDag| (0..graph.labels.len()).collect::<Vec<usize>>();
        let same = graph_key(&left) == graph_key(&right);
        let left_identity = dag_identity(&build(&left, &natural(&left))).unwrap();
        let right_identity = dag_identity(&build(&right, &natural(&right))).unwrap();
        prop_assert_eq!(left_identity == right_identity, same);
    }

    /// Two canonical values share a digest exactly when they are bit-equal.
    #[test]
    fn value_encoding_is_injective(left in value(), right in value()) {
        prop_assert_eq!(value_digest(&left) == value_digest(&right), left == right);
        // The pair a merged encoding would confuse first: the two zeros differ in sign bit.
        prop_assert_ne!(value_digest(&Value::Float64(0.0)), value_digest(&Value::Float64(-0.0)));
    }
}
