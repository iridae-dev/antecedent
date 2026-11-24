//! Identity sites of decisions across trade-offs and of preference learning, on the production
//! encoders through the public API: the value model, the preference contract, the choice record
//! set, and the member fingerprints of a Markov equivalence class and of a PAG class.
//!
//! The law each site obeys comes from its classification in formal/identity_sites.toml
//! (see tests/common/mod.rs).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_model::{
    Attribute, BeliefRows, LinearRelation, PreferenceExpr as P, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel, ValueRelation,
};
use antecedent_graph::{Dag, DenseNodeId, Pag, dag_fingerprint, mag_fingerprint};
use proptest::prelude::*;

use common::{distinct_permuted, injective, ordered_laws, set_laws};

const IDS: [&str; 4] = ["a", "b", "c", "d"];

fn parameter(id: &str) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "utility/unit".to_owned(),
        lower: 0.0,
        upper: 4.0,
        assumption_id: format!("{id}-judgment"),
    }
}

/// A relation over the three parameters, its terms in the order `indices` lists them; the
/// coefficient of a parameter depends on the relation and the parameter, never on the order.
fn relation(id: &str, indices: &[usize]) -> LinearRelation {
    let scale = id.bytes().map(f64::from).sum::<f64>();
    LinearRelation {
        id: id.to_owned(),
        terms: indices.iter().map(|&j| (j, scale + j as f64)).collect(),
        sense: RelationSense::AtLeast,
        bound: -scale,
        origin: RelationOrigin::Declared,
    }
}

fn value_relation(id: &str) -> ValueRelation {
    let scale = id.bytes().map(f64::from).sum::<f64>();
    ValueRelation {
        id: format!("value-{id}"),
        origin: RelationOrigin::Choice {
            record_id: format!("record-{id}"),
            rejected_option: "wait".to_owned(),
        },
        chosen: BeliefRows { rows: vec![scale, 1.0], weights: vec![1.0] },
        rejected: BeliefRows { rows: vec![1.0, scale], weights: vec![1.0] },
        tolerance: 0.0,
    }
}

/// `p0 * benefit + p1 * cost + p2 * benefit`, over three declared parameters.
fn model(relations: Vec<LinearRelation>, value_relations: Vec<ValueRelation>) -> ValueModel {
    let mut preferences =
        PreferenceSet::new(vec![parameter("p0"), parameter("p1"), parameter("p2")], relations);
    preferences.value_relations = value_relations;
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".to_owned(), units: "events".to_owned() },
            Attribute { id: "cost".to_owned(), units: "usd".to_owned() },
        ],
        preferences,
        value: P::sum(
            P::sum(
                P::product(P::Parameter(0), P::Attribute(0)),
                P::product(P::Parameter(1), P::Attribute(1)),
            ),
            P::product(P::Parameter(2), P::Attribute(0)),
        ),
        utility_units: "utility".to_owned(),
    }
}

fn quantity(variable: &str, regime: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn action(id: &str) -> PreferenceAction {
    PreferenceAction {
        id: id.to_owned(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("benefit", &format!("do({id})"), "events")),
            ActionInput::Fixed { value: id.bytes().map(f64::from).sum(), units: "usd".to_owned() },
        ],
    }
}

fn constraint(id: &str, applies_to: &[&str]) -> HardConstraint {
    HardConstraint {
        id: id.to_owned(),
        expr: UtilityExpr::Input(1),
        bound: 500.0,
        min_probability: 0.5,
        units: "usd".to_owned(),
        applies_to: applies_to.iter().map(|a| (*a).to_owned()).collect(),
    }
}

fn contract(actions: &[&str], constraints: &[&str], applies_to: &[&str]) -> PreferenceContract {
    PreferenceContract {
        model: model(vec![relation("r", &[0, 1])], Vec::new()),
        actions: actions.iter().map(|id| action(id)).collect(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: constraints.iter().map(|id| constraint(id, applies_to)).collect(),
        target_population: "target".to_owned(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

fn record(id: &str, options: &[&str], rows: &[String]) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: "clinic".to_owned(),
        options: options
            .iter()
            .map(|option| ChoiceOption {
                id: (*option).to_owned(),
                belief: OptionBelief::Means(vec![option.bytes().map(f64::from).sum(), 1.0]),
            })
            .collect(),
        chosen: options[0].to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast".to_owned() },
        observation_rows: rows.to_vec(),
    }
}

fn records(ids: &[&str]) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: vec![
            Attribute { id: "benefit".to_owned(), units: "events".to_owned() },
            Attribute { id: "cost".to_owned(), units: "usd".to_owned() },
        ],
        population: "adults".to_owned(),
        records: ids
            .iter()
            .map(|id| record(id, &["treat", "wait"], &[format!("visit#{id}")]))
            .collect(),
        pooling: Pooling::Pooled { assumption_id: "shared".to_owned() },
    }
}

fn node(index: usize) -> DenseNodeId {
    DenseNodeId::from_index(index)
}

/// Edges `i -> j` with `i < j` over four nodes: every list of them is acyclic.
fn forward_edges() -> Vec<(usize, usize)> {
    (0..4).flat_map(|i| ((i + 1)..4).map(move |j| (i, j))).collect()
}

proptest! {
    #![proptest_config(common::config())]

    /// `ValueModel::identity` and `encode_relation`: the linear relations are a set (sorted
    /// digests; validate refuses a repeated id), and the terms of one relation are a set of
    /// parameter indices (sorted; validate refuses a repeated index). Attributes and parameters
    /// are ordered: the expression indexes them.
    #[test]
    fn site_value_model_relations_and_terms(
        (ids, shuffled) in distinct_permuted(prop::sample::select(IDS.to_vec()), 0..5),
        (terms, shuffled_terms) in distinct_permuted(0usize..3, 1..4),
        i in 0usize..2,
        j in 0usize..2,
    ) {
        let relations = |ids: &[&str]| -> Vec<LinearRelation> {
            ids.iter().map(|id| relation(id, &[0, 1])).collect()
        };
        set_laws(&ids, &shuffled, |ids| model(relations(ids), Vec::new()).identity().ok())?;
        set_laws(&terms, &shuffled_terms, |terms| {
            model(vec![relation("r", terms)], Vec::new()).identity().ok()
        })?;
        let attributes = [
            Attribute { id: "benefit".to_owned(), units: "events".to_owned() },
            Attribute { id: "cost".to_owned(), units: "usd".to_owned() },
        ];
        ordered_laws(&attributes, i, j, |attributes| {
            let mut m = model(Vec::new(), Vec::new());
            m.attributes = attributes.to_vec();
            m.identity().ok()
        })?;
    }

    /// `ValueModel::identity` and `encode_value_relation`: the value relations are a set
    /// (sorted digests; validate refuses a repeated id, shared with the linear relations).
    #[test]
    fn site_value_model_value_relations(
        (ids, shuffled) in distinct_permuted(prop::sample::select(IDS.to_vec()), 0..5),
    ) {
        set_laws(&ids, &shuffled, |ids| {
            model(Vec::new(), ids.iter().map(|id| value_relation(id)).collect()).identity().ok()
        })?;
    }

    /// `PreferenceContract::identity` and `encode_action`: actions and constraints are ordered
    /// (results are reported in action order, so their declaration-order ids are written after
    /// the sorted digests); a constraint's `applies_to` is a set (`constraint_digest`).
    #[test]
    fn site_preference_contract_actions_ordered(
        (actions, _) in distinct_permuted(prop::sample::select(IDS.to_vec()), 2..5),
        (applies, shuffled_applies) in distinct_permuted(prop::sample::select(IDS.to_vec()), 1..3),
        i in 0usize..4,
        j in 0usize..4,
    ) {
        ordered_laws(&actions, i, j, |actions| contract(actions, &[], &[]).identity().ok())?;
        let constraints = ["budget", "floor"];
        ordered_laws(&constraints, i, j, |constraints| {
            contract(&IDS, constraints, &[]).identity().ok()
        })?;
        set_laws(&applies, &shuffled_applies, |applies| {
            contract(&IDS, &["budget"], applies).identity().ok()
        })?;
    }

    /// `ChoiceRecordSet::digest` and `encode_record`: records are a set (sorted by id;
    /// validate refuses a repeated id); a record's observation rows are a set (trimmed and
    /// sorted; validate refuses a repeat); a record's options keep their declared order.
    #[test]
    fn site_choice_record_set_digest(
        (ids, shuffled) in distinct_permuted(prop::sample::select(IDS.to_vec()), 0..5),
        (rows, shuffled_rows) in distinct_permuted(prop::sample::select(IDS.to_vec()), 1..5),
        pad in prop::sample::select(vec!["", " ", "  "]),
    ) {
        set_laws(&ids, &shuffled, |ids| records(ids).digest().ok())?;
        let with_rows = |rows: &[&str]| {
            let mut set = records(&["r"]);
            set.records[0].observation_rows =
                rows.iter().map(|row| format!("{pad}row#{row}{pad}")).collect();
            set.digest().ok()
        };
        set_laws(&rows, &shuffled_rows, with_rows)?;
        let options = ["treat", "wait"];
        ordered_laws(&options, 0, 1, |options| {
            let mut set = records(&["r"]);
            set.records[0] = record("r", options, &["visit#r".to_owned()]);
            set.records[0].chosen = "treat".to_owned();
            set.digest().ok()
        })?;
    }

    /// `dag_fingerprint`: the directed edges of a member are a set (sorted pairs; a repeated
    /// edge is refused when it is inserted), whatever order they were inserted in.
    #[test]
    fn site_mec_uniform_dag_fingerprint(
        (edges, shuffled) in distinct_permuted(prop::sample::select(forward_edges()), 0..6),
    ) {
        let identity = |edges: &[(usize, usize)]| {
            let mut dag = Dag::with_variables(4);
            for &(from, to) in edges {
                dag.insert_directed(node(from), node(to)).ok()?;
            }
            Some(dag_fingerprint(&dag))
        };
        set_laws(&edges, &shuffled, identity)?;
    }

    /// `mag_fingerprint`: the edges of a member with their two marks are a set (each node's
    /// higher neighbours sorted; a second edge between two nodes is refused when it is
    /// inserted), whatever order they were inserted in; the marks of an edge are framed.
    #[test]
    fn site_pag_uniform_mag_fingerprint(
        (edges, shuffled) in distinct_permuted(prop::sample::select(forward_edges()), 0..6),
    ) {
        let identity = |edges: &[(usize, usize)]| {
            let mut pag = Pag::with_variables(4);
            for &(from, to) in edges {
                if (from + to) % 2 == 0 {
                    pag.insert_directed(node(from), node(to)).ok()?;
                } else {
                    pag.insert_bidirected(node(from), node(to)).ok()?;
                }
            }
            Some(mag_fingerprint(&pag))
        };
        set_laws(&edges, &shuffled, identity)?;
        // One edge, directed or bidirected: the marks are part of the fingerprint.
        injective(&true, &false, |directed| *directed, |directed| {
            let mut pag = Pag::with_variables(2);
            let inserted = if *directed {
                pag.insert_directed(node(0), node(1))
            } else {
                pag.insert_bidirected(node(0), node(1))
            };
            inserted.ok().map(|()| mag_fingerprint(&pag))
        })?;
    }
}
