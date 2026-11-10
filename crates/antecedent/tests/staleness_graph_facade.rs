//! Staleness of a prepared study whose graph implies conditional independencies: the study
//! lists what its graph implies for every static graph class, an unmeasured common cause that
//! moves a measured column asks to identify again, and a graph that joins every pair says the
//! data cannot contradict it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::staleness::{
    FindingKind, GraphImplications, ImpliedIndependence, Recommendation, StalenessPolicy,
    assess_artifact_staleness,
};
use antecedent::{EstimatorId, IntoGraphInput, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag};
use antecedent_kernels::{sigmoid, standard_normal};

const EDGES: [(u32, u32); 6] = [(2, 0), (3, 0), (2, 4), (2, 1), (0, 1), (4, 1)];

/// `(t, y, z, w, s)`: `z` and `w` cause a binary `t`, `z` causes `s`, and `y` depends on `t`,
/// `s` and `z`. With `hidden`, an unmeasured cause raises `t`, `s` and `y` together.
fn columns(n: usize, seed: u64, hidden: bool) -> Vec<(String, Vec<f64>)> {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z, mut w, mut s) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let wi = standard_normal(&mut rng);
        let u = if hidden { standard_normal(&mut rng) } else { 0.0 };
        let ti = if rng.next_f64() < sigmoid(0.5 * zi + 0.5 * wi + u) { 1.0 } else { 0.0 };
        let si = 0.5 * zi + 0.8 * u + standard_normal(&mut rng);
        z.push(zi);
        w.push(wi);
        t.push(ti);
        s.push(si);
        y.push(1.0 + 2.0 * ti + zi + 0.5 * si + 0.8 * u + standard_normal(&mut rng));
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z), ("w".into(), w), ("s".into(), s)]
}

fn table(columns: &[(String, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(n, v)| (n.as_str(), v.as_slice()))).unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

fn dag() -> Dag {
    let mut dag = Dag::with_variables(5);
    for (from, to) in EDGES {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    dag
}

fn prepare(data: TabularData, graph: impl IntoGraphInput) -> PreparedStudy {
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
}

fn statement(x: &str, y: &str, given: &[&str]) -> ImpliedIndependence {
    ImpliedIndependence {
        x: x.into(),
        y: y.into(),
        given: given.iter().map(|g| (*g).to_owned()).collect(),
    }
}

/// What the graph implies, one statement per pair it separates, in node order.
fn expected() -> GraphImplications {
    GraphImplications::Listed(vec![
        statement("t", "s", &["z"]),
        statement("y", "w", &["t", "z"]),
        statement("z", "w", &[]),
        statement("w", "s", &[]),
    ])
}

#[test]
fn a_dag_and_an_admg_study_list_what_the_graph_implies() {
    let data = table(&columns(400, 1, false));
    assert_eq!(prepare(data.clone(), dag()).staleness_plan().implications, Some(expected()));
    let admg = Admg::from_dag(&dag()).unwrap();
    assert_eq!(prepare(data.clone(), admg).staleness_plan().implications, Some(expected()));
    // An unmeasured common cause of t and s joins them, so that statement is no longer implied.
    let mut confounded = Admg::from_dag(&dag()).unwrap();
    confounded.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(4)).unwrap();
    let GraphImplications::Listed(listed) =
        GraphImplications::from_admg(&confounded, &|node| match node {
            antecedent_core::NodeRef::Static(v) => {
                Some(["t", "y", "z", "w", "s"][v.raw() as usize].into())
            }
            _ => None,
        })
    else {
        panic!("an ADMG over measured columns is listed");
    };
    assert!(listed.iter().all(|i| !(i.x == "t" && i.y == "s")), "{listed:?}");
    assert!(listed.contains(&statement("z", "w", &[])));
}

#[test]
fn a_class_of_graphs_lists_what_its_members_share() {
    let name = |node: &antecedent_core::NodeRef| match node {
        antecedent_core::NodeRef::Static(v) => {
            Some(["t", "y", "z", "w", "s"][v.raw() as usize].to_owned())
        }
        _ => None,
    };
    // The CPDAG of the DAG stands for its Markov equivalence class: the same separations.
    let cpdag = Cpdag::from_dag(&dag());
    let GraphImplications::Listed(from_class) = GraphImplications::from_cpdag(&cpdag, &name) else {
        panic!("the class is listed");
    };
    let GraphImplications::Listed(from_dag) = GraphImplications::from_dag(&dag(), &name) else {
        panic!("the DAG is listed");
    };
    let pairs = |items: &[ImpliedIndependence]| {
        let mut pairs: Vec<_> = items.iter().map(|i| (i.x.clone(), i.y.clone())).collect();
        pairs.sort();
        pairs
    };
    assert_eq!(pairs(&from_class), pairs(&from_dag));
    // A PAG of the same adjacencies with every mark open keeps the separations of the pairs
    // every ancestral completion separates.
    let mut pag = Pag::with_variables(5);
    for (from, to) in EDGES {
        pag.insert_circle_circle(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    match GraphImplications::from_pag(&pag, &name) {
        GraphImplications::Listed(items) => {
            for item in &items {
                assert!(pairs(&from_dag).contains(&(item.x.clone(), item.y.clone())), "{item:?}");
            }
        }
        GraphImplications::Unavailable(reason) => {
            assert!(reason.contains("completes the PAG"), "{reason}");
        }
    }
}

#[test]
fn an_unmeasured_cause_that_moves_a_measured_column_asks_to_reidentify() {
    let policy = StalenessPolicy::default();
    let study = prepare(table(&columns(4000, 1, false)), dag());
    let unchanged = study.staleness(&table(&columns(4000, 3, false)), &policy, &ctx()).unwrap();
    assert!(
        unchanged.findings().all(|f| f.kind != FindingKind::ImpliedIndependence),
        "{:#?}",
        unchanged.findings().collect::<Vec<_>>()
    );
    let changed_data = table(&columns(4000, 2, true));
    let changed = study.staleness(&changed_data, &policy, &ctx()).unwrap();
    assert_eq!(changed.recommendation, Recommendation::Reidentify);
    let failed = changed
        .findings()
        .find(|f| f.kind == FindingKind::ImpliedIndependence && f.subject == "t _||_ s | z")
        .expect("the independence the unmeasured cause breaks is a finding");
    assert!(failed.statistic.abs() >= policy.implication_threshold);
    assert!(changed.invalidated_labels().iter().any(|stage| stage == "identification"));

    // The exported baseline carries the statements and gives the same report.
    let bytes = study.staleness_baseline(&policy, &ctx()).unwrap().export().unwrap();
    let consumed = assess_artifact_staleness(&bytes, &changed_data, &policy, &ctx()).unwrap();
    assert_eq!(consumed.identity, changed.identity);
}

#[test]
fn a_graph_that_joins_every_pair_says_the_data_cannot_contradict_it() {
    let keep = |columns: Vec<(String, Vec<f64>)>| -> TabularData {
        table(&columns.into_iter().take(3).collect::<Vec<_>>())
    };
    let mut complete = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        complete.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let study = prepare(keep(columns(4000, 1, false)), complete);
    assert_eq!(study.staleness_plan().implications, Some(GraphImplications::Listed(Vec::new())));
    let report = study
        .staleness(&keep(columns(4000, 3, false)), &StalenessPolicy::default(), &ctx())
        .unwrap();
    let (_, reason) = report
        .not_evaluated
        .iter()
        .find(|(what, _)| what == "implied_independence")
        .expect("the untested diagnostic is named");
    assert!(reason.contains("implies no conditional independence"), "{reason}");
}
