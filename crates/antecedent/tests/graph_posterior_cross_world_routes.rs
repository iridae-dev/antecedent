//! Mediation, path-restricted natural effects, natural direct effects and interventional
//! distributions over a graph posterior.
//!
//! Every expected value is counted from one contingency table with the formulas of the graph
//! each atom stands for (Pearl 2001 for the latent-free decomposition, the front-door formula
//! for the confounded atom), or read from the explicit cell the atom is answered by, not from
//! the mixture: the posterior moves only which structure answers, how much posterior mass
//! identifies, and how the atoms' values are weighted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, IdentificationStatus, Intervention,
    InterventionalDistributionQuery, MediationContrast, MediationQuery, NestedCounterfactualQuery,
    PathSpecificEffectQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

/// Graph-posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("graph-posterior-cross-world".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// `(t, m, y, count)` of the table.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 30),
    (0.0, 0.0, 1.0, 10),
    (0.0, 1.0, 0.0, 5),
    (0.0, 1.0, 1.0, 15),
    (1.0, 0.0, 0.0, 12),
    (1.0, 0.0, 1.0, 8),
    (1.0, 1.0, 0.0, 20),
    (1.0, 1.0, 1.0, 25),
];

fn table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (tv, mv, yv, count) in CELLS {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn count(t: Option<f64>, m: Option<f64>, y: Option<f64>) -> f64 {
    CELLS
        .iter()
        .filter(|(tv, mv, yv, _)| {
            t.is_none_or(|v| v == *tv) && m.is_none_or(|v| v == *mv) && y.is_none_or(|v| v == *yv)
        })
        .map(|(_, _, _, n)| *n as f64)
        .sum()
}

fn e_y(t: f64, m: f64) -> f64 {
    count(Some(t), Some(m), Some(1.0)) / count(Some(t), Some(m), None)
}

fn p_m(m: f64, t: f64) -> f64 {
    count(Some(t), Some(m), None) / count(Some(t), None, None)
}

fn p_t(t: f64) -> f64 {
    count(Some(t), None, None) / count(None, None, None)
}

fn counted_total() -> f64 {
    count(Some(1.0), None, Some(1.0)) / count(Some(1.0), None, None)
        - count(Some(0.0), None, Some(1.0)) / count(Some(0.0), None, None)
}

fn counted_direct() -> f64 {
    [0.0, 1.0].into_iter().map(|m| (e_y(1.0, m) - e_y(0.0, m)) * p_m(m, 0.0)).sum()
}

fn counted_indirect() -> f64 {
    [0.0, 1.0].into_iter().map(|m| e_y(0.0, m) * (p_m(m, 1.0) - p_m(m, 0.0))).sum()
}

/// The front-door interventional probability `P(y = 1 | do(t))` of `t -> m -> y` with `t <-> y`:
/// `sum_m P(m | t) sum_t' P(y = 1 | m, t') P(t')`.
fn front_door_probability(t: f64) -> f64 {
    let inner = |m: f64| -> f64 { [0.0, 1.0].into_iter().map(|u| e_y(u, m) * p_t(u)).sum() };
    [0.0, 1.0].into_iter().map(|m| p_m(m, t) * inner(m)).sum()
}

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("graph_posterior_cross_world"),
        rejected_invalid: 0,
    }
}

fn admg_of(directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(3);
    for &(from, to) in directed {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    for &(left, right) in bidirected {
        graph.insert_bidirected(node(left), node(right)).unwrap();
    }
    graph
}

/// `t -> m -> y` and `t -> y`: every component of the mediation is identified.
fn latent_free() -> Admg {
    admg_of(&[(0, 1), (0, 2), (1, 2)], &[])
}

/// `t -> m -> y` with `t <-> y` and no direct edge: only the front door identifies the total
/// effect.
fn front_door() -> Admg {
    admg_of(&[(0, 1), (1, 2)], &[(0, 2)])
}

/// An atom whose identifier refuses every component. A pair that is both directed and
/// bidirected cannot be packed into an adjacency mask, and no three-variable mixed graph that
/// packs leaves the total effect unidentified, so the refused mass is a directed cycle: a mask
/// that is no member of the posterior's graph class and so identifies nothing.
fn refused_mask() -> u64 {
    dag_mask(&[(0, 1), (1, 2), (2, 0)])
}

fn mask_of(graph: &Admg) -> u64 {
    adjacency_mask_from_admg(graph).unwrap()
}

fn dag_mask(edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |mask, &(from, to)| {
        set_edge(mask, 3, from, to, true)
    })
}

fn posterior_of(
    kind: GraphPosteriorAtomKind,
    weights: &[f64],
    masks: Vec<u64>,
) -> GraphPosterior {
    GraphPosterior::new(3, weights.to_vec(), masks, vec![0.0; 9], vec![0.0; 9], sampling())
        .unwrap()
        .with_atom_kind(kind)
}

fn admg_posterior(weights: &[f64], masks: &[u64]) -> GraphPosterior {
    posterior_of(GraphPosteriorAtomKind::Admg, weights, masks.to_vec())
}

fn mediation(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(v(0), v(2), [v(1)], contrast)
}

fn study(
    graphs: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
) -> antecedent::analysis::StudyBuilder {
    Study::tabular(table())
        .graph_posterior(graphs)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

fn value_of(result: &antecedent::StudyResult, key: u64) -> Option<f64> {
    let mixture = result.structural_response.as_ref().expect("a structural mixture");
    let atom = mixture.atoms.iter().find(|atom| atom.graph_key == key).expect("atom present");
    match atom.value {
        Some(ResponseValue::Scalar(value)) => Some(value),
        Some(_) => panic!("a scalar effect atom value is a scalar"),
        None => None,
    }
}

/// The weighted mean of the atoms' own values, computed from the mixture's atoms.
fn mean_of_atoms(result: &antecedent::StudyResult) -> f64 {
    let mixture = result.structural_response.as_ref().expect("a structural mixture");
    let (mut weighted, mut mass) = (0.0, 0.0);
    for atom in &mixture.atoms {
        if let Some(ResponseValue::Scalar(value)) = atom.value {
            weighted += atom.weight * value;
            mass += atom.weight;
        }
    }
    weighted / mass
}

#[test]
fn each_mediation_component_is_mixed_over_the_atoms_that_identify_it() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_001);
        let graphs = admg_posterior(&[0.5, 0.3, 0.2], &[mask_of(&latent_free()), mask_of(&front_door()), refused_mask()]);
        let keys = graphs.graph_keys.clone();
        let query = CausalQuery::Mediation(mediation(MediationContrast::Total));
        let built = study(graphs, query, InferenceMode::Frequentist).build().unwrap();
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap();

        // The latent-free atom decomposes by the counted formulas, the front-door atom
        // identifies the total effect by the front-door formula, and the confounded atom
        // identifies nothing: its mass is unidentified and is not redistributed.
        let front = front_door_probability(1.0) - front_door_probability(0.0);
        let expected = (0.5 * counted_total() + 0.3 * front) / 0.8;
        assert!((result.effect() - expected).abs() < 1e-12, "{} vs {expected}", result.effect());
        assert!((value_of(&result, keys[0]).unwrap() - counted_total()).abs() < 1e-12);
        assert!((value_of(&result, keys[1]).unwrap() - front).abs() < 1e-12);
        assert!(value_of(&result, keys[2]).is_none());
        let mixture = result.structural_response.as_ref().unwrap();
        assert_eq!(mixture.atoms.len(), 3);
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9, "{}", mixture.unidentified_mass);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("functional.effect"),
            "a mixed-graph posterior is evaluated by the finite-discrete plug-in"
        );

        // The decomposition beside the requested contrast carries the requested value, and the
        // other components are mixed over the atoms that identify them.
        let components = result.payload.mediation().expect("the decomposition");
        assert!((components.total.unwrap() - result.effect()).abs() < 1e-12);
        let direct = components.direct.expect("the latent-free atom identifies the direct effect");
        let indirect =
            components.mediated.expect("the latent-free atom identifies the indirect effect");
        assert!(direct.is_finite() && indirect.is_finite());
        let disclosure = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "estimate.graph_posterior.cross_world")
            .expect("the per-component accounting is disclosed");
        assert!(disclosure.message.contains("total: 2 atom(s)"), "{}", disclosure.message);
    });
}

#[test]
fn the_requested_direct_component_is_the_posterior_weighted_mean_of_the_atoms_values() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_002);
        let graphs = admg_posterior(&[0.6, 0.4], &[mask_of(&latent_free()), refused_mask()]);
        let keys = graphs.graph_keys.clone();
        let query = CausalQuery::Mediation(mediation(MediationContrast::NaturalDirect));
        let result =
            study(graphs, query, InferenceMode::Frequentist).build().unwrap().run(&ctx).unwrap();
        assert!((value_of(&result, keys[0]).unwrap() - counted_direct()).abs() < 1e-12);
        // Whatever the confounded atom decides, the published value is the weighted mean of
        // the values of the atoms that identify, and the masses account for the whole posterior.
        assert!((result.effect() - mean_of_atoms(&result)).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().unwrap();
        let accounted =
            mixture.identified_mass + mixture.unidentified_mass + mixture.unevaluable_mass;
        assert!((accounted - 1.0).abs() < 1e-9, "{accounted}");
        assert!(mixture.identified_mass >= 0.6 - 1e-9);
    });
}

#[test]
fn a_posterior_with_no_atom_that_identifies_the_requested_effect_is_not_identified() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_003);
        let graphs = admg_posterior(&[1.0], &[refused_mask()]);
        let query = CausalQuery::Mediation(mediation(MediationContrast::Total));
        let refusal = study(graphs, query, InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("effect_not_identified"), "{refusal}");
    });
}

#[test]
fn a_cpdag_atom_is_answered_by_its_completion_envelope() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_004);
        let mut dag = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            dag.insert_directed(node(a), node(b)).unwrap();
        }
        let mask = adjacency_mask_from_cpdag(&Cpdag::from_dag(&dag)).unwrap();
        // The same class sampled twice is one atom whose mass is the sum of its samples.
        let graphs = posterior_of(GraphPosteriorAtomKind::Cpdag, &[0.6, 0.4], vec![mask, mask]);
        for (contrast, expected) in [
            (MediationContrast::Total, counted_total()),
            (MediationContrast::NaturalDirect, counted_direct()),
            (MediationContrast::NaturalIndirect, counted_indirect()),
        ] {
            let result = study(
                graphs.clone(),
                CausalQuery::Mediation(mediation(contrast)),
                InferenceMode::Frequentist,
            )
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
            assert!(
                (result.effect() - expected).abs() < 1e-12,
                "{contrast:?}: {}",
                result.effect()
            );
            let mixture = result.structural_response.as_ref().unwrap();
            assert_eq!(mixture.atoms.len(), 1, "one class, however often it was sampled");
            assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
            let components = result.payload.mediation().unwrap();
            assert!((components.total.unwrap() - counted_total()).abs() < 1e-12);
            assert!((components.direct.unwrap() - counted_direct()).abs() < 1e-12);
            assert!((components.mediated.unwrap() - counted_indirect()).abs() < 1e-12);
        }
    });
}

#[test]
fn a_dag_atom_runs_the_dag_cells_own_estimator_and_a_non_member_mask_is_unidentified_mass() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_005);
        let member = dag_mask(&[(0, 1), (0, 2), (1, 2)]);
        // `t -> m -> y -> t`: not a DAG.
        let cyclic = dag_mask(&[(0, 1), (1, 2), (2, 0)]);
        let graphs = posterior_of(GraphPosteriorAtomKind::Dag, &[0.7, 0.3], vec![member, cyclic]);
        let keys = graphs.graph_keys.clone();
        let query = CausalQuery::Mediation(mediation(MediationContrast::NaturalDirect));
        let result = study(graphs, query.clone(), InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();

        // The explicit Dag cell on the same table.
        let mut dag = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            dag.insert_directed(node(a), node(b)).unwrap();
        }
        let explicit = Study::tabular(table())
            .graph(dag)
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!((result.effect() - explicit.effect()).abs() < 1e-12);
        let (theirs, ours) =
            (explicit.payload.mediation().unwrap(), result.payload.mediation().unwrap());
        assert!((ours.total.unwrap() - theirs.total.unwrap()).abs() < 1e-12);
        assert!((ours.mediated.unwrap() - theirs.mediated.unwrap()).abs() < 1e-12);
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("mediation.linear"));

        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.7).abs() < 1e-9);
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9);
        assert!(value_of(&result, keys[1]).is_none());
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    });
}

#[test]
fn a_natural_direct_effect_over_dag_and_admg_atoms_is_the_counted_direct_effect() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_006);
        let nested = CausalQuery::NestedCounterfactual(
            NestedCounterfactualQuery::new(v(0), v(1), v(2)).unwrap(),
        );

        // A latent-free DAG atom beside a cyclic mask.
        let member = dag_mask(&[(0, 1), (0, 2), (1, 2)]);
        let cyclic = dag_mask(&[(0, 1), (1, 2), (2, 0)]);
        let dag_atoms =
            posterior_of(GraphPosteriorAtomKind::Dag, &[0.7, 0.3], vec![member, cyclic]);
        let result = study(dag_atoms, nested.clone(), InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!((result.effect() - counted_direct()).abs() < 1e-12, "{}", result.effect());
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9);

        // An Admg atom beside a confounded one.
        let graphs = admg_posterior(&[0.6, 0.4], &[mask_of(&latent_free()), refused_mask()]);
        let keys = graphs.graph_keys.clone();
        let result =
            study(graphs, nested, InferenceMode::Frequentist).build().unwrap().run(&ctx).unwrap();
        assert!((value_of(&result, keys[0]).unwrap() - counted_direct()).abs() < 1e-12);
        assert!((result.effect() - mean_of_atoms(&result)).abs() < 1e-12);
    });
}

#[test]
fn a_path_restricted_effect_over_a_posterior_equals_the_explicit_cell_per_atom() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_007);
        let path = PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]);
        let query = CausalQuery::PathSpecific(path);
        let graphs = admg_posterior(&[0.75, 0.25], &[mask_of(&latent_free()), refused_mask()]);
        let keys = graphs.graph_keys.clone();
        let result = study(graphs, query.clone(), InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let explicit = Study::tabular(table())
            .graph(latent_free())
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!((value_of(&result, keys[0]).unwrap() - explicit.effect()).abs() < 1e-12);
        assert!((result.effect() - mean_of_atoms(&result)).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().unwrap();
        let accounted =
            mixture.identified_mass + mixture.unidentified_mass + mixture.unevaluable_mass;
        assert!((accounted - 1.0).abs() < 1e-9, "{accounted}");
    });
}

#[test]
fn an_interventional_distribution_is_the_posterior_weighted_mean_probability_of_each_level() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_008);
        let graphs = admg_posterior(&[0.5, 0.3, 0.2], &[mask_of(&latent_free()), mask_of(&front_door()), refused_mask()]);
        let keys = graphs.graph_keys.clone();
        let query = CausalQuery::Distribution(InterventionalDistributionQuery::new(
            v(2),
            [Intervention::set(v(0), Value::f64(1.0))],
        ));
        let result =
            study(graphs, query, InferenceMode::Frequentist).build().unwrap().run(&ctx).unwrap();

        // Without confounding `P(y = 1 | do(t = 1))` is the observed `P(y = 1 | t = 1)`; the
        // front-door atom has its own closed form; the confounded atom identifies nothing.
        let free = count(Some(1.0), None, Some(1.0)) / count(Some(1.0), None, None);
        let front = front_door_probability(1.0);
        let expected = (0.5 * free + 0.3 * front) / 0.8;
        let distribution = result.payload.distribution().expect("a distribution");
        let probability = |level: f64| {
            distribution
                .atoms
                .iter()
                .find(|atom| atom.outcomes[0].1 == Value::f64(level))
                .map(|atom| atom.probability)
                .expect("an outcome level")
        };
        assert!((probability(1.0) - expected).abs() < 1e-12, "{}", probability(1.0));
        assert!((probability(0.0) - (1.0 - expected)).abs() < 1e-12);
        assert!((distribution.mean - expected).abs() < 1e-12);
        assert!((value_of(&result, keys[0]).unwrap() - free).abs() < 1e-12);
        assert!((value_of(&result, keys[1]).unwrap() - front).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9);
    });
}

/// A conditional distribution has one law per conditioning assignment and so no scalar mean:
/// the atoms still mix level by level, and the mass the identifier refuses stays beside the mass
/// that is identified and evaluated, not lost as unevaluable mass.
#[test]
fn a_distribution_without_a_mean_is_mixed_level_by_level_with_its_mass_kept_apart() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_013);
        let query = CausalQuery::Distribution(
            InterventionalDistributionQuery::new(v(2), [Intervention::set(v(0), Value::f64(1.0))])
                .with_conditioning([v(1)]),
        );
        let member = dag_mask(&[(0, 1), (0, 2), (1, 2)]);
        let cyclic = dag_mask(&[(0, 1), (1, 2), (2, 0)]);
        let graphs = posterior_of(GraphPosteriorAtomKind::Dag, &[0.7, 0.3], vec![member, cyclic]);
        let result = study(graphs, query.clone(), InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let mut dag = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            dag.insert_directed(node(a), node(b)).unwrap();
        }
        let explicit = Study::tabular(table())
            .graph(dag)
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let (ours, theirs) = (
            result.payload.distribution().expect("a distribution"),
            explicit.payload.distribution().expect("a distribution"),
        );
        assert!(ours.mean.is_nan() && theirs.mean.is_nan(), "one law per conditioning assignment");
        assert_eq!(ours.atoms.len(), theirs.atoms.len());
        for (mixed, own) in ours.atoms.iter().zip(theirs.atoms.iter()) {
            assert_eq!(mixed.outcomes, own.outcomes);
            assert!((mixed.probability - own.probability).abs() < 1e-12);
        }
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.7).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9);
        assert!(mixture.unevaluable_mass.abs() < 1e-12, "{}", mixture.unevaluable_mass);
        assert!(result.effect().is_nan());
    });
}

#[test]
fn a_bayesian_mixture_keeps_one_atoms_posterior_and_pools_several() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_009);
        let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400));
        let query = CausalQuery::Mediation(mediation(MediationContrast::NaturalDirect));

        // One atom that covers the whole posterior keeps its own Bayesian-bootstrap posterior.
        let single =
            study(admg_posterior(&[1.0], &[mask_of(&latent_free())]), query.clone(), inference.clone())
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap();
        let posterior = single.posterior.as_ref().expect("the atom's own posterior");
        assert_eq!(posterior.draws.n_draws, 400);
        assert!((single.effect() - counted_direct()).abs() < 0.05, "{}", single.effect());

        // With posterior mass the identifier refuses, the aggregate is conditional on the
        // atoms that identify; the one contributing atom carries its own posterior, with the
        // mass the identifier refused stated on it.
        let lost = study(
            admg_posterior(&[0.7, 0.3], &[mask_of(&latent_free()), refused_mask()]),
            CausalQuery::Mediation(mediation(MediationContrast::Total)),
            inference.clone(),
        )
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
        let posterior = lost.posterior.as_ref().expect("the contributing atom's posterior");
        assert!(
            (posterior.unidentified_mass - 0.3).abs() < 1e-9,
            "{}",
            posterior.unidentified_mass
        );
        assert_eq!(lost.identification.status, IdentificationStatus::GraphDependent);
        assert!((lost.effect() - counted_total()).abs() < 0.05, "{}", lost.effect());

        // Several contributing atoms run in one context, so draw `d` of each is the same draw of
        // the row weights and the pooled posterior is the draw-wise weighted mixture.
        let pooled = study(
            admg_posterior(&[0.5, 0.5], &[mask_of(&latent_free()), mask_of(&front_door())]),
            CausalQuery::Mediation(mediation(MediationContrast::Total)),
            inference,
        )
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
        let posterior = pooled.posterior.as_ref().expect("a pooled posterior");
        assert_eq!(posterior.draws.n_draws, 400);
        let front = front_door_probability(1.0) - front_door_probability(0.0);
        let expected = 0.5 * (counted_total() + front);
        assert!((pooled.effect() - expected).abs() < 0.05, "{} vs {expected}", pooled.effect());
        assert!((posterior.summaries.mean[0] - pooled.effect()).abs() < 1e-9);
    });
}

/// A Frequentist mixture of several atoms re-runs every atom on shared resampled tables and reads
/// the spread of the draw-wise weighted mixture.
#[test]
fn a_frequentist_mixture_of_several_atoms_has_the_standard_error_of_its_shared_replicates() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_012);
        let query = CausalQuery::Mediation(mediation(MediationContrast::Total));
        let run = |graphs: GraphPosterior| {
            Study::tabular(table())
                .graph_posterior(graphs)
                .query(query.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(40)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap()
        };
        let mixture = run(admg_posterior(&[0.5, 0.5], &[mask_of(&latent_free()), mask_of(&front_door())]));
        let effect = mixture.estimate.as_effect().unwrap();
        let se = effect.se_bootstrap().expect("the spread of the shared replicates");
        assert!(se > 0.0 && se < 0.5, "{se}");
        let accounting = &effect.uncertainty.bootstrap;
        assert!(accounting.replicates_ok.unwrap() >= 30, "{accounting:?}");
        assert!(
            mixture
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.graph_posterior.cross_world_uncertainty")
        );
        // One atom carries its own bootstrap interval, which the mixture of a single atom keeps.
        let single = run(admg_posterior(&[1.0], &[mask_of(&latent_free())]));
        let own = single.estimate.as_effect().unwrap().se_bootstrap().expect("a bootstrap SE");
        assert!(own > 0.0 && own < 0.5, "{own}");
    });
}

#[test]
fn a_prepared_handle_answers_the_same_mixture() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_010);
        let graphs = admg_posterior(&[0.5, 0.3, 0.2], &[mask_of(&latent_free()), mask_of(&front_door()), refused_mask()]);
        let built = study(
            graphs,
            CausalQuery::Mediation(mediation(MediationContrast::Total)),
            InferenceMode::Frequentist,
        )
        .build()
        .unwrap();
        let fresh = built.run(&ctx).unwrap();
        let mut prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate(&table(), &ctx).unwrap();
        assert!((again.effect() - fresh.effect()).abs() < 1e-12);
        let mixture = again.structural_response.as_ref().unwrap();
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9);
    });
}

/// Cheap and full validation run each atom's own suite against that atom's own estimate and
/// mix the reports by posterior weight; validation observes the mixture and does not move it.
#[test]
fn each_atom_runs_its_own_suite_and_the_reports_mix_by_posterior_weight() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_011);
        let query = CausalQuery::Mediation(mediation(MediationContrast::Total));
        let run = |suite: RefuteSuite| {
            Study::tabular(table())
                .graph_posterior(admg_posterior(&[0.7, 0.3], &[mask_of(&latent_free()), refused_mask()]))
                .query(query.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap_or_else(|e| panic!("{suite:?}: {e}"))
                .run(&ctx)
                .unwrap_or_else(|e| panic!("{suite:?}: {e}"))
        };
        let none = run(RefuteSuite::None);
        assert!(none.refutations.is_empty());
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = run(suite);
            assert!((result.effect() - none.effect()).abs() < 1e-12);
            let found: Vec<&str> =
                result.refutations.iter().map(|report| report.refuter.as_ref()).collect();
            assert!(found.contains(&"mediation.sequential_ignorability"), "{found:?}");
            assert!(found.contains(&"mediation.joint_stability_subset"), "{found:?}");
            assert_eq!(
                found.contains(&"mediation.joint_bootstrap_range"),
                suite == RefuteSuite::Full
            );
            let mixture = result.structural_response.as_ref().unwrap();
            assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9);
        }
    });
}
