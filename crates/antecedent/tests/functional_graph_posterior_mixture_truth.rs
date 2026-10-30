//! A path-restricted natural effect, an interventional distribution and a mediation decomposition
//! over a graph posterior of Dag, ADMG, temporal-ADMG, CPDAG and PAG atoms, on both inference
//! sides and under every validation suite, against the posterior-weighted mean of each atom's
//! own identified functional (`conformance/estimate/functional_graph_posterior_mixtures`).
//!
//! An atom the identifier refuses is unidentified posterior mass; it is not redistributed, and
//! the status resolves to graph-dependent. Each atom runs its own refuter suite against its own
//! estimate and the reports mix by posterior weight.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "test fixtures: one coordinate walk per test"
)]
use antecedent::analysis::RetainedOperation;
use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, IdentificationStatus, Intervention,
    InterventionalDistributionQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery,
    Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value as Json;

fn pin() -> Json {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/functional_graph_posterior_mixtures/expected.json"
    ))
    .unwrap()
}

fn columns(table: &Json) -> Vec<String> {
    table["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_owned()).collect()
}

/// The table's rows in a fixed interleaved order (so that consecutive rows, the blocks a
/// temporal-ADMG draw weights together, do not follow the sorted cells), each cell repeated
/// `scale` times its count: the same law on more rows.
fn expand(table: &Json, scale: u64) -> TabularData {
    let names = columns(table);
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for cell in table["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap() * scale).unwrap();
        for (index, name) in names.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name.as_str()].as_f64().unwrap(), count));
        }
    }
    let rows = values[0].len();
    let order: Vec<usize> = (0..rows).map(|row| (row * 97) % rows).collect();
    assert_eq!(order.iter().collect::<std::collections::BTreeSet<_>>().len(), rows);
    let shuffled: Vec<Vec<f64>> =
        values.iter().map(|column| order.iter().map(|&row| column[row]).collect()).collect();
    let pairs: Vec<(&str, &[f64])> =
        names.iter().zip(&shuffled).map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn index_of(names: &[String], name: &str) -> u32 {
    u32::try_from(names.iter().position(|n| n == name).unwrap()).unwrap()
}

fn edges(names: &[String], list: &Json) -> Vec<(u32, u32)> {
    list.as_array()
        .map(|list| {
            list.iter()
                .map(|e| {
                    (
                        index_of(names, e[0].as_str().unwrap()),
                        index_of(names, e[1].as_str().unwrap()),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn weight(atom: &Json) -> f64 {
    let text = atom["weight"].as_str().unwrap();
    let (n, d) = text.split_once('/').unwrap();
    n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap()
}

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("x"),
        rejected_invalid: 0,
    }
}

fn atom_kind(class: &str) -> GraphPosteriorAtomKind {
    match class {
        "dag" => GraphPosteriorAtomKind::Dag,
        "admg" => GraphPosteriorAtomKind::Admg,
        "temporal_admg" => GraphPosteriorAtomKind::TemporalAdmg,
        "cpdag" => GraphPosteriorAtomKind::Cpdag,
        "pag" => GraphPosteriorAtomKind::Pag,
        other => panic!("unexpected class {other}"),
    }
}

fn mask(class: &str, names: &[String], graph: &Json) -> u64 {
    let n = names.len();
    let nodes = u32::try_from(n).unwrap();
    match class {
        "dag" | "pag" => edges(names, &graph["directed"])
            .iter()
            .fold(0_u64, |m, &(a, b)| set_edge(m, n, a as usize, b as usize, true)),
        "admg" | "temporal_admg" => {
            let mut g = Admg::with_variables(nodes);
            for (a, b) in edges(names, &graph["directed"]) {
                g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            for (a, b) in edges(names, &graph["bidirected"]) {
                g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            adjacency_mask_from_admg(&g).unwrap()
        }
        "cpdag" => {
            let mut g = Cpdag::with_variables(nodes);
            for (a, b) in edges(names, &graph["directed"]) {
                g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            for (a, b) in edges(names, &graph["undirected"]) {
                g.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            adjacency_mask_from_cpdag(&g).unwrap()
        }
        other => panic!("unexpected class {other}"),
    }
}

fn graph_posterior(class: &str, names: &[String], atoms: &[Json]) -> GraphPosterior {
    let n = names.len();
    let masks: Vec<u64> = atoms.iter().map(|a| mask(class, names, &a["graph"])).collect();
    let weights: Vec<f64> = atoms.iter().map(weight).collect();
    let kind = atom_kind(class);
    let mut g = GraphPosterior::new(
        n,
        weights,
        masks.clone(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling(),
    )
    .unwrap()
    .with_atom_kind(kind);
    if kind == GraphPosteriorAtomKind::Pag {
        g = g.with_mark_masks(vec![0; masks.len()]).unwrap();
    }
    g
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    PathSpecific,
    Distribution,
    Mediation,
}

impl Family {
    fn key(self) -> &'static str {
        match self {
            Self::PathSpecific => "path_specific",
            Self::Distribution => "distribution",
            Self::Mediation => "mediation",
        }
    }
}

/// What the fixture says one posterior's mixture is, per component.
struct Expected {
    /// `(component, value)`: `effect` for a path-specific effect, `y0` and `y1` for a
    /// distribution, `total`, `direct` and `indirect` for a mediation.
    components: Vec<(String, f64)>,
    unidentified: f64,
    set: (f64, f64),
}

struct Posterior {
    family: Family,
    class: &'static str,
    graphs: GraphPosterior,
    data: TabularData,
    query: CausalQuery,
    expected: Expected,
    /// The table's variable names and the fixture's atoms, to re-run one atom alone.
    names: Vec<String>,
    atoms: Vec<Json>,
    /// The conjugate posterior mean band of the declared estimator.
    bayesian_band: f64,
}

fn query_for(
    family: Family,
    names: &[String],
    pin: &Json,
    contrast: MediationContrast,
) -> CausalQuery {
    let v = |name: &str| VariableId::from_raw(index_of(names, name));
    let q = &pin["queries"][family.key()];
    match family {
        Family::PathSpecific => CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(
                v(q["treatment"].as_str().unwrap()),
                v(q["outcome"].as_str().unwrap()),
            )
            .with_path_nodes(
                q["path_nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| v(n.as_str().unwrap()))
                    .collect::<Vec<_>>(),
            )
            .with_max_len(4),
        ),
        Family::Distribution => CausalQuery::Distribution(InterventionalDistributionQuery::new(
            v(q["outcome"].as_str().unwrap()),
            [Intervention::set(v("t"), Value::f64(q["do"]["t"].as_f64().unwrap()))],
        )),
        Family::Mediation => CausalQuery::Mediation(MediationQuery::binary(
            v(q["treatment"].as_str().unwrap()),
            v(q["outcome"].as_str().unwrap()),
            q["mediators"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| v(n.as_str().unwrap()))
                .collect::<Vec<_>>(),
            contrast,
        )),
    }
}

fn expected_of(family: Family, entry: &Json) -> Expected {
    let mixture = &entry["mixture"];
    let (components, unidentified, set) = match family {
        Family::PathSpecific => (
            vec![("effect".to_owned(), mixture["value"].as_f64().unwrap())],
            mixture["unidentified_mass"].as_f64().unwrap(),
            (mixture["set"][0].as_f64().unwrap(), mixture["set"][1].as_f64().unwrap()),
        ),
        Family::Distribution | Family::Mediation => {
            let keys: &[&str] = if family == Family::Distribution {
                &["y0", "y1"]
            } else {
                &["total", "direct", "indirect"]
            };
            let pick = if family == Family::Distribution { "y1" } else { "direct" };
            (
                keys.iter()
                    .map(|k| ((*k).to_owned(), mixture[*k]["value"].as_f64().unwrap()))
                    .collect(),
                mixture[pick]["unidentified_mass"].as_f64().unwrap(),
                (
                    mixture[pick]["set"][0].as_f64().unwrap(),
                    mixture[pick]["set"][1].as_f64().unwrap(),
                ),
            )
        }
    };
    Expected { components, unidentified, set }
}

fn posterior(family: Family, class: &'static str, scale: u64) -> Posterior {
    let pin = pin();
    let fixture_class = if class == "temporal_admg" { "admg" } else { class };
    let entry = &pin["posteriors"][family.key()][fixture_class];
    let table = &pin["tables"][entry["table"].as_str().unwrap()];
    let names = columns(table);
    let atoms = entry["atoms"].as_array().unwrap();
    let linear = entry["table"] == "linear_mediation";
    Posterior {
        family,
        class,
        graphs: graph_posterior(class, &names, atoms),
        atoms: atoms.clone(),
        data: expand(table, if linear { 1 } else { scale }),
        query: query_for(family, &names, &pin, MediationContrast::NaturalDirect),
        names,
        expected: expected_of(family, entry),
        bayesian_band: if linear {
            pin["bayesian"]["linear_mean_tolerance"].as_f64().unwrap()
        } else {
            pin["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap()
        },
    }
}

const CLASSES: [&str; 5] = ["dag", "admg", "temporal_admg", "cpdag", "pag"];

fn posteriors(family: Family, scale: u64) -> Vec<Posterior> {
    CLASSES.iter().map(|class| posterior(family, class, scale)).collect()
}

fn inference(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).prior_scale(10.0))
    } else {
        InferenceMode::Frequentist
    }
}

fn builder(
    p: &Posterior,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::analysis::StudyBuilder {
    Study::tabular(p.data.clone())
        .graph_posterior(p.graphs.clone())
        .query(p.query.clone())
        .inference(inference(bayesian))
        .refute(suite)
        .bootstrap_replicates(0)
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().unwrap();
}

/// The mixed value of one component, read from the result.
fn component(p: &Posterior, result: &antecedent::StudyResult, name: &str) -> f64 {
    match p.family {
        Family::PathSpecific => result.effect(),
        Family::Distribution => {
            let level = if name == "y1" { 1.0 } else { 0.0 };
            result
                .distribution()
                .expect("the mixed distribution")
                .atoms
                .iter()
                .find(|atom| atom.outcomes.iter().any(|(_, value)| *value == Value::f64(level)))
                .map_or(0.0, |atom| atom.probability)
        }
        Family::Mediation => {
            let parts = result.payload().mediation().expect("the mixed decomposition");
            match name {
                "total" => parts.total,
                "direct" => parts.direct,
                _ => parts.mediated,
            }
            .unwrap_or_else(|| panic!("{} component {name} absent", p.class))
        }
    }
}

fn assert_values(label: &str, p: &Posterior, result: &antecedent::StudyResult, bayesian: bool) {
    let tolerance = if bayesian { p.bayesian_band } else { 1e-9 };
    for (name, want) in &p.expected.components {
        let got = component(p, result, name);
        assert!((got - want).abs() <= tolerance, "{label}: {name} {got} vs {want}");
    }
}

fn assert_mass_and_set(
    label: &str,
    p: &Posterior,
    result: &antecedent::StudyResult,
    tolerance: f64,
) {
    let mixture =
        result.structural_response.as_ref().unwrap_or_else(|| panic!("{label}: no mixture"));
    assert!(
        (mixture.unidentified_mass + mixture.unevaluable_mass - p.expected.unidentified).abs()
            < 1e-9,
        "{label}: unidentified {} unevaluable {} vs {}",
        mixture.unidentified_mass,
        mixture.unevaluable_mass,
        p.expected.unidentified
    );
    if p.expected.unidentified > 0.0 {
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent, "{label}");
    }
    if let Some(set) = mixture.identified_set.as_ref() {
        if p.family != Family::Distribution {
            assert!(
                (set.lower[0] - p.expected.set.0).abs() <= tolerance,
                "{label}: set {:?}",
                set.lower
            );
            assert!(
                (set.upper[0] - p.expected.set.1).abs() <= tolerance,
                "{label}: set {:?}",
                set.upper
            );
        }
    } else {
        assert!(p.class != "cpdag", "{label}: a class posterior must publish its identified set");
    }
}

fn run_family(family: Family) {
    on_large_stack(move || {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                // The validation states resample the table, so they run on the same law at 40
                // times the rows.
                let scale = if suite == RefuteSuite::None { 1 } else { 40 };
                for p in posteriors(family, scale) {
                    let label = format!("{family:?} {} bayesian={bayesian} {suite:?}", p.class);
                    let study = builder(&p, bayesian, suite)
                        .build()
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed), "{label}");
                    let result = study
                        .run(&ExecutionContext::for_tests(3))
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert_values(&label, &p, &result, bayesian);
                    assert_mass_and_set(
                        &label,
                        &p,
                        &result,
                        if bayesian { p.bayesian_band } else { 1e-9 },
                    );
                    if suite == RefuteSuite::None {
                        assert!(result.refutations.is_empty(), "{label}");
                        continue;
                    }
                    assert!(!result.refutations.is_empty(), "{label}: no refuter ran");
                    assert_mixed_verdicts(&label, &p, &result, bayesian, suite);
                }
            }
        }
    });
}

/// A mixed verdict passes only if every contributing atom passes: each refuter of the mixture
/// is compared with the conjunction of the verdicts the same refuter gives when each identified
/// atom runs alone as a one-atom posterior with the same suite.
fn assert_mixed_verdicts(
    label: &str,
    p: &Posterior,
    result: &antecedent::StudyResult,
    bayesian: bool,
    suite: RefuteSuite,
) {
    let mut alone: Vec<Vec<(String, bool, String)>> = Vec::new();
    for atom in p.atoms.iter().filter(|a| !a["value"].is_null()) {
        let mut only = atom.clone();
        only["weight"] = Json::from("1/1");
        let graphs = graph_posterior(p.class, &p.names, &[only]);
        let run = Study::tabular(p.data.clone())
            .graph_posterior(graphs)
            .query(p.query.clone())
            .inference(inference(bayesian))
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(3)))
            .unwrap_or_else(|e| panic!("{label}: atom {} alone: {e}", atom["name"]));
        alone.push(
            run.refutations
                .iter()
                .map(|r| (r.refuter.to_string(), r.passed, atom["name"].to_string()))
                .collect(),
        );
    }
    for report in &result.refutations {
        let name = report.refuter.as_ref();
        let verdicts: Vec<bool> = alone
            .iter()
            .flat_map(|reports| {
                reports.iter().filter(|(n, _, _)| n == name).map(|(_, passed, _)| *passed)
            })
            .collect();
        assert!(!verdicts.is_empty(), "{label}: refuter {name} is in no atom's own suite");
        // A permutation placebo draws its own permutation in each context and compares it with
        // an exact zero (no bootstrap spread), so the verdict of one atom alone and inside the
        // mixture is not the same draw: only its presence is compared.
        if name.contains("placebo") {
            continue;
        }
        assert_eq!(
            report.passed,
            verdicts.iter().all(|passed| *passed),
            "{label}: {name} ({:?}) mixed verdict against the atoms' own {verdicts:?} {alone:?}",
            report.failure_condition
        );
    }
}

fn run_retained(family: Family) {
    on_large_stack(move || {
        let ctx = ExecutionContext::for_tests(4);
        for bayesian in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let scale = if suite == RefuteSuite::None { 1 } else { 40 };
                for p in posteriors(family, scale) {
                    let label = format!("{family:?} {} bayesian={bayesian} {suite:?}", p.class);
                    let builder = builder(&p, bayesian, suite);
                    let study = builder.clone().build().unwrap();
                    let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                    drop(builder);
                    drop(study);
                    assert!(
                        !matches!(prepared.retained_operation(), RetainedOperation::FreshRun),
                        "{label}: {:?}",
                        prepared.retained_operation()
                    );
                    let result = prepared.estimate(&p.data, &ctx).unwrap();
                    assert_values(&label, &p, &result, bayesian);
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                }
            }
        }
    });
}

#[test]
fn path_specific_graph_posterior_is_the_weighted_mean_of_the_atoms_identified_functionals_under_every_suite()
 {
    run_family(Family::PathSpecific);
}

#[test]
fn interventional_distribution_graph_posterior_is_the_weighted_mean_of_the_atoms_identified_laws_under_every_suite()
 {
    run_family(Family::Distribution);
}

#[test]
fn mediation_graph_posterior_is_the_weighted_mean_of_the_atoms_components_under_every_suite() {
    run_family(Family::Mediation);
}

#[test]
fn path_specific_graph_posterior_executes_from_the_retained_plan_without_the_builder() {
    run_retained(Family::PathSpecific);
}

#[test]
fn interventional_distribution_graph_posterior_executes_from_the_retained_plan_without_the_builder()
{
    run_retained(Family::Distribution);
}

#[test]
fn mediation_graph_posterior_executes_from_the_retained_plan_without_the_builder() {
    run_retained(Family::Mediation);
}

/// The unidentified atom is mass, not an imputed value: with the refused atom's weight moved to
/// the identified atoms the mean would differ, and the result states the mass it does not cover.
#[test]
fn a_refused_atom_is_unidentified_mass_and_is_not_redistributed() {
    on_large_stack(|| {
        for family in [Family::PathSpecific, Family::Distribution, Family::Mediation] {
            for p in posteriors(family, 1) {
                if p.expected.unidentified == 0.0 {
                    continue;
                }
                let result = builder(&p, false, RefuteSuite::None)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(5))
                    .unwrap();
                let mixture = result.structural_response.as_ref().unwrap();
                let refused = mixture.atoms.iter().filter(|a| a.value.is_none()).count();
                assert!(refused >= 1, "{family:?} {}: no refused atom is listed", p.class);
                assert!(
                    (mixture.identified_mass - (1.0 - p.expected.unidentified)).abs() < 1e-9,
                    "{family:?} {}: identified mass {}",
                    p.class,
                    mixture.identified_mass
                );
            }
        }
    });
}

/// A posterior every atom of which refuses the query identifies nothing: the run is refused with
/// the not-identified error, never an imputed value.
#[test]
fn a_posterior_every_atom_of_which_refuses_the_query_is_not_identified() {
    on_large_stack(|| {
        for family in [Family::PathSpecific, Family::Mediation] {
            let p = posterior(family, "admg", 1);
            let pin = pin();
            let entry = &pin["posteriors"][family.key()]["admg"];
            let table = &pin["tables"][entry["table"].as_str().unwrap()];
            let names = columns(table);
            let refused: Vec<Json> = entry["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|a| a["value"].is_null())
                .cloned()
                .collect();
            assert_eq!(refused.len(), 1);
            let mut only = refused[0].clone();
            only["weight"] = Json::from("1/1");
            let graphs = graph_posterior("admg", &names, &[only]);
            let error = Study::tabular(p.data.clone())
                .graph_posterior(graphs)
                .query(p.query.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .and_then(|study| study.run(&ExecutionContext::for_tests(6)))
                .unwrap_err();
            assert!(
                matches!(error, antecedent::CausalError::NotIdentified { .. }),
                "{family:?}: {error}"
            );
        }
    });
}

/// A graph posterior stores each atom as an adjacency mask that cannot hold a directed and a
/// bidirected edge on one pair (a bow): such an atom is refused at construction with a typed
/// reason, never encoded as one of the two edges.
#[test]
fn a_bow_atom_cannot_enter_a_graph_posterior() {
    let mut g = Admg::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let error = adjacency_mask_from_admg(&g).unwrap_err().to_string();
    assert!(error.contains("bow"), "{error}");
}
