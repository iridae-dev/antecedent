//! The counterfactual (effect on the treated) over a graph posterior of Dag, ADMG, temporal-ADMG,
//! CPDAG and PAG atoms, on both inference sides and under every validation suite, against the
//! posterior-weighted mean of each atom's own closed form
//! (`conformance/estimate/cross_world_counterfactual_posterior`).
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
    CausalQuery, CounterfactualQuery, ExecutionContext, IdentificationStatus, Intervention, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;
use std::sync::Arc;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_counterfactual_posterior/expected.json"
    ))
    .unwrap()
}
fn node(r: u32) -> DenseNodeId {
    DenseNodeId::from_raw(r)
}
fn v(r: u32) -> VariableId {
    VariableId::from_raw(r)
}
fn table(pin: &serde_json::Value, name: &str) -> TabularData {
    let spec = &pin["tables"][name];
    let names: Vec<String> = spec["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for cell in spec["counts"].as_array().unwrap() {
        let levels: Vec<f64> =
            cell["levels"].as_array().unwrap().iter().map(|l| l.as_f64().unwrap()).collect();
        for _ in 0..cell["count"].as_u64().unwrap() {
            for (c, l) in data.iter_mut().zip(&levels) {
                c.push(*l);
            }
        }
    }
    TabularData::from_f64_columns(names.iter().zip(&data).map(|(n, c)| (n.as_str(), c.as_slice())))
        .unwrap()
}
fn dag_mask(n: usize, edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |m, &(a, b)| set_edge(m, n, a, b, true))
}
fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("x"),
        rejected_invalid: 0,
    }
}
fn gp(n: usize, kind: GraphPosteriorAtomKind, w: &[f64], masks: Vec<u64>) -> GraphPosterior {
    let mut g = GraphPosterior::new(
        n,
        w.to_vec(),
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
fn admg(edges: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(node(a), node(b)).unwrap();
    }
    for &(a, b) in bidirected {
        g.insert_bidirected(node(a), node(b)).unwrap();
    }
    g
}
fn dag(edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(node(a), node(b)).unwrap();
    }
    g
}

struct Posterior {
    kind: &'static str,
    graphs: GraphPosterior,
    data: TabularData,
    query: CausalQuery,
    expected: f64,
    tolerance: f64,
    unidentified: f64,
}

fn counterfactual(treatment: u32, outcome: u32) -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(outcome),
        Arc::from([Intervention::set(v(treatment), Value::f64(1.0))]),
    ))
}

/// Seeded standard-normal stream (xorshift64* uniforms, Box-Muller).
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, w) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * w).cos()
    }
}

/// Columns `z, t, y` with `y = 1 + 2 t + z` and a small deterministic wiggle (the generating
/// equations of the cross-world DAG validation fixture).
fn gaussian_table() -> TabularData {
    let n = 240_u32;
    let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.37).sin()).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 0.6 * z + (f64::from(i) * 0.91).cos()).collect();
    let mut noise = Normals(20_241_029);
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + z + 0.4 * noise.next()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn line_table(pin: &serde_json::Value) -> TabularData {
    let line = &pin["line"];
    let rows = line["rows"].as_u64().unwrap();
    let (intercept, slope) = (line["intercept"].as_f64().unwrap(), line["slope"].as_f64().unwrap());
    let x: Vec<f64> = (0..rows).map(|i| i as f64 / 20.0 - 2.0).collect();
    let y: Vec<f64> = x.iter().map(|x| intercept + slope * x).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn posteriors() -> Vec<Posterior> {
    let pin = pin();
    let expected = pin["posterior"]["expected"].as_f64().unwrap();
    let unidentified = pin["posterior"]["unidentified_mass"].as_f64().unwrap();
    let front_door = (&[(0, 1), (1, 2)][..], &[(0, 2)][..]);
    let unconfounded = (&[(0, 1), (0, 2), (1, 2)][..], &[][..]);
    let cyclic = dag_mask(3, &[(0, 1), (1, 2), (2, 0)]);
    let w = [0.5, 0.3, 0.2];
    let mut out = Vec::new();
    let masks = |kind: &str| -> Vec<u64> {
        let both = [front_door, unconfounded];
        let mut m: Vec<u64> =
            both.iter().map(|(d, b)| adjacency_mask_from_admg(&admg(d, b)).unwrap()).collect();
        let _ = kind;
        m.push(cyclic);
        m
    };
    for kind in ["Admg", "TemporalAdmg"] {
        out.push(Posterior {
            kind: if kind == "Admg" { "Admg" } else { "TemporalAdmg" },
            graphs: gp(
                3,
                if kind == "Admg" {
                    GraphPosteriorAtomKind::Admg
                } else {
                    GraphPosteriorAtomKind::TemporalAdmg
                },
                &w,
                masks(kind),
            ),
            data: table(&pin, "front_door"),
            query: counterfactual(0, 2),
            expected,
            tolerance: 1e-9,
            unidentified,
        });
    }
    // A PAG whose edges are all visible beside a reverse-causal atom that leaves the treatment edge
    // invisible: one identified atom carries the whole conditional value.
    let visible = [(0, 1), (0, 2), (2, 3), (1, 3)]
        .iter()
        .fold(0_u64, |m, &(x, z)| set_edge(m, 4, x, z, true));
    let reverse = [(1, 0), (0, 2), (2, 3), (1, 3)]
        .iter()
        .fold(0_u64, |m, &(x, z)| set_edge(m, 4, x, z, true));
    out.push(Posterior {
        kind: "Pag",
        graphs: gp(4, GraphPosteriorAtomKind::Pag, &[0.8, 0.2], vec![visible, reverse]),
        data: table(&pin, "side_path_chain"),
        query: counterfactual(1, 3),
        expected: pin["cases"]["pag_counterfactual"]["expected"].as_f64().unwrap(),
        tolerance: 1e-9,
        unidentified: 0.2,
    });
    // One CPDAG atom over the noiseless line: its two completions disagree, the value is their mean.
    let mut line = Cpdag::with_variables(2);
    line.insert_undirected(node(0), node(1)).unwrap();
    out.push(Posterior {
        kind: "Cpdag",
        graphs: gp(
            2,
            GraphPosteriorAtomKind::Cpdag,
            &[1.0],
            vec![adjacency_mask_from_cpdag(&line).unwrap()],
        ),
        data: line_table(&pin),
        query: counterfactual(0, 1),
        expected: pin["line"]["mean"].as_f64().unwrap(),
        tolerance: 1e-6,
        unidentified: 0.0,
    });
    out.push(Posterior {
        kind: "Dag",
        graphs: gp(
            3,
            GraphPosteriorAtomKind::Dag,
            &w,
            vec![dag_mask(3, &[(0, 1), (0, 2), (1, 2)]), dag_mask(3, &[(0, 2), (1, 2)]), cyclic],
        ),
        data: gaussian_table(),
        query: CausalQuery::Counterfactual(
            CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(1), Value::f64(1.5))]))
                .with_control_level(0.0),
        ),
        expected: pin["gaussian"]["expected"].as_f64().unwrap(),
        tolerance: pin["gaussian"]["tolerance"].as_f64().unwrap(),
        unidentified,
    });
    out
}

fn run(
    p: &Posterior,
    bayesian: bool,
    suite: RefuteSuite,
) -> Result<(antecedent::analysis::Study, antecedent::StudyResult), antecedent::CausalError> {
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
    } else {
        InferenceMode::Frequentist
    };
    let study = Study::tabular(p.data.clone())
        .graph_posterior(p.graphs.clone())
        .query(p.query.clone())
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()?;
    let result = study.run(&ExecutionContext::for_tests(3))?;
    Ok((study, result))
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().unwrap();
}

#[test]
fn counterfactual_graph_posterior_is_the_weighted_mean_of_the_atoms_closed_forms_under_every_suite()
{
    on_large_stack(|| {
        for p in posteriors() {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    if p.kind == "Pag" && suite != RefuteSuite::None {
                        continue;
                    }
                    let label = format!("{} bayesian={bayesian} {suite:?}", p.kind);
                    let (study, result) =
                        run(&p, bayesian, suite).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    let tolerance = if bayesian { p.tolerance.max(0.03) } else { p.tolerance };
                    assert!(
                        (result.effect() - p.expected).abs() <= tolerance,
                        "{label}: {} vs {}",
                        result.effect(),
                        p.expected
                    );
                    // The refused atom is unidentified posterior mass, not redistributed.
                    let mixture = result.structural_response.as_ref().expect(&label);
                    assert!(
                        (mixture.unidentified_mass + mixture.unevaluable_mass - p.unidentified)
                            .abs()
                            < 1e-9,
                        "{label}"
                    );
                    if p.unidentified > 0.0 {
                        assert_eq!(
                            result.identification.status,
                            IdentificationStatus::GraphDependent,
                            "{label}"
                        );
                    }
                    if suite == RefuteSuite::None {
                        assert!(result.refutations.is_empty(), "{label}");
                        continue;
                    }
                    assert!(!result.refutations.is_empty(), "{label}: no refuter ran");
                    // A mixed verdict passes only if every contributing atom passes.
                    assert!(
                        result.refutations.iter().all(|r| r.passed),
                        "{label}: {:?}",
                        result
                            .refutations
                            .iter()
                            .filter(|r| !r.passed)
                            .map(|r| format!("{} {:?}", r.refuter, r.failure_condition))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    });
}

#[test]
fn counterfactual_graph_posterior_executes_from_the_retained_plan_without_the_builder() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(4);
        for p in posteriors() {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    if p.kind == "Pag" && suite != RefuteSuite::None {
                        continue;
                    }
                    let label = format!("{} bayesian={bayesian} {suite:?}", p.kind);
                    let inference = if bayesian {
                        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
                    } else {
                        InferenceMode::Frequentist
                    };
                    let builder = Study::tabular(p.data.clone())
                        .graph_posterior(p.graphs.clone())
                        .query(p.query.clone())
                        .inference(inference)
                        .refute(suite)
                        .bootstrap_replicates(0);
                    let study = builder.clone().build().unwrap();
                    let prepared = study.prepare(&ctx).unwrap();
                    drop(builder);
                    drop(study);
                    let operation = format!("{:?}", prepared.retained_operation());
                    assert!(
                        !matches!(prepared.retained_operation(), RetainedOperation::FreshRun),
                        "{label}: {operation}"
                    );
                    let result = prepared.estimate(&p.data, &ctx).unwrap();
                    let tolerance = if bayesian { p.tolerance.max(0.03) } else { p.tolerance };
                    assert!(
                        (result.effect() - p.expected).abs() <= tolerance,
                        "{label}: {}",
                        result.effect()
                    );
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                }
            }
        }
    });
}

#[test]
fn a_validated_pag_counterfactual_posterior_is_refused_with_its_reason() {
    for (kind, suites, needle) in [("Pag", [RefuteSuite::Cheap, RefuteSuite::Full], "PAG atoms")] {
        let p = posteriors().into_iter().find(|p| p.kind == kind).unwrap();
        for bayesian in [false, true] {
            for suite in suites {
                let inference = if bayesian {
                    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
                } else {
                    InferenceMode::Frequentist
                };
                let error = Study::tabular(p.data.clone())
                    .graph_posterior(p.graphs.clone())
                    .query(p.query.clone())
                    .inference(inference)
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap_err()
                    .to_string();
                assert!(error.starts_with("refused:"), "{kind} {suite:?}: {error}");
                assert!(error.contains(needle), "{kind} {suite:?}: {error}");
            }
        }
    }
}
