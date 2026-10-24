//! Transport and interference over a graph posterior, and transport over a Cpdag or Pag.
//!
//! Every expected value is an independent reference: the closed-form two-row inverse-odds sum
//! pinned in `conformance/estimate/staged_transport`, the Dirichlet moments pinned in
//! `conformance/estimate/bayesian_trial_transport`, the enumerated two-unit Bernoulli design,
//! the cluster total effect, and the stated Gaussian exposure model, each computed without the
//! engine; and sID run directly on each atom's selection diagram.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    BayesianConfig, CellStatus, InferenceMode, InterferenceSpec, RefuteSuite, Study,
    TransportTrialSpec,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDomain, ExecutionContext, ExposureLevel,
    ExposureMapping, GridSpec, IdentificationStatus, InterferenceFunctional, InterferenceQuery,
    ResponseFunctional, ResponseQuery, ResponseValue, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, adjacency_masks_from_pag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, SelectionDiagram};
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_prob::InferenceDiagnostics;

/// Graph-posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("graph-posterior-exposure".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn sampling(label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

// ---------------------------------------------------------------------------
// Transport over a graph posterior.
// ---------------------------------------------------------------------------

const A: u32 = 0;
const Y: u32 = 1;
const TRIAL: u32 = 2;
const SEL: u32 = 3;
const PROP: u32 = 4;
const Z: u32 = 5;
const NODES: u32 = 6;

/// The pinned two-trial-row fixture with a covariate column: IPW = 4 / 2 = 2 exactly.
fn transport_data() -> TabularData {
    TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
        ("z", &[0.0, 1.0, 0.0, 1.0][..]),
    ])
    .unwrap()
}

fn transport_query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(A),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(A)])
}

fn trial_spec() -> TransportTrialSpec {
    TransportTrialSpec {
        trial: VariableId::from_raw(TRIAL),
        selection_probability: VariableId::from_raw(SEL),
        treatment_probability: VariableId::from_raw(PROP),
    }
}

fn admg(directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(NODES);
    for &(from, to) in directed {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for &(left, right) in bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(left), DenseNodeId::from_raw(right)).unwrap();
    }
    graph
}

/// The four atoms of the mixture, with the covariate `z` as the selection target:
///
/// * `direct`: `a -> y`; no population-varying mechanism is an ancestor of `y`.
/// * `standardized`: `a -> y <- z`; `z` is a parentless pre-treatment covariate.
/// * `uncertified`: `a -> z -> y` with `trial <-> s` elsewhere; `z` is downstream of the
///   treatment, no S-admissible set exists and the c-components are not singletons.
/// * `recursive`: `a -> z -> y`; sID certifies a recursive factorization the weighting cannot
///   evaluate.
fn atoms() -> [Admg; 4] {
    [
        admg(&[(A, Y)], &[]),
        admg(&[(A, Y), (Z, Y)], &[]),
        admg(&[(A, Z), (Z, Y)], &[(TRIAL, SEL)]),
        admg(&[(A, Z), (Z, Y)], &[]),
    ]
}

fn posterior(weights: [f64; 4]) -> GraphPosterior {
    let masks: Vec<_> =
        atoms().iter().map(|graph| adjacency_mask_from_admg(graph).unwrap()).collect();
    let n = NODES as usize;
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_transport"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

fn transport_study(
    posterior: GraphPosterior,
    targets: &[u32],
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let targets: Vec<VariableId> = targets.iter().map(|raw| VariableId::from_raw(*raw)).collect();
    Study::tabular(transport_data())
        .graph_posterior(posterior)
        .query(CausalQuery::Transport(transport_query()))
        .selection_targets(Arc::<[VariableId]>::from(targets))
        .transport_trial(trial_spec())
        .inference(inference)
        .refute(suite)
        .build()
        .unwrap()
}

/// `sqrt(sum of squared per-row influences) / target rows` for the pinned fixture: the trial
/// rows contribute `+6` and `-2`, the two target rows `-ipw = -2` each.
fn reference_se() -> f64 {
    (36.0_f64 + 4.0 + 4.0 + 4.0).sqrt() / 2.0
}

fn value_of(result: &antecedent::StudyResult, key: u64) -> Option<f64> {
    let mixture = result.structural_response.as_ref().expect("a structural mixture");
    let atom = mixture.atoms.iter().find(|atom| atom.graph_key == key).expect("atom present");
    match atom.value {
        Some(ResponseValue::Scalar(value)) => Some(value),
        Some(_) => panic!("a transport atom value is a scalar"),
        None => None,
    }
}

/// sID run directly on each atom's selection diagram names the rule that decides the atom.
#[test]
fn each_atom_is_decided_by_sid_on_its_own_selection_diagram() {
    let targets = vec![VariableId::from_raw(Z)];
    let rules: Vec<String> = atoms()
        .into_iter()
        .map(|graph| {
            let diagram = SelectionDiagram::try_new(graph, targets.clone()).unwrap();
            match TransportIdentifier::new().identify(&diagram, &transport_query()).unwrap() {
                TransportIdentification::Transportable { certificate, .. } => {
                    certificate.rule.to_string()
                }
                TransportIdentification::NotCertified(certificate) => {
                    format!("not_certified:{}", certificate.reason)
                }
                TransportIdentification::MissingEvidence(certificate) => {
                    format!("missing:{}", certificate.reason)
                }
            }
        })
        .collect();
    assert_eq!(
        rules,
        [
            "transport.sid.direct",
            "transport.sid.standardize",
            "not_certified:transport.sid.multinode_c_component_not_implemented",
            "transport.sid.singleton_c_components",
        ]
    );
}

#[test]
fn frequentist_transport_mixes_certified_atoms_and_states_the_lost_mass() {
    on_large_stack(|| {
        let weights = [0.35, 0.25, 0.2, 0.2];
        let graphs = posterior(weights);
        let study =
            transport_study(graphs.clone(), &[Z], InferenceMode::Frequentist, RefuteSuite::None);
        let ctx = ExecutionContext::for_tests(7_001);
        let result = study.run(&ctx).unwrap();

        // The pinned closed-form contrast and its delta-method interval.
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12, "{}", effect.ate);
        assert!((effect.se_analytic() - reference_se()).abs() < 1e-12);
        assert!((result.transport().unwrap().ipw - 2.0).abs() < 1e-12);
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("transport.trial_ipw"));

        // The posterior mass splits as sID decided each atom.
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.6).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9, "{}", mixture.unidentified_mass);
        assert!((mixture.unevaluable_mass - 0.2).abs() < 1e-9, "{}", mixture.unevaluable_mass);
        assert_eq!(mixture.atoms.len(), 4);
        let keys = &graphs.graph_keys;
        for certified in [keys[0], keys[1]] {
            let value = value_of(&result, certified).expect("a certified atom has a value");
            assert!((value - 2.0).abs() < 1e-12, "{value}");
        }
        assert!(value_of(&result, keys[2]).is_none(), "the uncertified atom has no value");
        assert!(value_of(&result, keys[3]).is_none(), "the recursive atom has no value");
        let status_of = |key: u64| {
            mixture.atoms.iter().find(|atom| atom.graph_key == key).expect("atom").status
        };
        assert_eq!(status_of(keys[0]), IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(status_of(keys[2]), IdentificationStatus::NotIdentified);
        assert_eq!(
            status_of(keys[3]),
            IdentificationStatus::NonparametricallyIdentified,
            "certified, but not evaluable by the weighting"
        );
        // Neither lost mass is redistributed, so the status is graph dependent.
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        let rules = result
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code.as_ref() == "estimate.graph_posterior.transport_sid")
            .expect("the per-atom sID rules are disclosed");
        assert!(rules.message.contains("transport.sid.direct=0.35"), "{}", rules.message);
        assert!(rules.message.contains("transport.sid.standardize=0.25"), "{}", rules.message);
        assert!(
            rules.message.contains("transport.sid.singleton_c_components=0.2"),
            "{}",
            rules.message
        );

        // The prepared handle retains the per-atom proofs and returns the same answer.
        let mut prepared = study.prepare(&ctx).unwrap();
        assert!(matches!(prepared.retained_operation(), RetainedOperation::EstimatorRoute));
        let again = prepared.estimate(&transport_data(), &ctx).unwrap();
        assert!((again.estimate.as_effect().unwrap().ate - effect.ate).abs() < 1e-12);
        assert!(again.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
        assert!((again.structural_response.as_ref().unwrap().unevaluable_mass - 0.2).abs() < 1e-9);
    });
}

/// With no selection target every atom is `Direct`: the posterior certifies all of its mass and
/// the same graphs identify the same contrast, so the lost mass is exactly what the diagram
/// decides and nothing else.
#[test]
fn the_selection_targets_decide_how_much_posterior_mass_identifies() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_002);
        let graphs = posterior([0.35, 0.25, 0.2, 0.2]);
        let open = transport_study(graphs, &[], InferenceMode::Frequentist, RefuteSuite::None)
            .run(&ctx)
            .unwrap();
        let mixture = open.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        assert_eq!(open.identification.status, IdentificationStatus::NonparametricallyIdentified);
        assert!((open.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);

        // The mass the diagram does not certify follows the posterior weights.
        let shifted = transport_study(
            posterior([0.1, 0.1, 0.7, 0.1]),
            &[Z],
            InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .run(&ctx)
        .unwrap();
        let mixture = shifted.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.2).abs() < 1e-9);
        assert!((mixture.unidentified_mass - 0.7).abs() < 1e-9);
        assert!((mixture.unevaluable_mass - 0.1).abs() < 1e-9);
        assert!((shifted.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);
    });
}

#[test]
fn bayesian_transport_mixes_the_trial_row_posterior_over_certified_atoms() {
    on_large_stack(|| {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/bayesian_trial_transport/expected.json"
        ))
        .unwrap();
        let study = transport_study(
            posterior([0.35, 0.25, 0.2, 0.2]),
            &[Z],
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)),
            RefuteSuite::None,
        );
        let ctx = ExecutionContext::for_tests(7_003);
        let result = study.run(&ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("a transport posterior");
        assert_eq!(posterior.draws.n_draws, 2_000);
        // Dirichlet(1, 1) over the two trial rows makes the draw 8U - 2.
        assert!(
            (posterior.summaries.mean[0] - fixture["expected_mean"].as_f64().unwrap()).abs()
                < fixture["mean_tolerance"].as_f64().unwrap()
        );
        assert!(
            (posterior.summaries.sd[0].powi(2) - fixture["expected_variance"].as_f64().unwrap())
                .abs()
                < fixture["variance_tolerance"].as_f64().unwrap()
        );
        assert!(
            (posterior.unidentified_mass - 0.2).abs() < 1e-9,
            "{}",
            posterior.unidentified_mass
        );
        assert!((posterior.unevaluable_mass - 0.2).abs() < 1e-9, "{}", posterior.unevaluable_mass);
        assert!(
            (result.estimate.as_effect().unwrap().ate - posterior.summaries.mean[0]).abs() < 1e-12
        );
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("transport.trial_bayesian_bootstrap")
        );
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.6).abs() < 1e-9);
    });
}

#[test]
fn the_transport_native_suite_runs_once_on_the_data_over_a_posterior() {
    on_large_stack(|| {
        let study = transport_study(
            posterior([0.35, 0.25, 0.2, 0.2]),
            &[Z],
            InferenceMode::Frequentist,
            RefuteSuite::Cheap,
        );
        let result = study.run(&ExecutionContext::for_tests(7_004)).unwrap();
        let ids: Vec<&str> =
            result.refutations.iter().map(|report| report.refuter.as_ref()).collect();
        assert!(ids.contains(&"transport.selection_calibration"), "{ids:?}");
        assert!(ids.contains(&"transport.arm_calibration"), "{ids:?}");
    });
}

#[test]
fn a_posterior_with_no_certified_atom_is_not_identified() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_005);
        let only_uncertified = GraphPosterior::new(
            NODES as usize,
            vec![1.0],
            vec![adjacency_mask_from_admg(&atoms()[2]).unwrap()],
            vec![0.0; 36],
            vec![0.0; 36],
            sampling("graph_posterior_transport_uncertified"),
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let study =
            transport_study(only_uncertified, &[Z], InferenceMode::Frequentist, RefuteSuite::None);
        assert!(study.run(&ctx).is_err());

        // A single certified atom that only the recursive formula decides cannot be evaluated.
        let only_recursive = GraphPosterior::new(
            NODES as usize,
            vec![1.0],
            vec![adjacency_mask_from_admg(&atoms()[3]).unwrap()],
            vec![0.0; 36],
            vec![0.0; 36],
            sampling("graph_posterior_transport_recursive"),
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let study =
            transport_study(only_recursive, &[Z], InferenceMode::Frequentist, RefuteSuite::None);
        assert!(study.run(&ctx).is_err());
    });
}

// ---------------------------------------------------------------------------
// Interference over a graph posterior.
// ---------------------------------------------------------------------------

fn network(data: &TabularData, edges: &[(u64, u64)]) -> NetworkData {
    NetworkData::try_new(
        data.clone(),
        edges.iter().map(|&(from, to)| NetworkEdge::new(from, to, 1.0)).collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Atoms over `[y, z]`: `y -> z`, the empty graph, `z -> y`, and a cyclic mask that is not a DAG
/// (weight 0.2).
fn exposure_posterior(kind: GraphPosteriorAtomKind, weights: [f64; 4]) -> GraphPosterior {
    let cyclic = set_edge(set_edge(0, 2, 0, 1, true), 2, 1, 0, true);
    GraphPosterior::new(
        2,
        weights.to_vec(),
        vec![set_edge(0, 2, 0, 1, true), 0, set_edge(0, 2, 1, 0, true), cyclic],
        vec![0.0; 4],
        vec![0.0; 4],
        sampling("graph_posterior_interference"),
    )
    .unwrap()
    .with_atom_kind(kind)
}

fn interference_study(
    data: &TabularData,
    query: InterferenceQuery,
    spec: InterferenceSpec,
    posterior: Option<GraphPosterior>,
    inference: InferenceMode,
) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = match posterior {
        Some(posterior) => builder.graph_posterior(posterior),
        None => builder.graph(Dag::with_variables(2)),
    };
    builder
        .query(CausalQuery::Interference(query))
        .interference(spec)
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn bernoulli_query() -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 1.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    )
}

fn cluster_query() -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from([0, 0, 1, 1, 2, 2, 3, 3]),
            treated_clusters: 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    )
}

/// The enumerated two-unit Bernoulli design of `conformance/response/randomized_interference`:
/// Horvitz-Thompson -6, Hajek -3, Young variance bound 204.
fn bernoulli_fixture() -> (TabularData, InterferenceSpec) {
    let data =
        TabularData::from_f64_columns([("y", &[1.0, 4.0][..]), ("z", &[0.0, 1.0][..])]).unwrap();
    let spec = InterferenceSpec {
        network: network(&data, &[(0, 1), (1, 0)]),
        assignment: Arc::from([true, false]),
    };
    (data, spec)
}

#[test]
fn the_bernoulli_contrast_is_the_enumerated_design_value_whatever_the_posterior() {
    on_large_stack(|| {
        let (data, spec) = bernoulli_fixture();
        let ctx = ExecutionContext::for_tests(7_101);
        let explicit = interference_study(
            &data,
            bernoulli_query(),
            spec.clone(),
            None,
            InferenceMode::Frequentist,
        )
        .run(&ctx)
        .unwrap();

        let weights = [0.4, 0.3, 0.1, 0.2];
        let over_dags = interference_study(
            &data,
            bernoulli_query(),
            spec.clone(),
            Some(exposure_posterior(GraphPosteriorAtomKind::Dag, weights)),
            InferenceMode::Frequentist,
        );
        let result = over_dags.run(&ctx).unwrap();
        let estimated = result.interference().expect("design-based contrast");
        assert!((estimated.contrast.horvitz_thompson + 6.0).abs() < 1e-12);
        assert!((estimated.contrast.hajek + 3.0).abs() < 1e-12);
        assert!((estimated.contrast.conservative_variance - 204.0).abs() < 1e-9);
        assert!(
            (result.estimate.as_effect().unwrap().ate - explicit.estimate.as_effect().unwrap().ate)
                .abs()
                < 1e-12,
            "the posterior does not change the contrast"
        );
        assert_eq!(result.support_status, Some(CellStatus::Refused));

        // The cyclic mask is not a DAG: its weight is the unidentified mass.
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9);
        assert_eq!(mixture.atoms.len(), 4);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        for atom in mixture.atoms.iter().filter(|atom| atom.value.is_some()) {
            match atom.value {
                Some(ResponseValue::Scalar(value)) => assert!((value + 6.0).abs() < 1e-12),
                ref other => panic!("an interference atom value is a scalar: {other:?}"),
            }
        }
        assert!(result.diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_ref() == "estimate.graph_posterior.interference_design"
        }));

        // Another weighting and another atom kind move the masses and not the contrast: the
        // same masks are acyclic ADMGs, one a bidirected edge, so nothing is unidentified.
        let over_admgs = interference_study(
            &data,
            bernoulli_query(),
            spec,
            Some(exposure_posterior(GraphPosteriorAtomKind::Admg, [0.1, 0.2, 0.3, 0.4])),
            InferenceMode::Frequentist,
        )
        .run(&ctx)
        .unwrap();
        let mixture = over_admgs.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        assert_eq!(over_admgs.identification.status, explicit.identification.status);
        assert_eq!(
            over_admgs.estimate.as_effect().unwrap().ate.to_bits(),
            result.estimate.as_effect().unwrap().ate.to_bits()
        );
    });
}

#[test]
fn the_cluster_total_effect_runs_the_checked_design_over_a_posterior() {
    on_large_stack(|| {
        let data = TabularData::from_f64_columns([
            ("y", &[4.0, 4.0, 6.0, 6.0, 1.0, 1.0, 3.0, 3.0][..]),
            ("z", &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
        ])
        .unwrap();
        let edges: Vec<(u64, u64)> = (0..4_u64)
            .flat_map(|cluster| [(2 * cluster, 2 * cluster + 1), (2 * cluster + 1, 2 * cluster)])
            .collect();
        let spec = InterferenceSpec {
            network: network(&data, &edges),
            assignment: Arc::from([true, true, true, true, false, false, false, false]),
        };
        let ctx = ExecutionContext::for_tests(7_102);
        let result = interference_study(
            &data,
            cluster_query(),
            spec.clone(),
            Some(exposure_posterior(GraphPosteriorAtomKind::Dag, [0.4, 0.3, 0.1, 0.2])),
            InferenceMode::Frequentist,
        )
        .run(&ctx)
        .unwrap();
        // Cluster totals 8 and 12 (treated) against 2 and 6 (control): a difference of means of 6
        // over two units per cluster is a total effect of 3, and the two arms' variances of 8
        // over two clusters each give a Neyman standard error of sqrt(8) / 2 = sqrt(2).
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - 3.0).abs() < 1e-12);
        assert!((effect.se_analytic() - 2.0_f64.sqrt()).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9);
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("interference.cluster_neyman"));

        // The prepared handle returns the same contrast with its posterior retained.
        let mut prepared = interference_study(
            &data,
            cluster_query(),
            spec,
            Some(exposure_posterior(GraphPosteriorAtomKind::Dag, [0.4, 0.3, 0.1, 0.2])),
            InferenceMode::Frequentist,
        )
        .prepare(&ctx)
        .unwrap();
        assert!(matches!(prepared.retained_operation(), RetainedOperation::EstimatorRoute));
        let again = prepared.estimate(&data, &ctx).unwrap();
        assert!((again.estimate.as_effect().unwrap().ate - 3.0).abs() < 1e-12);
    });
}

#[test]
fn the_gaussian_exposure_posterior_is_conditional_on_the_design_and_carries_the_atom_mass() {
    on_large_stack(|| {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/bayesian_interference/expected.json"
        ))
        .unwrap();
        let outcomes: Vec<f64> = fixture["outcomes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        let assignment: Vec<bool> = fixture["assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect();
        let edges: Vec<(u64, u64)> = fixture["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| (edge[0].as_u64().unwrap(), edge[1].as_u64().unwrap()))
            .collect();
        let data = TabularData::from_f64_columns([
            ("y", outcomes.as_slice()),
            ("z", &[0.0, 1.0, 0.0, 1.0][..]),
        ])
        .unwrap();
        let spec =
            InterferenceSpec { network: network(&data, &edges), assignment: Arc::from(assignment) };
        let study = interference_study(
            &data,
            bernoulli_query(),
            spec,
            Some(exposure_posterior(GraphPosteriorAtomKind::Dag, [0.4, 0.3, 0.1, 0.2])),
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(20_000)),
        );
        let result = study.run(&ExecutionContext::for_tests(7_103)).unwrap();
        // y = 2 + 3 own + 4 treated neighbors: (own 0, 1 neighbor) -> (own 1, 0 neighbors) is -1.
        let truth = fixture["expected_contrast"].as_f64().unwrap();
        let tolerance = fixture["tolerance"].as_f64().unwrap();
        assert!((result.estimate.as_effect().unwrap().ate - truth).abs() < tolerance);
        let posterior = result.posterior.as_ref().expect("a contrast posterior");
        assert_eq!(posterior.draws.n_draws, 20_000);
        assert!((posterior.unidentified_mass - 0.2).abs() < 1e-9);
        assert_eq!(
            result.identification.status,
            IdentificationStatus::GraphDependent,
            "the lost atom mass leaves the parametric identification graph dependent"
        );
    });
}

#[test]
fn a_posterior_that_does_not_cover_the_unit_schema_is_refused() {
    on_large_stack(|| {
        let (data, spec) = bernoulli_fixture();
        let three = GraphPosterior::new(
            3,
            vec![1.0],
            vec![0],
            vec![0.0; 9],
            vec![0.0; 9],
            sampling("graph_posterior_interference_wide"),
        )
        .unwrap();
        let study = interference_study(
            &data,
            bernoulli_query(),
            spec,
            Some(three),
            InferenceMode::Frequentist,
        );
        assert!(study.run(&ExecutionContext::for_tests(7_104)).is_err());
    });
}

// ---------------------------------------------------------------------------
// Transport over a Cpdag or a Pag.
// ---------------------------------------------------------------------------

fn class_study(graph: ClassGraph, targets: &[u32], inference: InferenceMode) -> Study {
    let targets: Vec<VariableId> = targets.iter().map(|raw| VariableId::from_raw(*raw)).collect();
    let builder = Study::tabular(transport_data());
    let builder = match graph {
        ClassGraph::Cpdag(graph) => builder.graph(graph),
        ClassGraph::Pag(graph) => builder.graph(graph),
    };
    builder
        .query(CausalQuery::Transport(transport_query()))
        .selection_targets(Arc::<[VariableId]>::from(targets))
        .transport_trial(trial_spec())
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// `a - y - z`: no collider, so three DAG completions (`a -> y -> z`, `a <- y -> z` and
/// `a <- y <- z`). With the covariate `z` as the selection target the first two certify the
/// direct formula (`z` is not an ancestor of `y`) and the third standardizes over `z`.
fn open_chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(NODES);
    graph.insert_undirected(node(A), node(Y)).unwrap();
    graph.insert_undirected(node(Z), node(Y)).unwrap();
    graph
}

#[test]
fn a_cpdag_whose_members_certify_different_formulas_is_partially_identified() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_201);
        let result = class_study(ClassGraph::Cpdag(open_chain()), &[Z], InferenceMode::Frequentist)
            .run(&ctx)
            .unwrap();
        assert_eq!(result.identification.status, IdentificationStatus::PartiallyIdentified);
        // The weighting reads a certified formula only through its certificate: the identified
        // set over the three cases is the pinned contrast.
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12);
        assert!((effect.se_analytic() - reference_se()).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().expect("an identified-set mixture");
        assert_eq!(mixture.atoms.len(), 3);
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        let set = mixture.identified_set.as_ref().expect("an identified set");
        assert!((set.lower[0] - 2.0).abs() < 1e-12 && (set.upper[0] - 2.0).abs() < 1e-12);
        let rules = result
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code.as_ref() == "estimate.cpdag.transport_sid")
            .expect("the case rules are disclosed");
        assert!(rules.message.contains("transport.sid.direct=2"), "{}", rules.message);
        assert!(rules.message.contains("transport.sid.standardize=1"), "{}", rules.message);
        assert_eq!(result.support_status, Some(CellStatus::Refused));
    });
}

#[test]
fn a_cpdag_whose_members_agree_is_a_point() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_202);
        // No selection target: every member certifies the direct formula.
        let result = class_study(ClassGraph::Cpdag(open_chain()), &[], InferenceMode::Frequentist)
            .run(&ctx)
            .unwrap();
        assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(result.structural_response.is_none(), "a point, not an identified set");
        assert!((result.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);
        let mut prepared =
            class_study(ClassGraph::Cpdag(open_chain()), &[], InferenceMode::Frequentist)
                .prepare(&ctx)
                .unwrap();
        assert!(matches!(prepared.retained_operation(), RetainedOperation::EstimatorRoute));
        let again = prepared.estimate(&transport_data(), &ctx).unwrap();
        assert!((again.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);
        assert!(again.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
    });
}

#[test]
fn a_bayesian_cpdag_transport_publishes_the_trial_row_posterior() {
    on_large_stack(|| {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/bayesian_trial_transport/expected.json"
        ))
        .unwrap();
        let result = class_study(
            ClassGraph::Cpdag(open_chain()),
            &[Z],
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)),
        )
        .run(&ExecutionContext::for_tests(7_203))
        .unwrap();
        let posterior = result.posterior.as_ref().expect("a transport posterior");
        assert!(
            (posterior.summaries.mean[0] - fixture["expected_mean"].as_f64().unwrap()).abs()
                < fixture["mean_tolerance"].as_f64().unwrap()
        );
        assert_eq!(result.identification.status, IdentificationStatus::PartiallyIdentified);
    });
}

#[test]
fn a_pag_is_read_through_the_graphs_its_completions_stand_for() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_204);
        // `a o-o y`: every completion and every ADMG sharing its m-separation model certifies the
        // direct formula when the selection target is not an ancestor of the outcome.
        let mut pag = Pag::with_variables(NODES);
        pag.insert_circle_circle(node(A), node(Y)).unwrap();
        let result =
            class_study(ClassGraph::Pag(pag), &[Z], InferenceMode::Frequentist).run(&ctx).unwrap();
        assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
        assert!((result.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);
        assert!(result.structural_response.is_none());
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code.as_ref() == "estimate.pag.transport_sid")
        );
    });
}

// ---------------------------------------------------------------------------
// Transport over a graph posterior of Cpdag or Pag atoms.
// ---------------------------------------------------------------------------

/// `a -> y <- z` as a Cpdag: the collider is compelled, so the class is the single DAG, and with
/// the covariate `z` as the selection target sID standardizes over it.
fn collider_cpdag() -> Cpdag {
    let mut dag = Dag::with_variables(NODES);
    dag.insert_directed(node(A), node(Y)).unwrap();
    dag.insert_directed(node(Z), node(Y)).unwrap();
    Cpdag::from_dag(&dag)
}

fn cpdag_posterior(weights: &[f64], graphs: &[Cpdag]) -> GraphPosterior {
    let n = NODES as usize;
    let masks: Vec<_> =
        graphs.iter().map(|graph| adjacency_mask_from_cpdag(graph).unwrap()).collect();
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_transport_cpdag"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn pag_posterior(weights: &[f64], graphs: &[Pag]) -> GraphPosterior {
    let n = NODES as usize;
    let mut masks = Vec::new();
    let mut marks = Vec::new();
    for graph in graphs {
        let (mask, mark) = adjacency_masks_from_pag(graph).unwrap();
        masks.push(mask);
        marks.push(mark);
    }
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_transport_pag"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms(marks)
    .unwrap()
}

/// `a o-o y`: every completion and every ADMG sharing its m-separation model certifies the
/// direct formula when the selection target is not an ancestor of the outcome.
fn open_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_circle_circle(node(A), node(Y)).unwrap();
    pag
}

/// `a -> y <- z` as a Pag: its one MAG completion is itself and it standardizes over `z`.
fn collider_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_directed(node(A), node(Y)).unwrap();
    pag.insert_directed(node(Z), node(Y)).unwrap();
    pag
}

/// `a -> z -> y` with `trial <-> s`: the non-singleton c-component leaves sID without a
/// certificate, as it does for the matching ADMG atom of the Dag/Admg posterior.
fn uncertified_pag() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_directed(node(A), node(Z)).unwrap();
    pag.insert_directed(node(Z), node(Y)).unwrap();
    pag.insert_bidirected(node(TRIAL), node(SEL)).unwrap();
    pag
}

/// Each Cpdag atom is run through the class envelope of sID on its own completions: the
/// open chain (two direct completions and one standardizing completion) and the compelled
/// collider (one standardizing completion) are both identified, and the posterior publishes
/// the one transported contrast the weighting evaluates on every certified case.
#[test]
fn a_posterior_of_cpdag_atoms_mixes_the_class_envelope_of_sid_outcomes() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_301);
        let graphs = cpdag_posterior(&[0.7, 0.3], &[open_chain(), collider_cpdag()]);
        let study =
            transport_study(graphs.clone(), &[Z], InferenceMode::Frequentist, RefuteSuite::None);
        let result = study.run(&ctx).unwrap();

        // The pinned closed-form contrast, with the delta-method interval of the weighting.
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12, "{}", effect.ate);
        assert!((effect.se_analytic() - reference_se()).abs() < 1e-12);
        assert_eq!(result.support_status, Some(CellStatus::Refused));

        // Every atom identifies and evaluates: no lost mass, and the value of each atom equals
        // the explicit class route on the same data.
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        assert_eq!(mixture.atoms.len(), 2);
        let explicit =
            class_study(ClassGraph::Cpdag(open_chain()), &[Z], InferenceMode::Frequentist)
                .run(&ctx)
                .unwrap();
        for key in graphs.graph_keys.iter() {
            let value = value_of(&result, *key).expect("an identified atom has a value");
            assert!((value - explicit.estimate.as_effect().unwrap().ate).abs() < 1e-12);
        }
        // The open chain's members certify different formulas, so it is partially identified
        // and so is the mixture that contains it.
        assert_eq!(result.identification.status, IdentificationStatus::PartiallyIdentified);

        // The classes the atoms stand for are disclosed, with the rules sID certified.
        let members = result
            .diagnostics
            .iter()
            .find(|item| item.code.as_ref() == "estimate.graph_posterior.transport_class_members")
            .expect("the class members are disclosed");
        assert!(members.message.contains("2 identified atom(s)"), "{}", members.message);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|item| { item.code.as_ref() == "estimate.graph_posterior.transport_sid" })
        );

        // The prepared handle retains the per-atom class proofs and returns the same answer.
        let mut prepared = study.prepare(&ctx).unwrap();
        let again = prepared.estimate(&transport_data(), &ctx).unwrap();
        assert!((again.estimate.as_effect().unwrap().ate - effect.ate).abs() < 1e-12);
        assert!(again.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
    });
}

/// A Bayesian posterior of Cpdag atoms carries the Dirichlet trial-row posterior of the
/// transported contrast, conditional on the atoms that identify.
#[test]
fn a_bayesian_posterior_of_cpdag_atoms_publishes_the_trial_row_posterior() {
    on_large_stack(|| {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/bayesian_trial_transport/expected.json"
        ))
        .unwrap();
        let study = transport_study(
            cpdag_posterior(&[0.7, 0.3], &[open_chain(), collider_cpdag()]),
            &[Z],
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)),
            RefuteSuite::None,
        );
        let result = study.run(&ExecutionContext::for_tests(7_302)).unwrap();
        let posterior = result.posterior.as_ref().expect("a transport posterior");
        assert_eq!(posterior.draws.n_draws, 2_000);
        assert!(
            (posterior.summaries.mean[0] - fixture["expected_mean"].as_f64().unwrap()).abs()
                < fixture["mean_tolerance"].as_f64().unwrap()
        );
        assert!(posterior.unidentified_mass.abs() < 1e-12);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("transport.trial_bayesian_bootstrap")
        );
    });
}

/// A Pag atom whose ADMG candidates sID does not certify is unidentified posterior mass, the
/// atoms that certify keep their weights, and neither is redistributed.
#[test]
fn a_posterior_of_pag_atoms_states_the_mass_sid_does_not_certify() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(7_303);
        let graphs =
            pag_posterior(&[0.5, 0.25, 0.25], &[open_pag(), uncertified_pag(), collider_pag()]);
        let study =
            transport_study(graphs.clone(), &[Z], InferenceMode::Frequentist, RefuteSuite::None);
        let result = study.run(&ctx).unwrap();
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12, "{}", effect.ate);
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.75).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.25).abs() < 1e-9, "{}", mixture.unidentified_mass);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        let keys = &graphs.graph_keys;
        assert!(value_of(&result, keys[1]).is_none(), "the uncertified atom has no value");
        for certified in [keys[0], keys[2]] {
            let value = value_of(&result, certified).expect("a certified atom has a value");
            assert!((value - 2.0).abs() < 1e-12, "{value}");
        }
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);

        // Moving the posterior mass moves the lost mass by exactly the shift.
        let shifted = transport_study(
            pag_posterior(&[0.1, 0.8, 0.1], &[open_pag(), uncertified_pag(), collider_pag()]),
            &[Z],
            InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .run(&ctx)
        .unwrap();
        let mixture = shifted.structural_response.as_ref().unwrap();
        assert!((mixture.unidentified_mass - 0.8).abs() < 1e-9);
        assert!((mixture.identified_mass - 0.2).abs() < 1e-9);
        assert!((shifted.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-12);
    });
}

/// A posterior of Pag atoms none of which sID certifies is not identified.
#[test]
fn a_posterior_of_pag_atoms_with_no_certified_atom_is_not_identified() {
    on_large_stack(|| {
        let study = transport_study(
            pag_posterior(&[1.0], &[uncertified_pag()]),
            &[Z],
            InferenceMode::Frequentist,
            RefuteSuite::None,
        );
        assert!(study.run(&ExecutionContext::for_tests(7_304)).is_err());
    });
}
