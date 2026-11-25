//! Compile-and-round-trip the registry rows the family loops do not build.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ContinuousDomain, ExecutionContext, ExposureLevel, ExposureMapping, GridSpec,
    InterferenceFunctional, InterferenceQuery, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    ObservationAssumption, PolicyValueQuery, ResponseFunctional, ResponseQuery, SurvivalFunctional,
    SurvivalQuery, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData, TimeSeriesData};
use antecedent_discovery::{BackgroundKnowledge, MultiDatasetConstraints};
use antecedent_graph::{Dag, DenseNodeId};

use crate::discovery::{DiscoverParams, StaticDiscoverParams};
use crate::discovery_defaults::resolve_ci;
use crate::{DiscoverySpec, Study};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied()).unwrap()
}

fn round_trip(study: Study, route: &str) {
    let plan = study
        .compile(&ExecutionContext::for_tests(5))
        .unwrap_or_else(|error| panic!("{route} does not compile: {error}"));
    let lowered = plan.lowered().unwrap_or_else(|| panic!("{route} has no lowered program"));
    assert_eq!(lowered.route(), route);
    crate::planner::lowered::round_trip(lowered)
        .unwrap_or_else(|error| panic!("{route} layers do not decode: {error}"));
}

fn outcome_column() -> TabularData {
    table(&[("outcome", &[0.0, 1.0, 0.0, 1.0])])
}

#[test]
fn the_remaining_design_cells_round_trip() {
    let n = 4;
    round_trip(
        Study::tabular(outcome_column())
            .query(PolicyValueQuery {
                outcome: v(0),
                assignment: Arc::from(vec![false; n]),
                propensity: Arc::from([0.5]),
                actions: Arc::from(vec![false; n]),
                reference: Arc::from(vec![false; n]),
                mu0: Arc::from([]),
                mu1: Arc::from([]),
                costs: Arc::from([0.0]),
                reference_costs: Arc::from([0.0]),
                evaluation_subject_ids: (0..n)
                    .map(|i| Arc::<str>::from(format!("eval-{i}")))
                    .collect(),
                disjoint_training_subjects: false,
                crossfit_fold_ownership_valid: false,
                global_constraints_present: false,
                multi_action: None,
                uplift_bins: Arc::from([]),
                uplift_bin_count: 0,
                uplift_training_subject_ids: Arc::from([]),
                regret: None,
            })
            .build()
            .unwrap(),
        "PolicyValue",
    );

    let doses = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let (mut outcome, mut dose) = (Vec::new(), Vec::new());
    for intercept in [1.0, 10.0] {
        for value in doses {
            outcome.push(intercept + 2.0 * value);
            dose.push(value);
        }
    }
    let groups: Vec<Arc<str>> = ["control", "treated"]
        .into_iter()
        .flat_map(|group| std::iter::repeat_n(Arc::<str>::from(group), 5))
        .collect();
    round_trip(
        Study::tabular(table(&[("y", &outcome), ("dose", &dose), ("density", &vec![0.5; 10])]))
            .query(antecedent_core::ContinuousDoseResponseQuery {
                outcome: v(0),
                dose: v(1),
                dose_density: v(2),
                baseline_groups: groups.into(),
                target_doses: Arc::from([0.0, 0.5]),
                bandwidth: 0.6,
                min_local_support: 3,
                density_provenance: Arc::from("known"),
                fixed_policy: None,
            })
            .build()
            .unwrap(),
        "ContinuousDoseResponse",
    );

    round_trip(
        Study::tabular(table(&[
            ("duration", &[1.0, 2.0, 3.0, 4.0]),
            ("event", &[1.0, 0.0, 1.0, 0.0]),
            ("treatment", &[0.0, 1.0, 0.0, 1.0]),
        ]))
        .query(SurvivalQuery {
            duration: v(0),
            event: v(1),
            treatment: v(2),
            tau: 5.0,
            delayed_entry: None,
            known_censoring: None,
            observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
            functional: SurvivalFunctional::SurvivalAndRmst,
        })
        .build()
        .unwrap(),
        "Survival",
    );

    round_trip(
        Study::tabular(table(&[("outcome", &[0.0, 1.0])]))
            .query(LongitudinalRegimeQuery {
                outcome: v(0),
                method: LongitudinalRegimeMethod::Ipw,
                period_outcome_predictions: None,
                known_fixed_outcome_predictions: false,
                stabilizing_numerator_probabilities: None,
                q_predictions: None,
                observation_history: None,
                prediction_fold_ids: None,
                periods: 2,
                treatment_history: Arc::from([true, false, true, false]),
                regime_actions: Arc::from([true, true, false, false]),
                treatment_probabilities: Arc::from([0.5, 0.5, 0.5, 0.5]),
                censoring_probabilities: Arc::from([1.0, 1.0, 1.0, 1.0]),
                outcome_observed: Arc::from([true, true]),
                subject_ids: Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]),
                fold_ids: Arc::from([0_u32, 1]),
                excluded_fold_predictions: false,
                probabilities_known_by_design: true,
                minimum_probability: 0.01,
                rule_id: None,
                rule_version: None,
                rule_provenance: None,
            })
            .build()
            .unwrap(),
        "LongitudinalRegime",
    );

    let mut graph = Dag::with_variables(6);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let columns = table(&[
        ("t", &[0.0, 1.0, 0.0, 1.0]),
        ("y", &[1.0, 2.0, 1.5, 2.5]),
        ("z", &[0.2, 0.4, 0.6, 0.8]),
        ("trial", &[1.0, 1.0, 0.0, 0.0]),
        ("selection", &[0.8, 0.7, 0.6, 0.5]),
        ("propensity", &[0.4, 0.6, 0.4, 0.6]),
    ]);
    round_trip(
        Study::tabular(columns.clone())
            .graph(graph.clone())
            .query(TransportQuery::new(response, "source", "target", [v(0)]))
            .selection_targets(Arc::from([v(2)]))
            .transport_trial(crate::TransportTrialSpec {
                trial: v(3),
                selection_probability: v(4),
                treatment_probability: v(5),
            })
            .build()
            .unwrap(),
        "TransportMixed",
    );
    let outcomes = [1.0, 2.0];
    let interference_data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(
        interference_data.clone(),
        vec![NetworkEdge::new(0, 1, 1.0), NetworkEdge::new(1, 0, 1.0)],
    )
    .unwrap();
    round_trip(
        Study::tabular(interference_data)
            .graph(Dag::with_variables(1))
            .query(InterferenceQuery::new(
                AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5, 0.4]) },
                ExposureMapping::OwnTreatment,
                InterferenceFunctional::ExposureContrast {
                    outcome: v(0),
                    from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                    to: ExposureLevel { own: 1.0, neighbors: 0.0 },
                },
            ))
            .interference(crate::InterferenceSpec { network, assignment: Arc::from([false, true]) })
            .build()
            .unwrap(),
        "InterferenceDag",
    );
}

#[test]
fn the_discovery_cells_round_trip() {
    let ci = resolve_ci("parcorr", None).unwrap();
    let static_params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: ci.clone(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    };
    let data = table(&[
        ("z", &[0.0, 1.0, 0.2, 0.8]),
        ("t", &[0.1, 0.9, 0.3, 0.7]),
        ("y", &[0.2, 1.1, 0.4, 0.9]),
    ]);
    let vars = vec![v(0), v(1), v(2)];
    round_trip(
        Study::tabular(data)
            .discovery(DiscoverySpec::pc(static_params, "parcorr"), vars)
            .build()
            .unwrap(),
        "DiscoverStatic",
    );

    let x: Vec<f64> = (0..20).map(|i| f64::from(i % 5)).collect();
    let y: Vec<f64> = x.iter().map(|value| value + 1.0).collect();
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1_000)
            .unwrap();
    let temporal = DiscoverParams {
        max_lag: 1,
        alpha: 0.05,
        fdr: None,
        ci,
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size: 2,
        knowledge: BackgroundKnowledge::default(),
    };
    round_trip(
        Study::series(series)
            .discovery(DiscoverySpec::pcmci_plus(temporal, "parcorr"), vec![v(0), v(1)])
            .build()
            .unwrap(),
        "DiscoverTemporal",
    );
}
