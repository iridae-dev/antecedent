//! Inline tests of the support-matrix lookup.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalCpdag, TemporalPag};

use super::matrix::refusal_reason;
use super::*;
use crate::accepted::{AcceptedGraph, GraphClass};
use crate::analysis::RefuteSuite;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::strategy_table::EstimatorId;

fn cell(
    query: &'static str,
    graph: &'static str,
    structure: &'static str,
    inference: &'static str,
    validation: &'static str,
) -> SupportCell {
    SupportCell::from_labels(query, graph, structure, inference, validation).unwrap_or_else(|| {
        panic!("support axis {query}:{graph}:{structure}:{inference}:{validation}")
    })
}

#[test]
fn dynamic_policy_classifies_by_schedule_shape() {
    use antecedent_core::{DynamicRuleId, TemporalEffectQuery, TemporalPolicy, VariableId};
    let make = |active_at: &[i32]| {
        let mut q =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        q.policy = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), active_at.to_vec());
        CausalQuery::TemporalEffect(q)
    };
    assert_eq!(query_axis_name(&make(&[0]), GraphClass::TemporalDag), Some("PulseEffect"));
    // A multi-step dynamic schedule is a sustained intervention; it must
    // hit the Sustained closure, not bypass the matrix.
    assert_eq!(query_axis_name(&make(&[0, 1]), GraphClass::TemporalDag), Some("SustainedEffect"));
}

#[test]
fn declared_off_axis_kinds_are_the_only_axisless_queries_that_build() {
    use antecedent_core::{TemporalEffectQuery, UnitChangeQuery, VariableId};
    // Unit change has no matrix cell for any class, and is on the closed allow-list.
    let unit = CausalQuery::UnitChange(UnitChangeQuery::new(VariableId::from_raw(1), 20));
    for class in [GraphClass::Dag, GraphClass::TemporalDag] {
        assert_eq!(query_axis_name(&unit, class), None);
    }
    assert!(is_declared_off_axis(&unit));
    assert!(refuse_undeclared_off_axis(&unit).is_ok());
    // An on-axis query never needs the allow-list.
    let pulse = CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        1.0,
    ));
    assert!(!is_declared_off_axis(&pulse));
    assert!(refuse_undeclared_off_axis(&pulse).is_ok());
}

#[test]
fn pulse_on_static_dag_is_not_applicable() {
    let status = classify(cell("PulseEffect", "Dag", "explicit", "Frequentist", "none"));
    assert!(matches!(status, CellStatus::NotApplicable { .. }));
}

#[test]
fn classify_estimator_licenses_calibration_and_route_estimators() {
    let c = cell("AverageEffect", "Dag", "explicit", "Frequentist", "none");
    assert_eq!(classify(c), CellStatus::Licensed);
    let named = licensed_estimators(c);
    assert!(named.contains(&"linear.adjustment.ate"), "{named:?}");
    // Route default.
    assert_eq!(licensed_route_estimator(c), Some(EstimatorId::LinearAdjustmentAte));
    // Calibration-record tokens on this row (plus the route).
    for est in [
        EstimatorId::LinearAdjustmentAte,
        EstimatorId::Aipw,
        EstimatorId::PropensityWeighting,
        EstimatorId::GlmAdjustment,
        EstimatorId::FrontDoorTwoStage,
        EstimatorId::IvWald,
        EstimatorId::Iv2Sls,
        EstimatorId::PropensityMatching,
        EstimatorId::RdSharp,
    ] {
        assert_eq!(
            classify_estimator(c, est),
            CellStatus::Licensed,
            "{est:?} is named by this cell's route or calibration ids; got refuse"
        );
    }
    // No record id names these on this cell.
    for est in [EstimatorId::DistanceMatching, EstimatorId::CausalForest] {
        assert_eq!(
            classify_estimator(c, est),
            CellStatus::Refused,
            "{est:?} must stay refused without a calibration token"
        );
    }
}

#[test]
fn classify_estimator_never_allowlists_unmeasured_estimators() {
    let c = cell("AverageEffect", "Dag", "explicit", "Frequentist", "none");
    let status = classify_estimator(c, EstimatorId::CausalForest);
    assert_eq!(status, CellStatus::Refused);
    assert!(!matches!(status, CellStatus::Allowlisted { .. }));
}

#[test]
fn classify_estimator_refuses_estimator_with_no_route_or_token() {
    // Stratification has neither a licensed_routes entry nor a calibration
    // token on the AverageEffect Dag frequentist/none cell.
    let c = cell("AverageEffect", "Dag", "explicit", "Frequentist", "none");
    assert_eq!(classify_estimator(c, EstimatorId::PropensityStratification), CellStatus::Refused);
}

#[test]
fn licensed_neighbors_keep_graph_class_and_never_relabel() {
    let pulse = cell("PulseEffect", "Dag", "explicit", "Frequentist", "none");
    let neighbors = licensed_neighbors(pulse);
    assert!(!neighbors.is_empty(), "Pulse on Dag should name a static licensed neighbor");
    assert!(
        neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Dag:")),
        "{neighbors:?}"
    );
    assert!(
        neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Dag")),
        "graph-class relabel is not a neighbor: {neighbors:?}"
    );

    let cpdag = cell("PulseEffect", "Cpdag", "explicit", "Frequentist", "none");
    let neighbors = licensed_neighbors(cpdag);
    assert!(
        neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Cpdag")),
        "Cpdag must not recommend a Dag relabel: {neighbors:?}"
    );
    assert!(neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Cpdag:")));

    let licensed_admg = cell("ConditionalEffect", "Admg", "explicit", "Frequentist", "none");
    assert!(
        licensed_neighbors(licensed_admg).is_empty(),
        "a licensed ADMG conditional effect has no neighbor"
    );
    let refused = cell("PathSpecificEffect", "Admg", "explicit", "Frequentist", "none");
    let neighbors = licensed_neighbors(refused);
    assert!(neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Admg:")));
    assert!(neighbors.iter().all(|n| !n.coordinate.as_ref().contains(":Dag:")));

    assert!(
        licensed_neighbors(cell("AverageEffect", "Dag", "explicit", "Frequentist", "none"))
            .is_empty()
    );
}

#[test]
fn average_effect_on_a_temporal_graph_is_the_executing_lag_zero_cell() {
    for graph in ["TemporalDag", "TemporalCpdag", "TemporalPag"] {
        for inference in ["Frequentist", "Bayesian"] {
            for validation in ["none", "cheap", "full"] {
                let c = cell("AverageEffect", graph, "explicit", inference, validation);
                assert!(matches!(classify(c), CellStatus::Refused), "{c:?}");
                assert!(super::executes_unlicensed(c), "{c:?}");
            }
        }
        let posterior = cell("AverageEffect", graph, "graph_posterior", "Frequentist", "none");
        assert!(matches!(classify(posterior), CellStatus::Refused));
        assert!(!super::executes_unlicensed(posterior));
        let conditional = cell("ConditionalEffect", graph, "explicit", "Frequentist", "none");
        assert!(matches!(classify(conditional), CellStatus::NotApplicable { .. }));
    }
}

#[test]
fn temporal_mediation_on_temporal_dag_is_not_n_a() {
    let status =
        classify(cell("TemporalMediationEffect", "TemporalDag", "explicit", "Frequentist", "none"));
    assert!(!matches!(status, CellStatus::NotApplicable { .. }));
}

#[test]
fn dag_average_effect_frequentist_none_is_licensed() {
    assert_eq!(
        classify(cell("AverageEffect", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("AverageEffect", "Dag", "accepted", "Frequentist", "none")),
        CellStatus::Licensed
    );
}

#[test]
fn dag_response_curve_frequentist_none_is_licensed() {
    assert_eq!(
        classify(cell("ResponseCurve", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("ResponseCurve", "Dag", "accepted", "Frequentist", "none")),
        CellStatus::Licensed
    );
}

#[test]
fn response_curve_graph_posterior_is_licensed_only_for_static_dag_atoms() {
    for graph in ["Dag", "Cpdag", "Pag", "Admg", "TemporalDag", "TemporalCpdag", "TemporalPag"] {
        let status =
            classify(cell("ResponseCurve", graph, "graph_posterior", "Frequentist", "none"));
        assert_eq!(status, CellStatus::Licensed, "{graph}: {status:?}");
    }
    for inference in ["Frequentist", "Bayesian"] {
        for graph in ["TemporalDag", "TemporalCpdag", "TemporalPag"] {
            for validation in ["none", "cheap", "full"] {
                let status = classify(cell(
                    "InterventionResponse",
                    graph,
                    "graph_posterior",
                    inference,
                    validation,
                ));
                assert_eq!(
                    status,
                    CellStatus::Licensed,
                    "{graph}/{inference}/{validation}: {status:?}"
                );
            }
            let status =
                classify(cell("ResponseCurve", graph, "graph_posterior", inference, "none"));
            assert_eq!(status, CellStatus::Licensed, "{graph}/{inference}: {status:?}");
            for validation in ["cheap", "full"] {
                let status = classify(cell(
                    "ResponseCurve",
                    graph,
                    "graph_posterior",
                    inference,
                    validation,
                ));
                assert!(
                    matches!(status, CellStatus::NotApplicable { .. }),
                    "{graph}/{inference}/{validation}: {status:?}"
                );
            }
        }
    }
}

#[test]
fn response_curve_cheap_on_dag_is_not_applicable() {
    let status = classify(cell("ResponseCurve", "Dag", "explicit", "Frequentist", "cheap"));
    assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
}

#[test]
fn dag_intervention_response_cheap_and_full_are_licensed() {
    for structure in ["explicit", "accepted"] {
        for validation in ["cheap", "full"] {
            let status =
                classify(cell("InterventionResponse", "Dag", structure, "Frequentist", validation));
            assert_eq!(status, CellStatus::Licensed, "{structure}/{validation}");
            refuse_if_not_applicable(cell(
                "InterventionResponse",
                "Dag",
                structure,
                "Frequentist",
                validation,
            ))
            .unwrap();
        }
    }
    for graph in ["TemporalDag", "Cpdag", "Pag", "TemporalCpdag", "TemporalPag"] {
        let status =
            classify(cell("InterventionResponse", graph, "explicit", "Frequentist", "cheap"));
        assert_eq!(status, CellStatus::Refused, "{graph}: {status:?}");
    }
}

#[test]
fn function_valued_and_class_response_cheap_are_not_applicable() {
    for graph in ["TemporalCpdag", "TemporalPag", "Admg"] {
        let status = classify(cell("ResponseCurve", graph, "explicit", "Frequentist", "cheap"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }), "{graph}: {status:?}");
    }
    let status = classify(cell("ResponseCurve", "Dag", "graph_posterior", "Frequentist", "full"));
    assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
    let status =
        classify(cell("ResponseCurve", "TemporalDag", "graph_posterior", "Frequentist", "full"));
    assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
    // The Bayesian intervention-level refuter state is a refusal that executes
    // unlicensed, not a typed impossibility.
    for structure in ["explicit", "accepted", "graph_posterior"] {
        for validation in ["cheap", "full"] {
            let c = cell("InterventionResponse", "Dag", structure, "Bayesian", validation);
            assert_eq!(classify(c), CellStatus::Refused, "{c:?}");
            assert!(super::executes_unlicensed(c), "{c:?}");
        }
    }
}

#[test]
fn refusals_formerly_recorded_as_not_applicable_execute_or_name_their_next_attempt() {
    // Built refuter states: refused (unlicensed) cells that execute.
    for query in ["PointDerivative", "AverageDerivative", "ResponseJacobian", "Elasticity"] {
        for inference in ["Frequentist", "Bayesian"] {
            let c = cell(query, "Dag", "explicit", inference, "cheap");
            assert_eq!(classify(c), CellStatus::Refused, "{c:?}");
            assert!(super::executes_unlicensed(c), "{c:?}");
        }
    }
    for graph in ["TemporalDag", "Cpdag", "Pag"] {
        let c = cell("InterventionResponse", graph, "explicit", "Frequentist", "full");
        assert!(super::executes_unlicensed(c), "{graph}");
    }
    let c = cell("InterventionResponse", "CoDetermined", "explicit", "Bayesian", "none");
    assert!(super::executes_unlicensed(c), "{c:?}");
    // The temporal sides execute too.
    for c in [
        cell("InterventionResponse", "TemporalCpdag", "explicit", "Frequentist", "full"),
        cell("InterventionResponse", "TemporalPag", "accepted", "Bayesian", "cheap"),
        cell("PointDerivative", "TemporalDag", "explicit", "Frequentist", "cheap"),
        cell("ResponseJacobian", "TemporalPag", "accepted", "Bayesian", "full"),
    ] {
        assert_eq!(classify(c), CellStatus::Refused, "{c:?}");
        assert!(super::executes_unlicensed(c), "{c:?}");
    }
    // Genuine impossibilities stay n/a.
    for c in [
        cell("PulseEffect", "Dag", "explicit", "Frequentist", "none"),
        cell("InterventionResponse", "Unknown", "explicit", "Bayesian", "none"),
        cell("InterventionResponse", "CoDetermined", "accepted", "Bayesian", "none"),
        cell("ResponseCurve", "Dag", "explicit", "Bayesian", "cheap"),
    ] {
        assert!(matches!(classify(c), CellStatus::NotApplicable { .. }), "{c:?}");
        assert!(!super::executes_unlicensed(c), "{c:?}");
    }
}

#[test]
fn frequentist_dbn_and_cpdag_mediation_suites_are_licensed() {
    for validation in ["cheap", "full"] {
        for query in ["PulseEffect", "SustainedEffect", "TemporalMediationEffect"] {
            let c = cell(query, "TemporalDag", "graph_posterior", "Frequentist", validation);
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
        }
        for structure in ["explicit", "accepted"] {
            let c = cell(
                "TemporalMediationEffect",
                "TemporalCpdag",
                structure,
                "Frequentist",
                validation,
            );
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
        }
        let c = cell("InterventionResponse", "Dag", "graph_posterior", "Frequentist", validation);
        assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
    }
    for graph in ["TemporalCpdag", "TemporalPag"] {
        for query in ["PulseEffect", "SustainedEffect"] {
            for inference in ["Frequentist", "Bayesian"] {
                for validation in ["none", "cheap", "full"] {
                    let c = cell(query, graph, "graph_posterior", inference, validation);
                    assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
                }
            }
        }
    }
    assert_eq!(
        classify(cell(
            "TemporalMediationEffect",
            "TemporalCpdag",
            "graph_posterior",
            "Frequentist",
            "none",
        )),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell(
            "TemporalMediationEffect",
            "TemporalPag",
            "graph_posterior",
            "Frequentist",
            "none",
        )),
        CellStatus::Refused
    );
}

#[test]
fn pag_average_effect_is_licensed() {
    assert_eq!(
        classify(cell("AverageEffect", "Pag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("AverageEffect", "Cpdag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("AverageEffect", "Cpdag", "accepted", "Bayesian", "full")),
        CellStatus::Licensed
    );
    assert_eq!(
        refuse_if_not_applicable(cell("AverageEffect", "Pag", "explicit", "Frequentist", "none"))
            .unwrap(),
        CellStatus::Licensed
    );
}

#[test]
fn closed_derivative_and_counterfactual_are_enforced() {
    let err =
        refuse_if_not_applicable(cell("Elasticity", "Pag", "explicit", "Frequentist", "none"))
            .unwrap_err();
    assert!(err.to_string().starts_with("refused:"), "{err}");
    let err =
        refuse_if_not_applicable(cell("Counterfactual", "Pag", "explicit", "Frequentist", "none"))
            .unwrap_err();
    assert!(err.to_string().starts_with("refused:"), "{err}");
    assert_eq!(
        classify(cell("ResponseCurve", "Pag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
}

#[test]
fn remaining_closed_dag_cells_are_named() {
    let cases = [
        (
            "AverageDerivative",
            "graph_posterior",
            "Bayesian",
            "none",
            "Graph-posterior derivative mixtures are not staged",
        ),
        ("Counterfactual", "graph_posterior", "Frequentist", "none", "graph-posterior"),
        (
            "Counterfactual",
            "explicit",
            "Frequentist",
            "cheap",
            "Counterfactual cheap/full are not licensed",
        ),
    ];
    for (query, structure, inference, validation, needle) in cases {
        let err = refuse_if_not_applicable(cell(query, "Dag", structure, inference, validation))
            .unwrap_err();
        let text = err.to_string();
        assert!(text.starts_with("refused:"), "{query}: {text}");
        assert!(text.contains(needle), "{query}: {text}");
    }
    assert_eq!(
        classify(cell("PointDerivative", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("MediationEffect", "Dag", "accepted", "Frequentist", "full")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("Counterfactual", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("Counterfactual", "Dag", "accepted", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("Counterfactual", "Dag", "accepted", "Bayesian", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("AnomalyAttribution", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("ChangeAttribution", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("TransportQuery", "Admg", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("InterferenceQuery", "Dag", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
}

#[test]
fn accepted_path_and_distribution_are_licensed() {
    for query in ["PathSpecificEffect", "InterventionalDistribution"] {
        for validation in ["none", "cheap", "full"] {
            let c = cell(query, "Dag", "accepted", "Frequentist", validation);
            assert_eq!(classify(c), CellStatus::Licensed);
            refuse_if_not_applicable(c).unwrap();
        }
        assert_eq!(
            classify(cell(query, "Dag", "graph_posterior", "Frequentist", "none")),
            CellStatus::Refused
        );
    }
}

#[test]
fn closed_mediation_is_enforced() {
    let err =
        refuse_if_not_applicable(cell("MediationEffect", "Pag", "explicit", "Frequentist", "none"))
            .unwrap_err();
    assert!(err.to_string().starts_with("refused:"), "{err}");
    for query in ["TransportQuery", "InterferenceQuery"] {
        let err = refuse_if_not_applicable(cell(query, "Pag", "explicit", "Frequentist", "none"))
            .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{query}: {err}");
    }
}

#[test]
fn licensed_pulse_temporal_class_is_open() {
    for query in ["PulseEffect", "SustainedEffect"] {
        for graph in ["TemporalCpdag", "TemporalPag"] {
            for structure in ["explicit", "accepted"] {
                let status = classify(cell(query, graph, structure, "Frequentist", "none"));
                assert_eq!(status, CellStatus::Licensed, "{query}/{graph}/{structure}");
                refuse_if_not_applicable(cell(query, graph, structure, "Frequentist", "none"))
                    .unwrap();
                for validation in ["none", "cheap", "full"] {
                    assert_eq!(
                        refuse_if_not_applicable(cell(
                            query, graph, structure, "Bayesian", validation
                        ))
                        .unwrap(),
                        CellStatus::Licensed,
                    );
                }
            }
        }
    }
}

#[test]
fn licensed_pulse_and_sustained_temporal_dag_are_open() {
    for query in ["PulseEffect", "SustainedEffect"] {
        for structure in ["explicit", "accepted"] {
            for inference in ["Frequentist", "Bayesian"] {
                let status = classify(cell(query, "TemporalDag", structure, inference, "none"));
                assert_eq!(status, CellStatus::Licensed, "{query}/{structure}/{inference}");
                refuse_if_not_applicable(cell(query, "TemporalDag", structure, inference, "none"))
                    .unwrap();
            }
        }
    }
}

#[test]
fn closed_conditional_effect_off_dag_is_enforced() {
    for structure in ["explicit", "accepted"] {
        for inference in ["Frequentist", "Bayesian"] {
            let status = classify(cell("ConditionalEffect", "Admg", structure, inference, "none"));
            assert_eq!(status, CellStatus::Licensed, "Admg/{structure}/{inference}");
            refuse_if_not_applicable(cell(
                "ConditionalEffect",
                "Admg",
                structure,
                inference,
                "none",
            ))
            .unwrap();
        }
    }
    let temporal = cell("ConditionalEffect", "TemporalAdmg", "accepted", "Frequentist", "none");
    assert_eq!(classify(temporal), CellStatus::Refused, "TemporalAdmg");
    let err = refuse_if_not_applicable(temporal).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("TemporalConditionalPulse"), "TemporalAdmg: {message}");
    assert!(message.contains("static Admg"), "TemporalAdmg: {message}");
    for graph in ["Cpdag", "Pag"] {
        for inference in ["Frequentist", "Bayesian"] {
            for validation in ["none", "cheap", "full"] {
                let status =
                    classify(cell("ConditionalEffect", graph, "accepted", inference, validation));
                assert_eq!(status, CellStatus::Licensed, "{graph}/{inference}/{validation}");
                refuse_if_not_applicable(cell(
                    "ConditionalEffect",
                    graph,
                    "accepted",
                    inference,
                    validation,
                ))
                .unwrap();
            }
        }
    }
}

#[test]
fn closed_path_and_distribution_on_explicit_admg_pag_is_enforced() {
    for graph in ["Admg", "Pag"] {
        let status = classify(cell("PathSpecificEffect", graph, "explicit", "Frequentist", "none"));
        assert_eq!(status, CellStatus::Refused, "PathSpecificEffect/{graph}");
        let err = refuse_if_not_applicable(cell(
            "PathSpecificEffect",
            graph,
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(
            err.to_string().starts_with(
                "refused: Path-specific queries execute only on a supplied static Dag"
            ),
            "PathSpecificEffect/{graph}: {err}"
        );
    }
    assert_eq!(
        classify(cell("InterventionalDistribution", "Admg", "explicit", "Frequentist", "none")),
        CellStatus::Licensed
    );
    assert_eq!(
        classify(cell("InterventionalDistribution", "Pag", "explicit", "Frequentist", "none")),
        CellStatus::Refused
    );
    for validation in ["cheap", "full"] {
        assert_eq!(
            classify(cell(
                "InterventionalDistribution",
                "Admg",
                "explicit",
                "Frequentist",
                validation
            )),
            CellStatus::Refused,
            "Admg distribution {validation}"
        );
    }
}

#[test]
fn closed_intervention_response_off_dag_is_enforced() {
    for inference in ["Frequentist", "Bayesian"] {
        for structure in ["explicit", "accepted"] {
            assert_eq!(
                classify(cell("InterventionResponse", "Admg", structure, inference, "none")),
                CellStatus::Licensed,
                "{structure}/{inference}"
            );
            refuse_if_not_applicable(cell(
                "InterventionResponse",
                "Admg",
                structure,
                inference,
                "none",
            ))
            .unwrap();
            assert_eq!(
                classify(cell("ResponseCurve", "Admg", structure, inference, "none")),
                CellStatus::Licensed,
                "curve {structure}/{inference}"
            );
        }
        for validation in ["cheap", "full"] {
            for structure in ["explicit", "accepted"] {
                let open = cell("InterventionResponse", "Admg", structure, inference, validation);
                assert_eq!(
                    classify(open),
                    CellStatus::Licensed,
                    "Admg {structure}/{inference}/{validation}"
                );
                refuse_if_not_applicable(open).unwrap();
            }
        }
    }
    for inference in ["Frequentist", "Bayesian"] {
        for validation in ["none", "cheap", "full"] {
            let open =
                cell("InterventionResponse", "Admg", "graph_posterior", inference, validation);
            assert_eq!(
                classify(open),
                CellStatus::Licensed,
                "Admg graph_posterior/{inference}/{validation}"
            );
            refuse_if_not_applicable(open).unwrap();
        }
    }
    for graph in ["Cpdag", "Pag"] {
        for inference in ["Frequentist", "Bayesian"] {
            assert_eq!(
                classify(cell("InterventionResponse", graph, "explicit", inference, "none")),
                CellStatus::Licensed,
                "{graph}/{inference}"
            );
            refuse_if_not_applicable(cell(
                "InterventionResponse",
                graph,
                "explicit",
                inference,
                "none",
            ))
            .unwrap();
        }
    }
}

#[test]
fn temporal_mediation_class_boundary_is_enforced() {
    for graph in ["TemporalCpdag", "TemporalPag"] {
        let status =
            classify(cell("TemporalMediationEffect", graph, "accepted", "Frequentist", "none"));
        assert_eq!(
            status,
            if graph == "TemporalCpdag" { CellStatus::Licensed } else { CellStatus::Refused },
            "{graph}"
        );
    }
}

#[test]
fn licensed_graph_posterior_frequentist_ate_is_open() {
    for validation in ["none", "cheap", "full"] {
        let c = cell("AverageEffect", "Dag", "graph_posterior", "Frequentist", validation);
        assert_eq!(classify(c), CellStatus::Licensed, "{validation}");
        refuse_if_not_applicable(c).unwrap();
    }
    for graph in ["Cpdag", "Pag"] {
        for inference in ["Frequentist", "Bayesian"] {
            for validation in ["none", "cheap", "full"] {
                let open = cell("AverageEffect", graph, "graph_posterior", inference, validation);
                assert_eq!(
                    classify(open),
                    CellStatus::Licensed,
                    "{graph}/{inference}/{validation}"
                );
                refuse_if_not_applicable(open).unwrap();
            }
        }
    }
    for inference in ["Frequentist", "Bayesian"] {
        for validation in ["none", "cheap", "full"] {
            let open = cell("AverageEffect", "Admg", "graph_posterior", inference, validation);
            assert_eq!(classify(open), CellStatus::Licensed, "Admg/{inference}/{validation}");
            refuse_if_not_applicable(open).unwrap();
        }
    }
    for inference in ["Frequentist", "Bayesian"] {
        for validation in ["none", "cheap", "full"] {
            let c = cell("ConditionalEffect", "Dag", "graph_posterior", inference, validation);
            assert_eq!(classify(c), CellStatus::Licensed);
            refuse_if_not_applicable(c).unwrap();
        }
    }
}

#[test]
fn class_and_mediation_graph_posterior_refusals_are_named() {
    for graph in ["Cpdag", "Pag"] {
        for inference in ["Frequentist", "Bayesian"] {
            for validation in ["none", "cheap", "full"] {
                let open =
                    cell("ConditionalEffect", graph, "graph_posterior", inference, validation);
                assert_eq!(
                    classify(open),
                    CellStatus::Licensed,
                    "{graph}/{inference}/{validation}"
                );
                refuse_if_not_applicable(open).unwrap();
            }
        }
    }
    for validation in ["none", "cheap", "full"] {
        let err = refuse_if_not_applicable(cell(
            "MediationEffect",
            "Dag",
            "graph_posterior",
            "Bayesian",
            validation,
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("Graph-posterior mediation mixtures are not staged"),
            "{validation}: {err}"
        );
    }
}

/// End-to-end: `Study::build` accepts the licensed Frequentist graph-posterior ATE cell.
#[test]
fn build_accepts_graph_posterior_under_frequentist() {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use std::sync::Arc;

    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [("t", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let n = 10;
    // Non-degenerate columns: Gaussian BIC no longer floors residual
    // variance, so an all-zero table scores every DAG as non-finite.
    let t: Vec<f64> =
        (0..n).map(|i| f64::from(u32::try_from(i % 2).expect("tiny fixture"))).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let i = u32::try_from(i).expect("tiny fixture");
            2.0 * f64::from(i % 2) + 0.1 * f64::from(i)
        })
        .collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    let gp = crate::discovery::discover_exact_dag_posterior(
        &data,
        &[VariableId::from_raw(0), VariableId::from_raw(1)],
        &crate::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();

    crate::analysis::Study::tabular(data)
        .graph_posterior(gp)
        .query(ate_query())
        .inference(crate::inference::InferenceMode::Frequentist)
        .build()
        .expect("AverageEffect × graph_posterior × Frequentist is licensed");
}

/// End-to-end class-aware static case: `Study::build` accepts a `ConditionalEffect`
/// query on a supplied Cpdag once the cell is licensed.
#[test]
fn build_accepts_conditional_effect_on_cpdag() {
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use std::sync::Arc;

    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("z", RoleHint::Context),
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let n = 10;
    let z = vec![0.0; n];
    let t = vec![0.0; n];
    let y = vec![0.0; n];
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);

    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();

    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
        .with_effect_modifiers(vec![VariableId::from_raw(0)]);
    let query = antecedent_core::ConditionalEffectQuery::try_new(inner).unwrap();

    crate::analysis::Study::tabular(data)
        .graph(AcceptedGraph::cpdag(cpdag).unwrap())
        .query(query)
        .build()
        .expect("ConditionalEffect × Cpdag is licensed");
}

// -- `effective_graph_class` --------------------------------------------

use antecedent_core::{AverageEffectQuery, TemporalEffectQuery, VariableId};

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ))
}

fn pulse_query() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        1.0,
    ))
}

fn distribution_query() -> CausalQuery {
    use antecedent_core::{Intervention, InterventionalDistributionQuery, Value};
    CausalQuery::Distribution(InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    ))
}

/// (a) An ADMG with no bidirected edges collapses to Dag under `AverageEffect`,
/// and the resulting cell is Licensed — the same Dag/explicit/Frequentist/none
/// row every supplied-DAG study licenses.
#[test]
fn admg_without_bidirected_collapses_to_dag_under_average_effect() {
    let mut admg = Admg::with_variables(2);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::from(admg);
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Dag);

    let sc = support_cell(
        &ate_query(),
        effective_graph_class(&graph, &ate_query()),
        StructureSource::Explicit,
        &InferenceMode::Frequentist,
        RefuteSuite::None,
    )
    .unwrap();
    assert_eq!(classify(sc), CellStatus::Licensed);
}

/// (b) An ADMG *with* a bidirected edge does not collapse: the ADMG path is
/// still live (dispatch runs `execute_admg`, not the static-DAG completion),
///         so the cell stays the Admg cell — licensed Frequentist ATE, not collapsed to Dag.
#[test]
fn admg_with_bidirected_edge_does_not_collapse() {
    let mut admg = Admg::with_variables(2);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::from(admg);
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Admg);

    let sc = support_cell(
        &ate_query(),
        effective_graph_class(&graph, &ate_query()),
        StructureSource::Explicit,
        &InferenceMode::Frequentist,
        RefuteSuite::None,
    )
    .unwrap();
    assert_eq!(classify(sc), CellStatus::Licensed);
}

#[test]
fn codetermined_joint_ir_is_a_licensed_matrix_cell() {
    let sc = cell("InterventionResponse", "CoDetermined", "explicit", "Frequentist", "none");
    assert_eq!(classify(sc), CellStatus::Licensed);
    let unknown = cell("AverageEffect", "Unknown", "explicit", "Frequentist", "none");
    assert_eq!(classify(unknown), CellStatus::Licensed);
    let off = cell("ConditionalEffect", "Unknown", "explicit", "Frequentist", "none");
    assert!(matches!(classify(off), CellStatus::NotApplicable { .. }));
}

/// The ADMG collapse is scoped to `AverageEffect`. Distribution on Admg is a
/// separate licensed cell (general ID, no DAG coercion); collapse must not
/// rewrite that class to Dag.
#[test]
fn admg_collapse_does_not_apply_outside_average_effect() {
    let mut admg = Admg::with_variables(2);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::from(admg);
    assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Admg);
}

/// A supplied Cpdag stays Cpdag under `AverageEffect`, including when it
/// has a unique orientation. Completing it to a `Dag` is a different cell.
#[test]
fn cpdag_does_not_collapse_under_average_effect() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::from(cpdag);
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Cpdag);
}

/// Undirected marks are MEC information: `AcceptedGraph::cpdag` accepts them.
#[test]
fn undirected_cpdag_is_accepted_and_stays_cpdag() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::cpdag(cpdag).unwrap();
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Cpdag);
}

/// Response / distribution queries on a Cpdag stay Cpdag (and refused):
/// there is no compile arm that would license them via collapse.
#[test]
fn cpdag_does_not_collapse_outside_average_effect() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::from(cpdag);
    assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Cpdag);
}

/// Fully oriented temporal CPDAGs retain their supplied class.
#[test]
fn complete_temporal_cpdag_retains_class_under_temporal_effect() {
    let mut cpdag = TemporalCpdag::empty();
    let a = cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let b =
        cpdag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_directed(a, b).unwrap();
    let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
    assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalCpdag);
}

/// An incomplete `TemporalCpdag` is accepted: undirected marks are the MEC.
#[test]
fn incomplete_temporal_cpdag_is_accepted_and_stays_temporal_cpdag() {
    let mut cpdag = TemporalCpdag::empty();
    let a = cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let b =
        cpdag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_undirected(a, b).unwrap();
    let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
    assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalCpdag);
}

/// Fully oriented temporal PAGs still require MAG adjustment certification.
#[test]
fn complete_temporal_pag_retains_class_under_temporal_effect() {
    use antecedent_graph::TemporalPag;
    let mut pag = TemporalPag::empty();
    let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let b = pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_directed(a, b).unwrap();
    let graph = AcceptedGraph::temporal_pag(pag);
    assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalPag);
}

/// A `TemporalPag` with a circle mark is accepted: circles are the incomplete class.
#[test]
fn temporal_pag_with_circle_mark_is_accepted_and_stays_temporal_pag() {
    let mut pag = TemporalPag::empty();
    let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let b = pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_circle_arrow(a, b).unwrap();
    let graph = AcceptedGraph::temporal_pag(pag);
    assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalPag);
}

/// The temporal collapse is scoped to `TemporalEffect`: `compile.rs` wires no
/// other query against `GraphClass::TemporalCpdag`/`TemporalPag` (temporal
/// `Mediation` only has a `GraphClass::TemporalDag` arm), so it must not fire
/// for e.g. a static query landing on a temporal class.
#[test]
fn temporal_collapse_does_not_apply_outside_temporal_effect() {
    let mut cpdag = TemporalCpdag::empty();
    let a = cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
    let b =
        cpdag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_directed(a, b).unwrap();
    let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
    // AverageEffect on a temporal class is typed-impossible regardless of
    // collapse, but the collapse itself must still not fire here.
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::TemporalCpdag);
}

/// A static Pag never collapses: circle marks are information the
/// class-aware generalized-adjustment identifier consumes, not incompleteness.
#[test]
fn pag_never_collapses() {
    let mut pag = Pag::with_variables(2);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let graph = AcceptedGraph::pag(pag);
    assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Pag);
}

/// End-to-end: a PAG ATE study is licensed (generalized-adjustment envelope),
/// under Frequentist none. Prepared-cache reuse is covered separately.
#[test]
fn build_pag_ate_is_licensed() {
    let mut pag = Pag::with_variables(3);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    let data = antecedent_data::TabularData::from_f64_columns([
        ("t", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0].as_slice()),
        ("y", [0.0, 2.0, 0.1, 2.1, 0.0, 2.0, 0.1, 2.1].as_slice()),
        ("r", [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].as_slice()),
    ])
    .unwrap();
    let result = crate::analysis::Study::tabular(data)
        .graph(pag)
        .query(ate_query())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&antecedent_core::ExecutionContext::for_tests(1))
        .unwrap();
    assert!(result.estimate.as_effect().unwrap().ate.is_finite());
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
    let trace = result.analysis_trace_wire();
    assert_eq!(trace.support_status.as_deref(), Some("licensed"));
}

/// A licensed ADMG Bayesian cell still requires identification of its graph.
#[test]
fn admg_bayesian_bow_requires_identification() {
    let mut admg = Admg::with_variables(2);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let data = antecedent_data::TabularData::from_f64_columns([
        ("t", [0.0, 1.0, 0.0, 1.0].as_slice()),
        ("y", [0.0, 2.0, 0.1, 2.1].as_slice()),
    ])
    .unwrap();
    let err = crate::analysis::Study::tabular(data)
        .graph(admg)
        .query(ate_query())
        .inference(InferenceMode::Bayesian(crate::inference::BayesianConfig::conjugate()))
        .refute(RefuteSuite::None)
        .build()
        .expect("ADMG Bayesian ATE is licensed")
        .run(&antecedent_core::ExecutionContext::for_tests(1))
        .unwrap_err();
    assert!(!matches!(err, CausalError::Support { .. }), "{err}");
    assert!(err.to_string().to_lowercase().contains("identif"), "{err}");
}

#[test]
fn sustained_dbn_posterior_all_suites_are_licensed() {
    assert_eq!(
        classify(cell("SustainedEffect", "TemporalDag", "graph_posterior", "Bayesian", "none")),
        CellStatus::Licensed
    );
    for v in ["cheap", "full"] {
        let c = cell("SustainedEffect", "TemporalDag", "graph_posterior", "Bayesian", v);
        assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
        assert!(refusal_reason(c).is_none());
    }
}

#[test]
fn temporal_mediation_dbn_posterior_all_suites_are_licensed() {
    for inference in ["Bayesian", "Frequentist"] {
        for v in ["none", "cheap", "full"] {
            let c = cell("TemporalMediationEffect", "TemporalDag", "graph_posterior", inference, v);
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
            assert!(refusal_reason(c).is_none());
        }
    }
}
