//! Parameter sweep of identified counterfactual transport against exact finite-SCM enumeration.
//!
//! The named fixtures fix one set of probabilities per structure. This file re-declares three
//! of the structures (a ternary-mediator nested natural effect with a covariate shift, the
//! confounded covariate model with its joint counterfactual and effect of treatment on the
//! treated, and the front-door district with a selection node on the mediator) as functions of
//! their probabilities and runs the identifier and the exact evaluator on 16 parameter sets of
//! each, including probabilities 2^-20 from the boundary (extreme weights, near-empty cells)
//! and unbalanced source and target covariate laws. The expected value of every target is
//! enumerated by the oracle from the target population's own structural model; the derivation
//! only ever sees the supplied regime laws. Identification must not depend on the numbers: every
//! parameter set must identify, and agree with the enumerated truth to 1e-12.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use antecedent_core::ExecutionContext;
use antecedent_estimate::evaluate_counterfactual_transport;
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    identify_counterfactual_transport,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, CaseGraph, ExpectedClass, Target};
use finite_scm_oracle::{
    Atom, CfVar, EvidenceSpec, FiniteScm, Mechanism, Population, Setting, TargetQuery, bern,
};

const TOLERANCE: f64 = 1e-12;

/// Deterministic dyadic probabilities, strictly inside (0, 1), with boundary-adjacent values.
struct Dyadic(u64);
impl Dyadic {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pool =
            [2f64.powi(-20), 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 - 2f64.powi(-20)];
        pool[(self.0 >> 33) as usize % pool.len()]
    }
    fn pmf3(&mut self) -> Vec<f64> {
        // three strictly positive dyadic masses summing to one
        let parts = [1u32, 2, 3, 4, 5, 6];
        let a = parts[(self.next() * 5.99) as usize % 6];
        let b = parts[(self.next() * 5.99) as usize % 6];
        let total = f64::from(a + b + 1);
        vec![f64::from(a) / total, f64::from(b) / total, 1.0 / total]
    }
}

fn pops(source: FiniteScm, target: FiniteScm) -> Vec<Population> {
    vec![Population::new("source", source), Population::new("target", target)]
}

fn target_of(label: &'static str, populations: &[Population], query: TargetQuery) -> Target {
    let expected_value = query.value(populations);
    Target { label, query, expected_value, class: ExpectedClass::Identified }
}

fn nested(var: &str, x: usize, m: &str, xp: usize) -> CfVar {
    CfVar::nested(
        var,
        vec![("X", Setting::Level(x)), (m, Setting::cf(CfVar::under(m, &[("X", xp)])))],
    )
}

fn mediation(uz: f64, ux: f64, um: &[f64], uy: f64) -> FiniteScm {
    FiniteScm::new()
        .exo("Uz", bern(uz))
        .exo("Ux", bern(ux))
        .exo("Um", um.to_vec())
        .exo("Uy", bern(uy))
        .endo("Z", 2, Mechanism::new(|_, u| u[0]))
        .endo("X", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("M", 3, Mechanism::new(|v, u| (v[1] + v[0] + u[2]) % 3))
        .endo("Y", 2, Mechanism::new(|v, u| usize::from(v[2] + v[1] * v[0] >= 2) ^ u[3]))
}

fn mediation_case(rng: &mut Dyadic) -> Case {
    let (ux, uy, um) = (rng.next(), rng.next(), rng.pmf3());
    let populations = pops(mediation(rng.next(), ux, &um, uy), mediation(rng.next(), ux, &um, uy));
    let q = |x, xp| TargetQuery::new("target", vec![Atom::new(nested("Y", x, "M", xp), 1)], vec![]);
    Case {
        name: "mediation_sweep",
        derivation: "mediation formula with a covariate-law shift",
        graph: CaseGraph {
            observed: vec!["Z", "X", "M", "Y"],
            directed: vec![("Z", "X"), ("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")],
            bidirected: vec![],
            selection: vec![("S_Z", "Z")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "M", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![
            target_of("Y_{1,M0}", &populations, q(1, 0)),
            target_of("Y_{0,M0}", &populations, q(0, 0)),
            target_of("Y_{1,M1}", &populations, q(1, 1)),
            target_of("Y_{0,M1}", &populations, q(0, 1)),
        ],
        populations,
        alternative: None,
    }
}

fn confounded(uz: f64, u: f64, ux: f64, uy: f64) -> FiniteScm {
    FiniteScm::new()
        .exo("Uz", bern(uz))
        .exo("U", bern(u))
        .exo("Ux", bern(ux))
        .exo("Uy", bern(uy))
        .endo("Z", 2, Mechanism::new(|_, u| u[0]))
        .endo("X", 2, Mechanism::new(|v, u| u[1] ^ (v[0] & u[2])))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ (v[0] & u[1]) ^ u[3]))
}

fn confounded_case(rng: &mut Dyadic) -> Case {
    let (u, ux, uy) = (rng.next(), rng.next(), rng.next());
    let populations = pops(confounded(rng.next(), u, ux, uy), confounded(rng.next(), u, ux, uy));
    let joint = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1), Atom::factual("X", 0)],
        vec![],
    );
    let ett = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)],
        vec![Atom::factual("X", 0)],
    );
    Case {
        name: "confounded_sweep",
        derivation: "joint counterfactual and ETT with a selection node on Z",
        graph: CaseGraph {
            observed: vec!["Z", "X", "Y"],
            directed: vec![("Z", "X"), ("Z", "Y"), ("X", "Y")],
            bidirected: vec![("X", "Y")],
            selection: vec![("S_Z", "Z")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 0)], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 1)], &["Z", "X", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![target_of("joint", &populations, joint), target_of("ett", &populations, ett)],
        populations,
        alternative: None,
    }
}

fn front_door(u: f64, ux: f64, um: f64, uy: f64) -> FiniteScm {
    FiniteScm::new()
        .exo("U", bern(u))
        .exo("Ux", bern(ux))
        .exo("Um", bern(um))
        .exo("Uy", bern(uy))
        .endo("X", 2, Mechanism::new(|_, u| u[0] ^ u[1]))
        .endo("M", 2, Mechanism::new(|v, u| v[0] ^ u[2]))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ u[0] ^ u[3]))
}

fn front_door_case(rng: &mut Dyadic) -> Case {
    let (u, ux, uy) = (rng.next(), rng.next(), rng.next());
    let populations = pops(front_door(u, ux, rng.next(), uy), front_door(u, ux, rng.next(), uy));
    let query =
        TargetQuery::new("target", vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)], vec![]);
    Case {
        name: "front_door_sweep",
        derivation: "front door with a selection node on the mediator",
        graph: CaseGraph {
            observed: vec!["X", "M", "Y"],
            directed: vec![("X", "M"), ("M", "Y")],
            bidirected: vec![("X", "Y")],
            selection: vec![("S_M", "M")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["X", "M", "Y"]),
            EvidenceSpec::new("target", &[], &["X", "M"]),
        ],
        targets: vec![target_of("Y_{x1}=1", &populations, query)],
        populations,
        alternative: None,
    }
}

#[test]
fn identification_does_not_depend_on_the_numbers_and_values_equal_enumeration() {
    let mut rng = Dyadic(20_261_028);
    let mut checked = 0;
    let mut worst = 0.0f64;
    for round in 0..16 {
        for case in [mediation_case(&mut rng), confounded_case(&mut rng), front_door_case(&mut rng)]
        {
            let adapted = adapter::catalog(&case);
            let data = adapter::laws(&case, &adapted);
            for target in &case.targets {
                let problem = adapter::problem(&case);
                let sources = adapter::sources(&case, &adapter::evidence_controllable(&case));
                let query = adapter::query(&case, &target.query, sources);
                let ctx = ExecutionContext::for_tests(1);
                let decision = identify_counterfactual_transport(
                    &problem,
                    &query,
                    &adapted.catalog,
                    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
                    &ctx,
                )
                .expect("a sweep case is a valid request");
                let CounterfactualTransportDecision::Identified(derivation) = decision else {
                    panic!("{} #{round} / {}: not identified", case.name, target.label);
                };
                let truth = target.expected_value.expect("positive conditioning mass");
                let point = evaluate_counterfactual_transport(
                    &derivation,
                    &data,
                    ExactEvaluationLimits::default(),
                    &ctx,
                )
                .unwrap_or_else(|e| panic!("{} #{round} / {}: {e}", case.name, target.label));
                let err = (point.probability - truth).abs();
                worst = worst.max(err);
                assert!(
                    err <= TOLERANCE,
                    "{} #{round} / {}: derived {} against enumerated {truth}",
                    case.name,
                    target.label,
                    point.probability
                );
                checked += 1;
            }
        }
    }
    eprintln!("checked {checked} targets, worst error {worst:e}");
    assert_eq!(checked, 16 * (4 + 2 + 1));
}
