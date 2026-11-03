//! The Dirichlet pseudo-count update behind the decision, inverse and marginal-structural
//! posterior-mean routes, against closed forms derived in this file.
//!
//! Conjugacy: a Dirichlet(c, ..., c) prior over `k` atoms updated by weights `w` is
//! Dirichlet(w_1 + c, ..., w_k + c), whose mean is `(w_i + c) / (sum w + k c)`. The expected
//! values below are built from that formula by hand for the two-atom law `{0: w_0, 2: w_1}`
//! (functionals of the law written out per atom) and for the two-stratum marginal structural
//! model (stratum masses and arm laws each updated as their own Dirichlet), at prior
//! concentrations that move every number visibly off the plug-in law. Each route is exercised
//! through its own entry point: the decision functionals, the inverse query, and
//! `msm_dirichlet_posterior`. Nothing here uses a vanishing prior, so a route that skipped the
//! update would fail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_functional;
use antecedent_design::inverse_query::{
    Comparison, FeasibilityStatus, ForwardClaim, ForwardEvidence, GridScope, InverseConstraint,
    InverseQuery, SelectionRule, evaluate_inverse_query,
};
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_posterior_mean,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use antecedent_validate::msm_sensitivity::{MsmOutcomeLaw, MsmStratum, msm_dirichlet_posterior};

/// f64 rounding of a handful of operations on numbers of order one.
const TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= TOLERANCE, "{what}: {actual} vs {expected}");
}

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// Two weighted draws: action `Y` takes the values 0 and 2, action `ref` is the constant 1.
fn law(weights: &[f64]) -> DistributionArtifact {
    let columns = [quantity("do(a=A)"), quantity("do(a=B)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 2],
            weights: Some(weights.to_vec()),
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![0.0, 1.0, 2.0, 1.0],
    )
    .unwrap()
}

fn contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "Y".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=A)")],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "ref".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=B)")],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn functional(weights: &[f64], which: DecisionFunctional) -> f64 {
    evaluate_functional(&contract(), "Y", which, &law(weights)).unwrap().value
}

fn posterior(weights: &[f64], concentration: f64, namespace: DirichletNamespace) -> Vec<f64> {
    dirichlet_posterior_mean(
        weights,
        PosteriorRequest::IndependentNormal { scale: concentration },
        namespace,
    )
    .unwrap()
}

/// Left inverse `inf {x : F(x) >= p}` of the two-atom law with masses `(m0, m2)` on `{0, 2}`.
fn hand_quantile(m0: f64, p: f64) -> f64 {
    if m0 >= p { 0.0 } else { 2.0 }
}

#[test]
fn decision_functionals_of_the_dirichlet_posterior_mean_match_the_conjugate_update() {
    let counts = [1.0, 3.0];
    for concentration in [0.5, 1.0, 2.0, 5.0] {
        // Dirichlet(c + 1, c + 3): posterior mean masses on the atoms 0 and 2.
        let denominator = 4.0 + 2.0 * concentration;
        let m0 = (1.0 + concentration) / denominator;
        let m2 = (3.0 + concentration) / denominator;
        let mean = posterior(&counts, concentration, DirichletNamespace::Decision);
        close(mean[0], m0, "mass on 0");
        close(mean[1], m2, "mass on 2");

        let expectation = 2.0 * m2;
        let variance = 4.0 * m2 - expectation * expectation;
        let label = format!("c = {concentration}");
        close(functional(&mean, DecisionFunctional::Expectation), expectation, &label);
        close(functional(&mean, DecisionFunctional::Variance), variance, &label);
        close(
            functional(&mean, DecisionFunctional::Probability { threshold: 1.0, tail: Tail::Upper }),
            m2,
            &label,
        );
        close(
            functional(&mean, DecisionFunctional::Probability { threshold: 0.0, tail: Tail::Lower }),
            m0,
            &label,
        );
        for p in [0.28, 0.5, 0.9] {
            close(
                functional(&mean, DecisionFunctional::Quantile { p }),
                hand_quantile(m0, p),
                &label,
            );
        }
        // The plug-in law has E = 3/2; the update must move the published number off it.
        assert!((expectation - 1.5).abs() > 0.05, "{label}: prior scale is moderate");
    }
}

fn inverse_report(
    namespace: DirichletNamespace,
    counts: &[f64],
    concentration: f64,
    target: f64,
    comparison: Comparison,
) -> (f64, FeasibilityStatus, Option<String>) {
    let mean = posterior(counts, concentration, namespace);
    let evidence = ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(law(&mean)))),
        ..ForwardEvidence::default()
    };
    let query = InverseQuery {
        contract: contract(),
        grid_order: vec!["Y".into(), "ref".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean { target, comparison }],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    };
    let result = evaluate_inverse_query(&query, &evidence).unwrap();
    let report = &result.actions[0];
    assert_eq!(report.id, "Y");
    (report.point_values[0].value.unwrap(), report.point.unwrap(), result.selected)
}

/// Both inverse routes read the Dirichlet mean of the law; the feasibility boundary of
/// `E[Y] >= t` must sit exactly at the conjugate posterior mean `2 (w_1 + c) / (W + 2 c)`.
fn inverse_boundary_tracks_the_posterior_mean(namespace: DirichletNamespace, counts: [f64; 2]) {
    let total = counts[0] + counts[1];
    for concentration in [0.5, 1.0, 2.0] {
        let expected = 2.0 * (counts[1] + concentration) / (total + 2.0 * concentration);
        let plug_in = 2.0 * counts[1] / total;
        // The prior must move the mean for the check to bite.
        assert!((expected - plug_in).abs() > 0.03);
        let (below, status_below, _) = inverse_report(
            namespace,
            &counts,
            concentration,
            expected - 1e-6,
            Comparison::AtLeast,
        );
        close(below, expected, "forward value of the posterior mean law");
        assert_eq!(status_below, FeasibilityStatus::Feasible, "target just under the mean");
        let (_, status_above, _) = inverse_report(
            namespace,
            &counts,
            concentration,
            expected + 1e-6,
            Comparison::AtLeast,
        );
        assert_eq!(status_above, FeasibilityStatus::Infeasible, "target just over the mean");
        // The mirror inequality flips with it.
        let (_, at_most_below, _) = inverse_report(
            namespace,
            &counts,
            concentration,
            expected - 1e-6,
            Comparison::AtMost,
        );
        assert_eq!(at_most_below, FeasibilityStatus::Infeasible);
        let (_, at_most_above, _) = inverse_report(
            namespace,
            &counts,
            concentration,
            expected + 1e-6,
            Comparison::AtMost,
        );
        assert_eq!(at_most_above, FeasibilityStatus::Feasible);
    }
}

#[test]
fn functional_inverse_boundary_is_the_conjugate_posterior_mean() {
    inverse_boundary_tracks_the_posterior_mean(DirichletNamespace::FunctionalInverse, [1.0, 3.0]);
}

#[test]
fn functional_inverse_selection_flips_where_the_update_flips_it() {
    // E[Y] = 2 (3 + c) / (4 + 2 c): 1.5 plug-in, 1.4 at c = 1/2, 4/3 at c = 1. Against the
    // target E[Y] >= 1.35 (the constant action `ref` has mean 1) the plug-in and c = 1/2
    // select Y while c = 1 selects nothing.
    let target = 1.35;
    let plug_in = inverse_report(
        DirichletNamespace::FunctionalInverse,
        &[1.0, 3.0],
        1e-12,
        target,
        Comparison::AtLeast,
    );
    assert_eq!(plug_in.2.as_deref(), Some("Y"));
    let half = inverse_report(
        DirichletNamespace::FunctionalInverse,
        &[1.0, 3.0],
        0.5,
        target,
        Comparison::AtLeast,
    );
    close(half.0, 1.4, "E at c = 1/2");
    assert_eq!(half.2.as_deref(), Some("Y"));
    let one = inverse_report(
        DirichletNamespace::FunctionalInverse,
        &[1.0, 3.0],
        1.0,
        target,
        Comparison::AtLeast,
    );
    close(one.0, 4.0 / 3.0, "E at c = 1");
    assert_eq!(one.2, None);
}

#[test]
fn finite_action_inverse_boundary_is_the_conjugate_posterior_mean() {
    // A different finite law (mass mostly on 0): E = 2 (1 + c) / (4 + 2 c).
    inverse_boundary_tracks_the_posterior_mean(DirichletNamespace::FiniteActionInverse, [3.0, 1.0]);
}

#[test]
fn finite_action_inverse_selection_flips_where_the_update_flips_it() {
    // Counts [3, 1]: plug-in E = 1/2, c = 1 gives 2 (1 + 1) / 6 = 2/3, c = 2 gives 3/4. Against
    // the target E[Y] >= 0.7 only c = 2 reaches the target, and the constant action `ref` (mean
    // 1) is then not selected because Y comes first in grid order.
    let target = 0.7;
    for (concentration, selected, mean) in
        [(1e-12, "ref", 0.5), (1.0, "ref", 2.0 / 3.0), (2.0, "Y", 0.75)]
    {
        let (value, _, picked) = inverse_report(
            DirichletNamespace::FiniteActionInverse,
            &[3.0, 1.0],
            concentration,
            target,
            Comparison::AtLeast,
        );
        close(value, mean, "forward value");
        // `ref` has mean 1 >= 0.7, so it is the selection whenever Y misses the target.
        assert_eq!(picked.as_deref(), Some(selected), "c = {concentration}");
    }
}

fn binary(rate: f64) -> MsmOutcomeLaw {
    MsmOutcomeLaw { values: vec![0.0, 1.0], probabilities: vec![1.0 - rate, rate] }
}

fn stratum(mass: f64, propensity: f64, treated: MsmOutcomeLaw, control: MsmOutcomeLaw) -> MsmStratum {
    MsmStratum { mass, propensity, treated, control }
}

fn expected_value(law: &MsmOutcomeLaw) -> f64 {
    law.values.iter().zip(&law.probabilities).map(|(v, p)| v * p).sum()
}

/// Hand posterior-mean law: cells with positive probability are a Dirichlet(c) update with
/// the probabilities as pseudo-counts; a structural zero stays zero.
fn hand_update(probabilities: &[f64], concentration: f64) -> Vec<f64> {
    let free = probabilities.iter().filter(|p| **p > 0.0).count() as f64;
    let sum: f64 = probabilities.iter().sum();
    probabilities
        .iter()
        .map(|p| if *p > 0.0 { (p + concentration) / (sum + free * concentration) } else { 0.0 })
        .collect()
}

#[test]
fn msm_posterior_is_the_identified_ate_of_the_conjugate_updated_laws() {
    // Unequal masses and a three-atom arm with a structural zero make every piece of the update
    // matter: masses (0.2, 0.8) become (0.2 + c, 0.8 + c) / (1 + 2 c); a binary arm with rate r
    // becomes rate (r + c) / (1 + 2 c); the {0, 1, 2} treated law (1/5, 0, 4/5) keeps its zero
    // and has two free cells.
    let three_atom = |probabilities: Vec<f64>| MsmOutcomeLaw {
        values: vec![0.0, 1.0, 2.0],
        probabilities,
    };
    let strata = vec![
        stratum(0.2, 0.5, binary(0.8), binary(0.4)),
        stratum(0.8, 0.25, binary(0.5), binary(0.3)),
    ];
    let with_zero = vec![
        stratum(0.2, 0.5, binary(0.8), binary(0.4)),
        stratum(
            0.8,
            0.25,
            three_atom(vec![0.2, 0.0, 0.8]),
            three_atom(vec![0.25, 0.5, 0.25]),
        ),
    ];
    for concentration in [0.5, 1.0, 3.0] {
        let request = PosteriorRequest::IndependentNormal { scale: concentration };
        for case in [&strata, &with_zero] {
            let masses: Vec<f64> = case.iter().map(|s| s.mass).collect();
            let updated_masses = hand_update(&masses, concentration);
            let mut ate = 0.0;
            for (stratum, mass) in case.iter().zip(&updated_masses) {
                let treated = hand_update(&stratum.treated.probabilities, concentration);
                let control = hand_update(&stratum.control.probabilities, concentration);
                let e1: f64 = stratum.treated.values.iter().zip(&treated).map(|(v, p)| v * p).sum();
                let e0: f64 = stratum.control.values.iter().zip(&control).map(|(v, p)| v * p).sum();
                ate += mass * (e1 - e0);
            }
            let produced = msm_dirichlet_posterior(case, request).unwrap();
            close(produced, ate, &format!("c = {concentration}"));
            // The plug-in identified ATE is different: the update must have moved it.
            let plug_in: f64 =
                case.iter().map(|s| s.mass * (expected_value(&s.treated) - expected_value(&s.control))).sum();
            assert!((produced - plug_in).abs() > 1e-3, "c = {concentration}: {produced} {plug_in}");
        }
    }
    // One hand number written out: two binary strata, c = 1, masses stay (0.5, 0.5) and every
    // contrast shrinks by 1 / (1 + 2 c) = 1/3, so the posterior-mean ATE is 0.3 / 3 = 0.1.
    let two = vec![
        stratum(0.5, 0.5, binary(0.8), binary(0.4)),
        stratum(0.5, 0.25, binary(0.5), binary(0.3)),
    ];
    let produced =
        msm_dirichlet_posterior(&two, PosteriorRequest::IndependentNormal { scale: 1.0 }).unwrap();
    close(produced, 0.1, "hand 0.3 / 3");
}
