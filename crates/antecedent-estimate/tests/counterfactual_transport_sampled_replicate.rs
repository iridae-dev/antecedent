//! A sampled mediation replicate under covariate shift returns a point and a
//! delta-method standard error, so a Wald interval can be formed.
//!
//! Hand example. The enumerated target truth of `Y_{x1, M_{x0}}` is supplied by
//! the finite-SCM oracle; at 40 000 units per table every cell of the ternary
//! mediator design is seen (the rarest cell has mass about 0.002, so about 80
//! expected units), the route does not refuse, and the plug-in lies within five
//! delta-method standard errors of the truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::counterfactual_transport_sampled::{
    OverlapDeclaration, SampledAxis, SampledCounterfactualTransportEstimate, SampledRegimeCounts,
    SampledRegimeCountsParts, SampledTransportLimits, SampledTransportRequest, SamplingDesign,
    UnseenCellRule, estimate_sampled_counterfactual_transport,
};
use antecedent_expr::{ExactDiscreteLaw, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, ExpectedClass, Target, all_cases};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn multinomial(probabilities: &[f64], n: u64, rng: &mut Rng) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for p in probabilities {
        running += p;
        cumulative.push(running);
    }
    let last_positive = probabilities.iter().rposition(|p| *p > 0.0).expect("a law has mass");
    let mut counts = vec![0u64; probabilities.len()];
    for _ in 0..n {
        let u = rng.unit();
        let cell = cumulative.partition_point(|c| *c <= u).min(last_positive);
        counts[cell] += 1;
    }
    counts
}

fn table_of(law: &ExactDiscreteLaw, counts: Vec<u64>, structural: bool) -> SampledRegimeCounts {
    let zeros: Vec<usize> = if structural {
        law.probabilities().iter().enumerate().filter(|(_, p)| **p == 0.0).map(|(i, _)| i).collect()
    } else {
        Vec::new()
    };
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().expect("a finite level")))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|a| SampledAxis {
                variable: a.variable,
                levels: a.values.iter().map(|v| v.as_f64().expect("a finite level")).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: zeros,
    })
    .expect("a sampled table of an enumerated law is valid")
}

fn sample_laws(data: &ExactTransportData, n: u64, rng: &mut Rng) -> Vec<SampledRegimeCounts> {
    data.laws()
        .iter()
        .map(|law| table_of(law, multinomial(law.probabilities(), n, rng), true))
        .collect()
}

fn controllable_for(case: &Case) -> Vec<&'static str> {
    if case.name == "missing_interventional_evidence" {
        vec!["X"]
    } else {
        adapter::evidence_controllable(case)
    }
}

fn identify(case: &Case, target: &Target) -> CounterfactualTransportDecision {
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, &controllable_for(case));
    let query = adapter::query(case, &target.query, sources);
    identify_counterfactual_transport(
        &problem,
        &query,
        &catalog.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .expect("a fixture is a valid request")
}

fn derive(case: &Case, target: &Target) -> CounterfactualTransportDerivation {
    match identify(case, target) {
        CounterfactualTransportDecision::Identified(derivation) => *derivation,
        other => panic!("{} / {}: expected a derivation, got {other:?}", case.name, target.label),
    }
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).expect("fixture exists")
}

fn request(tables: Vec<SampledRegimeCounts>, rule: UnseenCellRule) -> SampledTransportRequest {
    SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: rule,
        limits: SampledTransportLimits::default(),
    }
}

fn estimate(
    derivation: &CounterfactualTransportDerivation,
    tables: Vec<SampledRegimeCounts>,
    rule: UnseenCellRule,
) -> Result<SampledCounterfactualTransportEstimate, EstimationError> {
    estimate_sampled_counterfactual_transport(
        derivation,
        &request(tables, rule),
        &ExecutionContext::for_tests(7),
    )
}

/// The tables a derivation cites; evidence it does not read is refused by the
/// route, so a request carries exactly the cited tables.
fn cited_only(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
) -> Vec<SampledRegimeCounts> {
    tables
        .iter()
        .filter(|t| {
            derivation
                .cited
                .iter()
                .any(|c| c.regime == t.regime() && c.population.as_ref() == t.population())
        })
        .cloned()
        .collect()
}

fn laws_of(case: &Case) -> ExactTransportData {
    let adapted = adapter::catalog(case);
    adapter::laws(case, &adapted)
}

#[test]
fn a_sampled_mediation_replicate_returns_a_point_and_a_standard_error() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let target = case.targets.iter().find(|t| t.label == "Y_{x1,M_{x0}}").unwrap();
    assert_eq!(target.class, ExpectedClass::Identified);
    let truth = target.expected_value.expect("the oracle supplies the target truth");
    let derivation = derive(&case, target);
    let data = laws_of(&case);
    let mut rng = Rng(0x71a0_0001);
    let tables = cited_only(&derivation, &sample_laws(&data, 40_000, &mut rng));
    let result = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap();
    let point = result.point.as_ref().expect("a plug-in point").probability;
    let se = result.inference.standard_error.as_ref().expect("a delta-method standard error").value;
    assert!(se > 0.0 && se < 0.05, "standard error {se}");
    assert!((point - truth).abs() < 5.0 * se, "point {point} truth {truth} se {se}");
}

#[test]
fn a_sparse_replicate_with_an_unseen_cell_refuses_rather_than_filling_zero() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let target = case.targets.iter().find(|t| t.label == "Y_{x1,M_{x0}}").unwrap();
    let derivation = derive(&case, target);
    let data = laws_of(&case);
    // Ten units per table cannot see every one of the design's cells.
    let mut rng = Rng(7);
    let tables = cited_only(&derivation, &sample_laws(&data, 10, &mut rng));
    let error = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap_err();
    match error {
        EstimationError::Refused { code, message } => {
            assert_eq!(code, "transport_support_failure");
            assert!(message.contains("unseen"), "{message}");
        }
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}
