//! Point-only budget regression, using the finite-SCM test fixture.
#![allow(unused_imports, dead_code)]
use std::sync::Arc;

use crate::EstimationError;
use crate::counterfactual_transport_adapter as adapter;
use crate::counterfactual_transport_cases::{Case, CaseGraph, ExpectedClass, Target, all_cases};
use crate::counterfactual_transport_sampled::{
    DatasetOverlap, LABEL_ENVELOPE_ONLY, LABEL_EXACT_ON_ESTIMATED_TABLES, LABEL_IDENTIFIED,
    LABEL_INFERENCE_WITHHELD, LABEL_PLUG_IN, LABEL_SMOOTHED_PLUG_IN, OverlapDeclaration,
    SAMPLED_TRANSPORT_UNCERTAINTY_METHOD, STRUCTURAL_ENVELOPE_LABEL, SampledAxis,
    SampledCounterfactualTransportEstimate, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UNCALIBRATED_STANDARD_ERROR,
    UnseenCellRule, estimate_sampled_counterfactual_transport,
};
use crate::finite_scm_oracle::{
    Atom, CfVar, EvidenceSpec, FiniteScm, Mechanism, Population, Setting, TargetQuery, bern,
};
use antecedent_core::ExecutionContext;
use antecedent_expr::{ExactDiscreteLaw, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};

/// Absolute tolerance of a plug-in at the large-sample size below: about five
/// standard deviations of the least precise estimand of the named designs.
const PLUG_IN_TOLERANCE: f64 = 0.01;
/// Units per table of the large-sample designs.
const LARGE_N: u64 = 200_000;
/// Units per table of the sampling-variance design.
const MODEST_N: u64 = 1_500;
/// Replicates of the sampling-variance design.
const REPLICATES: u64 = 400;

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

fn chain_case() -> Case {
    let scm = FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Um", bern(0.25))
        .exo("Uy", bern(0.125))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("M", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ u[2]));
    let populations = vec![Population::new("pop", scm)];
    let query = TargetQuery::new("pop", vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)], vec![]);
    let expected_value = query.value(&populations);
    Case {
        name: "chain_unread_coordinate",
        derivation: "X -> M -> Y without confounding: P(y | do(x)) = sum_m P(m | x) P(y | m); the \
                     factor of Y reads M only, so its sampled conditional on (X, M) differs \
                     across X by noise and is pooled over X",
        graph: CaseGraph {
            observed: vec!["X", "M", "Y"],
            directed: vec![("X", "M"), ("M", "Y")],
            bidirected: vec![],
            selection: vec![],
        },
        populations,
        target_population: "pop",
        evidence: vec![EvidenceSpec::new("pop", &[], &["X", "M", "Y"])],
        targets: vec![Target {
            label: "Y_{x1}=1",
            query,
            expected_value,
            class: ExpectedClass::Identified,
        }],
        alternative: None,
    }
}

#[test]
fn point_fits_but_gradient_exhausts_shared_budget() {
    use super::counts::{LawFlavor, table_probabilities};
    use super::{SampledWork, evaluate_detailed, table_work};
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let mut tables = cited_only(&derivation, &sample_laws(&data, 20_000, &mut rng));
    let ctx = ExecutionContext::for_tests(7);
    let limits = SampledTransportLimits::default();
    let mut work = SampledWork { remaining: limits.evaluation.operations, ctx: &ctx };
    for table in &tables {
        work.charge(
            table_work(table).saturating_mul(
                tables.len().saturating_add(derivation.cited.len()).saturating_add(2),
            ),
        )
        .unwrap();
    }
    tables.sort_by_cached_key(|a| (a.key(), a.content_identity()));
    let probabilities: Vec<Vec<f64>> = tables
        .iter()
        .map(|table| table_probabilities(table, UnseenCellRule::Refuse, &limits).unwrap().0)
        .collect();
    let flavors = vec![LawFlavor::Empirical; tables.len()];
    let (point, _) =
        evaluate_detailed(&derivation, &tables, &probabilities, &flavors, &limits, &ctx, &mut work)
            .unwrap();
    assert!(point.probability.is_finite());
    let point_budget = limits.evaluation.operations - work.remaining;
    assert!(point_budget > 0);
    let request = SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits: SampledTransportLimits {
            evaluation: antecedent_expr::ExactEvaluationLimits {
                operations: point_budget,
                ..limits.evaluation
            },
            ..limits
        },
    };
    let error = estimate_sampled_counterfactual_transport(&derivation, &request, &ctx).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn final_evidence_identity_uses_the_shared_budget_and_checks_cancellation() {
    use super::{SampledWork, evidence_identity};
    use antecedent_core::identity::CanonicalWriter;

    let case = chain_case();
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&laws_of(&case), 20_000, &mut rng));
    let request = SampledTransportRequest {
        tables: tables.clone(),
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits: SampledTransportLimits::default(),
    };
    let ctx = ExecutionContext::for_tests(7);
    let mut insufficient = SampledWork { remaining: 0, ctx: &ctx };
    let error = evidence_identity(&request, &tables, &mut insufficient).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));

    let mut expected = CanonicalWriter::new(super::SAMPLED_TRANSPORT_IDENTITY_TAG);
    expected.word(tables.len() as u64);
    for table in &tables {
        table.write_identity(&mut expected);
    }
    request.overlap.write_identity(&mut expected);
    expected.text(request.unseen_rule.name()).opt_real(request.unseen_rule.parameter());
    expected.word(request.limits.min_table_total).real(request.limits.max_unseen_fraction);
    expected.word(u64::from(request.limits.max_envelope_evaluations));
    expected
        .word(request.limits.evaluation.operations as u64)
        .word(request.limits.evaluation.depth as u64);
    let mut sufficient = SampledWork { remaining: usize::MAX, ctx: &ctx };
    assert_eq!(
        evidence_identity(&request, &tables, &mut sufficient).unwrap(),
        expected.finish_hex()
    );

    ctx.cancellation.cancel();
    let mut cancelled = SampledWork { remaining: usize::MAX, ctx: &ctx };
    let error = evidence_identity(&request, &tables, &mut cancelled).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn checked_table_and_overlap_identities_preserve_framing_and_meter_large_scans() {
    use super::SampledWork;
    use antecedent_core::{CanonicalWriter, RegimeId, VariableId};
    let table = SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from("population"),
        regime: RegimeId::from_raw(1),
        interventions: Vec::new(),
        axes: vec![SampledAxis {
            variable: VariableId::from_raw(0),
            levels: (0..2048).map(f64::from).collect(),
        }],
        counts: vec![1; 2048],
        snapshot_identity: Arc::from("snapshot"),
        design: SamplingDesign::Stratified { variables: vec![VariableId::from_raw(0)] },
        structural_zeros: Vec::new(),
    })
    .unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let mut budget = SampledWork { remaining: 2500, ctx: &ctx };
    assert_eq!(
        table.checked_content_identity(&mut budget).unwrap_err().reason_code(),
        Some("transport_budget_cancel")
    );
    let mut budget = SampledWork { remaining: 20_000, ctx: &ctx };
    assert_eq!(table.checked_key(&mut budget).unwrap(), table.key());
    assert_eq!(table.checked_content_identity(&mut budget).unwrap(), table.content_identity());

    let first = table.key();
    let mut second = first.clone();
    second.regime = RegimeId::from_raw(2);
    let overlap = OverlapDeclaration::all_independent()
        .with_pair(first, second, DatasetOverlap::SharedUnits)
        .unwrap();
    let mut old = CanonicalWriter::new("overlap-test");
    overlap.write_identity(&mut old);
    let mut checked = CanonicalWriter::new("overlap-test");
    overlap.write_identity_checked(&mut checked, &mut budget).unwrap();
    assert_eq!(checked.finish_hex(), old.finish_hex());
    ctx.cancellation.cancel();
    let mut cancelled = SampledWork { remaining: 20_000, ctx: &ctx };
    assert_eq!(
        overlap.write_identity_checked(&mut checked, &mut cancelled).unwrap_err().reason_code(),
        Some("transport_budget_cancel")
    );
}

#[test]
fn cancelled_exact_law_is_mapped_to_transport_cancellation() {
    use super::counts::{LawFlavor, law_of, table_probabilities};
    let case = chain_case();
    let data = laws_of(&case);
    let mut rng = Rng(404);
    let tables = sample_laws(&data, 20_000, &mut rng);
    let limits = SampledTransportLimits::default();
    let probabilities = table_probabilities(&tables[0], UnseenCellRule::Refuse, &limits).unwrap().0;
    let ctx = ExecutionContext::for_tests(7);
    ctx.cancellation.cancel();
    let error = antecedent_core::numerical_admission::with_context(&ctx, || {
        law_of(&tables[0], &probabilities, LawFlavor::Empirical).unwrap_err()
    });
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}
