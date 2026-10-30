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
        let units = table_work(table, &mut work)
            .unwrap()
            .saturating_mul(tables.len().saturating_add(derivation.cited.len()).saturating_add(2));
        work.charge(units).unwrap();
    }
    tables.sort_by_cached_key(|a| (a.key(), a.content_identity()));
    let probabilities: Vec<Vec<f64>> = tables
        .iter()
        .map(|table| {
            table_probabilities(table, UnseenCellRule::Refuse, &limits, &mut work).unwrap().0
        })
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
    let ctx = ExecutionContext::for_tests(7);
    let mut work = super::SampledWork { remaining: limits.evaluation.operations, ctx: &ctx };
    let probabilities =
        table_probabilities(&tables[0], UnseenCellRule::Refuse, &limits, &mut work).unwrap().0;
    ctx.cancellation.cancel();
    let error = antecedent_core::numerical_admission::with_context(&ctx, || {
        law_of(&tables[0], &probabilities, LawFlavor::Empirical, &mut work).unwrap_err()
    });
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn checked_setup_matches_unchecked_table_identity_and_sort_order() {
    use super::{SampledWork, checked_sort_tables};
    let case = chain_case();
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&laws_of(&case), 20_000, &mut rng));
    let ctx = ExecutionContext::for_tests(7);
    let mut work = SampledWork { remaining: usize::MAX, ctx: &ctx };
    let table = &tables[0];
    let cloned = table.checked_clone(&mut work).unwrap();
    assert_eq!(&cloned, table);
    let mut input = Vec::new();
    for population in ["zz", "a", "aa", "z"] {
        let mut key = table.checked_key(&mut work).unwrap();
        key.population = Arc::from(population);
        input.push((key, table.checked_content_identity(&mut work).unwrap(), cloned.clone()));
    }
    let mut expected = input.clone();
    expected.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    assert_eq!(checked_sort_tables(input, &mut work).unwrap(), expected);
    ctx.cancellation.cancel();
    assert_eq!(
        tables[0].checked_clone(&mut work).unwrap_err().reason_code(),
        Some("transport_budget_cancel")
    );
}

#[test]
fn envelope_members_charge_preparation_and_keep_separate_allowances() {
    use super::SampledWork;
    use super::counts::table_probabilities;
    use super::envelope::evaluate_envelope;

    let case = chain_case();
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&laws_of(&case), 20_000, &mut rng));
    let ctx = ExecutionContext::for_tests(7);
    let defaults = SampledTransportLimits::default();
    let mut setup = SampledWork { remaining: defaults.evaluation.operations, ctx: &ctx };
    let probabilities: Vec<Vec<f64>> = tables
        .iter()
        .map(|table| {
            table_probabilities(table, UnseenCellRule::Refuse, &defaults, &mut setup).unwrap().0
        })
        .collect();
    let preparation: usize = probabilities.iter().map(|p| p.len() * 2).sum();
    assert!(preparation > 0);
    let tight = SampledTransportLimits {
        evaluation: antecedent_expr::ExactEvaluationLimits {
            operations: preparation - 1,
            ..defaults.evaluation
        },
        ..defaults.clone()
    };
    // Preparation is charged to the member even when the caller's shared
    // point/gradient allowance has already been consumed.
    let mut shared = SampledWork { remaining: 0, ctx: &ctx };
    let error =
        evaluate_envelope(&derivation, &tables, &probabilities, 0.01, &tight, &ctx, &mut shared)
            .unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
    let mut shared = SampledWork { remaining: 0, ctx: &ctx };
    let envelope =
        evaluate_envelope(&derivation, &tables, &probabilities, 0.01, &defaults, &ctx, &mut shared)
            .unwrap();
    assert!(envelope.defined_members > 0);
    assert_eq!(shared.remaining, 0);
    ctx.cancellation.cancel();
    let mut shared = SampledWork { remaining: defaults.evaluation.operations, ctx: &ctx };
    let error =
        evaluate_envelope(&derivation, &tables, &probabilities, 0.01, &defaults, &ctx, &mut shared)
            .unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn checked_report_key_copy_meters_and_preserves_identity() {
    use super::SampledWork;
    use super::counts::SampledTableKey;
    use antecedent_core::{RegimeId, VariableId};
    let key = SampledTableKey {
        population: Arc::from("α population"),
        regime: RegimeId::from_raw(3),
        interventions: (0..1025).map(|n| (VariableId::from_raw(n), u64::from(n))).collect(),
    };
    let ctx = ExecutionContext::for_tests(7);
    let mut enough = SampledWork { remaining: 1025, ctx: &ctx };
    assert_eq!(key.checked_clone(&mut enough).unwrap(), key);
    let mut insufficient = SampledWork { remaining: 1024, ctx: &ctx };
    assert_eq!(
        key.checked_clone(&mut insufficient).unwrap_err().reason_code(),
        Some("transport_budget_cancel")
    );
    ctx.cancellation.cancel();
    let mut cancelled = SampledWork { remaining: 1025, ctx: &ctx };
    assert_eq!(
        key.checked_clone(&mut cancelled).unwrap_err().reason_code(),
        Some("transport_budget_cancel")
    );
    let name = "α".repeat(200);
    assert!(super::name_preview(&name).len() <= 256);
    assert!(name.starts_with(super::name_preview(&name)));
}

#[test]
fn provider_quote_charges_empty_catalog_environments() {
    use super::{SampledWork, provider_work};
    use antecedent_core::Environment;
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut expanded = derivation.clone();
    let mut environments = expanded.catalog.environments.to_vec();
    for i in 0..64 {
        environments.push(Environment::try_new(format!("unused-{i}"), [], []).unwrap());
    }
    expanded.catalog.environments = environments.into();
    let mut rng = Rng(404);
    let table = sample_laws(&data, 20_000, &mut rng).remove(0);
    let ctx = ExecutionContext::for_tests(7);
    let mut base_work = SampledWork { remaining: usize::MAX - 1, ctx: &ctx };
    let base_quote = provider_work(&derivation, &table, 1, &mut base_work).unwrap();
    let base_scan = usize::MAX - 1 - base_work.remaining;
    let mut enlarged_work = SampledWork { remaining: usize::MAX - 1, ctx: &ctx };
    let enlarged_quote = provider_work(&expanded, &table, 1, &mut enlarged_work).unwrap();
    assert!(enlarged_quote > base_quote);
    assert_eq!(usize::MAX - 1 - enlarged_work.remaining, base_scan + 64);
    let mut limited = SampledWork { remaining: base_scan + base_quote, ctx: &ctx };
    let quote = provider_work(&derivation, &table, 1, &mut limited).unwrap();
    limited.charge(quote).unwrap();
    let mut limited = SampledWork { remaining: base_scan + base_quote, ctx: &ctx };
    let quote = provider_work(&expanded, &table, 1, &mut limited).unwrap();
    assert_eq!(limited.charge(quote).unwrap_err().reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn provider_quote_charges_worst_case_collision_bucket_each_time() {
    use super::{SampledWork, provider_work};
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let table = sample_laws(&data, 20_000, &mut rng).remove(0);
    let ctx = ExecutionContext::for_tests(7);
    let mut single = SampledWork { remaining: usize::MAX - 1, ctx: &ctx };
    let quote_one = provider_work(&derivation, &table, 1, &mut single).unwrap();
    let scan = usize::MAX - 1 - single.remaining;
    let mut many = SampledWork { remaining: usize::MAX - 1, ctx: &ctx };
    let quote_many = provider_work(&derivation, &table, 64, &mut many).unwrap();
    assert!(quote_many > quote_one);
    let mut tight = SampledWork { remaining: scan + quote_one, ctx: &ctx };
    let quote = provider_work(&derivation, &table, 64, &mut tight).unwrap();
    assert_eq!(tight.charge(quote).unwrap_err().reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn malformed_public_derivation_indices_and_cycles_refuse_before_execution() {
    use super::counts::{LawFlavor, table_probabilities};
    use super::{SampledWork, evaluate_detailed};
    use antecedent_identify::CounterfactualItemSource;
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    assert!(!derivation.items.is_empty());
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&data, 20_000, &mut rng));
    let ctx = ExecutionContext::for_tests(7);
    let limits = SampledTransportLimits::default();
    let mut setup = SampledWork { remaining: limits.evaluation.operations, ctx: &ctx };
    let probabilities: Vec<Vec<f64>> = tables.iter().map(|table|
        table_probabilities(table, UnseenCellRule::Refuse, &limits, &mut setup).unwrap().0
    ).collect();
    let flavors = vec![LawFlavor::Empirical; tables.len()];
    let mut malformed = derivation.clone();
    malformed.items[0].population = usize::MAX;
    let mut work = SampledWork { remaining: limits.evaluation.operations, ctx: &ctx };
    let error = evaluate_detailed(&malformed, &tables, &probabilities, &flavors, &limits, &ctx, &mut work).unwrap_err();
    assert_eq!(error.reason_code(), Some("invalid_argument"));
    let mut cyclic = derivation.clone();
    cyclic.items[0].source = CounterfactualItemSource::Transfer { from: 0 };
    let mut work = SampledWork { remaining: limits.evaluation.operations, ctx: &ctx };
    let error = evaluate_detailed(&cyclic, &tables, &probabilities, &flavors, &limits, &ctx, &mut work).unwrap_err();
    assert_eq!(error.reason_code(), Some("invalid_argument"));
}

#[test]
fn sampled_result_clones_and_escaped_collections_retain_admission() {
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&data, 20_000, &mut rng));
    let ctx = ExecutionContext::for_tests(7).with_memory_limit(Some(1 << 28));
    let pool = ctx.admission();
    let before = pool.available().bytes();
    let request = SampledTransportRequest {
        tables, overlap: OverlapDeclaration::all_independent(), unseen_rule: UnseenCellRule::Refuse,
        limits: SampledTransportLimits::default(),
    };
    let result = estimate_sampled_counterfactual_transport(&derivation, &request, &ctx).unwrap();
    let held = pool.available().bytes();
    assert!(held < before);
    assert!(result.tables.retained_admitted_bytes() > 0);
    let cloned = result.clone();
    assert_eq!(result, cloned);
    let escaped_tables = result.tables.clone();
    let escaped_datasets = result.datasets.clone();
    drop(result);
    drop(cloned);
    assert_eq!(pool.available().bytes(), held);
    drop(escaped_tables);
    assert_eq!(pool.available().bytes(), held);
    drop(escaped_datasets);
    assert_eq!(pool.available().bytes(), before);
    let tiny = ExecutionContext::for_tests(7).with_memory_limit(Some(64));
    let error = estimate_sampled_counterfactual_transport(&derivation, &request, &tiny).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}

#[test]
fn direct_exact_catalog_validation_shares_the_evaluation_allowance() {
    use crate::counterfactual_transport::evaluate_counterfactual_transport;
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let ctx = ExecutionContext::for_tests(7);
    let normal = antecedent_expr::ExactEvaluationLimits::default();
    let point = evaluate_counterfactual_transport(&derivation, &data, normal, &ctx).unwrap();
    assert!(point.probability.is_finite());
    let tiny = antecedent_expr::ExactEvaluationLimits { operations: 1, ..normal };
    let error = evaluate_counterfactual_transport(&derivation, &data, tiny, &ctx).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
    ctx.cancellation.cancel();
    let error = evaluate_counterfactual_transport(&derivation, &data, normal, &ctx).unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_budget_cancel"));
}
