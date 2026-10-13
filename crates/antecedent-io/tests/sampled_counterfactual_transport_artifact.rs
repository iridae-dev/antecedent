//! The sampled counterfactual transport artifact: a fresh consumer holding only
//! the bytes recomputes the plug-in estimate from the counts bit for bit, and
//! every semantic mutation (counts edited, overlap flipped, derivation swapped,
//! stored estimate edited), even resealed, is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceKind,
};
use antecedent_core::ExecutionContext;
use antecedent_estimate::counterfactual_transport_sampled::{
    DatasetOverlap, OverlapDeclaration, SampledAxis, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
};
use antecedent_expr::ExactDiscreteLaw;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SampledCounterfactualTransportArtifactError as ArtifactError,
    SampledCounterfactualTransportArtifactWire as Wire,
    SampledCounterfactualTransportConsumeLimits as ConsumeLimits,
    SampledCounterfactualTransportExpectedIdentity as Expected,
    consume_sampled_counterfactual_transport, consume_sampled_counterfactual_transport_with_limits,
    overlap_declaration_from_ledger, sampled_counterfactual_transport_artifact,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, Target, all_cases};

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
    let last = probabilities.iter().rposition(|p| *p > 0.0).unwrap();
    let mut counts = vec![0u64; probabilities.len()];
    for _ in 0..n {
        let u = rng.unit();
        counts[cumulative.partition_point(|c| *c <= u).min(last)] += 1;
    }
    counts
}

fn table_of(law: &ExactDiscreteLaw, counts: Vec<u64>) -> SampledRegimeCounts {
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().unwrap()))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|a| SampledAxis {
                variable: a.variable,
                levels: a.values.iter().map(|v| v.as_f64().unwrap()).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: law
            .probabilities()
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == 0.0)
            .map(|(i, _)| i)
            .collect(),
    })
    .unwrap()
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).unwrap()
}

fn derive(case: &Case, target: &Target) -> CounterfactualTransportDerivation {
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, &adapter::evidence_controllable(case));
    let query = adapter::query(case, &target.query, sources);
    match identify_counterfactual_transport(
        &problem,
        &query,
        &catalog.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
    {
        CounterfactualTransportDecision::Identified(d) => *d,
        other => panic!("expected a derivation, got {other:?}"),
    }
}

fn request_for(
    case: &Case,
    derivation: &CounterfactualTransportDerivation,
    seed: u64,
) -> SampledTransportRequest {
    let adapted = adapter::catalog(case);
    let data = adapter::laws(case, &adapted);
    let mut rng = Rng(seed);
    let tables: Vec<SampledRegimeCounts> = data
        .laws()
        .iter()
        .filter(|law| {
            derivation
                .cited
                .iter()
                .any(|c| c.regime == law.regime() && c.population.as_ref() == law.population())
        })
        .map(|law| table_of(law, multinomial(law.probabilities(), 5_000, &mut rng)))
        .collect();
    SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits: SampledTransportLimits::default(),
    }
}

fn refusal_code(error: &ArtifactError) -> &'static str {
    error.refusal().0
}

#[test]
fn fresh_consumer_recomputes_the_plug_in_from_the_counts_bit_for_bit() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let derivation = derive(&case, &case.targets[0]);
    let request = request_for(&case, &derivation, 1);
    let ctx = ExecutionContext::for_tests(3);
    let (wire, produced) =
        sampled_counterfactual_transport_artifact(&derivation, &request, &[], &ctx).unwrap();
    let bytes = wire.export().unwrap();

    // A fresh consumer: bytes and the independently held identity only.
    let consumed = consume_sampled_counterfactual_transport_with_limits(
        &bytes,
        &wire.identity(),
        ConsumeLimits::default(),
        &ExecutionContext::for_tests(99),
    )
    .unwrap();
    let produced_point = produced.point.as_ref().unwrap();
    let replayed_point = consumed.estimate.point.as_ref().unwrap();
    assert_eq!(produced_point.probability.to_bits(), replayed_point.probability.to_bits());
    assert_eq!(produced_point.numerator.to_bits(), replayed_point.numerator.to_bits());
    assert_eq!(produced_point.denominator.to_bits(), replayed_point.denominator.to_bits());
    assert_eq!(
        produced.inference.standard_error.as_ref().unwrap().value.to_bits(),
        consumed.estimate.inference.standard_error.as_ref().unwrap().value.to_bits()
    );
    assert_eq!(produced, consumed.estimate);
    assert_eq!(consumed.wire, wire);

    // Four distinct claims and no interval anywhere in the artifact.
    assert_eq!(wire.estimate.structural_identification, "identified_by_stored_derivation");
    assert_eq!(wire.estimate.point_estimate, "plug_in_point_estimate");
    assert_eq!(wire.estimate.statistical_inference, "withheld_cell_not_licensed");
    assert_eq!(wire.estimate.interval_code, "cell_not_licensed");
    assert_eq!(wire.estimate.standard_error.as_ref().unwrap().label, "uncalibrated_standard_error");
    assert_eq!(wire.overlap.default, "independent");
    assert_eq!(wire.unseen_rule.name, "refuse");

    // The number is the enumerated truth within sampling error at 5,000 units.
    let truth = case.targets[0].expected_value.unwrap();
    assert!(
        (replayed_point.probability - truth).abs() <= 0.06,
        "{} vs {truth}",
        replayed_point.probability
    );

    // The io-error form carries registered reason codes and the same result.
    let again = consume_sampled_counterfactual_transport(
        &bytes,
        &Expected::default(),
        ConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(again.estimate, produced);
}

fn reseal(mut wire: Wire, edit: impl FnOnce(&mut Wire)) -> Vec<u8> {
    edit(&mut wire);
    wire.sealed().unwrap().export().unwrap()
}

fn consume(bytes: &[u8]) -> Result<antecedent_io::sampled_counterfactual_transport_artifact::ConsumedSampledCounterfactualTransport, ArtifactError>{
    consume_sampled_counterfactual_transport_with_limits(
        bytes,
        &Expected::default(),
        ConsumeLimits::default(),
        &ExecutionContext::for_tests(5),
    )
}

#[test]
fn resealed_semantic_mutations_are_refused() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let a = derive(&case, &case.targets[0]);
    let b = derive(&case, &case.targets[1]);
    let request = request_for(&case, &a, 2);
    let ctx = ExecutionContext::for_tests(3);
    let (wire, _) = sampled_counterfactual_transport_artifact(&a, &request, &[], &ctx).unwrap();
    let (other, _) =
        sampled_counterfactual_transport_artifact(&b, &request_for(&case, &b, 2), &[], &ctx)
            .unwrap();
    consume(&wire.export().unwrap()).expect("the unmodified artifact replays");

    // Counts edited without resealing: the data digest refuses.
    let mut edited = wire.clone();
    let cell = edited.tables[0].counts.iter().position(|c| *c > 3).unwrap();
    edited.tables[0].counts[cell] -= 3;
    let next = (cell + 1) % edited.tables[0].counts.len();
    edited.tables[0].counts[next] += 3;
    let error = consume(&to_bytes_unchecked(&edited)).unwrap_err();
    assert!(matches!(error, ArtifactError::DataIdentityMismatch), "{error:?}");

    // Counts edited then resealed: both digests are consistent, and the stored
    // estimate no longer follows from the stored counts.
    let bytes = reseal(wire.clone(), |w| {
        let counts = &mut w.tables[0].counts;
        let cell = counts.iter().position(|c| *c > 3).unwrap();
        counts[cell] -= 3;
        let next = (cell + 1) % counts.len();
        counts[next] += 3;
    });
    let error = consume(&bytes).unwrap_err();
    assert!(matches!(error, ArtifactError::EstimateMismatch { .. }), "{error:?}");
    assert_eq!(error.slot(), "estimate");

    // The overlap declaration flipped to shared units between tables that are
    // not the same observations: no covariance exists in the counts.
    let bytes = reseal(wire.clone(), |w| {
        w.overlap.default = "shared_units".into();
    });
    let error = consume(&bytes).unwrap_err();
    assert_eq!(refusal_code(&error), "route_not_supported", "{error:?}");
    // ... and to unknown: never read as independent.
    let bytes = reseal(wire.clone(), |w| {
        w.overlap.default = "unknown".into();
    });
    let error = consume(&bytes).unwrap_err();
    assert_eq!(refusal_code(&error), "sampling_dependence_unknown", "{error:?}");

    // Derivation swapped for another query's, resealed.
    assert_ne!(wire.derivation, other.derivation);
    let bytes = reseal(wire.clone(), |w| {
        w.derivation = other.derivation.clone();
    });
    let error = consume(&bytes).unwrap_err();
    assert!(matches!(error, ArtifactError::DerivationMismatch(_)), "{error:?}");
    assert_eq!(refusal_code(&error), "transport_not_certified");

    // The stored probability edited and resealed.
    let bytes = reseal(wire.clone(), |w| {
        let point = w.estimate.point.as_mut().unwrap();
        point.probability = (f64::from_bits(point.probability) + 0.01).to_bits();
    });
    let error = consume(&bytes).unwrap_err();
    assert!(matches!(error, ArtifactError::EstimateMismatch { field: "point" }), "{error:?}");

    // The stored standard error edited and resealed.
    let bytes = reseal(wire.clone(), |w| {
        let se = w.estimate.standard_error.as_mut().unwrap();
        se.value = (f64::from_bits(se.value) * 2.0).to_bits();
    });
    let error = consume(&bytes).unwrap_err();
    assert!(
        matches!(error, ArtifactError::EstimateMismatch { field: "standard_error" }),
        "{error:?}"
    );

    // An interval smuggled in as an inference claim never seals.
    let mut smuggled = wire.clone();
    smuggled.estimate.interval_code = "nominal_interval".into();
    assert!(smuggled.sealed().is_err());

    // The unseen-cell rule changed and resealed: the labels and tables differ.
    let bytes = reseal(wire.clone(), |w| {
        w.unseen_rule.name = "additive_smoothing".into();
        w.unseen_rule.parameter = Some(0.5f64.to_bits());
    });
    // No table has an unseen cell, so smoothing leaves the estimate unchanged
    // except for the recorded rule; the artifact still must not replay as stored
    // under a different evidence identity.
    let error = consume(&bytes).unwrap_err();
    assert!(matches!(error, ArtifactError::EstimateMismatch { .. }), "{error:?}");

    // An identity obtained independently of the bytes: another artifact's.
    let error = consume_sampled_counterfactual_transport_with_limits(
        &wire.export().unwrap(),
        &other.identity(),
        ConsumeLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(error, ArtifactError::ExpectedIdentityMismatch { .. }), "{error:?}");

    // Consumer limits are the consumer's own.
    let error = consume_sampled_counterfactual_transport_with_limits(
        &wire.export().unwrap(),
        &Expected::default(),
        ConsumeLimits { max_tables: 1, ..ConsumeLimits::default() },
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(error, ArtifactError::LimitsExceeded("count tables")), "{error:?}");
}

/// Encode a wire without sealing or shape checks, as a tampered file would be.
fn to_bytes_unchecked(wire: &Wire) -> Vec<u8> {
    antecedent_io::to_cbor(wire).unwrap()
}

#[test]
fn a_stale_version_or_foreign_feature_is_refused_before_the_payload() {
    let case = case_named("transportable_latent_district");
    let derivation = derive(&case, &case.targets[0]);
    let request = request_for(&case, &derivation, 4);
    let (wire, _) = sampled_counterfactual_transport_artifact(
        &derivation,
        &request,
        &[],
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let mut stale = wire.clone();
    stale.version = 2;
    let error = consume(&to_bytes_unchecked(&stale)).unwrap_err();
    assert!(matches!(error, ArtifactError::UnsupportedVersion { version: 2 }), "{error:?}");
    let mut foreign = wire;
    foreign.required_features = vec!["counterfactual_transport_v1".into()];
    let error = consume(&to_bytes_unchecked(&foreign)).unwrap_err();
    assert!(matches!(error, ArtifactError::UnsupportedSemantics(_)), "{error:?}");
}

fn declared(range: std::ops::Range<u32>, label: &str) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet)
        .with_observations(range.map(|i| ObservationId::from_key("units", &i.to_le_bytes())))
}

#[test]
fn the_ledger_declares_overlap_and_unknown_stays_unknown() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let derivation = derive(&case, &case.targets[0]);
    let request = request_for(&case, &derivation, 6);
    assert!(request.tables.len() >= 3);
    let keys: Vec<_> = request.tables.iter().map(SampledRegimeCounts::key).collect();
    let ctx = ExecutionContext::for_tests(3);
    let run = |declaration: OverlapDeclaration| {
        let mut r = request.clone();
        r.overlap = declaration;
        sampled_counterfactual_transport_artifact(&derivation, &r, &[], &ctx)
    };

    // `source` builds a ledger with one source per table; `layout` gives the
    // observation range of each (None = observations not declared).
    let build = |layout: &[Option<std::ops::Range<u32>>]| {
        let mut ledger = EvidenceLedger::new();
        let mut bound = Vec::new();
        for (i, range) in layout.iter().enumerate() {
            let label = format!("table{i}");
            let declaration = match range {
                Some(r) => declared(r.clone(), &label),
                None => SourceDeclaration::new(label, SourceKind::ObservationSet),
            };
            bound.push((keys[i].clone(), ledger.add_source(declaration).unwrap()));
        }
        overlap_declaration_from_ledger(&ledger, &bound).unwrap()
    };
    let n = u32::try_from(keys.len()).expect("a handful of tables");

    // Disjoint declared observations behind every table: independent.
    let disjoint: Vec<_> = (0..n).map(|i| Some(i * 10_000..i * 10_000 + 5_000)).collect();
    let independent = build(&disjoint);
    assert!(!independent.pairs().is_empty());
    for pair in independent.pairs() {
        assert_eq!(pair.relation, DatasetOverlap::Independent);
    }
    let (wire, estimate) = run(independent).unwrap();
    assert_eq!(wire.overlap.default, "unknown");
    assert!(estimate.point.is_some());

    // A source whose observations are not declared is unknown; the estimator
    // refuses rather than assuming independence.
    let mut layout = disjoint.clone();
    layout[1] = None;
    let declaration = build(&layout);
    assert!(declaration.pairs().iter().any(|p| p.relation == DatasetOverlap::Unknown));
    let error = run(declaration).unwrap_err();
    assert_eq!(refusal_code(&error), "sampling_dependence_unknown", "{error:?}");

    // Overlapping declared observations: shared units, which counts of
    // different tables cannot give a covariance for.
    let mut layout = disjoint;
    layout[1] = Some(2_500..7_500);
    let declaration = build(&layout);
    assert!(declaration.pairs().iter().any(|p| p.relation == DatasetOverlap::SharedUnits));
    let error = run(declaration).unwrap_err();
    assert_eq!(refusal_code(&error), "route_not_supported", "{error:?}");
}
