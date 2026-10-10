//! X6: the checked adapter from `PriorCatalog` posterior sources to a future candidate
//! signal likelihood. All expected draws are enumerated by hand from the documented
//! rule: sources are processed in artifact-id order; each source's draws are mapped to
//! the target coordinate and shrunk toward the pooled weighted mean
//! (`weight * strength`) by `1 - conflict_shrinkage`; `resolution` draws are allocated
//! by largest remainder and each source contributes stratified quantiles
//! `sorted[floor((2j + 1) * len / (2 * count))]`.

use std::sync::Arc;

use antecedent_design::preposterior::DecisionPrior;
use antecedent_design::prior_signal::{
    CandidateSignalSource, NativeSignalFamily, PriorSignalError, PriorSignalRequest,
    PriorSourceInput, SourceToTarget, TransportPolicyDecl, adapt_prior_to_signal,
};
use antecedent_design::signal::SignalProvider;
use antecedent_design::signal::{NativeBinomialSignal, SignalLimits, SignalRequest};
use antecedent_io::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog, PriorSourceMeta,
    PriorSourceRef, TargetDesign,
};

fn meta(id: &str, outcome: &str) -> PriorSourceMeta {
    PriorSourceMeta::new(
        id,
        EstimandFingerprint::new("ate", "t", outcome),
        "NonparametricallyIdentified",
    )
    .with_design(vec![
        DesignVariableSummary::new("t", DesignVariableRole::Treatment),
        DesignVariableSummary::new("y", DesignVariableRole::Outcome),
    ])
}

fn catalog() -> PriorCatalog {
    PriorCatalog::from_sources(vec![
        PriorSourceRef::from_meta(meta("A", "y")),
        PriorSourceRef::from_meta(meta("B", "y")),
        // Estimand mismatch with no declared mapping: the catalog rejects it.
        PriorSourceRef::from_meta(meta("wrong", "other_y")),
    ])
}

fn target() -> TargetDesign {
    TargetDesign::new(EstimandFingerprint::new("ate", "t", "y"), ["t", "y"])
}

fn source(id: &str, draws: &[f64], weight: f64, observations: &[&str]) -> PriorSourceInput {
    PriorSourceInput {
        artifact_id: id.into(),
        source_population: "target".into(),
        draws: draws.to_vec(),
        weight,
        prior_strength: 1.0,
        conflict_shrinkage: 0.0,
        lineage: vec![format!("lineage:{id}")],
        observation_ids: observations.iter().map(|s| (*s).to_owned()).collect(),
        source_to_target: SourceToTarget::Identity,
    }
}

fn request(sources: Vec<PriorSourceInput>, resolution: usize) -> PriorSignalRequest {
    PriorSignalRequest {
        prior_id: "prior-1".into(),
        sources,
        signal: CandidateSignalSource::Native(NativeSignalFamily::Binomial),
        target_population: "target".into(),
        transport: None,
        candidate_observation_ids: vec!["future-1".into()],
        resolution,
    }
}

fn two_sources() -> Vec<PriorSourceInput> {
    vec![
        source("A", &[0.2, 0.4], 1.0, &["obs-a1", "obs-a2"]),
        source("B", &[0.6, 0.8], 1.0, &["obs-b1"]),
    ]
}

fn draws_of(prior: &DecisionPrior<f64>) -> Vec<f64> {
    match prior {
        DecisionPrior::Draws(d) => d.clone(),
        DecisionPrior::Normal { .. } => panic!("expected draws"),
    }
}

fn close_all(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn refusal_detail(error: &PriorSignalError) -> String {
    error.to_refusal().detail
}

#[test]
fn x6_equal_weight_sources_pool_into_the_enumerated_draws_and_a_signal() {
    let result = adapt_prior_to_signal(&catalog(), &target(), &request(two_sources(), 4)).unwrap();
    // quotas 2 and 2; A: sorted [0.2, 0.4] -> idx 0, 1; B: [0.6, 0.8] -> idx 0, 1.
    assert!(close_all(&draws_of(&result.prior), &[0.2, 0.4, 0.6, 0.8]));
    let d = &result.diagnostics;
    assert_eq!(d.source_ids, vec!["A".to_owned(), "B".to_owned()]);
    assert!(close_all(&d.effective_weights, &[0.5, 0.5]));
    assert_eq!(d.draws_allocated, vec![2, 2]);
    // pooled mean = 0.5 * mean(0.2, 0.4) + 0.5 * mean(0.6, 0.8) = 0.15 + 0.35.
    assert!((d.pooled_mean - 0.5).abs() < 1e-12);
    assert_eq!(d.lineage, vec!["lineage:A".to_owned(), "lineage:B".to_owned()]);
    assert_eq!(d.observations_checked, 3);
    assert!(d.overlapping_observations.is_empty());
    assert_eq!(result.prior_id, "prior-1");
    // The prior is paired with a real signal provider that answers an exact request.
    let signal_request = SignalRequest {
        candidate_id: "study".into(),
        prior_id: result.prior_id.clone(),
        state_quantity: antecedent_core::ScientificQuantity {
            variable_id: "schema:p".into(),
            variable_name: "p".into(),
            role: antecedent_core::QuantityRole::Outcome,
            units: "probability".into(),
            population_id: "target".into(),
            regime_id: "observational".into(),
            horizon: 0,
            functional_id: "state".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        },
        observation_quantity: antecedent_core::ScientificQuantity {
            variable_id: "schema:k".into(),
            variable_name: "k".into(),
            role: antecedent_core::QuantityRole::Outcome,
            units: "count".into(),
            population_id: "target".into(),
            regime_id: "observational".into(),
            horizon: 0,
            functional_id: "successes".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        },
        sample_size: 10,
        rng_seed: 1,
        evidence_lineage: d.lineage.clone(),
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    };
    assert!(result.provider.prepare(&signal_request).is_ok());
    assert!(NativeBinomialSignal.prepare(&signal_request).is_ok());
}

#[test]
fn x6_weights_and_prior_strength_set_the_allocation() {
    // Effective weights 3 * 1 and 1 * 1 -> quotas 3 and 1 of 4.
    // A: sorted [0.2, 0.4], count 3 -> idx floor(2/6 * ...) = 0, 1, 1 -> 0.2, 0.4, 0.4.
    // B: sorted [0.6, 0.8], count 1 -> idx floor(1 * 2 / 2) = 1 -> 0.8.
    let sources =
        vec![source("A", &[0.2, 0.4], 3.0, &["obs-a"]), source("B", &[0.6, 0.8], 1.0, &["obs-b"])];
    let result = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap();
    assert!(close_all(&draws_of(&result.prior), &[0.2, 0.4, 0.4, 0.8]));
    assert_eq!(result.diagnostics.draws_allocated, vec![3, 1]);
    // pooled = 0.75 * 0.3 + 0.25 * 0.7 = 0.4.
    assert!((result.diagnostics.pooled_mean - 0.4).abs() < 1e-12);

    // Halving B's strength: weights 3 and 0.5 -> same relative order, quotas 3.43 and 0.57
    // of 4 -> floors 3 and 0, remainder to the larger fraction (B, 0.57) -> [3, 1].
    let mut weak =
        vec![source("A", &[0.2, 0.4], 3.0, &["obs-a"]), source("B", &[0.6, 0.8], 1.0, &["obs-b"])];
    weak[1].prior_strength = 0.5;
    let result = adapt_prior_to_signal(&catalog(), &target(), &request(weak, 4)).unwrap();
    assert_eq!(result.diagnostics.draws_allocated, vec![3, 1]);
    assert!((result.diagnostics.effective_weights[1] - 0.5 / 3.5).abs() < 1e-12);
}

#[test]
fn x6_conflict_shrinkage_pulls_a_source_to_the_pooled_mean() {
    let mut sources = two_sources();
    sources[1].conflict_shrinkage = 1.0;
    let result = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap();
    // pooled mean 0.5; A unshrunk (0.2, 0.4); B collapsed to 0.5, 0.5.
    assert!(close_all(&draws_of(&result.prior), &[0.2, 0.4, 0.5, 0.5]));
}

#[test]
fn x6_source_to_target_map_is_applied_before_pooling() {
    let mut sources = vec![source("A", &[0.0, 1.0], 1.0, &["obs-a"])];
    sources[0].source_to_target = SourceToTarget::Affine { intercept: 0.5, slope: 0.25 };
    let result = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 2)).unwrap();
    // 0.5 + 0.25 * {0, 1} = {0.5, 0.75}; count 2 of len 2 -> idx 0, 1.
    assert!(close_all(&draws_of(&result.prior), &[0.5, 0.75]));
}

#[test]
fn x6_source_order_does_not_change_the_prior_or_its_digest() {
    let forward = adapt_prior_to_signal(&catalog(), &target(), &request(two_sources(), 4)).unwrap();
    let mut reversed_sources = two_sources();
    reversed_sources.reverse();
    let reversed =
        adapt_prior_to_signal(&catalog(), &target(), &request(reversed_sources, 4)).unwrap();
    assert_eq!(draws_of(&forward.prior), draws_of(&reversed.prior));
    assert_eq!(forward.diagnostics, reversed.diagnostics);
    // A different weight changes the digest.
    let mut heavier = two_sources();
    heavier[0].weight = 2.0;
    let other = adapt_prior_to_signal(&catalog(), &target(), &request(heavier, 4)).unwrap();
    assert_ne!(forward.diagnostics.source_digest, other.diagnostics.source_digest);
}

#[test]
fn x6_incompatible_or_unknown_artifacts_refuse() {
    let sources = vec![source("wrong", &[0.1, 0.2], 1.0, &["obs-w"])];
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap_err();
    assert!(matches!(error, PriorSignalError::IncompatibleArtifact { .. }), "{error:?}");
    assert_eq!(refusal_detail(&error), "prior_signal.incompatible_artifact");

    let sources = vec![source("ghost", &[0.1, 0.2], 1.0, &["obs-g"])];
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::ArtifactNotInCatalog("ghost".into()));
    assert_eq!(refusal_detail(&error), "prior_signal.artifact_not_in_catalog");
}

#[test]
fn x6_a_coefficient_prior_or_unlock_list_alone_is_not_a_signal() {
    let mut req = request(two_sources(), 4);
    req.signal = CandidateSignalSource::CoefficientPriorOnly;
    let error = adapt_prior_to_signal(&catalog(), &target(), &req).unwrap_err();
    assert_eq!(error, PriorSignalError::CoefficientPriorNotSignal);
    assert_eq!(refusal_detail(&error), "prior_signal.coefficient_prior_not_signal");

    let mut req = request(two_sources(), 4);
    req.signal = CandidateSignalSource::UnlockListOnly(vec!["z".into(), "w".into()]);
    let error = adapt_prior_to_signal(&catalog(), &target(), &req).unwrap_err();
    assert_eq!(error, PriorSignalError::UnlockListNotSignal(2));
    assert_eq!(refusal_detail(&error), "prior_signal.unlock_list_not_signal");
}

#[test]
fn x6_prior_data_overlap_refuses() {
    // The candidate study would reuse an observation the prior source was fit on.
    let mut req = request(two_sources(), 4);
    req.candidate_observation_ids = vec!["future-1".into(), "obs-a2".into()];
    let error = adapt_prior_to_signal(&catalog(), &target(), &req).unwrap_err();
    assert_eq!(error, PriorSignalError::ObservationReuse(vec!["obs-a2".into()]));
    assert_eq!(refusal_detail(&error), "prior_signal.observation_reuse");

    // Two prior sources fit on the same observation would count it twice.
    let sources = vec![
        source("A", &[0.2, 0.4], 1.0, &["obs-shared", "obs-a"]),
        source("B", &[0.6, 0.8], 1.0, &["obs-shared"]),
    ];
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::PriorSourcesOverlap(vec!["obs-shared".into()]));
    assert_eq!(refusal_detail(&error), "prior_signal.prior_sources_overlap");

    // A source that cannot show which observations it used cannot exclude reuse.
    let sources = vec![source("A", &[0.2, 0.4], 1.0, &[])];
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::ObservationLineageMissing("A".into()));
}

#[test]
fn x6_population_change_needs_a_declared_transport_policy() {
    let mut sources = two_sources();
    sources[0].source_population = "other-site".into();
    let error =
        adapt_prior_to_signal(&catalog(), &target(), &request(sources.clone(), 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::TransportPolicyRequired("A".into()));
    assert_eq!(error.to_refusal().detail, "prior_signal.transport_policy_required");

    let mut req = request(sources, 4);
    req.transport = Some(TransportPolicyDecl { policy_id: "policy:reweight".into() });
    let result = adapt_prior_to_signal(&catalog(), &target(), &req).unwrap();
    assert_eq!(result.diagnostics.transport_policy_id.as_deref(), Some("policy:reweight"));
}

#[test]
fn x6_invalid_parameters_and_resolution_refuse() {
    let bad = |edit: &dyn Fn(&mut PriorSourceInput)| {
        let mut sources = two_sources();
        edit(&mut sources[0]);
        adapt_prior_to_signal(&catalog(), &target(), &request(sources, 4)).unwrap_err()
    };
    assert_eq!(bad(&|s| s.weight = -1.0), PriorSignalError::InvalidParameter("weight"));
    assert_eq!(
        bad(&|s| s.prior_strength = 0.0),
        PriorSignalError::InvalidParameter("prior_strength")
    );
    assert_eq!(
        bad(&|s| s.conflict_shrinkage = 1.5),
        PriorSignalError::InvalidParameter("conflict_shrinkage")
    );
    assert_eq!(bad(&|s| s.draws.clear()), PriorSignalError::InvalidParameter("draws"));
    assert_eq!(
        bad(&|s| s.source_to_target = SourceToTarget::Affine { intercept: 0.0, slope: 0.0 }),
        PriorSignalError::InvalidParameter("source_to_target")
    );
    assert_eq!(bad(&|s| s.lineage.clear()), PriorSignalError::LineageMissing("A".into()));
    let error =
        adapt_prior_to_signal(&catalog(), &target(), &request(two_sources(), 0)).unwrap_err();
    assert_eq!(error, PriorSignalError::ResolutionInvalid);
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(Vec::new(), 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::NoSources);
    let duplicated = vec![source("A", &[0.2], 1.0, &["o1"]), source("A", &[0.3], 1.0, &["o2"])];
    let error = adapt_prior_to_signal(&catalog(), &target(), &request(duplicated, 4)).unwrap_err();
    assert_eq!(error, PriorSignalError::DuplicateSource("A".into()));
}

#[test]
fn x6_a_provider_can_supply_the_future_signal() {
    let mut req = request(two_sources(), 4);
    req.signal = CandidateSignalSource::Provider(Arc::new(NativeBinomialSignal));
    assert!(adapt_prior_to_signal(&catalog(), &target(), &req).is_ok());
    req.signal =
        CandidateSignalSource::Native(NativeSignalFamily::GaussianMean { noise_variance: -1.0 });
    let error = adapt_prior_to_signal(&catalog(), &target(), &req).unwrap_err();
    assert_eq!(error, PriorSignalError::InvalidParameter("noise_variance"));
}
