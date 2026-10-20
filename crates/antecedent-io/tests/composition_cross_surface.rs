//! The aligned joint draw, the external finite response, and the slice cases in
//! `slice_cases.json`: the Gaussian line as a joint draw, the finite-discrete
//! adjustment, and the named feedback policy.
//!
//! The fixtures in `conformance/cross_surface/` are read-only here. `py_*` files were built by
//! `python/tests/generate_cross_surface_fixtures.py`, `rust_*` files by the ignored
//! `regenerate_rust_fixtures` tests of `cross_surface_joint_law.rs` and
//! `cross_surface_external_claim.rs`; a missing file fails loudly. Every expected identity is
//! built from constants below, never read back from the artifact.
//!
//! Composes: `cross_surface_joint_law.rs`, `cross_surface_external_claim.rs` and
//! `external_claim_artifact.rs`, plus the `external_binding` and `external` unit tests of
//! `antecedent-core`. Python twins: `python/tests/test_composition_exit_joint_and_external.py`.
//!
//! Hand-derived joint truth: two equally weighted aligned draws (0, 0) and (1, 2) give means
//! (1/2, 1); `E[XY] = (0 + 2) / 2 = 1`, so `Cov = 1 - 1/2 * 1 = 1/2`; `E[max(X, Y)] =
//! (0 + 2) / 2 = 1`. Independently paired marginals would pair every x with every y:
//! `E[XY] = (0 + 0 + 0 + 2) / 4 = 1/2 = E[X] E[Y]`, i.e. zero covariance and a different
//! nonlinear value, which is why they are refused.
//!
//! Hand-derived external truth: `E[Y | do(a)] = 1 + 2a` on `a` in `0, 1, 2` is `[1, 3, 5]`.
//!
//! Hand-derived slice truth, from `slice_cases.json`: the line `y = 1 + 2t` has contrast 2;
//! the frequency table sums the confounder out to active `0.5625`, control `0.25`, contrast
//! `0.3125`; the two posterior draws mix to `0.4375`.

use std::sync::Arc;

use antecedent_contract::mixed_module::{CutFeedback, MixedModulePolicy, MixedModuleRefusal};
use antecedent_contract::{BoundExternalClaim, BoundTrustLevel};
use antecedent_contract::{
    CheckedCausalContract, CompositionStage, DistributionMeaning, ExternalBindingError,
    ExternalCapability, ExternalResponse, ExternalResult, ExternalResultHeader,
    ExternalScientificObject, ExternalTrustState, ExternalUncertaintyMeaning,
    ExternalVerificationError, LawProviderContract, OBSERVATIONAL_REGIME, ProviderObjectIdentity,
    QuantityMismatch, QuantityRole, ScientificQuantity, VerificationProbe, VerificationProbeKind,
    bind_external_result, verify_external_object,
};
use antecedent_core::{
    ExecutionContext, IdentificationStatus, RegimeId, SupportStatus, Value, VariableId,
};
use antecedent_expr::{
    Assignment, CausalExprArena, DiscreteAxis, DomainRef, ExactDiscreteLaw, ExactEvaluationLimits,
    ExactEvaluationPlan, ExactEvaluationSpec, ExactTransportData, ExprNode, LawTolerance,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::exact_law_wire::ExactLawWire;
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust, lineage_wire, values_digest,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_test_support::slice_cases::{self, FiniteDiscrete};
use antecedent_test_support::{
    assert_all_close, assert_close, conformance_dir, flip_last_byte, read_fixture_bytes, within,
};

/// Closeness of exact arithmetic on hand-derived values.
const TOLERANCE: f64 = 1e-12;

fn fixture_dir() -> std::path::PathBuf {
    conformance_dir("cross_surface")
}

fn read_fixture(name: &str) -> Vec<u8> {
    read_fixture_bytes(
        &fixture_dir(),
        name,
        "run \
            `python python/tests/generate_cross_surface_fixtures.py` from the repo root",
    )
}

// ------------------------------------------------------------------------ joint draw

fn joint_quantities(second: &str) -> Vec<ScientificQuantity> {
    ["x", second]
        .iter()
        .map(|name| ScientificQuantity {
            variable_id: (*name).into(),
            variable_name: (*name).into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        })
        .collect()
}

fn joint_identity(
    meaning: DistributionMeaningWire,
    alignment: DrawAlignment,
    second: &str,
) -> DistributionIdentity {
    DistributionIdentity::new(
        meaning,
        &joint_quantities(second),
        alignment,
        DistributionProvenance {
            source_id: "enumerated-law".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "law-snapshot".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

fn expected_joint_identity() -> DistributionIdentity {
    joint_identity(DistributionMeaningWire::InterventionalPredictive, DrawAlignment::Joint, "y")
}

fn law_with(identity: DistributionIdentity, draws: Vec<f64>) -> DistributionArtifact {
    let rows = draws.len() / 2;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn assert_joint_truth(artifact: &DistributionArtifact) {
    assert_close(artifact.mean(0).unwrap(), 0.5, TOLERANCE);
    assert_close(artifact.mean(1).unwrap(), 1.0, TOLERANCE);
    assert_close(artifact.covariance(0, 1).unwrap(), 0.5, TOLERANCE);
    assert_close(artifact.joint_expectation(0, 1, |x, y| x * y).unwrap(), 1.0, TOLERANCE);
    // A nonlinear utility of the aligned pair: E[max(X, Y)] = (0 + 2) / 2.
    assert_close(artifact.joint_expectation(0, 1, f64::max).unwrap(), 1.0, TOLERANCE);
}

/// Box 3: one aligned joint draw artifact round-trips Python -> Rust and Rust -> Python bytes,
/// reproduces covariance, a nonlinear utility and the published summary, and refuses both
/// independently paired marginals and a wrong distribution meaning.
#[test]
fn joint_draw_round_trips_and_refuses_marginals_and_wrong_meaning() {
    let python_bytes = read_fixture("py_joint_law.bin");
    let rust_bytes = read_fixture("rust_joint_law.bin");
    let expected = expected_joint_identity();

    let from_python = DistributionArtifact::from_bytes(&python_bytes, &expected).unwrap();
    let from_rust = DistributionArtifact::from_bytes(&rust_bytes, &expected).unwrap();
    for artifact in [&from_python, &from_rust] {
        assert_joint_truth(artifact);
        assert_eq!(artifact.metadata().identity, expected);
        assert_eq!(artifact.metadata().trust, DistributionTrust::Unverified);
        assert_eq!(artifact.metadata().calibration, DistributionCalibration::Exact);
        assert_eq!(artifact.shape(), [2, 2]);
        assert_all_close(artifact.draws(), &[0.0, 0.0, 1.0, 2.0], TOLERANCE);
    }
    assert_all_close(from_python.draws(), from_rust.draws(), TOLERANCE);

    // Re-export on this surface and re-consume: the round trip preserves every value.
    let again = from_python.to_bytes("joint-law").unwrap();
    let reloaded = DistributionArtifact::from_bytes(&again, &expected).unwrap();
    assert_joint_truth(&reloaded);

    // Independently paired marginals: the four (x, y) pairs {0,1} x {0,2} give E[XY] = 1/2,
    // a different nonlinear value, and the artifact refuses every joint operation on them.
    let paired = law_with(
        joint_identity(
            DistributionMeaningWire::InterventionalPredictive,
            DrawAlignment::IndependentMarginals,
            "y",
        ),
        vec![0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0],
    );
    assert_close(paired.mean(0).unwrap(), 0.5, TOLERANCE);
    let manual: f64 = paired.draws().chunks_exact(2).map(|row| row[0] * row[1]).sum::<f64>() / 4.0;
    assert_close(manual, 0.5, TOLERANCE);
    assert!(!within(manual, 1.0, TOLERANCE), "pairing changes the nonlinear value");
    let covariance = paired.covariance(0, 1).unwrap_err().to_string();
    assert!(covariance.contains("aligned_joint_draws.marginals_not_joint"), "{covariance}");
    let nonlinear = paired.joint_expectation(0, 1, |x, y| x * y).unwrap_err().to_string();
    assert!(nonlinear.contains("aligned_joint_draws.marginals_not_joint"), "{nonlinear}");

    // Wrong distribution meaning, changed quantity identity and changed bytes refuse.
    for changed in [
        joint_identity(DistributionMeaningWire::Bootstrap, DrawAlignment::Joint, "y"),
        joint_identity(
            DistributionMeaningWire::CausalFunctionalPosterior,
            DrawAlignment::Joint,
            "y",
        ),
        joint_identity(
            DistributionMeaningWire::InterventionalPredictive,
            DrawAlignment::Joint,
            "z",
        ),
    ] {
        for bytes in [&python_bytes, &rust_bytes] {
            let error = DistributionArtifact::from_bytes(bytes, &changed).unwrap_err();
            assert!(error.to_string().contains("aligned_joint_draws.identity_expected"), "{error}");
        }
    }
    for bytes in [&python_bytes, &rust_bytes] {
        assert!(DistributionArtifact::from_bytes(&flip_last_byte(bytes), &expected).is_err());
    }
}

// ------------------------------------------------------------------- external response

/// Outcome `y` in mmHg; the Python surface derives `variable_id == variable_name == "y"`.
fn quantity(dose: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn provider_identity(request: &str) -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "curve".into(),
        version_id: "v3".into(),
        snapshot_id: "snap-9".into(),
        request_id: request.into(),
    }
}

fn law_object(request: &str) -> ExternalScientificObject {
    ExternalScientificObject::Law(LawProviderContract {
        identity: provider_identity(request),
        quantities: (0..3).map(quantity).collect(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    })
}

fn causal_contract() -> CheckedCausalContract {
    CheckedCausalContract {
        graph_id: "graph-1".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: (0..3).map(quantity).collect(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec!["factor:z".into()],
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    }
}

fn response() -> ExternalResponse {
    ExternalResponse {
        header: ExternalResultHeader {
            object: law_object("req-1"),
            graph_id: "graph-1".into(),
            quantities: (0..3).map(quantity).collect(),
            evidence_ids: vec!["factor:z".into()],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        // Closed form: E[Y | do(a)] = 1 + 2a.
        values: vec![1.0, 3.0, 5.0],
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![
            SupportStatus::Supported,
            SupportStatus::Supported,
            SupportStatus::OutsideEmpiricalSupport,
        ]),
    }
}

fn bind(response: ExternalResponse) -> Result<BoundExternalClaim, ExternalBindingError> {
    bind_external_result(&causal_contract(), &ExternalResult::Response(response))
}

/// Retarget one coordinate in both the response header and the provider object, as a provider
/// that really labelled its values that way would.
fn relabel(response: &mut ExternalResponse, index: usize, edit: impl Fn(&mut ScientificQuantity)) {
    edit(&mut response.header.quantities[index]);
    if let ExternalScientificObject::Law(law) = &mut response.header.object {
        edit(&mut law.quantities[index]);
    }
}

fn probe(kind: VerificationProbeKind) -> VerificationProbe {
    VerificationProbe { kind, observed: 1.0, expected: 1.0, tolerance: 0.0 }
}

/// The probes a mean-capable law needs: known truth, shape, support and moments.
fn law_probes() -> Vec<VerificationProbe> {
    use VerificationProbeKind as P;
    [P::KnownTruth, P::Shape, P::Support, P::Moments].into_iter().map(probe).collect()
}

/// Consumer-retained identity, written from constants rather than any artifact.
fn expected_claim_identity() -> ExternalClaimIdentity {
    let mut identity = ExternalClaimIdentity {
        causal_contract_id: "checked-contract".into(),
        graph_id: "graph-1".into(),
        identification: "nonparametrically_identified".into(),
        quantities: (0..3).map(|d| ScientificQuantityWire::from(&quantity(d))).collect(),
        provider_id: "lab".into(),
        object_id: "curve".into(),
        version_id: "v3".into(),
        snapshot_id: "snap-9".into(),
        request_id: "req-1".into(),
        values_blake3: values_digest(&[1.0, 3.0, 5.0]),
        trust: ExternalClaimTrust::ExternallyAttested,
        point_status: vec![
            "supported".into(),
            "supported".into(),
            "outside_empirical_support".into(),
        ],
        uncertainty_method: None,
        evidence_ids: vec!["factor:z".into()],
        assumption_ids: vec!["ignorability".into()],
        equivalence_ids: vec![],
        lineage: lineage_wire(&[
            ("contract:checked-contract", "causal_contract", &[]),
            ("evidence:factor:z", "evidence", &["contract:checked-contract"]),
            ("provider:external:lab/curve@v3#snap-9", "external_provider", &["evidence:factor:z"]),
            (
                "claim",
                "claim",
                &["contract:checked-contract", "provider:external:lab/curve@v3#snap-9"],
            ),
        ])
        .unwrap(),
        provider_meaning: "interventional_predictive".into(),
        capabilities: vec!["mean".into()],
        provider_fingerprint: String::new(),
        verification: None,
    };
    identity.provider_fingerprint = identity.compute_fingerprint();
    identity
}

/// Box 4: one external finite interventional response is bound to a checked causal contract,
/// inspected and exported, then an observational law, a mismatched coordinate and an unverified
/// request are each refused with a structured detail; coordinate-level support and full
/// provenance are shown on the consumed artifacts of both surfaces.
#[test]
fn external_response_binds_exports_and_refuses_with_structured_details() {
    // Bound claim: values, per-coordinate support, trust, never native, provenance.
    let claim = bind(response()).unwrap();
    assert_all_close(claim.values().unwrap(), &[1.0, 3.0, 5.0], TOLERANCE);
    assert_eq!(
        claim.point_status(),
        Some(
            &[
                SupportStatus::Supported,
                SupportStatus::Supported,
                SupportStatus::OutsideEmpiricalSupport
            ][..]
        )
    );
    assert_eq!(claim.support_status(), Some(SupportStatus::OutsideEmpiricalSupport));
    assert_eq!(claim.trust(), BoundTrustLevel::ExternallyAttested);
    assert!(!claim.is_native_estimation());
    assert_eq!(claim.provenance_label(), "external:lab/curve@v3#snap-9");
    let chain = claim.provenance_chain("checked-contract").unwrap();
    let behind = chain.stages_behind("claim").unwrap();
    for stage in [
        CompositionStage::CausalContract,
        CompositionStage::Evidence,
        CompositionStage::ExternalProvider,
    ] {
        assert!(behind.contains(&stage), "missing lineage stage {stage:?}");
    }

    // Both surfaces' bytes load under the constants-built identity and carry the same claim.
    let expected = expected_claim_identity();
    for name in ["py_external_claim.bin", "rust_external_claim.bin"] {
        let bytes = read_fixture(name);
        let loaded = ExternalClaimArtifact::from_bytes(&bytes, &expected).unwrap();
        assert_all_close(loaded.values(), &[1.0, 3.0, 5.0], TOLERANCE);
        let identity = &loaded.metadata().identity;
        assert_eq!(identity, &expected, "{name}");
        assert_eq!(identity.quantities.len(), identity.point_status.len());
        assert_eq!(
            identity.point_status,
            ["supported", "supported", "outside_empirical_support"],
            "coordinate-level support, not one pooled flag: {name}"
        );
        assert_eq!(identity.trust, ExternalClaimTrust::ExternallyAttested);
        assert!(!loaded.metadata().native_estimation);
        assert_eq!(loaded.provenance_label(), "external:lab/curve@v3#snap-9");
        // Full provenance: contract, evidence, provider execution, claim, with Merkle digests
        // that each child records for its parents.
        let ids: Vec<&str> = identity.lineage.iter().map(|link| link.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "contract:checked-contract",
                "evidence:factor:z",
                "provider:external:lab/curve@v3#snap-9",
                "claim"
            ]
        );
        for link in &identity.lineage {
            assert_eq!(link.digest.len(), 64);
            for (parent, digest) in link.parents.iter().zip(&link.parent_digests) {
                let found = identity.lineage.iter().find(|candidate| &candidate.id == parent);
                assert_eq!(found.map(|candidate| &candidate.digest), Some(digest));
            }
        }
        // Inspect/export: re-exporting on this surface preserves the claim.
        let again = loaded.to_bytes("claim").unwrap();
        let reloaded = ExternalClaimArtifact::from_bytes(&again, &expected).unwrap();
        assert_all_close(reloaded.values(), loaded.values(), TOLERANCE);
        // A different retained identity or changed bytes refuse.
        let mut other = expected.clone();
        other.snapshot_id = "other-snapshot".into();
        other.provider_fingerprint = other.compute_fingerprint();
        assert!(ExternalClaimArtifact::from_bytes(&bytes, &other).is_err(), "{name}");
        assert!(ExternalClaimArtifact::from_bytes(&flip_last_byte(&bytes), &expected).is_err());
    }

    external_response_refusals_and_verification();
}

/// The refusal half of box 4: observational law, mismatched coordinates and unverified requests.
fn external_response_refusals_and_verification() {
    // Observational law offered for coordinate 1 with no checked equivalence.
    let mut observational = response();
    relabel(&mut observational, 1, |q| q.regime_id = OBSERVATIONAL_REGIME.into());
    assert_eq!(
        bind(observational).unwrap_err(),
        ExternalBindingError::UncheckedObservationalLaw(1)
    );

    // Mismatched coordinate: kPa for mmHg is refused at that coordinate, never converted.
    let mut wrong_units = response();
    relabel(&mut wrong_units, 1, |q| q.units = "kPa".into());
    assert_eq!(
        bind(wrong_units).unwrap_err(),
        ExternalBindingError::CoordinateMismatch(1, QuantityMismatch::Units)
    );
    let mut wrong_population = response();
    relabel(&mut wrong_population, 2, |q| q.population_id = "source".into());
    assert_eq!(
        bind(wrong_population).unwrap_err(),
        ExternalBindingError::CoordinateMismatch(2, QuantityMismatch::Population)
    );

    // Unverified request: probes missing or failing never verify, and a receipt for another
    // request (req-2) does not cover this one (req-1).
    assert_eq!(
        verify_external_object(&law_object("req-1"), &[]),
        Err(ExternalVerificationError::MissingProbe(VerificationProbeKind::KnownTruth))
    );
    let mut failing = law_probes();
    failing[0].observed = 1.5;
    assert_eq!(
        verify_external_object(&law_object("req-1"), &failing),
        Err(ExternalVerificationError::FailedProbe(VerificationProbeKind::KnownTruth))
    );
    let other_request = verify_external_object(&law_object("req-2"), &law_probes()).unwrap();
    let mut unverified = response();
    unverified.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(other_request));
    assert_eq!(bind(unverified).unwrap_err(), ExternalBindingError::TrustMismatch);
    // The receipt for the exact request does bind, and the claim records the stronger trust.
    let exact = verify_external_object(&law_object("req-1"), &law_probes()).unwrap();
    let mut verified = response();
    verified.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(exact));
    let claim = bind(verified).unwrap();
    assert_eq!(claim.trust(), BoundTrustLevel::ExactRequestVerified);
    assert!(!claim.is_native_estimation());
    assert_eq!(claim.verification().len(), 4);
    // A native-licensed label can never be claimed by a provider.
    let mut native = response();
    native.header.trust = ExternalTrustState::NativeLicensed;
    assert_eq!(bind(native).unwrap_err(), ExternalBindingError::TrustMismatch);
}

// ------------------------------------------------------------------------ slice cases

const TREATMENT: u32 = 0;
const CONFOUNDER: u32 = 1;
const OUTCOME: u32 = 2;

/// Cell probabilities of a table of counts.
fn probabilities(counts: &[u32]) -> Vec<f64> {
    let total: u32 = counts.iter().sum();
    counts.iter().map(|count| f64::from(*count) / f64::from(total)).collect()
}

fn wide_counts(counts: &[u32]) -> Vec<u64> {
    counts.iter().map(|count| u64::from(*count)).collect()
}

fn binary_axes() -> [DiscreteAxis; 3] {
    [TREATMENT, CONFOUNDER, OUTCOME].map(|raw| DiscreteAxis {
        variable: VariableId::from_raw(raw),
        values: Arc::from([Value::Int64(0), Value::Int64(1)]),
    })
}

fn adjustment(population: &str, regime: RegimeId) -> (CausalExprArena, antecedent_expr::ExprId) {
    let mut arena = CausalExprArena::new();
    let population = arena.intern_population(population);
    let empty = arena.empty_var_set();
    let none = arena.intern_intervention_set([]);
    let confounder = arena.intern_var_set([VariableId::from_raw(CONFOUNDER)]);
    let outcome = arena.intern_var_set([VariableId::from_raw(OUTCOME)]);
    let given =
        arena.intern_var_set([VariableId::from_raw(TREATMENT), VariableId::from_raw(CONFOUNDER)]);
    let marginal = arena.intern(ExprNode::Distribution {
        variables: confounder,
        conditioned_on: empty,
        intervention: none,
        domain: DomainRef::Observational,
        population,
        regime: Some(regime),
    });
    let conditional = arena.intern(ExprNode::Distribution {
        variables: outcome,
        conditioned_on: given,
        intervention: none,
        domain: DomainRef::Observational,
        population,
        regime: Some(regime),
    });
    let factors = arena.intern_list([marginal, conditional]);
    let product = arena.intern(ExprNode::Product(factors));
    let root = arena.intern(ExprNode::SumOut { variables: confounder, expr: product });
    (arena, root)
}

fn outcome_mean(law: &ExactDiscreteLaw, regime: RegimeId, level: i64) -> f64 {
    let (arena, root) = adjustment(law.population(), regime);
    let data = ExactTransportData::try_new(vec![law.clone()], 64).expect("one law");
    let ctx = ExecutionContext::for_tests(1);
    let plan = ExactEvaluationPlan::compile(
        ExactEvaluationSpec {
            arena: &arena,
            root,
            outcomes: Arc::from([VariableId::from_raw(OUTCOME)]),
            request: Assignment::from_pairs([(
                VariableId::from_raw(TREATMENT),
                Value::Int64(level),
            )]),
            limits: ExactEvaluationLimits::default(),
            tolerance: LawTolerance::default(),
        },
        data,
        &ctx,
    )
    .expect("adjustment compiles");
    plan.evaluate(&ctx).expect("adjustment evaluates").mean(VariableId::from_raw(OUTCOME)).unwrap()
}

fn reload_law(law: &ExactDiscreteLaw) -> ExactDiscreteLaw {
    let bytes = serde_json::to_vec(&ExactLawWire::from_law(law)).expect("law wire");
    serde_json::from_slice::<ExactLawWire>(&bytes).expect("law wire parses").to_law().unwrap()
}

fn empirical_law(discrete: &FiniteDiscrete, regime: u32) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_empirical(
        discrete.population.as_str(),
        RegimeId::from_raw(regime),
        [],
        binary_axes(),
        probabilities(&discrete.counts),
        discrete.snapshot.as_str(),
        LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(wide_counts(&discrete.counts))
    .unwrap()
}

fn posterior_law(discrete: &FiniteDiscrete, counts: &[u32], regime: u32) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_bayesian_posterior(
        discrete.population.as_str(),
        RegimeId::from_raw(regime),
        [],
        binary_axes(),
        probabilities(counts),
        discrete.snapshot.as_str(),
        LawTolerance::default(),
    )
    .unwrap()
}

fn assert_adjustment(law: &ExactDiscreteLaw, regime: RegimeId, discrete: &FiniteDiscrete) {
    let active = outcome_mean(law, regime, 1);
    let control = outcome_mean(law, regime, 0);
    assert_close(active, discrete.active, TOLERANCE);
    assert_close(control, discrete.control, TOLERANCE);
    assert_close(active - control, discrete.contrast, TOLERANCE);
    assert_eq!(law.empirical_counts().unwrap(), wide_counts(&discrete.counts));
}

/// The frequency table and the two posterior draws are loaded from their own law wires and
/// evaluated at the static regime id and at the horizon regime id.
///
/// This is a regime-label case, not a horizon producer: the table is the same and only its regime
/// id differs, because no study-facade producer builds a horizon table of counts. The static cells
/// are produced through the study facade by `composition_cross_surface_slices.rs`, which also
/// covers the temporal facade for Gaussian data.
/// `gaussian` and `linear_gaussian` are not law origins.
#[test]
fn finite_discrete_adjustment_reloads_at_each_regime_label() {
    let discrete = slice_cases::load().finite_discrete;
    assert_ne!(discrete.static_regime, discrete.horizon_regime);
    for regime in [discrete.static_regime, discrete.horizon_regime] {
        let label = RegimeId::from_raw(regime);
        let loaded = reload_law(&empirical_law(&discrete, regime));
        assert_eq!(loaded.regime(), label);
        assert_adjustment(&loaded, label, &discrete);

        let mut mixed = 0.0;
        for draw in &discrete.posterior_draws {
            let law = reload_law(&posterior_law(&discrete, &draw.counts, regime));
            let contrast = outcome_mean(&law, label, 1) - outcome_mean(&law, label, 0);
            assert_close(contrast, draw.contrast, TOLERANCE);
            mixed += draw.weight * contrast;
        }
        assert_close(mixed, discrete.posterior_mean, TOLERANCE);
    }

    let loaded = reload_law(&empirical_law(&discrete, discrete.static_regime));
    for origin in &discrete.refused_origins {
        let mut wire = ExactLawWire::from_law(&loaded);
        wire.origin.clone_from(origin);
        let error = wire.to_law().unwrap_err().to_string();
        assert!(error.contains("unknown law origin"), "{error}");
    }
}

fn line_quantity(name: &str, role: QuantityRole, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn line_identity(alignment: DrawAlignment) -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::EmpiricalOutcome,
        &[
            line_quantity("t", QuantityRole::Treatment, "treatment"),
            line_quantity("y", QuantityRole::Outcome, "outcome"),
        ],
        alignment,
        DistributionProvenance {
            source_id: "line-rows".into(),
            provider_id: "cross-surface-line".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "line-snapshot".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

/// The line rows are one aligned joint draw and keep that alignment through the artifact. The
/// slope below is the covariance ratio of the reloaded draws, a check that the pairing survived;
/// the estimators that fit this line are covered by `composition_cross_surface_slices.rs`.
#[test]
fn gaussian_line_rows_keep_their_alignment_through_the_artifact() {
    let gaussian = slice_cases::load().gaussian;
    let draws: Vec<f64> = gaussian
        .treatment
        .iter()
        .zip(&gaussian.outcome)
        .flat_map(|(arm, response)| [*arm, *response])
        .collect();
    let identity = line_identity(DrawAlignment::Joint);
    let bytes = law_with(identity.clone(), draws.clone()).to_bytes("gaussian-line").unwrap();
    let loaded = DistributionArtifact::from_bytes(&bytes, &identity).unwrap();
    let mean_arm = loaded.mean(0).unwrap();
    assert_close(mean_arm, 0.5, TOLERANCE);
    assert_close(
        loaded.mean(1).unwrap(),
        gaussian.intercept + gaussian.effect * mean_arm,
        TOLERANCE,
    );
    let slope = loaded.covariance(0, 1).unwrap() / loaded.covariance(0, 0).unwrap();
    assert_close(slope, gaussian.effect, TOLERANCE);

    // The same rows declared as independent marginals carry no joint operation.
    let marginals = line_identity(DrawAlignment::IndependentMarginals);
    let paired = law_with(marginals, draws);
    let error = paired.covariance(0, 1).unwrap_err().to_string();
    assert!(error.contains("aligned_joint_draws.marginals_not_joint"), "{error}");

    let wrong = joint_identity(DistributionMeaningWire::Bootstrap, DrawAlignment::Joint, "y");
    assert!(DistributionArtifact::from_bytes(&bytes, &wrong).is_err());
    assert!(DistributionArtifact::from_bytes(&flip_last_byte(&bytes), &identity).is_err());
}

/// Feedback is the named modular policy. A coherent joint stores no cut, and a blank
/// reconciliation is refused before a policy exists. The policy is a record: no public estimator
/// takes it, so there is no cut-versus-feedback contrast to reload from an artifact.
#[test]
fn specified_feedback_names_the_declared_policy() {
    let feedback = slice_cases::load().feedback;
    let specified = CutFeedback::from_name(&feedback.specified).unwrap();
    let cut = CutFeedback::from_name(&feedback.cut).unwrap();
    assert_eq!(specified, CutFeedback::Feedback);
    assert_eq!(cut, CutFeedback::Cut);
    assert!(CutFeedback::from_name("cycle").is_none());
    for policy in [specified, cut] {
        let declared =
            MixedModulePolicy::declared_modular(&feedback.reconciliation, policy).unwrap();
        assert_eq!(declared.cut_feedback(), Some(policy));
        assert_eq!(declared.reconciliation().unwrap().as_str(), feedback.reconciliation);
    }
    assert!(MixedModulePolicy::coherent_joint().cut_feedback().is_none());
    assert_eq!(
        MixedModulePolicy::declared_modular("  ", specified).unwrap_err(),
        MixedModuleRefusal::BlankReconciliation
    );
}
