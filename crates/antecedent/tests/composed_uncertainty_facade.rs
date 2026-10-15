//! Composed-uncertainty readings through the facade: the spread of a composed
//! quantity, the composed interval check, exceedance decisions and the spread
//! of a transported claim.
//!
//! Oracles are enumerated finite laws and closed forms computed here.
//!
//! * Ten rows `(x, y)`: `(0,0)` x1, `(0,2)` x4, `(1,0)` x3, `(1,2)` x2.
//!   `E[X] = 0.5`, `E[Y] = 1.2`, population `Var X = 0.25`, `Var Y = 0.96`,
//!   `E[XY] = 0.4`, `Cov = -0.2`, `Var(X - Y) = 1.61`. The rows of
//!   `g = x y` are two 2s and eight 0s: mean `0.4`, sample variance
//!   `6.4 / 9`, standard error of the mean `4 / 15`, `P(g > 1) = 0.2`.
//! * A mixture of `[0, 1, 2, 3]` (weight 0.4) and `[0, 0, 0, 2]` (weight 0.6)
//!   has `P(X > 1.5) = 0.4 * 0.5 + 0.6 * 0.25 = 0.35` with standard error
//!   `sqrt(0.16 * 0.25 / 4 + 0.36 * 0.1875 / 4)`.
//! * Transported pieces `s = [1, 2, 3, 4]`, `t = [2, 1, 4, 3]`,
//!   `f = [0, 0, 1, 2]` sum row-wise to `[3, 3, 8, 9]`: mean `5.75` and
//!   population variance `(2 * 2.75^2 + 2.25^2 + 3.25^2) / 4 = 7.6875`.

use antecedent::analysis::joint_law_propagation::{
    EnvelopeQueryWire, ExceedanceQuery, LinearDependence, MixtureDraws, MomentInput,
    PropagationDraws, ValueFunctional, claim_from_artifact, consume_joint_law_propagation_artifact,
    propagate_aligned_functional, propagate_aligned_linear_spread, propagate_composed_exceedance,
    propagate_dependence_envelope, propagate_linear_moments_spread, propagate_mixture_exceedance,
    propagate_spread, propagate_transported_spread, propagate_transported_spread_aligned,
};
use antecedent_contract::{InputRole, PropagationOperation};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// A distribution artifact whose columns are `columns` (all equally long).
fn artifact(snapshot: &str, alignment: DrawAlignment, columns: &[&[f64]]) -> DistributionArtifact {
    let names = ["a", "b", "c"];
    let quantities: Vec<ScientificQuantity> =
        (0..columns.len()).map(|i| quantity(names[i])).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        alignment,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: "engine".into(),
            rng_id: format!("splitmix-{snapshot}"),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .expect("identity");
    let rows = columns[0].len();
    let mut draws = Vec::with_capacity(rows * columns.len());
    for row in 0..rows {
        for column in columns {
            draws.push(column[row]);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, columns.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::NativeLicensed,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("artifact")
}

fn enumerated(scale: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (xv, yv, count) in [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)] {
        for _ in 0..count * scale {
            x.push(xv);
            y.push(yv);
        }
    }
    (x, y)
}

fn pair(snapshot: &str, scale: usize) -> (PropagationDraws, PropagationDraws) {
    let (x, y) = enumerated(scale);
    let joint = artifact(snapshot, DrawAlignment::Joint, &[&x, &y]);
    (
        PropagationDraws::from_artifact_column("x", &joint, 0).expect("x"),
        PropagationDraws::from_artifact_column("y", &joint, 1).expect("y"),
    )
}

fn pieces() -> [Vec<f64>; 3] {
    [vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 1.0, 4.0, 3.0], vec![0.0, 0.0, 1.0, 2.0]]
}

fn piece_draws(snapshot: &str) -> [PropagationDraws; 3] {
    let [s, t, f] = pieces();
    let joint = artifact(snapshot, DrawAlignment::Joint, &[&s, &t, &f]);
    [
        PropagationDraws::from_artifact_column("source", &joint, 0).expect("s"),
        PropagationDraws::from_artifact_column("target", &joint, 1).expect("t"),
        PropagationDraws::from_artifact_column("functional", &joint, 2).expect("f"),
    ]
}

fn moment(id: &str, evidence: &str, role: InputRole) -> MomentInput {
    MomentInput { id: id.into(), evidence_id: evidence.into(), role, mean: 0.0 }
}

const Z_95: f64 = 1.959_963_984_540_054;

// ------------------------------------------------------------------ spread

#[test]
fn an_aligned_functional_spread_matches_the_enumerated_rows() {
    let (x, y) = pair("snap-enum", 1);
    let claim =
        propagate_spread(PropagationOperation::Product, &x, &y, ValueFunctional::Product, &ctx())
            .expect("spread");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert_eq!(claim.route(), "aligned_joint_law");
    assert!(!claim.is_envelope());
    assert_eq!(claim.n_rows(), Some(10));
    assert!((claim.value("mean").expect("mean") - 0.4).abs() < 1e-12);
    assert!((claim.value("variance").expect("variance") - 6.4 / 9.0).abs() < 1e-12);
    assert!(
        (claim.value("mean_standard_error").expect("se") - 4.0 / 15.0).abs() < 1e-12,
        "se is sqrt(var / n)"
    );
    // The point reading of the same numbers keeps its own label.
    let point = propagate_aligned_functional(
        PropagationOperation::Product,
        &x,
        &y,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect("point");
    assert_eq!(point.label(), "point_only");
    assert_ne!(point.identity(), claim.identity());
}

#[test]
fn an_aligned_linear_spread_is_the_population_variance_of_the_contrast() {
    let (x, y) = pair("snap-enum", 1);
    let claim = propagate_aligned_linear_spread(
        PropagationOperation::Contrast,
        (1.0, -1.0),
        &x,
        &y,
        &ctx(),
    )
    .expect("linear spread");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert!((claim.value("variance").expect("variance") - 1.61).abs() < 1e-12);
    assert!((claim.value("mean").expect("mean") + 0.7).abs() < 1e-12);
    assert!((claim.value("covariance").expect("cov") + 0.2).abs() < 1e-12);
}

#[test]
fn a_linear_moments_spread_agrees_with_the_aligned_rows() {
    let inputs = [
        MomentInput {
            id: "x".into(),
            evidence_id: "snap-enum".into(),
            role: InputRole::Estimate,
            mean: 0.5,
        },
        MomentInput {
            id: "y".into(),
            evidence_id: "snap-enum".into(),
            role: InputRole::Estimate,
            mean: 1.2,
        },
    ];
    let claim = propagate_linear_moments_spread(
        PropagationOperation::Contrast,
        &inputs,
        &[1.0, -1.0],
        &[0.25, -0.2, -0.2, 0.96],
        &LinearDependence::Covariance { id: "enumeration".into() },
        &ctx(),
    )
    .expect("spread");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert_eq!(claim.route(), "linear_moments");
    assert!((claim.value("variance").expect("variance") - 1.61).abs() < 1e-12);
}

#[test]
fn a_spread_of_unpaired_draws_or_on_a_mismatched_route_is_refused() {
    let (x, y) = enumerated(1);
    let marginals = artifact("snap-marginals", DrawAlignment::IndependentMarginals, &[&x, &y]);
    let mx = PropagationDraws::from_artifact_column("mx", &marginals, 0).expect("mx");
    let my = PropagationDraws::from_artifact_column("my", &marginals, 1).expect("my");
    let error = propagate_spread(
        PropagationOperation::NonlinearFunctional,
        &mx,
        &my,
        ValueFunctional::Minimum,
        &ctx(),
    )
    .expect_err("index pairing");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.index_pairing_refused"]);

    // A contrast is a linear operation: its spread is read through the linear
    // route, not through a nonlinear functional's numbers.
    let (px, py) = pair("snap-enum", 1);
    let error = propagate_spread(
        PropagationOperation::Contrast,
        &px,
        &py,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect_err("numbers do not fit the route");
    assert_eq!(error.code(), "invalid_argument");
    assert_eq!(error.details(), vec!["joint_law_propagation.invalid_request"]);

    // Draws of two different joint laws are not paired.
    let (_, other_y) = pair("snap-other", 1);
    let error = propagate_aligned_linear_spread(
        PropagationOperation::Sum,
        (1.0, 1.0),
        &px,
        &other_y,
        &ctx(),
    )
    .expect_err("alignment");
    assert_eq!(error.details(), vec!["joint_law_propagation.alignment_mismatch"]);
}

// ------------------------------------------------------------------ interval

#[test]
fn a_composed_interval_opens_only_with_a_measured_record_at_its_coordinate() {
    let (x, y) = pair("snap-enum", 1);
    let claim = propagate_aligned_linear_spread(
        PropagationOperation::Contrast,
        (1.0, -1.0),
        &x,
        &y,
        &ctx(),
    )
    .expect("spread");
    let coordinate = claim.coordinate_id().expect("coordinate");
    assert_eq!(coordinate, "contrast|aligned_joint_law|estimate,estimate");
    // The caller supplies no record: the registry decides. The key is the claim's own
    // coordinate, construction and declared evidence size, so a caller cannot name a record.
    let basis = claim.composed_interval_basis("normal_approximation", 200).expect("key");
    assert_eq!(&*basis.functional, coordinate.as_str());
    assert_eq!(&*basis.interval_method, "normal_approximation");
    assert_eq!(&*basis.estimator, "joint_law.aligned_joint_law");
    assert_eq!(basis.row_count, 200);

    // Without a currently attesting record the interval reading is closed, whatever the
    // construction, and an evidence size of zero is invalid.
    for construction in ["normal_approximation", "aligned_draw_equal_tail_exchangeable_rank"] {
        match claim.check_composed_interval(construction, 200) {
            Ok(licence) => assert_eq!(licence.label(), "measured_interval"),
            Err(error) => {
                assert_eq!(error.code(), "cell_not_licensed");
                assert!(
                    error.details() == vec!["joint_law_propagation.interval_route_closed"]
                        || error.details()
                            == vec!["joint_law_propagation.coverage_record_unmeasured"],
                    "{:?}",
                    error.details()
                );
            }
        }
    }
    let error = claim.check_composed_interval("normal_approximation", 0).expect_err("rows");
    assert_eq!(error.code(), "invalid_argument");

    // The component-interval readings stay closed.
    assert_eq!(
        claim.interval().expect_err("closed").details(),
        vec!["joint_law_propagation.interval_route_closed"]
    );
}

#[test]
fn an_envelope_has_no_composed_interval_and_an_edited_artifact_is_refused() {
    let atoms: Vec<f64> = (0..200).map(|j| (f64::from(j) + 0.5) / 200.0).collect();
    let envelope = propagate_dependence_envelope(
        PropagationOperation::Sum,
        EnvelopeQueryWire::SumCdf { threshold: 1.0_f64.to_bits() },
        &PropagationDraws::independent_marginal("x", "s1", atoms.clone()),
        &PropagationDraws::independent_marginal("y", "s2", atoms),
        &ctx(),
    )
    .expect("envelope");
    let error =
        envelope.check_composed_interval("normal_approximation", 200).expect_err("envelope");
    assert_eq!(error.code(), "cell_not_licensed");
    assert_eq!(error.details(), vec!["joint_law_propagation.spread_needs_licensed_dependence"]);
    assert!(envelope.coordinate_id().is_err());

    // A truncated artifact does not rebuild a claim.
    let (x, y) = pair("snap-enum", 1);
    let claim = propagate_aligned_linear_spread(
        PropagationOperation::Contrast,
        (1.0, -1.0),
        &x,
        &y,
        &ctx(),
    )
    .expect("spread");
    let bytes = claim.export().expect("export");
    let rebuilt = claim_from_artifact(&bytes, Some(claim.identity()), &ctx()).expect("rebuilt");
    assert_eq!(rebuilt.identity(), claim.identity());
    assert_eq!(rebuilt.coordinate_id().expect("id"), claim.coordinate_id().expect("id"));
    assert!(claim_from_artifact(&bytes[..bytes.len() - 6], None, &ctx()).is_err());
    assert!(claim_from_artifact(&bytes, Some("not-the-identity"), &ctx()).is_err());
}

// ---------------------------------------------------------------- exceedance

#[test]
fn a_composed_exceedance_reports_the_probability_band_and_verdict() {
    // The enumerated law replicated 100 times: P(x y > 1) = 200 / 1000 = 0.2.
    let (x, y) = pair("snap-enum-large", 100);
    let decide = |cutoff: f64| {
        propagate_composed_exceedance(
            PropagationOperation::Product,
            &x,
            &y,
            ValueFunctional::Product,
            ExceedanceQuery { threshold: 1.0, cutoff, band_level: 0.95 },
            &ctx(),
        )
        .expect("decision")
    };
    let claim = decide(0.15);
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert!((claim.value("probability").expect("probability") - 0.2).abs() < 1e-12);
    assert_eq!(claim.n_rows(), Some(1000));
    // Agresti-Coull standard error at n = 1000, 200 hits.
    let adjusted = (200.0 + Z_95 * Z_95 / 2.0) / (1000.0 + Z_95 * Z_95);
    let standard_error = (adjusted * (1.0 - adjusted) / (1000.0 + Z_95 * Z_95)).sqrt();
    assert!((claim.value("standard_error").expect("se") - standard_error).abs() < 1e-6);
    assert!((claim.value("band").expect("band") - Z_95 * standard_error).abs() < 1e-6);
    // The cutoff below the band acts, inside the band is unresolved, above withholds.
    assert_eq!(claim.verdict(), Some("act"));
    assert_eq!(decide(0.2).verdict(), Some("unresolved"));
    assert_eq!(decide(0.25).verdict(), Some("withhold"));
    // The verdict code is not a float value.
    assert_eq!(claim.value("verdict_code"), None);
    assert!(claim.values().iter().all(|(name, _)| name != "verdict_code"));

    // Ten rows: the band is wide and the same cutoff is unresolved.
    let (sx, sy) = pair("snap-enum", 1);
    let small = propagate_composed_exceedance(
        PropagationOperation::Product,
        &sx,
        &sy,
        ValueFunctional::Product,
        ExceedanceQuery { threshold: 1.0, cutoff: 0.15, band_level: 0.95 },
        &ctx(),
    )
    .expect("decision");
    assert_eq!(small.verdict(), Some("unresolved"));

    // The sealed artifact replays the decision bit for bit.
    let bytes = claim.export().expect("export");
    let consumed = consume_joint_law_propagation_artifact(&bytes, Some(claim.identity()), &ctx())
        .expect("consume");
    assert_eq!(consumed.label, "uncalibrated_spread");
    assert_eq!(consumed.value("probability"), claim.value("probability"));
    assert!(consumed.result.iter().any(|(name, code)| name == "verdict_code" && *code == 0));
}

#[test]
fn an_exceedance_with_a_cutoff_outside_the_unit_interval_is_refused() {
    let (x, y) = pair("snap-enum", 1);
    let error = propagate_composed_exceedance(
        PropagationOperation::Product,
        &x,
        &y,
        ValueFunctional::Product,
        ExceedanceQuery { threshold: 1.0, cutoff: 1.5, band_level: 0.95 },
        &ctx(),
    )
    .expect_err("cutoff");
    assert_eq!(error.code(), "invalid_argument");
    assert_eq!(error.details(), vec!["joint_law_propagation.invalid_request"]);
}

#[test]
fn a_known_weight_mixture_exceedance_matches_the_weighted_fractions() {
    let components = [
        MixtureDraws {
            id: "a".into(),
            evidence_id: "s1".into(),
            weight: 0.4,
            draws: vec![0.0, 1.0, 2.0, 3.0],
        },
        MixtureDraws {
            id: "b".into(),
            evidence_id: "s2".into(),
            weight: 0.6,
            draws: vec![0.0, 0.0, 0.0, 2.0],
        },
    ];
    let claim = propagate_mixture_exceedance(&components, 1.5, &ctx()).expect("mixture");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert_eq!(claim.route(), "known_weight_mixture");
    assert!((claim.value("probability").expect("p") - 0.35).abs() < 1e-15);
    let expected = (0.16_f64 * 0.25 / 4.0 + 0.36 * 0.1875 / 4.0).sqrt();
    assert!((claim.value("standard_error").expect("se") - expected).abs() < 1e-15);

    // Draws that change at an unchanged shape change the identity.
    let mut changed = components.clone();
    changed[1].draws[3] = 1.0;
    let other = propagate_mixture_exceedance(&changed, 1.5, &ctx()).expect("mixture");
    assert_ne!(other.identity(), claim.identity());
    assert!((other.value("probability").expect("p") - 0.2).abs() < 1e-15);

    // Weights that are not a probability vector are refused.
    let mut skewed = components.clone();
    skewed[0].weight = 0.9;
    let error = propagate_mixture_exceedance(&skewed, 1.5, &ctx()).expect_err("weights");
    assert_eq!(error.code(), "invalid_argument");
}

// -------------------------------------------------------------- transported

#[test]
fn a_transported_spread_needs_all_three_pieces_and_their_dependence() {
    // Var = 1 + 4 + 9 + 2 * 0.5 = 15 with the source-target covariance 0.5.
    let covariance = [1.0, 0.5, 0.0, 0.5, 4.0, 0.0, 0.0, 0.0, 9.0];
    let all = [
        moment("source", "snap-source", InputRole::SourceSampling),
        moment("target", "snap-target", InputRole::TargetSampling),
        moment("functional", "snap-source", InputRole::TransportFunctional),
    ];
    let claim = propagate_transported_spread(&all, &[1.0, 1.0, 1.0], &covariance, "pieces", &ctx())
        .expect("transported spread");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert_eq!(claim.operation(), "transported_composition");
    assert!((claim.value("variance").expect("variance") - 15.0).abs() < 1e-12);

    // Two of the three pieces: the spread is incomplete.
    let two_covariance = [1.0, 0.5, 0.5, 4.0];
    let error =
        propagate_transported_spread(&all[..2], &[1.0, 1.0], &two_covariance, "pieces", &ctx())
            .expect_err("incomplete");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.transport_pieces_incomplete"]);
    assert_eq!(error.refusals[0].offending.as_deref(), Some("transport_functional"));

    // Declared independence of the pieces is not accepted for a transported claim.
    let diagonal = [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
    let error = propagate_linear_moments_spread(
        PropagationOperation::TransportedComposition,
        &all,
        &[1.0, 1.0, 1.0],
        &diagonal,
        &LinearDependence::DeclaredIndependent,
        &ctx(),
    )
    .expect_err("independence");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.transported_dependence_required"]);
}

#[test]
fn a_transported_spread_from_aligned_pieces_matches_the_row_sums() {
    let [s, t, f] = piece_draws("snap-pieces");
    let claim =
        propagate_transported_spread_aligned(&s, &t, &f, [1.0, 1.0, 1.0], &ctx()).expect("spread");
    assert_eq!(claim.label(), "uncalibrated_spread");
    assert!((claim.value("mean").expect("mean") - 5.75).abs() < 1e-12);
    assert!((claim.value("variance").expect("variance") - 7.6875).abs() < 1e-12);

    // A piece from another joint law is not paired with the others.
    let [_, other_t, _] = piece_draws("snap-other-pieces");
    let error = propagate_transported_spread_aligned(&s, &other_t, &f, [1.0, 1.0, 1.0], &ctx())
        .expect_err("alignment");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.alignment_mismatch"]);

    // Independently sampled pieces are never paired by index.
    let [a, b, c] = pieces();
    let marginals = artifact("snap-marginals", DrawAlignment::IndependentMarginals, &[&a, &b, &c]);
    let ms = PropagationDraws::from_artifact_column("source", &marginals, 0).expect("s");
    let mt = PropagationDraws::from_artifact_column("target", &marginals, 1).expect("t");
    let mf = PropagationDraws::from_artifact_column("functional", &marginals, 2).expect("f");
    let error = propagate_transported_spread_aligned(&ms, &mt, &mf, [1.0, 1.0, 1.0], &ctx())
        .expect_err("index pairing");
    assert_eq!(error.details(), vec!["joint_law_propagation.index_pairing_refused"]);
}
