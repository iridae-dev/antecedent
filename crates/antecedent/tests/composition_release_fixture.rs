//! Release-boundary composition fixture, through public Rust APIs only.
//!
//! One scenario whose numbers are checked by hand (see
//! `conformance/composition/release_boundary/README.md`):
//!
//! * a binary exposure with a declared confusion model (Se 0.9, Sp 0.8) is
//!   recorded as exposed for 410 of 1000, so the corrected prevalence is
//!   `(0.41 + 0.8 - 1) / 0.7 = 0.3`;
//! * two disjoint studies report a risk difference of 0.10 and 0.20 (variance
//!   0.0004 each); the pool is 0.15 with variance 0.0002;
//! * with baseline risk 0.10 the composed finite model has
//!   `P(Y=1 | do(X=1)) = 0.25`, `P(Y=1) = 0.145`, so the excess risk is
//!   `0.145 - 0.10 = 0.3 * 0.15 = 0.045`;
//! * the affine utility `1000 * excess - 30` is 15 for `act` and 0 for `hold`;
//! * a larger second study (0.07, variance 0.0002) makes the pool
//!   `600 / 7500 = 0.08`, the excess risk 0.024 and `act` worth -6.
//!
//! Every stage stays point-only. No interval, posterior decision or ranking
//! guarantee is produced here; each request for one is refused with
//! `cell_not_licensed`. Nothing below reaches into an internal constructor,
//! a test-only hook or a `calibration-internal` item.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "each test is one scenario read top to bottom")]

#[path = "composition_release_fixture/scenario.rs"]
mod scenario;

use antecedent::StudyBuilder;
use antecedent::analysis::composition_verification::{
    CompositionGraph, CompositionNode, CompositionVerification, DependencyKind, NodeKind,
    Revalidation,
};
use antecedent::analysis::evidence_ledger::{
    Lineage, Overlap, Revalidation as LedgerStanding, SourceDeclaration, SourceKind,
};
use antecedent::analysis::joint_law_propagation::{
    LinearDependence, MomentInput, PropagationDraws, ValueFunctional, propagate,
    propagate_aligned_functional, propagate_linear_moments,
};
use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, QuantityRole, RequestedReading, ScientificQuantity,
};
use antecedent_core::CancellationToken;
use antecedent_design::decision_eval::Verdict;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::joint_law_propagation_artifact::{PropagationRequestWire, float_bits};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use scenario::{
    Bundle, DecisionDeclaration, InterfaceOptions, Refused, World, check_interfaces, consume, ctx,
    decide, excess_risk, fixture, gate, ident, interface_pair_details, measure, num, outcome_risk,
    pool, rows_of, selected, study_source, text, value_of, verify,
};
use serde_json::Value;

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: expected {expected}, got {actual}");
}

/// Assert a refusal equals the `code` and `detail` the fixture file records.
fn assert_refused(refused: &Refused, entry: &Value) {
    assert_eq!(refused.code, text(&entry["code"]), "{refused:?}");
    if let Some(detail) = entry.get("detail") {
        assert_eq!(refused.detail, text(detail), "{refused:?}");
    }
}

fn exact(fx: &Value) -> f64 {
    num(&fx["tolerance"]["exact"])
}

fn loose(fx: &Value) -> f64 {
    num(&fx["tolerance"]["decision"])
}

// ---------------------------------------------------------------------------
// 1 and 2: a measured quantity enters a causal model; components contribute
// ---------------------------------------------------------------------------

#[test]
fn a_measured_exposure_enters_the_causal_model_through_the_declared_confusion_model() {
    let fx = fixture();
    let (sc, v) = (&fx["scenario"], &fx["values"]);
    let world = World::build(sc);

    // The corrected prevalence, the naive proportion and the delta-method
    // variance: p* = Se pi + (1 - Sp)(1 - pi), J = Se + Sp - 1 = 0.7.
    let m = &sc["measurement"];
    let (se, sp) = (num(&m["sensitivity"]), num(&m["specificity"]));
    let (vse, vsp) = (num(&m["sensitivity_variance"]), num(&m["specificity_variance"]));
    let (events, total) = (m["events"].as_f64().unwrap(), m["total"].as_f64().unwrap());
    let (p_star, j) = (events / total, se + sp - 1.0);
    let pi = (p_star + sp - 1.0) / j;
    close(world.measured.point, pi, exact(&fx), "closed-form prevalence");
    close(world.measured.point, num(&v["prevalence"]), exact(&fx), "hand-derived prevalence");
    close(world.measured.naive, num(&v["naive_prevalence"]), 1e-15, "naive proportion");
    let variance =
        (p_star * (1.0 - p_star) / total + (1.0 - pi).powi(2) * vsp + pi.powi(2) * vse) / (j * j);
    close(world.measured.standard_error.powi(2), variance, exact(&fx), "uncalibrated variance");

    // The pooled study evidence: inverse variance on disjoint rows.
    close(world.pooled.estimate, num(&v["pooled_effect"]), exact(&fx), "pooled estimate");
    close(world.pooled.variance, num(&v["pooled_variance"]), exact(&fx), "pooled variance");
    assert_eq!(world.pooled.declared_slots, v["declared_slots"].as_u64().unwrap());
    assert_eq!(world.pooled.distinct_observations, v["distinct_observations"].as_u64().unwrap());

    // The composed finite model reads the measured prevalence and the pooled effect.
    let composition = &world.composition;
    close(
        outcome_risk(composition, &[("X", "1")]),
        num(&v["outcome_risk_exposed"]),
        exact(&fx),
        "do(X=1)",
    );
    close(
        outcome_risk(composition, &[("X", "0")]),
        num(&v["outcome_risk_unexposed"]),
        exact(&fx),
        "do(X=0)",
    );
    close(outcome_risk(composition, &[]), num(&v["outcome_risk"]), exact(&fx), "P(Y=1)");
    close(excess_risk(composition), num(&v["excess_risk"]), exact(&fx), "excess risk");
    // The same number from the closed form: pi * d.
    close(excess_risk(composition), pi * world.pooled.estimate, exact(&fx), "pi * d");

    // The measurement carries an uncalibrated standard error, never an interval.
    assert!(world.measured.standard_error.is_finite());
}

// ---------------------------------------------------------------------------
// 3: shared evidence and dependence are preserved
// ---------------------------------------------------------------------------

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "risk".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn draws_artifact(
    snapshot: &str,
    alignment: DrawAlignment,
    x: &[f64],
    y: &[f64],
) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[quantity("prevalence"), quantity("effect")],
        alignment,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: "release_boundary".into(),
            rng_id: format!("enumerated-{snapshot}"),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::with_capacity(2 * x.len());
    for (a, b) in x.iter().zip(y) {
        draws.extend([*a, *b]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [x.len(), 2],
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

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario asserting each declared dependence case side by side against its oracle"
)]
fn shared_evidence_is_counted_once_and_dependence_is_declared_or_refused() {
    let fx = fixture();
    let (sc, v, r) = (&fx["scenario"], &fx["values"], &fx["variants"]);
    let studies = sc["studies"].as_array().unwrap();
    let overlapping = &sc["overlapping_study"];

    let mut session = StudyBuilder::evidence_ledger();
    let a = session
        .add_source(study_source("study_a", "cohort", rows_of(&studies[0]["rows"])))
        .unwrap();
    let c = session
        .add_source(study_source("study_c", "cohort", rows_of(&overlapping["rows"])))
        .unwrap();
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let (est_a, var_a) = (num(&studies[0]["estimate"]), num(&studies[0]["variance"]));
    let (est_c, var_c) = (num(&overlapping["estimate"]), num(&overlapping["variance"]));
    let cov = num(&overlapping["covariance"]);

    // The ledger reads the overlap from the rows: 200 observations are shared.
    let account = session.account_observations(&[a, c]).unwrap();
    assert_eq!(account.pairs[0].overlap, Overlap::SharedObservations { shared: 200 });
    assert_eq!(account.mass.declared_slots, 1000);
    assert_eq!(account.mass.distinct_observations, 800);
    assert_eq!(account.mass.repeated_slots, 200);

    // Declared independent, the shared rows would count twice: refused before any number.
    let head = session.head_hex();
    let refusal =
        pool(&mut session, "independent", &[(a, est_a, var_a), (c, est_c, var_c)], None, None)
            .unwrap_err();
    assert_refused(&refusal, &fx["mutations"]["evidence_overlap"]);
    // A diagonal covariance over shared rows is the same double count.
    let diagonal = vec![var_a, 0.0, 0.0, var_c];
    let refusal = pool(
        &mut session,
        "diagonal",
        &[(a, est_a, var_a), (c, est_c, var_c)],
        Some(diagonal),
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(refusal.detail, "evidence_ledger.diagonal_covariance_shared_rows");
    assert_eq!(session.head_hex(), head, "refusals leave the ledger unchanged");

    // Undeclared observations are an unknown overlap, never independent.
    let refusal =
        pool(&mut session, "vendor", &[(a, est_a, var_a), (vendor, est_c, var_c)], None, None)
            .unwrap_err();
    assert_refused(&refusal, &r["dependence_gap_ledger"]);

    // A declared covariance with cross terms licenses the shared pair.
    // GLS: y = (0.10, 0.20), S = [[0.0004, 0.0002], [0.0002, 0.0004]]: weights (1/2, 1/2),
    // estimate 0.15, variance (0.0004 + 0.0002) / 2 = 0.0003.
    let pooled = pool(
        &mut session,
        "shared",
        &[(a, est_a, var_a), (c, est_c, var_c)],
        Some(vec![var_a, cov, cov, var_c]),
        None,
    )
    .unwrap();
    close(pooled.estimate, num(&v["overlap_pooled_estimate"]), exact(&fx), "shared estimate");
    close(pooled.variance, num(&v["overlap_pooled_variance"]), exact(&fx), "shared variance");
    assert_eq!(pooled.declared_slots, v["overlap_declared_slots"].as_u64().unwrap());
    assert_eq!(pooled.distinct_observations, v["overlap_distinct_observations"].as_u64().unwrap());

    // The joint-law rule reads the same overlap: the shared rows carry one
    // evidence identity, so independence is refused and the covariance accepted.
    let shared_evidence = a.to_hex();
    let inputs = [
        MomentInput {
            id: "study_a".into(),
            evidence_id: shared_evidence.clone(),
            role: InputRole::Estimate,
            mean: est_a,
        },
        MomentInput {
            id: "study_c".into(),
            evidence_id: shared_evidence,
            role: InputRole::Estimate,
            mean: est_c,
        },
    ];
    let error = propagate_linear_moments(
        PropagationOperation::WeightedSum,
        &inputs,
        &[0.5, 0.5],
        &[var_a, 0.0, 0.0, var_c],
        &LinearDependence::DeclaredIndependent,
        &ctx(),
    )
    .expect_err("shared evidence declared independent");
    assert_eq!(error.code(), text(&r["shared_evidence_declared_independent"]["code"]));
    assert_eq!(error.details(), vec![text(&r["shared_evidence_declared_independent"]["detail"])]);
    let claim = propagate_linear_moments(
        PropagationOperation::WeightedSum,
        &inputs,
        &[0.5, 0.5],
        &[var_a, cov, cov, var_c],
        &LinearDependence::Covariance { id: "shared-rows-300-500".into() },
        &ctx(),
    )
    .expect("a declared covariance is accepted");
    close(claim.value("mean").unwrap(), 0.15, exact(&fx), "weighted mean");
    // The rule's variance is the ledger's GLS variance.
    close(claim.value("variance").unwrap(), pooled.variance, exact(&fx), "variance agrees");
    assert_eq!(claim.label(), "point_only");
    assert!(claim.interval().is_err());
    assert_eq!(claim.interval().unwrap_err().code(), text(&r["interval"]["code"]));

    // Unknown dependence is a gap, not an independence.
    let moments = |id: &str, evidence: &str| PropagationInput {
        id: id.into(),
        evidence_id: evidence.into(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    };
    let unknown = PropagationRequest {
        operation: PropagationOperation::WeightedSum,
        inputs: vec![moments("study_a", "evidence-a"), moments("study_b", "evidence-b")],
        dependence: PropagationDependence::Unknown,
        reading: RequestedReading::Point,
    };
    let numbers = PropagationRequestWire::LinearCombination {
        coefficients: float_bits(&[0.5, 0.5]),
        means: float_bits(&[0.10, 0.20]),
        covariance: float_bits(&[var_a, 0.0, 0.0, var_a]),
    };
    let error = propagate(&unknown, numbers, &ctx()).expect_err("unknown dependence");
    assert_eq!(error.code(), text(&r["dependence_gap_rule"]["code"]));
    assert_eq!(error.details(), vec![text(&r["dependence_gap_rule"]["detail"])]);

    // The claim graph reports the same overlap from the ledger's rows.
    let mut g = CompositionGraph::new();
    for (key, source) in [("study_a", a), ("study_c", c)] {
        g.add_node(
            CompositionNode::new(key, NodeKind::CohortSource, ident(&[key]))
                .with_evidence(vec![source]),
        )
        .unwrap();
    }
    g.add_node(CompositionNode::new("overlap_check", NodeKind::Claim, ident(&["overlap_check"])))
        .unwrap();
    g.add_edge("overlap_check", "study_a", DependencyKind::Evidence).unwrap();
    g.add_edge("overlap_check", "study_c", DependencyKind::Evidence).unwrap();
    let verification = CompositionVerification::verify(
        None,
        &g,
        &std::collections::BTreeMap::new(),
        Some((session.ledger(), &CancellationToken::new())),
    )
    .unwrap();
    let view = verification.view("overlap_check").unwrap();
    assert_eq!(view.shared_data.len(), 1);
    assert_eq!(view.shared_data[0].relation, "shared_data");
    assert_eq!(view.shared_data[0].shared_observations, 200);
}

#[test]
fn unpaired_marginals_are_never_paired_by_index_and_an_enumerated_joint_law_is() {
    let fx = fixture();
    let sc = &fx["scenario"];
    let prevalence: Vec<f64> =
        sc["marginal_draws"]["prevalence"].as_array().unwrap().iter().map(num).collect();
    let effect: Vec<f64> =
        sc["marginal_draws"]["effect"].as_array().unwrap().iter().map(num).collect();

    // Marginals with distinct evidence and no joint law: index pairing is refused.
    let x: Vec<f64> = prevalence.iter().cycle().take(4).copied().collect();
    let y: Vec<f64> = effect.iter().cycle().take(4).copied().collect();
    let marginals = draws_artifact("marginals", DrawAlignment::IndependentMarginals, &x, &y);
    let mx = PropagationDraws::from_artifact_column("prevalence", &marginals, 0).unwrap();
    let my = PropagationDraws::from_artifact_column("effect", &marginals, 1).unwrap();
    let error = propagate_aligned_functional(
        PropagationOperation::NonlinearFunctional,
        &mx,
        &my,
        ValueFunctional::Minimum,
        &ctx(),
    )
    .expect_err("unpaired marginals");
    let entry = &fx["variants"]["unpaired_marginals"];
    assert_eq!(error.code(), text(&entry["code"]));
    assert_eq!(error.details(), vec![text(&entry["detail"])]);

    // The declared independent coupling, enumerated as the full product lattice,
    // is one aligned joint law: E[pi d] = E[pi] E[d] = 0.3 * 0.15 = 0.045.
    let (mut px, mut dy) = (Vec::new(), Vec::new());
    for p in &prevalence {
        for d in &effect {
            px.push(*p);
            dy.push(*d);
        }
    }
    let joint = draws_artifact("lattice", DrawAlignment::Joint, &px, &dy);
    let jx = PropagationDraws::from_artifact_column("prevalence", &joint, 0).unwrap();
    let jy = PropagationDraws::from_artifact_column("effect", &joint, 1).unwrap();
    let claim = propagate_aligned_functional(
        PropagationOperation::Product,
        &jx,
        &jy,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect("an aligned product");
    close(
        claim.value("mean").unwrap(),
        num(&fx["values"]["aligned_product_mean"]),
        exact(&fx),
        "E[pi d]",
    );
    assert_eq!(claim.route(), "aligned_joint_law");
    assert_eq!(claim.label(), "point_only");
    // The aligned moment is the point claim of the composed model.
    let world = World::build(sc);
    close(claim.value("mean").unwrap(), world.excess_risk, loose(&fx), "joint law = composition");
    assert_eq!(claim.posterior().unwrap_err().code(), "cell_not_licensed");
}

// ---------------------------------------------------------------------------
// 4: a downstream decision depends on the resulting causal claim
// ---------------------------------------------------------------------------

#[test]
fn the_decision_reads_the_licensed_mean_and_matches_the_hand_computed_utilities() {
    let fx = fixture();
    let (sc, v) = (&fx["scenario"], &fx["values"]);
    let world = World::build(sc);
    let d = &world.result;

    // 1000 * 0.045 - 30 = 15 for act; the hold action is worth 0.
    close(value_of(d, "act"), num(&v["act"]), loose(&fx), "act");
    close(value_of(d, "hold"), num(&v["hold"]), loose(&fx), "hold");
    assert_eq!(d.verdict, Verdict::UniquelyOptimal(text(&v["selected"])));
    assert_eq!(selected(d), "act");
    // A mean source carries no sampling error and no regret or EVPI.
    assert!(d.evpi.is_none());
    assert!(d.actions.iter().all(|a| a.standard_error.is_none() && a.expected_regret.is_none()));
    assert!(d.assumptions.iter().any(|a| a.contains("means only")));
    // The result names the evidence state it was computed at.
    assert_eq!(d.source.snapshot_id, world.session.state_digest_hex());
    assert_eq!(d.source.causal_contract_id, world.plan_identity);

    // The decision follows the declaration, not the library: a programme
    // costing 50 is worth 1000 * 0.045 - 50 = -5, so hold.
    let dearer = DecisionDeclaration { cost: 50.0, ..world.declaration.clone() };
    let result = decide(&dearer, world.excess_risk, "snapshot", "plan").unwrap();
    close(value_of(&result, "act"), -5.0, loose(&fx), "dearer act");
    assert_eq!(selected(&result), "hold");
}

// ---------------------------------------------------------------------------
// 5: new observations revise the evidence state; claims are revalidated
// ---------------------------------------------------------------------------

#[test]
fn new_observations_revalidate_the_claim_graph_and_the_decision_is_reevaluated() {
    let fx = fixture();
    let (sc, rv) = (&fx["scenario"], &fx["revised_values"]);
    let mut world = World::build(sc);
    let before_pool = world.pooled.result;
    let previous_plan = world.plan_identity.clone();

    // Nothing has changed: every claim is still valid and the decision is usable.
    let steady = verify(Some(&world.graph), &world.graph, &world.session, before_pool);
    assert!(steady.revalidation().keys_of(Revalidation::Recompute).is_empty());
    assert!(gate(&world.graph, &world.session, before_pool).is_none());

    // Study B grows. The ledger classifies the pooled result for recomputation
    // and the claim graph carries that to everything that rests on it.
    let step = world.revise(sc);
    assert_eq!(world.session.standing(before_pool), Some(LedgerStanding::Recompute));
    let refusal = gate(&world.graph, &world.session, before_pool).expect("the decision is stale");
    assert_refused(&refusal, &fx["variants"]["revalidation_pending"]);
    let stale = verify(Some(&world.graph), &world.graph, &world.session, before_pool);
    assert_eq!(
        stale.revalidation().keys_of(Revalidation::Recompute),
        vec!["claim", "decision", "mechanisms", "pooled_effect"]
    );
    assert_eq!(
        stale.revalidation().keys_of(Revalidation::StillValid),
        vec!["measurement", "study_a", "study_b", "survey"]
    );
    assert!(stale.revalidation().keys_of(Revalidation::Withdrawn).is_empty());
    // The stale pooled numbers are not served.
    let served = world.session.synthesis(before_pool).unwrap_err();
    assert_eq!(served.code, "evidence_source_withdrawn");
    assert_eq!(served.detail, "evidence_ledger.result_not_valid");

    // Recompute on the grown study, naming the result replaced.
    let (estimate, variance) = step.summary;
    let studies = sc["studies"].as_array().unwrap();
    let pooled = pool(
        &mut world.session,
        "pooled_effect_recomputed",
        &[
            (world.study_a, num(&studies[0]["estimate"]), num(&studies[0]["variance"])),
            (step.grown, estimate, variance),
        ],
        None,
        Some(before_pool),
    )
    .unwrap();
    // (0.10 / 0.0004 + 0.07 / 0.0002) / (1 / 0.0004 + 1 / 0.0002) = 600 / 7500 = 0.08.
    close(pooled.estimate, num(&rv["pooled_effect"]), exact(&fx), "revised pool");
    close(pooled.variance, num(&rv["pooled_variance"]), exact(&fx), "revised variance");
    assert_eq!(world.session.standing(before_pool), Some(LedgerStanding::Withdrawn));
    assert_eq!(world.session.standing(pooled.result), Some(LedgerStanding::StillValid));

    // Everything downstream of the pool is recomputed from the new evidence.
    let previous = world.graph;
    let survey = world.survey;
    let revised = World::downstream_of(
        sc,
        world.session,
        (survey, world.study_a, step.grown),
        pooled.clone(),
        world.measured,
    );
    close(revised.excess_risk, num(&rv["excess_risk"]), exact(&fx), "revised excess risk");
    close(
        outcome_risk(&revised.composition, &[("X", "1")]),
        num(&rv["outcome_risk_exposed"]),
        exact(&fx),
        "do(X=1)",
    );
    close(outcome_risk(&revised.composition, &[]), num(&rv["outcome_risk"]), exact(&fx), "P(Y=1)");
    close(value_of(&revised.result, "act"), num(&rv["act"]), loose(&fx), "act");
    close(value_of(&revised.result, "hold"), num(&rv["hold"]), loose(&fx), "hold");
    // The decision flips: the programme no longer pays for itself.
    assert_eq!(selected(&revised.result), text(&rv["selected"]));
    assert_ne!(revised.plan_identity, previous_plan);

    // The old claims that rested on study B lose authority when its evidence
    // changes; independent claims remain valid. The replacement graph is usable.
    let moved = verify(Some(&previous), &revised.graph, &revised.session, pooled.result);
    assert_eq!(
        moved.revalidation().keys_of(Revalidation::Withdrawn),
        vec!["claim", "decision", "mechanisms", "pooled_effect", "study_b"]
    );
    assert_eq!(
        moved.revalidation().keys_of(Revalidation::StillValid),
        vec!["measurement", "study_a", "survey"]
    );
    assert!(moved.revalidation().keys_of(Revalidation::Recompute).is_empty());
    for key in ["claim", "decision", "mechanisms", "pooled_effect", "study_b"] {
        let claim = moved.revalidation().claim(key).expect("revalidated claim");
        assert!(
            claim.withdrawals.iter().any(|withdrawal| {
                withdrawal.kind.as_str() == "retired"
                    && withdrawal.origin == "study_b"
                    && withdrawal.detail == "evidence_changed"
            }),
            "{key} should reflect the replaced study evidence: {claim:?}"
        );
    }
    // And the recomputed decision is usable against its own graph.
    assert!(gate(&revised.graph, &revised.session, pooled.result).is_none());

    // A fresh consumer of the revised evidence state reaches the same decision
    // from its bytes and the declared studies alone.
    let mut sealed = revised.bundle(sc);
    sealed.studies = scenario::revised_study_declarations(sc);
    let report = consume(&sealed).expect("the revised bundle is consumed");
    close(report.pooled_effect, num(&rv["pooled_effect"]), exact(&fx), "consumed pool");
    close(report.excess_risk, num(&rv["excess_risk"]), loose(&fx), "consumed excess risk");
    close(report.act, num(&rv["act"]), loose(&fx), "consumed act");
    assert_eq!(report.selected, text(&rv["selected"]));
    assert!(report.snapshot_bound && report.claim_graph_valid);
}

// ---------------------------------------------------------------------------
// 6: duplicated evidence cannot be counted twice
// ---------------------------------------------------------------------------

#[test]
fn duplicated_evidence_cannot_be_counted_twice() {
    let fx = fixture();
    let sc = &fx["scenario"];
    let studies = sc["studies"].as_array().unwrap();
    let (est_a, var_a) = (num(&studies[0]["estimate"]), num(&studies[0]["variance"]));
    let mut session = StudyBuilder::evidence_ledger();
    let a = session
        .add_source(study_source("study_a", "cohort", rows_of(&studies[0]["rows"])))
        .unwrap();
    // The same study re-entering through its lineage.
    let copy = session
        .add_source(
            SourceDeclaration::new("study_a_again", SourceKind::ObservationSet)
                .with_lineage(Lineage::Duplicate { of: a }),
        )
        .unwrap();
    let head = session.head_hex();
    let refusal =
        pool(&mut session, "twice", &[(a, est_a, var_a), (copy, est_a, var_a)], None, None)
            .unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(refusal.detail, "evidence_ledger.reentry_duplicate");
    // The same source listed twice.
    let refusal =
        pool(&mut session, "listed_twice", &[(a, est_a, var_a), (a, est_a, var_a)], None, None)
            .unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(refusal.detail, "evidence_ledger.repeated_source");
    assert_eq!(session.head_hex(), head, "a refused pool registers nothing");
    assert!(session.ledger().results().is_empty());

    // A consumer that re-derives the pool from declarations in which the
    // second study re-uses the first study's rows is refused the same way.
    let world = World::build(sc);
    let mut bundle = world.bundle(sc);
    bundle.studies[1].rows = bundle.studies[0].rows.clone();
    let refusal = consume(&bundle).expect_err("overlapping declarations");
    assert_refused(&refusal, &fx["mutations"]["evidence_overlap"]);
}

// ---------------------------------------------------------------------------
// 7: unsupported semantic conversions fail closed
// ---------------------------------------------------------------------------

#[test]
fn unsupported_semantic_conversions_fail_closed() {
    let fx = fixture();
    let (sc, c) = (&fx["scenario"], &fx["conversions"]);
    let world = World::build(sc);
    let model = &world.measured.identity;

    // The measured column offered as the construct it proxies.
    let refusal = measure(sc, scenario::INDICATOR).unwrap_err();
    assert_refused(&refusal, &c["measured_as_construct"]);

    // The value used without the model it was corrected under.
    let details = interface_pair_details(
        model,
        &InterfaceOptions { omit_measurement_model: true, ..InterfaceOptions::default() },
    );
    assert!(
        details.contains(&(
            text(&c["measurement_model_missing"]["code"]),
            text(&c["measurement_model_missing"]["detail"])
        )),
        "{details:?}"
    );

    // A mean offered where the consumer requires a law.
    let refusal = check_interfaces(
        model,
        &InterfaceOptions { decision_requires_law: true, ..InterfaceOptions::default() },
    )
    .unwrap_err();
    assert_refused(&refusal, &c["mean_for_law"]);

    // The honest declarations connect, producers first.
    let order = check_interfaces(model, &InterfaceOptions::default()).unwrap();
    assert_eq!(order, ["measurement", "mechanisms", "decision"]);

    // A mean source never answers a criterion that needs a law.
    let law = DecisionDeclaration { needs_law: true, ..world.declaration.clone() };
    let refusal = decide(&law, world.excess_risk, "snapshot", "plan").unwrap_err();
    assert_refused(&refusal, &fx["mutations"]["law"]);
}

// ---------------------------------------------------------------------------
// 8: a fresh consumer checks the decision against an independent oracle
// ---------------------------------------------------------------------------

#[test]
fn a_fresh_consumer_recomputes_the_decision_and_every_mutation_is_refused_as_expected() {
    let fx = fixture();
    let (sc, v, m) = (&fx["scenario"], &fx["values"], &fx["mutations"]);
    // The producer is gone before the consumer starts: only a bundle survives.
    let bundle = {
        let world = World::build(sc);
        world.bundle(sc)
    };
    let report = consume(&bundle).expect("the unmutated bundle is consumed");
    close(report.prevalence, num(&v["prevalence"]), exact(&fx), "prevalence");
    close(report.pooled_effect, num(&v["pooled_effect"]), exact(&fx), "pooled effect");
    close(report.pooled_variance, num(&v["pooled_variance"]), exact(&fx), "pooled variance");
    close(report.exposed_risk, num(&v["outcome_risk_exposed"]), exact(&fx), "do(X=1)");
    close(report.unexposed_risk, num(&v["outcome_risk_unexposed"]), exact(&fx), "do(X=0)");
    close(report.excess_risk, num(&v["excess_risk"]), loose(&fx), "excess risk");
    close(report.act, num(&v["act"]), loose(&fx), "act");
    close(report.hold, num(&v["hold"]), loose(&fx), "hold");
    assert_eq!(report.selected, text(&v["selected"]));
    assert!(report.snapshot_bound, "the decision is bound to the evidence state consumed");
    assert!(report.claim_graph_valid);
    // The producer's own report agrees with the consumer's closed form.
    close(bundle.reported_act, report.act, loose(&fx), "reported act");

    // Quantity: the mean is declared in other units than the contract's input.
    let mut mutated: Bundle = bundle.clone();
    mutated.decision.mean_units = "percent".into();
    assert_refused(&consume(&mutated).unwrap_err(), &m["quantity"]);

    // Law: the criterion needs the outcome law, which a mean cannot supply.
    let mut mutated = bundle.clone();
    mutated.decision.needs_law = true;
    assert_refused(&consume(&mutated).unwrap_err(), &m["law"]);

    // Support: the producer declares a real support where the consumer needs [0, 1].
    let mut mutated = bundle.clone();
    mutated.interface.producer_support_real = true;
    assert_refused(&consume(&mutated).unwrap_err(), &m["support"]);

    // Snapshot: the ledger bytes are another history's under the retained identity.
    let mut other = World::build(sc);
    other.session.add_source(study_source("late_arrival", "late", 0..10)).unwrap();
    let other_bundle = other.bundle(sc);
    let mut mutated = bundle.clone();
    mutated.ledger_bytes = other_bundle.ledger_bytes.clone();
    assert_refused(&consume(&mutated).unwrap_err(), &m["snapshot"]);

    // Snapshot, second form: the claim graph was verified at another ledger state.
    let mut mutated = bundle.clone();
    mutated.ledger_bytes = other_bundle.ledger_bytes.clone();
    mutated.ledger_identity = other_bundle.ledger_identity.clone();
    let refusal = consume(&mutated).unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "composition_verification.ledger_mismatch");

    // Evidence overlap: the second study re-uses rows of the first.
    let mut mutated = bundle.clone();
    mutated.studies[1].rows = mutated.studies[0].rows.start..(mutated.studies[0].rows.start + 100);
    assert_refused(&consume(&mutated).unwrap_err(), &m["evidence_overlap"]);

    // The same bundle is still consumed: no mutation reached the original.
    assert!(consume(&bundle).is_ok());
}

// ---------------------------------------------------------------------------
// 10: variants in which the downstream decision refuses
// ---------------------------------------------------------------------------

#[test]
fn an_upstream_withdrawal_makes_the_downstream_decision_refuse_end_to_end() {
    let fx = fixture();
    let sc = &fx["scenario"];
    let mut world = World::build(sc);
    assert!(gate(&world.graph, &world.session, world.pooled.result).is_none());

    // Consent for study A is withdrawn.
    world
        .session
        .revise_source(antecedent::SourceRevision::Withdraw {
            target: world.study_a,
            reason: "consent withdrawn".into(),
        })
        .unwrap();
    let pooled = world.pooled.result;
    assert_eq!(world.session.standing(pooled), Some(LedgerStanding::Withdrawn));

    // The claim graph withdraws everything resting on the pool, with no
    // numerical reason; the unrelated measurement stays valid.
    let verification = verify(Some(&world.graph), &world.graph, &world.session, pooled);
    assert_eq!(
        verification.revalidation().keys_of(Revalidation::Withdrawn),
        vec!["claim", "decision", "mechanisms", "pooled_effect"]
    );
    assert!(verification.revalidation().keys_of(Revalidation::StillValid).contains(&"measurement"));
    let refusal = gate(&world.graph, &world.session, pooled).expect("the decision refuses");
    assert_refused(&refusal, &fx["variants"]["withdrawal"]);
    // No number is served from the withdrawn pool, and new work cannot rest on it.
    let studies = sc["studies"].as_array().unwrap();
    let served = world.session.synthesis(pooled).unwrap_err();
    assert_eq!(served.code, "evidence_source_withdrawn");
    let again = pool(
        &mut world.session,
        "again",
        &[
            (world.study_a, num(&studies[0]["estimate"]), num(&studies[0]["variance"])),
            (world.study_b, num(&studies[1]["estimate"]), num(&studies[1]["variance"])),
        ],
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(again.code, "evidence_source_withdrawn");
    assert_eq!(again.detail, "evidence_ledger.source_withdrawn");
}

#[test]
fn a_dependence_gap_makes_the_downstream_decision_refuse_end_to_end() {
    let fx = fixture();
    let (sc, r) = (&fx["scenario"], &fx["variants"]);
    let studies = sc["studies"].as_array().unwrap();
    let vendor_claim = &sc["vendor_claim"];

    // The second contribution is a vendor claim whose observations are not
    // declared: its overlap with the first study is unknown.
    let mut session = StudyBuilder::evidence_ledger();
    let a = session
        .add_source(study_source("study_a", "cohort", rows_of(&studies[0]["rows"])))
        .unwrap();
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let pooled = pool(
        &mut session,
        "pooled_effect",
        &[
            (a, num(&studies[0]["estimate"]), num(&studies[0]["variance"])),
            (vendor, num(&vendor_claim["estimate"]), num(&vendor_claim["variance"])),
        ],
        None,
        None,
    );
    // There is no pooled result, hence no composition, claim or decision.
    let refusal = pooled.unwrap_err();
    assert_refused(&refusal, &r["dependence_gap_ledger"]);
    assert!(session.ledger().results().is_empty());

    // The joint-law rule refuses the same gap when asked for the combination.
    let moments = |id: &str, evidence: &str| PropagationInput {
        id: id.into(),
        evidence_id: evidence.into(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    };
    let request = PropagationRequest {
        operation: PropagationOperation::WeightedSum,
        inputs: vec![moments("study_a", &a.to_hex()), moments("vendor", &vendor.to_hex())],
        dependence: PropagationDependence::Unknown,
        reading: RequestedReading::Point,
    };
    let numbers = PropagationRequestWire::LinearCombination {
        coefficients: float_bits(&[0.5, 0.5]),
        means: float_bits(&[0.10, 0.20]),
        covariance: float_bits(&[0.0004, 0.0, 0.0, 0.0004]),
    };
    let error = propagate(&request, numbers, &ctx()).expect_err("an unknown dependence");
    assert_eq!(error.code(), text(&r["dependence_gap_rule"]["code"]));

    // Declaring the dependence is what licenses the combination: the vendor's
    // claim is entered with its rows, and a declared covariance pools it.
    let covered = session.add_source(study_source("vendor_rows", "cohort", 400..900)).unwrap();
    let licensed = pool(
        &mut session,
        "pooled_with_declared_dependence",
        &[
            (a, num(&studies[0]["estimate"]), num(&studies[0]["variance"])),
            (covered, num(&vendor_claim["estimate"]), num(&vendor_claim["variance"])),
        ],
        Some(vec![0.0004, 0.0002, 0.0002, 0.0004]),
        None,
    )
    .unwrap();
    close(licensed.variance, 0.0003, exact(&fx), "declared dependence");
}

// ---------------------------------------------------------------------------
// Point-only
// ---------------------------------------------------------------------------

#[test]
fn every_stage_is_point_only_and_no_interval_or_posterior_decision_ships() {
    let fx = fixture();
    let sc = &fx["scenario"];
    let world = World::build(sc);
    let code = text(&fx["variants"]["interval"]["code"]);

    // The ledger: a pooled interval is never licensed.
    let refusal = world.session.pooled_interval(world.pooled.result).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.code, code);
    // The claim graph holds no calibration record for the decision.
    let verification =
        verify(Some(&world.graph), &world.graph, &world.session, world.pooled.result);
    let decision = verification.classify("decision").unwrap();
    assert_eq!(decision.class, Revalidation::StillValid);
    // The decision under a criterion beyond a mean is refused, not approximated.
    let law = DecisionDeclaration { needs_law: true, ..world.declaration.clone() };
    let refused = decide(&law, world.excess_risk, "s", "p").unwrap_err();
    assert_refused(&refused, &fx["mutations"]["law"]);
    assert_eq!(text(&fx["calibration"]), "unmeasured");
    assert_eq!(text(&fx["label"]), "point_only");
    // The result of the decision has no statistical guarantee fields set.
    assert!(world.result.evpi.is_none());
}
