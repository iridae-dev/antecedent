//! Path-specific natural effects on ADMGs with latent confounding among the outcome's
//! ancestors, checked against exact enumerated latent-variable structural models.
//!
//! The identified functional is evaluated on the observational joint of a random model and
//! compared with the effect computed by solving the model with the treatment read at one
//! level along the selected edges and at the control level along the others, sharing every
//! exogenous state between the worlds. The district formula shares no code with that
//! evaluation. A recanting district and an unidentified district kernel are refused with
//! their own derivation step.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

#[path = "support/path_specific_scm.rs"]
mod path_specific_scm;

use antecedent_core::{
    CausalQuery, IdentificationStatus, MediationContrast, MediationQuery, PathSpecificEffectQuery,
};
use path_specific_scm::latent_scm::{LatentScm, Rng};
use path_specific_scm::{admg, evaluate, has_rule, identify, path_specific_truth, v};

/// `T -> A -> Y`, `T -> B -> Y` with `A <-> Y`: the effect along the paths through `A`
/// leaves the treatment into the district `{A, Y}` at the active level and into `{B}` at
/// the control level, so it is the district formula and not the node-wise product.
#[test]
fn district_formula_matches_the_enumerated_effect_with_latent_confounding() {
    let directed = [(0, 1), (0, 2), (1, 3), (2, 3)];
    let bidirected = [(1, 3)];
    let graph = admg(4, &directed, &bidirected);
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]),
    );
    let result = identify(&graph, &query);
    assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(has_rule(&result, "path_specific.district_gformula"), "{:?}", result.derivation.steps);

    let mut distinct = std::collections::BTreeSet::new();
    for seed in 0..8u64 {
        let scm = LatentScm::random(&mut Rng::new(seed), &[2, 2, 2, 2], &directed, &bidirected, 3);
        let truth = path_specific_truth(&scm, 0, 3, &[1]);
        let actual = evaluate(&result, &scm);
        assert!(
            (actual - truth).abs() < 1e-10,
            "seed {seed}: district formula {actual}, enumerated effect {truth}"
        );
        distinct.insert((truth * 1e9).round() as i64);
    }
    assert!(distinct.len() > 4, "the sweep must move the effect, got {distinct:?}");
}

/// Natural direct effect with a pre-treatment covariate that shares a latent cause with the
/// outcome: `W -> T`, `W -> M`, `W -> Y`, `T -> M`, `T -> Y`, `M -> Y`, `W <-> Y`.
#[test]
fn natural_direct_effect_with_covariate_confounding_matches_the_enumerated_effect() {
    let directed = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let bidirected = [(0, 3)];
    let graph = admg(4, &directed, &bidirected);
    let query = CausalQuery::Mediation(MediationQuery::binary(
        v(1),
        v(3),
        [v(2)],
        MediationContrast::NaturalDirect,
    ));
    let result = identify(&graph, &query);
    assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(has_rule(&result, "path_specific.district_gformula"), "{:?}", result.derivation.steps);

    for seed in 0..8u64 {
        let scm =
            LatentScm::random(&mut Rng::new(100 + seed), &[2, 2, 2, 2], &directed, &bidirected, 3);
        // The direct edge `T -> Y` is the selected one; the mediator reads the control level.
        let truth = path_specific_truth(&scm, 1, 3, &[3]);
        let actual = evaluate(&result, &scm);
        assert!(
            (actual - truth).abs() < 1e-10,
            "seed {seed}: natural direct effect {actual}, enumerated effect {truth}"
        );
    }
}

/// `A <-> B` puts the selected child `A` and the complementary child `B` in one district.
#[test]
fn a_district_entered_by_both_kinds_of_treatment_edge_is_refused() {
    let graph = admg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[(1, 2)]);
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]),
    );
    let result = identify(&graph, &query);
    assert_eq!(result.status, IdentificationStatus::NotIdentified);
    assert!(has_rule(&result, "path_specific.recanting_district"), "{:?}", result.derivation.steps);
    assert!(result.estimands.is_empty());
}

/// Mediator-outcome confounding is the classic recanting district of the natural direct
/// effect.
#[test]
fn mediator_outcome_confounding_refuses_the_natural_direct_effect() {
    let graph = admg(3, &[(0, 1), (1, 2), (0, 2)], &[(1, 2)]);
    let query = CausalQuery::Mediation(MediationQuery::binary(
        v(0),
        v(2),
        [v(1)],
        MediationContrast::NaturalDirect,
    ));
    let result = identify(&graph, &query);
    assert_eq!(result.status, IdentificationStatus::NotIdentified);
    assert!(has_rule(&result, "path_specific.recanting_district"), "{:?}", result.derivation.steps);
}

/// `T <-> B` leaves the kernel `Q[{B}] = P(b | do(t, a, y))` with a hedge, so the effect that
/// reads the complementary edge into `B` at the control level is not identified.
#[test]
fn an_unidentified_district_kernel_leaves_the_effect_unidentified() {
    let graph = admg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[(0, 2)]);
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]),
    );
    let result = identify(&graph, &query);
    assert_eq!(result.status, IdentificationStatus::NotIdentified);
    assert!(has_rule(&result, "path_specific.district_hedge"), "{:?}", result.derivation.steps);
}
