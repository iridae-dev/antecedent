//! Temporal instrument identification on the unfolded window graph.
//!
//! A lagged variable is a valid instrument iff the static graphical conditions hold on the
//! unfolded window: it is not a descendant of the treatment, it is connected to the treatment, and
//! it is d-separated from the outcome once the treatment's out-edges are cut. Each case below is a
//! stationary template whose unfolded verdict is derived by hand.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TemporalNodeKey;
use antecedent_graph::TemporalDag;
use antecedent_identify::{IdentificationError, IdentificationStatus, TemporalIvIdentifier};

const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const U: u32 = 3;

fn node(graph: &mut TemporalDag, variable: u32, lag: u32) -> antecedent_graph::DenseNodeId {
    graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
}

/// `z_{t-2} -> x_{t-1} -> y_t` with the confounder `u_{t-1} -> x_{t-1}`, `u_{t-1} -> y_t`.
fn confounded_chain() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let (z2, x1, y0, u1) =
        (node(&mut g, Z, 2), node(&mut g, X, 1), node(&mut g, Y, 0), node(&mut g, U, 1));
    g.insert_directed(z2, x1).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g.insert_directed(u1, x1).unwrap();
    g.insert_directed(u1, y0).unwrap();
    g
}

fn query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(X), VariableId::from_raw(Y), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn key(variable: u32, offset: i32) -> TemporalNodeKey {
    TemporalNodeKey { variable: VariableId::from_raw(variable), offset }
}

#[test]
fn a_lagged_cause_of_the_treatment_is_the_instrument_and_the_confounder_is_not() {
    let found =
        TemporalIvIdentifier::new().identify_temporal_iv(&confounded_chain(), &query()).unwrap();
    assert_eq!(found.result.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
    assert_eq!(found.instruments.as_ref(), &[key(Z, -2)]);
    assert_eq!(found.treatment_key, key(X, -1));
    assert_eq!(found.outcome_key, key(Y, 0));
    let estimand = &found.result.estimands[0];
    assert_eq!(estimand.method.as_ref(), "temporal.iv.unfolded");
    assert_eq!(estimand.instruments.len(), 1);
    // The dense instrument id is the unfolded node of z at offset -2.
    let dense = found.indexer.dense_id(key(Z, -2)).unwrap();
    assert_eq!(estimand.instruments[0], VariableId::from_raw(dense));
    // The unobserved confounder must never be reported as an instrument.
    assert!(!found.instruments.iter().any(|k| k.variable == VariableId::from_raw(U)));
    // Exclusion and the Wald restriction are carried as declared premises.
    assert!(
        found.result.required_assumptions.entries.iter().any(|a| matches!(
            a.assumption,
            antecedent_core::Assumption::ExclusionRestriction { .. }
        ))
    );
}

#[test]
fn a_lagged_confounder_of_the_instrument_and_the_outcome_violates_exogeneity() {
    // `u_{t-2} -> z_{t-2}` and `u_{t-2} -> y_t`: Z and Y share an unblocked back-door path.
    let mut g = confounded_chain();
    let (u2, z2, y0) = (node(&mut g, U, 2), node(&mut g, Z, 2), node(&mut g, Y, 0));
    g.insert_directed(u2, z2).unwrap();
    g.insert_directed(u2, y0).unwrap();
    let found = TemporalIvIdentifier::new().identify_temporal_iv(&g, &query()).unwrap();
    assert_eq!(found.result.status, IdentificationStatus::NotIdentified);
    assert!(found.instruments.is_empty());
    assert!(found.result.estimands.is_empty());
    assert!(found.result.derivation.steps.iter().any(|s| s.rule.as_ref() == "temporal.iv.none"));
}

#[test]
fn a_direct_path_from_the_lagged_instrument_to_the_outcome_violates_exclusion() {
    let mut g = confounded_chain();
    let (z2, y0) = (node(&mut g, Z, 2), node(&mut g, Y, 0));
    g.insert_directed(z2, y0).unwrap();
    let found = TemporalIvIdentifier::new().identify_temporal_iv(&g, &query()).unwrap();
    assert_eq!(found.result.status, IdentificationStatus::NotIdentified);
    assert!(found.instruments.is_empty());
}

#[test]
fn an_instrument_independent_of_the_treatment_is_irrelevant() {
    // z_{t-2} -> y_t only: exclusion fails and there is no first stage.
    let mut g = TemporalDag::empty();
    let (x1, y0, z2) = (node(&mut g, X, 1), node(&mut g, Y, 0), node(&mut g, Z, 2));
    g.insert_directed(x1, y0).unwrap();
    let _ = z2;
    let found = TemporalIvIdentifier::new().identify_temporal_iv(&g, &query()).unwrap();
    assert_eq!(found.result.status, IdentificationStatus::NotIdentified);
}

#[test]
fn an_autoregressive_instrument_chain_is_not_certified() {
    // `z_{t-1} -> z_t` makes the ancestry of the instrument and of the treatment unbounded: no
    // finite window certifies it, so no instrument is reported valid.
    let mut g = confounded_chain();
    let (z1, z0) = (node(&mut g, Z, 1), node(&mut g, Z, 0));
    g.insert_directed(z1, z0).unwrap();
    let error = TemporalIvIdentifier::new().identify_temporal_iv(&g, &query()).unwrap_err();
    assert!(matches!(error, IdentificationError::NotCertified { .. }), "{error}");
}

#[test]
fn a_multi_step_schedule_is_unsupported() {
    let multi =
        TemporalEffectQuery::sustained(VariableId::from_raw(X), VariableId::from_raw(Y), -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1);
    let error =
        TemporalIvIdentifier::new().identify_temporal_iv(&confounded_chain(), &multi).unwrap_err();
    assert!(matches!(error, IdentificationError::UnsupportedQuery { .. }), "{error}");
}
