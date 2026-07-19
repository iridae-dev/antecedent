//! Property tests: identification on tiny generated SCMs.
//!
//! Prefer graph-structure asserts (expected adjustment / mediator sets). Where a
//! discrete table is cheap, also evaluate the returned functional against a
//! known interventional ATE (`ToleranceClass::StableFloat`). `causal-identify`
//! does not depend on `causal-model` / `causal-estimate`, so Monte Carlo
//! sampling from a fitted SCM is out of scope here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(test)]

use causal_core::{AverageEffectQuery, CausalQuery, ToleranceClass, Value, VariableId};
use causal_expr::{
    Assignment, DomainRef, EmpiricalTableProvider, EvalContext, FactorSpec,
    InterventionAssignment,
};
use causal_graph::{Dag, DenseNodeId};

use crate::backdoor::BackdoorIdentifier;
use crate::frontdoor::FrontDoorIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::result::IdentificationStatus;

fn f(x: f64) -> Value {
    Value::f64(x)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Classic confounding DAG: Z → T, Z → Y, T → Y. Backdoor set is `{Z}`.
fn confounding_dag() -> Dag {
    let mut g = Dag::with_variables(3);
    let t = DenseNodeId::from_raw(0);
    let y = DenseNodeId::from_raw(1);
    let z = DenseNodeId::from_raw(2);
    g.insert_directed(z, t).unwrap();
    g.insert_directed(z, y).unwrap();
    g.insert_directed(t, y).unwrap();
    g
}

/// Front-door DAG with unmeasured confounder U: U→T, U→Y, T→M→Y.
fn frontdoor_dag() -> Dag {
    let mut g = Dag::with_variables(4);
    let t = DenseNodeId::from_raw(0);
    let m = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    let u = DenseNodeId::from_raw(3);
    g.insert_directed(t, m).unwrap();
    g.insert_directed(m, y).unwrap();
    g.insert_directed(u, t).unwrap();
    g.insert_directed(u, y).unwrap();
    g
}

/// Discrete factors matching `causal-expr` backdoor eval: ATE = 0.45.
fn confounding_provider(t: VariableId, y: VariableId, z: VariableId) -> EmpiricalTableProvider {
    let mut p = EmpiricalTableProvider::new();
    p.set_domain(z, [f(0.0), f(1.0)]);
    p.set_domain(y, [f(0.0), f(1.0)]);
    p.set_domain(t, [f(0.0), f(1.0)]);

    for (zval, prob) in [(0.0, 0.5), (1.0, 0.5)] {
        let spec = FactorSpec {
            variables: &[z],
            conditioned_on: &[],
            intervention: &[],
            domain: DomainRef::Observational,
        };
        let assign = Assignment::from_pairs([(z, f(zval))]);
        p.insert_probability(&spec, &assign, prob).unwrap();
    }

    // E[Y|T=1,Z=0]=0.8, E[Y|T=1,Z=1]=0.6, E[Y|T=0,Z=0]=0.3, E[Y|T=0,Z=1]=0.2
    // → E[Y|do(1)]=0.7, E[Y|do(0)]=0.25, ATE=0.45
    let ey = |tlev: f64, zlev: f64| -> f64 {
        match (tlev.to_bits(), zlev.to_bits()) {
            (tb, zb) if tb == 1.0f64.to_bits() && zb == 0.0f64.to_bits() => 0.8,
            (tb, zb) if tb == 1.0f64.to_bits() && zb == 1.0f64.to_bits() => 0.6,
            (tb, zb) if tb == 0.0f64.to_bits() && zb == 0.0f64.to_bits() => 0.3,
            (tb, zb) if tb == 0.0f64.to_bits() && zb == 1.0f64.to_bits() => 0.2,
            _ => panic!("bad levels"),
        }
    };
    for tlev in [0.0, 1.0] {
        let interv = [InterventionAssignment { variable: t, value: f(tlev) }];
        for zlev in [0.0, 1.0] {
            let p_y1 = ey(tlev, zlev);
            for (yval, prob) in [(1.0, p_y1), (0.0, 1.0 - p_y1)] {
                let spec = FactorSpec {
                    variables: &[y],
                    conditioned_on: &[z],
                    intervention: &interv,
                    domain: DomainRef::Interventional,
                };
                let assign = Assignment::from_pairs([(y, f(yval)), (z, f(zlev))]);
                p.insert_probability(&spec, &assign, prob).unwrap();
            }
        }
    }
    p
}

#[test]
fn id_scm_backdoor_confounding_structure_and_functional() {
    let g = confounding_dag();
    let id = BackdoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(0), v(1)));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        res.estimands.iter().any(|e| e.adjustment_set.as_ref() == [v(2)]),
        "expected adjustment {{Z}}; got {:?}",
        res.estimands.iter().map(|e| e.adjustment_set.clone()).collect::<Vec<_>>()
    );

    let est = res
        .estimands
        .iter()
        .find(|e| e.adjustment_set.as_ref() == [v(2)])
        .expect("Z estimand");
    let provider = confounding_provider(v(0), v(1), v(2));
    let ate = res
        .arena
        .compile(est.functional)
        .unwrap()
        .evaluate(&res.arena, &provider, &EvalContext::default())
        .unwrap();
    assert!(
        ToleranceClass::StableFloat.close(ate, 0.45),
        "identified functional ate={ate}, expected 0.45"
    );
}

#[test]
fn id_scm_frontdoor_structure_on_known_pattern() {
    let g = frontdoor_dag();
    let id = FrontDoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::average_effect(AverageEffectQuery::with_levels(v(0), v(2), 0.0, 1.0));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        res.estimands.iter().any(|e| e.mediators.as_ref() == [v(1)]),
        "expected mediator {{M}}; got {:?}",
        res.estimands.iter().map(|e| e.mediators.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn id_scm_chain_empty_adjustment() {
    // T → M → Y: no backdoor paths; empty Z is valid.
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let id = BackdoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(0), v(2)));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(res.estimands.iter().any(|e| e.adjustment_set.is_empty()));
}

/// Graphs that match the classic confounding pattern: identifier recovers Z.
#[test]
fn id_scm_random_confounding_pattern_recovers_z() {
    let mut rng = causal_core::CausalRng::from_seed(2026);
    let id = BackdoorIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut hits = 0u32;
    for _ in 0..25 {
        let n = if rng.next_u64() % 2 == 0 { 3u32 } else { 4u32 };
        let mut g = Dag::with_variables(n);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();
        if n == 4 && rng.next_u64() % 2 == 0 {
            let _ = g.insert_directed(DenseNodeId::from_raw(3), y);
        }
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(0), v(1)));
        let Ok(res) = id.identify(&prep, &q, &mut ws) else {
            continue;
        };
        if res.status != IdentificationStatus::NonparametricallyIdentified {
            continue;
        }
        assert!(
            res.estimands.iter().any(|e| {
                e.adjustment_set.as_ref() == [v(2)] || e.adjustment_set.iter().any(|x| *x == v(2))
            }),
            "expected Z in some adjustment set; got {:?}",
            res.estimands.iter().map(|e| e.adjustment_set.clone()).collect::<Vec<_>>()
        );
        hits += 1;
    }
    assert!(hits >= 10, "too few successful identifies: {hits}");
}

#[test]
fn id_scm_nonbinary_levels_still_identify() {
    let g = confounding_dag();
    let id = BackdoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::average_effect(AverageEffectQuery::with_levels(v(0), v(1), -2.0, 3.0));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(res.estimands.iter().any(|e| e.adjustment_set.as_ref() == [v(2)]));
}
