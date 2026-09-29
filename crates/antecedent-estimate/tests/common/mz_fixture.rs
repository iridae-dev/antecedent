//! The R-443 Figure 1(c,d) multi-source fixture shared by the mz-transport tests.
//!
//! `Z1 -> X -> Z2 -> Y` with `Z1 <-> X`, `Z1 <-> Z2`, `Z1 <-> Y`. Source `a` changes
//! the `Z1` and `Z2` mechanisms and can experiment on `Z2`; source `b` changes `Z1`
//! and `Y` and can experiment on `Z1`. Every law is enumerated from each
//! population's structural model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use std::sync::Arc;

use antecedent_core::{EvidenceCatalog, InterventionAssignment, RegimeKind, Value};
use antecedent_expr::{ExactDiscreteLaw, ExactTransportData};
use antecedent_identify::{MzTransportQuery, ZTransportSourceSpec};

use super::z_scm::{Mechanism, Scm, Spec, bit, build, diagram, vid};

pub const Z1: usize = 0;
pub const X: usize = 1;
pub const Z2: usize = 2;
pub const Y: usize = 3;

// Exogenous bits: e0 = Z1<->X, e1 = Z1<->Z2, e2 = Z1<->Y; e3..e6 private to Z1, X, Z2, Y.
const EXO: [f64; 7] = [0.35, 0.6, 0.45, 0.3, 0.7, 0.65, 0.4];

fn x_mechanism() -> Mechanism {
    Box::new(|v, e| bit(v[Z1] == 1) ^ bit(e[0] == 1 && e[4] == 1))
}
fn z2_target() -> Mechanism {
    Box::new(|v, e| bit((v[X] == 1 && e[5] == 1) || (v[X] == 0 && e[1] == 1)))
}
fn y_target() -> Mechanism {
    Box::new(|v, e| bit((v[Z2] == 1 && e[6] == 1) || (e[2] == 1 && e[6] == 0)))
}

pub fn target_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit((e[0] == 1 && e[3] == 1) || (e[1] == 1 && e[2] == 1))),
            x_mechanism(),
            z2_target(),
            y_target(),
        ],
    }
}

/// Source `a`: the Z1 and Z2 mechanisms differ from the target.
pub fn source_a_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit(e[3] == 1 || e[1] == 1)),
            x_mechanism(),
            Box::new(|v, e| bit(v[X] == 1) ^ bit(e[1] == 1 && e[5] == 1)),
            y_target(),
        ],
    }
}

/// Source `b`: the Z1 and Y mechanisms differ from the target.
pub fn source_b_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit(e[0] == 1 && e[2] == 1)),
            x_mechanism(),
            z2_target(),
            Box::new(|v, e| bit(v[Z2] == 1) ^ bit(e[2] == 1 && e[6] == 1)),
        ],
    }
}

pub fn graph() -> antecedent_graph::Admg {
    diagram(4, &[(Z1, X), (X, Z2), (Z2, Y)], &[(Z1, X), (Z1, Z2), (Z1, Y)], &[])
        .causal_graph()
        .clone()
}

pub fn sources() -> Vec<ZTransportSourceSpec> {
    vec![
        ZTransportSourceSpec {
            population: Arc::from("a"),
            controllable: Arc::from([vid(Z2)]),
            experiment_assignment: Arc::from([]),
            selection_targets: Arc::from([vid(Z1), vid(Z2)]),
        },
        ZTransportSourceSpec {
            population: Arc::from("b"),
            controllable: Arc::from([vid(Z1)]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: vid(Z1),
                value: Value::Bool(false),
            }]),
            selection_targets: Arc::from([vid(Z1), vid(Y)]),
        },
    ]
}

pub fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
    MzTransportQuery {
        outcomes: Arc::from([vid(Y)]),
        treatments: Arc::from([vid(X)]),
        target: Arc::from("target"),
        sources: sources.into(),
    }
}

/// Target observational law; `do(Z2 = 0/1)` in `a`; `do(Z1 = 0)` in `b`.
pub fn evidence() -> (EvidenceCatalog, ExactTransportData) {
    let specs = [
        Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 0)] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 1)] },
        Spec { population: "b", kind: RegimeKind::Experimental, assignments: vec![(Z1, 0)] },
    ];
    let (target, a, b) = (target_scm(), source_a_scm(), source_b_scm());
    let scm_for = |population: &str| -> &Scm {
        match population {
            "a" => &a,
            "b" => &b,
            _ => &target,
        }
    };
    build(&scm_for, 4, &specs, &["target", "a", "b"])
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "fixture probabilities are in [0, 1] and the sample sizes fit u64"
)]
pub fn empirical(data: &ExactTransportData, n: f64) -> ExactTransportData {
    let laws = data
        .laws()
        .iter()
        .map(|law| {
            let counts =
                law.probabilities().iter().map(|p| (p * n).round() as u64).collect::<Vec<_>>();
            let total = counts.iter().sum::<u64>() as f64;
            ExactDiscreteLaw::try_empirical(
                law.population(),
                law.regime(),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
                law.snapshot_identity(),
                law.tolerance(),
            )
            .unwrap()
            .with_empirical_counts(counts)
            .unwrap()
        })
        .collect::<Vec<_>>();
    ExactTransportData::try_new(laws, data.max_support_rows()).unwrap()
}

/// Declare every regime its own study, so independence between them is known.
pub fn with_studies(catalog: &EvidenceCatalog) -> EvidenceCatalog {
    let mut out = catalog.clone();
    let mut regimes = out.regimes.to_vec();
    for regime in &mut regimes {
        regime.study = Some(Arc::from(format!("study-{}", regime.id.raw())));
    }
    out.regimes = regimes.into();
    out
}
