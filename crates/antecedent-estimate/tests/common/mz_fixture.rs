//! The R-443 Figure 1(c,d) multi-source fixture shared by the mz-transport tests.
//!
//! `Z1 -> X -> Z2 -> Y` with `Z1 <-> X`, `Z1 <-> Z2`, `Z1 <-> Y`. Source `a` changes
//! the `Z1` and `Z2` mechanisms and can experiment on `Z2` (diagram (c)); source
//! `b` changes `Z1` and `Y` and can experiment on `Z1` (diagram (d) plus a
//! selection node into `Z1`, which the paper's (d) lacks; `do(Z1)` cuts it, so
//! the paper's transport formula still applies). Every law is enumerated from
//! each population's structural model.
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

/// How a second table of source `b`'s `do(Z1 = 0)` trial relates to its joint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedTable {
    /// A second regime publishing the identical joint table.
    Identical,
    /// A second regime publishing the `(X, Z2)` margin of the joint.
    Margin,
}

/// Regime id of the second table [`with_shared_b_trial`] adds.
pub const SHARED_REGIME: u32 = 4;
/// Regime id of source `b`'s `do(Z1 = 0)` joint.
pub const B_TRIAL_REGIME: u32 = 3;

/// Declared studies plus a second table of `b`'s `do(Z1 = 0)` trial that names
/// the same forwarded dataset (`b-trial`) and study as the joint: either the
/// identical table or its recorded `(X, Z2)` margin, with counts that are exactly
/// that margin of the joint's counts.
pub fn with_shared_b_trial(
    catalog: &EvidenceCatalog,
    data: &ExactTransportData,
    table: SharedTable,
) -> (EvidenceCatalog, ExactTransportData) {
    use antecedent_core::{DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeId};
    let mut out = with_studies(catalog);
    let joint_id = RegimeId::from_raw(B_TRIAL_REGIME);
    let shared_id = RegimeId::from_raw(SHARED_REGIME);
    let joint = out.regimes.iter().find(|r| r.id == joint_id).unwrap().clone();
    let measured: Vec<_> = match table {
        SharedTable::Identical => joint.measured.to_vec(),
        SharedTable::Margin => vec![vid(X), vid(Z2)],
    };
    let mut shared = EvidenceRegime::try_new(
        shared_id,
        joint.kind,
        EvidenceKind::Available,
        joint.interventions.to_vec(),
        joint.intervention_values.to_vec(),
        measured,
        joint.population.as_ref(),
        DistributionAvailability::Joint,
    )
    .unwrap();
    shared.study.clone_from(&joint.study);
    let mut regimes = out.regimes.to_vec();
    regimes.push(shared);
    out.regimes = regimes.into();
    let mut bindings = out.bindings.to_vec();
    let mut shared_binding = bindings.iter().find(|b| b.regime == joint_id).unwrap().clone();
    shared_binding.regime = shared_id;
    shared_binding.snapshot_identity = Arc::from("b-3-second");
    bindings.push(shared_binding);
    for binding in &mut bindings {
        if binding.regime == joint_id || binding.regime == shared_id {
            binding.dataset_identity = Some(Arc::from("b-trial"));
        }
    }
    out.bindings = bindings.into();
    let law = data.laws().iter().find(|law| law.regime() == joint_id).unwrap();
    let joint_counts = law.empirical_counts().unwrap();
    // Joint axes are (X, Z2, Y), last fastest: the (X, Z2) margin sums Y out.
    let (axes, counts) = match table {
        SharedTable::Identical => (law.axes().to_vec(), joint_counts.to_vec()),
        SharedTable::Margin => (
            law.axes()[..2].to_vec(),
            joint_counts.chunks(2).map(|pair| pair.iter().sum()).collect::<Vec<u64>>(),
        ),
    };
    #[expect(clippy::cast_precision_loss, reason = "fixture counts fit f64 exactly")]
    let total = counts.iter().sum::<u64>() as f64;
    #[expect(clippy::cast_precision_loss, reason = "fixture counts fit f64 exactly")]
    let probabilities = counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>();
    let shared_law = ExactDiscreteLaw::try_empirical(
        law.population(),
        shared_id,
        law.interventions().to_vec(),
        axes,
        probabilities,
        "b-3-second",
        law.tolerance(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap();
    let mut laws = data.laws().to_vec();
    laws.push(shared_law);
    (out, ExactTransportData::try_new(laws, data.max_support_rows()).unwrap())
}

/// Declared studies, with `a`'s two `do(Z2)` arms naming one forwarded dataset
/// (`a-trial`): two different tables that are neither identical nor a margin of
/// one another, so they cannot be resampled jointly.
pub fn with_conflicting_a_trial(catalog: &EvidenceCatalog) -> EvidenceCatalog {
    let mut out = with_studies(catalog);
    let mut bindings = out.bindings.to_vec();
    for binding in &mut bindings[1..3] {
        binding.dataset_identity = Some(Arc::from("a-trial"));
    }
    out.bindings = bindings.into();
    out
}

/// Source `c`: a twin of `a` that certifies the same factor `Q[Y]` (same
/// controllable `Z2` and selection on `Z1`, `Z2`) from a different population:
/// its `Z1` and `Z2` mechanisms differ from both the target's and `a`'s, while
/// `X` and `Y` keep the target's mechanisms.
pub fn source_c_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit(e[0] == 1 || e[3] == 1)),
            x_mechanism(),
            Box::new(|v, e| bit(v[X] == 1) ^ bit(e[1] == 1 && e[5] == 0)),
            y_target(),
        ],
    }
}

/// `a`, `b` and the twin `c` (declared last, so canonical order is `a`, `b`, `c`).
pub fn twin_sources() -> Vec<ZTransportSourceSpec> {
    let mut all = sources();
    all.push(ZTransportSourceSpec {
        population: Arc::from("c"),
        controllable: Arc::from([vid(Z2)]),
        experiment_assignment: Arc::from([]),
        selection_targets: Arc::from([vid(Z1), vid(Z2)]),
    });
    all
}

/// [`evidence`] plus `c`'s `do(Z2 = 0/1)` laws as regimes 4 and 5 (regimes 0 to 3
/// are exactly those of [`evidence`]).
pub fn evidence_with_twin() -> (EvidenceCatalog, ExactTransportData) {
    let specs = [
        Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 0)] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 1)] },
        Spec { population: "b", kind: RegimeKind::Experimental, assignments: vec![(Z1, 0)] },
        Spec { population: "c", kind: RegimeKind::Experimental, assignments: vec![(Z2, 0)] },
        Spec { population: "c", kind: RegimeKind::Experimental, assignments: vec![(Z2, 1)] },
    ];
    let (target, a, b, c) = (target_scm(), source_a_scm(), source_b_scm(), source_c_scm());
    let scm_for = |population: &str| -> &Scm {
        match population {
            "a" => &a,
            "b" => &b,
            "c" => &c,
            _ => &target,
        }
    };
    build(&scm_for, 4, &specs, &["target", "a", "b", "c"])
}
