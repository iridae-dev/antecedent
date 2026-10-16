//! Whole-method calibration of the posterior interval of a count-fitted finite table
//! model: fit every conditional from a sample (Dirichlet posterior per parent
//! configuration), condition on the evidence by exact enumeration, and score the
//! equal-tailed posterior quantiles of `P(Y = 1 | evidence)` against the value under
//! the true tables.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs
//! only at the final measurement step.
//!
//! Gated designs (levels 0.90 and 0.95, sample size `grid_n(300)`):
//!
//! * `confounded_signal`: a common cause `Z` of `S` and `Y` with `S -> Y`, evidence
//!   `S = 1`, so `P(Y | S)` and `P(Y | do(S))` differ.
//! * `measured_signal`: the same model with a measured signal `M` of `S` with known
//!   sensitivity 0.90 and specificity 0.85 (a declared node), evidence `M = 1`.
//!
//! Named boundaries, each under its own calibration structure:
//!
//! * `sparse_parent_configuration`: a rare parent configuration (`P(Z = 1) = 0.04`)
//!   that is empty or nearly empty in small samples.
//! * `rare_evidence`: the conditioning event has probability about 0.03.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "calibration-internal")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;

use antecedent_estimate::count_fitted_posterior::{
    CountFittedModel, CountNode, INTERVAL_METHOD, NodeTable, PosteriorQuery,
    boundary_calibration_basis, calibration_basis,
};
use calibration::{CoverageTally, RecordKey, grid_n, map_replicates, n_sim};
use candidate::Generator;

const DRAWS: u32 = 2000;
const PRIOR: f64 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    None,
    SparseParentConfiguration,
    RareEvidence,
}

impl Boundary {
    const fn label(self) -> &'static str {
        match self {
            Self::None => "gated",
            Self::SparseParentConfiguration => "sparse_parent_configuration",
            Self::RareEvidence => "rare_evidence",
        }
    }
}

#[derive(Clone, Copy)]
struct Design {
    test: &'static str,
    dgp: &'static str,
    signal: bool,
    boundary: Boundary,
    level: f64,
}

/// True tables: `P(Z=1)`, `P(S=1 | Z)`, `P(Y=1 | Z, S)` indexed `z * 2 + s`.
struct Truth {
    z: f64,
    s_given_z: [f64; 2],
    y_given_zs: [f64; 4],
}

fn truth_of(design: Design) -> Truth {
    match design.boundary {
        Boundary::None => {
            Truth { z: 0.4, s_given_z: [0.25, 0.7], y_given_zs: [0.15, 0.4, 0.45, 0.8] }
        }
        Boundary::SparseParentConfiguration => {
            Truth { z: 0.04, s_given_z: [0.3, 0.7], y_given_zs: [0.15, 0.4, 0.45, 0.8] }
        }
        Boundary::RareEvidence => {
            Truth { z: 0.4, s_given_z: [0.01, 0.06], y_given_zs: [0.15, 0.4, 0.45, 0.8] }
        }
    }
}

const SE: f64 = 0.90;
const SP: f64 = 0.85;

fn declared_signal() -> CountNode {
    CountNode {
        cardinality: 2,
        parents: vec![1],
        table: NodeTable::Declared(vec![SP, 1.0 - SP, 1.0 - SE, SE]),
    }
}

fn nodes(design: Design, z: Vec<u64>, s: Vec<u64>, y: Vec<u64>) -> Vec<CountNode> {
    let mut nodes = vec![
        CountNode { cardinality: 2, parents: vec![], table: NodeTable::Counts(z) },
        CountNode { cardinality: 2, parents: vec![0], table: NodeTable::Counts(s) },
        CountNode { cardinality: 2, parents: vec![0, 1], table: NodeTable::Counts(y) },
    ];
    if design.signal {
        nodes.push(declared_signal());
    }
    nodes
}

fn query(design: Design) -> PosteriorQuery {
    PosteriorQuery {
        target: 2,
        level: 1,
        evidence: vec![if design.signal { (3, 1) } else { (1, 1) }],
    }
}

/// The value under the true tables, from the same enumeration with declared nodes.
fn true_value(design: Design) -> f64 {
    let t = truth_of(design);
    let row = |p: f64| [1.0 - p, p];
    let mut nodes = vec![
        CountNode {
            cardinality: 2,
            parents: vec![],
            table: NodeTable::Declared(row(t.z).to_vec()),
        },
        CountNode {
            cardinality: 2,
            parents: vec![0],
            table: NodeTable::Declared(t.s_given_z.iter().flat_map(|&p| row(p)).collect()),
        },
        CountNode {
            cardinality: 2,
            parents: vec![0, 1],
            table: NodeTable::Declared(t.y_given_zs.iter().flat_map(|&p| row(p)).collect()),
        },
    ];
    if design.signal {
        nodes.push(declared_signal());
    }
    CountFittedModel::new(nodes, 0)
        .expect("true model")
        .point(&query(design), PRIOR)
        .expect("true value")
}

fn replicate(design: Design, n: usize, seed: u64) -> Option<(f64, f64)> {
    let t = truth_of(design);
    let mut g = Generator::new(seed);
    let (mut z, mut s, mut y) = (vec![0_u64; 2], vec![0_u64; 4], vec![0_u64; 8]);
    for _ in 0..n {
        let zi = g.binary(t.z);
        let si = g.binary(t.s_given_z[zi]);
        let yi = g.binary(t.y_given_zs[zi * 2 + si]);
        z[zi] += 1;
        s[zi * 2 + si] += 1;
        y[(zi * 2 + si) * 2 + yi] += 1;
    }
    // Rows are `[level 0, level 1]` per parent configuration.
    let z = vec![z[0], z[1]];
    let s: Vec<u64> = s;
    let model = CountFittedModel::new(nodes(design, z, s, y), n as u64).ok()?;
    let interval =
        model.posterior_interval(&query(design), design.level, PRIOR, DRAWS, seed).ok()?;
    Some((interval.lower, interval.upper))
}

fn run(design: Design, base: usize) {
    let n = grid_n(base);
    let truth = true_value(design);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: design.test, dgp: design.dgp, interval: INTERVAL_METHOD },
        design.level,
    );
    let results = map_replicates(n_sim(), |replicate_index| {
        replicate(design, n, calibration::grid_seed(0x5b_0000 + replicate_index))
    });
    for result in results {
        let Some(interval) = result else {
            tally.skip();
            continue;
        };
        let basis = match design.boundary {
            Boundary::None => calibration_basis(n as u64, design.level, DRAWS),
            other => boundary_calibration_basis(other.label(), n as u64, design.level, DRAWS),
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some(interval), truth);
    }
    match design.boundary {
        Boundary::None => tally.assert(),
        _ => tally.emit_named_boundary(),
    }
}

macro_rules! design {
    ($name:ident, $dgp:literal, $signal:expr, $boundary:expr, $level:expr, $gated:expr) => {
        #[test]
        #[ignore = "calibration: run only at the final measurement step"]
        fn $name() {
            let _ = $gated;
            run(
                Design {
                    test: stringify!($name),
                    dgp: $dgp,
                    signal: $signal,
                    boundary: $boundary,
                    level: $level,
                },
                300,
            );
        }
    };
}

design!(
    count_fitted_confounded_signal_l90,
    "common_cause_z_of_s_and_y_with_s_to_y_counts_fitted_dirichlet_posterior_evidence_s1",
    false,
    Boundary::None,
    0.90,
    true
);
design!(
    count_fitted_confounded_signal_l95,
    "common_cause_z_of_s_and_y_with_s_to_y_counts_fitted_dirichlet_posterior_evidence_s1",
    false,
    Boundary::None,
    0.95,
    true
);
design!(
    count_fitted_measured_signal_l90,
    "common_cause_z_of_s_and_y_with_declared_measured_signal_se_090_sp_085_evidence_m1",
    true,
    Boundary::None,
    0.90,
    true
);
design!(
    count_fitted_measured_signal_l95,
    "common_cause_z_of_s_and_y_with_declared_measured_signal_se_090_sp_085_evidence_m1",
    true,
    Boundary::None,
    0.95,
    true
);
design!(
    count_fitted_sparse_parent_configuration_boundary_l95,
    "rare_parent_configuration_pz1_004_counts_fitted_dirichlet_posterior_evidence_s1",
    false,
    Boundary::SparseParentConfiguration,
    0.95,
    false
);
design!(
    count_fitted_rare_evidence_boundary_l95,
    "rare_conditioning_event_ps1_about_003_counts_fitted_dirichlet_posterior_evidence_s1",
    false,
    Boundary::RareEvidence,
    0.95,
    false
);
