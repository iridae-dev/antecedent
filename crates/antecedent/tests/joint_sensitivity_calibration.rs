//! Coverage of the 2.2B X3 conservative endpoint percentile bootstrap (record
//! `2.2B.X3.joint_sensitivity_uncertainty`).
//!
//! The registered surrogate formula's cited joint is `P(W) P(X) P(Y | W, X)`
//! under `do(Z = 0)` with a W-dependent effect. The truth is `psi(delta0)` at
//! the EXTREMAL deviation of the declared box: the vertex that attains the
//! upper end `U` of the exact range on the true law (the tight side of the
//! one-sided target; an interior point such as `delta0 = 0` is the easiest
//! point and would not test it). At the zero box that vertex is `delta0 = 0`.
//!
//! * `z_joint_sensitivity_zero_box`: the zero box, where the composition is
//!   exactly the ordinary percentile bootstrap; a two-sided gated record.
//! * `z_joint_sensitivity_positive_box`: a positive box, where the target is
//!   one-sided: coverage of the true upper extremal bound `U` at least nominal,
//!   bound by the conservative upper endpoint `q_{(1+a)/2}(U*)`. It is a
//!   `one_sided` record (`CoverageTally::assert_one_sided`, side `upper`): a
//!   floor at the level with no upper band or ceiling, so the interval's
//!   designed over-coverage passes.
//!
//! Both are registered in `scripts/gate_calibration.sh` (`run_js`, measured over
//! the sample-size grid) and are measured once at the release cut; until then
//! the interval route stays closed (`cell_not_licensed`). The estimator is
//! compiled only under `calibration-internal`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

mod common;

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExactTransportData, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundZTransportFunctional, SidLimits, ZTransportQuery, ZTransportResult,
    bind_z_transport_catalog, identify_z_transport,
};
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound,
    joint_sensitivity_bootstrap_interval_internal,
};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    coverage_mcse, grid_n, map_replicates, n_sim, needs_one_sided_recheck, passes_one_sided,
    passes_precision, stream_seed, unit_uniform,
};

const INTERVAL: &str = "percentile_bootstrap";
const REPLICATES: u32 = 199;
/// The positive-box record's one-sided target.
const UPPER_TARGET: &str = "true upper extremal bound U (psi at the maximizing vertex of the true \
                            law) covered by the conservative upper endpoint q_{(1+a)/2}(U*)";
const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn p_y1(w: bool, x: bool) -> f64 {
    match (w, x) {
        (false, false) => 0.2,
        (false, true) => 0.5,
        (true, false) => 0.3,
        (true, true) => 0.8,
    }
}

fn probabilities() -> Vec<f64> {
    let mut out = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { 0.35 } else { 0.65 };
                let p_x = if x { 0.4 } else { 0.6 };
                out.push(p_w * p_x * if y { p_y1(w, x) } else { 1.0 - p_y1(w, x) });
            }
        }
    }
    out
}

/// `psi(delta0)` at the maximizing vertex of the box with both fractions
/// `fraction`, built from the generating model as an explicit contaminated
/// target: the kernel replacement puts all mass on `y = 1` in the active arm
/// and on `y = 0` in the control arm, and the parent replacement puts all mass
/// on `argmax_w D+(w)` (here `w = 1`, whose effect is larger). At `fraction = 0`
/// this is `psi(0) = sum_w P(w) [P(y | w, 1) - P(y | w, 0)]`.
fn truth(fraction: f64) -> f64 {
    let e = fraction;
    let q_y1 = |w: bool, x: bool| (1.0 - e) * p_y1(w, x) + e * if x { 1.0 } else { 0.0 };
    let p_w1 = 0.35;
    let q_w1 = (1.0 - e) * p_w1 + e;
    let effect = |w: bool| q_y1(w, true) - q_y1(w, false);
    (1.0 - q_w1) * effect(false) + q_w1 * effect(true)
}

fn diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

fn functional() -> BoundZTransportFunctional {
    let query = ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram(),
        &query,
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("registered surrogate");
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [Environment::try_new(
            "source",
            [W, Z, X, Y]
                .into_iter()
                .map(|variable| VariableCoordinate {
                    variable,
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("snapshot-0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap();
    bind_z_transport_catalog(&diagram(), &query, &proof, &catalog).unwrap()
}

fn z_joint_surrogate_counts(n: usize, seed: u64) -> ExactTransportData {
    let probabilities = probabilities();
    let mut counts = vec![0u64; probabilities.len()];
    for row in 0..n {
        let mut threshold = unit_uniform(stream_seed(seed, row as u64));
        let mut chosen = probabilities.len() - 1;
        for (index, probability) in probabilities.iter().enumerate() {
            if threshold < *probability {
                chosen = index;
                break;
            }
            threshold -= probability;
        }
        counts[chosen] += 1;
    }
    let binary = |variable| DiscreteAxis {
        variable,
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let law = ExactDiscreteLaw::try_empirical(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary(W), binary(X), binary(Y)],
        counts.iter().map(|c| *c as f64 / n as f64).collect::<Vec<_>>(),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap();
    ExactTransportData::try_new([law], 64).unwrap()
}

fn construction() -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.z_joint_sensitivity_endpoint_bootstrap".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_interventional_mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn box_spec(fraction: f64) -> JointDeviationSpec {
    JointDeviationSpec::new(vec![
        JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: fraction },
        JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: fraction },
    ])
}

fn measure(tally: &mut CoverageTally, fraction: f64, n: usize) {
    let functional = functional();
    let diagram = diagram();
    let spec = box_spec(fraction);
    let rows = map_replicates(n_sim(), |rep| {
        let data = z_joint_surrogate_counts(n, rep.wrapping_mul(1_000_037));
        let ctx = ExecutionContext::for_tests(rep);
        joint_sensitivity_bootstrap_interval_internal(
            &diagram,
            &functional,
            &data,
            &spec,
            REPLICATES,
            REPORTED_LEVEL,
            &ctx,
        )
        .ok()
        .and_then(Result::ok)
        .map(|interval| (interval.replicates_ok, (interval.lower, interval.upper)))
    });
    for row in rows {
        match row {
            Some((replicates_ok, band)) => {
                tally.bind(
                    &construction(),
                    ScopeFacts {
                        row_count: n as u64,
                        replicates_ok: Some(replicates_ok),
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(Some(band), truth(fraction));
            }
            None => tally.skip(),
        }
    }
}

/// The coverage truths are the extremal vertex of the exact range on the true
/// law: the upper end `U` the joint evaluator computes, and the baseline at the
/// zero box.
#[test]
fn the_calibration_truths_are_the_extremal_vertex_of_the_true_law() {
    let ctx = ExecutionContext::for_tests(1);
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [W, X, Y].map(|variable| DiscreteAxis {
            variable,
            values: Arc::from([Value::Bool(false), Value::Bool(true)]),
        }),
        probabilities(),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    let exact = ExactTransportData::try_new([law], 64).unwrap();
    for fraction in [0.0, 0.05, 0.3] {
        let result = antecedent_validate::z_transport_joint_mechanism_sensitivity(
            &diagram(),
            &functional(),
            &exact,
            &box_spec(fraction),
            &ctx,
        )
        .unwrap();
        assert!((result.range.maximum - truth(fraction)).abs() <= 1e-12, "fraction {fraction}");
        if fraction > 0.0 {
            assert_eq!(result.range.maximizing_parent_level, Some(1));
            assert!(truth(fraction) > result.baseline, "the vertex is not the interior point");
        } else {
            assert_eq!(result.range.maximum.to_bits(), result.baseline.to_bits());
            assert_eq!(result.range.maximizing_parent_level, None);
        }
    }
}

/// The positive-box record's acceptance is one-sided: a floor at the level and
/// no ceiling, so the conservative interval's over-coverage passes and a
/// shortfall fails, and a short run that shortfalls by more than the recheck
/// margin is rechecked rather than passed.
#[test]
fn the_one_sided_role_accepts_over_coverage_and_refuses_a_shortfall() {
    let level = REPORTED_LEVEL;
    // Over-coverage that the two-sided gate refuses passes one-sided.
    assert!(!passes_precision(2000, level, 0.995));
    assert!(passes_one_sided(2000, level, 0.995));
    assert!(passes_one_sided(400, level, 1.0));
    assert!(!needs_one_sided_recheck(400, level, 1.0));
    // A shortfall below the floor fails at either count.
    let mcse = coverage_mcse(2000, level);
    assert!(!passes_one_sided(2000, level, level - 2.5 * mcse));
    assert!(passes_one_sided(2000, level, level - 1.5 * mcse));
    assert!(!passes_one_sided(400, level, level - 3.5 * coverage_mcse(400, level)));
    // A short run under the level by more than the recheck margin is rechecked.
    assert!(needs_one_sided_recheck(400, level, level - 0.025));
    assert!(!needs_one_sided_recheck(2000, level, level - 0.025));
    assert_eq!(CoverageSide::Upper.as_str(), "upper");
    assert_eq!(CoverageSide::Lower.as_str(), "lower");
}

/// A record tally of `covered` out of `n` synthetic replicates of the
/// positive-box construction, for the role's own acceptance tests.
fn synthetic_one_sided_tally(test: &'static str, n: u32, covered: u32) -> CoverageTally {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "z_joint_surrogate_counts", interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for rep in 0..n {
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: 400,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        let truth = if rep < covered { 0.5 } else { 2.0 };
        tally.record(Some((0.0, 1.0)), truth);
    }
    tally
}

/// The conservative interval's over-coverage (1000 of 1000) passes the
/// one-sided assertion, where the two-sided gate's ceiling would refuse it.
#[test]
fn assert_one_sided_accepts_the_over_coverage_of_a_conservative_interval() {
    let tally = synthetic_one_sided_tally("synthetic_one_sided_over_coverage", 1000, 1000);
    assert!(!passes_precision(tally.attempts(), REPORTED_LEVEL, tally.rate()));
    tally.assert_one_sided(CoverageSide::Upper, UPPER_TARGET);
}

/// A shortfall (900 of 1000, below the precision floor) fails the one-sided
/// assertion.
#[test]
#[should_panic(expected = "one-sided (upper) coverage=0.900")]
fn assert_one_sided_refuses_a_shortfall() {
    synthetic_one_sided_tally("synthetic_one_sided_shortfall", 1000, 900)
        .assert_one_sided(CoverageSide::Upper, UPPER_TARGET);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn z_joint_sensitivity_zero_box() {
    let n = grid_n(400);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "z_joint_sensitivity_zero_box",
            dgp: "z_joint_surrogate_counts",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    measure(&mut tally, 0.0, n);
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn z_joint_sensitivity_positive_box() {
    let n = grid_n(400);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "z_joint_sensitivity_positive_box",
            dgp: "z_joint_surrogate_counts",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    measure(&mut tally, 0.05, n);
    tally.assert_one_sided(CoverageSide::Upper, UPPER_TARGET);
}
