//! The direct design-based interference estimator and the compiled `InterferenceDag` route are
//! one computation under complete randomization and under Bernoulli assignment, for every
//! built-in exposure mapping.
//!
//! `estimate_interference_with_context` and the prepared study are run on the same ring network
//! and compared bit for bit: Horvitz-Thompson and Hajek contrasts, the conservative variance and
//! its standard error, and the smallest exposure probability. The route calls the same kernel with
//! the same inputs. The one difference is the Monte Carlo exposure-probability seed of a design
//! too large to enumerate: the direct entry takes the caller's seed and the route derives it from
//! the execution context, so the direct call here is given the derived seed.
//!
//! Every design runs the estimator `interference.ht_hajek`, but its coverage records were
//! measured under Bernoulli assignment with `NeighborCount` exposure only. Each other assignment
//! x mapping pair carries its own calibration key (functional label
//! `interference.<assignment>.<mapping>`) and reports coverage not measured, while the measured
//! Bernoulli/`NeighborCount` design keeps the key its records carry.
#![allow(
    deprecated,
    clippy::cast_precision_loss,
    reason = "integration test calls the direct entry it compares and builds fixtures from small counts"
)]
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::estimate::estimate_interference_with_context;
use antecedent::{InterferenceSpec, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    InterferenceFunctional, InterferenceQuery, StreamDomain, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;
use antecedent_io::calibration::{
    CONSTRUCTION_NOT_MEASURED, CalibrationBasisWire, CalibrationKeyWire, calibration_slot,
};

fn same(direct: f64, compiled: f64, label: &str) {
    assert_eq!(direct.to_bits(), compiled.to_bits(), "{label}: direct {direct} vs {compiled}");
}

/// A ring of `n` units, each reading both neighbours with unequal weights.
fn ring(n: usize) -> Vec<NetworkEdge> {
    (0..n)
        .flat_map(|unit| {
            [(unit + n - 1) % n, (unit + 1) % n].into_iter().map(move |from| {
                NetworkEdge::new(
                    from as u64,
                    unit as u64,
                    1.0 + 0.5 * ((from + 2 * unit) % 3) as f64,
                )
            })
        })
        .collect()
}

fn outcomes(assignment: &[bool]) -> Vec<f64> {
    let n = assignment.len();
    (0..n)
        .map(|unit| {
            let neighbors = [(unit + n - 1) % n, (unit + 1) % n];
            let treated_neighbors = neighbors.iter().filter(|&&other| assignment[other]);
            1.0 + 0.25 * unit as f64
                + 2.0 * f64::from(u8::from(assignment[unit]))
                + 1.5 * treated_neighbors.count() as f64
        })
        .collect()
}

fn compiled(
    data: &TabularData,
    query: &InterferenceQuery,
    edges: Vec<NetworkEdge>,
    assignment: &[bool],
    ctx: &ExecutionContext,
) -> StudyResult {
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query.clone()))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
        .estimate(data, ctx)
        .unwrap()
}

fn assert_twin(
    label: &str,
    design: AssignmentDesign,
    exposure: ExposureMapping,
    from: (f64, f64),
    to: (f64, f64),
    assignment: &[bool],
) {
    let data = TabularData::from_f64_columns([("y", outcomes(assignment).as_slice())]).unwrap();
    let edges = ring(assignment.len());
    let query = InterferenceQuery::new(
        design,
        exposure,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: from.0, neighbors: from.1 },
            to: ExposureLevel { own: to.0, neighbors: to.1 },
        },
    );
    let ctx = ExecutionContext::for_tests(241);
    let result = compiled(&data, &query, edges.clone(), assignment, &ctx);
    // The route draws the exposure-probability seed from the execution context's stream.
    let seed = ctx.rng.stream_for(StreamDomain::Transport, 0x1F7E).next_u64();
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let direct =
        estimate_interference_with_context(&query, &network, assignment, seed, &ctx).unwrap();
    let route = result.interference().expect("an interference estimate");
    same(direct.contrast.horvitz_thompson, route.contrast.horvitz_thompson, &format!("{label} ht"));
    same(direct.contrast.hajek, route.contrast.hajek, &format!("{label} hajek"));
    same(
        direct.contrast.conservative_variance,
        route.contrast.conservative_variance,
        &format!("{label} variance"),
    );
    same(
        direct.minimum_exposure_probability,
        route.minimum_exposure_probability,
        &format!("{label} minimum probability"),
    );
    assert_eq!(direct.from_probability_method, route.from_probability_method, "{label}");
    assert_eq!(direct.to_probability_method, route.to_probability_method, "{label}");
    let estimate = result.estimate().as_effect().expect("an effect estimate");
    same(direct.contrast.horvitz_thompson, estimate.ate, &format!("{label} published point"));
    same(
        direct.contrast.conservative_variance.sqrt(),
        estimate.se_analytic(),
        &format!("{label} published standard error"),
    );
}

/// Treated units 0, 1, 2 and 5 of a ring of eight.
const EIGHT: [bool; 8] = [true, true, true, false, false, true, false, false];

#[test]
fn complete_randomization_route_equals_the_direct_estimator_under_every_mapping() {
    let complete = || AssignmentDesign::CompleteRandomization { treated: 4 };
    assert_twin(
        "complete own",
        complete(),
        ExposureMapping::OwnTreatment,
        (0.0, 0.0),
        (1.0, 0.0),
        &EIGHT,
    );
    assert_twin(
        "complete count",
        complete(),
        ExposureMapping::NeighborCount,
        (0.0, 1.0),
        (1.0, 1.0),
        &EIGHT,
    );
    assert_twin(
        "complete fraction",
        complete(),
        ExposureMapping::NeighborFraction,
        (0.0, 0.5),
        (1.0, 0.5),
        &EIGHT,
    );
    // Weighted exposures between zero and one are unit-specific, so the contrast compares a
    // treated unit with no treated neighbour against one with both.
    assert_twin(
        "complete weighted",
        complete(),
        ExposureMapping::WeightedNeighborExposure,
        (1.0, 0.0),
        (1.0, 1.0),
        &EIGHT,
    );
}

#[test]
fn bernoulli_route_equals_the_direct_estimator_under_the_other_mappings() {
    let bernoulli = || AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) };
    assert_twin(
        "bernoulli own",
        bernoulli(),
        ExposureMapping::OwnTreatment,
        (0.0, 0.0),
        (1.0, 0.0),
        &EIGHT,
    );
    assert_twin(
        "bernoulli fraction",
        bernoulli(),
        ExposureMapping::NeighborFraction,
        (0.0, 0.5),
        (1.0, 0.5),
        &EIGHT,
    );
    assert_twin(
        "bernoulli weighted",
        bernoulli(),
        ExposureMapping::WeightedNeighborExposure,
        (1.0, 0.0),
        (1.0, 1.0),
        &EIGHT,
    );
}

#[test]
fn monte_carlo_exposure_probabilities_agree_under_the_derived_seed() {
    // Twenty-four units exceed the exact-enumeration bound, so both paths draw Monte Carlo
    // exposure probabilities from the seed the route derives.
    let assignment = (0..24).map(|unit| unit % 3 == 0 || unit % 5 == 0).collect::<Vec<_>>();
    let treated = assignment.iter().filter(|&&on| on).count();
    assert_twin(
        "complete fraction monte carlo",
        AssignmentDesign::CompleteRandomization { treated },
        ExposureMapping::NeighborFraction,
        (0.0, 0.5),
        (1.0, 0.5),
        &assignment,
    );
}

/// A ring of `n` units, each reading both neighbours with unit weight: the network the
/// Bernoulli/`NeighborCount` coverage records were measured on.
fn unit_ring(n: usize) -> Vec<NetworkEdge> {
    (0..n)
        .flat_map(|unit| {
            [(unit + n - 1) % n, (unit + 1) % n]
                .into_iter()
                .map(move |from| NetworkEdge::new(from as u64, unit as u64, 1.0))
        })
        .collect()
}

/// The calibration basis of the primary interval of a one-shot run, bound to the inspected
/// contract as the coverage tests bind it.
fn primary_basis(
    design: AssignmentDesign,
    exposure: ExposureMapping,
    from: (f64, f64),
    to: (f64, f64),
    assignment: &[bool],
) -> CalibrationBasisWire {
    let data = TabularData::from_f64_columns([("y", outcomes(assignment).as_slice())]).unwrap();
    let network = NetworkData::try_new(data.clone(), unit_ring(assignment.len())).unwrap();
    let query = InterferenceQuery::new(
        design,
        exposure,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: from.0, neighbors: from.1 },
            to: ExposureLevel { own: to.0, neighbors: to.1 },
        },
    );
    let study = Study::tabular(data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(0x1F7E)).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("interference.ht_hajek"));
    let contract = study.inspect().unwrap();
    result.calibration_bases(&contract).unwrap().remove(0)
}

/// The coverage of `interference.ht_hajek` was measured under Bernoulli assignment with
/// `NeighborCount` exposure on a unit-weight ring of 200 to 800 units. That design still reads its
/// record; complete randomization and the other mappings run the same estimator under their own
/// calibration key, which differs from the measured key only in the design label, and report
/// coverage not measured.
#[test]
fn only_bernoulli_neighbor_count_reads_the_measured_interference_coverage() {
    // Two hundred units, the smallest measured size; every requested exposure level is realized.
    let pattern = [true, true, true, false, false, true, false, false, false, false];
    let assignment = (0..200).map(|unit| pattern[unit % pattern.len()]).collect::<Vec<_>>();
    let bernoulli = || AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) };
    let measured = primary_basis(
        bernoulli(),
        ExposureMapping::NeighborCount,
        (0.0, 0.0),
        (1.0, 0.0),
        &assignment,
    );
    assert_eq!(measured.key.functional, "", "the measured design keeps its key");
    let slot = calibration_slot(&measured);
    assert_ne!(slot.status.as_str(), "not_measured", "{slot:#?}");
    assert!(
        slot.record_id
            .as_deref()
            .is_some_and(|id| id.contains("interference_dag_bernoulli_neighbor_count")),
        "{slot:#?}"
    );

    let treated = assignment.iter().filter(|&&on| on).count();
    let complete = || AssignmentDesign::CompleteRandomization { treated };
    let designs = [
        (
            "interference.complete_randomization.own_treatment",
            complete(),
            ExposureMapping::OwnTreatment,
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        (
            "interference.complete_randomization.neighbor_count",
            complete(),
            ExposureMapping::NeighborCount,
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        (
            "interference.complete_randomization.neighbor_fraction",
            complete(),
            ExposureMapping::NeighborFraction,
            (0.0, 0.5),
            (1.0, 0.5),
        ),
        (
            "interference.complete_randomization.weighted_neighbor_exposure",
            complete(),
            ExposureMapping::WeightedNeighborExposure,
            (1.0, 0.0),
            (1.0, 1.0),
        ),
        (
            "interference.bernoulli.own_treatment",
            bernoulli(),
            ExposureMapping::OwnTreatment,
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        (
            "interference.bernoulli.neighbor_fraction",
            bernoulli(),
            ExposureMapping::NeighborFraction,
            (0.0, 0.5),
            (1.0, 0.5),
        ),
        (
            "interference.bernoulli.weighted_neighbor_exposure",
            bernoulli(),
            ExposureMapping::WeightedNeighborExposure,
            (1.0, 0.0),
            (1.0, 1.0),
        ),
    ];
    for (label, design, exposure, from, to) in designs {
        let basis = primary_basis(design, exposure, from, to, &assignment);
        assert_eq!(basis.key.functional, label);
        let slot = calibration_slot(&basis);
        assert_eq!(slot.status.as_str(), "not_measured", "{label}: {slot:#?}");
        assert_eq!(slot.reason.as_deref(), Some(CONSTRUCTION_NOT_MEASURED), "{label}");
        assert!(slot.record_id.is_none(), "{label}: {slot:#?}");
        // Only the design label separates this key from the measured one.
        let unlabelled = CalibrationKeyWire { functional: String::new(), ..basis.key.clone() };
        assert_eq!(unlabelled, measured.key, "{label}");
    }
}
