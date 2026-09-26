//! Coverage of the registered surrogate z-transport percentile interval.
//!
//! The cited joint is `P(W)P(X)P(Y|X)` under `do(Z=0)`, with `P(Y=1|X=0)=0.2`.
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::nominal_z_transport_interval;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    SidLimits, ZTransportQuery, ZTransportResult, bind_z_transport_catalog, identify_z_transport,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, unit_uniform,
};

const INTERVAL: &str = "percentile_bootstrap";
const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);
const TRUTH: f64 = 0.2;

fn binary_axis(variable: VariableId) -> DiscreteAxis {
    DiscreteAxis { variable, values: Arc::from([Value::Bool(false), Value::Bool(true)]) }
}

fn probabilities() -> Vec<f64> {
    let mut probabilities = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { 0.25 } else { 0.75 };
                let p_x = if x { 0.35 } else { 0.65 };
                let p_y = if y == x { 0.8 } else { 0.2 };
                probabilities.push(p_w * p_x * p_y);
            }
        }
    }
    probabilities
}

fn diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, []).unwrap()
}

fn query() -> ZTransportQuery {
    ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn catalog() -> EvidenceCatalog {
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
    EvidenceCatalog::try_new(
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
    .unwrap()
}

fn draw_counts(n: usize, seed: u64, probabilities: &[f64]) -> Vec<u64> {
    let mut counts = vec![0u64; probabilities.len()];
    for row in 0..n {
        let mut threshold = unit_uniform(seed.wrapping_add(row as u64).wrapping_mul(0x9E37_79B9));
        let mut chosen = probabilities.len() - 1;
        for (index, probability) in probabilities.iter().enumerate() {
            if threshold <= *probability {
                chosen = index;
                break;
            }
            threshold -= probability;
        }
        counts[chosen] += 1;
    }
    counts
}

fn construction() -> Construction {
    Construction {
        query: "ZTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.z_empirical_plugin".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_interventional_mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

/// The 400-replicate band passes, but the project's 2000-replicate recheck does not
/// (n=200 covered 0.932 against [0.935, 0.965]). The route therefore keeps
/// `estimator_grid_not_measured` and this test stays out of `gate_calibration.sh`.
#[test]
#[ignore = "coverage: the 2000-replicate recheck misses the nominal band"]
fn surrogate_cited_margin_nominal_coverage() {
    let diagram = diagram();
    let query = query();
    let catalog = catalog();
    let ZTransportResult::Identified(derivation) = identify_z_transport(
        &diagram,
        &query,
        SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("registered surrogate");
    };
    let functional = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let probabilities = probabilities();
    let n = grid_n(200);
    let rows = map_replicates(n_sim(), |rep| {
        let counts = draw_counts(n, rep.wrapping_mul(1_000_003), &probabilities);
        let probs = counts.iter().map(|count| *count as f64 / n as f64).collect::<Vec<_>>();
        let law = ExactDiscreteLaw::try_empirical(
            "source",
            RegimeId::from_raw(0),
            [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
            [binary_axis(W), binary_axis(X), binary_axis(Y)],
            probs,
            "snapshot-0",
            LawTolerance::default(),
        )
        .unwrap()
        .with_empirical_counts(counts)
        .unwrap();
        let data = ExactTransportData::try_new([law], 64).unwrap();
        let ctx = ExecutionContext::for_tests(rep);
        nominal_z_transport_interval(
            &functional,
            &data,
            &Assignment::from_pairs([(X, Value::Bool(false))]),
            ExactEvaluationLimits::default(),
            199,
            REPORTED_LEVEL,
            &ctx,
        )
        .ok()
        .and_then(Result::ok)
        .map(|interval| {
            let band = interval.mean_intervals.iter().find(|(variable, _, _)| *variable == Y);
            (interval.replicates_ok, band.map(|(_, lo, hi)| (*lo, *hi)))
        })
    });
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "surrogate_cited_margin_nominal_coverage",
            dgp: "cited_wyx_joint",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
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
                tally.record(band, TRUTH);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}
