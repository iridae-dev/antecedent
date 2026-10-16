//! Complete-procedure coverage of the discovered-PAG transport contrast interval.
//!
//! Every replicate runs the whole procedure on fresh data: FCI on the source rows, the
//! enumeration of the discovered PAG's members, the identification of the effect in every
//! member, the estimate over the source and target rows, and the percentile interval on the
//! contrast of a point answer. A replicate whose procedure refuses (the class is not identified
//! in every member, or the answer is a structural range) is scored as a non-report, so the
//! coverage is of the whole procedure including its discovery and the rate at which it reports.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build). The interval stays
//! unbound to any record until this design is measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::analysis::discovered_transport::discover_and_transport;
use antecedent::analysis::discovered_transport_estimate::{
    AnswerKind, TransportEvidence, estimate_discovered_transport,
};
use antecedent::analysis::discovered_transport_temporal::{
    TemporalHorizon, estimate_discovered_temporal_transport,
    transport_discovered_temporal_structure,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, TemporalAlgorithm,
    TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed, unit_uniform,
};

const INTERVAL: &str = "percentile_bootstrap";
const REPLICATES: u32 = 200;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn bern(seed: u64, p: f64) -> f64 {
    f64::from(unit_uniform(seed) < p)
}

/// `P(y = 1 | x, z, L)` of the structural model `x -> y <- z <- {c, d}`, `y -> r`, `z -> q`,
/// with a latent `L -> w` and `L -> y`.
fn p_y(x: f64, z: f64, l: f64) -> f64 {
    0.1 + 0.3 * x + 0.2 * z + 0.2 * l + 0.15 * x * z
}

/// `P(z = 1 | c, d)`: the mechanism that differs between the populations.
fn p_z(c: f64, d: f64, target: bool) -> f64 {
    if target { 0.85 - 0.3 * c - 0.3 * d } else { 0.15 + 0.3 * c + 0.3 * d }
}

fn study(rows: usize, target: bool, seed: u64) -> TabularData {
    let mut columns: Vec<Vec<f64>> = (0..8).map(|_| Vec::with_capacity(rows)).collect();
    for i in 0..rows as u64 {
        let draw = |k: u64| stream_seed(seed, i * 16 + k);
        let (c, d, l, x) =
            (bern(draw(0), 0.5), bern(draw(1), 0.5), bern(draw(2), 0.5), bern(draw(3), 0.5));
        let z = bern(draw(4), p_z(c, d, target));
        let w = bern(draw(5), 0.2 + 0.6 * l);
        let y = bern(draw(6), p_y(x, z, l));
        let q = bern(draw(7), 0.1 + 0.7 * z);
        let r = bern(draw(8), 0.2 + 0.6 * y);
        for (column, value) in columns.iter_mut().zip([x, y, z, c, d, w, q, r]) {
            column.push(value);
        }
    }
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

/// `E*[y | do(x = a)]` by enumerating the target population's structural model.
fn truth(a: f64) -> f64 {
    let mut total = 0.0;
    for c in [0.0, 1.0] {
        for d in [0.0, 1.0] {
            for l in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    let pz = p_z(c, d, true);
                    total += 0.125 * if z > 0.5 { pz } else { 1.0 - pz } * p_y(a, z, l);
                }
            }
        }
    }
    total
}

fn construction() -> Construction {
    Construction {
        query: "DiscoveredTransport".into(),
        graph_class: "Pag".into(),
        structure: "discovered".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.empirical_table_plugin".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_interventional_contrast".into(),
        identification: "discovered_class".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn query() -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "clinic",
        [v(0)],
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovered_pag_transport_contrast_complete_procedure_coverage() {
    let n = grid_n(4000);
    let effect = truth(1.0) - truth(0.0);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "discovered_pag_transport_contrast_complete_procedure_coverage",
            dgp: "latent_confounded_selection_on_z",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    for (rep, outcome) in map_replicates(n_sim(), |rep| {
        let seed = rep.wrapping_mul(1_000_003);
        let source = study(n, false, seed);
        let target = study(n, true, seed ^ 0x5eed);
        let ctx = ExecutionContext::for_tests(rep);
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            (0..8).map(VariableId::from_raw).collect(),
            CiTestSpec::named("gsquared"),
            rep,
        );
        let interval = discover_and_transport(&contract, &source, &[v(2)], &query(), 256, &ctx)
            .ok()
            .and_then(|run| run.transport.ok())
            .and_then(|report| {
                let mut evidence = TransportEvidence::new(&source, &target);
                evidence.bootstrap_replicates = REPLICATES;
                estimate_discovered_transport(&report, &query(), &evidence, &ctx).ok()
            })
            .filter(|estimate| estimate.answer().kind == AnswerKind::Point)
            .and_then(|estimate| estimate.answer().interval.map(|i| (i.low, i.high)));
        (rep, interval)
    }) {
        let _ = rep;
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(outcome, effect);
    }
    tally.assert();
}

fn p_y_lagged(x: f64, z: f64) -> f64 {
    0.1 + 0.4 * x + 0.3 * z + 0.1 * x * z
}

/// A binary series: `z` autoregressive with a population-specific mechanism, `x` randomized,
/// `y_t` caused by `x_{t-1}` and `z_{t-1}`.
fn series(steps: usize, target: bool, seed: u64) -> (TimeSeriesData, TabularData) {
    let (mut xs, mut ys, mut zs) = (vec![0.0], vec![0.0], vec![0.0]);
    for t in 1..steps {
        let draw = |k: u64| stream_seed(seed, t as u64 * 4 + k);
        let pz = if target { 0.6 - 0.4 * zs[t - 1] } else { 0.2 + 0.5 * zs[t - 1] };
        zs.push(bern(draw(0), pz));
        xs.push(bern(draw(1), 0.5));
        ys.push(bern(draw(2), p_y_lagged(xs[t - 1], zs[t - 1])));
    }
    let columns = [("v0", xs.as_slice()), ("v1", ys.as_slice()), ("v2", zs.as_slice())];
    (
        TimeSeriesData::from_f64_columns(columns, 1_000_000_000).unwrap(),
        TabularData::from_f64_columns(columns).unwrap(),
    )
}

fn temporal_construction() -> Construction {
    Construction {
        graph_class: "TemporalCpdag".into(),
        dependence: "block".into(),
        interval_method: "block_percentile_bootstrap".into(),
        ..construction()
    }
}

/// Complete-procedure coverage of the temporal contrast interval: PCMCI+ on the source series,
/// the temporal class at horizon one, identification in every member, the estimate over the
/// windowed tables and the circular block bootstrap interval, in every replicate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovered_temporal_transport_contrast_complete_procedure_coverage() {
    let steps = grid_n(4000);
    let pi = 0.6 / 1.4;
    let truth = |a: f64| (1.0 - pi) * p_y_lagged(a, 0.0) + pi * p_y_lagged(a, 1.0);
    let effect = truth(1.0) - truth(0.0);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "discovered_temporal_transport_contrast_complete_procedure_coverage",
            dgp: "lagged_selection_on_autoregressive_z",
            interval: "block_percentile_bootstrap",
        },
        REPORTED_LEVEL,
    );
    for outcome in map_replicates(n_sim(), |rep| {
        let seed = rep.wrapping_mul(1_000_003);
        let (source, source_table) = series(steps, false, seed);
        let (_, target_table) = series(steps, true, seed ^ 0x5eed);
        let ctx = ExecutionContext::for_tests(rep);
        let contract = TemporalDiscoveryContract::new(
            TemporalAlgorithm::PcmciPlus,
            (0..3).map(VariableId::from_raw).collect(),
            CiTestSpec::named("gsquared"),
            rep,
        );
        let mut adapter = TemporalSelectiveDiscovery::new();
        adapter.run(&contract, &source, &ctx).ok()?;
        let report = transport_discovered_temporal_structure(
            adapter.outcome()?,
            TemporalHorizon { horizon_steps: 1, history: None },
            &[v(2)],
            &query(),
            256,
        )
        .ok()?;
        let mut evidence = TransportEvidence::new(&source_table, &target_table);
        evidence.bootstrap_replicates = REPLICATES;
        let estimate =
            estimate_discovered_temporal_transport(&report, &query(), &evidence, &ctx).ok()?;
        (estimate.answer().kind == AnswerKind::Point)
            .then(|| estimate.answer().interval.map(|i| (i.low, i.high)))
            .flatten()
    }) {
        tally.bind(
            &temporal_construction(),
            ScopeFacts {
                row_count: steps as u64,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(outcome, effect);
    }
    tally.assert();
}
