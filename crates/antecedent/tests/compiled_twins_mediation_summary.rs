//! The Python mediation summary reads its three components from the compiled temporal
//! mediation route, and they are the components its direct estimator call produced.
//!
//! The summary used to call `TemporalMediationEstimator::effect_surface` on a hand-built
//! front-door estimand: treatment at lag one, mediator and outcome contemporaneous, no
//! adjustment. It now runs `TemporalMediationEffect` over the temporal DAG of that structure
//! (`t(-1) -> m`, `t(-1) -> y`, `m -> y`), whose identification certifies an empty adjustment
//! set, so the route fits the same three mechanism regressions on the same lag-aligned rows.
//! Each test runs both on the same series and compares the total, direct and mediated effects
//! bit for bit. No resampling is involved, so no seed enters either number.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "deterministic fixture draws scale a 53-bit integer to the unit interval"
)]

use std::sync::Arc;

use antecedent::estimate::{CausalExprArena, TemporalMediationEstimator};
use antecedent::prelude::IdentifiedEstimand;
use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::TemporalDag;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Deterministic standard normal draws (splitmix64 and Box-Muller).
struct Normals(u64);

impl Normals {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, w) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos()
    }
}

/// `m_t = 0.8 t_{t-1} + e`, `y_t = 0.25 t_{t-1} + 0.55 m_t + e`, columns `t, m, y`, and, when
/// `extra`, an unrelated column `z` the summary never reads.
fn series(n: usize, seed: u64, extra: bool) -> TimeSeriesData {
    let mut draws = Normals(seed);
    let t: Vec<f64> = (0..n).map(|_| draws.normal()).collect();
    let (mut m, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.2 * draws.normal();
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.1 * draws.normal();
    }
    if extra {
        let z: Vec<f64> = (0..n).map(|_| draws.normal()).collect();
        TimeSeriesData::from_f64_columns(
            [("t", &t[..]), ("m", &m[..]), ("y", &y[..]), ("z", &z[..])],
            1,
        )
        .unwrap()
    } else {
        TimeSeriesData::from_f64_columns([("t", &t[..]), ("m", &m[..]), ("y", &y[..])], 1).unwrap()
    }
}

/// The components the summary computed by its direct estimator call.
fn direct(data: &TimeSeriesData, ctx: &ExecutionContext) -> [f64; 3] {
    let (t, m, y) = (v(0), v(1), v(2));
    let query = MediationQuery::binary(t, y, [m], MediationContrast::Total);
    let mut arena = CausalExprArena::new();
    let functional = arena.frontdoor_ate(t, y, &[m], Value::f64(1.0), Value::f64(0.0));
    let estimand =
        IdentifiedEstimand::frontdoor("temporal_mediation.total", Arc::from([m]), functional);
    let surface =
        TemporalMediationEstimator::new().effect_surface(data, &estimand, &query, ctx).unwrap();
    [surface.total, surface.direct, surface.mediated]
}

/// The components the compiled temporal mediation route publishes, as the summary reads them.
fn compiled(data: &TimeSeriesData, ctx: &ExecutionContext) -> [f64; 3] {
    let (t, m, y) = (v(0), v(1), v(2));
    let mut graph = TemporalDag::empty();
    let t1 = graph.add_lagged(t, Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(m, Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(y, Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(t1, m0), (t1, y0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    let result = Study::series(data.clone())
        .graph(graph)
        .query(CausalQuery::Mediation(MediationQuery::binary(t, y, [m], MediationContrast::Total)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap();
    let decomposition = result.mediation().expect("the route publishes the decomposition");
    [
        decomposition.total.expect("total"),
        decomposition.direct.expect("direct"),
        decomposition.mediated.expect("mediated"),
    ]
}

fn assert_twin(label: &str, data: &TimeSeriesData) -> [f64; 3] {
    let ctx = ExecutionContext::for_tests(1);
    let (left, right) = (direct(data, &ctx), compiled(data, &ctx));
    for (name, (a, b)) in ["total", "direct", "mediated"].iter().zip(left.iter().zip(&right)) {
        assert_eq!(a.to_bits(), b.to_bits(), "{label}: {name} {a} against {b}");
    }
    right
}

#[test]
fn the_summary_components_equal_the_compiled_temporal_mediation_route() {
    let [total, direct, mediated] = assert_twin("t, m, y", &series(2_000, 21, false));
    // The fixture's structural effects (0.25 direct, 0.8 x 0.55 mediated), so the twin compares
    // a material decomposition; the in-sample identity of nested least-squares fits holds.
    assert!((direct - 0.25).abs() < 0.05, "direct {direct}");
    assert!((mediated - 0.44).abs() < 0.05, "mediated {mediated}");
    assert!((total - direct - mediated).abs() < 1e-9, "{total} = {direct} + {mediated}");
    assert_twin("short series", &series(40, 5, false));
}

#[test]
fn a_column_the_summary_never_reads_changes_neither_path() {
    let [total, ..] = assert_twin("t, m, y, z", &series(2_000, 21, true));
    let [without, ..] = assert_twin("t, m, y", &series(2_000, 21, false));
    assert_eq!(total.to_bits(), without.to_bits(), "the unread column moves nothing");
}
