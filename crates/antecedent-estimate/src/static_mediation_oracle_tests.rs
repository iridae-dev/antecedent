//! Cross-world structural-evaluation reference for the static linear mediation estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, ExecutionContext, Intervention, MediationContrast, MediationQuery, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

use super::estimate_static_mediation;

/// Two mediators in series and in parallel, an exogenous mediator-outcome confounder and
/// a treatment contrast of two, on exactly recoverable mechanisms: the disturbances are
/// distinct whole harmonics over one period, so each is orthogonal in sample to every
/// regressor of its own equation and least squares returns the structural coefficients to
/// rounding. Node ids are deliberately not in topological order. The reference evaluates
/// the structural equations across worlds, `Y(a1, M(a0)) - Y(a0, M(a0))` and
/// `Y(a1, M(a1)) - Y(a0, M(a0))`, rather than summing path products.
#[test]
fn serial_and_parallel_mediators_match_cross_world_structural_evaluation() {
    const N: usize = 1024;
    // ids: 0 = y, 1 = m2, 2 = z, 3 = m1, 4 = x
    let harmonic = |k: f64, i: usize, cosine: bool| {
        let angle = std::f64::consts::TAU * k * i as f64 / N as f64;
        if cosine { angle.cos() } else { angle.sin() }
    };
    let (mut y, mut m2, mut z, mut m1, mut x) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..N {
        let xi = 1.0 + harmonic(1.0, i, true);
        let zi = 0.5 + harmonic(2.0, i, false);
        let m1i = 0.25 + 2.0 * xi + harmonic(3.0, i, true);
        let m2i = -0.5 + 0.7 * xi - 1.5 * m1i + 0.8 * zi + harmonic(5.0, i, false);
        let yi = 2.0 + 0.4 * xi + 0.9 * m1i - 0.6 * m2i + 1.3 * zi + harmonic(7.0, i, true);
        x.push(xi);
        z.push(zi);
        m1.push(m1i);
        m2.push(m2i);
        y.push(yi);
    }
    let data = TabularData::from_f64_columns([
        ("y", &y[..]),
        ("m2", &m2[..]),
        ("z", &z[..]),
        ("m1", &m1[..]),
        ("x", &x[..]),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(5);
    let [y_id, m2_id, z_id, m1_id, x_id] = [0, 1, 2, 3, 4].map(DenseNodeId::from_raw);
    for (from, to) in [
        (x_id, m1_id),
        (x_id, m2_id),
        (x_id, y_id),
        (m1_id, m2_id),
        (m1_id, y_id),
        (m2_id, y_id),
        (z_id, m2_id),
        (z_id, y_id),
    ] {
        dag.insert_directed(from, to).unwrap();
    }
    // Structural equations without disturbances, at z = 0.5.
    let m1_of = |x: f64| 0.25 + 2.0 * x;
    let m2_of = |x: f64, m1: f64| -0.5 + 0.7 * x - 1.5 * m1 + 0.8 * 0.5;
    let y_of = |x: f64, m1: f64, m2: f64| 2.0 + 0.4 * x + 0.9 * m1 - 0.6 * m2 + 1.3 * 0.5;
    let (a0, a1) = (1.0, 3.0);
    let natural = |x: f64| {
        let m1 = m1_of(x);
        y_of(x, m1, m2_of(x, m1))
    };
    let total = natural(a1) - natural(a0);
    let m1_0 = m1_of(a0);
    let m2_0 = m2_of(a0, m1_0);
    let direct = y_of(a1, m1_0, m2_0) - y_of(a0, m1_0, m2_0);
    assert!((total - 7.16).abs() < 1e-12 && (direct - 0.8).abs() < 1e-12, "{total} {direct}");
    let ctx = ExecutionContext::for_tests(3);
    for (contrast, want) in [
        (MediationContrast::Total, total),
        (MediationContrast::Direct, direct),
        (MediationContrast::NaturalDirect, direct),
        (MediationContrast::NaturalIndirect, total - direct),
        (MediationContrast::Mediated, total - direct),
    ] {
        let mut q = MediationQuery::binary(
            VariableId::from_raw(4),
            VariableId::from_raw(0),
            [VariableId::from_raw(3), VariableId::from_raw(1)],
            contrast,
        );
        q.control = Intervention::set(q.treatment, Value::f64(a0));
        q.active = Intervention::set(q.treatment, Value::f64(a1));
        let fit = estimate_static_mediation(&data, &dag, &q, AssumptionSet::new(), 0, &[], &ctx)
            .unwrap();
        assert!((fit.total.unwrap() - total).abs() < 1e-9, "{contrast:?} total");
        assert!((fit.direct.unwrap() - direct).abs() < 1e-9, "{contrast:?} direct");
        assert!(
            (fit.effect.ate - want).abs() < 1e-9,
            "{contrast:?}: {} vs {want}",
            fit.effect.ate
        );
    }
}
