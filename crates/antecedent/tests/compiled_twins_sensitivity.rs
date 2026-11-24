//! The compiled mediation validation and the direct `mediation_sensitivity_fit` /
//! `mediation_range` calls read the same working model, so the sensitivity the compiled study
//! reports is the curve the direct range tabulates.
//!
//! The compiled `mediation.sequential_ignorability` report carries the break-even
//! mediator-outcome error correlation (`comparison`), the indirect effect at the end of the
//! probed range `rho = +-0.2` nearer zero (`refuted_ate`) and whether the effect keeps its sign
//! over that range (`passed`). The direct range tabulates the same curve at its grid, which
//! contains both ends, and names the break-even. Both run `mediation_sensitivity_fit` on the
//! table's columns, so the numbers agree bit for bit.
//!
//! The compiled report does not carry the whole curve, only these three readings. The other
//! sensitivity ranges the facade re-exports (E-value, omitted-variable bias, Rosenbaum bounds,
//! the component and posterior ranges, the MSM parameter ranges and the z-transport mechanism
//! ranges) have no compiled refuter yet and are not compared here.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use antecedent::validate::{mediation_range, mediation_sensitivity_fit};
use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_validate::target_native::{RefutationTarget, TargetQuadrant};

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `(t, m, y, count)` cells of a binary treatment, mediator and outcome.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 30),
    (0.0, 0.0, 1.0, 10),
    (0.0, 1.0, 0.0, 5),
    (0.0, 1.0, 1.0, 15),
    (1.0, 0.0, 0.0, 12),
    (1.0, 0.0, 1.0, 8),
    (1.0, 1.0, 0.0, 20),
    (1.0, 1.0, 1.0, 25),
];
/// A table whose mediator carries little of the outcome.
const WEAK: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 22),
    (0.0, 0.0, 1.0, 18),
    (0.0, 1.0, 0.0, 14),
    (0.0, 1.0, 1.0, 16),
    (1.0, 0.0, 0.0, 17),
    (1.0, 0.0, 1.0, 23),
    (1.0, 1.0, 0.0, 15),
    (1.0, 1.0, 1.0, 15),
];

fn columns(cells: &[(f64, f64, f64, usize)]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for &(tv, mv, yv, count) in cells {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    (t, m, y)
}

fn latent_free() -> Admg {
    let mut admg = Admg::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    admg
}

fn assert_twin(cells: &[(f64, f64, f64, usize)]) {
    let (t, m, y) = columns(cells);
    let summary = mediation_sensitivity_fit(&t, &m, &y, &[]).unwrap();
    let target =
        RefutationTarget::scalar_average_effect(TargetQuadrant::STATIC_FREQUENTIST, [0; 32]);
    let range = mediation_range(target, &summary).unwrap();
    let at = |rho: f64| {
        range.tested.iter().find(|point| point.parameter == rho).expect("a probed rho").target
    };
    let (centre, low, high) = (at(0.0), at(-0.2), at(0.2));
    let sign = centre.signum();
    let worst = if sign * low < sign * high { low } else { high };
    let survives = centre.abs() > 0.0 && sign * low > 0.0 && sign * high > 0.0;

    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let ctx = ExecutionContext::for_tests(31);
    let result = Study::tabular(data)
        .graph(latent_free())
        .query(CausalQuery::Mediation(MediationQuery::binary(
            var(0),
            var(2),
            [var(1)],
            MediationContrast::NaturalIndirect,
        )))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let report = result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == "mediation.sequential_ignorability")
        .expect("the compiled sensitivity report");

    assert_eq!(range.break_even.map(f64::to_bits), Some(summary.rho_tilde.to_bits()));
    assert_eq!(report.comparison.to_bits(), summary.rho_tilde.to_bits(), "break-even rho");
    assert_eq!(report.refuted_ate.to_bits(), worst.to_bits(), "indirect effect at the range end");
    assert_eq!(report.passed, survives);
    assert!(report.informative);
}

#[test]
fn compiled_mediation_sensitivity_equals_the_direct_range_on_the_mixture_table() {
    assert_twin(&CELLS);
}

#[test]
fn compiled_mediation_sensitivity_equals_the_direct_range_when_the_mediator_is_weak() {
    assert_twin(&WEAK);
}
