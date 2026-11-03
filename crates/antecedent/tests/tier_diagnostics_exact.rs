//! Tier diagnostics on exact designs with hand-computed values.
//!
//! The CoDetermined design is noise-free (`y = 2 t + z - 0.5 u`) on a cell-count design, so
//! AIPW's per-arm outcome regressions are exact and the reported effect is the closure
//! functional 2 to machine precision. The expected point E-value is then derived entirely by
//! hand from that exactly known effect and the outcome SD of the design, with no value taken
//! from the code that attaches it.
//!
//! Design (360 rows). `z, u` in `{-1, 0, 1}`; each `(z, u)` cell holds 20 controls and
//! `10 (z + 2)` treated rows. Let `w = 2 t + z`; in each `u` slice (120 rows) the control
//! values are `-1, 0, 1` (20 rows each) and the treated values `1, 2, 3` (10, 20, 30 rows):
//!
//! * `sum w = -20 + 0 + 20 + 10 + 40 + 90 = 140` and `sum w^2 = 20 + 0 + 20 + 10 + 80 + 270 =
//!   400` per slice, so over three slices `sum w = 420`, mean `7/6`,
//!   `SS_w = 1200 - 420^2 / 360 = 710`.
//! * `u` has `sum u^2 = 240` and is uncorrelated with `w` inside every slice (each slice's mean of
//!   `w` is `7/6`), so `sum (w - 7/6) u = 0`.
//! * `y = w - 0.5 u`: `SS_y = 710 + 0.25 * 240 = 770`, sample variance `770 / 359`.
//!
//! The VanderWeele-Ding point E-value is `RR + sqrt(RR (RR - 1))` with
//! `RR = exp(0.91 * 2 / sqrt(770 / 359))`.
//!
//! The Unknown design has binary `t` and cells `(era, t)` in `{-1, 1} x {0, 1}` weighted 3:1:1:3,
//! each crossed fully with `(e_m, e_y)` in `{-1, 1}^2`; its two scenario effects are exactly 1
//! and -1 (derivation in `tiered_average_effect_closed_form.rs`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, RefuteSuite, Study, TierDesign, tier_diagnostics};
use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::Availability;
use antecedent_graph::{TieredBackground, WithinTier};

fn codetermined_data() -> TabularData {
    let mut rows = Vec::new();
    for u in [-1.0, 0.0, 1.0] {
        for (index, z) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
            for _ in 0..20 {
                rows.push((z, u, 0.0));
            }
            for _ in 0..10 * (index + 1) {
                rows.push((z, u, 1.0));
            }
        }
    }
    let n = rows.len();
    assert_eq!(n, 360);
    // A stride coprime to 360 interleaves the cells so no cross-fit fold is one cell.
    let rows: Vec<(f64, f64, f64)> = (0..n).map(|i| rows[(i * 7) % n]).collect();
    let z: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let u: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let t: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r.2 + r.0 - 0.5 * r.1).collect();
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn unknown_data() -> TabularData {
    let (mut era, mut t, mut m, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..4 {
        for a in [-1.0, 1.0] {
            for t_i in [0.0, 1.0] {
                // Cell weights 3:1 / 1:3 make the treatment depend on era.
                let weight = if (a < 0.0) == (t_i < 0.5) { 3 } else { 1 };
                for _ in 0..weight {
                    for e_m in [-1.0, 1.0] {
                        for e_y in [-1.0, 1.0] {
                            let m_i = t_i + 0.2 * a + 0.5 * e_m;
                            era.push(a);
                            t.push(t_i);
                            m.push(m_i);
                            y.push(-t_i + 2.0 * m_i + 0.2 * a + e_y);
                        }
                    }
                }
            }
        }
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn query(data: &TabularData) -> AverageEffectQuery {
    let schema = data.schema();
    AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap())
}

#[test]
fn the_point_evalue_equals_the_hand_derived_value_of_the_exactly_known_effect() {
    let data = codetermined_data();
    let ctx = ExecutionContext::for_tests(41);
    let background = TieredBackground::from_named(
        data.schema(),
        &[vec!["z", "u"], vec!["t"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let result = Study::tabular(data.clone())
        .tiered_background(background)
        .unwrap()
        .query(query(&data))
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    // The effect is exactly the closure functional on this noise-free design.
    let effect = result.estimate().as_effect().unwrap().ate;
    assert!((effect - 2.0).abs() < 1e-9, "aipw effect {effect}");

    let diagnostics = tier_diagnostics(&result, &ctx).unwrap();
    let schema = data.schema();
    let TierDesign::CoDeterminedClosure { adjustment, rows, propensity_scored } =
        &diagnostics.design
    else {
        panic!("a CoDetermined study reports the closure design, got {:?}", diagnostics.design);
    };
    assert_eq!(adjustment, &[schema.id_of("z").unwrap(), schema.id_of("u").unwrap()]);
    assert_eq!(*rows, Some(360));
    assert!(*propensity_scored);
    let Availability::Available(report) = &diagnostics.overlap else {
        panic!("AIPW fits a propensity, so the overlap report is carried");
    };
    assert!(0.0 < report.propensity_min && report.propensity_min < report.propensity_max);
    assert!(report.propensity_max < 1.0);

    // Hand-derived: SS_y = 770 over 360 rows, so sd = sqrt(770 / 359); effect 2.
    let sd = (770.0_f64 / 359.0).sqrt();
    let risk_ratio = (0.91 * 2.0 / sd).exp();
    let expected = risk_ratio + (risk_ratio * (risk_ratio - 1.0)).sqrt();
    let Availability::Available(point) = &diagnostics.evalue.point else {
        panic!("the tier cell attaches a point E-value");
    };
    assert!(
        (point.value - expected).abs() <= 1e-8 * expected,
        "E-value {} vs hand-derived {expected}",
        point.value
    );
    assert_eq!(point.method.as_ref(), "vanderweele_outcome_sd");
    // The hand value itself is a nontrivial number (RR is about 3.47, E-value about 6.4), so
    // a constant 1 or a wrong SD convention cannot pass.
    assert!(expected > 6.0 && expected < 7.0, "hand-derived {expected}");
}

#[test]
fn unknown_scenario_diagnostics_carry_the_exact_scenario_effects_and_adjustment_sets() {
    let data = unknown_data();
    let ctx = ExecutionContext::for_tests(47);
    let background = TieredBackground::from_named(
        data.schema(),
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
    )
    .unwrap();
    let result = Study::tabular(data.clone())
        .tiered_background(background)
        .unwrap()
        .query(query(&data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let diagnostics = tier_diagnostics(&result, &ctx).unwrap();
    let schema = data.schema();
    let (era, m) = (schema.id_of("era").unwrap(), schema.id_of("m").unwrap());
    let TierDesign::UnknownScenarios { scenarios } = &diagnostics.design else {
        panic!("an Unknown study reports its scenarios, got {:?}", diagnostics.design);
    };
    assert_eq!(scenarios.len(), 2);
    // y = t + 0.6 era + e_m + e_y given {era}: total effect -1 + 2 * 1 = 1; given {era, m}: -1.
    assert_eq!(scenarios[0].adjustment, vec![era]);
    assert!((scenarios[0].effect - 1.0).abs() < 1e-9, "total effect {}", scenarios[0].effect);
    assert_eq!(scenarios[1].adjustment, vec![era, m]);
    assert!((scenarios[1].effect + 1.0).abs() < 1e-9, "direct effect {}", scenarios[1].effect);
    assert!(diagnostics.overlap.unavailable().is_some());
    assert!(diagnostics.evalue.point.unavailable().is_some());
}
