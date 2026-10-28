//! Influence-function standard errors against exact rational references.
//!
//! `oracle_data/influence_se_reference.py` (frozen in the JSON beside it) writes the variance of
//! a mean of influence values for every declared dependence from its definition in `Fraction`
//! arithmetic: the homoskedastic, HC0 and HC1 sums of squares, the cluster form with the
//! `G / (G - 1)` factor, the Cameron-Gelbach-Miller inclusion-exclusion for two and three
//! dimensions, the Bartlett Newey-West sum over consecutive indices and over calendar labels
//! (with a repeated label and gaps beyond the bandwidth), and the within-cluster panel HAC with
//! its lag-zero cluster convention. The product dispatcher must agree, refuse HC2 and HC3, and
//! refuse a materially negative two-way variance rather than report it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::AnalyticSeKind;
use antecedent_estimate::se::influence_se_under;
use serde_json::Value;

fn ints<T: TryFrom<i64>>(v: &Value) -> Vec<T> {
    v.as_array().unwrap().iter().map(|x| T::try_from(x.as_i64().unwrap()).ok().unwrap()).collect()
}

fn spec(kind: AnalyticSeKind, data: &Value) -> antecedent_estimate::InferenceSpec {
    let mut spec = antecedent_estimate::InferenceSpec::new(0);
    spec.se_kind = kind;
    match kind {
        AnalyticSeKind::Cluster => spec.cluster_ids = Some(ints(&data["clusters"])),
        AnalyticSeKind::PanelClusterHac { .. } => {
            spec.cluster_ids = Some(ints(&data["clusters"]));
            spec.panel_times = Some(ints(&data["panel_times"]));
        }
        AnalyticSeKind::NeweyWest { .. } => {}
        _ => {}
    }
    spec
}

fn check(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * want.abs(), "{what}: {got} vs {want}");
}

#[test]
fn every_declared_dependence_matches_the_exact_variance() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/influence_se_reference.json")).unwrap();
    let psi: Vec<f64> =
        data["psi"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let want = |key: &str| data["se"][key].as_f64();
    let se = |kind, extra: &dyn Fn(&mut antecedent_estimate::InferenceSpec)| {
        let mut s = spec(kind, &data);
        extra(&mut s);
        influence_se_under(&s, &psi)
    };
    let none = |_: &mut antecedent_estimate::InferenceSpec| {};
    check(
        se(AnalyticSeKind::Homoskedastic, &none).unwrap(),
        want("homoskedastic").unwrap(),
        "homoskedastic",
    );
    check(se(AnalyticSeKind::Hc0, &none).unwrap(), want("hc0").unwrap(), "hc0");
    check(se(AnalyticSeKind::Hc1, &none).unwrap(), want("hc1").unwrap(), "hc1");
    check(se(AnalyticSeKind::Cluster, &none).unwrap(), want("cluster").unwrap(), "cluster");
    for lag in [0_usize, 1, 3, 20] {
        check(
            se(AnalyticSeKind::NeweyWest { lag }, &none).unwrap(),
            want(&format!("newey_west_{lag}")).unwrap(),
            &format!("newey west {lag}"),
        );
    }
    for lag in [0_usize, 2, 5, 30] {
        let with_times = |s: &mut antecedent_estimate::InferenceSpec| {
            s.panel_times = Some(ints(&data["times"]));
        };
        check(
            se(AnalyticSeKind::NeweyWest { lag }, &with_times).unwrap(),
            want(&format!("newey_west_calendar_{lag}")).unwrap(),
            &format!("calendar newey west {lag}"),
        );
    }
    for lag in [0_usize, 1, 2, 9] {
        check(
            se(AnalyticSeKind::PanelClusterHac { lag }, &none).unwrap(),
            want(&format!("panel_hac_{lag}")).unwrap(),
            &format!("panel hac {lag}"),
        );
    }
    let three = |s: &mut antecedent_estimate::InferenceSpec| {
        s.multiway_ids = Some(vec![ints(&data["dim1"]), ints(&data["dim2"]), ints(&data["dim3"])]);
    };
    check(se(AnalyticSeKind::Multiway, &three).unwrap(), want("multiway3").unwrap(), "three-way");
    // The two-way inclusion-exclusion variance of these labels is negative: refused, not reported.
    assert!(want("multiway2").is_none());
    let two = |s: &mut antecedent_estimate::InferenceSpec| {
        s.multiway_ids = Some(vec![ints(&data["clusters"]), ints(&data["dim1"])]);
    };
    assert!(se(AnalyticSeKind::Multiway, &two).is_err());
    for kind in [AnalyticSeKind::Hc2, AnalyticSeKind::Hc3] {
        assert!(
            se(kind, &none).is_err(),
            "{kind:?} has no leverage correction for influence values"
        );
    }
}

#[test]
fn a_dependence_declared_without_its_labels_is_refused_not_ignored() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/influence_se_reference.json")).unwrap();
    let psi: Vec<f64> =
        data["psi"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mut bare = antecedent_estimate::InferenceSpec::new(0);
    for kind in [
        AnalyticSeKind::Cluster,
        AnalyticSeKind::Multiway,
        AnalyticSeKind::PanelClusterHac { lag: 1 },
    ] {
        bare.se_kind = kind;
        assert!(influence_se_under(&bare, &psi).is_err(), "{kind:?}");
    }
}
