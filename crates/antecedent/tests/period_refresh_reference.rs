//! Autoregressive period refresh against an exact rational replay of the declared periods.
//!
//! `oracle_data/period_refresh_reference.py` (frozen in the JSON beside it) forms the lag rows of
//! every period, solves the normal equations of the retained rows and the conjugate
//! normal-inverse-gamma posterior under the stated prior with Fractions, and replays the three
//! outcomes the refresh can reach: a contiguous extension (rows that reach back into the carried
//! tail are formed), an extension over a gap (they are not), and a restart on the new period
//! alone. Cases cover an offset of 1e6, three periods, periods shorter than the lag order, a
//! nearly collinear ramp with a weak prior and a strong prior against few rows. The test also
//! pins the row identities (the absolute time index of each target), the refusal of an overlapping
//! or snapshot-reusing period and of a restart that leaves no row, and that a refused period
//! leaves the state unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_state::ConjugateOlsPrior;
use serde_json::Value;

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn window(start: i64, end: i64, snapshot: &str) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "ar_mechanism".into(),
        horizon: 2,
        lag_alignment: vec![("t".into(), 1)],
        intervention_history: vec!["a1=1".into()],
        selection_targets: set(&["s_t1"]),
        regimes: set(&["observational"]),
        period: ObservationPeriod { start, end },
        unit_ids: set(&["u1", "u2"]),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn cadence() -> ObservationCadence {
    ObservationCadence::new(1).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

/// The prior of the reference: diagonal precision `(1/2 + i/5) * scale`, shape 3, rate 2.
fn prior(p: usize, scale: f64) -> ConjugateOlsPrior {
    let mut precision = vec![0.0; p * p];
    for i in 0..p {
        precision[i * p + i] = (0.5 + i as f64 / 5.0) * scale;
    }
    ConjugateOlsPrior::new(vec![0.0; p], precision, 3.0, 2.0).unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1.0),
        "{what}: got {got:.17e} want {want:.17e}"
    );
}

fn check(refresh: &PeriodRefresh, step: &Value, rel: f64, what: &str) {
    let want_n = step["n"].as_u64().unwrap();
    let snapshot = refresh.fit().snapshot();
    assert_eq!(snapshot.n, want_n, "{what}: rows");
    let ids: Vec<u64> =
        step["row_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let got_ids: Vec<u64> = refresh.fit().rows().iter().map(|r| r.id).collect();
    assert_eq!(got_ids, ids, "{what}: row identities are the absolute time indices of the targets");
    for (k, (g, w)) in snapshot.beta.as_ref().unwrap().iter().zip(nums(&step["beta"])).enumerate() {
        close(*g, w, rel, &format!("{what}: least squares {k}"));
    }
    let posterior = refresh.summary().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(posterior.n, want_n, "{what}: posterior rows");
    for (k, (g, w)) in posterior.mean.iter().zip(nums(&step["mean"])).enumerate() {
        close(*g, w, rel, &format!("{what}: posterior mean {k}"));
    }
    close(posterior.shape, num(&step["shape"]), 1e-12, &format!("{what}: shape"));
    // The rate is a difference of two sums near 1e8 at the ramp's offset, evaluated from a mean
    // that itself carries the conditioning of the raw Gram; it is held to a percent there.
    let rate_rel = if what.starts_with("near_collinear_ramp") { 5e-2 } else { rel };
    close(posterior.rate, num(&step["rate"]), rate_rel, &format!("{what}: rate"));
    refresh.check_against_fresh().expect("cached results equal a fresh reconstruction");
}

#[test]
fn every_period_outcome_matches_the_exact_replay() {
    let frozen: Value =
        serde_json::from_str(include_str!("oracle_data/period_refresh_reference.json")).unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let lags = usize::try_from(case["lags"].as_u64().unwrap()).unwrap();
        let standing = match case["standing"].as_str().unwrap() {
            "declared" => MechanismStanding::DeclaredInvariant,
            _ => MechanismStanding::Undeclared,
        };
        // The nearly collinear ramp is solved to a looser relative tolerance: the Gram of
        // [1, v, v'] at an offset of 2000 has a condition number near 1e12.
        let rel = if name == "near_collinear_ramp" || name == "offset_million_may_refuse" {
            1e-3
        } else if name == "offset_three_contiguous" {
            1e-4
        } else {
            1e-7
        };
        let may_refuse = case["may_refuse"].as_bool().unwrap_or(false);
        let p = lags + 1;
        let scale = num(&case["prior_scale"]);
        let prior = prior(p, scale);
        let spec = PeriodRefreshSpec {
            series_id: "t",
            cadence: cadence(),
            initial_law: "law_1",
            standing,
            lags,
        };
        let mut refresh: Option<PeriodRefresh> = None;
        let mut end = 0_i64;
        let mut refused = 0;
        for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let start = step["start"].as_i64().unwrap();
            let values = nums(&step["values"]);
            end = start + i64::try_from(values.len()).unwrap();
            let snap = format!("snap_{index}");
            let win = window(start, end, &snap);
            let what = format!("{name} step {index}");
            if index == 0 {
                refresh = Some(PeriodRefresh::start(&spec, &win, &prior, &values).unwrap());
            } else {
                let outcome =
                    refresh.as_mut().unwrap().refresh(&win, cadence(), "law_1", &values).unwrap();
                match (step["outcome"].as_str().unwrap(), &outcome) {
                    (
                        "extended_contiguous",
                        PeriodRefreshOutcome::Extended { contiguous: true },
                    )
                    | ("extended_gap", PeriodRefreshOutcome::Extended { contiguous: false })
                    | ("restarted", PeriodRefreshOutcome::Restarted { .. }) => {}
                    other => panic!("{what}: outcome {other:?}"),
                }
            }
            if may_refuse && refresh.as_ref().unwrap().summary().is_err() {
                // Fail closed: a posterior that cannot be resolved is refused, never answered.
                refused += 1;
                continue;
            }
            check(refresh.as_ref().unwrap(), step, rel, &what);
        }
        assert_eq!(refresh.unwrap().window().period.end, end);
        assert!(may_refuse || refused == 0, "{name}: refused a posterior it should resolve");
    }
}

#[test]
fn a_refused_period_leaves_the_state_unchanged() {
    let series: Vec<f64> = (0..12).map(|i| 3.0 + f64::from(i % 5) * 0.25).collect();
    let spec = PeriodRefreshSpec {
        series_id: "t",
        cadence: cadence(),
        initial_law: "law_1",
        standing: MechanismStanding::DeclaredInvariant,
        lags: 1,
    };
    let prior = prior(2, 1.0);
    let mut refresh =
        PeriodRefresh::start(&spec, &window(0, 12, "snap_a"), &prior, &series).unwrap();
    let before_rows = refresh.fit().rows().to_vec();
    let before = refresh.summary().unwrap();
    let events = refresh.events().len();
    // Overlap, a reused snapshot and a wrong-length series are refused.
    for (win, values) in [
        (window(10, 15, "snap_b"), vec![1.0; 5]),
        (window(12, 15, "snap_a"), vec![1.0; 3]),
        (window(12, 15, "snap_c"), vec![1.0; 2]),
    ] {
        assert!(refresh.refresh(&win, cadence(), "law_1", &values).is_err());
        assert_eq!(refresh.events().len(), events);
        assert_eq!(refresh.fit().rows(), before_rows.as_slice());
        let after = refresh.summary().unwrap();
        assert_eq!(after.mean, before.mean);
        assert_eq!(after.rate.to_bits(), before.rate.to_bits());
    }
    // A restart whose new period leaves no row is refused too, and keeps the old fit.
    let spec = PeriodRefreshSpec { standing: MechanismStanding::Undeclared, ..spec };
    let mut restarting =
        PeriodRefresh::start(&spec, &window(0, 12, "snap_a"), &prior, &series).unwrap();
    assert!(restarting.refresh(&window(12, 13, "snap_b"), cadence(), "law_1", &[1.0]).is_err());
    assert_eq!(restarting.fit().rows(), before_rows.as_slice());
    assert_eq!(restarting.events().len(), 1);
}
