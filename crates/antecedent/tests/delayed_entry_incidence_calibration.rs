//! Coverage of the compiled pointwise target-cause incidence interval under delayed entry.
//!
//! Cause-1 masses at times one and two are 0.15 in control and 0.10 in treatment.
//! Both arms have cause-2 mass 0.10 at each event time. The source-population CIF
//! contrast at horizon three is therefore -0.10. Entry is drawn independently of
//! event time and cause. The weighted design adds subject-specific independent
//! censoring at 1.5 and 2.5 with fixed known probabilities at both times.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::{
    CausalQuery, ExecutionContext, KnownCensoringSurvival, ObservationAssumption,
    SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, uniform,
};
use common::calibration_bind::constructions;

const TIMES: [f64; 6] = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
const TRUTH: f64 = -0.10;
const BOOTSTRAP: u32 = 299;

fn trial(fixed_known: bool, seed: u64) -> (TabularData, SurvivalQuery) {
    let minimum_arm = antecedent_io::graphless_support::incidence_min_arm(fixed_known);
    let arm_size = grid_n(2 * minimum_arm);
    let mut rng = uniform(seed, 0xD3E1_C1F0);
    let (mut duration, mut causes, mut treatment, mut entries) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut grid: [Vec<f64>; TIMES.len()] = std::array::from_fn(|_| Vec::new());
    for arm in [false, true] {
        let target = if arm { 0.10 } else { 0.15 };
        let mut observed = 0;
        while observed < arm_size {
            let entry = f64::from(rng() < 0.5);
            let u = rng();
            let (event_time, cause): (f64, f64) = if u < target {
                (1.0, 1.0)
            } else if u < target + 0.10 {
                (1.0, 2.0)
            } else if u < 2.0 * target + 0.10 {
                (2.0, 1.0)
            } else if u < 2.0 * target + 0.20 {
                (2.0, 2.0)
            } else {
                (3.0, 0.0)
            };
            let hazard = if rng() < 0.5 { 0.10 } else { 0.20 };
            let (c1, c2) = (rng(), rng());
            let censor_time = if fixed_known && c1 < hazard {
                1.5
            } else if fixed_known && c2 < hazard {
                2.5
            } else {
                3.0
            };
            let exit = event_time.min(censor_time);
            if entry >= exit {
                continue;
            }
            duration.push(exit);
            causes.push(if event_time > censor_time {
                0.0
            } else {
                cause
            });
            treatment.push(f64::from(arm));
            entries.push(entry);
            let keep = 1.0 - hazard;
            for (values, probability) in
                grid.iter_mut().zip([1.0, 1.0, 1.0, keep, keep, keep * keep])
            {
                values.push(probability);
            }
            observed += 1;
        }
    }
    let columns = ["g_zero", "g_one", "g_first_censor", "g_two", "g_second_censor", "g_tau"];
    let mut data_columns = vec![
        ("duration", duration.as_slice()),
        ("cause", causes.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entries.as_slice()),
    ];
    if fixed_known {
        data_columns.extend(columns.into_iter().zip(grid.iter().map(Vec::as_slice)));
    }
    let data = TabularData::from_f64_columns(data_columns).expect("simulated subjects align");
    let query = SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(1),
        treatment: VariableId::from_raw(2),
        tau: 3.0,
        delayed_entry: Some(VariableId::from_raw(3)),
        known_censoring: fixed_known.then(|| KnownCensoringSurvival {
            times: Arc::from(TIMES),
            columns: (4..10).map(VariableId::from_raw).collect(),
            minimum_probability: 0.01,
        }),
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::CumulativeIncidence { target_cause: 1 },
    };
    (data, query)
}

fn replicate(fixed_known: bool, seed: u64) -> Option<((f64, f64), Construction, ScopeFacts)> {
    let (data, query) = trial(fixed_known, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query))
        .bootstrap_replicates(BOOTSTRAP)
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let curve = result.survival()?;
    let [lower, upper] = curve.difference_at_tau_interval?;
    let contract = prepared.contract_for_result(&result).ok()?;
    let (construction, scope, validation) = constructions(&contract, &result)
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == "percentile_bootstrap")?;
    assert_eq!(validation, "not_applicable");
    if construction.functional.ends_with("+scope=insufficient") {
        return None;
    }
    let censoring = if fixed_known { "fixed_known" } else { "unweighted" };
    assert_eq!(
        construction.functional,
        format!(
            "incidence_difference+entry=delayed+censoring={censoring}+cause=1+tau=4008000000000000"
        )
    );
    assert_eq!(scope.replicates_ok, Some(BOOTSTRAP));
    Some(((lower, upper), construction, scope))
}

fn measure(test: &'static str, dgp: &'static str, fixed_known: bool, seed: u64) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: "percentile_bootstrap" },
        REPORTED_LEVEL,
    );
    for run in map_replicates(n_sim(), |rep| replicate(fixed_known, seed + rep)) {
        if let Some((interval, construction, scope)) = run {
            tally.bind_validation(&construction, scope, "not_applicable");
            tally.record(Some(interval), TRUTH);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_unweighted_incidence_percentile_coverage() {
    measure(
        "delayed_entry_unweighted_incidence_percentile_coverage",
        "independent_delayed_entry_two_cause_administrative_censoring",
        false,
        0xD3E1_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_fixed_known_incidence_percentile_coverage() {
    measure(
        "delayed_entry_fixed_known_incidence_percentile_coverage",
        "independent_delayed_entry_two_cause_fixed_known_stratified_censoring",
        true,
        0xD3E2_0000,
    );
}
