//! Coverage of the compiled pointwise target-cause incidence interval under delayed entry.
//!
//! Cause-1 masses at times one and two are 0.15 in control and 0.10 in treatment.
//! Both arms have cause-2 mass 0.10 at each event time. The source-population CIF
//! contrast at horizon three is therefore -0.10. Entry is drawn independently of
//! event time and cause. The weighted design adds subject-specific independent
//! censoring at 1.5 and 2.5 with fixed known probabilities at both times.
//! Coverage uses all attempted studies. An unpublished interval or band is a miss.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::support::CellStatus;
use antecedent::{PreparedStudy, Study, StudyResult};
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

fn trial(fixed_known: bool, incidence: bool, seed: u64) -> (TabularData, SurvivalQuery) {
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
            } else if incidence {
                cause
            } else {
                f64::from(cause > 0.0)
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
        stratified_censoring: None,
        known_censoring: fixed_known.then(|| KnownCensoringSurvival {
            times: Arc::from(TIMES),
            columns: (4..10).map(VariableId::from_raw).collect(),
            minimum_probability: 0.01,
        }),
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: if incidence {
            SurvivalFunctional::CumulativeIncidence { target_cause: 1 }
        } else {
            SurvivalFunctional::SurvivalAndRmst
        },
    };
    (data, query)
}

fn published_construction(
    prepared: &PreparedStudy,
    result: &StudyResult,
    method: &str,
) -> (Construction, ScopeFacts) {
    let contract = prepared.contract_for_result(result).expect("published result has a contract");
    let mut published = constructions(&contract, result);
    assert_eq!(published.len(), 1, "a band and scalar interval cannot share the calibration slot");
    let (construction, scope, validation) = published.pop().unwrap();
    assert_eq!(construction.interval_method, method);
    assert_eq!(
        [
            construction.query.as_str(),
            construction.graph_class.as_str(),
            construction.structure.as_str(),
            construction.modality.as_str(),
            construction.inference.as_str(),
            construction.estimator.as_str(),
            construction.se_kind.as_str(),
            construction.dependence.as_str(),
            construction.posterior.as_str(),
            construction.identification.as_str(),
            validation,
        ],
        [
            "Survival",
            "RandomizedTrial",
            "design",
            "tabular",
            "Frequentist",
            "randomized.survival_product_limit",
            "",
            "iid",
            "",
            "point",
            "not_applicable",
        ]
    );
    assert_eq!(construction.reported_level.to_bits(), REPORTED_LEVEL.to_bits());
    (construction, scope)
}

fn replicate(fixed_known: bool, seed: u64) -> Option<((f64, f64), Construction, ScopeFacts)> {
    let (data, query) = trial(fixed_known, true, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query))
        .bootstrap_replicates(BOOTSTRAP)
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    if result.support_status() != Some(CellStatus::Supported) {
        return None;
    }
    let curve = result.survival()?;
    let [lower, upper] = curve.difference_at_tau_interval?;
    let (construction, scope) = published_construction(&prepared, &result, "percentile_bootstrap");
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

fn band_replicate(
    fixed_known: bool,
    incidence: bool,
    seed: u64,
) -> Option<((f64, f64), f64, Construction, ScopeFacts)> {
    const BAND_BOOTSTRAP: u32 = 399;
    let (data, query) = trial(fixed_known, incidence, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query))
        .bootstrap_replicates(BAND_BOOTSTRAP)
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let band = result.survival()?.difference_band.as_ref()?;
    if band.support_status != Some(CellStatus::Supported) {
        return None;
    }
    let exact = |time| {
        let sign = if incidence { -1.0 } else { 1.0 };
        sign * (0.05 * f64::from(time >= 1.0) + 0.05 * f64::from(time >= 2.0))
    };
    let slack = |(lower, upper, truth): (f64, f64, f64)| (truth - lower).min(upper - truth);
    let (lower, upper, truth) = band
        .times
        .iter()
        .zip(&*band.lower)
        .zip(&*band.upper)
        .map(|((&time, &lower), &upper)| (lower, upper, exact(time)))
        .min_by(|&left, &right| slack(left).total_cmp(&slack(right)))?;
    let (construction, scope) = published_construction(&prepared, &result, "simultaneous_band");
    assert_eq!(scope.replicates_ok, Some(BAND_BOOTSTRAP));
    let censoring = if fixed_known { "fixed_known" } else { "unweighted" };
    let endpoint = if incidence { "incidence:1" } else { "survival" };
    assert_eq!(
        construction.functional,
        format!(
            "difference_band+entry=delayed+censoring={censoring}+endpoint={endpoint}+tau=4008000000000000"
        )
    );
    Some(((lower, upper), truth, construction, scope))
}

fn measure_band(test: &'static str, fixed_known: bool, incidence: bool, seed: u64) {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "independent_delayed_entry_two_cause_complete_analysis_grid",
            interval: "simultaneous_band",
        },
        REPORTED_LEVEL,
    );
    for run in map_replicates(n_sim(), |rep| band_replicate(fixed_known, incidence, seed + rep)) {
        if let Some((interval, truth, construction, scope)) = run {
            tally.bind_validation(&construction, scope, "not_applicable");
            // Coverage holds jointly exactly when the least-covered grid point is inside.
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_unweighted_survival_band_coverage() {
    measure_band("delayed_entry_unweighted_survival_band_coverage", false, false, 0xD3E3_0000);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_unweighted_incidence_band_coverage() {
    measure_band("delayed_entry_unweighted_incidence_band_coverage", false, true, 0xD3E4_0000);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_fixed_known_survival_band_coverage() {
    measure_band("delayed_entry_fixed_known_survival_band_coverage", true, false, 0xD3E5_0000);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn delayed_entry_fixed_known_incidence_band_coverage() {
    measure_band("delayed_entry_fixed_known_incidence_band_coverage", true, true, 0xD3E6_0000);
}

#[path = "common/conditional_survival_fixture.rs"]
mod conditional_survival_fixture;

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn jointly_conditional_entry_and_censoring_survival_band_coverage() {
    use conditional_survival_fixture as fixture;
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "jointly_conditional_entry_and_censoring_survival_band_coverage",
            dgp: "binary_baseline_entry_and_conditional_loss_pre_entry_survival_source",
            interval: "simultaneous_band",
        },
        REPORTED_LEVEL,
    );
    let run = |rep| -> Option<((f64, f64), f64, Construction, ScopeFacts)> {
        let seed = 0xD3E7_0000 + rep;
        let data = fixture::table(&fixture::sampled(
            grid_n(2 * fixture::MIN_ARM),
            true,
            true,
            uniform(seed, 0xED0F),
        ));
        let ctx = ExecutionContext::for_tests(seed);
        let prepared = Study::tabular(data.clone())
            .query(CausalQuery::Survival(fixture::query(true)))
            .bootstrap_replicates(fixture::DRAWS)
            .build()
            .ok()?
            .prepare(&ctx)
            .ok()?;
        let result = prepared.estimate(&data, &ctx).ok()?;
        let band = result.survival()?.difference_band.as_ref()?;
        if band.support_status != Some(CellStatus::Supported) {
            return None;
        }
        let slack = |(lower, upper, truth): (f64, f64, f64)| (truth - lower).min(upper - truth);
        let (lower, upper, truth) = band
            .times
            .iter()
            .zip(&*band.lower)
            .zip(&*band.upper)
            .map(|((&time, &lower), &upper)| {
                (lower, upper, fixture::truth(1, time) - fixture::truth(0, time))
            })
            .min_by(|&left, &right| slack(left).total_cmp(&slack(right)))?;
        let (construction, scope) = published_construction(&prepared, &result, "simultaneous_band");
        assert_eq!(scope.replicates_ok, Some(fixture::DRAWS));
        assert_eq!(
            construction.functional,
            concat!(
                "difference_band+entry=standardized+censoring=stratified_refit+endpoint=survival",
                "+tau=4008000000000000+population=3fe0000000000000/3fe0000000000000+floor=3fa999999999999a"
            )
        );
        Some(((lower, upper), truth, construction, scope))
    };
    for run in map_replicates(n_sim(), run) {
        if let Some((interval, truth, construction, scope)) = run {
            tally.bind_validation(&construction, scope, "not_applicable");
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
