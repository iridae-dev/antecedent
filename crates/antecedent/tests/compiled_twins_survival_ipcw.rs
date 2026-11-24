//! The facade IPCW survival summary and the compiled known-censoring survival route give
//! identical numbers.
//!
//! `antecedent::estimate::survival::randomized_survival_ipcw_summary` is the kernel call the
//! compiled `SurvivalQuery` with `known_censoring` makes for its point summary, for both the
//! survival/RMST endpoint and the target-cause cumulative-incidence endpoint. The assertions
//! are on `to_bits()` of every curve value and of the restricted mean survival times. The
//! direct path publishes points only, so the compiled result without bootstrap replicates is
//! the comparison; with replicates the compiled route adds pointwise intervals whose draws are
//! seeded from the execution context's master seed.
#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "bit-identical equality is the property under test; fixtures widen small integers"
)]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::Study;
use antecedent::estimate::survival::{
    KnownCensoring, SurvivalEndpoint, SurvivalRows, randomized_survival_ipcw_summary,
};
use antecedent::prelude::ExecutionContext;
use antecedent_core::{
    CausalQuery, KnownCensoringSurvival, ObservationAssumption, SurvivalFunctional, SurvivalQuery,
    VariableId,
};
use antecedent_data::TabularData;

const GRID: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const TAU: f64 = 3.0;
const FLOOR: f64 = 0.01;

struct Fixture {
    duration: Vec<f64>,
    cause: Vec<f64>,
    treatment: Vec<f64>,
    /// Row-major subjects by grid points.
    censoring: Vec<f64>,
}

fn fixture() -> Fixture {
    let n = 120;
    let mut fixture = Fixture {
        duration: Vec::new(),
        cause: Vec::new(),
        treatment: Vec::new(),
        censoring: Vec::new(),
    };
    for i in 0..n {
        let arm = i % 2;
        // Durations on the grid, with a share followed through the horizon in each arm.
        let duration = match (i / 2) % 5 {
            0 | 1 => 1.0,
            2 => 2.0,
            _ => 3.0,
        };
        // Cause 0 censors; the failures split between two causes.
        let cause = if duration >= TAU {
            0.0
        } else if (i / 2) % 3 == 0 && duration == 2.0 {
            0.0
        } else if (i / 2) % 4 == 0 {
            2.0
        } else {
            1.0
        };
        fixture.duration.push(duration);
        fixture.cause.push(cause);
        fixture.treatment.push(arm as f64);
        let slope = 0.01 * (i % 5) as f64;
        let g1 = 0.92 - slope;
        let g2 = g1 - 0.05 - slope / 2.0;
        let g3 = g2 - 0.04;
        fixture.censoring.extend([1.0, g1, g2, g3]);
    }
    fixture
}

fn table(fixture: &Fixture) -> TabularData {
    // Survival treats any positive cause code as the event.
    let event: Vec<f64> = fixture.cause.iter().map(|cause| f64::from(*cause > 0.0)).collect();
    let column = |j: usize| -> Vec<f64> {
        fixture.censoring.chunks_exact(GRID.len()).map(|row| row[j]).collect()
    };
    let (g0, g1, g2, g3) = (column(0), column(1), column(2), column(3));
    TabularData::from_f64_columns([
        ("duration", fixture.duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", fixture.treatment.as_slice()),
        ("cause", fixture.cause.as_slice()),
        ("g0", g0.as_slice()),
        ("g1", g1.as_slice()),
        ("g2", g2.as_slice()),
        ("g3", g3.as_slice()),
    ])
    .unwrap()
}

fn query(event: u32, functional: SurvivalFunctional) -> SurvivalQuery {
    SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(event),
        treatment: VariableId::from_raw(2),
        tau: TAU,
        delayed_entry: None,
        known_censoring: Some(KnownCensoringSurvival {
            times: Arc::from(GRID),
            columns: Arc::from([4, 5, 6, 7].map(VariableId::from_raw)),
            minimum_probability: FLOOR,
        }),
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn compiled_known_censoring_survival_reproduces_the_direct_summary() {
    let fixture = fixture();
    let data = table(&fixture);
    let event_codes: Vec<i64> = fixture.cause.iter().map(|cause| i64::from(*cause > 0.0)).collect();
    let treated: Vec<bool> = fixture.treatment.iter().map(|arm| *arm == 1.0).collect();
    let (direct, direct_minimum) = randomized_survival_ipcw_summary(
        &SurvivalRows {
            duration: &fixture.duration,
            event_code: &event_codes,
            treated: &treated,
            delayed_entry: None,
        },
        &KnownCensoring { times: &GRID, survival: &fixture.censoring, minimum_probability: FLOOR },
        TAU,
        SurvivalEndpoint::Survival,
    )
    .unwrap();

    let context = ExecutionContext::for_tests(11);
    let result = Study::tabular(data)
        .query(CausalQuery::Survival(query(1, SurvivalFunctional::SurvivalAndRmst)))
        .build()
        .unwrap()
        .run(&context)
        .unwrap();
    let compiled = result.survival().unwrap();
    assert_eq!(bits(&compiled.times), bits(&direct.times));
    assert_eq!(bits(&compiled.control), bits(&direct.control));
    assert_eq!(bits(&compiled.treated), bits(&direct.treated));
    assert_eq!(compiled.rmst_control.map(f64::to_bits), direct.rmst_control.map(f64::to_bits));
    assert_eq!(compiled.rmst_treated.map(f64::to_bits), direct.rmst_treated.map(f64::to_bits));
    assert_eq!(compiled.minimum_event_risk_set, direct.minimum_event_risk_set);
    assert!(direct_minimum >= FLOOR);
    // Neither path publishes an interval without bootstrap replicates, and the compiled
    // route records that the censoring survival was supplied, not fitted.
    assert_eq!(compiled.uncertainty.as_ref(), "point_only_no_interval");
    assert!(compiled.rmst_difference_interval.is_none());
    assert_eq!(
        compiled.censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified")
    );
}

#[test]
fn compiled_known_censoring_cumulative_incidence_reproduces_the_direct_summary() {
    let fixture = fixture();
    let data = table(&fixture);
    let event_codes: Vec<i64> = fixture.cause.iter().map(|cause| *cause as i64).collect();
    let treated: Vec<bool> = fixture.treatment.iter().map(|arm| *arm == 1.0).collect();
    let target_cause = 1;
    let (direct, _) = randomized_survival_ipcw_summary(
        &SurvivalRows {
            duration: &fixture.duration,
            event_code: &event_codes,
            treated: &treated,
            delayed_entry: None,
        },
        &KnownCensoring { times: &GRID, survival: &fixture.censoring, minimum_probability: FLOOR },
        TAU,
        SurvivalEndpoint::CumulativeIncidence { target_cause },
    )
    .unwrap();

    let context = ExecutionContext::for_tests(11);
    let result = Study::tabular(data)
        .query(CausalQuery::Survival(query(
            3,
            SurvivalFunctional::CumulativeIncidence { target_cause },
        )))
        .build()
        .unwrap()
        .run(&context)
        .unwrap();
    let compiled = result.survival().unwrap();
    assert_eq!(compiled.target_cause, Some(target_cause));
    assert_eq!(bits(&compiled.times), bits(&direct.times));
    assert_eq!(bits(&compiled.control), bits(&direct.control));
    assert_eq!(bits(&compiled.treated), bits(&direct.treated));
    assert!(compiled.rmst_control.is_none() && direct.rmst_control.is_none());
    assert_eq!(compiled.minimum_event_risk_set, direct.minimum_event_risk_set);
}
