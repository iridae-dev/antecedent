//! Source law for the standardized conditional-survival checks.
#![allow(dead_code, reason = "two test binaries share this module and each uses a subset")]

use std::sync::Arc;

use antecedent_core::{
    ObservationAssumption, StratifiedCensoringSurvival, SurvivalFunctional, SurvivalQuery,
    VariableId,
};
use antecedent_data::TabularData;

pub const TAU: f64 = 3.0;
pub const DRAWS: u32 = 399;
pub const MIN_ARM: usize = 1_000;

const EVENT: [[[[usize; 2]; 2]; 2]; 2] =
    [[[[2, 1], [3, 2]], [[1, 1], [2, 1]]], [[[1, 1], [2, 1]], [[2, 2], [4, 2]]]];

type Row = [f64; 6];

/// One pre-entry study unit: its arm and history strata, and its event, entry and censoring draws.
struct Unit {
    arm: usize,
    entry: usize,
    censor: usize,
    event_draw: usize,
    entry_draw: usize,
    censor_draw: usize,
}

fn observed(unit: &Unit, delayed: bool, censoring: bool) -> Option<Row> {
    let Unit { arm, entry, censor, event_draw, entry_draw, censor_draw } = *unit;
    let [early, later] = EVENT[arm][entry][censor];
    let time = if event_draw < early {
        1.0
    } else if event_draw < early + later {
        2.0
    } else {
        TAU
    };
    let entry_time = f64::from(delayed && entry_draw < [1, 6][entry]);
    let censor_time = if censoring && censor_draw < [1, 3][censor] { 1.5_f64 } else { TAU };
    let exit = time.min(censor_time);
    (entry_time < exit).then(|| {
        [
            exit,
            f64::from(time < TAU && time <= censor_time),
            arm as f64,
            entry_time,
            entry as f64,
            censor as f64,
        ]
    })
}

/// Enumerate the full pre-entry law without sampling; exclusion happens by observation time.
pub fn exact(delayed: bool, censoring: bool) -> Vec<Row> {
    let mut observed_rows = Vec::new();
    for arm in 0..2 {
        for entry in 0..2 {
            for censor in 0..2 {
                for time in 0..8 {
                    for left in 0..8 {
                        for drop in 0..8 {
                            let unit = Unit {
                                arm,
                                entry,
                                censor,
                                event_draw: time,
                                entry_draw: left,
                                censor_draw: drop,
                            };
                            observed_rows.extend(observed(&unit, delayed, censoring));
                        }
                    }
                }
            }
        }
    }
    observed_rows
}

/// Sample independent study units under one prespecified observation design.
pub fn sampled(
    arm_size: usize,
    delayed: bool,
    censoring: bool,
    mut uniform: impl FnMut() -> f64,
) -> Vec<Row> {
    let mut rows = Vec::with_capacity(2 * arm_size);
    for arm in 0..2 {
        while rows.len() < (arm + 1) * arm_size {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a uniform draw in [0, 1) scaled by a small count floors to an index"
            )]
            let mut sample = |n: u32| (uniform() * f64::from(n)) as usize;
            let unit = Unit {
                arm,
                entry: sample(2),
                censor: sample(2),
                event_draw: sample(8),
                entry_draw: sample(8),
                censor_draw: sample(8),
            };
            rows.extend(observed(&unit, delayed, censoring));
        }
    }
    rows
}

/// Closed-form standardized arm survival at every analysis knot in the source population.
pub fn truth(arm: usize, time: f64) -> f64 {
    let event_probability = EVENT[arm]
        .iter()
        .flatten()
        .map(|[early, later]| early * usize::from(time >= 1.0) + later * usize::from(time >= 2.0))
        .sum::<usize>() as f64
        / 32.0;
    1.0 - event_probability
}

pub fn table(rows: &[Row]) -> TabularData {
    let columns = (0..6)
        .map(|index| rows.iter().map(|row| row[index]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    TabularData::from_f64_columns(
        ["duration", "event", "treatment", "entry", "entry_history", "censor_history"]
            .into_iter()
            .zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

pub fn query(delayed: bool) -> SurvivalQuery {
    let id = VariableId::from_raw;
    SurvivalQuery {
        duration: id(0),
        event: id(1),
        treatment: id(2),
        tau: TAU,
        delayed_entry: delayed.then(|| id(3)),
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([id(4), id(5)])),
        stratified_censoring: Some(StratifiedCensoringSurvival {
            entry_stratum: id(4),
            censoring_stratum: id(5),
            population_weights: [0.5, 0.5],
            minimum_probability: 0.05,
        }),
        functional: SurvivalFunctional::SurvivalAndRmst,
    }
}
