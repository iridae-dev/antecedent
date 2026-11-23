//! Independent branching-cluster oracle for a single exponential event type.

use antecedent::analysis::events::{Arrival, EventHistory, EventWindow};
use antecedent_core::CausalRng;

pub fn branching_history(seed: u64, cutoff: f64, end: f64) -> (EventHistory, usize) {
    let mut rng = CausalRng::from_seed(seed);
    let mut times = Vec::new();
    let mut time = 0.0;
    let exponential = |rng: &mut CausalRng, rate: f64| -(-rng.next_f64()).ln_1p() / rate;
    loop {
        time += exponential(&mut rng, 0.9);
        if time >= end {
            break;
        }
        times.push(time);
    }
    let mut index = 0;
    while index < times.len() {
        // Offspring arrivals are a Poisson process on cumulative offspring mass.
        // Inversion samples ages with the exponential branching kernel.
        let mut mass = 0.0;
        loop {
            mass += exponential(&mut rng, 1.0);
            if mass >= 0.45 {
                break;
            }
            let child = times[index] - (1.0 - mass / 0.45).ln();
            if child < end {
                times.push(child);
            }
        }
        index += 1;
    }
    times.sort_unstable_by(f64::total_cmp);
    let split = times.partition_point(|&t| t < cutoff);
    let future = times.len() - split;
    (
        EventHistory {
            catalog: vec!["mention".into()],
            time_unit: "hour".into(),
            origin: "oracle".into(),
            source: "independent_cluster_truth".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: cutoff },
            left: vec![],
            gaps: vec![],
            events: times[..split].iter().map(|&time| Arrival { time, kind: 0 }).collect(),
        },
        future,
    )
}
