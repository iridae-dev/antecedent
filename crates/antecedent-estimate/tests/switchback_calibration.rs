//! Frozen known-truth evidence for independent-sequence switchback inference.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::switchback::switchback_itt;

fn uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
}

// Every one of the declared Bernoulli allocations contributes to the coverage
// denominator; no conditioning on within-sequence realized arm support.
fn unconditional_coverage(probability: f64, sequences: usize, draws: usize,
    heterogeneous: bool, serial_assignment: bool) -> (usize, usize) {
    const PERIODS: usize = 24;
    const TRUTH: f64 = 0.7;
    let ids = (0..sequences * PERIODS)
        .map(|row| format!("sequence-{}", row / PERIODS)).collect::<Vec<_>>();
    let refs = ids.iter().map(String::as_str).collect::<Vec<_>>();
    let probabilities = (0..refs.len()).map(|row| {
        if heterogeneous { 0.2 + 0.6 * (row % PERIODS) as f64 / (PERIODS - 1) as f64 }
        else { probability }
    }).collect::<Vec<_>>();
    let mut state = 0x4da3_2f90_6ac1_d3e7_u64;
    let mut accepted = 0;
    let mut covered = 0;
    let mut rejected = 0;
    while accepted < draws {
        let mut outcomes = Vec::with_capacity(refs.len());
        let mut assignment = Vec::with_capacity(refs.len());
        for _sequence in 0..sequences {
            let sequence_shock = 1.4 * (uniform(&mut state) - 0.5);
            let sequence_trend = 0.4 * (uniform(&mut state) - 0.5);
            let mut previous = false;
            for period in 0..PERIODS {
                let row = outcomes.len();
                let treated = if serial_assignment && period > 0 && uniform(&mut state) < 0.65 {
                    previous
                } else {
                    uniform(&mut state) < probabilities[row]
                };
                previous = treated;
                let base = sequence_shock + sequence_trend * period as f64 / PERIODS as f64
                    + 0.2 * (uniform(&mut state) - 0.5);
                assignment.push(treated);
                outcomes.push(base + if treated { TRUTH } else { 0.0 });
            }
        }
        let fit = match switchback_itt(&outcomes, &assignment, &probabilities, &refs) {
            Some(fit) if fit.interval_95.is_some() => fit,
            _ => { rejected += 1; continue; },
        };
        let interval = fit.interval_95.unwrap();
        covered += usize::from(interval[0] <= TRUTH && TRUTH <= interval[1]);
        accepted += 1;
    }
    (covered, rejected)
}

#[test]
fn switchback_student_interval_covers_known_truth() {
    for (probability, sequences, heterogeneous, serial_assignment) in [
        (0.2, 30, false, false), (0.5, 30, false, false),
        (0.8, 30, false, false), (0.5, 40, false, false),
        (0.5, 30, true, false), (0.5, 30, false, true),
    ] {
        let (covered, rejected) = unconditional_coverage(probability, sequences, 2_000, heterogeneous, serial_assignment);
        eprintln!("switchback p={probability}, sequences={sequences}, heterogeneous={heterogeneous}, serial_assignment={serial_assignment}: {covered}/2000 unconditional allocations, {rejected} unsupported draws");
        assert_eq!(rejected, 0, "calibration fixture cannot condition on observed assignment");
        assert!((1_840..=1_960).contains(&covered),
            "p={probability}, sequences={sequences} coverage {covered}/2000");
    }
}
