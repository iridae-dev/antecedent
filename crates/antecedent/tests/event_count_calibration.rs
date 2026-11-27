//! Count calibration against an independent simulator. Run through the calibration lane.

use antecedent::Study;
use antecedent::analysis::events::{EventForecastRequest, EventSource, EventWindow, HawkesFitSpec};
use antecedent::prelude::ExecutionContext;

#[path = "common/calibration.rs"]
mod calibration;
#[path = "events_support/cluster_oracle.rs"]
mod truth;

use calibration::{
    Construction, CoverageTally, RecordKey, SampleGrid, ScopeFacts, grid_seed, map_replicates,
    n_sim,
};

const TEST: &str = "hawkes_recursive_refit_count_l90";

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn hawkes_recursive_refit_count_l90() {
    let cutoff = SampleGrid::STANDARD.n(256) as f64;
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: TEST,
            dgp: "hawkes_exponential_cluster_interior",
            interval: "smoothed_rank_90",
        },
        0.90,
    );
    let replicates = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x4556_4e54_4341_4c00 ^ rep);
        let (history, future) = truth::branching_history(seed, cutoff, cutoff + 2.0);
        let context = ExecutionContext::for_tests(seed);
        let request = EventForecastRequest {
            history,
            forecast: EventWindow { start: cutoff, end: cutoff + 2.0 },
            source: EventSource::Fitted {
                fit: HawkesFitSpec { decay: vec![1.0], column_cap: 0.90, iterations: 1400 },
                refits: 16,
            },
            paths: 256,
            max_events: 2048,
            seed: seed ^ 0x89f2_4be7,
        };
        let result = Study::prepare_event_forecast(request, None, &context)
            .ok()?
            .count_coverage_candidate(0, &context)
            .ok()?;
        Some((result, future))
    });
    for entry in replicates {
        if let Some((candidate, future)) = entry {
            let basis = candidate.basis;
            let construction = Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
                unit: None,
            };
            tally.bind_validation(
                &construction,
                ScopeFacts {
                    row_count: basis.row_count,
                    replicates_ok: basis.replicates_ok,
                    posterior_draws: basis.posterior_draws,
                    unidentified_mass: basis.unidentified_mass,
                },
                &basis.validation,
            );
            tally.record(
                Some((candidate.bounds[0] as f64, candidate.bounds[1] as f64)),
                future as f64,
            );
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
