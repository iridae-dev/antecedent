//! Simultaneous coverage of the complete source-target trial mapping
//! discrepancy grid. Every replicate regenerates independent trial terminal
//! effects, refits both regressions through the compiler and checks every
//! band against the known target-minus-source line.

mod common;

use antecedent::analysis::bridge_transport_premise_assessment::{
    BridgeTransportPremiseAssessmentRequest, compile_bridge_transport_premise_assessment,
};
use antecedent_contract::bridge::{
    BridgeScope, TransportabilityPremise, TrialEffectPair, TrialLevelValidation,
};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim,
};

const TEST: &str = "bridge_target_mapping_discrepancy_grid_l95";
const DGP: &str = "independent_gaussian_trial_lines_fixed_surrogate_design";
const INTERVAL: &str = "bonferroni_normal_wald_95_grid";

fn replicate(seed: u64, n: usize) -> Option<(bool, CalibrationBasis)> {
    let mut source_noise = gaussian(seed ^ 0xB71D_1001);
    let mut target_noise = gaussian(seed ^ 0xB71D_1002);
    let make_trials = |offset: f64, noise: &mut dyn FnMut() -> f64| TrialLevelValidation {
        trials: (0..n)
            .map(|index| {
                let x = -1.0 + 2.0 * index as f64 / (n - 1) as f64;
                TrialEffectPair {
                    surrogate_effect: x,
                    terminal_effect: 0.3 + 1.2 * x + offset + 0.15 * noise(),
                }
            })
            .collect(),
        min_trials: 3,
        min_r_squared: 0.0,
    };
    let request = BridgeTransportPremiseAssessmentRequest {
        premise: TransportabilityPremise {
            id: "trial-surrogate-transfer".into(),
            statement: "the source trial line is proposed for the target".into(),
            established_in: BridgeScope {
                population_id: "source".into(),
                regime_id: "trial".into(),
            },
            applies_to: BridgeScope { population_id: "target".into(), regime_id: "trial".into() },
            evidence: Some("independent-target-trials".into()),
        },
        source: make_trials(0.0, &mut source_noise),
        target: make_trials(0.05, &mut target_noise),
        source_trial_ids: (0..n).map(|i| format!("source-trial-{i}")).collect(),
        target_trial_ids: (0..n).map(|i| format!("target-trial-{i}")).collect(),
        surrogate_grid: vec![-0.8, -0.4, 0.0, 0.4, 0.8],
        equivalence_margin: 0.2,
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_bridge_transport_premise_assessment(request).ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    let covered =
        candidate.result.grid.iter().all(|point| point.lower <= 0.05 && 0.05 <= point.upper);
    Some((covered, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bridge_target_mapping_discrepancy_grid_l95() {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0xB71D_7001 + rep, n)) {
        if let Some((covered, basis)) = result {
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
            tally.record(Some(if covered { (-1.0, 1.0) } else { (1.0, 2.0) }), 0.0);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
