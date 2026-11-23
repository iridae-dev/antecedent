//! Coverage of a learned policy using less information than its nuisance
//! adjustment. The two policy strata make the learned rule's true population
//! value observable from the executed held-out actions: each stratum has one
//! action, while both nuisance models use the wider x,w set.

mod common;

use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    ObservationalPolicyRequest, ObservationalPolicySpecWire, PolicyGraphDeclaration,
    compile_observational_policy,
};
use antecedent_core::ExecutionContext;
use antecedent_io::calibration::CalibrationBasisWire;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim, uniform,
};

const TEST: &str = "graph_narrow_held_out_policy_l95";
const DGP: &str = "binary_strata_known_assignment_wider_nuisance";
const INTERVAL: &str = "held_out_policy_score_wald_v1";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasisWire)> {
    let mut next = uniform(seed, 0xA11C_E001);
    let (mut outcome, mut treatment, mut x, mut w, mut p0, mut p1) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let xi = if next() < 0.5 { -1.0 } else { 1.0 };
        let wi = if next() < 0.5 { -1.0 } else { 1.0 };
        let propensity = 0.35 + 0.10 * xi + 0.05 * wi;
        let action = f64::from(u8::from(next() < propensity));
        let noise = 0.3 * (2.0 * next() - 1.0);
        outcome.push(1.0 + 0.3 * xi + 0.5 * wi + action * (0.6 + 0.9 * xi) + noise);
        treatment.push(action);
        x.push(xi);
        w.push(wi);
        p0.push(1.0 - propensity);
        p1.push(propensity);
    }
    let split_seed = grid_seed(seed);
    let spec = ObservationalPolicySpecWire {
        support: Some(vec![0.0, 1.0]),
        objective: Some("maximize".into()),
        evaluation: Some("held_out".into()),
        evaluation_fraction: Some(0.4),
        folds: Some(3),
        known_propensity_columns: Some(vec![p0, p1]),
        causal_graph: Some(PolicyGraphDeclaration {
            nodes: vec!["x".into(), "w".into(), "a".into(), "y".into()],
            edges: vec![
                ("x".into(), "a".into()),
                ("w".into(), "a".into()),
                ("x".into(), "y".into()),
                ("w".into(), "y".into()),
                ("a".into(), "y".into()),
            ],
            treatment: "a".into(),
            outcome: "y".into(),
            information_set: vec!["x".into()],
        }),
        seed: split_seed,
        ..ObservationalPolicySpecWire::default()
    };
    let request = ObservationalPolicyRequest {
        outcome,
        treatment,
        adjustment: vec![
            AdjustmentColumn { name: "x".into(), values: x.clone() },
            AdjustmentColumn { name: "w".into(), values: w },
        ],
        spec,
    };
    let ctx = ExecutionContext::for_tests(split_seed);
    let plan = compile_observational_policy(request, &ctx).ok()?;
    let candidate = plan.held_out_value_candidate(&ctx).ok()?;
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
    assert_eq!(candidate.basis.key.interval_method, INTERVAL);
    let mut action_negative = None;
    let mut action_positive = None;
    for (&row, &action) in candidate.report.row_index.iter().zip(&candidate.report.policy_actions) {
        let i = usize::try_from(row).expect("row index fits usize");
        let slot = if x[i] < 0.0 { &mut action_negative } else { &mut action_positive };
        if let Some(previous) = slot {
            #[allow(
                clippy::float_cmp,
                reason = "a deterministic rule must return the identical action for every row in a stratum"
            )]
            {
                assert_eq!(*previous, action);
            }
        } else {
            *slot = Some(action);
        }
    }
    let negative = action_negative?;
    let positive = action_positive?;
    // E[X] = E[W] = E[noise] = 0. The held-out rule is a function of X
    // alone, so this is its exact population truth conditional on learning.
    let truth = 1.0 + 0.5 * (1.5 * positive - 0.3 * negative);
    Some(((candidate.interval.lower, candidate.interval.upper), truth, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_narrow_held_out_policy_l95() {
    let n = grid_n(1000);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0xB01D_0001 + rep, n)) {
        if let Some((interval, truth, basis)) = result {
            let key = &basis.key;
            let construction = Construction {
                query: key.query.clone(),
                graph_class: key.graph_class.clone(),
                structure: key.structure.clone(),
                modality: key.modality.clone(),
                inference: key.inference.clone(),
                estimator: key.estimator.clone(),
                interval_method: key.interval_method.clone(),
                se_kind: key.se_kind.clone(),
                dependence: key.dependence.clone(),
                posterior: key.posterior.clone(),
                functional: key.functional.clone(),
                identification: key.identification.clone(),
                reported_level: key.level,
            };
            tally.bind_validation(
                &construction,
                ScopeFacts {
                    row_count: basis.scope.row_count,
                    replicates_ok: basis.scope.replicates_ok,
                    posterior_draws: basis.scope.posterior_draws,
                    unidentified_mass: basis.scope.unidentified_mass,
                },
                &key.validation,
            );
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
