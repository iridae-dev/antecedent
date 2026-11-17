//! Prospective near-tie subsampling coverage of the cross-fitted learning procedure on its
//! empirical scored-row frame. The wider x,w adjustment set enters nuisance
//! fitting, while the policy uses x. Every realized fold action has an exact
//! conditional outcome mean under the planted structural law, so the target
//! is known even when fold policies differ. This ignored harness earns no
//! record.

mod common;

use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    compile_policy_subsampling, consume_policy_subsampling, ObservationalPolicyRequest,
    ObservationalPolicySpecWire, PolicyGraphDeclaration,
};
use antecedent_core::ExecutionContext;
use antecedent_io::calibration::CalibrationBasisWire;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    grid_n, grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "graph_narrow_cross_fitted_subsampling_policy_l95";
const DGP: &str = "binary_near_tie_fold_policies_known_assignment_wider_nuisance";
const INTERVAL: &str = "cross_fitted_policy_subsampling_basic_v1";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasisWire)> {
    let mut next = uniform(seed, 0xA11C_E101);
    let (mut outcome, mut treatment, mut x, mut w, mut p0, mut p1) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n),
         Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let xi = if next() < 0.5 { -1.0 } else { 1.0 };
        let wi = if next() < 0.5 { -1.0 } else { 1.0 };
        let propensity = 0.35 + 0.10 * xi + 0.05 * wi;
        let action = f64::from(u8::from(next() < propensity));
        let noise = 0.1 * (2.0 * next() - 1.0);
        outcome.push(1.0 + 0.3 * xi + 0.1 * wi + action * (0.02 * xi) + noise);
        treatment.push(action);
        x.push(xi);
        w.push(wi);
        p0.push(1.0 - propensity);
        p1.push(propensity);
    }
    let split_seed = grid_seed(seed);
    let spec = ObservationalPolicySpecWire {
        support: Some(vec![0.0, 1.0]), objective: Some("maximize".into()),
        evaluation: Some("cross_fitted_learning".into()),
        folds: Some(3), known_propensity_columns: Some(vec![p0, p1]),
        causal_graph: Some(PolicyGraphDeclaration {
            nodes: vec!["x".into(), "w".into(), "a".into(), "y".into()],
            edges: vec![("x".into(), "a".into()), ("w".into(), "a".into()),
                ("x".into(), "y".into()), ("w".into(), "y".into()),
                ("a".into(), "y".into())],
            treatment: "a".into(), outcome: "y".into(),
            information_set: vec!["x".into()],
        }),
        seed: split_seed, ..ObservationalPolicySpecWire::default()
    };
    let request = ObservationalPolicyRequest {
        outcome, treatment,
        adjustment: vec![
            AdjustmentColumn { name: "x".into(), values: x.clone() },
            AdjustmentColumn { name: "w".into(), values: w.clone() },
        ],
        spec,
    };
    let ctx = ExecutionContext::for_tests(split_seed);
    let plan = compile_policy_subsampling(request, n / 2, 100, &ctx).ok()?;
    assert_eq!(plan.plan_view().route().as_str(), "PolicySubsampling");
    let candidate = plan.execute(&ctx).ok()?;
    let artifact = plan.export(&candidate, &ctx).ok()?;
    let replay = consume_policy_subsampling(&artifact, plan.lowered().identity(), &ctx).ok()?;
    assert_eq!(replay.interval, candidate.interval);
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
    assert_eq!(candidate.basis.key.interval_method, INTERVAL);
    let truth = candidate.report.row_index.iter().zip(&candidate.report.policy_actions)
        .map(|(&row, &action)| {
            let i = row as usize;
            1.0 + 0.3 * x[i] + 0.1 * w[i]
                + action * (0.02 * x[i])
        })
        .sum::<f64>() / candidate.report.evaluation_rows as f64;
    Some(((candidate.interval.lower, candidate.interval.upper), truth, candidate.basis))
}

#[test]
#[ignore = "source-only candidate; near-tie subsampling coverage is not earned"]
fn graph_narrow_cross_fitted_subsampling_policy_l95() {
    let n = grid_n(1000);
    let mut tally = CoverageTally::for_record(RecordKey {
        test: TEST, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for result in map_replicates(n_sim(), |rep| replicate(0xB01D_1001 + rep, n)) {
        if let Some((interval, truth, basis)) = result {
            let key = &basis.key;
            let construction = Construction {
                query: key.query.clone(), graph_class: key.graph_class.clone(),
                structure: key.structure.clone(), modality: key.modality.clone(),
                inference: key.inference.clone(), estimator: key.estimator.clone(),
                interval_method: key.interval_method.clone(), se_kind: key.se_kind.clone(),
                dependence: key.dependence.clone(), posterior: key.posterior.clone(),
                functional: key.functional.clone(), identification: key.identification.clone(),
                reported_level: key.level,
            };
            tally.bind_validation(&construction, ScopeFacts {
                row_count: basis.scope.row_count,
                replicates_ok: basis.scope.replicates_ok,
                posterior_draws: basis.scope.posterior_draws,
                unidentified_mass: basis.scope.unidentified_mass,
            }, &key.validation);
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
