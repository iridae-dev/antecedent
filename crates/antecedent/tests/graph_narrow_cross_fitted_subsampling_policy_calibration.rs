//! Near-tie subsampling coverage of the cross-fitted learning procedure's fold-policy
//! average on its empirical scored-row frame. The wider x,w adjustment set enters
//! nuisance fitting, while the policy uses x. The target is what the route states: the
//! mean outcome `E[Y(pi_k(X))]` of each realized fold policy of the original fit, weighted
//! by the fold's share of the scored rows. A fold policy reads only the binary x, so its
//! two stratum actions fix that population value under the planted law, even when fold
//! policies differ. Every subsample keeps a different half of the rows, so the
//! subsampling law includes the variation of the conditional mean over the rows and the
//! row-wise conditional means of the scored rows are not its target.

mod common;

use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    ObservationalPolicyRequest, ObservationalPolicySpecWire, PolicyGraphDeclaration,
    compile_policy_subsampling, consume_policy_subsampling,
};
use antecedent_core::ExecutionContext;
use antecedent_io::calibration::CalibrationBasisWire;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim, uniform,
};

const TEST: &str = "graph_narrow_cross_fitted_subsampling_policy_l95";
const DGP: &str = "binary_near_tie_fold_policies_known_assignment_wider_nuisance";
const INTERVAL: &str = "cross_fitted_policy_subsampling_basic_v2";
const FOLDS: usize = 3;

/// Population value of the realized cross-fitted learner: each fold policy's
/// mean outcome, weighted by the fold's share of the scored rows. The report
/// lists the scored rows fold by fold, each fold in ascending row order, so a
/// descending row index opens the next fold. `value(x, action)` is the planted
/// conditional mean averaged over the independent fair sign W; X is a fair
/// sign, so a fold policy's value is the mean of its two stratum values.
fn fold_policy_average_value(
    rows: &[u64],
    actions: &[f64],
    x: &[f64],
    value: impl Fn(f64, f64) -> f64,
) -> Option<f64> {
    let mut total = 0.0;
    let mut folds = 0;
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && rows[end] > rows[end - 1] {
            end += 1;
        }
        let mut stratum = [None; 2];
        for (&row, &action) in rows[start..end].iter().zip(&actions[start..end]) {
            let i = usize::try_from(row).expect("row index fits usize");
            let slot = &mut stratum[usize::from(x[i] > 0.0)];
            if let Some(previous) = slot {
                #[allow(
                    clippy::float_cmp,
                    reason = "a fold policy of x alone returns the identical action for every row in a stratum"
                )]
                {
                    assert_eq!(*previous, action);
                }
            } else {
                *slot = Some(action);
            }
        }
        let [negative, positive] = stratum;
        let fold_value = 0.5 * value(-1.0, negative?) + 0.5 * value(1.0, positive?);
        total += (end - start) as f64 * fold_value;
        folds += 1;
        start = end;
    }
    assert_eq!(folds, FOLDS, "each fold's scored rows form one ascending run");
    Some(total / rows.len() as f64)
}

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasisWire)> {
    let mut next = uniform(seed, 0xA11C_E101);
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
        support: Some(vec![0.0, 1.0]),
        objective: Some("maximize".into()),
        evaluation: Some("cross_fitted_learning".into()),
        folds: Some(FOLDS),
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
    let plan = compile_policy_subsampling(request, n / 2, 100, &ctx).ok()?;
    assert_eq!(plan.plan_view().route().as_str(), "PolicySubsampling");
    let candidate = plan.execute(&ctx).ok()?;
    let artifact = plan.export(&candidate, &ctx).ok()?;
    let replay = consume_policy_subsampling(&artifact, plan.lowered().identity(), &ctx).ok()?;
    assert_eq!(replay.interval, candidate.interval);
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
    assert_eq!(candidate.basis.key.interval_method, INTERVAL);
    // E[W] = E[noise] = 0, so the planted mean averaged over W is
    // 1 + 0.3 x + action 0.02 x.
    let truth = fold_policy_average_value(
        &candidate.report.row_index,
        &candidate.report.policy_actions,
        &x,
        |xi, action| 1.0 + 0.3 * xi + action * (0.02 * xi),
    )?;
    Some(((candidate.interval.lower, candidate.interval.upper), truth, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_narrow_cross_fitted_subsampling_policy_l95() {
    let n = grid_n(1000);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0xB01D_1001 + rep, n)) {
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
