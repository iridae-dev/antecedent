//! Joint-endpoint coverage of a compiler-owned whole-cluster bootstrap
//! assumption union. The true set is [1, 2]: mean 1 plus fixed bias [0, 1].
//! Each independent cluster contributes two correlated rows; every outer
//! replicate regenerates clusters and reruns the complete chain.

mod common;

use antecedent::analysis::propagation_chain::PropagationChain;
use antecedent::analysis::sampling_assumption_union::{
    ClusterSamplingAssumptionUnionRequest, compile_cluster_sampling_assumption_union,
};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_io::propagation_chain_record::{ChainOpWire, ChainSourceWire, ChainStageWire};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim,
};

const TEST: &str = "cluster_sampling_assumption_union_joint_endpoints_l95";
const DGP: &str = "independent_gaussian_clusters_two_units_plus_fixed_bias";
const INTERVAL: &str = "bonferroni_basic_cluster_bootstrap";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut cluster_noise = gaussian(seed ^ 0xC105_1001);
    let mut unit_noise = gaussian(seed ^ 0xC105_1002);
    let observations: Vec<f64> = (0..n / 2)
        .flat_map(|_| {
            let shared = cluster_noise();
            [1.0 + shared + 0.3 * unit_noise(), 1.0 + shared + 0.3 * unit_noise()]
        })
        .collect();
    let original = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws(
                "raw.units",
                LawMeaning::Sampling,
                "cluster-raw-units",
                &observations,
            ),
            ChainSourceWire::interval("fixed.bias", 0.0, 1.0),
        ],
        vec![ChainStageWire::new(
            "answer",
            StageKind::DerivedQuantity,
            &["raw.units", "fixed.bias"],
            ChainOpWire::binary("sum"),
        )],
    )
    .with_output_quantity(ScientificQuantityWire {
        version: 1,
        variable_id: "target.mean_plus_bias".into(),
        variable_name: "target mean plus fixed bias".into(),
        role: "outcome".into(),
        units: "score".into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 0,
        functional_id: "identified_set_mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let request = ClusterSamplingAssumptionUnionRequest {
        original,
        unit_ids: (0..n).map(|row| format!("unit-{row}")).collect(),
        cluster_ids: (0..n).map(|row| format!("cluster-{}", row / 2)).collect(),
        bootstrap_replicates: 200,
        level: REPORTED_LEVEL,
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_cluster_sampling_assumption_union(request, &ctx).ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    Some(((candidate.result.union_lower, candidate.result.union_upper - 1.0), 1.0, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_sampling_assumption_union_joint_endpoints_l95() {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0xC105_7001 + rep, n)) {
        if let Some((interval, truth, basis)) = result {
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
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
