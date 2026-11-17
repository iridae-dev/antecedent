//! Prospective joint-endpoint coverage of a compiler-owned IID bootstrap
//! assumption union. The true set is [1, 2]: a Gaussian mean of 1 plus the
//! fixed assumption interval [0, 1]. Each outer replicate samples new raw
//! units and runs all inner bootstrap chains through the compiler. The event
//! is `union_lower <= 1 && union_upper >= 2`, represented as containment of
//! 1 in `[union_lower, union_upper - 1]` for the scalar coverage tally.
//! This ignored harness earns no record.

mod common;

use antecedent::analysis::propagation_chain::PropagationChain;
use antecedent::analysis::sampling_assumption_union::{
    compile_iid_sampling_assumption_union, IidSamplingAssumptionUnionRequest,
};
use antecedent_contract::propagation_chain::{ChainReading, LawMeaning, StageKind, TableConstruction};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_io::propagation_chain_record::{ChainOpWire, ChainSourceWire, ChainStageWire};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    gaussian, grid_n, grid_seed, map_replicates, n_sim,
};

const TEST: &str = "iid_sampling_assumption_union_joint_endpoints_l95";
const DGP: &str = "iid_gaussian_mean_plus_fixed_unit_assumption";
const INTERVAL: &str = "bonferroni_basic_bootstrap";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut noise = gaussian(seed ^ 0xA01C_F201);
    let observations: Vec<f64> = (0..n).map(|_| 1.0 + noise()).collect();
    let original = PropagationChain::draft(
        TableConstruction::JointDraws, ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws(
                "raw.units", LawMeaning::Sampling, "iid-raw-units", &observations,
            ),
            ChainSourceWire::interval("fixed.bias", 0.0, 1.0),
        ],
        vec![ChainStageWire::new(
            "answer", StageKind::DerivedQuantity, &["raw.units", "fixed.bias"],
            ChainOpWire::binary("sum"),
        )],
    ).with_output_quantity(ScientificQuantityWire {
        version: 1, variable_id: "target.mean_plus_bias".into(),
        variable_name: "target mean plus fixed bias".into(),
        role: "outcome".into(), units: "score".into(),
        population_id: "target".into(), regime_id: "observed".into(),
        horizon: 0, functional_id: "identified_set_mean".into(),
        conditioning: Vec::new(), transform_id: "identity".into(),
    });
    let request = IidSamplingAssumptionUnionRequest {
        original,
        unit_ids: (0..n).map(|row| format!("unit-{row}")).collect(),
        bootstrap_replicates: 200, level: REPORTED_LEVEL,
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_iid_sampling_assumption_union(request, &ctx).ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    Some(((candidate.result.union_lower, candidate.result.union_upper - 1.0),
        1.0, candidate.basis))
}

#[test]
#[ignore = "source-only prospective whole-route joint-set coverage; no record earned"]
fn iid_sampling_assumption_union_joint_endpoints_l95() {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(RecordKey {
        test: TEST, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for result in map_replicates(n_sim(), |rep| replicate(0xD0C0_7001 + rep, n)) {
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
            tally.bind_validation(&construction, ScopeFacts {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            }, &basis.validation);
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
