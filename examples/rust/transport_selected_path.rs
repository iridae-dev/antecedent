// SPDX-License-Identifier: MIT OR Apache-2.0
//! Trace a treatment effect through a selected time-ordered pathway.
//!
//! One treatment feeds two stages and the outcome. The planted coefficients give a hand-checkable contrast.

use antecedent::analysis::temporal_path_transport::{
    ContinuousChainSourceRow, ContinuousChainTransportRequest, compile_continuous_chain_transport,
    consume_continuous_chain_transport,
};
use antecedent_core::ExecutionContext;

fn main() {
    let ctx = ExecutionContext::for_tests(991);
    let source = (0..200)
        .map(|i| {
            let x = (i % 25) as f64 / 25.0;
            let treated = i >= 100;
            let a = if treated { 1.0 } else { 0.0 };
            let noise1 = if (i / 25) % 2 == 0 { -1.0 } else { 1.0 };
            let noise2 = if (i / 50) % 2 == 0 { -1.0 } else { 1.0 };
            let m1 = x + a + noise1;
            let m2 = 2.0 * m1 + 3.0 * a + noise2;
            ContinuousChainSourceRow {
                baseline: x,
                treated,
                mediators: vec![m1, m2],
                outcome: 3.0 + x + m2 + 4.0 * a,
            }
        })
        .collect();
    let request = ContinuousChainTransportRequest {
        source,
        target_baseline: (0..25).map(|i| i as f64 / 25.0).collect(),
        mediator_parents: vec![vec![], vec![0], vec![0, 1]],
        treatment_edges: vec![true, true, true],
        active_treatment_edges: vec![true, false, false],
        bootstrap_replicates: 100,
    };
    let plan = compile_continuous_chain_transport(request.clone(), &ctx).unwrap();
    let result = plan.execute(&ctx).unwrap();
    assert!((result.contrast - 2.0).abs() < 1e-10);
    assert_eq!(result.calibration, "not_measured");
    let artifact = plan.export(&result, &ctx).unwrap();
    let replay =
        consume_continuous_chain_transport(&artifact, plan.lowered().identity(), &ctx).unwrap();
    assert_eq!(result, replay);
    let mut alternate = request.clone();
    alternate.active_treatment_edges = vec![false, true, true];
    let alternate_plan = compile_continuous_chain_transport(alternate, &ctx).unwrap();
    assert_ne!(plan.lowered().identity(), alternate_plan.lowered().identity());
    let alternate_result = alternate_plan.execute(&ctx).unwrap();
    assert!((alternate_result.contrast - 7.0).abs() < 1e-10);
    assert!(
        consume_continuous_chain_transport(&artifact, alternate_plan.lowered().identity(), &ctx)
            .is_err()
    );
    println!("Effect through the first stage: {}", replay.contrast);
    println!("Effect through the other treatment links: {}", alternate_result.contrast);
}
