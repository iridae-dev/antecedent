//! One-shot AIPW uses its complete prepared operation on the licensed slice.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, EstimatorSpec, RefuteSuite, Study};
use antecedent_core::ExecutionContext;

mod common;

use common::fixtures::confounded_scm;

#[test]
fn one_shot_aipw_matches_prepared_and_nearby_linear_stays_supported() {
    let ctx = ExecutionContext::for_tests(29);
    let (data, dag, query) = confounded_scm(512, 73);
    let aipw = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let one_shot = aipw.run(&ctx).unwrap();
    let prepared = aipw.prepare(&ctx).unwrap();
    let clicked = prepared.estimate(&data, &ctx).unwrap();
    assert!((one_shot.effect() - clicked.effect()).abs() < 1e-10);

    let linear = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!((linear.run(&ctx).unwrap().effect() - 2.0).abs() < 0.2);
}

/// The trimmed cross-fitted AIPW study, sealed. The pinned bits are what the
/// ordinary study dispatcher computed for this seed and data before the trim
/// rule was lowered into the checked procedure; the sealed route must agree
/// bit for bit, bootstrap included.
#[test]
fn one_shot_trimmed_aipw_is_sealed_and_matches_the_ordinary_dispatcher_bitwise() {
    const LEGACY_ATE_BITS: u64 = 0x3fff_e021_1571_42a3;
    const LEGACY_SE_BITS: u64 = 0x3fa6_00b0_1a9f_2da1;
    const LEGACY_SE_BOOTSTRAP_BITS_8_REPLICATES: u64 = 4_586_416_601_391_101_200;
    let overlap = antecedent_estimate::OverlapPolicy::RequireDiagnostics {
        clip: Some(0.01),
        trim: Some(0.02),
    };
    for replicates in [0u32, 8] {
        let ctx = ExecutionContext::for_tests(31);
        let (data, dag, query) = confounded_scm(512, 73);
        let mut fitter = antecedent_estimate::AipwAte::new();
        fitter.bootstrap_replicates = replicates;
        fitter.overlap = overlap;
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query)
            .estimator(EstimatorSpec::Aipw(Box::new(fitter)))
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap();

        // A trimming policy lowers to the common-support procedure and is
        // sealed like the untrimmed one ...
        let prepared = study.prepare(&ctx).unwrap();
        let lowering = prepared.checked_aipw_ate().expect("trimmed AIPW is sealed").lowering();
        assert_eq!(
            lowering.procedure,
            antecedent_estimate::CheckedAipwProcedure::TrimmedLogisticOls
        );
        assert_eq!(lowering.overlap, overlap);
        assert_eq!(lowering.folds, 0);
        // ... so the one-shot run executes that retained operation.
        let one_shot = study.run(&ctx).unwrap();
        let clicked = prepared.estimate(&data, &ctx).unwrap();
        for result in [&one_shot, &clicked] {
            assert_eq!(result.effect().to_bits(), LEGACY_ATE_BITS, "replicates={replicates}");
            assert_eq!(
                result.estimate.se_analytic.to_bits(),
                LEGACY_SE_BITS,
                "replicates={replicates}"
            );
            assert_eq!(
                result.estimate.se_bootstrap.map(f64::to_bits),
                (replicates > 0).then_some(LEGACY_SE_BOOTSTRAP_BITS_8_REPLICATES),
                "replicates={replicates}"
            );
            assert!((result.effect() - 2.0).abs() < 0.2, "ate={}", result.effect());
            let codes: Vec<&str> = result.diagnostics.iter().map(|d| d.code.as_ref()).collect();
            assert!(
                codes.contains(&"estimate.overlap.require_diagnostics"),
                "the trimmed policy must be the one that ran: {codes:?}"
            );
            assert!(
                codes.contains(&"estimate.aipw.full_sample_residualized"),
                "the trimmed fit is the full-sample common-support procedure: {codes:?}"
            );
            assert!(
                codes.contains(&"exec.identify.cached"),
                "both clicks execute the retained checked operation: {codes:?}"
            );
            assert_eq!(result.refutations.len(), 2, "replicates={replicates}");
        }
    }
}
