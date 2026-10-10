//! Internal unmeasured joint posterior retention; independent Gaussian algebra.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_bayesian::PosteriorSummarySpec;
use antecedent::analysis::recalc_joint_bayesian::{
    JointBayesianRequest, JointBayesianSession, execute_joint_bayesian_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::execution_counts::count_checked_identifications;
use antecedent_io::joint_bayesian_transport_artifact::{
    JointBayesianArtifactWire, JointBayesianConsumeLimits,
};
use antecedent_prob::fit_counts::count_bayesian_work;
fn request() -> JointBayesianRequest {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(vec![0.0, 1.0].into()),
            ),
        }),
        "source",
        "target",
        vec![VariableId::from_raw(0)],
    );
    let sources = ["a", "b"]
        .into_iter()
        .map(|id| {
            let treatment = (0..128).map(|row| row % 2 == 1).collect::<Vec<_>>();
            let outcome = treatment.iter().map(|t| if *t { 3.0 } else { 1.0 }).collect();
            SourceData {
                id: id.into(),
                identity: DataIdentity {
                    snapshot_digest: format!("snapshot-{id}"),
                    datum_ids: (0..128).map(|row| format!("{id}-{row}")).collect(),
                },
                treatment,
                outcome,
                covariates: vec![],
                noise_variance: 1.0,
            }
        })
        .collect();
    JointBayesianRequest {
        diagram: SelectionDiagram::try_new(graph, vec![]).unwrap(),
        query,
        model: JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features: vec![],
            varying: VaryingBlock::Intercept,
            sharing: SourceSharing::SharedVaryingBlock,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 100.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        },
        sources,
        target: TargetData {
            identity: DataIdentity {
                snapshot_digest: "target".into(),
                datum_ids: (0..40).map(|row| format!("target-{row}")).collect(),
            },
            rows: 40,
            covariates: vec![],
        },
        options: JointTransportOptions { draws: 1024, seed: 19 },
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    }
}
#[test]
fn actual_checked_joint_posterior_retention_and_independent_replay() {
    let ctx = ExecutionContext::for_tests(19);
    let mut request = request();
    let mut session = JointBayesianSession::new();
    let ((out, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_joint_bayesian_with_receipt(&mut session, &request, &ctx).unwrap()
        })
    });
    assert_eq!(out.receipt.totals().identifications, checks);
    assert!(checks > 0);
    assert_eq!(work.model_fits, 3); // actual joint solve + two source-disagreement solves
    assert_eq!(work.posterior_draws, 1024);
    assert_eq!(out.receipt.totals().model_fits, work.model_fits);
    assert_eq!(out.receipt.totals().posterior_draws, work.posterior_draws);
    let a = 128.01;
    let b = 128.0;
    let c = 256.25;
    let det = a * c - b * b;
    let mean = (c * 384.0 - b * 512.0) / det;
    let variance = c / det;
    let oracle: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/recalculation/bayesian_retention/expected.json"
    )))
    .unwrap();
    assert!((mean - oracle["joint_internal"]["mean"].as_f64().unwrap()).abs() < 1e-12);
    assert!((variance - oracle["joint_internal"]["variance"].as_f64().unwrap()).abs() < 1e-12);
    let fit = session.fit().unwrap();
    assert!((fit.target_effect_mean - mean).abs() < 1e-12);
    assert!((fit.target_effect_variance - variance).abs() < 1e-12);
    let ptr = fit.draws.values.as_ptr();
    request.summary.threshold = 2.0;
    let (out, work) = count_bayesian_work(|| {
        execute_joint_bayesian_with_receipt(&mut session, &request, &ctx).unwrap()
    });
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    assert_eq!(out.receipt.totals().law_summaries, 1);
    assert_eq!(session.fit().unwrap().draws.values.as_ptr(), ptr);
    let bytes = session.export_result().unwrap();
    let (wire, replayed) = JointBayesianArtifactWire::consume_with_limits(
        &bytes,
        JointBayesianConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(wire.calibration, "unmeasured");
    assert_eq!(replayed, *session.fit().unwrap());
    request.model.priors.varying.provenance = PriorProvenance::Bank {
        bank_id: "used-bank".into(),
        consumed: vec![request.sources[0].identity.clone()],
    };
    let (failure, work) =
        count_bayesian_work(|| execute_joint_bayesian_with_receipt(&mut session, &request, &ctx));
    assert!(failure.is_err());
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    request.model.priors.varying.provenance = PriorProvenance::Bank {
        bank_id: "oversized-bank".into(),
        consumed: vec![request.sources[0].identity.clone(); 17],
    };
    let (failure, work) =
        count_bayesian_work(|| execute_joint_bayesian_with_receipt(&mut session, &request, &ctx));
    assert!(matches!(
        failure,
        Err(antecedent::analysis::recalc_receipt::RecalcRunError::Request(
            "recalc.bayesian_workspace_limit"
        ))
    ));
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    request.model.priors.varying.provenance = PriorProvenance::Declared;
    assert_eq!(
        execute_joint_bayesian_with_receipt(&mut session, &request, &ctx)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
}
