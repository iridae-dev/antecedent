//! The two-step temporal transport route under a Dirichlet posterior mean, against the
//! enumerated interventional truth of the target dynamic SCM (`common/temporal_fixture.rs`).
//!
//! Every supplied source and target law is the exact preimage of the enumerated law under the
//! update (`w = u (1 + K c) - c` over the positive cells, so the Dirichlet mean with
//! concentration `c` is exactly `u`); the route (mean of every law, then the identified whole
//! sequence functional) must return the target truth for each action sequence, and the supplied
//! laws must differ from the enumerated ones.

mod common;

use antecedent_core::{ExecutionContext, SearchLimits, Value};
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_supported_mean,
};
use antecedent_estimate::temporal_transport::prepare_temporal_sequence;
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::sid::temporal_sequence::{
    TemporalSequenceDecision, decide_temporal_transport_sequence,
};
use common::temporal_fixture::{catalog, laws, sequence, source_scm, spec, target_scm, truth};

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn decide(seq: &[Value]) -> TemporalSequenceDecision {
    decide_temporal_transport_sequence(&spec(), seq, "source", "target", &catalog(), BUDGET, &ctx())
        .unwrap()
}

#[test]
fn two_step_sequence_on_dirichlet_updated_laws_equals_the_target_interventional_truth() {
    let c = 1e-6;
    let data = laws(&source_scm(), &target_scm(), &[]);
    let mut moved = 0.0_f64;
    let updated_laws: Vec<ExactDiscreteLaw> = data
        .laws()
        .iter()
        .map(|law| {
            let u = law.probabilities();
            let k = u.iter().filter(|p| **p > 0.0).count() as f64;
            let counts: Vec<f64> = u
                .iter()
                .map(|&p| {
                    if p > 0.0 {
                        let w = p * (1.0 + k * c) - c;
                        assert!(w >= 0.0, "cell {p} is below the update floor");
                        moved = moved.max((w - p).abs());
                        w
                    } else {
                        0.0
                    }
                })
                .collect();
            let mean = dirichlet_supported_mean(
                &counts,
                PosteriorRequest::IndependentNormal { scale: c },
                DirichletNamespace::TwoStepTemporal,
            )
            .unwrap();
            for (m, p) in mean.iter().zip(u) {
                assert!((m - p).abs() < 1e-12, "{m} vs {p}");
            }
            ExactDiscreteLaw::try_new(
                law.population(),
                law.regime(),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                mean,
                law.snapshot_identity(),
                law.tolerance(),
            )
            .unwrap()
        })
        .collect();
    assert!(moved > 1e-8, "the update must be nontrivial: {moved}");
    let updated = ExactTransportData::try_new(updated_laws, data.max_support_rows()).unwrap();
    for (a1, a2) in [(0_u8, 0_u8), (0, 1), (1, 0), (1, 1)] {
        let seq = sequence(f64::from(a1), f64::from(a2));
        let report = prepare_temporal_sequence(
            decide(&seq),
            updated.clone(),
            ExactEvaluationLimits::default(),
            &ctx(),
        )
        .unwrap()
        .evaluate(&ctx())
        .unwrap();
        let expected = truth(&target_scm(), [a1, a2]);
        assert!((report.mean - expected).abs() < 1e-9, "{a1}{a2}: {} vs {expected}", report.mean);
    }
}
