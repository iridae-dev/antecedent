//! The multi-source limited-experiment route under a Dirichlet posterior mean, against the
//! enumerated interventional truth of the target model (R-443 Figure 1(c,d), enumerated in
//! `common/mz_fixture.rs`).
//!
//! Every supplied source and target law is the exact preimage of the enumerated law under the
//! update (`w = u (1 + K c) - c` over the law's positive cells, so the Dirichlet mean with
//! concentration `c` is exactly `u`); the route (mean of every law, then the checked combined
//! formula) must return the target's own `P*(y | do(x))`, and the supplied laws must differ
//! from the enumerated ones.

mod common;

use antecedent_core::{EvidenceCatalog, ExecutionContext, Value};
use antecedent_estimate::evaluate_exact_mz_transport;
use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_supported_mean,
};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision,
    ZTransportSourceSpec, bind_mz_transport_catalog, decide_mz_transport,
};
use common::mz_fixture::{X, Y, evidence, graph, query, sources, target_scm};
use common::z_scm::{risk_of, vid};

fn bound(
    sources: Vec<ZTransportSourceSpec>,
    catalog: &EvidenceCatalog,
) -> BoundMzTransportFunctional {
    let q = query(sources);
    let ctx = ExecutionContext::for_tests(3);
    let MzTransportDecision::Identified { derivation, .. } =
        decide_mz_transport(&graph(), &q, catalog, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("the complementary catalog identifies");
    };
    bind_mz_transport_catalog(&graph(), &derivation, catalog).unwrap()
}

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

#[test]
fn multi_source_formula_on_dirichlet_updated_laws_equals_the_target_interventional_truth() {
    let c = 1e-5;
    let (catalog, data) = evidence();
    let functional = bound(sources(), &catalog);
    let ctx = ExecutionContext::for_tests(3);
    let mut moved = 0.0_f64;
    let laws: Vec<ExactDiscreteLaw> = data
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
                DirichletNamespace::MultiSource,
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
    let updated = ExactTransportData::try_new(laws, data.max_support_rows()).unwrap();
    for x in [0_u8, 1] {
        let point = evaluate_exact_mz_transport(
            &functional,
            updated.clone(),
            request(x == 1),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let want = target_scm().risk(&[(X, x)], Y);
        assert!((risk_of(&point) - want).abs() < 1e-9, "do(X={x}): {} vs {want}", risk_of(&point));
    }
}
