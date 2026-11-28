//! The compiled trial-to-target transport route and the direct facade calls compute the same
//! contrast.
//!
//! The compiled route (`TransportQuery` with a declared trial design) forms its point estimate
//! with `trial_to_target_effect` and its standard error with `trial_to_target_ipw_se`. The
//! facade entry `trial_to_target_effect_with_context` is the same kernel call under the caller's
//! execution policy, so the two agree bit for bit on the same rows, as does the Wald interval the
//! study publishes from that point and standard error.
//!
//! Both are deterministic: neither draws from the seed, so the seed differences between a direct
//! call and a study context do not arise for the frequentist route.
//!
//! Not covered by a twin: the direct call with caller-supplied outcome regressions (the AIPW
//! contrast) and the augmented response grid evaluate equations the compiled route has no query
//! for.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits of both paths are compared exactly")]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::estimate::trial_to_target_effect_with_context;
use antecedent::result::PublishedScalarUncertainty;
use antecedent::{CausalQuery, InferenceMode, RefuteSuite, Study, TransportTrialSpec};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{EffectEstimate, OverlapPolicy, trial_to_target_ipw_se};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};

struct Rows {
    treatment: Vec<bool>,
    outcome: Vec<f64>,
    trial: Vec<bool>,
    selection: Vec<f64>,
    propensity: Vec<f64>,
}

/// A deterministic trial of `n` units with row-varying selection and treatment probabilities.
fn rows(n: usize) -> Rows {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let mut out = Rows {
        treatment: Vec::new(),
        outcome: Vec::new(),
        trial: Vec::new(),
        selection: Vec::new(),
        propensity: Vec::new(),
    };
    for _ in 0..n {
        let selection = 0.3 + 0.4 * uniform();
        let propensity = 0.4 + 0.2 * uniform();
        let member = uniform() < selection;
        let treated = uniform() < propensity;
        out.treatment.push(treated);
        out.outcome.push(1.0 + 2.0 * f64::from(u8::from(treated)) + uniform());
        out.trial.push(member);
        out.selection.push(selection);
        out.propensity.push(propensity);
    }
    out
}

fn column(flags: &[bool]) -> Vec<f64> {
    flags.iter().map(|&flag| f64::from(u8::from(flag))).collect()
}

fn table(rows: &Rows) -> TabularData {
    let (a, trial) = (column(&rows.treatment), column(&rows.trial));
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", rows.outcome.as_slice()),
        ("trial", trial.as_slice()),
        ("s", rows.selection.as_slice()),
        ("e", rows.propensity.as_slice()),
    ])
    .unwrap()
}

fn query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)])
}

fn study(data: &TabularData) -> Study {
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Transport(query()))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn direct_identification() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Direct(PopulationFactor {
            regime: None,
            population: Arc::from("source"),
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }),
        certificate: TransportCertificate {
            rule: Arc::from("transport.direct"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn twin thread")
        .join()
        .expect("twin body panicked");
}

#[test]
fn compiled_trial_transport_equals_the_direct_ipw_point_standard_error_and_interval() {
    run_on_large_stack(|| {
        for (n, seed) in [(400_usize, 5_u64), (1_000, 6)] {
            let fixture = rows(n);
            let ctx = ExecutionContext::for_tests(seed);
            let direct = trial_to_target_effect_with_context(
                &direct_identification(),
                &fixture.outcome,
                &fixture.treatment,
                &fixture.trial,
                &fixture.selection,
                &fixture.propensity,
                None,
                &ctx,
            )
            .unwrap();
            let direct_se = trial_to_target_ipw_se(
                &fixture.outcome,
                &fixture.treatment,
                &fixture.trial,
                &fixture.selection,
                &fixture.propensity,
                direct.ipw,
            )
            .unwrap();

            let result = study(&table(&fixture)).run(&ctx).unwrap();
            let effect = result.estimate().as_effect().unwrap();
            assert_eq!(effect.ate.to_bits(), direct.ipw.to_bits(), "n={n}: point");
            assert_eq!(effect.se_analytic().to_bits(), direct_se.to_bits(), "n={n}: se");
            let transported = result.transport().expect("transport estimate");
            assert_eq!(transported.ipw.to_bits(), direct.ipw.to_bits(), "n={n}: ipw");
            assert_eq!(transported.overlap, direct.overlap, "n={n}: overlap");
            assert!(transported.aipw.is_none() && direct.aipw.is_none());

            // The interval the study publishes is the Wald interval of that point and error.
            let published = PublishedScalarUncertainty::select(effect).interval(effect);
            let reference = EffectEstimate::new(
                direct.ipw,
                direct_se,
                effect.provenance.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let wald = PublishedScalarUncertainty::select(&reference).interval(&reference);
            let (published, wald) = (published.unwrap(), wald.unwrap());
            assert_eq!(published.0.to_bits(), wald.0.to_bits(), "n={n}: interval lower");
            assert_eq!(published.1.to_bits(), wald.1.to_bits(), "n={n}: interval upper");
        }
    });
}

#[test]
fn the_direct_call_with_a_different_context_seed_gives_the_same_bits() {
    // Neither path reads the seed under frequentist IPW, so a context seed cannot separate them.
    run_on_large_stack(|| {
        let fixture = rows(300);
        let point = |seed: u64| {
            trial_to_target_effect_with_context(
                &direct_identification(),
                &fixture.outcome,
                &fixture.treatment,
                &fixture.trial,
                &fixture.selection,
                &fixture.propensity,
                None,
                &ExecutionContext::for_tests(seed),
            )
            .unwrap()
            .ipw
            .to_bits()
        };
        let compiled = study(&table(&fixture))
            .run(&ExecutionContext::for_tests(99))
            .unwrap()
            .estimate()
            .as_effect()
            .unwrap()
            .ate
            .to_bits();
        assert_eq!(point(1), point(2));
        assert_eq!(point(1), compiled);
    });
}
