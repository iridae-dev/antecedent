//! The direct cyclic parameter posterior is the compiled cyclic family's posterior.
//!
//! `sample_cyclic_posterior` samples the conditional-structure parameter posterior of a
//! specified equilibrium model from known-regime rows. The compiled `parameter_posterior`
//! cyclic family (`Study::prepare_cyclic_family`, plan route `CyclicFamily`) calls the same
//! function with the same template, parameter coordinates, rows, prior and sampler schedule,
//! under an execution context whose random stream is seeded from the request's sampling seed.
//! The twin test samples directly with the caller's context reseeded from that seed (as the
//! Python `PreparedCyclicSystem.posterior` does) and compares every draw and diagnostic with
//! the compiled answer, bit for bit. The known-truth test checks the compiled posterior mean
//! against the planted feedback weight.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::Study;
use antecedent::inference::metropolis::MetropolisOptions;
use antecedent_core::{ExecutionContext, RngFactory};
use antecedent_estimate::cyclic_likelihood::{
    CyclicParameter, CyclicPosterior, sample_cyclic_posterior,
};
use antecedent_io::cyclic_family_artifact::{
    CyclicFamilyAnswer, CyclicFamilyRequest, CyclicSamplingPolicy,
};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};

const SEED: u64 = 0x5EED_C1C1;
const PLANTED_WEIGHT: f64 = 0.3;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

/// A deterministic standard-normal stream (xorshift uniforms through Box-Muller).
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn equation(id: &str, weights: Vec<f64>) -> Equation {
    Equation {
        id: id.into(),
        intercept: 0.0,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: Vec::new(),
    }
}

/// `wishlists = b traffic + 0.2 ads + e_w`, `traffic = 0.2 wishlists + e_t`, `ads` clamped to
/// 0 or 1; the template's `b` starts at zero and the rows are drawn with `b = 0.3`.
fn model() -> EquilibriumModel {
    EquilibriumModel {
        equations: vec![
            equation("wishlists", vec![0.0, 0.0, 0.2]),
            equation("traffic", vec![0.2, 0.0, 0.0]),
            equation("ads", vec![0.0, 0.0, 0.0]),
        ],
        noise_covariance: vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
    }
}

fn rows(n: usize) -> Vec<RegimeRow> {
    let mut stream = Stream(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|i| {
            let ads = if i % 2 == 0 { 0.0 } else { 1.0 };
            let (ew, et) = (0.2 * stream.normal(), 0.2 * stream.normal());
            let wishlists = (0.2 * ads + ew + PLANTED_WEIGHT * et) / (1.0 - PLANTED_WEIGHT * 0.2);
            let traffic = 0.2 * wishlists + et;
            RegimeRow {
                values: vec![wishlists, traffic, ads],
                clamps: vec![Clamp { variable: 2, value: ads }],
            }
        })
        .collect()
}

fn parameters() -> Vec<CyclicParameter> {
    vec![CyclicParameter::Weight { equation: 0, parent: 1 }]
}

fn sampling() -> CyclicSamplingPolicy {
    CyclicSamplingPolicy { chains: 4, warmup: 300, draws: 400, proposal_scale: 0.1, seed: SEED }
}

fn compiled_posterior(rows: &[RegimeRow]) -> (CyclicPosterior, &'static str) {
    let request = CyclicFamilyRequest::ParameterPosterior {
        model: model(),
        parameters: parameters(),
        rows: rows.to_vec(),
        prior_sd: 1.0,
        sampling: sampling(),
        worlds: vec![Vec::new()],
        noise: vec![vec![0.0; 3]],
        initial: vec![0.0; 3],
        solve_policy: SolvePolicy::default(),
    };
    let prepared = Study::prepare_cyclic_family(&request, &ctx()).expect("prepares");
    let route = prepared.plan_view().route().as_str();
    let report = prepared.estimate(&ctx()).expect("executes");
    match report.answer {
        CyclicFamilyAnswer::ParameterPosterior { posterior, .. } => (posterior, route),
        other => panic!("expected a parameter posterior, got {other:?}"),
    }
}

fn direct_posterior(rows: &[RegimeRow]) -> CyclicPosterior {
    let mut ctx = ctx();
    ctx.rng = RngFactory::from_seed(SEED);
    let policy = sampling();
    sample_cyclic_posterior(
        &model(),
        &parameters(),
        rows,
        1.0,
        MetropolisOptions {
            chains: policy.chains,
            warmup: policy.warmup,
            draws: policy.draws,
            proposal_scale: policy.proposal_scale,
        },
        &ctx,
    )
    .expect("samples")
}

#[test]
fn the_direct_posterior_equals_the_compiled_parameter_posterior_family() {
    let rows = rows(300);
    let (compiled, route) = compiled_posterior(&rows);
    let direct = direct_posterior(&rows);
    assert_eq!(route, "CyclicFamily");
    assert_eq!(direct.samples.len(), compiled.samples.len());
    for (draw, (a, b)) in direct.samples.iter().zip(&compiled.samples).enumerate() {
        let a: Vec<u64> = a.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u64> = b.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "draw {draw}");
    }
    assert_eq!(direct.chains, compiled.chains);
    assert_eq!(direct.draws_per_chain, compiled.draws_per_chain);
    assert_eq!(direct.accepted, compiled.accepted);
    assert_eq!(direct.max_rhat.map(f64::to_bits), compiled.max_rhat.map(f64::to_bits));
    assert_eq!(direct.min_bulk_ess.to_bits(), compiled.min_bulk_ess.to_bits());
    assert_eq!(direct.min_tail_ess.to_bits(), compiled.min_tail_ess.to_bits());
    assert_eq!(direct.diagnostics_pass, compiled.diagnostics_pass);
}

/// Known truth: the rows are drawn with feedback weight 0.3, so the compiled posterior mean of
/// the weight recovers it.
#[test]
fn the_compiled_parameter_posterior_recovers_the_planted_weight() {
    let (posterior, _) = compiled_posterior(&rows(300));
    let mean =
        posterior.samples.iter().map(|draw| draw[0]).sum::<f64>() / posterior.samples.len() as f64;
    assert!(
        (mean - PLANTED_WEIGHT).abs() < 0.1,
        "posterior mean {mean} is not near the planted weight {PLANTED_WEIGHT}"
    );
}
